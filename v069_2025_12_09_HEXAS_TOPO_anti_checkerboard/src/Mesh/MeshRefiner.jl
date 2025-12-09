# FILE: .\src\Mesh\MeshRefiner.jl

module MeshRefiner

using LinearAlgebra
using Printf

using ..Mesh
using ..Helpers

export refine_mesh_and_fields

function estimate_element_memory_cost_bytes(hard_element_limit::Int)
    # Heuristic: 
    # If the limit is massive (>500M), we assume H-series GPU with optimized packing.
    # We lower the 'safe' byte estimate per element because purely Matrix-Free on H100 is very efficient.
    
    if hard_element_limit > 500_000_000
        return 180 # Matrix-free compressed
    else
        return 250 # Standard safety
    end
end

function get_safe_element_limit(hard_element_limit::Int)
    
    free_ram = Float64(Sys.free_memory())
    
    # H-Series usually has ECC and headless operation, requiring less safety buffer.
    # Standard workstation needs more.
    is_h_series = hard_element_limit > 500_000_000
    
    safety_buffer = if is_h_series
        max(free_ram * 0.05, 2.0 * 1024^3) # 5% buffer for H200
    else
        max(free_ram * 0.20, 4.0 * 1024^3) # 20% buffer for RTX
    end
        
    usable_ram = free_ram - safety_buffer

    if usable_ram <= 0
        @warn "[MeshRefiner] System RAM is critically low. Limiting mesh severely."
        return 1_000_000 
    end

    bytes_per_elem = estimate_element_memory_cost_bytes(hard_element_limit)
    max_elements = floor(Int, usable_ram / bytes_per_elem)
    
    return max_elements
end

function refine_mesh_and_fields(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                density::Vector{Float32}, 
                                alpha_field::Vector{Float32}, 
                                current_dims::Tuple{Int, Int, Int},
                                target_active_count::Int,
                                domain_bounds::NamedTuple;
                                max_growth_rate::Float64=1.2,
                                hard_element_limit::Int=800_000_000) 

    println("\n" * "="^60)
    println("[MeshRefiner] Evaluating Mesh Refinement...")

    n_total_old = length(density)
    
    n_active_old = count(d -> d > 0.05f0, density) 
    active_ratio = max(0.01, n_active_old / n_total_old) 
    
    println("  Current Total:   $(n_total_old)")
    println("  Current Active:  $(n_active_old) ($(round(active_ratio*100, digits=2))%)")
    println("  Target Active:   $(target_active_count)")

    ideal_total_elements = round(Int, target_active_count / active_ratio)
    
    rate_limit_elements = round(Int, n_total_old * max_growth_rate)
    ram_limit_elements = get_safe_element_limit(hard_element_limit)
    final_new_total = min(ideal_total_elements, rate_limit_elements, ram_limit_elements, hard_element_limit)

    println("  > Ideal Requirement: $ideal_total_elements")
    println("  > Rate Limit ($(max_growth_rate)x): $rate_limit_elements")
    println("  > RAM Limit:         $ram_limit_elements")
    println("  > Config Limit:      $hard_element_limit")
    
    if final_new_total < ideal_total_elements
        println("  [⚠️ LIMIT APPLIED] Mesh growth constrained by safeguards.")
    end
    
    println("  ---> FINAL NEW MESH: $final_new_total")

    if final_new_total < (n_total_old * 1.05)
        println("  [ℹ️ INFO] Growth too small (< 5%). Skipping refinement.")
        println("="^60 * "\n")
        return nodes, elements, density, alpha_field, current_dims
    end

    len_x, len_y, len_z = domain_bounds.len_x, domain_bounds.len_y, domain_bounds.len_z
    
    new_nx, new_ny, new_nz, new_dx, new_dy, new_dz, actual_count = 
        Helpers.calculate_element_distribution(len_x, len_y, len_z, final_new_total)
        
    println("  > Generating grid: $(new_nx)x$(new_ny)x$(new_nz) = $actual_count")
    println("  > Resolution: $(new_dx) x $(new_dy) x $(new_dz)")

    if actual_count > get_safe_element_limit(hard_element_limit)
        error("[MeshRefiner] Aborting: Race condition on memory. RAM dropped during calculation.")
    end

    new_nodes, new_elements, new_dims = Mesh.generate_mesh(
        new_nx, new_ny, new_nz;
        dx=new_dx, dy=new_dy, dz=new_dz
    )
    
    min_pt = domain_bounds.min_pt
    new_nodes[:, 1] .+= min_pt[1]
    new_nodes[:, 2] .+= min_pt[2]
    new_nodes[:, 3] .+= min_pt[3]
    
    println("  > Mapping density and thermal fields...")
    n_new_total = size(new_elements, 1)
    new_density = zeros(Float32, n_new_total)
    new_alpha   = zeros(Float32, n_new_total) 
    
    old_nx = current_dims[1] - 1
    old_ny = current_dims[2] - 1
    old_nz = current_dims[3] - 1
    
    old_dx = len_x / old_nx
    old_dy = len_y / old_ny
    old_dz = len_z / old_nz

    Threads.@threads for e_new in 1:n_new_total
        iz = div(e_new - 1, new_nx * new_ny) + 1
        rem_z = (e_new - 1) % (new_nx * new_ny)
        iy = div(rem_z, new_nx) + 1
        ix = rem_z % new_nx + 1
        
        cx = (ix - 0.5f0) * new_dx
        cy = (iy - 0.5f0) * new_dy
        cz = (iz - 0.5f0) * new_dz
        
        old_ix = clamp(floor(Int, cx / old_dx) + 1, 1, old_nx)
        old_iy = clamp(floor(Int, cy / old_dy) + 1, 1, old_ny)
        old_iz = clamp(floor(Int, cz / old_dz) + 1, 1, old_nz)
        
        old_linear = old_ix + (old_iy - 1)*old_nx + (old_iz - 1)*old_nx*old_ny
        
        new_density[e_new] = density[old_linear]
        new_alpha[e_new]   = alpha_field[old_linear] 
    end
    
    println("[MeshRefiner] Success.")
    println("="^60 * "\n")

    return new_nodes, new_elements, new_density, new_alpha, new_dims
end

end