# // # FILE: .\src\Optimization\TopOpt.jl

module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using Base.Threads
using ..Element
using ..Mesh
using ..GPUHelmholtz 
using ..Helpers

export update_density!, reset_filter_cache!

# ------------------------------------------------------------------------------
# FILTER CACHE
# ------------------------------------------------------------------------------

mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

# ------------------------------------------------------------------------------
# HELMHOLTZ FILTER ASSEMBLY (CPU)
# ------------------------------------------------------------------------------

function assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R)
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    nElem = nElem_x * nElem_y * nElem_z
    Ke_local, Me_local = Element.get_scalar_canonical_matrices(dx, dy, dz)
    
    entries_per_elem = 64 
    total_entries = nElem * entries_per_elem
    
    I_vec = Vector{Int64}(undef, total_entries)
    J_vec = Vector{Int64}(undef, total_entries)
    V_vec = Vector{Float64}(undef, total_entries)
    
    nx, ny = nElem_x + 1, nElem_y + 1
    Re_local = (R^2) .* Ke_local .+ Me_local
    
    idx_counter = 0
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        n1 = i        + (j-1)*nx        + (k-1)*nx*ny
        n2 = (i+1) + (j-1)*nx        + (k-1)*nx*ny
        n3 = (i+1) + j*nx            + (k-1)*nx*ny
        n4 = i        + j*nx            + (k-1)*nx*ny
        n5 = i        + (j-1)*nx        + k*nx*ny
        n6 = (i+1) + (j-1)*nx        + k*nx*ny
        n7 = (i+1) + j*nx            + k*nx*ny
        n8 = i        + j*nx            + k*nx*ny
        nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        
        for r in 1:8
            row = nodes[r]
            for c in 1:8
                col = nodes[c]
                idx_counter += 1
                I_vec[idx_counter] = Int64(row)
                J_vec[idx_counter] = Int64(col)
                V_vec[idx_counter] = Float64(Re_local[r, c])
            end
        end
    end
    
    K_global = sparse(I_vec, J_vec, V_vec, nNodes, nNodes)
    n = size(K_global, 1)
    K_global = K_global + sparse(1:n, 1:n, fill(1e-9, n), n, n) 
    
    return cholesky(K_global)
end

function apply_helmholtz_filter_cpu(field_elem::Vector{Float32}, F_fact, nElem_x, nElem_y, nElem_z, dx, dy, dz)
    nx, ny = nElem_x + 1, nElem_y + 1
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    nElem = length(field_elem)
    
    elem_vol = dx * dy * dz
    nodal_weight = elem_vol / 8.0f0
    RHS = zeros(Float64, nNodes)
    
    idx_e = 1
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        val = Float64(field_elem[idx_e] * nodal_weight)
        n1 = i + (j-1)*nx + (k-1)*nx*ny
        n2 = n1 + 1
        n3 = n1 + nx + 1
        n4 = n1 + nx
        n5 = n1 + nx*ny
        n6 = n2 + nx*ny
        n7 = n3 + nx*ny
        n8 = n4 + nx*ny
        
        RHS[n1] += val; RHS[n2] += val; RHS[n3] += val; RHS[n4] += val;
        RHS[n5] += val; RHS[n6] += val; RHS[n7] += val; RHS[n8] += val;
        idx_e += 1
    end
    
    nodal_filtered = F_fact \ RHS
    
    filtered_elem = zeros(Float32, nElem)
    
    Threads.@threads for e in 1:nElem
        iz = div(e - 1, nElem_x * nElem_y) + 1
        rem_z = (e - 1) % (nElem_x * nElem_y)
        iy = div(rem_z, nElem_x) + 1
        ix = rem_z % nElem_x + 1
        
        n1 = ix + (iy-1)*nx + (iz-1)*nx*ny
        n2 = n1 + 1; n3 = n1 + nx + 1; n4 = n1 + nx;
        n5 = n1 + nx*ny; n6 = n2 + nx*ny; n7 = n3 + nx*ny; n8 = n4 + nx*ny;
        
        sum_nodes = nodal_filtered[n1] + nodal_filtered[n2] + nodal_filtered[n3] + nodal_filtered[n4] +
                    nodal_filtered[n5] + nodal_filtered[n6] + nodal_filtered[n7] + nodal_filtered[n8]
        
        filtered_elem[e] = Float32(sum_nodes / 8.0)
    end
    
    return filtered_elem
end

# ------------------------------------------------------------------------------
# OPTIMIZATION UPDATE ROUTINE
# ------------------------------------------------------------------------------

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_stress_allowable::Float32, 
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict,
                         elements::Matrix{Int}, 
                         is_annealing::Bool=false) 

    nElem = length(density)
    
    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]
    solver_params = config["solver_parameters"]

    # --------------------------------------------------------------------------
    # 1. PARSE GEOMETRY & CONSTANTS
    # --------------------------------------------------------------------------
    nElem_x = Int(geom_params["nElem_x_computed"]) 
    nElem_y = Int(geom_params["nElem_y_computed"])
    nElem_z = Int(geom_params["nElem_z_computed"])
    dx = Float32(geom_params["dx_computed"])
    dy = Float32(geom_params["dy_computed"])
    dz = Float32(geom_params["dz_computed"])
    max_domain_dim = geom_params["max_domain_dim"]
    filter_tol = Float32(get(solver_params, "filter_tolerance", 1.0e-5))

    avg_element_size = (dx + dy + dz) / 3.0f0

    # --------------------------------------------------------------------------
    # 2. STRESS REGULARIZATION (Optional Pre-Filtering)
    # --------------------------------------------------------------------------
    radius_multiplier = Float32(get(opt_params, "stress_regularization_ratio", 0.0f0))
    
    if radius_multiplier > 1.0e-6
        stress_filter_radius = radius_multiplier * avg_element_size
        
        ran_stress_gpu = false
        filtered_stress = zeros(Float32, nElem)

        if CUDA.functional()
             s_gpu, _, _, _ = GPUHelmholtz.apply_gpu_filter!(
                l1_stress_norm_field, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz, stress_filter_radius, elements, filter_tol
            )
            if !isempty(s_gpu)
                filtered_stress = s_gpu
                ran_stress_gpu = true
            end
        end

        if !ran_stress_gpu
            # Note: We don't cache this factorization to avoid overwriting the main density filter cache
            fact = assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, stress_filter_radius)
            filtered_stress = apply_helmholtz_filter_cpu(
                l1_stress_norm_field, 
                fact, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz
            )
        end

        # Update the stress field with the smoothed version
        Threads.@threads for e in 1:nElem
            l1_stress_norm_field[e] = filtered_stress[e]
        end
    end

    # --------------------------------------------------------------------------
    # 3. CALCULATE FILTER RADIUS SCHEDULE
    # --------------------------------------------------------------------------
    R_init_perc = Float32(get(opt_params, "filter_R_init_perc", 0.0f0))
    R_interm_perc = Float32(get(opt_params, "filter_R_interm_perc", 0.0f0))
    R_final_perc = Float32(get(opt_params, "filter_R_final_perc", 0.0f0))
    R_interm_iter_perc = Float32(get(opt_params, "filter_R_interm_iter_perc", 50.0f0)) / 100.0f0
    
    R_init_length = R_init_perc / 100.0f0 * max_domain_dim
    R_interm_length = R_interm_perc / 100.0f0 * max_domain_dim
    R_final_length = R_final_perc / 100.0f0 * max_domain_dim

    iter_interm = max(1, round(Int, R_interm_iter_perc * number_of_iterations))
    calc_iter = min(iter, number_of_iterations)
    
    R_length = 0.0f0

    if calc_iter <= iter_interm
        t = (iter_interm > 1) ? Float32(calc_iter - 1) / Float32(iter_interm - 1) : 0.0f0
        R_length = R_init_length * (1.0f0 - t) + R_interm_length * t
    else 
        t = (number_of_iterations > iter_interm) ? Float32(calc_iter - iter_interm) / Float32(number_of_iterations - iter_interm) : 0.0f0
        R_length = R_interm_length * (1.0f0 - t) + R_final_length * t
    end
    
    # --------------------------------------------------------------------------
    # [RADIUS GUARD] AUTOMATIC CHECKERBOARD DETECTION & CORRECTION
    # --------------------------------------------------------------------------
    # The Helmholtz filter requires a radius of at least 1.2x the element size
    # to effectively suppress numerical checkerboarding (jagged patterns).
    
    R_safe_min = 1.2f0 * avg_element_size 

    if R_length < R_safe_min
        if iter % 10 == 0 # Log periodically to avoid spam
            println("  [Filter Guard] ⚠️ INSTABILITY DETECTED: Radius ($R_length) < Element Size ($avg_element_size).")
            println("  [Filter Guard] 🛡️ Auto-increasing Radius to $R_safe_min to prevent checkerboarding.")
        end
        R_length = R_safe_min
    end

    # The effective radius is adjusted by a heuristic factor (2.5) for the specific kernel type
    R_effective = R_length / 2.5f0
    
    # --------------------------------------------------------------------------
    # 4. CALCULATE PROPOSED DENSITY
    # --------------------------------------------------------------------------
    proposed_density_field = zeros(Float32, nElem)

    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            current_l1_stress = l1_stress_norm_field[e]
            # Heuristic update: Density ~ Stress / Allowable
            val = (current_l1_stress / l1_stress_allowable) / E
            proposed_density_field[e] = clamp(val, min_density, max_density)
        else
            proposed_density_field[e] = original_density[e]
        end
    end

    # --------------------------------------------------------------------------
    # 5. APPLY DENSITY FILTER
    # --------------------------------------------------------------------------
    filtered_density_field = proposed_density_field
    filter_time = 0.0
    filter_iters = 0
    filter_res = 0.0
    
    if R_effective > 1e-4
        ran_gpu_successfully = false
        
        # Try GPU Filter first
        if CUDA.functional()
            filtered_gpu, t_gpu, it_gpu, res_gpu = GPUHelmholtz.apply_gpu_filter!(
                proposed_density_field, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz, R_effective, elements, filter_tol
            )
            
            if !isempty(filtered_gpu)
                filtered_density_field = filtered_gpu
                filter_time = t_gpu
                filter_iters = it_gpu
                filter_res = res_gpu
                ran_gpu_successfully = true
            end
        end

        # Fallback to CPU Filter
        if !ran_gpu_successfully
            t_start = time()
            if !GLOBAL_FILTER_CACHE.is_initialized || abs(GLOBAL_FILTER_CACHE.radius - R_effective) > 1e-5
                println("  [CPU Filter] Building system for $nElem elements (R=$R_effective)...")
                fact = assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R_effective)
                GLOBAL_FILTER_CACHE.K_filter = fact
                GLOBAL_FILTER_CACHE.radius = R_effective
                GLOBAL_FILTER_CACHE.is_initialized = true
            end
            
            filtered_density_field = apply_helmholtz_filter_cpu(
                proposed_density_field, 
                GLOBAL_FILTER_CACHE.K_filter, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz
            )
            filter_time = time() - t_start
            filter_iters = 1 
            filter_res = 0.0
        end
    end
    
    # --------------------------------------------------------------------------
    # 6. UPDATE DENSITY & CALCULATE DELTA
    # --------------------------------------------------------------------------
    n_threads = Threads.nthreads()
    max_changes = zeros(Float32, n_threads)

    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            old_val = density[e]
            new_val = clamp(filtered_density_field[e], min_density, max_density)
            density[e] = new_val
            
            diff = abs(new_val - old_val)
            tid = Threads.threadid()
            if diff > max_changes[tid]
                max_changes[tid] = diff
            end
        end
    end
    max_change = maximum(max_changes)
    
    # --------------------------------------------------------------------------
    # 7. CULLING (VOID REMOVAL)
    # --------------------------------------------------------------------------
    # Gradually increase the threshold for what is considered "void"
    
    current_threshold = 0.0f0
    final_threshold_val = Float32(get(opt_params, "final_density_threshold", 0.95))
    max_culling_ratio = Float32(get(opt_params, "max_culling_ratio", 0.1)) # Default 0.1
    
    if iter > number_of_iterations
        current_threshold = final_threshold_val
    else
        progress = Float32(iter) / Float32(number_of_iterations)
        current_threshold = final_threshold_val * progress
    end
    
    cull_candidates = Int[]
    active_count = 0
    
    for e in 1:nElem
        if !protected_elements_mask[e]
            if density[e] > min_density
                active_count += 1
                if density[e] < current_threshold
                    push!(cull_candidates, e)
                end
            end
        else
             # Count protected non-voids as active
             if original_density[e] > min_density
                 active_count += 1
             end
        end
    end
    
    # Safety: Don't remove too many elements at once (Mesh Collapse prevention)
    max_allowed_culls = floor(Int, active_count * max_culling_ratio)
    
    if length(cull_candidates) > max_allowed_culls
        # If too many candidates, only remove the weakest ones
        sort!(cull_candidates, by = idx -> density[idx])
        
        for i in 1:max_allowed_culls
            idx = cull_candidates[i]
            density[idx] = min_density
        end
    else
        for idx in cull_candidates
            density[idx] = min_density
        end
    end

    # Re-enforce protected elements
    Threads.@threads for e in 1:nElem
        if protected_elements_mask[e]
            density[e] = original_density[e]
        end
    end
    
    return max_change, R_length, current_threshold, filter_time, filter_iters, filter_res
end

end