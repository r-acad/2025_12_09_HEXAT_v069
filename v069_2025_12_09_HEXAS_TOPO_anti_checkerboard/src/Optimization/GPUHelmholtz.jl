// # FILE: .\src\Optimization\GPUHelmholtz.jl
module GPUHelmholtz

using CUDA
using LinearAlgebra
using Printf
using ..Element
using ..Mesh

export HelmholtzWorkspace, setup_helmholtz_workspace, apply_gpu_filter!

mutable struct HelmholtzWorkspace{T}
    is_initialized::Bool
    radius::T
    
    elements::CuVector{Int32} 
    Ae_base::CuMatrix{T}         
    inv_diag::CuVector{T}        
    
    r::CuVector{T}
    p::CuVector{T}
    z::CuVector{T}
    Ap::CuVector{T}
    x::CuVector{T} 
    b::CuVector{T} 
    
    nNodes::Int
    nElem::Int
    
    HelmholtzWorkspace{T}() where T = new{T}(false, T(0))
end

const GLOBAL_HELMHOLTZ_CACHE = HelmholtzWorkspace{Float32}()

# ==============================================================================
# KERNELS
# ==============================================================================

function compute_rhs_kernel!(b, density, elements, val_scale, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        val = density[e] * val_scale
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            CUDA.atomic_add!(pointer(b, node), val)
        end
    end
    return nothing
end

function matvec_kernel!(y, x, elements, Ae, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        
        # Load x values for this element's nodes
        x_loc_1 = x[elements[base_idx + 1]]
        x_loc_2 = x[elements[base_idx + 2]]
        x_loc_3 = x[elements[base_idx + 3]]
        x_loc_4 = x[elements[base_idx + 4]]
        x_loc_5 = x[elements[base_idx + 5]]
        x_loc_6 = x[elements[base_idx + 6]]
        x_loc_7 = x[elements[base_idx + 7]]
        x_loc_8 = x[elements[base_idx + 8]]
        
        @inbounds for r in 1:8
            val = Ae[r,1]*x_loc_1 + Ae[r,2]*x_loc_2 + Ae[r,3]*x_loc_3 + Ae[r,4]*x_loc_4 +
                  Ae[r,5]*x_loc_5 + Ae[r,6]*x_loc_6 + Ae[r,7]*x_loc_7 + Ae[r,8]*x_loc_8
            
            node = elements[base_idx + r]
            CUDA.atomic_add!(pointer(y, node), val)
        end
    end
    return nothing
end

function extract_solution_kernel!(filtered_density, x, elements, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        sum_val = 0.0f0
        @inbounds for i in 1:8
            sum_val += x[elements[base_idx + i]]
        end
        filtered_density[e] = sum_val / 8.0f0
    end
    return nothing
end

function compute_diagonal_kernel!(diag, elements, Ae_diag, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            val = Ae_diag[i]
            CUDA.atomic_add!(pointer(diag, node), val)
        end
    end
    return nothing
end

# ==============================================================================
# WORKSPACE MANAGEMENT
# ==============================================================================

function setup_helmholtz_workspace(elements_cpu::Matrix{Int}, 
                                   dx::T, dy::T, dz::T, radius::T) where T
    
    ws = GLOBAL_HELMHOLTZ_CACHE
    nElem = size(elements_cpu, 1)
    nNodes = maximum(elements_cpu)
    
    # Only re-allocate if necessary
    if !ws.is_initialized || ws.nElem != nElem || abs(ws.radius - radius) > 1e-5
        
        # Proactively clear memory to ensure we have space for the new workspace
        # This prevents "0 bytes available" errors when the pool is fragmented
        if CUDA.available_memory() < (nElem * 200) 
            GC.gc()
            CUDA.reclaim()
        end

        # We construct everything in a try-catch to allow safe fallback upstream
        # rather than erroring out here.
        
        Ke, Me = Element.get_scalar_canonical_matrices(dx, dy, dz)
        Ae_cpu = (radius^2) .* Ke .+ Me
        
        elements_flat = vec(elements_cpu') 
        
        # Allocations
        ws.elements = CuArray(Int32.(elements_flat))
        ws.Ae_base = CuArray(Ae_cpu)
        
        diag_vec = CUDA.zeros(T, nNodes)
        Ae_diag_gpu = CuArray(diag(Ae_cpu))
        
        threads = 256
        blocks = cld(nElem, threads)
        @cuda threads=threads blocks=blocks compute_diagonal_kernel!(diag_vec, ws.elements, Ae_diag_gpu, nElem)
        
        ws.inv_diag = 1.0f0 ./ diag_vec
        
        ws.r  = CUDA.zeros(T, nNodes)
        ws.p  = CUDA.zeros(T, nNodes)
        ws.z  = CUDA.zeros(T, nNodes)
        ws.Ap = CUDA.zeros(T, nNodes)
        ws.x  = CUDA.zeros(T, nNodes)
        ws.b  = CUDA.zeros(T, nNodes)
        
        ws.nNodes = nNodes
        ws.nElem = nElem
        ws.radius = radius
        ws.is_initialized = true
    end
    return ws
end

# ==============================================================================
# SOLVERS
# ==============================================================================

function solve_helmholtz_on_gpu(density_cpu::Vector{T}, ws::HelmholtzWorkspace{T}, 
                                dx, dy, dz, tol::T) where T
    
    density_gpu = CuArray(density_cpu) 
    filtered_gpu = CUDA.zeros(T, ws.nElem)
    
    threads = 256
    blocks = cld(ws.nElem, threads)
    
    fill!(ws.b, 0.0f0)
    elem_vol = dx * dy * dz
    val_scale = elem_vol / 8.0f0
    @cuda threads=threads blocks=blocks compute_rhs_kernel!(ws.b, density_gpu, ws.elements, val_scale, ws.nElem)
    
    norm_b = norm(ws.b)
    if norm_b == 0.0f0
        return density_cpu, 0.0, 0, 0.0
    end

    fill!(ws.x, 0.0f0) 
    ws.r .= ws.b
    ws.z .= ws.r .* ws.inv_diag 
    ws.p .= ws.z
    
    rho_old = dot(ws.r, ws.z)
    
    max_iter = 200 
    final_rel_res = 0.0f0
    final_iter = 0

    filter_start_time = time()

    for iter in 1:max_iter
        final_iter = iter
        
        fill!(ws.Ap, 0.0f0)
        @cuda threads=threads blocks=blocks matvec_kernel!(ws.Ap, ws.p, ws.elements, ws.Ae_base, ws.nElem)
        
        alpha = rho_old / dot(ws.p, ws.Ap)
        
        ws.x .+= alpha .* ws.p
        ws.r .-= alpha .* ws.Ap
        
        if iter % 10 == 0
            norm_r = norm(ws.r)
            final_rel_res = norm_r / norm_b
            if final_rel_res < tol
                break
            end
        end
        
        ws.z .= ws.r .* ws.inv_diag 
        
        rho_new = dot(ws.r, ws.z)
        beta = rho_new / rho_old
        ws.p .= ws.z .+ beta .* ws.p
        
        rho_old = rho_new
    end
    
    filter_time = time() - filter_start_time
    @cuda threads=threads blocks=blocks extract_solution_kernel!(filtered_gpu, ws.x, ws.elements, ws.nElem)
    
    return Array(filtered_gpu), filter_time, final_iter, final_rel_res
end

function apply_blocked_gpu_filter!(density_full::Vector{T}, nElem_x, nElem_y, nElem_z, 
                                   dx, dy, dz, radius, tol::T) where T
    
    # --- CRITICAL: Force Clean before checking memory ---
    # The linear solver likely left the GPU full. We must clear it to see true availability.
    GC.gc()
    CUDA.reclaim()
    
    # Conservative estimate for Blocked Helmholtz (80-100 bytes/element)
    bytes_per_elem_heuristic = 100 
    
    free_mem = CUDA.available_memory()
    # Use 85% of what's truly free to avoid OOM during allocation spikes
    safe_mem = Int(floor(free_mem * 0.85)) 
    
    # Ensure we don't calculate 0 size if memory is weirdly reported
    safe_mem = max(safe_mem, 100 * 1024 * 1024) # Minimum 100MB block size
    
    max_elems_per_block = div(safe_mem, bytes_per_elem_heuristic)
    
    # Calculate Halo size (Overlap) - 3x Radius for physics safety
    min_dim = min(dx, dy, dz)
    halo_cells = ceil(Int, (3.0 * radius) / min_dim)
    halo_cells = max(halo_cells, 2)
    
    cube_root = cbrt(max_elems_per_block)
    
    n_chunks_x = ceil(Int, nElem_x / cube_root)
    n_chunks_y = ceil(Int, nElem_y / cube_root)
    n_chunks_z = ceil(Int, nElem_z / cube_root)
    
    blk_nx = ceil(Int, nElem_x / n_chunks_x)
    blk_ny = ceil(Int, nElem_y / n_chunks_y)
    blk_nz = ceil(Int, nElem_z / n_chunks_z)

    filtered_full = zeros(T, length(density_full))
    
    total_time = 0.0
    total_iters = 0
    blocks_processed = 0
    n_total_blocks = n_chunks_x * n_chunks_y * n_chunks_z
    
    if n_total_blocks > 1
        println("  [GPU Filter] Blocked Mode: Split into $n_total_blocks blocks ($n_chunks_x x $n_chunks_y x $n_chunks_z). Halo: $halo_cells cells.")
    end

    for cz in 1:n_chunks_z, cy in 1:n_chunks_y, cx in 1:n_chunks_x
        
        # 1. Core Range
        x_start = (cx - 1) * blk_nx + 1; x_end = min(cx * blk_nx, nElem_x)
        y_start = (cy - 1) * blk_ny + 1; y_end = min(cy * blk_ny, nElem_y)
        z_start = (cz - 1) * blk_nz + 1; z_end = min(cz * blk_nz, nElem_z)
        
        # 2. Halo Range
        x_start_halo = max(1, x_start - halo_cells); x_end_halo = min(nElem_x, x_end + halo_cells)
        y_start_halo = max(1, y_start - halo_cells); y_end_halo = min(nElem_y, y_end + halo_cells)
        z_start_halo = max(1, z_start - halo_cells); z_end_halo = min(nElem_z, z_end + halo_cells)
        
        loc_nx = x_end_halo - x_start_halo + 1
        loc_ny = y_end_halo - y_start_halo + 1
        loc_nz = z_end_halo - z_start_halo + 1
        n_loc_elem = loc_nx * loc_ny * loc_nz
        
        # 3. Extract Data (CPU side)
        rho_local = zeros(T, n_loc_elem)
        loc_idx = 1
        for k in z_start_halo:z_end_halo
            for j in y_start_halo:y_end_halo
                global_start = (k-1)*(nElem_x*nElem_y) + (j-1)*nElem_x + x_start_halo
                rho_local[loc_idx : loc_idx+loc_nx-1] = density_full[global_start : global_start+loc_nx-1]
                loc_idx += loc_nx
            end
        end
        
        # 4. Solve Local Block on GPU
        _, elems_local, _ = Mesh.generate_mesh(loc_nx, loc_ny, loc_nz; dx=T(dx), dy=T(dy), dz=T(dz))
        ws = setup_helmholtz_workspace(elems_local, T(dx), T(dy), T(dz), T(radius))
        
        res_local, t_solve, iters, _ = solve_helmholtz_on_gpu(rho_local, ws, dx, dy, dz, tol)
        
        total_time += t_solve
        total_iters += iters
        blocks_processed += 1
        
        # 5. Reassemble Core (Discard Halo)
        off_x = x_start - x_start_halo
        off_y = y_start - y_start_halo
        off_z = z_start - z_start_halo
        core_dim_x = x_end - x_start + 1
        
        for k in 1:(z_end - z_start + 1)
            loc_k = k + off_z
            for j in 1:(y_end - y_start + 1)
                loc_j = j + off_y
                loc_start_idx = (loc_k-1)*(loc_nx*loc_ny) + (loc_j-1)*loc_nx + (1 + off_x)
                
                global_k = z_start + k - 1
                global_j = y_start + j - 1
                global_start_idx = (global_k-1)*(nElem_x*nElem_y) + (global_j-1)*nElem_x + x_start
                
                filtered_full[global_start_idx : global_start_idx+core_dim_x-1] = 
                    res_local[loc_start_idx : loc_start_idx+core_dim_x-1]
            end
        end
        
        # Periodic cleanup to prevent fragmentation
        if blocks_processed % 10 == 0; CUDA.reclaim(); end
    end
    
    return filtered_full, total_time, (blocks_processed > 0 ? round(Int, total_iters/blocks_processed) : 0), 0.0
end


function apply_gpu_filter!(density_cpu::Vector{T}, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, elements_cpu, tol::T=1.0f-5) where T
    
    # 1. Attempt Single Pass first (Fastest)
    # Estimate size: 80 bytes/elem. 
    nElem = length(density_cpu)
    
    # Force a check of *actual* free memory
    GC.gc()
    CUDA.reclaim()
    
    free_mem = CUDA.available_memory()
    req_mem = nElem * 100 
    
    try
        if req_mem < (free_mem * 0.9)
            ws = setup_helmholtz_workspace(elements_cpu, T(dx), T(dy), T(dz), T(radius))
            return solve_helmholtz_on_gpu(density_cpu, ws, dx, dy, dz, tol)
        else
            # 2. Blocked Mode (Memory Efficient)
            return apply_blocked_gpu_filter!(density_cpu, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, tol)
        end
    catch e
        # 3. Ultimate Fallback (Safety Net)
        println("\n" * "!"^60)
        println("  [GPU Filter] CRITICAL ERROR: $(e).")
        println("  [GPU Filter] Falling back to CPU Filter to save simulation...")
        println("!"^60 * "\n")
        
        # Signal failure to TopOpt.jl, so it can run the CPU routine
        # We return a specific tuple that TopOpt will recognize as failure
        return Float32[], 0.0, 0, 0.0 
    end
end

end