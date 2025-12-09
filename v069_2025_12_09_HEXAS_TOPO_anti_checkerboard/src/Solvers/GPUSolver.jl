# // # FILE: .\src\Solvers\GPUSolver.jl

module GPUSolver

using LinearAlgebra, Printf
using CUDA
using SparseArrays
using AlgebraicMultigrid 
using Base.Threads
using ..Element

export solve_system_gpu

# ------------------------------------------------------------------------------
# CONSTANTS & LOGGING UTILS
# ------------------------------------------------------------------------------
const C_RESET  = "\e[0m"
const C_BOLD   = "\e[1m"
const C_CYAN   = "\e[36m"
const C_GREEN  = "\e[32m"
const C_YELLOW = "\e[33m"
const C_RED    = "\e[31m"

function print_section_header(title::String, outer_iter::Any="?")
    width = 80
    println("\n" * C_CYAN * "="^width * C_RESET)
    full_title = "$title [Topo Opt Iter: $outer_iter]"
    padding = max(0, (width - length(full_title) - 2) ÷ 2)
    println(" "^padding * C_BOLD * full_title * C_RESET)
    println(C_CYAN * "="^width * C_RESET)
end

# ------------------------------------------------------------------------------
# GPU KERNELS
# ------------------------------------------------------------------------------

function matvec_kernel!(y::CuDeviceArray{T}, x::CuDeviceArray{T}, 
                        elements::CuDeviceArray{Int32}, 
                        Ke::CuDeviceArray{T},
                        factors::CuDeviceArray{T},
                        nActive::Int) where T
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nActive
        factor = factors[e]
        @inbounds begin
            n1 = elements[e, 1]; n2 = elements[e, 2]; n3 = elements[e, 3]; n4 = elements[e, 4]
            n5 = elements[e, 5]; n6 = elements[e, 6]; n7 = elements[e, 7]; n8 = elements[e, 8]
        end
        @inbounds begin
            x1 = x[3*(n1-1)+1]; x2 = x[3*(n1-1)+2]; x3 = x[3*(n1-1)+3]
            x4 = x[3*(n2-1)+1]; x5 = x[3*(n2-1)+2]; x6 = x[3*(n2-1)+3]
            x7 = x[3*(n3-1)+1]; x8 = x[3*(n3-1)+2]; x9 = x[3*(n3-1)+3]
            x10 = x[3*(n4-1)+1]; x11 = x[3*(n4-1)+2]; x12 = x[3*(n4-1)+3]
            x13 = x[3*(n5-1)+1]; x14 = x[3*(n5-1)+2]; x15 = x[3*(n5-1)+3]
            x16 = x[3*(n6-1)+1]; x17 = x[3*(n6-1)+2]; x18 = x[3*(n6-1)+3]
            x19 = x[3*(n7-1)+1]; x20 = x[3*(n7-1)+2]; x21 = x[3*(n7-1)+3]
            x22 = x[3*(n8-1)+1]; x23 = x[3*(n8-1)+2]; x24 = x[3*(n8-1)+3]
        end
        
        @inbounds for i in 1:24
            val = T(0.0)
            val += Ke[i, 1] * x1 + Ke[i, 2] * x2 + Ke[i, 3] * x3 + Ke[i, 4] * x4
            val += Ke[i, 5] * x5 + Ke[i, 6] * x6 + Ke[i, 7] * x7 + Ke[i, 8] * x8
            val += Ke[i, 9] * x9 + Ke[i, 10] * x10 + Ke[i, 11] * x11 + Ke[i, 12] * x12
            val += Ke[i, 13] * x13 + Ke[i, 14] * x14 + Ke[i, 15] * x15 + Ke[i, 16] * x16
            val += Ke[i, 17] * x17 + Ke[i, 18] * x18 + Ke[i, 19] * x19 + Ke[i, 20] * x20
            val += Ke[i, 21] * x21 + Ke[i, 22] * x22 + Ke[i, 23] * x23 + Ke[i, 24] * x24
            
            node_idx = (i - 1) ÷ 3 + 1
            dof_local = (i - 1) % 3 + 1
            node = (node_idx == 1 ? n1 : node_idx == 2 ? n2 : node_idx == 3 ? n3 : 
                    node_idx == 4 ? n4 : node_idx == 5 ? n5 : node_idx == 6 ? n6 : 
                    node_idx == 7 ? n7 : n8)
            global_dof = 3 * (node - 1) + dof_local
            CUDA.atomic_add!(pointer(y, global_dof), factor * val)
        end
    end
    return nothing
end

function expand_kernel!(x_full::CuDeviceArray{T}, x_free::CuDeviceArray{T}, 
                        free_to_full::CuDeviceArray{Int32}, n_free::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_full[free_to_full[idx]] = x_free[idx]
    end
    return nothing
end

function contract_kernel!(x_free::CuDeviceArray{T}, x_full::CuDeviceArray{T},
                          free_to_full::CuDeviceArray{Int32}, n_free::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_free[idx] = x_full[free_to_full[idx]]
    end
    return nothing
end

function jacobi_precond_kernel!(z::CuDeviceArray{T}, r::CuDeviceArray{T}, 
                                M_inv::CuDeviceArray{T}, n::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds z[idx] = r[idx] * M_inv[idx]
    end
    return nothing
end

# ------------------------------------------------------------------------------
# SETUP UTILS
# ------------------------------------------------------------------------------

function get_free_dofs(bc_indicator::Matrix{T}) where T
    nNodes = size(bc_indicator, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    @inbounds for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    return findall(!, constrained)
end

function setup_matrix_free_operator(nodes::Matrix{T}, elements::Matrix{Int}, 
                                    E::T, nu::T, density::Vector{T}, 
                                    min_stiffness_threshold::T) where T
    setup_start = time()
    active_mask = density .>= min_stiffness_threshold
    active_indices = findall(active_mask)
    nActive = length(active_indices)
    
    if nActive == 0
        error(C_RED * "❌ No active elements found (Threshold: $min_stiffness_threshold)." * C_RESET)
    end

    n1 = nodes[elements[1,1], :]
    n2 = nodes[elements[1,2], :] 
    n4 = nodes[elements[1,4], :] 
    n5 = nodes[elements[1,5], :] 
    
    dx = norm(n2 - n1)
    dy = norm(n4 - n1)
    dz = norm(n5 - n1)
    
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, nu)
    active_elements = elements[active_indices, :]
    element_factors = E .* density[active_indices]
    
    return active_elements, Ke_base, element_factors, active_indices, setup_start
end

function apply_matrix_free_operator!(y::CuVector{T}, x::CuVector{T},
                                     elements_gpu::CuMatrix{Int32},
                                     Ke_gpu::CuMatrix{T},
                                     factors_gpu::CuVector{T},
                                     nActive::Int) where T
    fill!(y, T(0.0))
    threads = 256
    blocks = cld(nActive, threads)
    @cuda threads=threads blocks=blocks matvec_kernel!(y, x, elements_gpu, Ke_gpu, factors_gpu, nActive)
    CUDA.synchronize()
end

# ------------------------------------------------------------------------------
# AMG HELPERS (PARALLEL CPU ASSEMBLY)
# ------------------------------------------------------------------------------

function assemble_cpu_matrix_for_amg(nodes::Matrix{T}, 
                                     active_elements::Matrix{Int}, 
                                     active_indices::Vector{Int}, 
                                     density::Vector{T}, 
                                     E::T, nu::T,
                                     free_dofs::Vector{Int}) where T
    
    println("  [AMG Setup] Assembling sparse matrix on CPU (Parallel)...")
    t_asm = time()
    
    n_full_dofs = size(nodes, 1) * 3
    
    # 1. Fast Lookup Table
    full_to_reduced = zeros(Int32, n_full_dofs)
    Threads.@threads for i in 1:length(free_dofs)
        @inbounds full_to_reduced[free_dofs[i]] = Int32(i)
    end

    nActive = length(active_indices)
    
    # 2. Pre-calculate Element Stiffness Matrix (Ke)
    n1 = nodes[active_elements[1,1], :]
    n2 = nodes[active_elements[1,2], :] 
    n4 = nodes[active_elements[1,4], :] 
    n5 = nodes[active_elements[1,5], :] 
    dx = norm(n2 - n1); dy = norm(n4 - n1); dz = norm(n5 - n1)
    Ke_base = Element.get_canonical_stiffness(Float32(dx), Float32(dy), Float32(dz), Float32(nu))
    
    # 3. Parallel Assembly Setup
    n_threads = Threads.nthreads()
    
    entries_per_elem = 576 # 24 x 24
    total_entries = nActive * entries_per_elem
    
    # Pre-allocate global arrays to avoid concatenation overhead
    I_global = Vector{Int32}(undef, total_entries)
    J_global = Vector{Int32}(undef, total_entries)
    V_global = Vector{Float64}(undef, total_entries)
    
    chunk_size = div(nActive, n_threads)
    
    Threads.@threads for t in 1:n_threads
        start_idx = (t - 1) * chunk_size + 1
        end_idx = (t == n_threads) ? nActive : t * chunk_size
        
        # Determine global offset for this thread
        global_ptr = (start_idx - 1) * entries_per_elem
        
        for idx in start_idx:end_idx
            e = active_indices[idx] 
            factor = Float64(E * density[e]) 
            conn = view(active_elements, idx, :)
            
            for i in 1:8
                node_i = conn[i]
                base_i = 3 * (node_i - 1)
                
                for d_i in 1:3
                    full_dof_i = base_i + d_i
                    red_dof_i = full_to_reduced[full_dof_i]
                    local_dof_i = 3*(i-1) + d_i
                    
                    for j in 1:8
                        node_j = conn[j]
                        base_j = 3 * (node_j - 1)
                        
                        for d_j in 1:3
                            full_dof_j = base_j + d_j
                            red_dof_j = full_to_reduced[full_dof_j]
                            local_dof_j = 3*(j-1) + d_j
                            
                            global_ptr += 1
                            
                            # Write dummy 0.0 if constrained to keep array aligned without branching
                            if red_dof_i > 0 && red_dof_j > 0
                                @inbounds I_global[global_ptr] = red_dof_i
                                @inbounds J_global[global_ptr] = red_dof_j
                                @inbounds V_global[global_ptr] = factor * Ke_base[local_dof_i, local_dof_j]
                            else
                                @inbounds I_global[global_ptr] = 1
                                @inbounds J_global[global_ptr] = 1
                                @inbounds V_global[global_ptr] = 0.0
                            end
                        end
                    end
                end
            end
        end
    end
    
    n_reduced = length(free_dofs)
    
    # sparse() handles summing duplicates (including the dummy 0.0s) efficiently
    K_reduced = sparse(I_global, J_global, V_global, n_reduced, n_reduced)
    
    # Regularization
    K_reduced += sparse(I, n_reduced, n_reduced) * 1e-9

    println("  [AMG Setup] CPU Parallel Assembly finished in $(round(time() - t_asm, digits=3))s.")
    return K_reduced
end

# ------------------------------------------------------------------------------
# MAIN SOLVER ROUTINE
# ------------------------------------------------------------------------------

function gpu_matrix_free_cg_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                                  bc_indicator::Matrix{T}, f::Vector{T},
                                  density::Vector{T};
                                  max_iter=40000, tol=1e-6, 
                                  shift_factor::T=Float32(1.0e-7),
                                  min_stiffness_threshold::T=Float32(1.0e-3),
                                  u_guess::Vector{T}=T[], 
                                  config::Dict=Dict()) where T
    CUDA.allowscalar(false)
    outer_iter = get(config, "current_outer_iter", "?")
    
    active_elements, Ke_base, element_factors, active_indices, setup_start = setup_matrix_free_operator(
        nodes, elements, E, nu, density, min_stiffness_threshold
    )
    
    nActive = length(active_indices)
    ndof = size(nodes, 1) * 3
    free_dofs = get_free_dofs(bc_indicator)
    n_free = length(free_dofs)
    free_to_full = Int32.(free_dofs)
    
    print_section_header("GPU SOLVER - ADAPTIVE PCG", outer_iter)
    @printf("  Free DOFs:             %12d\n", n_free)
    @printf("  Active Elements:       %12d\n", nActive)
    @printf("  Setup Time:            %12.3f s\n", time() - setup_start)

    # 1. Compute Diagonal for Jacobi
    diag_full = zeros(T, ndof)
    @inbounds for t_idx in 1:nActive
        e = active_indices[t_idx]
        factor = E * density[e]
        conn = view(active_elements, t_idx, :)
        for i in 1:8
            node = conn[i]
            local_base = 3 * (i - 1)
            for dof in 1:3
                global_dof = 3 * (node - 1) + dof
                local_dof = local_base + dof
                diag_full[global_dof] += Ke_base[local_dof, local_dof] * factor
            end
        end
    end
    diag_free_base = diag_full[free_dofs]
    max_diag_val = maximum(abs.(diag_free_base))

    # 2. Config & Strategy
    solver_conf = get(config, "solver_parameters", Dict())
    pcs_config_str = lowercase(get(solver_conf, "preconditioner", "jacobi"))
    
    use_amg_initially = (pcs_config_str == "amg")
    allow_amg_fallback = (pcs_config_str == "jacobi_amg") 
    
    active_prec_is_amg = use_amg_initially
    ml = nothing
    amg_pl = nothing
    
    # Buffers for CPU <-> GPU transfer in hybrid mode
    r_cpu = Vector{Float64}(undef, n_free)
    z_cpu = Vector{Float64}(undef, n_free)

    if active_prec_is_amg
        try
            println("  [AMG Setup] Initializing Algebraic Multigrid Hierarchy...")
            K_cpu = assemble_cpu_matrix_for_amg(nodes, active_elements, active_indices, density, E, nu, free_dofs)
            ml = AlgebraicMultigrid.smoothed_aggregation(K_cpu)
            amg_pl = AlgebraicMultigrid.aspreconditioner(ml)
            println("  [AMG Setup] Hierarchy built. Levels: $(length(ml)). Grid complexity: $(AlgebraicMultigrid.grid_complexity(ml))")
        catch e
            println(C_RED * "  [AMG Setup] FAILED: $e" * C_RESET)
            println(C_YELLOW * "  [AMG Setup] Falling back to Jacobi preconditioner." * C_RESET)
            active_prec_is_amg = false
            allow_amg_fallback = false 
            ml = nothing
        end
    else
        if allow_amg_fallback
            println("  [Preconditioner] Strategy: Jacobi (Adaptive -> AMG on instability).")
        else
            println("  [Preconditioner] Strategy: Standard Jacobi (Diagonal) scaling.")
        end
    end

    # 3. GPU Allocation
    elements_gpu = CuArray{Int32}(active_elements)
    Ke_gpu = CuArray(Ke_base)
    factors_gpu = CuArray(element_factors)
    free_to_full_gpu = CuArray(free_to_full)
    b_gpu = CuVector(f[free_dofs])
    
    x_gpu = CUDA.zeros(T, n_free)
    r_gpu = CUDA.zeros(T, n_free)
    z_gpu = CUDA.zeros(T, n_free)
    p_gpu = CUDA.zeros(T, n_free)
    Ap_free_gpu = CUDA.zeros(T, n_free)
    x_full_gpu = CUDA.zeros(T, ndof)
    Ap_full_gpu = CUDA.zeros(T, ndof)
    M_inv_gpu = CUDA.zeros(T, n_free) 

    norm_b = norm(b_gpu)
    if norm_b == 0; return (zeros(T, ndof), 0.0, "None"); end

    threads_map = 256
    blocks_map = cld(n_free, threads_map)

    current_shift_factor = shift_factor
    max_retries = 3
    solve_success = false
    global_best_x = zeros(T, ndof)
    global_lowest_res = T(Inf)
    
    final_rel_res_val = 0.0
    prec_name_str = "None"

    # --------------------------------------------------------------------------
    # RETRY LOOP
    # --------------------------------------------------------------------------
    for attempt in 1:max_retries
        actual_shift = current_shift_factor * max_diag_val
        
        if attempt > 1
            @printf("\n  %s>>> RETRY %d/%d | Shift Factor: %.1e (Val: %.2e)%s\n", 
                    C_YELLOW, attempt, max_retries, current_shift_factor, actual_shift, C_RESET)
        else
            @printf("  Diagonal shift:         %.4e\n", actual_shift)
        end

        # Setup Preconditioner on GPU (if Jacobi)
        if !active_prec_is_amg
            M_inv_cpu = 1.0f0 ./ (diag_free_base .+ actual_shift)
            copyto!(M_inv_gpu, M_inv_cpu)
        end

        # Reset Solver State
        fill!(x_gpu, T(0.0))
        use_warm_start = !isempty(u_guess) && length(u_guess) == ndof && attempt == 1
        if use_warm_start
            copyto!(x_gpu, u_guess[free_dofs])
            println("  [Warm Start] Using previous solution.")
        end

        # Compute Initial Residual
        fill!(x_full_gpu, T(0.0))
        @cuda threads=threads_map blocks=blocks_map expand_kernel!(x_full_gpu, x_gpu, free_to_full_gpu, n_free)
        apply_matrix_free_operator!(Ap_full_gpu, x_full_gpu, elements_gpu, Ke_gpu, factors_gpu, nActive)
        @cuda threads=threads_map blocks=blocks_map contract_kernel!(Ap_free_gpu, Ap_full_gpu, free_to_full_gpu, n_free)
        CUDA.synchronize()
        
        Ap_free_gpu .+= actual_shift .* x_gpu
        r_gpu .= b_gpu .- Ap_free_gpu
        
        # Apply Preconditioner (z = M^-1 * r)
        if active_prec_is_amg
            copyto!(r_cpu, r_gpu)
            ldiv!(z_cpu, amg_pl, r_cpu)
            copyto!(z_gpu, z_cpu)
        else
            @cuda threads=threads_map blocks=blocks_map jacobi_precond_kernel!(z_gpu, r_gpu, M_inv_gpu, n_free)
            CUDA.synchronize()
        end

        p_gpu .= z_gpu
        rz_old = dot(r_gpu, z_gpu)
        
        # Track Best within this attempt
        best_res_attempt = T(Inf)
        best_rel_res_attempt = T(Inf)
        best_x_attempt = CUDA.copy(x_gpu)
        
        cg_start = time()
        diverged_in_attempt = false
        
        @printf("\n  %s%8s %12s %12s %10s%s\n", 
                C_BOLD, "Iter", "Res Norm", "Rel Res", "Time(s)", C_RESET)
        @printf("  %s\n", "-"^46)

        # ----------------------------------------------------------------------
        # CG ITERATION LOOP
        # ----------------------------------------------------------------------
        for iter in 1:max_iter
            fill!(x_full_gpu, T(0.0))
            @cuda threads=threads_map blocks=blocks_map expand_kernel!(x_full_gpu, p_gpu, free_to_full_gpu, n_free)
            apply_matrix_free_operator!(Ap_full_gpu, x_full_gpu, elements_gpu, Ke_gpu, factors_gpu, nActive)
            @cuda threads=threads_map blocks=blocks_map contract_kernel!(Ap_free_gpu, Ap_full_gpu, free_to_full_gpu, n_free)
            CUDA.synchronize()

            Ap_free_gpu .+= actual_shift .* p_gpu
            denom = dot(p_gpu, Ap_free_gpu)

            if abs(denom) < 1e-20 || isnan(denom)
                @warn "      [CG BREAKDOWN] Zero/NaN denominator."
                diverged_in_attempt = true
                break
            end

            alpha = rz_old / denom
            x_gpu .+= alpha .* p_gpu
            r_gpu .-= alpha .* Ap_free_gpu

            res_norm_sq = dot(r_gpu, r_gpu)
            res_norm = sqrt(res_norm_sq)
            rel_res = res_norm / norm_b
            
            final_rel_res_val = rel_res

            # DIVERGENCE CHECK (Strictness increases if we have a fallback)
            # If we were doing well, but then exploded > 100x our best, abort.
            if rel_res > (best_rel_res_attempt * 100.0) && iter > 50
                 @printf("  %s%8d %12.4e [DIVERGED - INSTABILITY DETECTED]%s\n", C_RED, iter, res_norm, C_RESET)
                 diverged_in_attempt = true
                 break
            end

            if isnan(res_norm) || isinf(res_norm) || res_norm > (norm_b * 1000.0)
                @printf("  %s%8d %12.4e [DIVERGED - EXPLOSION]%s\n", C_RED, iter, res_norm, C_RESET)
                diverged_in_attempt = true
                break
            end

            if res_norm < best_res_attempt
                best_res_attempt = res_norm
                best_rel_res_attempt = rel_res
                # Only snapshot the vector if it's somewhat decent to save bandwidth
                if rel_res < 1e-2 
                     best_x_attempt .= x_gpu
                end
            end

            if (iter == 1) || (iter % 1000 == 0) || (rel_res < tol)
                color = (rel_res < tol) ? C_GREEN : C_RESET
                @printf("  %s%8d %12.4e %12.4e %10.3f%s\n", 
                        color, iter, res_norm, rel_res, time() - cg_start, C_RESET)
            end

            if rel_res < tol
                solve_success = true
                best_x_attempt .= x_gpu 
                break
            end

            if active_prec_is_amg
                copyto!(r_cpu, r_gpu)      
                ldiv!(z_cpu, amg_pl, r_cpu) 
                copyto!(z_gpu, z_cpu)      
            else
                @cuda threads=threads_map blocks=blocks_map jacobi_precond_kernel!(z_gpu, r_gpu, M_inv_gpu, n_free)
                CUDA.synchronize()
            end

            rz_new = dot(r_gpu, z_gpu)
            beta = rz_new / rz_old
            p_gpu .= z_gpu .+ beta .* p_gpu
            rz_old = rz_new
        end
        # ----------------------------------------------------------------------

        prec_name_str = active_prec_is_amg ? "AMG" : "Jacobi"

        if solve_success
            copyto!(x_gpu, best_x_attempt)
            break
        end

        # ----------------------------------------------------------------------
        # FAILURE HANDLING & SWITCHING LOGIC
        # ----------------------------------------------------------------------
        
        # We enforce stricter standards if we have a powerful fallback option.
        # If we can switch to AMG, we DO NOT accept "acceptable stagnation" from Jacobi.
        can_switch = allow_amg_fallback && !active_prec_is_amg
        
        # If we diverged (hit infinity/NaN or exploded relative to best), that is a hard fail.
        # If we didn't diverge, but just stagnated, we check tolerance.
        
        failed_badly = diverged_in_attempt || (best_rel_res_attempt > 0.1) # Hard limit 10%

        if !failed_badly && !can_switch
            # If we cannot switch, we might have to accept mediocrity.
            if best_rel_res_attempt < 5e-2
                @printf("  %sResidual %.2e > Tol, but acceptable stagnation (No fallback). Accepting.%s\n", C_YELLOW, best_rel_res_attempt, C_RESET)
                solve_success = true
                copyto!(x_gpu, best_x_attempt)
                break
            end
        end

        if failed_badly || (can_switch && best_rel_res_attempt > tol)
            # Logic: If we failed OR if we just stagnated but CAN switch, we switch.
            
            if can_switch
                println(C_CYAN * "\n  [Adaptive Strategy] Instability or Stagnation detected with Jacobi." * C_RESET)
                println(C_CYAN * "  [Adaptive Strategy] Switching preconditioner to AMG for next attempt..." * C_RESET)
                try
                    K_cpu = assemble_cpu_matrix_for_amg(nodes, active_elements, active_indices, density, E, nu, free_dofs)
                    ml = AlgebraicMultigrid.smoothed_aggregation(K_cpu)
                    amg_pl = AlgebraicMultigrid.aspreconditioner(ml)
                    active_prec_is_amg = true 
                catch e
                    println(C_RED * "  [Adaptive Strategy] AMG Setup Failed: $e" * C_RESET)
                    allow_amg_fallback = false
                end
            elseif active_prec_is_amg
                println(C_RED * "  [Adaptive Strategy] AMG failed. Falling back to Jacobi." * C_RESET)
                active_prec_is_amg = false
                allow_amg_fallback = false
            end
            
            @printf("  %sAttempt %d failed (RelRes: %.2e). Increasing shift/Switching.%s\n", C_YELLOW, attempt, best_rel_res_attempt, C_RESET)
            current_shift_factor *= 10.0
        else
             # If we get here, it means we didn't fail badly, but we didn't converge, 
             # AND we can't switch. We check for "Acceptable Stagnation" one last time.
             if best_rel_res_attempt < global_lowest_res
                global_lowest_res = best_rel_res_attempt
                x_gpu .= best_x_attempt
                expand_gpu = CUDA.zeros(T, ndof)
                @cuda threads=threads_map blocks=blocks_map expand_kernel!(expand_gpu, x_gpu, free_to_full_gpu, n_free)
                global_best_x = Array(expand_gpu)
            end
            current_shift_factor *= 5.0
        end
    end

    println(C_CYAN * "-"^80 * C_RESET)
    x_full = zeros(T, ndof)
    
    if solve_success
        fill!(x_full_gpu, T(0.0))
        @cuda threads=threads_map blocks=blocks_map expand_kernel!(x_full_gpu, x_gpu, free_to_full_gpu, n_free)
        copyto!(x_full, x_full_gpu)
    else
        @warn "All solver attempts failed. Returning best stagnated result."
        x_full = global_best_x
    end

    return (x_full, final_rel_res_val, prec_name_str)
end

function solve_system_gpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T},
                          density::Vector{T};
                          max_iter=40000, tol=1e-6, 
                          method=:native, solver=:cg, use_precond=true,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3),
                          u_guess::Vector{T}=T[], 
                          config::Dict=Dict()) where T
    if !CUDA.functional(); error("CUDA not found."); end
    
    return gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc_indicator, f, density,
                                    max_iter=max_iter, tol=tol, shift_factor=shift_factor,
                                    min_stiffness_threshold=min_stiffness_threshold,
                                    u_guess=u_guess,
                                    config=config)
end

end