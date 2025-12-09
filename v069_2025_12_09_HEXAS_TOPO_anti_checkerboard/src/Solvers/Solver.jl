# // # FILE: .\src\Solvers\Solver.jl
module Solver 
 
using CUDA 
using Printf
using ..Helpers  
using ..DirectSolver: solve_system as solve_system_direct 
using ..IterativeSolver: solve_system_iterative 
using ..MeshPruner 
 
export solve_system 
 
function choose_solver(nNodes, nElem, config) 
    solver_params = config["solver_parameters"] 
    configured_type = Symbol(lowercase(get(solver_params, "solver_type", "direct"))) 
 
    if configured_type == :direct 
        if nElem > 100_000 
            @warn "Direct solver requested for large mesh ($(nElem) elements). Switching to Matrix-Free iterative." 
            return :matrix_free 
        end 
        return :direct 
    elseif configured_type == :gpu 
        if CUDA.functional() && Helpers.has_enough_gpu_memory(nNodes, nElem, true) 
            return :gpu 
        else 
            @warn "Not enough GPU memory even for matrix-free. Falling back to CPU."
            return :matrix_free 
        end 
    elseif configured_type == :matrix_free 
        return :matrix_free 
    else 
        @warn "Unknown solver_type: $(configured_type). Defaulting to matrix_free." 
        return :matrix_free 
    end 
end 
 
function solve_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      E::Float32, 
                      nu::Float32, 
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32}; 
                      density::Vector{Float32}=nothing, 
                      config::Dict, 
                      min_stiffness_threshold::Float32=Float32(1.0e-3),
                      prune_voids::Bool=true,
                      u_prev::Vector{Float32}=Float32[]) 
                            
    active_system = nothing
    
    solve_nodes = nodes
    solve_elements = elements
    solve_bc = bc_indicator
    solve_F = F
    solve_density = density
    
    solve_u_guess = Float32[]

    if prune_voids && density !== nothing
        prune_threshold = min_stiffness_threshold * 1.01f0 
        nElem_total = size(elements, 1)
        nActive = count(d -> d > prune_threshold, density)
          
        if nActive < (nElem_total * 0.99)
            active_system = MeshPruner.prune_system(nodes, elements, density, prune_threshold, bc_indicator, F)
              
            solve_nodes = active_system.nodes
            solve_elements = active_system.elements
            solve_bc = active_system.bc_indicator
            solve_F = active_system.F
            solve_density = active_system.density
            
            if !isempty(u_prev) && length(u_prev) == length(F)
                nActiveNodes = length(active_system.new_to_old_node_map)
                solve_u_guess = zeros(Float32, nActiveNodes * 3)
                
                for (new_idx, old_idx) in enumerate(active_system.new_to_old_node_map)
                    base_old = 3 * (old_idx - 1)
                    base_new = 3 * (new_idx - 1)
                    solve_u_guess[base_new+1] = u_prev[base_old+1]
                    solve_u_guess[base_new+2] = u_prev[base_old+2]
                    solve_u_guess[base_new+3] = u_prev[base_old+3]
                end
            end
        else
            solve_u_guess = u_prev
        end
    else
        solve_u_guess = u_prev
    end

    nNodes_solve = size(solve_nodes, 1) 
    nElem_solve = size(solve_elements, 1) 
      
    solver_params = config["solver_parameters"] 
    solver_type = choose_solver(nNodes_solve, nElem_solve, config) 
      
    base_tol = Float32(get(solver_params, "tolerance", 1.0e-6)) 
    max_iter = Int(get(solver_params, "max_iterations", 1000)) 
    shift_factor = Float32(get(solver_params, "diagonal_shift_factor", 1.0e-6)) 
      
    iter_current = get(config, "current_outer_iter", 1)
    iter_max = get(config, "number_of_iterations", 30)
    
    progress = clamp(Float32(iter_current) / Float32(iter_max), 0f0, 1f0)
    
    start_tol_log = -5.0 
    end_tol_log = log10(base_tol)
    
    adaptive_tol = Float32(10.0^((1.0 - progress) * start_tol_log + progress * end_tol_log))
    
    if solver_type == :gpu
        gpu_limit = 5.0e-7
        if adaptive_tol < gpu_limit
            adaptive_tol = Float32(gpu_limit)
        end
        tol_str = @sprintf("%.1e", adaptive_tol)
        println("   [Solver] Adaptive Tol: $tol_str (Iter $iter_current/$iter_max)")
    end

    use_precond = true 
    
    U_solved_tuple = if solver_type == :direct 
        solve_system_direct(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                            density=solve_density, 
                            shift_factor=shift_factor, 
                            min_stiffness_threshold=min_stiffness_threshold) 
                             
    elseif solver_type == :gpu 
        gpu_method = Symbol(lowercase(get(solver_params, "gpu_method", "krylov"))) 
        krylov_solver = Symbol(lowercase(get(solver_params, "krylov_solver", "cg"))) 
 
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:gpu, max_iter=max_iter, tol=adaptive_tol, 
                             density=solve_density, 
                             use_precond=use_precond,  
                             gpu_method=gpu_method, krylov_solver=krylov_solver, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold, 
                             u_guess=solve_u_guess, 
                             config=config) 
                             
    else   
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:matrix_free, max_iter=max_iter, tol=adaptive_tol, 
                             use_precond=use_precond, 
                             density=solve_density, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold, 
                             config=config) 
    end 
 
    U_solved_vec = U_solved_tuple[1]
    res_val = U_solved_tuple[2]
    prec_str = U_solved_tuple[3]

    if active_system !== nothing
        U_full = MeshPruner.reconstruct_full_solution(U_solved_vec, active_system.new_to_old_node_map, size(nodes, 1))
        return (U_full, res_val, prec_str)
    else
        return U_solved_tuple
    end
end 
 
end