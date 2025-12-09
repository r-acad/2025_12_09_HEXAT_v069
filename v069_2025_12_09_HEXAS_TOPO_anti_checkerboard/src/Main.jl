# // # FILE: .\src\Main.jl

using Pkg

const PROJECT_ROOT = joinpath(@__DIR__, "..")

const REQUIRED_PACKAGES = [
    "CUDA", 
    "JSON", 
    "JSON3", 
    "Krylov", 
    "LinearOperators", 
    "MarchingCubes", 
    "YAML", 
    "AlgebraicMultigrid", 
    "SparseArrays"
]

function check_and_install_packages()
    println(">>> [SETUP] Verifying dependencies...")
    project_deps = Pkg.project().dependencies
    
    missing_pkgs = []
    for pkg_name in REQUIRED_PACKAGES
        if !haskey(project_deps, pkg_name)
            push!(missing_pkgs, pkg_name)
        end
    end

    if !isempty(missing_pkgs)
        println(">>> [SETUP] The following packages are missing from the environment: $missing_pkgs")
        println(">>> [SETUP] Adding them now (relaxed constraints)...")
        try
            for pkg in missing_pkgs
                Pkg.add(pkg)
            end
        catch e
            println("!!! Error adding packages: $e")
        end
    else
        println(">>> [SETUP] Dependencies appear correct.")
    end
end

check_and_install_packages()

println("\n>>> SCRIPT START: Loading Modules...")

module HEXA

using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics 
using CUDA
using YAML
using AlgebraicMultigrid

using ..Main: PROJECT_ROOT

include("Utils/Diagnostics.jl")
include("Utils/Helpers.jl")
using .Diagnostics
using .Helpers

include("Core/Element.jl")
include("Core/Boundary.jl")
include("Core/Stress.jl")

using .Element
using .Boundary
using .Stress

include("Mesh/Mesh.jl")
include("Mesh/MeshUtilities.jl")
include("Mesh/MeshPruner.jl") 
include("Mesh/MeshRefiner.jl") 
include("Mesh/MeshShapeProcessing.jl") 

using .Mesh
using .MeshUtilities
using .MeshPruner 
using .MeshRefiner 
using .MeshShapeProcessing

include("Solvers/CPUSolver.jl")
include("Solvers/GPUSolver.jl")
include("Solvers/DirectSolver.jl")
include("Solvers/IterativeSolver.jl")
include("Solvers/Solver.jl") 

using .CPUSolver
using .GPUSolver
using .DirectSolver
using .IterativeSolver
using .Solver

include("IO/Configuration.jl")
include("IO/ExportVTK.jl")
include("IO/Postprocessing.jl")
include("Optimization/GPUHelmholtz.jl") 
include("Optimization/TopOpt.jl") 

using .Configuration
using .ExportVTK
using .Postprocessing
using .TopologyOptimization 

function __init__()
    Diagnostics.log_status("HEXA Finite Element Solver initialized")
    Helpers.clear_gpu_memory()
end


function apply_hardware_profile!(config::Dict)
    gpu_type = get(config, "gpu_profile", "RTX")
    println("\n>>> [HARDWARE] Detected Profile in Config: $gpu_type")
    growth = get(config, "growth_settings", Dict())
    solver = get(config, "solver_parameters", Dict())
    
    if uppercase(gpu_type) in ["H", "H200", "H100"]
        println("    -> High-Performance Data Center GPU Detected (H-Series).")
        growth["max_background_elements"] = 1_500_000_000 
        growth["max_growth_rate"] = 1.5       
        growth["gpu_solver_safety_factor"] = 0.98 
        solver["tolerance"] = 1.0e-12 
        solver["diagonal_shift_factor"] = 1.0e-10
        solver["solver_type"] = "gpu"
    elseif uppercase(gpu_type) == "V100"
        println("    -> Legacy Data Center GPU Detected (V100).")
        growth["max_background_elements"] = 150_000_000 
        growth["max_growth_rate"] = 1.3       
        growth["gpu_solver_safety_factor"] = 0.95 
        solver["tolerance"] = 1.0e-10 
        solver["diagonal_shift_factor"] = 1.0e-9
        solver["solver_type"] = "gpu"
    elseif uppercase(gpu_type) == "RTX"
        println("    -> Consumer/Workstation GPU Detected (RTX-Series).")
        growth["max_background_elements"] = 200_000_000
        growth["max_growth_rate"] = 1.2       
        growth["gpu_solver_safety_factor"] = 0.90 
        solver["tolerance"] = 1.0e-6
        solver["solver_type"] = "gpu"
    else
        println("    -> Unknown GPU profile '$gpu_type'. Defaulting to RTX safe mode.")
        growth["max_background_elements"] = 100_000_000
    end
    
    config["growth_settings"] = growth
    config["solver_parameters"] = solver
    config["hardware_profile_applied"] = gpu_type
end

function run_main(input_file=nothing)
    try
        _run_safe(input_file)
    catch e
        if isa(e, InterruptException)
            println("\n" * "="^60)
            println(">>> [USER INTERRUPT] Simulation stopped by user.")
            println(">>> Check RESULTS directory for the last successful iteration.")
            println("="^60)
        else
            println("\n" * "!"^60)
            println("!!! FATAL ERROR DETECTED !!!")
            println("!"^60)
            showerror(stderr, e, catch_backtrace())
        end
    end
end

function _run_safe(input_file)
    println(">>> [INIT] Clearing GPU Memory from previous runs...")
    if CUDA.functional()
        Helpers.clear_gpu_memory()
        CUDA.device!(0) 
        name = CUDA.name(CUDA.device())
        mem = CUDA.total_memory() / 1024^3
        println(">>> [INIT] GPU Detected: $name ($(@sprintf("%.2f", mem)) GB VRAM)")
    end
    GC.gc()
    
    if input_file === nothing
        input_file = joinpath(PROJECT_ROOT, "configs", "default.yaml")
    end
    
    if !isfile(input_file)
        error("Input file not found: $input_file")
    end

    println(">>> [INIT] Loading configuration from: $input_file")
    current_config = Configuration.load_configuration(input_file)
    apply_hardware_profile!(current_config)
    
    restart_conf = get(current_config, "restart_configuration", Dict())
    enable_restart = get(restart_conf, "enable_restart", false)
    restart_path = get(restart_conf, "file_path", "")
    
    config = Dict{Any,Any}()
    density = Float32[]
    start_iter = 1
    restart_radius = 0.0f0
    restart_threshold = 0.0f0
    is_restart_active = false

    if enable_restart
        if isfile(restart_path)
            println("\n>>> RESTART MODE ACTIVE")
            println("    Loading checkpoint: $restart_path")
            saved_config, density, saved_iter, restart_radius, restart_threshold = Configuration.load_checkpoint(restart_path)
            config = merge(saved_config, current_config)
            apply_hardware_profile!(config)
            start_iter = saved_iter + 1
            is_restart_active = true
        else
            println("\n!!! WARNING: Restart requested but file not found: '$restart_path'")
            println("!!! Falling back to FRESH START.")
            config = current_config
            is_restart_active = false
        end
    else
        println("\n>>> FRESH START MODE")
        config = current_config
        is_restart_active = false
    end
    
    hard_stop_iter = get(config, "hard_stop_after_iteration", -1)
    if hard_stop_iter > -1
        println(">>> HARD STOP ENABLED: Execution will stop after iteration $hard_stop_iter.")
    end

    out_settings = get(config, "output_settings", Dict())
    default_freq = get(out_settings, "export_frequency", 5)
    save_bin_freq = get(out_settings, "save_bin_frequency", default_freq)
    save_stl_freq = get(out_settings, "save_STL_frequency", default_freq)
    save_vtk_freq = get(out_settings, "save_VTK_frequency", default_freq)

    RESULTS_DIR = joinpath(PROJECT_ROOT, "RESULTS")
    if !isdir(RESULTS_DIR); mkpath(RESULTS_DIR); end

    raw_log_name = get(out_settings, "log_filename", "simulation_log.txt")
    log_filename = joinpath(PROJECT_ROOT, basename(raw_log_name))
    iso_threshold_val = get(out_settings, "iso_surface_threshold", 0.8)
    iso_threshold = Float32(iso_threshold_val)
    
    if !is_restart_active
        Diagnostics.init_log_file(log_filename, config)
    else
        Diagnostics.log_status("--- RESTARTING SIMULATION (Iter $start_iter) ---")
    end
    
    geom = Configuration.setup_geometry(config)
    nodes, elements, dims = generate_mesh(geom.nElem_x, geom.nElem_y, geom.nElem_z; dx = geom.dx, dy = geom.dy, dz = geom.dz)
    initial_target_count = size(elements, 1)
    
    if is_restart_active && length(density) != initial_target_count
        error("Restart Mismatch: Checkpoint density size ($(length(density))) != Generated Mesh size ($initial_target_count).")
    end
    
    domain_bounds = (min_pt=[0.0f0,0.0f0,0.0f0], len_x=geom.dx*geom.nElem_x, len_y=geom.dy*geom.nElem_y, len_z=geom.dz*geom.nElem_z)
    
    config["geometry"]["nElem_x_computed"] = geom.nElem_x
    config["geometry"]["nElem_y_computed"] = geom.nElem_y
    config["geometry"]["nElem_z_computed"] = geom.nElem_z
    config["geometry"]["dx_computed"] = geom.dx
    config["geometry"]["dy_computed"] = geom.dy
    config["geometry"]["dz_computed"] = geom.dz
    config["geometry"]["max_domain_dim"] = geom.max_domain_dim
    
    nNodes = size(nodes, 1)
    ndof = nNodes * 3
    bc_data = config["boundary_conditions"]
    forces_data = config["external_forces"]
    
    bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
    F_external = zeros(Float32, ndof)
    Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
    println("Boundary Conditions & External Forces Mapped.")

    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    material_density = Float32(get(config["material"], "material_density", 0.0))
    gravity_accel = Float32(get(config["material"], "gravity_acceleration", 9.81))
    delta_T = Float32(get(config["material"], "delta_temperature", 0.0))
    if abs(delta_T) > 1e-6; println(">>> THERMOELASTICITY ENABLED: Delta T = $delta_T"); end

    original_density = ones(Float32, size(elements, 1)) 
    protected_elements_mask = falses(size(elements, 1)) 
    alpha_field = zeros(Float32, size(elements, 1))

    if !is_restart_active
        density, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
    else
        _, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
    end
    
    opt_params = config["optimization_parameters"]
    min_density = Float32(get(opt_params, "min_density", 1.0e-3))
    max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))
    base_name = splitext(basename(input_file))[1]
    
    growth_conf = get(config, "growth_settings", Dict())
    nominal_iterations = get(config, "number_of_iterations", 30)
    raw_active_target = get(growth_conf, "target_active_elements", initial_target_count)
    final_target_active = isa(raw_active_target, String) ? parse(Int, replace(raw_active_target, "_" => "")) : Int(raw_active_target)

    max_growth_rate = Float64(get(growth_conf, "max_growth_rate", 1.2))
    raw_bg_limit = get(growth_conf, "max_background_elements", 800_000_000)
    hard_elem_limit = isa(raw_bg_limit, String) ? parse(Int, replace(raw_bg_limit, "_" => "")) : Int(raw_bg_limit)

    println(">>> [LIMITS] Hard Element Limit set to: $(Base.format_bytes(hard_elem_limit * 100)) approx ($hard_elem_limit elems)")

    l1_stress_allowable = Float32(get(config, "l1_stress_allowable", 1.0))
    if l1_stress_allowable == 0.0f0; l1_stress_allowable = 1.0f0; end

    U_full = zeros(Float32, ndof)
    max_change = 1.0f0
    filter_R = is_restart_active ? restart_radius : 0.0f0
    curr_threshold = is_restart_active ? restart_threshold : 0.0f0
    
    iter = start_iter
    keep_running = true
    is_annealing = false
    
    println("\n--- Starting Optimization (Iter $iter / $nominal_iterations) ---")
    println("Log File: $log_filename")
    
    while keep_running
        iter_start_time = time()
        status_msg = "Nominal"
        current_target_active = final_target_active
        phase_refinement_needed = false
        gravity_scale = 0.0f0
        
        if iter <= nominal_iterations
            status_msg = "Nominal"
            is_annealing = false
            current_target_active = initial_target_count 
            current_active = count(d -> d > 0.01, density)
            nominal_ref_thresh = Float64(get(growth_conf, "nominal_refinement_threshold", 0.8))
            if current_active < (initial_target_count * nominal_ref_thresh); phase_refinement_needed = true; end
            gravity_scale = 0.0f0
        else
            status_msg = "Growth/Ann"
            is_annealing = true 
            current_active = count(d -> d > 0.01, density)
            if current_active < (final_target_active * 0.90); phase_refinement_needed = true; end
            gravity_scale = 1.0f0 
        end

        if phase_refinement_needed
             prev_elem_count = size(elements, 1)
             nodes, elements, density, alpha_field, dims = MeshRefiner.refine_mesh_and_fields(
                nodes, elements, density, alpha_field, dims, current_target_active, domain_bounds;
                max_growth_rate = max_growth_rate, hard_element_limit = hard_elem_limit
            )
            GC.gc()
            
            if size(elements, 1) > prev_elem_count
                status_msg = "Refined"
                nElem_x_new, nElem_y_new, nElem_z_new = dims[1]-1, dims[2]-1, dims[3]-1
                current_dx = domain_bounds.len_x / nElem_x_new
                current_dy = domain_bounds.len_y / nElem_y_new
                current_dz = domain_bounds.len_z / nElem_z_new
                
                config["geometry"]["nElem_x_computed"] = nElem_x_new
                config["geometry"]["nElem_y_computed"] = nElem_y_new
                config["geometry"]["nElem_z_computed"] = nElem_z_new
                config["geometry"]["dx_computed"] = current_dx
                config["geometry"]["dy_computed"] = current_dy
                config["geometry"]["dz_computed"] = current_dz
                
                geom = (nElem_x=nElem_x_new, nElem_y=nElem_y_new, nElem_z=nElem_z_new, dx=current_dx, dy=current_dy, dz=current_dz, shapes=geom.shapes, actual_elem_count=size(elements, 1), max_domain_dim=geom.max_domain_dim)

                println("    [Refinement] Re-mapping Boundary Conditions & Forces...")
                nNodes = size(nodes, 1)
                ndof = nNodes * 3
                bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
                F_external = zeros(Float32, ndof)
                Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
                _, original_density, protected_elements_mask, _ = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
                println("    [Refinement] Resetting solution guess.")
                U_full = zeros(Float32, ndof)
                TopologyOptimization.reset_filter_cache!()
            else
                status_msg = "RefineSkip"
            end
        else
            if iter > nominal_iterations; status_msg = "Annealing"; end
        end

        if iter > 1
            Threads.@threads for e in 1:size(elements, 1)
                if protected_elements_mask[e]; density[e] = original_density[e]; end
            end
        end
        
        config["current_outer_iter"] = iter
        F_total = copy(F_external)
        
        if gravity_scale > 1e-4 && material_density > 1e-9
             dx_curr = Float32(config["geometry"]["dx_computed"]); dy_curr = Float32(config["geometry"]["dy_computed"]); dz_curr = Float32(config["geometry"]["dz_computed"])
             Boundary.add_self_weight!(F_total, density, material_density, gravity_scale, elements, dx_curr, dy_curr, dz_curr, gravity_accel)
        end
        
        if abs(delta_T) > 1e-6
             Boundary.compute_global_thermal_forces!(F_total, nodes, elements, alpha_field, delta_T, E, nu, density)
        end
        
        
        sol_tuple = Solver.solve_system(
            nodes, elements, E, nu, bc_indicator, F_total;
            density=density, config=config, min_stiffness_threshold=min_density, 
            prune_voids=true, u_prev=U_full 
        )
        
        U_new = sol_tuple[1]
        last_residual = sol_tuple[2]
        prec_used = sol_tuple[3]
        
        U_full = U_new
        
        if CUDA.functional(); GC.gc(); CUDA.reclaim(); end
        
        compliance = dot(F_total, U_full)
        strain_energy = 0.5 * compliance
        
        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_dir_field = Stress.compute_stress_field(nodes, elements, U_full, E, nu, density)
        
        if iter == 1
            println("    -> Exporting INITIAL REFERENCE STATE (Iter 0 - Before Optimization)...")
            do_bin_init = (save_bin_freq > 0); do_stl_init = (save_stl_freq > 0); do_vtk_init = (save_vtk_freq > 0)
            if do_bin_init || do_stl_init || do_vtk_init
                export_iteration_results(0, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_dir_field, density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(filter_R), config=config, save_bin=do_bin_init, save_stl=do_stl_init, save_vtk=do_vtk_init)
            end
            if hard_stop_iter == 0; println(">>> HARD STOP: Stopping after background analysis (Iter 0)."); keep_running = false; break; end
        end
        
        active_stress_indices = findall(d -> d > 0.1f0, density)
        avg_l1_stress = isempty(active_stress_indices) ? 0.0f0 : mean(view(l1_stress_norm_field, active_stress_indices))
        vol_total = length(density); active_non_soft = count(d -> d > min_density, density); vol_frac = sum(density) / vol_total
        
        res_tuple = TopologyOptimization.update_density!(density, l1_stress_norm_field, protected_elements_mask, E, l1_stress_allowable, iter, nominal_iterations + 10, original_density, min_density, max_density_clamp, config, elements, is_annealing)
        max_change, filter_R, curr_threshold = res_tuple
        
        iter_time = time() - iter_start_time
        cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
        
        
        Diagnostics.write_iteration_log(
            log_filename, iter, cur_dims_str, vol_total, active_non_soft, 
            filter_R, curr_threshold, compliance, strain_energy, avg_l1_stress, vol_frac, max_change, 
            status_msg, iter_time, last_residual, prec_used
        )

        do_bin = (save_bin_freq > 0) && (iter % save_bin_freq == 0); do_stl = (save_stl_freq > 0) && (iter % save_stl_freq == 0); do_vtk = (save_vtk_freq > 0) && (iter % save_vtk_freq == 0)
        should_export = do_bin || do_stl || do_vtk

        if should_export
            println("    -> Exporting results...")
            export_iteration_results(iter, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_dir_field, density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(filter_R), config=config, save_bin=do_bin, save_stl=do_stl, save_vtk=do_vtk)
        end
        
        if hard_stop_iter > 0 && iter >= hard_stop_iter; println(">>> HARD STOP: Reached target iteration $hard_stop_iter."); keep_running = false; break; end
        if iter > (nominal_iterations + 50); println("\n>>> STOPPING: Reached iteration limit."); keep_running = false; end
        if nominal_iterations == 0; keep_running = false; end

        if CUDA.functional(); Helpers.clear_gpu_memory(); end
        iter += 1
        GC.gc() 
    end
    Diagnostics.log_status("Finished.")
end

end

using .HEXA

config_file = joinpath(Main.PROJECT_ROOT, "configs", "default.yaml")
if length(ARGS) >= 1; if endswith(lowercase(ARGS[1]), ".yaml") || endswith(lowercase(ARGS[1]), ".json"); config_file = ARGS[1]; end; end

if isfile(config_file)
    HEXA.run_main(config_file)
else
    println("\n!!! ERROR: Input file not found at: $config_file")
    println("Usage: julia src/Main.jl [config.yaml]")
end