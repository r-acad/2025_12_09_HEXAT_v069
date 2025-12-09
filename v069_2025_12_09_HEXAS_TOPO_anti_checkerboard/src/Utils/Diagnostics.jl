# // # FILE: .\src\Utils\Diagnostics.jl
module Diagnostics

using CUDA
using Printf
using Dates

export log_status, check_memory, init_log_file, write_iteration_log

# Added "Lin Res" and "Prec" columns to header
const LOG_HEADER = """
| Iter | Mesh Size | Total El | Active El | Radius | Cutoff | Compliance | Strain Energy | Avg L1 Stress | Vol Frac | Delta Rho | Refine? | Lin Res  | Prec   | Time (s) | Wall Time | VRAM |
|------|-----------|----------|-----------|--------|--------|------------|---------------|---------------|----------|-----------|---------|----------|--------|----------|-----------|------|"""

function log_status(msg::String)
    timestamp = Dates.format(now(), "HH:MM:SS")
    println("[$timestamp] $msg")
    flush(stdout) 
end

function check_memory()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        return free_gpu
    end
    return 0
end

function format_memory_str()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        used_gb = (total_gpu - free_gpu) / 1024^3
        return @sprintf("%.1fG", used_gb)
    end
    return "CPU"
end

function init_log_file(filename::String, config::Dict)
    open(filename, "w") do io
        write(io, "HEXA FEM TOPOLOGY OPTIMIZATION LOG\n")
        write(io, "Start Date: $(now())\n")
        write(io, "Config Geometry: $(config["geometry"])\n")
        write(io, "="^200 * "\n") 
        write(io, LOG_HEADER * "\n")
    end
end

# FIX: Removed `::Float64` type constraint on `lin_residual` to allow `Float32`
function write_iteration_log(filename::String, iter, mesh_dims_str, nTotal, nActive, 
                             filter_R, threshold, compliance, strain_energy, avg_l1, 
                             vol_frac, delta_rho, refine_status, time_sec,
                             lin_residual=0.0, precond_type="-")
    
    vram_str = format_memory_str()
    wall_time = Dates.format(now(), "HH:MM:SS")
    
    f_R = Float64(filter_R)
    f_th = Float64(threshold)
    f_comp = Float64(compliance)
    f_se = Float64(strain_energy)
    f_l1 = Float64(avg_l1)
    f_vf = Float64(vol_frac)
    f_dr = Float64(delta_rho)
    f_time = Float64(time_sec)
    
    # Ensure residual is cast to float for formatting
    f_res = Float64(lin_residual)

    line = @sprintf("| %4d | %9s | %8d | %9d | %6.3f | %6.3f | %10.3e | %13.3e | %13.3e | %8.4f | %8.2f%% | %7s | %8.1e | %-6s | %8.2f | %9s | %4s |",
                    iter, mesh_dims_str, nTotal, nActive, f_R, f_th,
                    f_comp, f_se, f_l1, f_vf, 
                    f_dr*100, refine_status, f_res, precond_type, f_time, wall_time, vram_str)
    
    open(filename, "a") do io
        println(io, line)
    end
    
    if iter == 1 || iter % 10 == 0 || refine_status != "Nominal"
        println(LOG_HEADER)
    end
    println(line)
    flush(stdout)
end

end