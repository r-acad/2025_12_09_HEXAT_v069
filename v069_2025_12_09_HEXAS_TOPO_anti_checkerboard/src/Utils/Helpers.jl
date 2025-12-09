# FILE: .\src\Utils\Helpers.jl

module Helpers 

using CUDA 
using Printf

export expand_element_indices, nodes_from_location, parse_location_component 
export calculate_element_distribution, has_enough_gpu_memory, clear_gpu_memory, get_max_feasible_elements

function expand_element_indices(elem_inds, dims) 
    nElem_x = dims[1] - 1 
    nElem_y = dims[2] - 1 
    nElem_z = dims[3] - 1 
    inds = Vector{Vector{Int}}() 
    for d in 1:3 
        if (typeof(elem_inds[d]) == String && elem_inds[d] == ":") 
            if d == 1 
                push!(inds, collect(1:nElem_x)) 
            elseif d == 2 
                push!(inds, collect(1:nElem_y)) 
            elseif d == 3 
                push!(inds, collect(1:nElem_z)) 
            end 
        else 
            push!(inds, [Int(elem_inds[d])]) 
        end 
    end 
    result = Int[] 
    for i in inds[1], j in inds[2], k in inds[3] 
        eidx = i + (j-1)*nElem_x + (k-1)*nElem_x*nElem_y 
        push!(result, eidx) 
    end 
    return result 
end 

function nodes_from_location(loc::Vector, dims) 
    nNodes_x, nNodes_y, nNodes_z = dims 
    ix = parse_location_component(loc[1], nNodes_x) 
    iy = parse_location_component(loc[2], nNodes_y) 
    iz = parse_location_component(loc[3], nNodes_z) 
    nodes = Int[] 
    for k in iz, j in iy, i in ix 
        node = i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y 
        push!(nodes, node) 
    end 
    return nodes 
end 

function parse_location_component(val, nNodes::Int) 
    if val == ":" 
        return collect(1:nNodes) 
    elseif isa(val, String) && endswith(val, "%") 
        perc = parse(Float64, replace(val, "%"=>"")) / 100.0 
        idx = round(Int, 1 + perc*(nNodes-1)) 
        return [idx] 
    elseif isa(val, Number) 
        if 0.0 <= val <= 1.0 
            idx = round(Int, 1 + val*(nNodes-1)) 
            return [idx] 
        else 
            idx = clamp(round(Int, val), 1, nNodes) 
            return [idx] 
        end 
    else 
        error("Invalid location component: $val") 
    end 
end 

function clear_gpu_memory() 
    if !CUDA.functional() 
        return (0, 0) 
    end 
    GC.gc() 
    CUDA.reclaim() 

    final_free, total = CUDA.available_memory(), CUDA.total_memory() 
    return (final_free, total) 
end 

"""
    estimate_bytes_per_element(matrix_free::Bool)

Refined estimation for RTX 3080 vs H100 optimization.
Matrix-Free CG requires: 
- 7 Vectors (Float32) * 24 DOF/elem = 168 * 4 bytes = 672 bytes (worst case full node overlap)
- Connectivity (Int32) = 32 bytes
- Density field = 4 bytes
- Geometry overhead
Realistically, nodes are shared 8x, so per-element vector cost is lower.
Conservative estimate: ~220 bytes per element.
"""
function estimate_bytes_per_element(matrix_free::Bool=true)
    if matrix_free
        return 220 
    else
        return 12000 
    end
end

"""
    get_max_feasible_elements(matrix_free::Bool=true; safety_factor::Float64=0.95, bytes_per_elem::Int=0)

Calculates the maximum number of elements the GPU can hold based on CURRENT available memory.
Accepts custom safety factors and byte estimates for hardware tuning.
"""
function get_max_feasible_elements(matrix_free::Bool=true; safety_factor::Float64=0.95, bytes_per_elem::Int=0)
    if !CUDA.functional() 
        return 5_000_000 
    end 
      
    free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
    
    # Check if we are on massive hardware (H100/H200) by checking total memory > 70GB
    # If so, we can be less conservative with safety factors if not specified
    if total_mem > (70 * 1024^3) && safety_factor == 0.95
         safety_factor = 0.98
    end
    
    usable_mem = free_mem * safety_factor 
      
    bpe = (bytes_per_elem > 0) ? bytes_per_elem : estimate_bytes_per_element(matrix_free)
    
    max_elems = floor(Int, usable_mem / bpe) 
      
    return max_elems
end
 
function estimate_gpu_memory_required(nNodes, nElem, matrix_free::Bool=true) 
    return nElem * estimate_bytes_per_element(matrix_free)
end 
 
function has_enough_gpu_memory(nNodes, nElem, matrix_free::Bool=true) 
    if !CUDA.functional() 
        return false 
    end 
    try 
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
        required_mem = estimate_gpu_memory_required(nNodes, nElem, matrix_free) 
          
        # If running on H-series (huge memory), use 98% utilization, otherwise 90%
        utilization_limit = (total_mem > 70 * 1024^3) ? 0.98 : 0.90
        
        usable_mem = free_mem * utilization_limit 
          
        req_gb = required_mem / 1024^3
        avail_gb = usable_mem / 1024^3

        if required_mem > usable_mem
            @warn "GPU Memory Estimate:"
            @printf("   Required:  %.2f GB\n", req_gb)
            @printf("   Available: %.2f GB\n", avail_gb)
            println("   ⚠️ WARNING: Memory estimate exceeds safe limits.")
            return true 
        end
        return true 
    catch e 
        println("Error checking GPU memory: $e") 
        return true 
    end 
end 
 
function calculate_element_distribution(length_x, length_y, length_z, target_elem_count) 
    total_volume = length_x * length_y * length_z 
      
    ratio_x = length_x / cbrt(total_volume) 
    ratio_y = length_y / cbrt(total_volume) 
    ratio_z = length_z / cbrt(total_volume) 

    base_count = cbrt(target_elem_count) 
    nElem_x = max(1, round(Int, base_count * ratio_x)) 
    nElem_y = max(1, round(Int, base_count * ratio_y)) 
    nElem_z = max(1, round(Int, base_count * ratio_z)) 

    dx = length_x / nElem_x 
    dy = length_y / nElem_y 
    dz = length_z / nElem_z 
    actual_elem_count = nElem_x * nElem_y * nElem_z 
    return nElem_x, nElem_y, nElem_z, Float32(dx), Float32(dy), Float32(dz), actual_elem_count 
end 
 
end