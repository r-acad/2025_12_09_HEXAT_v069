# // # FILE: .\src\Mesh\MeshUtilities.jl

module MeshUtilities 
 
export inside_sphere, inside_box, element_centroid,
       check_element_quality, fix_inverted_elements!, 
       calculate_element_quality 
 
using LinearAlgebra 
 
""" 
    element_centroid(e, nodes, elements) 
 
Computes the centroid of element `e`.
OPTIMIZED: Uses scalar math to avoid allocating vectors inside threaded loops.
Returns a Tuple (x, y, z).
""" 
function element_centroid(e::Int, nodes::Matrix{Float32}, elements::Matrix{Int}) 
    cx = 0.0f0
    cy = 0.0f0
    cz = 0.0f0

    @inbounds for i in 1:8
        node_idx = elements[e, i]
        cx += nodes[node_idx, 1]
        cy += nodes[node_idx, 2]
        cz += nodes[node_idx, 3]
    end

    inv8 = 0.125f0 # 1/8
    return (cx * inv8, cy * inv8, cz * inv8)  
end 

""" 
    inside_sphere(pt, center, diam) 
Return true if point `pt` is inside a sphere of diameter `diam` at `center`. 
Accepts pt as Vector or Tuple.
""" 
function inside_sphere(pt, center::Tuple{Float32,Float32,Float32}, diam::Float32) 
    r = diam / 2f0 
    # Use scalar indexing to handle both Tuple (from centroid) and Vector input efficiently
    dx = pt[1] - center[1]
    dy = pt[2] - center[2]
    dz = pt[3] - center[3]
    return (dx*dx + dy*dy + dz*dz) <= (r*r)
end 

""" 
    inside_box(pt, center, size) 
Return true if point `pt` is inside a box with dimensions `size` (x, y, z) centered at `center`. 
Accepts pt as Vector or Tuple.
""" 
function inside_box(pt, center::Tuple{Float32,Float32,Float32}, box_size::Tuple{Float32,Float32,Float32}) 
    half_x = box_size[1] / 2f0 
    half_y = box_size[2] / 2f0 
    half_z = box_size[3] / 2f0 
    
    return abs(pt[1] - center[1]) <= half_x && 
           abs(pt[2] - center[2]) <= half_y && 
           abs(pt[3] - center[3]) <= half_z 
end 
 
""" 
    check_element_quality(nodes, elements) -> poor_elements 
Mark which elements are degenerate, etc. (Placeholder for future expansion)
""" 
function check_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    nElem = size(elements,1) 
    poor_elements = Int[] 
    return poor_elements 
end 
 
""" 
    fix_inverted_elements!(nodes, elements) -> (fixed_count, warning_count) 
Swap node ordering to fix negative Jacobians. (Placeholder)
""" 
function fix_inverted_elements!(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    return (0, 0) 
end 
 
""" 
    calculate_element_quality(nodes, elements) 
Returns (aspect_ratios, min_jacobians) (Placeholder)
""" 
function calculate_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    nElem = size(elements, 1) 
    return zeros(Float32, nElem), zeros(Float32, nElem) 
end 
 
end