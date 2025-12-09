# FILE: .\src\Mesh\MeshShapeProcessing.jl
module MeshShapeProcessing 
 
export apply_geometric_modifiers!
 
using LinearAlgebra 
using Base.Threads
using ..MeshUtilities     
 
""" 
    apply_geometric_modifiers!(density, alpha_field, nodes, elements, shapes, min_density)

Iterates over elements and modifies the `density` and `alpha_field` based on the 
`stiffness_ratio` of the geometric shapes defined in the configuration.
""" 
function apply_geometric_modifiers!(density::Vector{Float32}, 
                                    alpha_field::Vector{Float32},
                                    nodes::Matrix{Float32}, 
                                    elements::Matrix{Int}, 
                                    shapes::Vector{Any},
                                    min_density::Float32)
    
    if isempty(shapes)
        return
    end

    nElem = size(elements, 1)
    
    Threads.@threads for e in 1:nElem
        
        centroid = MeshUtilities.element_centroid(e, nodes, elements)
        
        for shape in shapes
            shape_type = lowercase(get(shape, "type", ""))
            is_inside = false

            if shape_type == "sphere"
                if haskey(shape, "center") && haskey(shape, "diameter")
                    center = tuple(Float32.(shape["center"])...)
                    diam   = Float32(shape["diameter"])
                    is_inside = MeshUtilities.inside_sphere(centroid, center, diam)
                end
            elseif shape_type == "box"
                if haskey(shape, "center")
                    center = tuple(Float32.(shape["center"])...)
                    
                    # Logic to handle 'size' vector, replacing 'side'
                    if haskey(shape, "size")
                        sz_raw = shape["size"]
                        # Ensure it matches [x, y, z] structure
                        if isa(sz_raw, AbstractVector) && length(sz_raw) >= 3
                            box_sz = (Float32(sz_raw[1]), Float32(sz_raw[2]), Float32(sz_raw[3]))
                            is_inside = MeshUtilities.inside_box(centroid, center, box_sz)
                        end
                    elseif haskey(shape, "side")
                        # Fallback for old configs if needed, treating side as a cube
                        s = Float32(shape["side"])
                        box_sz = (s, s, s)
                        is_inside = MeshUtilities.inside_box(centroid, center, box_sz)
                    end
                end
            end

            if is_inside
                # Get stiffness ratio (default to 0 if missing, effectively removing it)
                ratio = Float32(get(shape, "stiffness_ratio", 0.0))
                
                if ratio == 0.0f0
                    # Void / Remove
                    density[e] = min_density
                    alpha_field[e] = 0.0f0
                elseif ratio < 0.0f0
                    # Negative ratio: Abs(stiffness) + Thermal Expansion
                    density[e] = abs(ratio)
                    alpha_field[e] = 1.0f0
                else
                    # Positive ratio: Stiffness scaling, no thermal expansion
                    density[e] = ratio
                    alpha_field[e] = 0.0f0
                end
            end
        end
    end
end 
 
end