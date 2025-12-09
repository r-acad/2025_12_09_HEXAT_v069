# FILE: .\src\IO\Configuration.jl
module Configuration 
 
using YAML  
using JSON
using ..Mesh  
using ..Helpers 
using ..MeshShapeProcessing 
 
export load_configuration, setup_geometry, initialize_density_field, load_checkpoint
 
""" 
    load_configuration(filename::String) 
 
Load and parse a JSON/YAML configuration file. 
""" 
function load_configuration(filename::String) 
    if !isfile(filename) 
        error("Configuration file '$(filename)' not found") 
    end 
      
    return YAML.load_file(filename) 
end 

"""
    load_checkpoint(filename::String)

Reads a .bin checkpoint file. 
Returns: (config, density, restart_iter, restart_radius, restart_threshold)
"""
function load_checkpoint(filename::String)
    println(">>> [Checkpoint] Reading restart data from: $filename")
    
    if !isfile(filename); error("Checkpoint file not found."); end

    data = open(filename, "r") do io
        
        magic = read(io, UInt32) # 0x48455841 "HEXA"
        version = read(io, UInt32)
        
        if magic != 0x48455841
            error("Invalid file format. Not a HEXA checkpoint.")
        end

        iter = Int(read(io, Int32))
        radius = Float32(read(io, Float32))
        threshold = Float32(read(io, Float32))
        
        count = Int(read(io, UInt32))
        dx = read(io, Float32)
        dy = read(io, Float32)
        dz = read(io, Float32)

        seek(io, position(io) + (count * 3 * 4))

        density = Vector{Float32}(undef, count)
        read!(io, density)

        seek(io, position(io) + (count * 4))

        json_len = Int(read(io, UInt32))
        json_bytes = Vector{UInt8}(undef, json_len)
        read!(io, json_bytes)
        
        config_str = String(json_bytes)
        config = JSON.parse(config_str)

        println("    Restarting at Iteration: $iter")
        println("    Filter Radius: $radius")
        println("    Threshold: $threshold")
        println("    Elements: $count")

        return (config, density, iter, radius, threshold)
    end

    return data
end
 
""" 
    setup_geometry(config) 
 
Process the geometry configuration and return parameters for mesh generation. 
""" 
function setup_geometry(config) 
      
    length_x = config["geometry"]["length_x"] 
    length_y = config["geometry"]["length_y"] 
    length_z = config["geometry"]["length_z"] 
     
    raw_count = config["geometry"]["target_elem_count"]
    target_elem_count = if isa(raw_count, String)
        parse(Int, replace(raw_count, "_" => ""))
    else
        Int(raw_count)
    end
      
    println("Domain dimensions:") 
    println("  X: 0 to $(length_x)") 
    println("  Y: 0 to $(length_y)") 
    println("  Z: 0 to $(length_z)") 
      
    shapes = Any[] 
     
    for (key, shape) in config["geometry"] 
        if key in ["length_x", "length_y", "length_z", "target_elem_count", "shape_notes", "nElem_x_computed", "nElem_y_computed", "nElem_z_computed", "dx_computed", "dy_computed", "dz_computed", "max_domain_dim"] 
            continue 
        end 
          
        if haskey(shape, "type") 
            # We treat all shapes equally now, storing them in a single list.
            # The 'stiffness_ratio' determines the behavior.
            push!(shapes, shape)
        end 
    end 
 
    println("Found $(length(shapes)) geometric modification shapes.") 
      
    nElem_x, nElem_y, nElem_z, dx, dy, dz, actual_elem_count = 
        Helpers.calculate_element_distribution(length_x, length_y, length_z, target_elem_count) 
      
    println("Mesh parameters:") 
    println("  Domain: $(length_x) x $(length_y) x $(length_z) meters") 
    println("  Elements: $(nElem_x) x $(nElem_y) x $(nElem_z) = $(actual_elem_count)") 
    println("  Element sizes: $(dx) x $(dy) x $(dz)") 
      
    max_domain_dim = max(length_x, length_y, length_z) 
 
    return ( 
        nElem_x = nElem_x,  
        nElem_y = nElem_y,  
        nElem_z = nElem_z, 
        dx = dx, 
        dy = dy, 
        dz = dz, 
        shapes = shapes, # Consolidated list
        actual_elem_count = actual_elem_count, 
        max_domain_dim = Float32(max_domain_dim)  
    ) 
end 
 
""" 
    initialize_density_field(nodes, elements, shapes, config)
 
Processes geometric shapes to set the initial density array AND the alpha (thermal expansion) field.
Returns `density`, `original_density`, `protected_elements_mask`, and `alpha_field`.
""" 
function initialize_density_field(nodes::Matrix{Float32}, 
                                  elements::Matrix{Int}, 
                                  shapes::Vector{Any}, 
                                  config::Dict) 
      
    min_density = Float32(get(config["optimization_parameters"], "min_density", 1e-3)) 
 
    nElem = size(elements, 1)
    println("Processing geometric density and thermal modifiers...") 
     
    # Default: Background Material (Density=1.0, Alpha=0.0)
    density = ones(Float32, nElem) 
    alpha_field = zeros(Float32, nElem)

    # Apply modifiers based on stiffness_ratio
    MeshShapeProcessing.apply_geometric_modifiers!(density, alpha_field, nodes, elements, shapes, min_density)
      
    println("Element density and thermal processing complete. Min Density floor: $(min_density)") 
 
    original_density = copy(density) 
      
    protected_elements_mask = (original_density .!= 1.0f0) 
    num_protected = sum(protected_elements_mask) 
    println("Found $(num_protected) protected elements (voids/rigid/thermal) that will not be iterated.") 
 
    return density, original_density, protected_elements_mask, alpha_field
end 
 
end