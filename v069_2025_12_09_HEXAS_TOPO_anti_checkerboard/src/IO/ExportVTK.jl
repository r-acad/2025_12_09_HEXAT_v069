# FILE: .\src\IO\ExportVTK.jl
module ExportVTK 

using Printf 

export export_mesh, export_solution 

function export_mesh(nodes::Matrix{Float32}, 
                     elements::Matrix{Int}; 
                     bc_indicator=nothing, 
                     filename::String="mesh_output.vtu")
      
    if !endswith(lowercase(filename), ".vtk") && !endswith(lowercase(filename), ".vtu") 
        filename *= ".vtk" 
    end
    
    # Mesh export logic (placeholder if needed, currently unused in main loop)
end 

function export_solution(nodes::Matrix{Float32}, 
                         elements::Matrix{Int}, 
                         U_full::Vector{Float32}, 
                         F::Vector{Float32}, 
                         bc_indicator::Matrix{Float32}, 
                         principal_field::Matrix{Float32}, 
                         vonmises_field::Vector{Float32}, 
                         full_stress_voigt::Matrix{Float32}, 
                         l1_stress_norm_field::Vector{Float32},
                         principal_dir_field::Matrix{Float32}; 
                         density::Union{Vector{Float32}, Nothing}=nothing,
                         scale::Float32=Float32(1.0),
                         threshold::Float32=0.01f0, # MODIFIED: Added threshold argument
                         filename::String="solution_output.vtu") 

    function sanitize_data(data) 
        data = replace(data, NaN => Float32(0.0), Inf => Float32(0.0), -Inf => Float32(0.0)) 
        max_val = maximum(abs.(data)) 
        if max_val > Float32(1.0e10) 
            return clamp.(data, Float32(-1.0e10), Float32(1.0e10)) 
        end 
        return data 
    end 
      
    U_full = sanitize_data(U_full) 
    F_sanitized = sanitize_data(F) 
    bc_sanitized = sanitize_data(bc_indicator)
    nodes = sanitize_data(nodes) 
    
    l1_stress_norm_field = sanitize_data(l1_stress_norm_field) 

    nNodes = size(nodes, 1) 
    nElem  = size(elements, 1) 

    valid_elements = Int[] 
    
    # MODIFIED: Use the passed threshold variable
    export_threshold = threshold

    if density !== nothing
        for e = 1:nElem 
            if density[e] >= export_threshold 
                push!(valid_elements, e) 
            end
        end 
    else
        valid_elements = collect(1:nElem)
    end

    nElem_valid = length(valid_elements) 

    displacement = zeros(Float32, nNodes, 3) 
    forces_vec   = zeros(Float32, nNodes, 3)
    bcs_vec      = zeros(Float32, nNodes, 3)
      
    for i in 1:nNodes 
        base_idx = 3*(i-1)
        if base_idx + 3 <= length(U_full) 
            displacement[i, 1] = U_full[base_idx + 1] 
            displacement[i, 2] = U_full[base_idx + 2] 
            displacement[i, 3] = U_full[base_idx + 3] 
        end 
        
        if base_idx + 3 <= length(F_sanitized)
            forces_vec[i, 1] = F_sanitized[base_idx + 1]
            forces_vec[i, 2] = F_sanitized[base_idx + 2]
            forces_vec[i, 3] = F_sanitized[base_idx + 3]
        end

        if i <= size(bc_sanitized, 1)
            bcs_vec[i, 1] = bc_sanitized[i, 1]
            bcs_vec[i, 2] = bc_sanitized[i, 2]
            bcs_vec[i, 3] = bc_sanitized[i, 3]
        end
    end 
    
    disp_mag = sqrt.(sum(displacement.^2, dims=2))[:,1]       

    l1_stress_norm_field_valid = l1_stress_norm_field[valid_elements] 
    vonmises_field_valid = vonmises_field[valid_elements]
    
    if endswith(lowercase(filename), ".vtu"); combined_filename = filename[1:end-4] * ".vtk"
    elseif !endswith(lowercase(filename), ".vtk"); combined_filename = filename * ".vtk"
    else; combined_filename = filename; end

    try 
        open(combined_filename, "w") do file 
            write(file, "# vtk DataFile Version 3.0\n") 
            write(file, "HEXA FEM Solution (Undeformed)\n") 
            write(file, "BINARY\n") 
            write(file, "DATASET UNSTRUCTURED_GRID\n") 
              
            write(file, "POINTS $(nNodes) float\n") 
            coords_flat = vec(nodes') 
            write(file, hton.(Float32.(coords_flat))) 
              
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n") 
            cell_data = Vector{Int32}(undef, nElem_valid * 9)
            idx = 1
            for e in valid_elements
                cell_data[idx] = Int32(8) 
                idx += 1
                for j in 1:8
                    cell_data[idx] = Int32(elements[e, j] - 1) 
                    idx += 1
                end
            end
            write(file, hton.(cell_data)) 
              
            write(file, "\nCELL_TYPES $(nElem_valid)\n") 
            cell_types = fill(Int32(12), nElem_valid) 
            write(file, hton.(cell_types)) 
              
            write(file, "\nPOINT_DATA $(nNodes)\n") 
              
            write(file, "VECTORS Displacement float\n") 
            disp_flat = vec(displacement')
            write(file, hton.(Float32.(disp_flat))) 
              
            write(file, "\nVECTORS External_Forces float\n")
            forces_flat = vec(forces_vec')
            write(file, hton.(Float32.(forces_flat)))

            write(file, "\nVECTORS Boundary_Conditions float\n")
            bcs_flat = vec(bcs_vec')
            write(file, hton.(Float32.(bcs_flat)))

            write(file, "\nSCALARS Displacement_Magnitude float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(disp_mag))) 
              
            write(file, "\nCELL_DATA $(nElem_valid)\n") 
              
            write(file, "SCALARS l1_stress_norm float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(l1_stress_norm_field_valid))) 
            
            write(file, "\nSCALARS Von_Mises float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(vonmises_field_valid))) 
              
            if density !== nothing
                write(file, "\nSCALARS Element_Density float 1\n")
                write(file, "LOOKUP_TABLE default\n")
                density_valid = density[valid_elements]
                write(file, hton.(Float32.(density_valid)))
            end
        end 
    catch e 
        @error "Failed to save combined VTK file: $e" 
    end 
    return nothing 
end 

end