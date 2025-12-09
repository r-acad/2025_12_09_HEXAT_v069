// # FILE: .\src\Core\Stress.jl";

module Stress 
 
using LinearAlgebra 
using Base.Threads
using ..Element 
export compute_stress_field 
 
function compute_element_stress(element_nodes::Array{Float32,2}, 
                                 element_disp::Array{Float32,1}, 
                                 E::Float32, nu::Float32) 
    D = Element.material_matrix(E, nu) 
    xi, eta, zeta = 0.0f0, 0.0f0, 0.0f0
    _, dN_dxi = Element.shape_functions(xi, eta, zeta) 
    J = transpose(dN_dxi)*element_nodes 
    detJ = det(J) 
    if detJ <= 0.0f0; error("Non-positive Jacobian!"); end 
    invJ = inv(J) 
    dN_dx = dN_dxi * transpose(invJ) 
 
    B = zeros(Float32, 6, 24) 
    for i in 1:8 
        idx = 3*(i-1)+1 
        dN_i = dN_dx[i, :] 
        B[1, idx]    = dN_i[1]; B[2, idx+1] = dN_i[2]; B[3, idx+2] = dN_i[3] 
        B[4, idx]    = dN_i[2]; B[4, idx+1] = dN_i[1] 
        B[5, idx+1] = dN_i[3]; B[5, idx+2] = dN_i[2] 
        B[6, idx]    = dN_i[3]; B[6, idx+2] = dN_i[1] 
    end 
 
    strain = B * element_disp 
    stress_voigt = D * strain 
 
    σ = zeros(Float32, 3, 3) 
    σ[1,1] = stress_voigt[1]; σ[2,2] = stress_voigt[2]; σ[3,3] = stress_voigt[3]    
    σ[1,2] = stress_voigt[4]; σ[2,1] = stress_voigt[4]    
    σ[2,3] = stress_voigt[5]; σ[3,2] = stress_voigt[5]    
    σ[1,3] = stress_voigt[6]; σ[3,1] = stress_voigt[6]    
    return σ 
end 
 
function compute_principal_data(σ::Matrix{Float32}) 
    F = eigen(σ)
    perm = sortperm(F.values, rev=true)
    principal_stresses = F.values[perm]
    principal_vectors  = F.vectors[:, perm]
    max_dir = principal_vectors[:, 1]

    σxx, σyy, σzz = σ[1,1], σ[2,2], σ[3,3]
    σxy, σyz, σxz = σ[1,2], σ[2,3], σ[1,3]
    vm = sqrt(0.5f0 * ((σxx-σyy)^2 + (σyy-σzz)^2 + (σzz-σxx)^2) + 3.0f0*(σxy^2 + σyz^2 + σxz^2))  
 
    return principal_stresses, vm, max_dir
end 
 
function compute_stress_field(nodes, elements, U, E::Float32, nu::Float32, density::Vector{Float32}) 
    nElem = size(elements, 1) 
    
    # Pre-allocate output arrays
    principal_field      = zeros(Float32, 3, nElem) 
    vonmises_field       = zeros(Float32, nElem) 
    full_stress_voigt    = zeros(Float32, 6, nElem) 
    l1_stress_norm_field = zeros(Float32, nElem)  
    principal_dir_field  = zeros(Float32, 3, nElem)

    # Parallel loop over elements
    Threads.@threads for e in 1:nElem 
        conn = elements[e, :] 
        element_nodes = nodes[conn, :] 
        
        # Local buffer for displacement
        element_disp = zeros(Float32, 24) 
        for i in 1:8 
            global_node = conn[i] 
            base_idx = 3*(global_node-1)
            element_disp[3*(i-1)+1] = U[base_idx+1]
            element_disp[3*(i-1)+2] = U[base_idx+2]
            element_disp[3*(i-1)+3] = U[base_idx+3]
        end 
 
        # Only compute full stress if element has significant density to save time
        # but for visualization consistency we might compute it anyway. 
        # We check density > 1e-6 to avoid numerical noise in voids.
        if density[e] > 1e-6
            E_local = E * density[e] 
            σ = compute_element_stress(element_nodes, element_disp, E_local, nu) 
            (principal, vm, max_dir) = compute_principal_data(σ) 
     
            l1_norm = abs(principal[1]) + abs(principal[2]) + abs(principal[3]) 
                
            principal_field[:, e] = principal 
            vonmises_field[e]        = vm 
            l1_stress_norm_field[e] = l1_norm  
            principal_dir_field[:, e] = max_dir
            full_stress_voigt[:, e] .= (σ[1,1], σ[2,2], σ[3,3], σ[1,2], σ[2,3], σ[1,3]) 
        end
    end 
    return principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_dir_field
end 
 
end