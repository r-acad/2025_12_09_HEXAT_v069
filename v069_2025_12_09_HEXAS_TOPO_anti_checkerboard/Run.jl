# // # FILE: .\Run.jl
using Pkg

println(">>> [LAUNCHER] Activating Project Environment...")
Pkg.activate(@__DIR__)

println(">>> [LAUNCHER] Checking Dependencies...")

try
    # Try to instantiate using the existing Manifest
    Pkg.instantiate()
catch e
    println("!!! [LAUNCHER] Existing Manifest is incompatible with this Julia version.")
    println("!!! [LAUNCHER] Deleting Manifest.toml and resolving dependencies...")
    
    # Path to the Manifest file
    manifest_path = joinpath(@__DIR__, "Manifest.toml")
    if isfile(manifest_path)
        rm(manifest_path, force=true)
    end
    
    # Force a full update/resolve to get versions compatible with THIS Julia
    Pkg.resolve()
    Pkg.instantiate()
    println(">>> [LAUNCHER] Dependencies resolved successfully.")
end

const MAIN_SCRIPT = joinpath(@__DIR__, "src", "Main.jl")

# The 'ARGS' passed to this script are automatically visible to Main.jl
println(">>> [LAUNCHER] Starting Solver...")
println("-"^60)
include(MAIN_SCRIPT)