using Pkg
const REPO=normpath(joinpath(@__DIR__,"../../../.."))
const PRIVATE_ENV=joinpath(@__DIR__,"environment")
Pkg.activate(PRIVATE_ENV)
cd(PRIVATE_ENV) do
    Pkg.develop(PackageSpec(path=relpath(joinpath(REPO,"TDAPersistenceDiagrams.jl"),PRIVATE_ENV)))
    Pkg.instantiate()
end
