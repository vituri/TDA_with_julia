# Run from any directory. This changes only this reproduction's environment.
using Pkg
const REPO = normpath(joinpath(@__DIR__, "../../../../"))
const ENV = joinpath(@__DIR__, "environment")
Pkg.activate(ENV)
cd(ENV) do
    Pkg.develop([
        PackageSpec(path=relpath(joinpath(REPO, "MetricSpaces.jl"), ENV)),
        PackageSpec(path=relpath(joinpath(REPO, "ToMATo.jl"), ENV)),
    ])
    Pkg.instantiate()
end
