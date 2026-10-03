# Run once from any working directory; only the private environment is updated.
using Pkg
experiment = @__DIR__
workspace = normpath(joinpath(experiment, "..", "..", "..", ".."))
Pkg.activate(joinpath(experiment, "environment"))
Pkg.develop([Pkg.PackageSpec(path=joinpath(workspace, name * ".jl"))
    for name in ("MetricSpaces", "TDAPersistenceDiagrams", "TDARipserer", "TDAplots")])
Pkg.instantiate()
