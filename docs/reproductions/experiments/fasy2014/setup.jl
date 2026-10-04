ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
using Pkg
root = @__DIR__
workspace = abspath(root, "../../../..")
Pkg.activate(root)
Pkg.develop([PackageSpec(path=joinpath(workspace, name * ".jl"))
    for name in ("TDAPersistenceDiagrams", "TDARipserer", "PersistenceInference")])
Pkg.instantiate()
