# Run from any directory. Prepare the book environment first for frozen sources.
using Pkg
project = dirname(@__DIR__)
book_root = abspath(project, "../../..")
package_root = isdir(joinpath(book_root,".book-packages")) ?
    joinpath(book_root,".book-packages") : dirname(book_root)
lockfile = joinpath(project,"environment.lock.toml")
manifest = joinpath(project,"Manifest.toml")
if !isfile(manifest) && isfile(lockfile)
    cp(lockfile,manifest)
end
Pkg.activate(project)
packages = ("MetricSpaces", "TDAmapper", "TDAPersistenceDiagrams", "TDAplots")
specs = [PackageSpec(path=relpath(joinpath(package_root,name*".jl"),project)) for name in packages]
all(isdir(joinpath(project,spec.path)) for spec in specs) ||
    error("Prepare the book environment or provide the JuliaTDA sibling checkouts first")
cd(project) do
    Pkg.develop(specs; preserve=Pkg.PRESERVE_ALL)
    Pkg.instantiate()
end
