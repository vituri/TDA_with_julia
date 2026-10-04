using SHA, TOML, Dates
const ROOT = @__DIR__
workspace = abspath(ROOT, "../../../..")
filehash(path) = bytes2hex(sha256(read(path)))

lock = TOML.parsefile(joinpath(ROOT, "Manifest.toml"))
for entries in values(lock["deps"]), entry in entries
    haskey(entry, "path") || continue
    resolved = isabspath(entry["path"]) ? entry["path"] : abspath(ROOT, entry["path"])
    entry["path"] = relpath(resolved, ROOT)
end
open(joinpath(ROOT, "environment.lock.toml"), "w") do io
    TOML.print(io, lock; sorted=true)
end
packages = Dict{String,Any}()
for name in ("TDARipserer", "TDAPersistenceDiagrams", "PersistenceInference")
    dir = joinpath(workspace, name * ".jl")
    files = [joinpath(base, name) for (base,_,names) in walkdir(joinpath(dir,"src"))
             for name in names if endswith(name,".jl")]
    push!(files, joinpath(dir,"Project.toml"))
    packages[name] = Dict("git_head"=>strip(read(`git -C $dir rev-parse HEAD`,String)),
        "source_files_sha256"=>Dict(relpath(file,dir)=>filehash(file) for file in sort(files)))
end
artifacts = Dict(relpath(joinpath(base,name),ROOT)=>filehash(joinpath(base,name))
    for (base,_,names) in walkdir(joinpath(ROOT,"results")) for name in names if endswith(name,".csv"))
metadata = Dict("recorded_at_utc"=>string(now(UTC)), "packages"=>packages,
    "numerical_artifacts_sha256"=>artifacts,
    "environment_lock_sha256"=>filehash(joinpath(ROOT,"environment.lock.toml")),
    "paper_url"=>"https://arxiv.org/pdf/1303.7117v3",
    "paper_sha256"=>"33fde1a70ddbadaaf55c69326c4ff7e855c638e32da19335b9f2babd3c52025e",
    "data_origin"=>"New IID uniform unit-circle realization of Example 13; original seed was not supplied")
open(joinpath(ROOT,"results","provenance.toml"),"w") do io
    TOML.print(io,metadata;sorted=true)
end
