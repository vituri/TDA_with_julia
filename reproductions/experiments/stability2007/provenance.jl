using SHA, TOML, Dates
const ROOT=@__DIR__
workspace=abspath(ROOT,"../../../..")
filehash(path)=bytes2hex(sha256(read(path)))

# Record the environment used by run.jl with relocatable local source paths.
lock=TOML.parsefile(joinpath(ROOT,"Manifest.toml"))
for entries in values(lock["deps"]), entry in entries
    haskey(entry,"path") || continue
    entry["path"]=relpath(entry["path"],ROOT)
end
open(joinpath(ROOT,"environment.lock.toml"),"w") do io
    TOML.print(io,lock;sorted=true)
end

packages=Dict{String,Any}()
for name in ("TDARipserer","TDAPersistenceDiagrams")
    dir=joinpath(workspace,name*".jl")
    files=String[]
    for (base,_,names) in walkdir(joinpath(dir,"src")), filename in names
        endswith(filename,".jl") && push!(files,joinpath(base,filename))
    end
    push!(files,joinpath(dir,"Project.toml"))
    hashes=Dict(relpath(file,dir)=>filehash(file) for file in sort(files))
    packages[name]=Dict("git_head"=>strip(read(`git -C $dir rev-parse HEAD`,String)),
                        "source_files_sha256"=>hashes)
end
artifacts=Dict(relpath(joinpath(base,name),ROOT)=>filehash(joinpath(base,name))
               for (base,_,names) in walkdir(joinpath(ROOT,"results"))
               for name in names if endswith(name,".csv"))
metadata=Dict("recorded_when"=>"After the numerical analysis; source snapshot for identifying local revisions",
              "recorded_at_utc"=>string(now(UTC)),"packages"=>packages,
              "numerical_artifacts_sha256"=>artifacts,
              "environment_lock_sha256"=>filehash(joinpath(ROOT,"environment.lock.toml")))
open(joinpath(ROOT,"results","provenance.toml"),"w") do io
    TOML.print(io,metadata;sorted=true)
end
