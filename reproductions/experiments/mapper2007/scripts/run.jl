include("common.jl")
using Test

config = TOML.parsefile(joinpath(ROOT, "config.toml"))
Random.seed!(config["seed"])
data = load_data()
mkpath(joinpath(ROOT, "results"))
write_csv(joinpath(ROOT, "results", "preprocessing.csv"),
          [(; feature=FEATURES[i], mean=mean(data.measurements[:,i]),
             sample_std=std(data.measurements[:,i])) for i in 1:5])

rows = NamedTuple[]
selected = NamedTuple[]
models = Dict{String,Any}()
audits = Dict{String,Any}()
for normalization in config["normalizations"]
    X = prepare_space(data, normalization)
    base_bandwidth = nearest_neighbor_bandwidth(X)
    for factor in config["bandwidth_factors"]
        bandwidth = factor*base_bandwidth
        f = kde(X; bandwidth)
        for bins in config["histogram_bins"], intervals in config["interval_counts"],
            histogram_range in config["histogram_ranges"], overlap in config["overlaps"]
            model = construct(X, f; intervals, overlap, bins, histogram_range)
            summary = graph_summary(model.M)
            summary.covered_patients == 145 || error("Incomplete coverage")
            row = (; normalization, bandwidth_factor=factor, bandwidth, bins,
                   histogram_range, intervals, overlap, summary...)
            push!(rows, row)
            if normalization == config["illustration_normalization"] &&
               bins == config["illustration_bins"] &&
               overlap == config["illustration_overlap"] &&
               histogram_range == config["illustration_histogram_range"] &&
               factor in config["illustration_factors"]
                name = "k$(intervals)_h$(Int(factor))"
                saved = export_model(name, X, f, data, model)
                models[name] = saved
                audits[name] = audit_model(saved; bandwidth)
                push!(selected, (; name, row...))
                println(name, ": ", summary)
            end
        end
    end
end
write_csv(joinpath(ROOT, "results", "sensitivity.csv"), rows)
write_csv(joinpath(ROOT, "results", "selected_models.csv"), selected)
open(joinpath(ROOT, "results", "audit.toml"), "w") do io
    TOML.print(io, Dict("models"=>audits, "configurations"=>length(rows),
                       "all_configurations_cover_145_patients"=>true,
                       "checks"=>["dataset SHA256 and schema", "KDE vs explicit Gaussian sum",
                                   "each nerve edge vs patient intersection",
                                   "single linkage vs distance-threshold connected components"]))
end

# Fingerprint the actual local source, including uncommitted changes.
book_root = abspath(ROOT, "../../..")
workspace = isdir(joinpath(book_root, ".book-packages")) ? joinpath(book_root, ".book-packages") : dirname(book_root)
manifest = joinpath(ROOT,"Manifest.toml")
lockfile = joinpath(ROOT,"environment.lock.toml")
function write_portable_lock(manifest, lockfile)
    manifest_text = read(manifest,String)
    for entries in values(TOML.parse(manifest_text)["deps"])
        for entry in entries
            if haskey(entry,"path")
                absolute = normpath(joinpath(ROOT,entry["path"]))
                portable = replace(relpath(absolute,ROOT), '\\'=>'/')
                manifest_text = replace(manifest_text, "path = \"" * entry["path"] * "\"" => "path = \"" * portable * "\"")
            end
        end
    end
    write(lockfile,manifest_text)
end
write_portable_lock(manifest,lockfile)
source_state = Dict{String,Any}()
for name in ("MetricSpaces", "TDAmapper", "TDAplots", "TDAPersistenceDiagrams")
    dir = joinpath(workspace, name*".jl")
    files = sort([joinpath(d,f) for (d,_,fs) in walkdir(joinpath(dir,"src")) for f in fs if endswith(f,".jl")])
    append!(files, [joinpath(dir,"Project.toml")])
    fingerprints = [relpath(path,dir)*":"*filehash(path) for path in files]
    is_checkout = isdir(joinpath(dir,".git"))
    git_head = is_checkout ? strip(read(`git -C $dir rev-parse HEAD`,String)) : "frozen book source"
    dirty = is_checkout ? !isempty(read(`git -C $dir status --porcelain`,String)) : false
    source_state[name] = Dict("git_head"=>git_head,
                              "dirty"=>dirty,
                              "source_sha256"=>bytes2hex(sha256(join(fingerprints,"\n"))))
end
open(joinpath(ROOT, "results", "run_metadata.toml"), "w") do io
    TOML.print(io, Dict("julia_version"=>string(VERSION), "threads"=>Threads.nthreads(),
                       "dataset_sha256"=>filehash(joinpath(ROOT,"data","diabetes.csv")),
                       "config_sha256"=>filehash(joinpath(ROOT,"config.toml")),
                       "manifest_sha256"=>filehash(joinpath(ROOT,"Manifest.toml")),
                       "environment_lock_sha256"=>filehash(lockfile),
                       "sources"=>source_state))
end
@testset "scientific integrity" begin
    @test length(rows) == 480
    @test all(values(audits))
    @test all(r.covered_patients==145 for r in rows)
    @test [count(==(g),data.groups) for g in GROUPS] == [76,36,33]
end
if !("--no-plots" in ARGS)
    include("plot.jl")
    plot_results(models, selected, rows, data, config)
end
println("Saved ", length(rows), " configurations and ", length(models), " illustrated models in results/.")
