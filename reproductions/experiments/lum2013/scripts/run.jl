include("common.jl")
using Test
if !("--no-plots" in ARGS)
    include("plot.jl")
end

function main()
    LinearAlgebra.BLAS.set_num_threads(1)
    Random.seed!(2034)
    data=load_data(); mkpath(joinpath(ROOT,"results"))
    csv(joinpath(ROOT,"results","patients.csv"),[(;gsm=data.ids[i],patient_id=data.pid[i],relapse=data.relapse[i],er=data.er[i],followup_months=data.followup[i],esr1=data.esr[i],four_marker_score=data.marker_score[i]) for i in eachindex(data.ids)])
    models=Dict{String,Any}(); summaries=NamedTuple[]; audits=Dict{String,Any}()
    for centered in (false,true), nprobes in (500,1553,3212)
        geometry=correlation_geometry(data;nprobes,center_genes=centered)
        for resolution in (20,35,70), bins in (5,10,20)
            model=construct(geometry,data.relapse;resolution,bins)
            row=(;nprobes,center_genes=centered,resolution,bins,gain=3.0,graph_summary(model.M,data)...)
            push!(summaries,row)
            row.covered_patients==286 || error("Coverage failure")
            row.all_nodes_outcome_pure || error("Unexpected mixed outcome node")
            if nprobes==1553 && bins==10 && resolution in (20,70)
                name="$(centered ? "gene_centered" : "log2")_r$(resolution)"
                models[name]=model; export_model(name,model,data)
                audits[name]=Dict(pairs(audit(model,data)))
                println(name,": ",graph_summary(model.M,data))
            end
        end
    end
    csv(joinpath(ROOT,"results","sensitivity.csv"),summaries)
    baseline=models["log2_r70"]
    control=construct(baseline.geometry,data.relapse;supervised=false)
    models["outcome_free_r70"]=control;export_model("outcome_free_r70",control,data)
    audits["outcome_free_r70"]=Dict(pairs(audit(control,data)))
    csv(joinpath(ROOT,"results","selected_models.csv"),[(;name,graph_summary(model.M,data)...) for (name,model) in sort(collect(models);by=first)])
    permutation=randperm(length(data.ids)); shuffled=data.relapse[permutation]
    permuted=construct(baseline.geometry,shuffled)
    permuted_data=merge(data,(;relapse=shuffled))
    csv(joinpath(ROOT,"results","outcome_control.csv"),[(;name="real_relapse",graph_summary(baseline.M,data)...),(;name="permuted_relapse",graph_summary(permuted.M,permuted_data)...),(;name="no_outcome_lens",graph_summary(control.M,data)...)])
    csv(joinpath(ROOT,"results","selected_probes.csv"),[(;rank=i,probe=data.probes[j],log2_variance=baseline.geometry.variances[j],symbols=join(get(data.annotation,data.probes[j],String[]),'|')) for (i,j) in enumerate(baseline.geometry.selected)])
    # Descriptive ER- summaries use the deposited clinical status, not graph-tuned
    # thresholds or the authors' undisclosed lowERHS memberships.
    groups=[(;name="all",ids=collect(eachindex(data.ids))),
            (;name="clinical_ER_negative_relapse",ids=findall((data.er.=="ER-").&(data.relapse.==1))),
            (;name="clinical_ER_negative_nonrelapse",ids=findall((data.er.=="ER-").&(data.relapse.==0)))]
    csv(joinpath(ROOT,"results","marker_summary.csv"),[(;group=g.name,patients=length(g.ids),mean_esr1=mean(data.esr[g.ids]),mean_four_marker_score=mean(data.marker_score[g.ids]),median_four_marker_score=median(data.marker_score[g.ids])) for g in groups])
    markerprobes=Dict(data.markers[i]=>data.probes[data.marker_rows[i]] for i in eachindex(data.markers))
    metadata=Dict{String,Any}("julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),"seed"=>2034,
        "configurations"=>length(summaries),"patients"=>286,"probes"=>22283,"relapse"=>107,"nonrelapse"=>179,
        "source_intensity_range"=>collect(extrema(data.raw)),"esr1_probes"=>data.probes[data.esr_probes],"marker_probes"=>markerprobes,
        "environment_manifest_sha256"=>filehash(joinpath(ROOT,"environment","Manifest.toml")),"audits"=>audits,
        "baseline"=>Dict("top_variable_probes"=>1553,"log2"=>true,"gene_centering"=>false,"rank_equalization"=>true,"resolution"=>70,"binary_relapse_resolution"=>30,"gain"=>3.0,"assumed_overlap"=>2/3,"histogram_bins"=>10))
    workspace=abspath(ROOT,"../../../..")
    source_state=Dict{String,Any}()
    for name in ("MetricSpaces","TDAmapper","TDAplots")
        dir=joinpath(workspace,name*".jl")
        files=sort([joinpath(d,f) for (d,_,fs) in walkdir(joinpath(dir,"src")) for f in fs if endswith(f,".jl")])
        append!(files,[joinpath(dir,"Project.toml")])
        fingerprints=[relpath(path,dir)*":"*filehash(path) for path in files]
        source_state[name]=Dict("source_sha256"=>bytes2hex(sha256(join(fingerprints,"\n"))),"git_head"=>strip(read(`git -C $dir rev-parse HEAD`,String)),"dirty"=>!isempty(read(`git -C $dir status --porcelain`,String)))
    end
    metadata["sources"]=source_state
    open(joinpath(ROOT,"results","run_metadata.toml"),"w") do io; TOML.print(io,metadata); end
    @testset "Lum 2013 scientific integrity" begin
        @test length(summaries)==54
        @test all(s.covered_patients==286 for s in summaries)
        @test all(s.all_nodes_outcome_pure for s in summaries)
        @test length(data.esr_probes)==9
        @test sum(data.relapse)==107
        @test graph_summary(permuted.M,permuted_data).all_nodes_outcome_pure
        @test any(c -> length(unique(data.relapse[c]))==2, control.M.C)
    end
    println("Outcome-free control: ",graph_summary(control.M,data))
    if !("--no-plots" in ARGS)
        plot_results(models,summaries,data)
    end
    println("Saved 54 sensitivity configurations, 5 graph exports, source hashes and audits.")
end
main()
