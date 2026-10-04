include("common.jl")
using Test
LinearAlgebra.BLAS.set_num_threads(1)
Random.seed!(2011)
mkpath(joinpath(ROOT,"results"))
data=load_data()
println("Aligned genes: ",length(data.genes),"; KNN: ",data.imputed,"; missing values: ",data.missing_before)
csv(joinpath(ROOT,"results","gene_alignment.csv"),data.mapping)
models=Dict{String,Any}();sensitivity=NamedTuple[];dsga_checks=Dict{String,Any}()
baseline=nothing;selected=nothing
for r in (8,10,12)
    result=dsga(data.T,data.N;r)
    selection=select_genes(result.Dc)
    A=hcat(result.L1,result.Dc)[selection.selected,:]
    G=geometry(A)
    normal_dot=norm(result.U'*result.Dc)/norm(result.Dc)
    decomposition=norm(result.T-result.Nc-result.Dc)/norm(result.T)
    dsga_checks[string(r)]=Dict("orthogonality_relative"=>normal_dot,"decomposition_relative"=>decomposition,
                              "selected_genes"=>length(selection.selected),"relaxed_genes"=>length(selection.lax),"stringent_genes"=>length(selection.strict),
                              "relaxed_threshold"=>selection.relaxed,"stringent_threshold"=>selection.stringent)
    for k in (1,4),bins in (5,10,20)
        model=mapper(G;k,bins)
        push!(sensitivity,(;hsm_dimension=r,p=2,power=k,histogram_bins=bins,summary(model)...))
        if r==10 && bins==10
            name="power$(k)";models[name]=model;export_graph(name,model,data)
            audit(model)
        end
    end
    if r==10
        global baseline=result;global selected=selection
        csv(joinpath(ROOT,"results","genes.csv"),[(;gene=data.genes[j],selected=j in selection.selected,deviation_quantile=selection.q[j]) for j in eachindex(data.genes)])
        csv(joinpath(ROOT,"results","wold.csv"),[(;dimension=i,singular_value=result.singular_values[i],wold=result.wold[i]) for i in eachindex(result.wold)])
    end
end
csv(joinpath(ROOT,"results","sensitivity.csv"),sensitivity)
base_model=models["power4"]
row_my=only(findall(==("MYB"),data.genes)); row_er=only(findall(==("ESR1"),data.genes))
all_residuals=hcat(baseline.L1,baseline.Dc)
csv(joinpath(ROOT,"results","samples.csv"),[(;sample=data.ids[j],is_normal=j<=13,filter=base_model.f[j],
    residual_norm=norm(all_residuals[:,j]),
    myb_residual=all_residuals[row_my,j],esr1_residual=all_residuals[row_er,j],
    clinical_ER=data.clinical_er[j],death=j<=13 ? NaN : data.death[j-13],
    followup_years=j<=13 ? NaN : data.followup[j-13]) for j in eachindex(data.ids)])
keep=findall(c->length(c)>1,base_model.M.C)
g,vertices=Graphs.induced_subgraph(base_model.M.g,keep)
retained=sort(unique(vcat(base_model.M.C[keep]...)))
pruned=Dict("nodes"=>Graphs.nv(g),"edges"=>Graphs.ne(g),"components"=>length(Graphs.connected_components(g)),"covered_samples"=>length(retained),"removed_samples"=>setdiff(data.ids,data.ids[retained]))
csv(joinpath(ROOT,"results","selected_models.csv"),[(;name,summary(model)...) for (name,model) in sort(collect(models);by=first)])
# Clinical data are only descriptive annotations. No subgroup is selected here.
csv(joinpath(ROOT,"results","cohort.csv"),[(;group="all_NKI",patients=295,ER_positive=sum(data.er),deaths=sum(data.death),median_followup_years=median(data.followup))])
csv(joinpath(ROOT,"results","tumor_normal_magnitude.csv"),[(;group="BCN_leave_one_out",samples=13,median_residual_norm=median(vec(sqrt.(sum(abs2,baseline.L1;dims=1)))),median_filter=median(base_model.f[1:13])),
    (;group="NKI_tumors",samples=295,median_residual_norm=median(vec(sqrt.(sum(abs2,baseline.Dc;dims=1)))),median_filter=median(base_model.f[14:end]))])
state=Dict{String,Any}()
workspace=abspath(ROOT,"../../../..")
for name in ("MetricSpaces","TDAmapper","TDAplots","TDAPersistenceDiagrams")
    dir=joinpath(workspace,name*".jl")
    files=sort([joinpath(d,f) for (d,_,fs) in walkdir(joinpath(dir,"src")) for f in fs if endswith(f,".jl")])
    push!(files,joinpath(dir,"Project.toml"))
    state[name]=Dict("source_sha256"=>bytes2hex(sha256(join([relpath(p,dir)*":"*hashfile(p) for p in files],"\n"))),
                     "git_head"=>strip(read(`git -C $dir rev-parse HEAD`,String)),"dirty"=>!isempty(read(`git -C $dir status --porcelain`,String)))
end
metadata=Dict{String,Any}("status"=>"partial historical-data reconstruction; symbol alignment replaces unavailable NKI UniGene219 mapping",
    "julia_version"=>string(VERSION),"seed"=>2011,"blas_threads"=>1,"tumors"=>295,"normals"=>13,"matched_genes"=>length(data.genes),
    "missing_nki_values"=>data.missing_before,"imputed_genes"=>data.imputed.imputed_genes,"complete_donor_genes"=>data.imputed.complete_donors,
    "target_column_norm"=>baseline.target,"dimension"=>10,"selected_genes"=>length(selected.selected),
    "normal_ids"=>data.normal_ids,"clinical_ER_positive"=>sum(data.er),"clinical_deaths"=>sum(data.death),
    "dimension_checks"=>dsga_checks,"baseline_graph"=>Dict(pairs(summary(base_model))),"singleton_pruned_graph"=>pruned,
    "baseline_audit"=>Dict(pairs(audit(base_model))),"source_state"=>state,
    "manifest_sha256"=>hashfile(joinpath(ROOT,"environment","Manifest.toml")),"input_checksums_sha256"=>hashfile(joinpath(ROOT,"data","checksums.toml")),
    "scripts_sha256"=>Dict(f=>hashfile(joinpath(@__DIR__,f)) for f in ("common.jl","run.jl","plot.jl","prepare.py","export_annotation.R")))
open(joinpath(ROOT,"results","run_metadata.toml"),"w") do io;TOML.print(io,metadata);end
@testset "Nicolau 2011 scientific integrity" begin
    @test length(data.ids)==308 && length(unique(data.ids))==308
    @test all(isfinite,data.T) && all(isfinite,data.N)
    @test all(s.covered_samples==308 for s in sensitivity)
    @test length(sensitivity)==18
    @test dsga_checks["10"]["orthogonality_relative"]<1e-12
    @test dsga_checks["10"]["decomposition_relative"]<1e-12
    @test maximum(abs.(vec(sqrt.(sum(abs2,baseline.N;dims=1))).-baseline.target))<1e-10
    @test maximum(abs.(vec(sqrt.(sum(abs2,baseline.T;dims=1))).-baseline.target))<1e-10
    @test all(j->norm(baseline.L1[:,j])>1e-4,1:13)
    @test !isempty(selected.selected)
    @test !(base_model.f≈models["power1"].f)
    @test all(isnan,data.clinical_er[1:13])
end
println("Baseline: ",summary(base_model));println("Singleton-pruned: ",pruned)
if !("--no-plots" in ARGS)
    include("plot.jl")
    plot_results(models,sensitivity,data,baseline,selected)
end
println("Saved 18 Mapper configurations with all 308 samples, DSGA diagnostics and provenance.")
