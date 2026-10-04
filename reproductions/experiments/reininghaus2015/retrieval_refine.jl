# Optional retrieval-only refinement, never used by the SVM model selection.
include("common.jl")
using TOML
data=load_data()
CACHE=get(ENV,"REININGHAUS_CACHE",joinpath(tempdir(),"reininghaus2015-kernels"))
mkpath(CACHE)
coarse=readdlm(joinpath(ROOT,"results","retrieval.csv"),',',Float64;header=true)[1]
grid=readdlm(joinpath(ROOT,"results","retrieval_grid.csv"),',',Float64;header=true)[1]
started=time()
signature=(SOURCE_SHA256,bytes2hex(sha256(read(joinpath(@__DIR__,"../../../..","TDAPersistenceDiagrams.jl","src","additional_kernels.jl")))))
open(joinpath(ROOT,"results","retrieval_refined_grid.csv"),"w") do gio
open(joinpath(ROOT,"results","retrieval_refined.csv"),"w") do bio
open(joinpath(ROOT,"results","retrieval_refined_predictions.csv"),"w") do pio
    println(gio,"hks_index,sigma,accuracy_percent,normalized_accuracy_percent,kernel_seconds")
    println(bio,"hks_index,oracle_sigma,correct,total,accuracy_percent,paper_percent,difference_pp,normalized_max_percent")
    println(pio,"hks_index,query_shape_id,neighbor_shape_id,true_class,neighbor_class,squared_distance")
    for t in 1:10
        center=log2(coarse[t,2])
        # Refinement uses the labeled benchmark, consistent with oracle retrieval.
        # It is not a generalization estimate or a parameter grid for the SVM.
        scales=sort(unique(vcat(SIGMAS,2.0 .^ collect(center-1.5:0.5:center+1.5))))
        best=-Inf; best_s=0.0; best_nn=nothing; best_norm=-Inf
        for s in scales
            file=joinpath(CACHE,"hks$(t)-sigma$(s).jls")
            saved=isfile(file) ? deserialize(file) : nothing
            if saved!==nothing&&saved.signature==signature
                K=saved.K;seconds=saved.seconds
            else
                seconds=@elapsed K=pss_matrix(data.diagrams[t],s)
                serialize(file,(;K,seconds,signature))
            end
            nn=nearest_neighbors(K,data.labels)
            normalized_nn=nearest_neighbors(normalize_gram(K),data.labels)
            println(gio,"$t,$s,$(nn.accuracy),$(normalized_nn.accuracy),$seconds");flush(gio)
            if nn.accuracy>best
                best=nn.accuracy;best_s=s;best_nn=nn
            end
            best_norm=max(best_norm,normalized_nn.accuracy)
        end
        println(bio,"$t,$best_s,$(round(Int,3best)),300,$best,$(PAPER_RETRIEVAL[t]),$(best-PAPER_RETRIEVAL[t]),$best_norm");flush(bio)
        for i in 1:300
            j=best_nn.predicted[i]
            println(pio,"$t,$(i-1),$(j-1),$(data.labels[i]),$(data.labels[j]),$(best_nn.distance[i])")
        end
        flush(pio)
        println("Refined retrieval HKS$t: raw=$best normalized=$best_norm sigma=$best_s");flush(stdout)
    end
end;end;end
open(joinpath(ROOT,"results","retrieval_refine_execution.toml"),"w") do io
    TOML.print(io,Dict("rule"=>"coarse winner exponent plus -1.5:0.5:1.5; union with coarse grid","elapsed_seconds"=>time()-started,"selection"=>"oracle over all300 labeled queries; excluded from classification"))
end
