# Run with the private environment; see README.md. All preprocessing is label-free.
include("common.jl")
using LIBSVM, Random, TOML
BLAS.set_num_threads(1)
mkpath(RESULTS)
const CACHE = get(ENV,"REININGHAUS_CACHE",joinpath(tempdir(),"reininghaus2015-kernels"))
mkpath(CACHE)
data = load_data()

# Same 10 deterministic outer splits for every HKS setting; never select a HKS
# time from local test results. Indices are 1-based; source shape IDs are 0-based.
function make_splits(labels)
    splits=[]
    open(joinpath(RESULTS,"splits.csv"),"w") do io
        println(io,"repetition,shape_id,class_id,partition,inner_fold")
        for trial in 1:10
            rng=MersenneTwister(SEED+trial)
            train=Int[]; test=Int[]; fold=zeros(Int,length(labels))
            for c in 1:15
                ids=shuffle(rng,findall(==(c),labels))
                append!(train,ids[1:14]); append!(test,ids[15:20])
                # Distribute each identity over all folds, with shuffled extra
                # folds, then assign to randomized training points.
                fs=shuffle(rng,vcat(collect(1:10),shuffle(rng,collect(1:10))[1:4]))
                fold[ids[1:14]]=fs
            end
            sort!(train); sort!(test)
            push!(splits,(;train,test,fold))
            for i in eachindex(labels)
                println(io,"$trial,$(i-1),$(labels[i]),$(fold[i]==0 ? "test" : "train"),$(fold[i])")
            end
        end
    end
    return splits
end
splits=make_splits(data.labels)

open(joinpath(RESULTS,"data_summary.csv"),"w") do io
    println(io,"hks_index,shapes,finite_positive_h1,diagonal_h1,negative_dim_records,min_points,max_points,mean_points,min_birth,max_death")
    for t in 1:10
        ds=data.diagrams[t]
        println(io,join([t,300,sum(length,ds),data.diagonal[t],data.essential[t],minimum(length,ds),maximum(length,ds),mean(length.(ds)),minimum(birth(p) for d in ds for p in d),maximum(death(p) for d in ds for p in d)],','))
    end
end

function get_kernel(t,s)
    file=joinpath(CACHE,"hks$(t)-sigma$(s).jls")
    # Cache is tied to exact source data and current kernel implementation.
    signature=(SOURCE_SHA256,bytes2hex(sha256(read(joinpath(@__DIR__,"../../../..","TDAPersistenceDiagrams.jl","src","additional_kernels.jl")))))
    if isfile(file)
        saved=deserialize(file)
        saved.signature==signature && return saved.K,saved.seconds
    end
    seconds=@elapsed K=pss_matrix(data.diagrams[t],s)
    serialize(file,(;K,seconds,signature))
    return K,seconds
end

function score(K,labels,fit_ids,predict_ids,cost)
    model=svmtrain(K[fit_ids,fit_ids],labels[fit_ids];kernel=LIBSVM.Kernel.Precomputed,cost,tolerance=0.001,nt=1)
    predicted,_=svmpredict(model,K[fit_ids,predict_ids];nt=1)
    return count(predicted.==labels[predict_ids]),predicted
end

function classify(kernels,labels,split)
    # The identical fold assignment is reused for every candidate. The score is
    # micro accuracy on all210 held-out training observations, not the90 tests.
    best_correct=-1; best_s=1; best_c=1
    cv_scores=zeros(Int,length(SIGMAS),length(COSTS))
    for (si,K) in enumerate(kernels), (ci,cost) in enumerate(COSTS)
        correct=0
        for f in 1:10
            validation=[i for i in split.train if split.fold[i]==f]
            fit=[i for i in split.train if split.fold[i]!=f]
            n,_=score(K,labels,fit,validation,cost)
            correct+=n
        end
        cv_scores[si,ci]=correct
        # Deterministic ties: first sigma then first C (both increasing).
        if correct>best_correct
            best_correct=correct; best_s=si; best_c=ci
        end
    end
    correct,predicted=score(kernels[best_s],labels,split.train,split.test,COSTS[best_c])
    return (;best_s,best_c,best_correct,correct,predicted,cv_scores)
end

targets="--all-classification" in ARGS ? collect(1:10) : [2,10]
retrieval_only="--retrieval-only" in ARGS
started=time()
open(joinpath(RESULTS,"retrieval_grid.csv"),"w") do rio
open(joinpath(RESULTS,"retrieval.csv"),"w") do rbest
open(joinpath(RESULTS,"classification.csv"),"w") do cio
open(joinpath(RESULTS,"cv_grid.csv"),"w") do cvio
open(joinpath(RESULTS,"predictions.csv"),"w") do pio
    println(rio,"hks_index,sigma,correct,total,accuracy_percent,kernel_seconds")
    println(rbest,"hks_index,oracle_sigma,correct,total,accuracy_percent,paper_percent,difference_pp")
    println(cio,"hks_index,repetition,sigma,cost,cv_correct,cv_total,test_correct,test_total,accuracy_percent,svm_seconds")
    println(cvio,"hks_index,repetition,sigma,cost,correct,total")
    println(pio,"hks_index,repetition,shape_id,true_class,predicted_class")
    for t in 1:10
        kernels=Matrix{Float64}[]; accuracies=Float64[]
        for s in SIGMAS
            K,seconds=get_kernel(t,s)
            NORMALIZED&&(K=normalize_gram(K))
            nn=nearest_neighbors(K,data.labels)
            push!(kernels,K);push!(accuracies,nn.accuracy)
            println(rio,"$t,$s,$(round(Int,3nn.accuracy)),300,$(nn.accuracy),$seconds")
            flush(rio)
            println("HKS$t sigma=$s NN=$(round(nn.accuracy;digits=2)) kernel=$(round(seconds;digits=2))s")
            flush(stdout)
        end
        si=argmax(accuracies)
        a=accuracies[si]
        println(rbest,"$t,$(SIGMAS[si]),$(round(Int,3a)),300,$a,$(PAPER_RETRIEVAL[t]),$(a-PAPER_RETRIEVAL[t])");flush(rbest)
        if t in targets && !retrieval_only
            for trial in 1:10
                seconds=@elapsed out=classify(kernels,data.labels,splits[trial])
                a=100out.correct/90
                println(cio,"$t,$trial,$(SIGMAS[out.best_s]),$(COSTS[out.best_c]),$(out.best_correct),210,$(out.correct),90,$a,$seconds");flush(cio)
                for si in eachindex(SIGMAS),ci in eachindex(COSTS)
                    println(cvio,"$t,$trial,$(SIGMAS[si]),$(COSTS[ci]),$(out.cv_scores[si,ci]),210")
                end
                flush(cvio)
                for (i,p) in zip(splits[trial].test,out.predicted)
                    println(pio,"$t,$trial,$(i-1),$(data.labels[i]),$p")
                end
                flush(pio)
                println("HKS$t split$trial CV=$(out.best_correct)/210 test=$(out.correct)/90 sigma=$(SIGMAS[out.best_s]) C=$(COSTS[out.best_c]) $(round(seconds;digits=2))s")
                flush(stdout)
            end
        end
    end
end;end;end;end;end

cls=readdlm(joinpath(RESULTS,"classification.csv"),',',Float64;header=true)[1]
if !isempty(cls)
    open(joinpath(RESULTS,"classification_summary.csv"),"w") do io
        println(io,"hks_index,repetitions,mean_percent,sample_sd_percent,paper_mean,paper_sd,difference_pp")
        for t in targets
            a=cls[cls[:,1].==t,9]
            println(io,"$t,$(length(a)),$(mean(a)),$(std(a)),$(PAPER_CLASSIFICATION[t]),$(PAPER_SD[t]),$(mean(a)-PAPER_CLASSIFICATION[t])")
        end
    end
end
open(joinpath(RESULTS,"execution.toml"),"w") do io
    TOML.print(io,Dict("julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"seed"=>SEED,"source_sha256"=>SOURCE_SHA256,"sigma_grid"=>SIGMAS,"cost_grid"=>COSTS,"classified_hks"=>targets,"elapsed_seconds"=>time()-started,"kernel_normalization"=>NORMALIZED ? "unit diagonal, zero rows retained" : "none","library_kernel"=>"TDAPersistenceDiagrams.PersistenceScaleSpaceKernel","kernel_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../../..","TDAPersistenceDiagrams.jl","src","additional_kernels.jl"))))))
end
println("Finished in $(round(time()-started;digits=2))s")
