include("common.jl")
using Test, TOML
BLAS.set_num_threads(1)
data=load_data()
checks=Dict{String,Any}()

# Independent high precision equation10, evaluated with the reflected point.
function analytic(F,G,sigma)
    s=BigFloat(sigma)
    value=BigFloat(0)
    for x in F,y in G
        isfinite(x)&&isfinite(y)||continue
        b,d=BigFloat(birth(x)),BigFloat(death(x))
        u,v=BigFloat(birth(y)),BigFloat(death(y))
        value+=exp(-((b-u)^2+(d-v)^2)/(8s))-exp(-((b-v)^2+(d-u)^2)/(8s))
    end
    value/(8BigFloat(pi)*s)
end

function author_time_kernel(F,G,time)
    T=BigFloat(time);total=BigFloat(0)
    for x in F,y in G
        isfinite(x)&&isfinite(y)||continue
        b,d=BigFloat(birth(x)),BigFloat(death(x))
        u,v=BigFloat(birth(y)),BigFloat(death(y))
        for (a,p1,p2) in [(1,b,d),(-1,d,b)], (c,q1,q2) in [(1,u,v),(-1,v,u)]
            total+=a*c*exp(-((p1-q1)^2+(p2-q2)^2)/(2T))
        end
    end
    total/(4BigFloat(pi)*T)
end

@testset "SHREC source and analytic PSS audit" begin
    @test size(data.rows)==(210816,5)
    @test sort(unique(Int.(data.rows[:,1])))==collect(0:299)
    @test sort(unique(Int.(data.rows[:,2])))==collect(1:10)
    @test [count(==(c),data.labels) for c in 1:15]==fill(20,15)
    @test sum(data.diagonal)==5576
    @test sum(sum(length,ds) for ds in data.diagrams)==147316
    @test sum(data.essential)==9290
    @test all(d->all(p->isfinite(p)&&birth(p)<death(p),d),Iterators.flatten(data.diagrams))
    fixtures=[PersistenceDiagram([(0.0,1.0)]),
              PersistenceDiagram([(0.0,1.0),(0.0,1.0),(0.2,0.20000000001)]),
              PersistenceDiagram(Tuple{Float64,Float64}[]),
              PersistenceDiagram([(1.0,1.0),(2.0,Inf)]),
              data.diagrams[10][1],data.diagrams[10][101],data.diagrams[10][201]]
    max_error=0.0
    max_relative=0.0
    for s in [2.0^-12,0.25,1.0,64.0], F in fixtures,G in fixtures
        actual=PersistenceScaleSpaceKernel(;sigma=s)(F,G)
        expected=Float64(analytic(F,G,s))
        max_error=max(max_error,abs(actual-expected))
        abs(expected)>1e-200&&(max_relative=max(max_relative,abs(actual-expected)/abs(expected)))
        @test isapprox(actual,expected;atol=1e-13,rtol=1e-11)
        @test isapprox(actual,Float64(author_time_kernel(F,G,4s));atol=1e-13,rtol=1e-11)
    end
    ds=data.diagrams[10][1:12]
    @test pss_matrix(ds,0.25)≈Matrix(kernel_matrix(PersistenceScaleSpaceKernel(sigma=0.25),ds))
    @test PersistenceScaleSpaceKernel()(fixtures[2],fixtures[1])≈2PersistenceScaleSpaceKernel()(fixtures[1],fixtures[1])
    checks["analytic_max_absolute_error"]=max_error
    checks["analytic_max_relative_error_above_1e_minus_200"]=max_relative
    checks["empty_diagrams_hks10"]=count(isempty,data.diagrams[10])
end

@testset "outer split and inner CV isolation" begin
    rows=readdlm(joinpath(RESULTS,"splits.csv"),',';header=true)[1]
    @test size(rows)==(3000,5)
    for trial in 1:10
        r=rows[rows[:,1].==trial,:]
        @test sort(Int.(r[:,2]))==collect(0:299)
        tr=r[r[:,4].=="train",:];te=r[r[:,4].=="test",:]
        @test size(tr,1)==210&&size(te,1)==90
        @test isempty(intersect(Int.(tr[:,2]),Int.(te[:,2])))
        @test all(te[:,5].==0)
        @test all(x->1<=x<=10,tr[:,5])
        for c in 1:15
            @test count(==(c),tr[:,3])==14&&count(==(c),te[:,3])==6
            @test sort(unique(Int.(tr[tr[:,3].==c,5])))==collect(1:10)
        end
    end
end

@testset "published comparisons and prediction accounting" begin
    cls=readdlm(joinpath(RESULTS,"classification.csv"),',',Float64;header=true)[1]
    cv=readdlm(joinpath(RESULTS,"cv_grid.csv"),',',Float64;header=true)[1]
    predictions=readdlm(joinpath(RESULTS,"predictions.csv"),',',Float64;header=true)[1]
    @test size(cls,2)==10
    @test size(cv,1)==size(cls,1)*length(SIGMAS)*length(COSTS)
    @test sort(unique(cv[:,3]))==SIGMAS
    @test sort(unique(cv[:,4]))==COSTS
    @test size(predictions,1)==90size(cls,1)
    for r in eachrow(cls)
        t,trial=Int(r[1]),Int(r[2])
        candidate=cv[(cv[:,1].==t).&(cv[:,2].==trial),:]
        @test r[5]==maximum(candidate[:,5])
        winner=findfirst(==(r[5]),candidate[:,5])
        @test candidate[winner,3:4]==r[3:4]
        ps=predictions[(predictions[:,1].==t).&(predictions[:,2].==trial),:]
        @test r[7]==count(ps[:,4].==ps[:,5])
        @test r[9]≈100r[7]/90
    end
    nnrows=readdlm(joinpath(RESULTS,"retrieval.csv"),',',Float64;header=true)[1]
    grid=readdlm(joinpath(RESULTS,"retrieval_grid.csv"),',',Float64;header=true)[1]
    @test size(nnrows)==(10,7)
    @test size(grid,1)==10length(SIGMAS)
    min_eigen=Inf
    CACHE=get(ENV,"REININGHAUS_CACHE",joinpath(tempdir(),"reininghaus2015-kernels"))
    open(joinpath(RESULTS,"retrieval_predictions.csv"),"w") do io
        println(io,"hks_index,query_shape_id,neighbor_shape_id,true_class,neighbor_class,squared_distance")
        for r in eachrow(nnrows)
            t=Int(r[1]);s=r[2]
            @test r[5]==maximum(grid[grid[:,1].==t,5])
            K=deserialize(joinpath(CACHE,"hks$(t)-sigma$(s).jls")).K
            NORMALIZED&&(K=normalize_gram(K))
            @test issymmetric(K)&&all(isfinite,K)
            lambda=minimum(eigvals(Symmetric(K)))
            min_eigen=min(min_eigen,lambda)
            @test lambda>=-1e-10max(1.0,maximum(diag(K)))
            nn=nearest_neighbors(K,data.labels)
            @test nn.accuracy≈r[5]
            @test all(nn.predicted.!=collect(1:300))
            for i in 1:300
                j=nn.predicted[i]
                println(io,"$t,$(i-1),$(j-1),$(data.labels[i]),$(data.labels[j]),$(nn.distance[i])")
            end
        end
    end
    checks["minimum_selected_gram_eigenvalue"]=min_eigen
    checks["classified_settings"]=sort(unique(Int.(cls[:,1])))
    checks["classification_repetitions"]=size(cls,1)
end

if !NORMALIZED
@testset "refined retrieval accounting" begin
    table=readdlm(joinpath(RESULTS,"retrieval_refined.csv"),',',Float64;header=true)[1]
    grid=readdlm(joinpath(RESULTS,"retrieval_refined_grid.csv"),',',Float64;header=true)[1]
    predictions=readdlm(joinpath(RESULTS,"retrieval_refined_predictions.csv"),',',Float64;header=true)[1]
    @test size(table)==(10,8)
    @test size(predictions)==(3000,6)
    for t in 1:10
        g=grid[grid[:,1].==t,:];p=predictions[predictions[:,1].==t,:]
        @test table[t,5]==maximum(g[:,3])
        @test table[t,8]==maximum(g[:,4])
        @test table[t,3]==count(p[:,4].==p[:,5])
        @test all(p[:,2].!=p[:,3])
        @test sort(Int.(p[:,2]))==collect(0:299)
        @test all(s->s in g[:,2],SIGMAS)
    end
end
end
checks["status"]="passed"
open(joinpath(RESULTS,"verification.toml"),"w") do io
    TOML.print(io,checks)
end
println(checks)
