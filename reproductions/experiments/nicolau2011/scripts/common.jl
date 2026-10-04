using TDAmapper
using TDAmapper.ImageCovers, TDAmapper.IntervalCovers, TDAmapper.Refiners, TDAmapper.Nerves
using Statistics, LinearAlgebra, DelimitedFiles, SHA, TOML, Random
import LinearAlgebra: norm
import Graphs, Clustering
const ROOT = dirname(@__DIR__)
hashfile(p) = bytes2hex(sha256(read(p)))

function csv(path, rows; header=isempty(rows) ? String[] : string.(keys(first(rows))))
    mkpath(dirname(path))
    escape(x) = occursin(r"[,\"\n]", string(x)) ? "\""*replace(string(x),'"'=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows; println(io,join(escape.(collect(values(row))),',')); end
    end
end

function readmatrix(path;skip=0,metadata=1)
    ids=String[]; genes=String[]; rows=Vector{Vector{Float64}}(); names=String[]
    open(`gzip -dc $path`) do io
        ids=String.(split(readline(io),'\t')[metadata+1:end])
        for (i,line) in enumerate(eachline(io))
            i<=skip && continue
            a=split(line,'\t';keepempty=true)
            push!(genes,a[1]);push!(names,metadata>1 ? a[2] : a[1])
            push!(rows,[isempty(x) ? NaN : parse(Float64,x) for x in a[metadata+1:end]])
        end
    end
    (;genes,names,ids,A=permutedims(reduce(hcat,rows)))
end

# Gene-oriented KNN, k=10, over the complete genes in the SAME tumor cohort.
# Euclidean distance uses only the target gene's observed patients; inverse-distance
# weights interpolate each missing entry. This deterministic, unpartitioned rule is
# an explicit local choice: the historical impute.knn options/seed were not given.
function impute_knn!(A;k=10)
    complete=findall(i->all(isfinite,view(A,i,:)),axes(A,1))
    donors=A[complete,:]; donor_norms=vec(sum(abs2,donors;dims=2))
    bad=findall(i->any(!isfinite,view(A,i,:)),axes(A,1))
    for i in bad
        missing=findall(!isfinite,view(A,i,:)); observed=setdiff(axes(A,2),missing)
        target=copy(A[i,:]);target[missing].=0
        d2=donor_norms .-vec(sum(abs2,donors[:,missing];dims=2)).+sum(abs2,target).-2 .* (donors*target)
        near=partialsortperm(eachindex(d2),1:k;by=j->(d2[j],complete[j]))
        d=sqrt.(max.(d2[near],0)./length(observed))
        weights=any(<(1e-12),d) ? Float64.(d.<1e-12) : 1 ./d
        weights ./=sum(weights)
        A[i,missing].=vec(weights'*donors[near,missing])
    end
    (;imputed_genes=length(bad),complete_donors=length(complete))
end

function load_data()
    for (name,hash) in TOML.parsefile(joinpath(ROOT,"data","checksums.toml"))
        hashfile(joinpath(ROOT,"data",name))==hash || error("Input hash changed: $name")
    end
    tumor=readmatrix(joinpath(ROOT,"data","nki_log10.tsv.gz"))
    normal=readmatrix(joinpath(ROOT,"data","BCN.ugc219.pcl.gz");skip=1,metadata=3)
    size(tumor.A)==(24453,295) || error("Incorrect filtered historical tumor assay")
    size(normal.A)==(18971,13) || error("Incorrect archived BCN matrix")
    missing_before=count(!isfinite,tumor.A)
    imputed=impute_knn!(tumor.A)
    tumor.A .*=log2(10) # convert already logarithmic ratios; do not log these again
    all(isfinite,normal.A) || error("Archived collapsed normals contain missing values")
    ann,head=readdlm(joinpath(ROOT,"data","nki_annotation.tsv"),'\t',String;header=true)
    col=only(findall(==("HUGO.gene.symbol"),vec(head)))
    symbols=Dict(ann[i,1]=>ann[i,col] for i in axes(ann,1))
    normal_symbols=[first(split(s,"||")) for s in normal.names]
    groups=Dict{String,Vector{Int}}()
    for (i,probe) in enumerate(tumor.genes)
        symbol=get(symbols,probe,"")
        isempty(symbol) || push!(get!(groups,symbol,Int[]),i)
    end
    # No alias guesses, no contemporary UniGene remapping, no many-to-many join.
    unique_normal=Dict(s=>i for (i,s) in enumerate(normal_symbols) if !isempty(s) && count(==(s),normal_symbols)==1)
    genes=sort(collect(intersect(Set(keys(groups)),Set(keys(unique_normal)))))
    T=permutedims(reduce(hcat,[vec(mean(tumor.A[groups[s],:];dims=1)) for s in genes]))
    N=normal.A[[unique_normal[s] for s in genes],:]
    ids=vcat(normal.ids,tumor.ids)
    clinical,chead=readdlm(joinpath(ROOT,"data","clinical.tsv"),'\t',Any;header=true)
    ci=Int.(clinical[:,1]);ti=parse.(Int,replace.(tumor.ids,"Sample "=>""))
    length(unique(ci))==295 && Set(ci)==Set(ti) || error("Clinical/expression patient mismatch")
    align=[only(findall(==(j),ci)) for j in ti]
    cidx(s)=only(findall(==(s),vec(chead)))
    er=Int.(clinical[align,cidx("ESR1")]);death=Int.(clinical[align,cidx("EVENTdeath")])
    followup=Float64.(clinical[align,cidx("TIMEsurvival")])
    clinical_er=vcat(fill(NaN,13),Float64.(er))
    mapping=[(;symbol=s,normal_unigene=normal.genes[unique_normal[s]],nki_probes=join(tumor.genes[groups[s]],'|')) for s in genes]
    (;T,N,ids,genes,mapping,tumor_ids=tumor.ids,normal_ids=normal.ids,er,clinical_er,death,followup,missing_before,imputed)
end

# No intercept, no centering. These vectors live in a LINEAR subspace as in the SI.
function flat(N)
    F=similar(N)
    for j in axes(N,2)
        other=N[:,setdiff(axes(N,2),[j])]
        F[:,j]=other*(other\N[:,j])
    end
    F
end
function basis(N,r=10)
    S=svd(flat(N);full=false)
    r<=length(S.S) || error("HSM dimension exceeds available normals")
    S.U[:,1:r]
end
function dsga(T,N;r=10)
    target=mean(vec(sqrt.(sum(abs2,N;dims=1))))
    N=N .* (target ./sqrt.(sum(abs2,N;dims=1)))
    T=T .* (target ./sqrt.(sum(abs2,T;dims=1)))
    U=basis(N,r);Nc=U*(U'*T);Dc=T-Nc
    L1=similar(N)
    for j in axes(N,2)
        Uj=basis(N[:,setdiff(axes(N,2),[j])],r)
        L1[:,j]=N[:,j]-Uj*(Uj'*N[:,j])
    end
    S=svdvals(flat(N));n=size(N,1);k=size(N,2)
    wold=[S[l]^2/sum(abs2,S[l+1:end])*((n-l-1)*(k-l)/(n+k-2l)) for l in 1:k-1]
    (;T,N,U,Nc,Dc,L1,target,wold,singular_values=S)
end

function select_genes(Dc)
    q=[max(abs(quantile(view(Dc,i,:),.05)),abs(quantile(view(Dc,i,:),.95))) for i in axes(Dc,1)]
    relaxed=quantile(q,.85);stringent=quantile(q,.98)
    lax=findall(>(relaxed),q);strict=findall(>(stringent),q)
    C=Dc .-mean(Dc;dims=2)
    C ./=sqrt.(sum(abs2,C;dims=2))
    corr=C[lax,:]*C[strict,:]'
    counts=vec(sum(corr .>.6;dims=2))
    for (i,j) in enumerate(lax);j in strict && (counts[i]-=1);end
    selected=lax[counts .>=3]
    (;selected,q,relaxed,stringent,lax,strict,counts)
end
function geometry(A)
    C=A .-mean(A;dims=1)
    C ./=sqrt.(sum(abs2,C;dims=1))
    D=Matrix(Symmetric(clamp.(1 .-C'*C,0,2)))
    D[diagind(D)].=0
    (;A,D)
end

struct CachedSingleLinkage <: TDAmapper.Refiners.AbstractRefiner
    distances::Matrix{Float64}
    bins::Int
end
function (r::CachedSingleLinkage)(ids::Vector{Int})
    length(ids)==1 && return [1]
    D=r.distances[ids,ids];tree=Clustering.hclust(D;linkage=:single)
    h=TDAmapper.Refiners.cutoff_at_first_empty_bin(tree.heights,maximum(D),r.bins)
    Clustering.cutree(tree;h)
end
function interval_cover(f;resolution=15,overlap=.8)
    lo,hi=extrema(f)
    width=(hi-lo)/(resolution-(resolution-1)*overlap)
    step=width*(1-overlap)
    ManualCover([Interval(lo+(i-1)*step,i==resolution ? hi : lo+(i-1)*step+width) for i in 1:resolution])
end
function mapper(G;k=4,resolution=15,overlap=.8,bins=10)
    f=vec(sum(abs2,G.A;dims=1)).^(k/2)
    cover=R1Cover(f,interval_cover(f;resolution,overlap));refiner=CachedSingleLinkage(G.D,bins)
    M=classical_mapper(collect(eachindex(f)),cover,refiner,SimpleNerve())
    (;M,f,cover,G,refiner)
end
function summary(model)
    M=model.M; cc=Graphs.connected_components(M.g)
    groups=[sort(unique(vcat(M.C[c]...))) for c in cc]
    (;nodes=Graphs.nv(M.g),edges=Graphs.ne(M.g),components=length(cc),
      cycle_rank=Graphs.ne(M.g)-Graphs.nv(M.g)+length(cc),
      covered_samples=length(unique(vcat(M.C...))),singleton_nodes=count(c->length(c)==1,M.C),
      largest_component_samples=maximum(length.(groups)))
end
function audit(model)
    M=model.M;n=length(model.f)
    sort(unique(vcat(M.C...)))==collect(1:n) || error("Missing samples")
    for i in eachindex(M.C),j in i+1:length(M.C)
        Graphs.has_edge(M.g,i,j)==!isempty(intersect(M.C[i],M.C[j])) || error("Incorrect nerve")
    end
    for ids in TDAmapper.make_cover(model.cover)
        length(ids)<=1 && continue
        D=model.G.D[ids,ids];tree=Clustering.hclust(D;linkage=:single)
        h=TDAmapper.Refiners.cutoff_at_first_empty_bin(tree.heights,maximum(D),model.refiner.bins)
        g=Graphs.SimpleGraph(length(ids))
        for i in eachindex(ids),j in i+1:length(ids);D[i,j]<=h && Graphs.add_edge!(g,i,j);end
        lab=zeros(Int,length(ids))
        for (j,c) in enumerate(Graphs.connected_components(g));lab[c].=j;end
        computed=model.refiner(ids)
        all((computed[i]==computed[j])==(lab[i]==lab[j]) for i in eachindex(ids),j in eachindex(ids)) || error("Single-linkage audit")
    end
    err=maximum(abs(model.G.D[i,j]-(1-cor(model.G.A[:,i],model.G.A[:,j]))) for (i,j) in [(1,2),(1,308),(14,23),(72,140)])
    err<1e-12 || error("Pearson audit")
    (;all_sample_membership_checked=true,all_nerve_edges_checked=true,single_linkage_threshold_audit=true,pearson_max_abs_error=err)
end
function export_graph(name,model,data)
    out=joinpath(ROOT,"results",name);M=model.M
    csv(joinpath(out,"membership.csv"),[(;node=i,sample=data.ids[j],is_normal=j<=13) for (i,c) in enumerate(M.C) for j in c])
    csv(joinpath(out,"nodes.csv"),[(;node=i,samples=length(c),normal_samples=count(<=(13),c),mean_filter=mean(model.f[c]),mean_ER=any(>(13),c) ? mean(data.clinical_er[filter(>(13),c)]) : NaN) for (i,c) in enumerate(M.C)])
    csv(joinpath(out,"edges.csv"),[(;source=Graphs.src(e),target=Graphs.dst(e),shared_samples=length(intersect(M.C[Graphs.src(e)],M.C[Graphs.dst(e)]))) for e in Graphs.edges(M.g)];header=["source","target","shared_samples"])
end
