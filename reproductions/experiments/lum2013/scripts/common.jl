using TDAmapper
using TDAmapper.ImageCovers, TDAmapper.IntervalCovers, TDAmapper.Refiners, TDAmapper.Nerves
using Statistics, LinearAlgebra, DelimitedFiles, SHA, TOML, Random
import Graphs, Clustering, Distances

const ROOT = dirname(@__DIR__)
filehash(path) = bytes2hex(sha256(read(path)))
unquote(s) = strip(s, '"')

function load_data()
    sources = TOML.parsefile(joinpath(ROOT, "data", "sources.toml"))
    for filename in ("GSE2034_series_matrix.txt.gz", "GPL96.annot.gz", "clinical.tsv")
        filehash(joinpath(ROOT, "data", filename)) == sources[filename]["sha256"] || error("Source checksum changed: $filename")
    end
    ids = String[]; probes = String[]; rows = Vector{Vector{Float64}}()
    matrix_path = joinpath(ROOT, "data", "GSE2034_series_matrix.txt.gz")
    open(`gzip -dc $matrix_path`) do io
        for line in eachline(io)
            startswith(line, "\"ID_REF\"") && (ids = unquote.(split(line, '\t')); popfirst!(ids); continue)
            startswith(line, "\"") || continue
            cells = split(line, '\t')
            push!(probes, unquote(cells[1]))
            push!(rows, parse.(Float64, cells[2:end]))
        end
    end
    raw = permutedims(reduce(hcat, rows))
    size(raw) == (22283, 286) || error("Wrong expression dimensions")
    length(unique(ids)) == 286 || error("Non-unique sample accessions")
    all(isfinite, raw) && minimum(raw) > 0 || error("Invalid positive GEO intensity")
    clinical, header = readdlm(joinpath(ROOT, "data", "clinical.tsv"), '\t', Any; header=true)
    String(header[1,2]) == "GEO asscession number" || error("Unexpected clinical schema")
    clinical_ids = String.(clinical[:,2])
    Set(ids) == Set(clinical_ids) && length(unique(clinical_ids)) == 286 || error("Clinical sample mismatch")
    alignment = [only(findall(==(gsm), clinical_ids)) for gsm in ids]
    relapse = Int.(clinical[alignment,5]); er = String.(clinical[alignment,6])
    pid = Int.(clinical[alignment,1]); followup = Float64.(clinical[alignment,4])
    sum(relapse) == 107 && count(==("ER-"), er) == 77 || error("Unexpected original clinical labels")
    # Keep annotated probes separately; controls may enter the variance ranking,
    # as the exact historical GSE2034 feature-selection rule was not published.
    annotation = Dict{String,Vector{String}}()
    ann_path = joinpath(ROOT,"data","GPL96.annot.gz")
    open(`gzip -dc $ann_path`) do io
        for line in eachline(io)
            startswith(line, "#") && continue
            cells = split(line, '\t'; keepempty=true)
            length(cells) >= 3 || continue
            annotation[cells[1]] = String.(split(cells[3], "///"))
        end
    end
    logexpr = log2.(raw)
    esr_probes = findall(p -> "ESR1" in get(annotation, p, String[]), probes)
    isempty(esr_probes) && error("No ESR1 probes")
    # Average log-expression over all platform probes annotated to each marker.
    esr = vec(mean(logexpr[esr_probes,:]; dims=1))
    markers = ("CCL13", "CCL3", "CXCL13", "PF4V1")
    marker_rows = [findall(p -> marker in get(annotation,p,String[]), probes) for marker in markers]
    all(!isempty,marker_rows) || error("Missing published chemokine marker")
    marker_values = reduce(hcat, [vec(mean(logexpr[ii,:]; dims=1)) for ii in marker_rows])
    marker_score = vec(mean(marker_values; dims=2))
    (; ids, probes, raw, logexpr, relapse, er, pid, followup, annotation, esr, esr_probes, markers, marker_rows, marker_values, marker_score)
end

function correlation_geometry(data; nprobes=1553, center_genes=false)
    variances = vec(var(data.logexpr; dims=2))
    order = sortperm(eachindex(variances); by=i -> (-variances[i], data.probes[i]))
    selected = order[1:nprobes]
    A = copy(data.logexpr[selected,:])
    center_genes && (A .-= mean(A; dims=2))
    Z = A .- mean(A; dims=1)
    Z ./= sqrt.(sum(abs2,Z; dims=1))
    D = clamp.(1 .- Z' * Z, 0, 2)
    D = Matrix(Symmetric(D))
    D[diagind(D)] .= 0
    centrality = vec(maximum(D; dims=1))
    # Equalization is a local empirical-rank approximation, independent of outcome.
    order_c = sortperm(centrality)
    rank = zeros(length(centrality))
    for (r,i) in enumerate(order_c); rank[i] = (r-1)/(length(rank)-1); end
    (; A, D, centrality, rank, selected, variances)
end

# This adapter clusters ORIGINAL PATIENT INDICES using the cached exact distance
# submatrix. Integers are identifiers, never scalar geometric coordinates.
struct CachedSingleLinkage <: TDAmapper.Refiners.AbstractRefiner
    distances::Matrix{Float64}
    bins::Int
end
function (r::CachedSingleLinkage)(ids::Vector{Int})
    length(ids) == 1 && return [1]
    D = r.distances[ids,ids]
    tree = Clustering.hclust(D; linkage=:single)
    h = TDAmapper.Refiners.cutoff_at_first_empty_bin(tree.heights, maximum(D), r.bins)
    Clustering.cutree(tree; h)
end

# Local interpretation of Gain g: width / step = g, hence overlap = 1 - 1/g.
# This is a declared convention, not a verified Ayasdi reimplementation.
function gain_cover(f; resolution=70, gain=3.0)
    lo,hi = extrema(f)
    width = (hi-lo) * gain / (resolution + gain - 1)
    step = width/gain
    intervals = [Interval(lo + (i-1)*step, i==resolution ? hi : lo+(i-1)*step+width) for i in 1:resolution]
    ManualCover(intervals)
end

function construct(geometry, relapse; resolution=70, bins=10, supervised=true)
    f = geometry.rank
    cover = supervised ? R2Cover(collect(zip(f,relapse)), gain_cover(f; resolution), gain_cover(Float64.(relapse); resolution=30)) : R1Cover(f, gain_cover(f; resolution))
    refiner = CachedSingleLinkage(geometry.D, bins)
    M = classical_mapper(collect(eachindex(f)), cover, refiner, SimpleNerve())
    (; M, cover, refiner, geometry, supervised)
end

function graph_summary(M, data)
    components = Graphs.connected_components(M.g)
    groups = [sort(unique(vcat(M.C[c]...))) for c in components]
    nonrelapse = [length(ii) for ii in groups if all(==(0), data.relapse[ii])]
    relapsed = [length(ii) for ii in groups if all(==(1), data.relapse[ii])]
    (; nodes=Graphs.nv(M.g), edges=Graphs.ne(M.g), components=length(components),
       graph_cycle_rank=Graphs.ne(M.g)-Graphs.nv(M.g)+length(components),
       covered_patients=length(unique(vcat(M.C...))),
       largest_component_patients=maximum(length.(groups)),
       largest_nonrelapse_component_patients=isempty(nonrelapse) ? 0 : maximum(nonrelapse),
       largest_relapse_component_patients=isempty(relapsed) ? 0 : maximum(relapsed),
       singleton_nodes=count(c -> length(c)==1, M.C),
       all_nodes_outcome_pure=all(c -> length(unique(data.relapse[c]))==1, M.C))
end

function csv(path, rows; header=isempty(rows) ? String[] : string.(keys(first(rows))))
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows
            println(io,join(values(row),','))
        end
    end
end

function export_model(name, model, data)
    M = model.M; out=joinpath(ROOT,"results",name); mkpath(out)
    csv(joinpath(out,"nodes.csv"),[(; node_id=i, patients=length(c), degree=Graphs.degree(M.g,i), mean_centrality=mean(model.geometry.centrality[c]), mean_rank=mean(model.geometry.rank[c]), mean_esr1=mean(data.esr[c]), mean_four_marker_score=mean(data.marker_score[c]), relapse_fraction=mean(data.relapse[c]), er_negative_fraction=mean(data.er[c].=="ER-")) for (i,c) in enumerate(M.C)])
    csv(joinpath(out,"membership.csv"),[(;node_id=i,gsm=data.ids[j],patient_id=data.pid[j]) for (i,c) in enumerate(M.C) for j in c])
    csv(joinpath(out,"edges.csv"),[(;source=Graphs.src(e),target=Graphs.dst(e),shared_patients=length(intersect(M.C[Graphs.src(e)],M.C[Graphs.dst(e)]))) for e in Graphs.edges(M.g)];header=["source","target","shared_patients"])
    csv(joinpath(out,"components.csv"),[(;component_id=i,nodes=length(c),patients=length(unique(vcat(M.C[c]...))),relapse_fraction=mean(data.relapse[unique(vcat(M.C[c]...))]),mean_esr1=mean(data.esr[unique(vcat(M.C[c]...))]),mean_four_marker_score=mean(data.marker_score[unique(vcat(M.C[c]...))])) for (i,c) in enumerate(Graphs.connected_components(M.g))])
end

function audit(model, data)
    M=model.M; n=length(data.ids)
    sort(unique(vcat(M.C...))) == collect(1:n) || error("Patient coverage incomplete")
    for i in 1:length(M.C), j in i+1:length(M.C)
        Graphs.has_edge(M.g,i,j) == !isempty(intersect(M.C[i],M.C[j])) || error("Incorrect nerve edge")
    end
    for ids in TDAmapper.make_cover(model.cover)
        length(ids)<2 && continue
        D=model.geometry.D[ids,ids]; hcl=Clustering.hclust(D; linkage=:single)
        cutoff=TDAmapper.Refiners.cutoff_at_first_empty_bin(hcl.heights,maximum(D),model.refiner.bins)
        g=Graphs.SimpleGraph(length(ids))
        for i in eachindex(ids),j in i+1:length(ids)
            D[i,j]<=cutoff && Graphs.add_edge!(g,i,j)
        end
        cc=Graphs.connected_components(g); lab=zeros(Int,length(ids))
        for (k,c) in enumerate(cc); lab[c].=k; end
        computed=model.refiner(ids)
        all((computed[i]==computed[j])==(lab[i]==lab[j]) for i in eachindex(ids),j in eachindex(ids)) || error("Single-linkage audit failed")
    end
    A=model.geometry.A; D=model.geometry.D
    err=maximum(abs(D[i,j]-(1-cor(A[:,i],A[:,j]))) for (i,j) in [(1,2),(1,286),(30,72),(140,235)])
    err<1e-12 || error("Pearson-distance audit failed")
    (;covered_patients=n,all_pairwise_nerve_edges_checked=true,single_linkage_checked_against_threshold_components=true,pearson_max_abs_error=err)
end
