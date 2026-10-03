using TDAplots
using TDAmapper.ImageCovers, TDAmapper.IntervalCovers, TDAmapper.Refiners, TDAmapper.Nerves
using Statistics, DelimitedFiles, SHA, TOML, Random, LinearAlgebra
import Graphs, Clustering, Distances

const ROOT = dirname(@__DIR__)
const FEATURES = ("rw", "fpg", "glucose", "insulin", "sspg")
const GROUPS = ("normal", "chemical", "overt")

filehash(path) = bytes2hex(sha256(read(path)))

function load_data()
    path = joinpath(ROOT, "data", "diabetes.csv")
    provenance = TOML.parsefile(joinpath(ROOT, "data", "provenance.toml"))
    filehash(path) == provenance["csv_sha256"] || error("Dataset checksum mismatch")
    values, header = readdlm(path, ',', Any; header=true)
    vec(String.(header)) == ["patient_id", FEATURES..., "group"] || error("Unexpected schema")
    size(values) == (145, 7) || error("Unexpected dataset shape")
    ids = Int.(values[:, 1])
    ids == collect(1:145) || error("Patient IDs are not in source order")
    measurements = Float64.(values[:, 2:6])
    all(isfinite, measurements) || error("Non-finite measurement")
    groups = String.(values[:, 7])
    [count(==(g), groups) for g in GROUPS] == [76, 36, 33] || error("Unexpected labels")
    (; ids, measurements, groups)
end

function prepare_space(data, normalization)
    raw = EuclideanSpace(permutedims(data.measurements))
    normalization == "raw" && return raw
    normalization == "zscore" && return EuclideanSpace(standardize(raw))
    error("Unknown normalization: $normalization")
end

# Endpoint-aligned equal-width intervals. Overlap means intersection / interval width,
# not the JuliaTDA Uniform.expansion parameter. ManualCover preserves this convention.
function paper_cover(f; intervals, overlap)
    intervals >= 2 || error("Need at least two intervals")
    0 <= overlap < 1 || error("Overlap must be in [0,1)")
    lo, hi = extrema(f)
    hi > lo || error("Constant filter")
    width = (hi-lo) / (intervals-(intervals-1)*overlap)
    step = width*(1-overlap)
    cover = [Interval(lo+(i-1)*step, i == intervals ? hi : lo+(i-1)*step+width)
             for i in 1:intervals]
    R1Cover(f, ManualCover(cover))
end

# Sensitivity variant: histogram range is the range of merge heights, as suggested
# by a literal reading of section 3.1. This is NOT claimed to recover their MATLAB
# bin convention. The standard FirstEmptyBin instead extends the range to diameter.
struct MergeHeightHistogram <: TDAmapper.Refiners.AbstractRefiner
    num_bins::Int
end

function (r::MergeHeightHistogram)(X)
    length(X) == 1 && return [1]
    distances = Distances.pairwise(Distances.Euclidean(), as_matrix(X); dims=2)
    hc = Clustering.hclust(distances; linkage=:single)
    cutoff = TDAmapper.Refiners.cutoff_at_first_empty_bin(
        hc.heights, maximum(hc.heights), r.num_bins)
    Clustering.cutree(hc; h=cutoff)
end

function nearest_neighbor_bandwidth(X)
    distances = Distances.pairwise(Distances.Euclidean(), as_matrix(X); dims=2)
    for i in axes(distances, 1)
        distances[i, i] = Inf
    end
    median(vec(minimum(distances; dims=1)))
end

function construct(X, f; intervals, overlap, bins, histogram_range)
    cover = paper_cover(f; intervals, overlap)
    refiner = histogram_range == "diameter" ? FirstEmptyBin(num_bins=bins) :
              histogram_range == "merge_heights" ? MergeHeightHistogram(bins) :
              error("Unknown histogram range")
    M = classical_mapper(X, cover, refiner, SimpleNerve())
    (; M, cover, refiner)
end

function graph_summary(M)
    g = M.g
    degrees = Graphs.degree(g)
    components = Graphs.connected_components(g)
    (; nodes=Graphs.nv(g), edges=Graphs.ne(g), components=length(components),
       graph_cycle_rank=Graphs.ne(g)-Graphs.nv(g)+length(components),
       leaves=count(==(1), degrees), branch_nodes=count(>=(3), degrees),
       isolated_nodes=count(==(0), degrees),
       largest_component_nodes=maximum(length.(components)),
       covered_patients=length(unique(reduce(vcat, M.C))),
       smallest_node=minimum(length.(M.C)), largest_node=maximum(length.(M.C)))
end

function write_csv(path, rows)
    isempty(rows) && error("No rows for $path")
    mkpath(dirname(path))
    open(path, "w") do io
        writedlm(io, permutedims(collect(keys(first(rows)))), ',')
        for row in rows
            writedlm(io, permutedims(collect(values(row))), ',')
        end
    end
end

function export_model(name, X, f, data, model)
    M, cover, refiner = model.M, model.cover, model.refiner
    out = joinpath(ROOT, "results", name)
    mkpath(out)
    nodes = [(; node_id=i, size=length(c), degree=Graphs.degree(M.g, i),
               mean_density=mean(f[c]),
               normal=count(==("normal"), data.groups[c]),
               chemical=count(==("chemical"), data.groups[c]),
               overt=count(==("overt"), data.groups[c])) for (i,c) in enumerate(M.C)]
    write_csv(joinpath(out, "nodes.csv"), nodes)
    write_csv(joinpath(out, "membership.csv"),
              [(; node_id=i, patient_id=data.ids[j]) for (i,c) in enumerate(M.C) for j in c])
    write_csv(joinpath(out, "edges.csv"),
              [(; source=Graphs.src(e), target=Graphs.dst(e),
                 shared_patients=length(intersect(M.C[Graphs.src(e)], M.C[Graphs.dst(e)])))
               for e in Graphs.edges(M.g)])
    write_csv(joinpath(out, "filter.csv"),
              [(; patient_id=data.ids[j], density=f[j], group=data.groups[j]) for j in eachindex(f)])
    write_csv(joinpath(out, "intervals.csv"),
              [(; interval_id=i, lower=iv.a, upper=iv.b,
                 patients=count(x -> x in iv, f)) for (i,iv) in enumerate(cover.U)])
    (; name, X, f, M, cover, refiner, nodes)
end

# Integrity checks, rather than assertions that a particular drawing must appear.
function audit_model(model; bandwidth)
    X, f, M, cover, refiner = model.X, model.f, model.M, model.cover, model.refiner
    n = length(X)
    all_patients = sort(unique(reduce(vcat, M.C)))
    all_patients == collect(1:n) || error("Patients lost during cover refinement")
    expected_density = [mean(exp(-sum(abs2, X[i]-X[j])/(2bandwidth^2))
                             for j in 1:n) for i in 1:n]
    isapprox(f, expected_density; atol=1e-12, rtol=1e-12) || error("KDE mismatch")
    for i in 1:length(M.C), j in i+1:length(M.C)
        Graphs.has_edge(M.g, i, j) == !isempty(intersect(M.C[i], M.C[j])) || error("Nerve mismatch")
    end
    for ids in TDAmapper.make_cover(cover)
        length(ids) <= 1 && continue
        subset = X[ids]
        D = Distances.pairwise(Distances.Euclidean(), as_matrix(subset); dims=2)
        hc = Clustering.hclust(D; linkage=:single)
        upper = refiner isa FirstEmptyBin ? maximum(D) : maximum(hc.heights)
        cutoff = TDAmapper.Refiners.cutoff_at_first_empty_bin(hc.heights, upper, refiner.num_bins)
        graph = Graphs.SimpleGraph(length(ids))
        for i in eachindex(ids), j in i+1:length(ids)
            D[i,j] <= cutoff && Graphs.add_edge!(graph,i,j)
        end
        labels = refiner(subset)
        reference = zeros(Int, length(ids))
        for (i,c) in enumerate(Graphs.connected_components(graph))
            reference[c] .= i
        end
        all((labels[i]==labels[j]) == (reference[i]==reference[j])
            for i in eachindex(ids) for j in eachindex(ids)) || error("Single-linkage mismatch")
    end
    true
end
