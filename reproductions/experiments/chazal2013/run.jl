using ToMATo, MetricSpaces, Graphs, CairoMakie
using DelimitedFiles, Downloads, SHA, Statistics, TOML, Test
include("audit.jl")

const ROOT = @__DIR__
const CONFIG = TOML.parsefile(joinpath(ROOT, "config.toml"))
const OUT = joinpath(ROOT, "results")
mkpath(OUT)

function csvfile(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
end

function load_original()
    archive = joinpath(ROOT, "data", "ToMATo_code.tar.gz")
    mkpath(dirname(archive))
    isfile(archive) || Downloads.download(CONFIG["archive_url"], archive)
    bytes2hex(sha256(read(archive))) == CONFIG["archive_sha256"] ||
        error("Archive checksum mismatch; refusing unverified input")
    member = CONFIG["archive_member"]
    sourcebytes = read(`tar -xOzf $archive $member`)
    bytes2hex(sha256(sourcebytes)) == CONFIG["dataset_sha256"] ||
        error("Dataset checksum mismatch")
    raw = readdlm(IOBuffer(sourcebytes), Float64)
    @test size(raw) == (CONFIG["expected_points"], 3)
    @test all(isfinite, raw)
    @test all(>(0), raw[:, 3])
    return permutedims(raw[:, 1:2]), raw[:, 3]
end

function draw_cloud(coords, density, labels_by_radius)
    fig = Figure(size=(1440, 500), fontsize=16)
    ax = Axis(fig[1, 1], title="Released density", aspect=DataAspect(),
              xlabel="x (source units)", ylabel="y (source units)")
    sc = scatter!(ax, coords[1, :], coords[2, :]; color=density,
                  colormap=:viridis, markersize=1.1)
    Colorbar(fig[2, 1], sc; vertical=false, label="Precomputed density")
    colors = [:grey70, :dodgerblue3, :darkorange2]
    for (col, radius) in enumerate(CONFIG["radii"])
        labels = labels_by_radius[radius]
        ax = Axis(fig[1, col+1], title="Radius $(Int(radius)); τ = 0.001",
                  aspect=DataAspect(), xlabel="x (source units)")
        # Background first, then both retained spirals. All points are plotted.
        for label in 0:2
            ids = findall(==(label), labels)
            scatter!(ax, coords[1, ids], coords[2, ids];
                     color=colors[label+1], markersize=1.1)
        end
        hidedecorations!(ax; label=false)
    end
    Label(fig[2, 2:3], "Blue/orange: retained clusters. Gray: low-peak filtering (label 0).")
    save(joinpath(OUT, "spirals.png"), fig; px_per_unit=2)
end

function draw_diagrams(pairs_by_radius)
    fig = Figure(size=(1280, 680), fontsize=14, figure_padding=30)
    ticks = ([0.0, 0.0004, 0.0008, 0.0012, 0.0016],
             ["0", "0.0004", "0.0008", "0.0012", "0.0016"])
    for (col, radius) in enumerate(CONFIG["radii"])
        pairs = pairs_by_radius[radius]
        finite = [p for p in values(pairs) if isfinite(p[2])]
        essential = [p[1] for p in values(pairs) if !isfinite(p[2])]
        ax = Axis(fig[1, col], title="Radius $(Int(radius)): library output",
                  xlabel="Birth density", ylabel="Lifetime (birth − saddle)", xticks=ticks)
        scatter!(ax, first.(finite), [p[1]-p[2] for p in finite];
                 color=:dodgerblue3, markersize=5)
        hlines!(ax, [CONFIG["tau"]]; color=:darkorange2, linestyle=:dash)
        limits!(ax, 0, 0.0016, 0, 0.0013)
        records = length(essential) == 1 ? "record" : "records"
        ax2 = Axis(fig[2, col], title="$(length(essential)) essential $records; no finite death",
                   xlabel="Birth density", ylabel="", xticks=ticks)
        scatter!(ax2, essential, zeros(length(essential));
                 color=[b < CONFIG["height_cutoff"] ? :grey65 : :dodgerblue3 for b in essential],
                 marker=:utriangle, markersize=8)
        vlines!(ax2, [CONFIG["height_cutoff"]]; color=:darkorange2, linestyle=:dash)
        xlims!(ax2, 0, 0.0016)
        hideydecorations!(ax2)
    end
    rowsize!(fig.layout, 2, Relative(0.3))
    save(joinpath(OUT, "persistence.png"), fig; px_per_unit=2)
end

function main()
    coords, density = load_original()
    X = EuclideanSpace(coords)
    n = length(X)
    summary_rows, sweep_rows, audit_rows = [], [], []
    labels_by_radius = Dict{Float64,Vector{Int}}()
    pairs_by_radius = Dict{Float64,Any}()
    runtime_rows = []
    for (ri, radius) in enumerate(CONFIG["radii"])
        graph_seconds = @elapsed g = proximity_graph(X, radius;
            min_k_ball=0, max_k_ball=n, k_nn=0)
        components = length(connected_components(g))
        @test nv(g) == n
        @test ne(g) == CONFIG["expected_edges"][ri]
        @test components == CONFIG["expected_components"][ri]
        @test edge_radius_check(g, coords, radius)

        diagram_seconds = @elapsed _, pairs = tomato(X, g, density, Inf)
        _, reference_pairs = reference_tomato(g, density, Inf)
        _, standard_pairs = reference_tomato(g, density, Inf; strict_ties=false)
        @test Set(keys(pairs)) == Set(keys(reference_pairs))
        lifetimes = finite_lifetimes(pairs)
        reference_lifetimes = finite_lifetimes(standard_pairs)
        @test lifetimes[1:2] == reference_lifetimes[1:2]
        @test lifetimes[2] < CONFIG["tau"] < lifetimes[1]
        essential = count(p -> !isfinite(p[2]), values(pairs))
        @test count(p -> !isfinite(p[2]), values(standard_pairs)) == components
        mismatched_deaths = 0
        for peak in sort(collect(keys(pairs)))
            b, d = pairs[peak]
            ref_d = reference_pairs[peak][2]
            disagree = d != ref_d
            mismatched_deaths += disagree
            push!(audit_rows, (radius, peak, b, d, ref_d, disagree))
        end
        csvfile(joinpath(OUT, "peaks_radius$(Int(radius)).csv"),
                ["peak_row", "birth", "library_death", "lifetime"],
                [(i, pairs[i][1], pairs[i][2],
                  isfinite(pairs[i][2]) ? pairs[i][1]-pairs[i][2] : Inf)
                 for i in sort(collect(keys(pairs)))])

        selected_seconds = @elapsed labels, _ = tomato(X, g, density, CONFIG["tau"];
            max_cluster_height=CONFIG["height_cutoff"])
        @test length(labels) == n
        @test Set(unique(labels)) ⊆ Set(0:2)
        @test [count(==(k), labels) for k in 1:2] == CONFIG["expected_cluster_sizes"][ri]
        filtered = count(==(0), labels)
        @test filtered == CONFIG["expected_filtered"][ri]
        corrected, _ = reference_tomato(g, density, CONFIG["tau"];
            height_cutoff=CONFIG["height_cutoff"])
        standard, _ = reference_tomato(g, density, CONFIG["tau"];
            height_cutoff=CONFIG["height_cutoff"], strict_ties=false)
        disagreement = count(labels .!= corrected)
        standard_disagreement = count(labels .!= standard)
        csvfile(joinpath(OUT, "labels_radius$(Int(radius)).csv"),
                ["source_row", "library_label", "reference_same_ties", "reference_total_order"],
                [(i, labels[i], corrected[i], standard[i]) for i in 1:n])
        @test length(filter(>(0), unique(corrected))) == 2
        @test length(filter(>(0), unique(standard))) == 2
        if radius == 25
            @test labels == corrected == standard
        end

        push!(summary_rows, (radius, n, ne(g), components, length(pairs), essential,
              count(==(1), labels), count(==(2), labels), filtered,
              lifetimes[1], lifetimes[2], mismatched_deaths, disagreement, standard_disagreement))
        push!(runtime_rows, (radius, graph_seconds, diagram_seconds, selected_seconds))
        for tau in CONFIG["tau_sweep"]
            labs, _ = tau == CONFIG["tau"] ? (labels, nothing) :
                tomato(X, g, density, tau; max_cluster_height=tau)
            push!(sweep_rows, (radius, tau, length(filter(>(0), unique(labs))),
                               count(==(0), labs)))
        end
        labels_by_radius[radius] = labels
        pairs_by_radius[radius] = pairs
        println("radius=$radius edges=$(ne(g)) components=$components clusters=2 " *
                "sizes=$([count(==(k),labels) for k in 1:2]) filtered=$filtered " *
                "death_disagreements=$mismatched_deaths label_disagreements=$disagreement")
        flush(stdout)
    end
    csvfile(joinpath(OUT, "summary.csv"),
            ["radius", "points", "edges", "graph_components", "library_peaks", "library_essential",
             "cluster_1", "cluster_2", "filtered", "largest_finite_lifetime", "second_finite_lifetime",
             "death_audit_mismatches", "label_audit_mismatches", "label_total_order_mismatches"], summary_rows)
    csvfile(joinpath(OUT, "threshold_sweep.csv"), ["radius", "tau", "positive_clusters", "filtered"], sweep_rows)
    csvfile(joinpath(OUT, "death_audit.csv"),
            ["radius", "peak_row", "birth", "library_death", "reference_same_ties_death", "disagree"], audit_rows)
    csvfile(joinpath(OUT, "runtime.csv"), ["radius", "graph_seconds", "diagram_seconds", "selected_clustering_seconds"], runtime_rows)
    draw_cloud(coords, density, labels_by_radius)
    draw_diagrams(pairs_by_radius)
    repo = normpath(joinpath(ROOT, "../../../../"))
    files = ["ToMATo.jl/src/tomato algorithm.jl", "ToMATo.jl/src/graph.jl",
             "ToMATo.jl/src/density.jl", "MetricSpaces.jl/src/extra/filters.jl"]
    provenance = Dict(
        "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "source_access_date" => CONFIG["source_access_date"],
        "source_hashes" => Dict(f => bytes2hex(sha256(read(joinpath(repo, f)))) for f in files),
        "environment_manifest_sha256" => bytes2hex(sha256(read(joinpath(ROOT, "environment/Manifest.toml")))),
        "dataset_sha256" => CONFIG["dataset_sha256"],
        "archive_sha256" => CONFIG["archive_sha256"],
        "density_range" => collect(extrema(density)),
        "coordinate_min" => vec(minimum(coords; dims=2)),
        "coordinate_max" => vec(maximum(coords; dims=2)),
        "distinct_density_values" => length(unique(density)),
        "validation" => "Data/graph/selected cluster checks passed. Audit disagreements are recorded, not hidden.")
    open(joinpath(OUT, "provenance.toml"), "w") do io
        TOML.print(io, provenance; sorted=true)
    end
    println("Completed original released benchmark and independent audit; results in $OUT")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
