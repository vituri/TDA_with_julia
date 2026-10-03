# Validate the saved experiment without network access or rebuilding the graphs.
using DelimitedFiles, Graphs, MetricSpaces, ToMATo, TOML, Test, SHA
include("audit.jl")
const CFG = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
const OUT = joinpath(@__DIR__, "results")

@testset "Original released benchmark" begin
    archive = joinpath(@__DIR__, "data/ToMATo_code.tar.gz")
    @test bytes2hex(sha256(read(archive))) == CFG["archive_sha256"]
    summary = readdlm(joinpath(OUT, "summary.csv"), ',', Float64; skipstart=1)
    @test size(summary) == (2, 14)
    for i in 1:2
        radius = Int(CFG["radii"][i])
        @test summary[i, 2] == CFG["expected_points"]
        @test summary[i, 3] == CFG["expected_edges"][i]
        @test summary[i, 4] == CFG["expected_components"][i]
        @test summary[i, 7:8] == CFG["expected_cluster_sizes"][i]
        @test summary[i, 9] == CFG["expected_filtered"][i]
        @test summary[i, 11] < CFG["tau"] < summary[i, 10]
        labels = Int.(readdlm(joinpath(OUT, "labels_radius$(radius).csv"), ','; skipstart=1))
        @test labels[:, 1] == 1:CFG["expected_points"]
        @test Set(labels[:, 2]) ⊆ Set(0:2)
        @test count(labels[:, 2] .!= labels[:, 3]) == Int(summary[i, 13])
        @test count(labels[:, 2] .!= labels[:, 4]) == Int(summary[i, 14])
        if radius == 25
            @test labels[:, 2] == labels[:, 3] == labels[:, 4]
        end
    end
    sweep = readdlm(joinpath(OUT, "threshold_sweep.csv"), ',', Float64; skipstart=1)
    for radius in CFG["radii"]
        rows = sweep[sweep[:, 1] .== radius, :]
        @test all(diff(rows[:, 3]) .<= 0)
        @test rows[rows[:, 2] .== CFG["tau"], 3] == [2]
    end
    for file in ["spirals.png", "persistence.png"]
        @test isfile(joinpath(OUT, file))
        @test filesize(joinpath(OUT, file)) > 10000
    end
end

@testset "Independent audit on a three-peak saddle" begin
    # At height 6, three components meet. Both lower peaks must die at 6.
    g = SimpleGraph(7)
    for (i, j) in [(3,4), (2,5), (1,6), (4,7), (5,7), (6,7)]
        add_edge!(g, i, j)
    end
    density = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    X = EuclideanSpace(reshape(collect(1.0:7.0), 1, :))
    labels, pairs = reference_tomato(g, density, Inf)
    @test length(unique(labels)) == 1
    @test pairs[1] == [12.0, Inf]
    @test pairs[2] == [11.0, 6.0]
    @test pairs[3] == [10.0, 6.0]
    library_labels, library_pairs = tomato(X, g, density, Inf)
    println("Three-peak diagnostic: library components=", length(unique(library_labels)),
            "; exact components=1. This is an audit, not a hidden library patch.")
end
