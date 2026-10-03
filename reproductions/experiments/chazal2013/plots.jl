# Redraw figures from saved outputs without recomputing any clustering.
# Main guard in run.jl leaves its helper functions available to this script.
include("run.jl")
coords, density = load_original()
labels_by_radius = Dict{Float64,Vector{Int}}()
pairs_by_radius = Dict{Float64,Any}()
for radius in CONFIG["radii"]
    r = Int(radius)
    labels = readdlm(joinpath(OUT, "labels_radius$r.csv"), ',', Int; skipstart=1)
    peaks = readdlm(joinpath(OUT, "peaks_radius$r.csv"), ',', Float64; skipstart=1)
    labels_by_radius[radius] = labels[:, 2]
    pairs_by_radius[radius] = Dict(Int(row[1]) => collect(row[2:3]) for row in eachrow(peaks))
end
draw_cloud(coords, density, labels_by_radius)
draw_diagrams(pairs_by_radius)
println("Figures redrawn from saved outputs.")
