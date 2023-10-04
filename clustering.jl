using ToMATo
import GeometricDatasets as gd
using AlgebraOfGraphics

# usar distance to measure aqui?

X = hcat(randn(2, 800), randn(2, 800) .* 0.8 .+ 4, randn(2, 100) .* 0.3 .- 2)
k = x -> exp(-(x / 0.8)^2)
ds = gd.density_estimation(X, h = 0.5)
# ds = gd.distance_to_measure(X, k = 1500)
# ds .= maximum(ds) .- ds

df = (x1 = X[1, :], x2 = X[2, :], ds = ds)
plt = data(df) * mapping(:x1, :x2)
draw(plt)

df = (x1 = X[1, :], x2 = X[2, :], ds = ds)
plt = data(df) * mapping(:x1, :x2, color = :ds)
draw(plt)

g = proximity_graph(X, 0.2, max_k_ball = 6, k_nn = 4, min_k_ball = 2)

fig, ax, plt = graph_plot(X, g, ds)
fig

X2 = vcat(X, ds')
fig, ax, plt = graph_plot(X2, g, ds)
fig

clusters, births_and_deaths = tomato(X, g, ds, 0)
plot_births_and_deaths(births_and_deaths)

fig, ax, plt = graph_plot(X, g, clusters .|> string)
fig

fig, ax, plt = graph_plot(X2, g, clusters .|> string)
fig

τ = 0.02
clusters, _ = tomato(X, g, ds, τ, max_cluster_height = τ)

df = (x1 = X[1, :], x2 = X[2, :], ds = clusters .|> string)
plt = data(df) * mapping(:x1, :x2, color = :ds)
draw(plt)

fig, ax, plt = graph_plot(X, g, clusters .|> string)
fig

fig, ax, plt = graph_plot(X2, g, clusters .|> string)
fig







# datasets

using DelimitedFiles

X = readdlm("clustering datasets/spiral with density.txt") |> 
    transpose |> 
    collect