using MLDatasets
using Images, Makie, CairoMakie
using Distances
using Ripserer, PersistenceDiagrams#, Plots
using StatsBase: mean

train_x, train_y = MNIST(split=:train)[:]
train_x

n = 15
mosaicview(train_x[:, :, 1:(n^2)] .|> Gray, nrow = n)

"""
For each digit, return a list of coordinates in R^2 where the pixels are white
"""
function img_to_points(img, threshold = 0.5)
    ids = findall(x -> x >= threshold, img)
    pts = getindex.(ids, [1 2])
    pts
end

n = 1000
ms = [train_x[:, :, i] for i in 1:n]
m = ms[1]

img = m .|> Gray

pts = img_to_points.(ms, 0.5)
pts_vec = [x for x in eachrow(pts)]
pt = pts[1]




pt_vec = [x for x ∈ eachrow(pt)]
pd = ripserer(pt_vec, dim_max = 1, verbose = true, cutoff = 1.5, sparse = true)
pd[1]
pd[2]
barcode(pd)

PersistenceImage(pd; sigma = 0.1)


pd = ripserer(Cubical(-m))
pd |> barcode

ms = [train_x[:, :, i] for i in 1:1000]
diagrams = @showprogress [ripserer(Cubical(m)) for m in ms]
diagrams[1] |> barcode

PersistenceImage(diagrams[1][1])


dist_function = Cityblock()
dist_function = Euclidean()
dists = pairwise(dist_function, pts[1]')
excentricity = [mean(c) for c ∈ eachcol(dists)]
plot_digit(pt, excentricity)

exp_density(x, h = 5) = @. exp(-((x/h)^2))
dens = [exp_density(c, 5) |> mean for c ∈ eachcol(dists)]
plot_digit(pt, dens)

function plot_digit(pt, values = :black)
    f = Figure();
    ax = Makie.Axis(f[1, 1], autolimitaspect = 1)
    scatter!(ax, pt; markersize = 40, marker = :rect, color = values)
    if values isa Vector{<:Real}
        Colorbar(f[1, 2])
    end
    f
end

plot_digit(pt, excentricity)

# pega imagens

# calcula excentricidade, densidade

#



# Now we will use a random forest classifier to predict each class. The training process consists of two lines

# ```{julia}
# using DecisionTree
# model = RandomForestClassifier(n_subfeatures = 50, n_trees = 50, max_depth = 10)
# fit!(model, X, y)
# ```

# We can get the predicted labels 

# ```{julia}
# pred_y = predict(model, X)
# ```

# and calculate the accuracy:

# ```{julia}
# accuracy = sum(pred_y .== y) / length(y)
# accuracy = round(accuracy * 100, digits = 2)
# println("The accuracy was $accuracy %!")
# ```

# We got a `julia accuracy`% accuracy! That was pretty good, taking into account that we only used the excentricity sublevel filtration. 