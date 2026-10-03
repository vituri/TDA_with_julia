# Independent union-find implementation of Algorithm 1, used only for auditing
# the local library output. Final chapter cluster assignments come from ToMATo.
# Ties can follow the library's strict comparison or a deterministic total order.
function reference_tomato(g, ds, tau; height_cutoff=0.0, strict_ties=true)
    n = length(ds)
    parent = collect(1:n)
    processed = falses(n)
    pairs = Dict{Int,Vector{Float64}}()
    function root(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end
    for i in sortperm(ds; rev=true)
        ns = [j for j in neighbors(g, i)
              if processed[j] && (!strict_ties || ds[j] > ds[i])]
        if isempty(ns)
            pairs[i] = [ds[i], Inf]
        else
            gradient = ns[argmax(ds[ns])]
            parent[i] = root(gradient)
            for j in ns
                a, b = root(i), root(j)
                a == b && continue
                if min(ds[a], ds[b]) < ds[i] + tau
                    low, high = ds[a] < ds[b] ? (a, b) : (b, a)
                    pairs[low][2] = ds[i]
                    parent[low] = high
                end
            end
        end
        processed[i] = true
    end
    roots = root.(1:n)
    surviving = sort(unique(roots); by=i -> ds[i], rev=true)
    labelmap = Dict(c => (ds[c] < height_cutoff ? 0 : k)
                    for (k, c) in enumerate(surviving))
    return [labelmap[c] for c in roots], pairs
end

finite_lifetimes(pairs) = sort([p[1] - p[2] for p in values(pairs)
                               if isfinite(p[2])]; rev=true)

function edge_radius_check(g, coords, radius)
    # Check every edge, with only an allowance for floating-point roundoff.
    r2 = radius^2
    for edge in edges(g)
        i, j = src(edge), dst(edge)
        d2 = sum(abs2, coords[:, i] - coords[:, j])
        d2 <= r2 + 16eps(r2) || return false
    end
    return true
end
