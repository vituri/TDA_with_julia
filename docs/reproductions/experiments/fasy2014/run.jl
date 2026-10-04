using TDARipserer, TDAPersistenceDiagrams, PersistenceInference, CairoMakie
using Random, Statistics, DelimitedFiles, SHA, TOML, Test, LinearAlgebra

const ROOT = @__DIR__
const OUT = joinpath(ROOT, "results")
const CFG = TOML.parsefile(joinpath(ROOT, "config.toml"))
mkpath(OUT)
BLAS.set_num_threads(2)

function write_csv(name, rows)
    open(joinpath(OUT, name), "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
end

circle(n, seed) = [(cos(t), sin(t)) for t in 2pi .* rand(MersenneTwister(seed), n)]
paper_quantile(x, alpha) = sort(x)[ceil(Int, (1-alpha)*length(x))]

function grid_kernel(points, grid; h=CFG["bandwidth"])
    # Each column is one i.i.d. observation's normalized Gaussian contribution.
    locations = [(x,y) for x in grid, y in grid]
    matrix = [exp(-((x-u)^2+(y-v)^2)/(2h^2))/(2pi*h^2)
              for (x,y) in vec(locations), (u,v) in points]
    return matrix
end

function kde_bootstrap(matrix, B, seed)
    n = size(matrix, 2)
    estimate = vec(mean(matrix; dims=2))
    rng = MersenneTwister(seed)
    counts = zeros(n)
    resampled = similar(estimate)
    deviations = zeros(B)
    for b in 1:B
        fill!(counts, 0)
        for i in rand(rng, 1:n, n)
            counts[i] += 1/n
        end
        mul!(resampled, matrix, counts)
        deviations[b] = maximum(abs, resampled .- estimate)
    end
    return estimate, deviations
end

function triangulated_filtration(values)
    # Fixed PL triangulation: split every square along the same diagonal.
    m = size(values, 1)
    index(i,j) = i + (j-1)*m
    simplices = Pair{Tuple,Float64}[(index(i,j),) => -values[i,j]
                                   for i in 1:m for j in 1:m]
    for i in 1:m-1, j in 1:m-1
        for vertices in ((index(i,j), index(i+1,j), index(i+1,j+1)),
                         (index(i,j), index(i,j+1), index(i+1,j+1)))
            # Explicit edges are required: Custom's automatic missing-face
            # insertion uses the coface time, which is not a vertex lower star.
            for (a,b) in ((1,2),(1,3),(2,3))
                edge=(vertices[a],vertices[b])
                push!(simplices, edge => maximum(-values[v] for v in edge))
            end
            push!(simplices, vertices => maximum(-values[v] for v in vertices))
        end
    end
    return Custom(simplices)
end

triangulated_diagrams(values) = ripserer(triangulated_filtration(values);
    dim_max=1, modulus=CFG["coefficient_field"])

function export_diagrams(name, diagrams)
    write_csv(name, [(; dimension=k-1, birth=birth(bar), death=death(bar),
        persistence=persistence(bar)) for (k,d) in enumerate(diagrams) for bar in d])
end

function density_detection(diagrams, c)
    # H0 uses the nonnegative-density baseline zero for the paper's finite plot.
    # The native essential interval remains infinite in all metric calculations.
    component_height = -birth(only(filter(bar -> !isfinite(death(bar)), diagrams[1])))
    return (; component_height, significant_component=component_height > 2c,
        significant_finite_H0=count(bar -> isfinite(death(bar)) && persistence(bar)>2c, diagrams[1]),
        significant_H1=count(bar -> persistence(bar)>2c, diagrams[2]),
        longest_H1=isempty(diagrams[2]) ? 0.0 : maximum(persistence, diagrams[2]))
end

function subsample_radii(points, sizes, B, seed)
    n = length(points)
    distances = [hypot(a[1]-b[1], a[2]-b[2]) for a in points, b in points]
    rows = NamedTuple[]
    draws = NamedTuple[]
    for b in sizes
        rng = MersenneTwister(seed+b)
        hausdorff = zeros(B)
        for j in 1:B
            selected = randperm(rng, n)[1:b] # WITHOUT replacement.
            hausdorff[j] = maximum(minimum(view(distances, i, selected)) for i in 1:n)
            push!(draws, (;subsample_size=b, replicate=j, hausdorff=hausdorff[j]))
        end
        q = paper_quantile(hausdorff, CFG["alpha"])
        push!(rows, (;subsample_size=b, quantile=q, radius=2q))
    end
    write_csv("subsample_draws.csv", draws)
    return rows
end

function population_values(grid)
    # Deterministic angular quadrature for E[K_h(x-X)], X uniform on S^1.
    m = CFG["population_quadrature_points"]
    points = [(cos(2pi*j/m), sin(2pi*j/m)) for j in 0:m-1]
    return reshape(vec(mean(grid_kernel(points, grid); dims=2)), length(grid), length(grid))
end

function main()
    n, h, alpha = CFG["n"], CFG["bandwidth"], CFG["alpha"]
    primary_b = floor(Int, n^CFG["subsample_exponent"])
    points = circle(n, CFG["sample_seed"])
    write_csv("points.csv", [(;id=i, x=p[1], y=p[2]) for (i,p) in enumerate(points)])
    grid = collect(range(CFG["grid_lower"], CFG["grid_upper"]; length=CFG["grid_size"]))
    matrix = grid_kernel(points, grid)
    estimate, maxima = kde_bootstrap(matrix, CFG["bootstrap_replicates"], CFG["bootstrap_seed"])
    c_boot = paper_quantile(maxima, alpha)
    # Solve Lemma 11 equation (34), using the actual D-dimensional K(0).
    c_finite = (1/(2pi*h^2))*sqrt(log(2length(grid)^2/alpha)/(2n))
    values = reshape(estimate, length(grid), length(grid))
    writedlm(joinpath(OUT, "kde.csv"), values, ',')
    population = population_values(grid)
    writedlm(joinpath(OUT, "population_kde.csv"), population, ',')
    density_diagrams = triangulated_diagrams(values)
    population_diagrams = triangulated_diagrams(population)
    export_diagrams("density_diagrams.csv", density_diagrams)
    export_diagrams("population_diagrams.csv", population_diagrams)
    write_csv("bootstrap_draws.csv", [(;replicate=j, sup_norm=z, scaled_statistic=sqrt(n*h^2)*z)
        for (j,z) in enumerate(maxima)])
    detections = [(;method, radius=c, density_detection(density_diagrams,c)...)
        for (method,c) in (("finite_grid_Hoeffding",c_finite),("KDE_bootstrap",c_boot))]
    write_csv("density_detection.csv", detections)

    # Native confidence_band resamples i.i.d. kernel contribution functions.
    # Validate its bootstrap distribution against a separate counts implementation.
    smallgrid = collect(range(-2,2; length=21))
    smallmatrix = grid_kernel(points, smallgrid)
    native = confidence_band(permutedims(smallmatrix); method=:bootstrap,
        studentize=false, n_boot=100, alpha, rng=MersenneTwister(CFG["bootstrap_seed"]))
    _, independent = kde_bootstrap(smallmatrix, 100, CFG["bootstrap_seed"])
    write_csv("native_band_audit.csv", [(;replicate=j,
        native_sup_norm=native.maxima[j]/sqrt(n), counts_sup_norm=independent[j],
        absolute_difference=abs(native.maxima[j]/sqrt(n)-independent[j])) for j in 1:100])

    # Native Rips uses edge length. Divide endpoints by two for the ball-radius
    # convention suggested by Figure 6; preserve both in the export.
    rips = ripserer(points; dim_max=1, modulus=2)
    export_diagrams("rips_edge_length_diagrams.csv", rips)
    scaled = [PersistenceDiagram([(birth(bar)/2, death(bar)/2) for bar in d]; dim=k-1)
              for (k,d) in enumerate(rips)]
    export_diagrams("rips_radius_diagrams.csv", scaled)
    subsampling = subsample_radii(points, CFG["subsample_sizes"], CFG["subsample_replicates"], CFG["subsample_seed"])
    subrows = [(;row..., significant_H0=count(bar -> persistence(bar)>2row.radius, scaled[1]),
        significant_H1=count(bar -> persistence(bar)>2row.radius, scaled[2]),
        longest_H1=maximum(persistence, scaled[2])) for row in subsampling]
    write_csv("subsampling.csv", subrows)

    # Distribution-known diagnostic: exact Hausdorff error of S_n vs unit circle.
    angles = sort([mod(atan(y,x), 2pi) for (x,y) in points])
    largest_gap = maximum(diff(vcat(angles, first(angles)+2pi)))
    actual_hausdorff = 2sin(largest_gap/4)
    errors = [(;dimension=k-1, bottleneck=Bottleneck()(density_diagrams[k], population_diagrams[k]),
        actual_sup_norm=maximum(abs, values.-population), bootstrap_radius=c_boot,
        finite_radius=c_finite) for k in 1:2]
    write_csv("population_error.csv", errors)

    sensitivity = NamedTuple[]
    for seed in CFG["sensitivity_seeds"], m in CFG["sensitivity_grid_sizes"]
        g = collect(range(-2,2; length=m))
        mat = grid_kernel(circle(n,seed),g)
        field, deviations = kde_bootstrap(mat, CFG["bootstrap_replicates"], CFG["bootstrap_seed"]+seed)
        ds = triangulated_diagrams(reshape(field,m,m))
        cb = paper_quantile(deviations,alpha)
        cf = 1/(2pi*h^2)*sqrt(log(2m^2/alpha)/(2n))
        for (method,c) in (("KDE_bootstrap",cb),("finite_grid_Hoeffding",cf))
            push!(sensitivity,(;seed, grid_size=m, method, radius=c, density_detection(ds,c)...))
        end
    end
    write_csv("sensitivity.csv",sensitivity)

    checks = @testset "Fasy finite-grid bootstrap and subsampling" begin
        @test all(isapprox(hypot(x,y),1;atol=1e-14) for (x,y) in points)
        @test length(points)==500
        @test primary_b in CFG["subsample_sizes"]
        @test all(isapprox(native.maxima[j]/sqrt(n),independent[j];atol=1e-14) for j in 1:100)
        @test native.estimate ≈ vec(mean(smallmatrix;dims=2))
        @test maximum(abs,native.upper.-native.estimate) ≈ native.critical_value/sqrt(n)
        @test c_boot == sort(maxima)[950]
        @test all(row.radius==2row.quantile for row in subsampling)
        @test actual_hausdorff>0 && actual_hausdorff<first(subsampling).radius
        @test population[(length(grid)+1)÷2,(length(grid)+1)÷2] ≈ exp(-1/(2h^2))/(2pi*h^2)
        @test length(filter(bar->!isfinite(death(bar)),density_diagrams[1]))==1
        @test all(isfinite(death(bar)) for bar in density_diagrams[2])
        @test all(row.bottleneck<=row.actual_sup_norm+1e-12 for row in errors)
        @test length(sensitivity)==30
        filtration=triangulated_filtration(values)
        @test all(birth(simplex)==maximum(-values[v] for v in vertices(simplex))
                  for dim in 0:2 for simplex in filtration[dim])
        @test length(filtration[1])==2length(grid)*(length(grid)-1)+(length(grid)-1)^2
        @test length(filtration[2])==2(length(grid)-1)^2
        # Translation sanity check for the independently constructed PL domain.
        shifted=triangulated_diagrams(values .+ 0.02)
        @test Bottleneck()(density_diagrams[1],shifted[1]) ≈ 0.02
    end

    metadata=Dict("target"=>CFG["target"], "julia_version"=>string(VERSION),
        "sample_seed"=>CFG["sample_seed"], "bootstrap_radius"=>c_boot,
        "primary_subsample_size"=>primary_b,
        "finite_grid_radius"=>c_finite, "actual_support_hausdorff"=>actual_hausdorff,
        "native_band_max_absolute_difference"=>maximum(abs.(native.maxima./sqrt(n).-independent)),
        "bootstrap_H1_sensitivity_successes"=>count(r->r.method=="KDE_bootstrap" && r.significant_H1==1,sensitivity),
        "bootstrap_sensitivity_comparisons"=>15, "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "config_sha256"=>bytes2hex(sha256(read(joinpath(ROOT,"config.toml")))))
    open(joinpath(OUT,"summary.toml"),"w") do io; TOML.print(io,metadata;sorted=true); end

    CairoMakie.activate!()
    fig=Figure(size=(1160,920))
    ax=Axis(fig[1,1];title="Uniform unit circle: n = 500",xlabel="x",ylabel="y",aspect=DataAspect())
    scatter!(ax,first.(points),last.(points);markersize=4,color="#245f75")
    ax=Axis(fig[1,2];title="Rips persistence: ball-radius scale",xlabel="Birth",ylabel="Death",aspect=DataAspect())
    lines!(ax,[0,1],[0,1];color=:black)
    chosen=only(filter(r->r.subsample_size==primary_b,subrows))
    lines!(ax,[0,1],[2chosen.radius,1+2chosen.radius];color="#c05546",linestyle=:dash,label="Subsampling: 2c")
    for (k,d) in enumerate(scaled)
        bars=filter(bar->isfinite(death(bar)),d)
        scatter!(ax,birth.(bars),death.(bars);color=k==1 ? :black : "#c05546",marker=k==1 ? :circle : :utriangle,markersize=k==1 ? 6 : 14,label="H$(k-1)")
    end
    axislegend(ax;position=:rb)
    limits!(ax,0,1,0,1.05)
    ax=Axis(fig[2,1];title="Gaussian KDE: h = 0.3",xlabel="x",ylabel="y",aspect=DataAspect())
    hm=heatmap!(ax,grid,grid,values;colormap=:viridis)
    Colorbar(fig[2,1][1,2],hm;label="Density")
    ax=Axis(fig[2,2];title="Upper-level density persistence",xlabel="Death density",ylabel="Birth density",aspect=DataAspect())
    lines!(ax,[0,.3],[0,.3];color=:black)
    for (name,c,color) in (("Finite bound",c_finite,"#506fb6"),("Bootstrap",c_boot,"#c05546"))
        lines!(ax,[0,.3],[2c,.3+2c];color,linestyle=:dash,label=name)
    end
    for (k,d) in enumerate(density_diagrams)
        xs=[isfinite(death(bar)) ? -death(bar) : 0.0 for bar in d]
        ys=[-birth(bar) for bar in d]
        scatter!(ax,xs,ys;color=k==1 ? :black : "#c05546",marker=k==1 ? :circle : :utriangle,markersize=11,label="H$(k-1)")
    end
    limits!(ax,0,.3,0,.48)
    axislegend(ax;position=:rb)
    save(joinpath(OUT,"circle_confidence.png"),fig;px_per_unit=1.25)
    save(joinpath(OUT,"circle_confidence.svg"),fig)
    fig2=Figure(size=(960,440))
    ax=Axis(fig2[1,1];title="Subsampling choice is visible in the decision",xlabel="Subsample size b",ylabel="Persistence / exclusion threshold")
    scatterlines!(ax,[r.subsample_size for r in subrows],[2r.radius for r in subrows];marker=:circle,color="#c05546",label="Exclusion threshold 2c")
    hlines!(ax,[maximum(persistence,scaled[2])];color="#245f75",label="Longest H1 lifetime")
    axislegend(ax)
    save(joinpath(OUT,"subsample_sensitivity.png"),fig2;px_per_unit=1.3)
    println("KDE detections: ",detections)
    println("Subsampling: ",subrows)
    println("Saved executed results to ",OUT)
end

main()
