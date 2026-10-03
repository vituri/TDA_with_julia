# From the JuliaTDA workspace:
# julia --project=TDA_with_julia/reproductions/experiments/stability2007 --startup-file=no TDA_with_julia/reproductions/experiments/stability2007/run.jl
using TDARipserer, TDAPersistenceDiagrams, CairoMakie
using Random, Statistics, DelimitedFiles, SHA, TOML, Test

const ROOT = @__DIR__
const OUT = joinpath(ROOT,"results")
mkpath(OUT)
config = TOML.parsefile(joinpath(ROOT,"config.toml"))

function write_csv(path,rows)
    open(path,"w") do io
        writedlm(io,permutedims(collect(keys(first(rows)))),',')
        for row in rows
            writedlm(io,permutedims(collect(values(row))),',')
        end
    end
end
function export_diagrams(name,diagrams)
    rows = [(; homology_dimension=d-1,birth=birth(bar),death=death(bar),
              persistence=persistence(bar)) for (d,diagram) in enumerate(diagrams) for bar in diagram]
    write_csv(joinpath(OUT,name*"_diagrams.csv"),rows)
end
function field_diagrams(values)
    # Full filtration: each call includes all cell births, with no persistence cutoff.
    ripserer(Cubical(values);dim_max=1,modulus=config["coefficient_field"])
end
function finite_points(diagram)
    [Point2f(birth(bar),death(bar)) for bar in diagram if isfinite(death(bar))]
end

function main()
grid=collect(range(config["grid_lower"],config["grid_upper"];length=config["grid_size"]))
base=[(hypot(x,y)-1)^2+0.12*cos(3x)*cos(2y) for x in grid,y in grid]
writedlm(joinpath(OUT,"base_field.csv"),base,',')
original=field_diagrams(base)
export_diagrams("base",original)
rows=NamedTuple[]
illustration=nothing
illustration_diagrams=nothing
for seed in config["seeds"]
    rng=MersenneTwister(seed)
    noise=2rand(rng,size(base)...).-1
    noise./=maximum(abs,noise) # Sup norm is exactly one, up to arithmetic precision.
    writedlm(joinpath(OUT,"noise_seed_$(seed).csv"),noise,',')
    for epsilon in config["noise_amplitudes"]
        perturbed=base.+epsilon.*noise
        diagrams=field_diagrams(perturbed)
        bound=maximum(abs,perturbed.-base)
        for dimension in 0:1
            distance=Bottleneck()(original[dimension+1],diagrams[dimension+1])
            push!(rows,(;seed,epsilon,dimension,sup_norm=bound,bottleneck=distance,
                        slack=bound-distance,
                        original_intervals=length(original[dimension+1]),
                        perturbed_intervals=length(diagrams[dimension+1]),
                        bound_holds=distance<=bound+config["absolute_tolerance"]))
        end
        export_diagrams("seed_$(seed)_epsilon_$(replace(string(epsilon),'.'=>'p'))",diagrams)
        if seed==first(config["seeds"]) && epsilon==0.1
            illustration=perturbed
            illustration_diagrams=diagrams
            writedlm(joinpath(OUT,"illustration_field.csv"),perturbed,',')
        end
    end
end
write_csv(joinpath(OUT,"perturbations.csv"),rows)

# An exact sanity case: f -> f+c translates every birth/death by c.
c=config["translation"]
translated=field_diagrams(base.+c)
translation_rows=NamedTuple[]
for dimension in 0:1
    distance=Bottleneck()(original[dimension+1],translated[dimension+1])
    push!(translation_rows,(;dimension,translation=c,bottleneck=distance,sup_norm=c))
end
write_csv(joinpath(OUT,"translation.csv"),translation_rows)
export_diagrams("translation",translated)

# Essential H0 interval has an analytically known birth at the minimum field value.
essential=filter(bar->!isfinite(death(bar)),original[1])
checks=@testset "stability on a fixed cubical domain" begin
    @test length(rows)==80
    @test all(row.bound_holds for row in rows)
    @test length(essential)==1
    @test birth(only(essential))≈minimum(base)
    @test all(isfinite(row.bottleneck) for row in rows)
    @test all(row.bottleneck==0 for row in rows if row.epsilon==0)
    @test Bottleneck()(original[1],original[1])==0
    @test Bottleneck()(original[2],original[2])==0
    @test length(original[2])>0
    @test translation_rows[1].bottleneck≈c
    for dimension in 0:1
        expected=sort([(birth(bar)+c,death(bar)+c) for bar in original[dimension+1]])
        observed=sort([(birth(bar),death(bar)) for bar in translated[dimension+1]])
        @test length(expected)==length(observed)
        @test all(isapprox(a[1],b[1];atol=1e-12) &&
                  (isinf(a[2]) ? isinf(b[2]) : isapprox(a[2],b[2];atol=1e-12))
                  for (a,b) in zip(expected,observed))
    end
end

hashfile(path)=bytes2hex(sha256(read(path)))
metadata=Dict("julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),
              "target"=>config["target"],"field_sha256"=>hashfile(joinpath(OUT,"base_field.csv")),
              "config_sha256"=>hashfile(joinpath(ROOT,"config.toml")),
              "script_sha256"=>hashfile(@__FILE__),"comparisons"=>length(rows),
              "violations"=>count(row->!row.bound_holds,rows),
              "max_positive_excess"=>maximum(max(0,row.bottleneck-row.sup_norm) for row in rows),
              "maximum_bottleneck"=>maximum(row.bottleneck for row in rows))
project=Base.active_project()
metadata["project_file"]=relpath(project,ROOT)
metadata["project_sha256"]=hashfile(project)
manifest=joinpath(dirname(project),"Manifest.toml")
isfile(manifest) && (metadata["manifest_sha256"]=hashfile(manifest))
open(joinpath(OUT,"summary.toml"),"w") do io
    TOML.print(io,metadata)
end

CairoMakie.activate!()
with_theme(Theme(fontsize=16)) do
    fig=Figure(size=(1160,460))
    for (col,(name,field)) in enumerate([("Original field f",base),("Perturbed field g; epsilon = 0.1",illustration)])
        ax=Axis(fig[1,col];title=name,xlabel="x",ylabel="y",aspect=DataAspect())
        hm=heatmap!(ax,grid,grid,field;colormap=:viridis,colorrange=extrema(base))
        col==2 && Colorbar(fig[1,3],hm;label="Function value")
    end
    save(joinpath(OUT,"fields.png"),fig;px_per_unit=1.4)

    fig2=Figure(size=(1140,520))
    for dimension in 0:1
        ax=Axis(fig2[1,dimension+1];title="H$(dimension): diagrams remain within the perturbation bound",
                xlabel="Actual sup norm of perturbation",ylabel="Bottleneck distance")
        chosen=filter(row->row.dimension==dimension,rows)
        lines!(ax,[0,0.21],[0,0.21];color=:black,linestyle=:dash,label="Theorem: dB ≤ epsilon")
        scatter!(ax,[row.sup_norm for row in chosen],[row.bottleneck for row in chosen];
                 color=dimension==0 ? "#247c85" : "#b95b4f",markersize=11,label="Computed comparisons")
        axislegend(ax;position=:lt)
    end
    save(joinpath(OUT,"stability_bound.png"),fig2;px_per_unit=1.4)
    save(joinpath(OUT,"stability_bound.svg"),fig2)

    fig3=Figure(size=(1150,480))
    for dimension in 0:1
        ax=Axis(fig3[1,dimension+1];title="H$(dimension), finite intervals; epsilon = 0.1",xlabel="Birth",ylabel="Death",aspect=DataAspect())
        bars=filter(bar->isfinite(death(bar)),vcat(collect(original[dimension+1]),collect(illustration_diagrams[dimension+1])))
        low,high=extrema(vcat(birth.(bars),death.(bars)))
        padding=0.08*(high-low)
        low-=padding;high+=padding
        lines!(ax,[low,high],[low,high];color=:gray,linestyle=:dash)
        points=finite_points(original[dimension+1]);!isempty(points) && scatter!(ax,points;color="#247c85",marker=:circle,markersize=12,label="f")
        points=finite_points(illustration_diagrams[dimension+1]);!isempty(points) && scatter!(ax,points;color="#b95b4f",marker=:utriangle,markersize=10,label="g")
        axislegend(ax;position=:rb)
        limits!(ax,low,high,low,high)
    end
    save(joinpath(OUT,"diagrams.png"),fig3;px_per_unit=1.4)
end
println("Verified ",length(rows)," bottleneck bounds; violations = ",metadata["violations"])
println("Saved results to ",OUT)
end

main()
