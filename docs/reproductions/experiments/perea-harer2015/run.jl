# Reproduce Perea & Harer, arXiv:1307.6188v2, Section 6.3 / Figure 3.
# All arrays use columns as points. No original experimental dataset is required:
# the paper explicitly supplies g1, g2, M, tau, and all sample times.
using TDARipserer, TDAPersistenceDiagrams, MetricSpaces, TDAplots
using CairoMakie, LinearAlgebra, Statistics, Random, TOML, SHA
import LinearAlgebra: norm

const HERE = @__DIR__
const DATA = joinpath(HERE, "data")
const RESULTS = joinpath(HERE, "results")
const FIGURES = joinpath(HERE, "figures")
foreach(mkpath, (DATA, RESULTS, FIGURES))
const CONFIG = TOML.parsefile(joinpath(HERE, "config.toml"))
const M = CONFIG["published_experiment"]["M"]
const TAU = 2π / (M + 1)
const TIMES = 2π .* collect(0:150) ./ 150
const FORMULAS = [t -> 0.6cos(t) + 0.8cos(2t),
                  t -> 0.8cos(t) + 0.6cos(2t)]
const AMPLITUDES = [(0.6, 0.8), (0.8, 0.6)]

function csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
end

sliding_window(f, ts, m, tau) = [f(t + j * tau) for j in 0:m, t in ts]
function center_normalize(A)
    C = A .- mean(A; dims=1)
    radii = sqrt.(sum(abs2, C; dims=1))
    minimum(radii) > 1e-12 || error("A window is constant; normalization undefined")
    return C ./ radii
end
function diagrams(A, p=3)
    # threshold=Inf prevents the default enclosing-radius shortcut from censoring H1.
    ripserer(Rips(EuclideanSpace(A); threshold=Inf); dim_max=1, modulus=p)
end
function longest(diag)
    isempty(diag) && return (birth=0.0, death=0.0, mp=0.0, n=0)
    all(isfinite, diag) || error("Censored or essential H1 interval")
    bar = diag[argmax(persistence.(diag))]
    return (; birth=birth(bar), death=death(bar), mp=persistence(bar), n=length(diag))
end
pairwise(A) = [norm(A[:, i] - A[:, j]) for i in axes(A, 2), j in axes(A, 2)]
function savecloud(name, ts, A, Z)
    csv(joinpath(DATA, name * ".csv"),
        ["time"; ["raw_$(j)" for j in axes(A,1)]; ["normalized_$(j)" for j in axes(Z,1)]],
        ([ts[i]; A[:,i]; Z[:,i]] for i in eachindex(ts)))
end

println("Exact Figure 3 reproduction: 151 windows, including the endpoint")
flush(stdout)
clouds = Matrix{Float64}[]
results = Dict{Tuple{Int, Int}, Any}()
summary = []
checks = []
for i in 1:2
    A = sliding_window(FORMULAS[i], TIMES, M, TAU)
    Z = center_normalize(A)
    push!(clouds, Z)
    savecloud("g$(i)_published", TIMES, A, Z)
    r1, r2 = AMPLITUDES[i]
    iso = permutedims(hcat(r1 .* cos.(TIMES), r1 .* sin.(TIMES),
                          r2 .* cos.(2 .* TIMES), r2 .* sin.(2 .* TIMES)))
    err_iso = maximum(abs, pairwise(Z) - pairwise(iso))
    err_cent = maximum(abs, vec(mean(Z; dims=1)))
    err_norm = maximum(abs, vec(sqrt.(sum(abs2,Z;dims=1))) .- 1)
    err_raw = maximum(abs, vec(sqrt.(sum(abs2,A;dims=1))) .- sqrt(2.5))
    endpoint_err = norm(Z[:,1] - Z[:,end])
    push!(checks, ("g$(i)_isometry_distance_max_error", err_iso, 1e-12, err_iso < 1e-12))
    push!(checks, ("g$(i)_coordinate_mean_max_error", err_cent, 1e-12, err_cent < 1e-12))
    push!(checks, ("g$(i)_unit_norm_max_error", err_norm, 1e-12, err_norm < 1e-12))
    push!(checks, ("g$(i)_raw_radius_max_error", err_raw, 1e-12, err_raw < 1e-12))
    push!(checks, ("g$(i)_endpoint_distance", endpoint_err, 1e-12, endpoint_err < 1e-12))
    for p in (2,3)
        res = diagrams(Z, p)
        results[(i,p)] = res
        s = longest(res[2])
        metricresult=ripserer(Rips(pairwise(Z);threshold=Inf);dim_max=1,modulus=p)[2]
        metricdb=Bottleneck()(res[2],metricresult)
        push!(checks,("g$(i)_F$(p)_distance_matrix_bottleneck",metricdb,1e-12,metricdb<1e-12))
        csv(joinpath(RESULTS,"g$(i)_F$(p)_H1.csv"),["birth","death","persistence"],
            [(birth(bar), death(bar), persistence(bar)) for bar in sort(res[2];by=persistence,rev=true)])
        # Strictly exceed the sampling Hausdorff radius pi/150 as required in 6.8.
        delta = (π/150) * (1 + 1e-10)
        kappa = 2sqrt(2) * sqrt(r1^2 + 4r2^2)
        lower = sqrt(3)*max(r1,r2) - delta*kappa
        push!(summary,("g$(i)",p,size(Z,2),s.n,s.birth,s.death,s.mp,s.mp/sqrt(3),lower,p>2))
        println("g$(i), F$(p): ", s, ", score=", s.mp/sqrt(3), ", lower bound=",lower)
        flush(stdout)
        if p > 2
            push!(checks,("g$(i)_F3_theorem_6_8_lower_bound_margin", s.mp-lower, 0.0, s.mp>=lower))
            isores = diagrams(iso,p)
            db = Bottleneck()(res[2],isores[2])
            push!(checks,("g$(i)_F3_isometric_bottleneck",db,1e-12,db<1e-12))
        end
    end
end
csv(joinpath(RESULTS,"published_summary.csv"),
    ["signal","modulus","n_windows","h1_intervals","longest_birth","longest_death","max_persistence","score","theorem_6_8_lower_bound","bound_field_applicable"],summary)

# Figure 3 layout: rows = signal, columns = field; same limits for all four panels.
fig = Figure(size=(900,780), fontsize=17)
for i in 1:2, (col,p) in enumerate((2,3))
    ax = Axis(fig[i,col], title="g$(i), coefficients F$(p)",xlabel="Birth (edge length)",ylabel="Death (edge length)",aspect=DataAspect())
    lines!(ax,[0,1.9],[0,1.9];color=:gray70,linewidth=1)
    d=results[(i,p)][2]
    scatter!(ax,birth.(d),death.(d);color=p==2 ? :darkorange : :dodgerblue,markersize=9)
    xlims!(ax,0,1.9); ylims!(ax,0,1.9)
end
save(joinpath(FIGURES,"figure3_reproduction.png"),fig;px_per_unit=1.5)
save(joinpath(FIGURES,"g1_F3_barcode.png"),barcode_plot(results[(1,3)][2]);px_per_unit=1.5)
save(joinpath(FIGURES,"g1_F3_diagram_tdaplots.png"),persistence_plot(results[(1,3)][2]);px_per_unit=1.5)

# An independently evaluable cosine benchmark at resonance: a regular unit n-gon.
convergence = []
for n in CONFIG["extensions"]["cosine_sample_sizes"]
    ts=2π.*collect(0:n-1)./n
    A=sliding_window(cos,ts,M,TAU)
    Z=center_normalize(A)
    s=longest(diagrams(Z,3)[2])
    exact_birth=2sin(π/n)
    # All chosen n are divisible by 3; the first triangle spanning the hole is equilateral.
    exact_death=sqrt(3)
    err=max(abs(s.birth-exact_birth),abs(s.death-exact_death))
    push!(checks,("cosine_n$(n)_exact_regular_polygon_endpoints",err,1e-12,err<1e-12))
    push!(convergence,(n,s.birth,s.death,s.mp,s.mp/sqrt(3),exact_birth,exact_death,err))
    savecloud("cosine_n$(n)",ts,A,Z)
end
csv(joinpath(RESULTS,"cosine_convergence.csv"),
    ["n","birth","death","max_persistence","score","exact_birth","exact_death","max_endpoint_error"],convergence)

# A sensitivity extension, explicitly distinct from the original Figure 3 grid.
# For the pure cosine, globally scale (without per-window normalization) to retain
# the ellipse geometry in Section 2.1. The mixtures use the paper's normalization.
windowrows=[]
for ratio in CONFIG["extensions"]["window_ratios"]
    tau=TAU*ratio
    for i in 1:2
        A=sliding_window(FORMULAS[i],TIMES,M,tau)
        Z=center_normalize(A)
        s=longest(diagrams(Z,3)[2])
        push!(windowrows,("g$(i)",ratio,tau,M*tau/(2π),"centered_unit_norm",s.mp,s.mp/sqrt(3),NaN,NaN))
    end
    A=sliding_window(cos,TIMES,M,tau)
    X=A/sqrt((M+1)/2)
    s=longest(diagrams(X,3)[2])
    q=abs(sin((M+1)*tau)/sin(tau))
    major=sqrt(((M+1)+q)/(M+1))
    minor=sqrt(((M+1)-q)/(M+1))
    B=hcat(cos.(collect(0:M).*tau),-sin.(collect(0:M).*tau))/sqrt((M+1)/2)
    axiserr=maximum(abs,eigvals(Symmetric(B'B))-[minor^2,major^2])
    push!(checks,("cosine_ratio$(ratio)_ellipse_squared_axes",axiserr,1e-12,axiserr<1e-12))
    push!(windowrows,("cosine",ratio,tau,M*tau/(2π),"global_scale_only",s.mp,s.mp/sqrt(3),minor,major))
end
csv(joinpath(RESULTS,"window_sensitivity.csv"),
    ["signal","tau_ratio","tau","window_fraction_of_period","normalization","max_persistence","score","minor_axis","major_axis"],windowrows)
println("Window sensitivity completed"); flush(stdout)

# Same normalized metric after positive affine rescaling of the original signal.
ZA=center_normalize(sliding_window(t -> 3.2 * FORMULAS[1](t)+7,TIMES,M,TAU))
err_affine=maximum(abs,pairwise(ZA)-pairwise(clouds[1]))
db_affine=Bottleneck()(results[(1,3)][2],diagrams(ZA,3)[2])
push!(checks,("affine_pairwise_distance_max_error",err_affine,1e-12,err_affine<1e-12))
push!(checks,("affine_H1_bottleneck",db_affine,1e-12,db_affine<1e-12))

# Controls use ordinary time-series delay windows, not cyclic wrapping or splines.
controlrows=[]
for seed in CONFIG["extensions"]["control_seeds"]
    count=CONFIG["extensions"]["control_samples"]
    times=2π.*collect(0:count-1)./150
    for kind in ("permuted_g1", "white_noise")
        rng=MersenneTwister(seed)
        observations=kind=="permuted_g1" ? FORMULAS[1].(times)[randperm(rng,count)] : randn(rng,count)
        lag=CONFIG["extensions"]["control_lag"]
        nwin=length(observations)-M*lag
        A=[observations[k+j*lag] for j in 0:M,k in 1:nwin]
        Z=center_normalize(A)
        s=longest(diagrams(Z,3)[2])
        push!(controlrows,(kind,seed,nwin,s.birth,s.death,s.mp,s.mp/sqrt(3)))
        csv(joinpath(DATA,"$(kind)_seed$(seed)_observations.csv"),["time","value"],zip(times,observations))
        savecloud("$(kind)_seed$(seed)_windows",times[1:nwin],A,Z)
    end
end
csv(joinpath(RESULTS,"controls.csv"),
    ["control","seed","n_windows","birth","death","max_persistence","score"],controlrows)

# A concise composite showing dependence on window, sample density, and controls.
fig=Figure(size=(1250,850),fontsize=16)
ax1=Axis(fig[1,1],xlabel="Window length / period",ylabel="mp / sqrt(3)",title="Mixtures: centered unit windows")
for (signal,color) in (("g1",:darkorange),("g2",:dodgerblue))
    mixture_rows=filter(x->x[1]==signal,windowrows)
    scatterlines!(ax1,[x[4] for x in mixture_rows],[x[7] for x in mixture_rows];color=color,label=signal,markersize=5)
end
vlines!(ax1,[M/(M+1)];color=:gray50,linestyle=:dash,label="paper window")
axislegend(ax1;position=:rb)
ax2=Axis(fig[1,2],xlabel="Window length / period",ylabel="Normalized axis / mp",title="Cosine: ellipse before unit normalization")
rr=filter(x->x[1]=="cosine",windowrows)
lines!(ax2,[x[4] for x in rr],[x[8] for x in rr];label="minor axis",color=:seagreen)
lines!(ax2,[x[4] for x in rr],[x[9] for x in rr];label="major axis",color=:purple)
scatterlines!(ax2,[x[4] for x in rr],[x[7] for x in rr];label="mp / sqrt(3)",color=:black,markersize=5)
axislegend(ax2;position=:rb)
ax3=Axis(fig[2,1],xlabel="Number of unique cosine windows",ylabel="mp / sqrt(3)",title="Finite-sample score approaches 1")
scatterlines!(ax3,[x[1] for x in convergence],[x[5] for x in convergence];color=:seagreen,markersize=10)
hlines!(ax3,[1.0];color=:gray50,linestyle=:dash)
ylims!(ax3,0.85,1.01)
ax4=Axis(fig[2,2],xlabel="Control (10 independent seeds)",ylabel="mp / sqrt(3)",title="Ordering and white-noise controls",xticks=([1,2],["permuted g1","white noise"]))
for (j,kind) in enumerate(("permuted_g1","white_noise"))
    control_rows=filter(x->x[1]==kind,controlrows)
    scatter!(ax4,[j+(k-5.5)*0.02 for k in 1:length(control_rows)],[x[7] for x in control_rows];color=:gray50,markersize=9)
    hlinescore=median([x[7] for x in control_rows])
    lines!(ax4,[j-0.17,j+0.17],[hlinescore,hlinescore];color=:black,linewidth=3)
end
hlines!(ax4,[summary[2][8]];color=:darkorange,label="published g1, F3")
hlines!(ax4,[summary[4][8]];color=:dodgerblue,label="published g2, F3")
axislegend(ax4;position=:rt)
save(joinpath(FIGURES,"sensitivity_and_controls.png"),fig;px_per_unit=1.3)

csv(joinpath(RESULTS,"checks.csv"),["check","value","comparison_threshold","passed"],checks)
all(last,checks) || error("A scientific consistency check failed")

# Record exact executable environment and local source file content, without changing it.
packages=Dict{String,Any}()
for module_ in (TDARipserer,TDAPersistenceDiagrams,MetricSpaces,TDAplots)
    entry=pathof(module_)
    root=dirname(dirname(entry))
    sourcefiles=sort(vcat([joinpath(root,"Project.toml")],
        [joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(root,"src")) for file in files]))
    treehash=bytes2hex(sha256(join([relpath(file,root)*"="*bytes2hex(sha256(read(file)))
        for file in sourcefiles],"\n")))
    packages[string(module_)]=Dict("entrypoint"=>entry,"entrypoint_sha256"=>bytes2hex(sha256(read(entry))),
        "source_tree_sha256"=>treehash,
        "git_head"=>strip(read(`git -C $root rev-parse HEAD`,String)),
        "git_worktree_clean"=>isempty(strip(read(`git -C $root status --porcelain`,String))))
end
project=Base.active_project()
manifest=joinpath(dirname(project),"Manifest.toml")
provenance=Dict("julia_version"=>string(VERSION),"active_project"=>project,
    "project_sha256"=>bytes2hex(sha256(read(project))),
    "manifest_sha256"=>bytes2hex(sha256(read(manifest))),
    "config_sha256"=>bytes2hex(sha256(read(joinpath(HERE,"config.toml")))),
    "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),"packages"=>packages)
open(joinpath(RESULTS,"environment.toml"),"w") do io
    TOML.print(io,provenance)
end
files=sort(filter(f->isfile(f),vcat([joinpath(DATA,f) for f in readdir(DATA)],
    [joinpath(RESULTS,f) for f in readdir(RESULTS) if f!="sha256.csv" && !endswith(f,".log")])))
csv(joinpath(RESULTS,"sha256.csv"),["path","sha256"],
    [(relpath(f,HERE),bytes2hex(sha256(read(f)))) for f in files])
println("Passed ",length(checks)," consistency checks. Artifacts written to ",HERE)
for kind in ("permuted_g1","white_noise")
    scores=[x[7] for x in controlrows if x[1]==kind]
    println(kind," score median/min/max: ",median(scores)," / ",minimum(scores)," / ",maximum(scores))
end
