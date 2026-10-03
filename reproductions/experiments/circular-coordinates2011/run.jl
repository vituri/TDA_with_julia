# Published targets: full 2011 author manuscript, Sections 3.3 and 3.7.
# No original realization/seed is supplied; generate fresh samples from its laws.
using TDARipserer, TDAPersistenceDiagrams, MetricSpaces, TDAplots, CairoMakie
using Random, Statistics, LinearAlgebra, SparseArrays, TOML, SHA
import LinearAlgebra: norm
const HERE=@__DIR__
const CONFIG=TOML.parsefile(joinpath(HERE,"config.toml"))
const DATA=joinpath(HERE,"data")
const RESULTS=joinpath(HERE,"results")
const FIGURES=joinpath(HERE,"figures")
const PRIME=CONFIG["analysis"]["modulus"]
foreach(mkpath,(DATA,RESULTS,FIGURES))
function csv(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','))
        foreach(row->println(io,join(row,',')),rows)
    end
end
wrap(x)=mod(x+0.5,1)-0.5
const CHECKS=[]
function check(name,value,threshold,passed)
    push!(CHECKS,(name,value,threshold,passed))
    passed || error("Scientific check failed: "*name*", value="*string(value))
end

function circle_sample()
    rng=MersenneTwister(CONFIG["circle"]["seed"])
    n=CONFIG["circle"]["n"]
    u=rand(rng,n)
    clean=permutedims(hcat(cos.(2π.*u),sin.(2π.*u)))
    jitter=permutedims(0.4.*rand(rng,n,2))
    return clean+jitter,reshape(u,1,:),jitter
end
function torus_sample()
    rng=MersenneTwister(CONFIG["torus"]["seed"])
    n=CONFIG["torus"]["n"]
    u=rand(rng,n);v=rand(rng,n)
    clean=permutedims(hcat((2 .+cos.(2π.*v)).*cos.(2π.*u),
        (2 .+cos.(2π.*v)).*sin.(2π.*u),sin.(2π.*v)))
    jitter=permutedims(0.2.*rand(rng,n,3))
    return clean+jitter,permutedims(hcat(u,v)),jitter
end
function save_sample(name,X,phases,jitter)
    csv(joinpath(DATA,name*".csv"),
        [["x$(j)" for j in axes(X,1)];["phase$(j)" for j in axes(phases,1)];["jitter$(j)" for j in axes(jitter,1)]],
        ([X[:,i];phases[:,i];jitter[:,i]] for i in axes(X,2)))
end

# Compact oriented incidence matrix. TDARipserer stores edges (u,v), u>v;
# B*f therefore equals f(v)-f(u), consistently with its cocycle convention.
function skeleton(filtration,delta)
    n=TDARipserer.nv(filtration)
    edge_list=filter(e->birth(e)<delta,TDARipserer.edges(filtration))
    rows=Int[];cols=Int[];vals=Float64[]
    adjacency=falses(n,n)
    lookup=Dict{Int,Int}()
    neighbors=[Int[] for _ in 1:n]
    for (r,e) in enumerate(edge_list)
        u,v=Tuple(e)
        append!(rows,(r,r));append!(cols,(u,v));append!(vals,(-1.0,1.0))
        adjacency[u,v]=adjacency[v,u]=true
        push!(neighbors[u],v);push!(neighbors[v],u)
        lookup[index(e)]=r
    end
    B=sparse(rows,cols,vals,length(edge_list),n)
    anchors=Int[];parents=zeros(Int,n);components=zeros(Int,n)
    for root in 1:n
        components[root]!=0 && continue
        push!(anchors,root);components[root]=length(anchors)
        queue=[root];k=1
        while k<=length(queue)
            u=queue[k];k+=1
            for v in neighbors[u]
                if components[v]==0
                    components[v]=components[root];parents[v]=u;push!(queue,v)
                end
            end
        end
    end
    triangles=NTuple{3,Int}[]
    for c in 1:n,b in 1:c-1
        adjacency[c,b] || continue
        for a in 1:b-1
            adjacency[c,a]&&adjacency[b,a]&&push!(triangles,(a,b,c))
        end
    end
    return (;B,edge_list,lookup,anchors,parents,components,triangles,adjacency)
end
edgeid(u,v)=index((max(u,v),min(u,v)))
function triangle_residual(skel,alpha)
    greatest=0.0
    for (a,b,c) in skel.triangles
        value=alpha[skel.lookup[edgeid(c,b)]]-alpha[skel.lookup[edgeid(c,a)]]+alpha[skel.lookup[edgeid(b,a)]]
        greatest=max(greatest,abs(value))
    end
    greatest
end
function circular_fit(theta,truth)
    best=(R=-Inf,degree=zeros(Int,size(truth,1)),offset=0.0,rmse=Inf)
    grid=CONFIG["analysis"]["integer_degree_search"]
    candidates=size(truth,1)==1 ? [(k,) for k in grid if k!=0] :
        [(m,n) for m in grid,n in grid if m!=0||n!=0]
    for degree in candidates
        target=vec(transpose(collect(degree))*truth)
        z=mean(exp.(2π*im.*(theta-target)))
        R=abs(z);offset=angle(z)/(2π)
        rmse=sqrt(mean(abs2,wrap.(theta-target.-offset)))
        if R>best.R
            best=(;R,degree=collect(degree),offset,rmse)
        end
    end
    best
end
function tree_path(skel,u,v)
    ancestry=Dict{Int,Int}();left=Int[]
    current=u
    while current!=0
        push!(left,current);ancestry[current]=length(left);current=skel.parents[current]
    end
    right=Int[];current=v
    while !haskey(ancestry,current)
        push!(right,current);current=skel.parents[current]
        current==0 && error("Path crosses disconnected components")
    end
    [left[1:ancestry[current]];reverse(right)]
end
# Certify integer periods on an actual closed cycle of the chosen Rips graph,
# not merely on a scatter plot sorted by a known angular parameter.
function nonzero_period_cycle(skel,alpha,harmonic,phases)
    for (r,e) in enumerate(skel.edge_list)
        u,v=Tuple(e)
        (skel.parents[u]==v || skel.parents[v]==u) && continue
        path=tree_path(skel,u,v)
        walk=[[(path[i],path[i+1]) for i in 1:length(path)-1];[(v,u)]]
        period=sum((a>b ? 1 : -1)*alpha[skel.lookup[edgeid(a,b)]] for (a,b) in walk)
        if period!=0
            smooth=sum((a>b ? 1 : -1)*harmonic[skel.lookup[edgeid(a,b)]] for (a,b) in walk)
            winding=[sum(wrap(phases[j,b]-phases[j,a]) for (a,b) in walk) for j in axes(phases,1)]
            return (;walk,period,smooth,winding)
        end
    end
    error("Lifted cocycle has no nonzero fundamental-cycle period")
end
function coordinate_case(name,filtration,interval,delta,phases)
    skel=skeleton(filtration,delta)
    alpha=zeros(Int,length(skel.edge_list))
    raw=zeros(Int,length(alpha))
    for (e,c) in interval.representative
        birth(e)<delta || continue
        r=skel.lookup[index(e)]
        residue=Int(c)
        raw[r]=residue
        alpha[r]=residue>(PRIME-1)÷2 ? residue-PRIME : residue
    end
    # If this fails, do not silently claim the integer lift exists.
    int_residual=triangle_residual(skel,alpha)
    check(name*"_integer_triangle_coboundary",int_residual,0.0,int_residual==0)
    check(name*"_lift_modular_congruence",maximum(abs,mod.(alpha-raw,PRIME)),0.0,all(iszero,mod.(alpha-raw,PRIME)))
    free=setdiff(collect(1:size(skel.B,2)),skel.anchors)
    f=zeros(size(skel.B,2))
    if !isempty(free)
        f[free]=-(skel.B[:,free]\Float64.(alpha))
    end
    harmonic=Float64.(alpha)+skel.B*f
    theta=mod.(f,1.0)
    stationarity=norm(skel.B'*harmonic,Inf)
    closed=triangle_residual(skel,harmonic)
    before=sum(abs2,alpha);after=sum(abs2,harmonic)
    check(name*"_harmonic_stationarity",stationarity,1e-9,stationarity<1e-9)
    check(name*"_harmonic_triangle_coboundary",closed,1e-10,closed<1e-10)
    check(name*"_energy_reduction",after-before,1e-10,after<=before+1e-10)
    cycle=nonzero_period_cycle(skel,alpha,harmonic,phases)
    check(name*"_cycle_period_preservation",abs(cycle.smooth-cycle.period),1e-10,abs(cycle.smooth-cycle.period)<1e-10)
    check(name*"_primitive_integer_period",abs(cycle.period),1.0,abs(cycle.period)==1)
    fit=circular_fit(theta,phases)
    counts=[count(t->k/20<=t<(k+1)/20,theta) for k in 0:19]
    probabilities=counts/length(theta)
    entropy=-sum(p*log(p) for p in probabilities if p>0)
    largest_bin=maximum(probabilities)
    csv(joinpath(RESULTS,name*"_edges.csv"),["tail","head","length","field_residue","integer_lift","harmonic","df"],
        ((Tuple(e)[1],Tuple(e)[2],birth(e),raw[r],alpha[r],harmonic[r],(skel.B*f)[r]) for (r,e) in enumerate(skel.edge_list)))
    csv(joinpath(RESULTS,name*"_vertices.csv"),["vertex","component","potential","coordinate"],
        ((i,skel.components[i],f[i],theta[i]) for i in eachindex(f)))
    csv(joinpath(RESULTS,name*"_cycle.csv"),["tail","head"],cycle.walk)
    println(name,": birth=",birth(interval),", death=",death(interval),", R=",fit.R,
        ", degree=",fit.degree,", RMSEcycles=",fit.rmse,", components=",length(skel.anchors),
        ", energy=",before," -> ",after,", period=",cycle.period,", truth winding=",cycle.winding,
        ", maxbin=",largest_bin,", entropy=",entropy)
    flush(stdout)
    return (;name,theta,f,harmonic,alpha,skel,fit,cycle,delta,before,after,entropy,largest_bin,
        birth=birth(interval),death=death(interval),stationarity,closed)
end
function save_diagram(name,diagram,threshold)
    csv(joinpath(RESULTS,name*"_H1.csv"),["birth","death","death_censored","observed_lifetime"],
        ((birth(bar),death(bar),!isfinite(bar),min(death(bar),threshold)-birth(bar)) for bar in diagram))
end
active(diagram,delta,threshold)=sort(filter(bar->birth(bar)<delta<death(bar),diagram);
    by=bar->min(death(bar),threshold)-birth(bar),rev=true)

circle,uc,noise_c=circle_sample()
torus,uv,noise_t=torus_sample()
save_sample("circle_seed2011",circle,uc,noise_c)
save_sample("torus_seed2011",torus,uv,noise_t)
check("circle_uniform_noise_bounds",maximum(noise_c),0.4,minimum(noise_c)>=0&&maximum(noise_c)<0.4)
check("torus_uniform_noise_bounds",maximum(noise_t),0.2,minimum(noise_t)>=0&&maximum(noise_t)<0.2)

fc=Rips(EuclideanSpace(circle);threshold=0.5)
dc=ripserer(fc;dim_max=1,modulus=PRIME,reps=true)[2]
save_diagram("circle",dc,0.5)
globalbars=active(dc,0.4,0.5)
localbars=active(dc,0.14,0.5)
check("circle_global_active_class_count",length(globalbars),1.0,length(globalbars)==1)
check("circle_local_has_multiple_classes",length(localbars),1.0,length(localbars)>1)
localbar=last(filter(b->isfinite(b)&&persistence(b)>1e-6,localbars))
globalcoord=coordinate_case("circle_global",fc,first(globalbars),0.4,uc)
clocal=coordinate_case("circle_local",fc,localbar,0.14,uc)
check("circle_global_degree_abs_one",abs(only(globalcoord.fit.degree)),1.0,abs(only(globalcoord.fit.degree))==1)
check("circle_global_angular_association",globalcoord.fit.R,0.9,globalcoord.fit.R>0.9)
check("circle_local_histogram_more_concentrated",clocal.largest_bin-globalcoord.largest_bin,0.0,clocal.largest_bin>globalcoord.largest_bin)

ft=Rips(EuclideanSpace(torus);threshold=sqrt(3))
dt=ripserer(ft;dim_max=1,modulus=PRIME,reps=true)[2]
save_diagram("torus",dt,sqrt(3))
torusbars=active(dt,1.6,sqrt(3))
check("torus_two_active_classes",length(torusbars),2.0,length(torusbars)>=2)
tcoords=[coordinate_case("torus_coordinate$(j)",ft,torusbars[j],1.6,uv) for j in 1:2]
A=reduce(vcat,[permutedims(c.fit.degree) for c in tcoords])
check("torus_degree_matrix_unimodular",det(Float64.(A)),1.0,abs(det(Float64.(A)))==1)
for (j,c) in enumerate(tcoords)
    check("torus_coord$(j)_angular_association",c.fit.R,0.8,c.fit.R>0.8)
end

# API comparison is a later-method extension, not the exact 2011 construction.
# All points are landmarks: no random maxmin subsampling. Its full-coverage policy
# selects smoothing delta0.5 here; the published vertex coordinate used delta0.4.
points=EuclideanSpace(circle)
cc=CircularCoordinates(points,collect(eachindex(points));modulus=PRIME,threshold=0.5,coverage=1.0,dim_max=1)
api_raw=cc(points)[:,1]
check("all_landmarks_api_no_missing",count(ismissing,api_raw),0.0,!any(ismissing,api_raw))
api_theta=Float64.(api_raw)
api_fit=circular_fit(api_theta,uc)
check("all_landmarks_api_degree_abs_one",abs(only(api_fit.degree)),1.0,abs(only(api_fit.degree))==1)
api_pair=circular_fit(api_theta,reshape(globalcoord.theta,1,:))
csv(joinpath(RESULTS,"later_api_comparison.csv"),
    ["method","smoothing_delta","n_landmarks","angular_resultant","degree","angular_rmse_cycles","resultant_vs_2011","rmse_vs_2011_cycles"],
    [("Perea2020_all_landmarks",2cc.coordinate_data[1].radius,length(cc.landmarks),api_fit.R,only(api_fit.degree),api_fit.rmse,api_pair.R,api_pair.rmse)])
csv(joinpath(RESULTS,"later_api_coordinates.csv"),["vertex","coordinate"],enumerate(api_theta))
println("Perea2020 API: R=",api_fit.R,", degree=",api_fit.degree,", smoothing_delta=",2cc.coordinate_data[1].radius,
    ", phase RMSE against 2011 vertex map=",api_pair.rmse)

cases=[globalcoord,clocal,tcoords[1],tcoords[2]]
summary=[]
for c in cases
    degrees=[c.fit.degree;zeros(Int,2-length(c.fit.degree))]
    push!(summary,(c.name,c.delta,c.birth,c.death,length(c.skel.anchors),length(c.skel.edge_list),length(c.skel.triangles),
        c.fit.R,degrees[1],degrees[2],c.fit.offset,c.fit.rmse,c.before,c.after,c.stationarity,c.closed,
        c.cycle.period,c.cycle.smooth,c.largest_bin,c.entropy))
end
csv(joinpath(RESULTS,"summary.csv"),["case","delta","birth","death","components","edges","triangles",
    "circular_resultant","degree_phase1","degree_phase2","offset","angular_rmse_cycles","energy_before","energy_after",
    "stationarity_inf","triangle_residual_inf","integer_cycle_period","harmonic_cycle_period","max_histogram_bin_fraction","histogram_entropy_nats"],summary)
sc=skeleton(fc,0.5);st=skeleton(ft,sqrt(3))
csv(joinpath(RESULTS,"complex_sizes.csv"),["data","n","threshold","edges","triangles","total_2_skeleton","published_total"],
    [("circle",200,0.5,length(sc.edge_list),length(sc.triangles),200+length(sc.edge_list)+length(sc.triangles),23475),
     ("torus",400,sqrt(3),length(st.edge_list),length(st.triangles),400+length(st.edge_list)+length(st.triangles),61522)])

function draw_diagram(ax,diagram,threshold,delta)
    finitebars=filter(isfinite,diagram);censored=filter(!isfinite,diagram)
    lines!(ax,[0,threshold],[0,threshold];color=:gray65)
    scatter!(ax,birth.(finitebars),death.(finitebars);color=:dodgerblue,markersize=6)
    if !isempty(censored)
        scatter!(ax,birth.(censored),fill(threshold,length(censored));color=:darkorange,marker=:utriangle,markersize=11,label="death beyond cutoff")
    end
    vlines!(ax,[delta];color=:seagreen,linestyle=:dash)
    hlines!(ax,[delta];color=:seagreen,linestyle=:dash)
    xlims!(ax,0,threshold*1.03);ylims!(ax,0,threshold*1.06)
end
fig=Figure(size=(1120,1150),fontsize=17)
ax=Axis(fig[1,1:3],title="Noisy circle: H1 over F47 (triangles indicate censored deaths)",xlabel="Birth",ylabel="Death",aspect=DataAspect())
draw_diagram(ax,dc,0.5,0.4)
for (row,c) in enumerate((globalcoord,clocal))
    ax1=Axis(fig[row+1,1],title=c.name*": coordinate histogram",xlabel="Circular coordinate",ylabel="Count")
    hist!(ax1,c.theta;bins=range(0,1;length=21),color=row==1 ? :dodgerblue : :darkorange)
    ax2=Axis(fig[row+1,2],title="Inferred versus original phase",xlabel="Original phase",ylabel="Inferred (offset aligned)")
    scatter!(ax2,vec(uc),mod.(c.theta.-c.fit.offset,1);markersize=4,color=row==1 ? :dodgerblue : :darkorange)
    xlims!(ax2,0,1);ylims!(ax2,0,1)
    ax3=Axis(fig[row+1,3],title="delta = $(c.delta)",xlabel="x",ylabel="y",aspect=DataAspect())
    scatter!(ax3,circle[1,:],circle[2,:];color=c.theta,colormap=:hsv,colorrange=(0,1),markersize=7)
end
save(joinpath(FIGURES,"noisy_circle_reproduction.png"),fig;px_per_unit=1.3)

fig=Figure(size=(1300,450),fontsize=17)
ax=Axis(fig[1,1],title="Torus H1, delta = 1.6",xlabel="Birth",ylabel="Death",aspect=DataAspect())
draw_diagram(ax,dt,sqrt(3),1.6)
for j in 1:2
    ax3=Axis3(fig[1,j+1],title="Inferred coordinate $(j)",xlabel="x",ylabel="y",zlabel="z")
    scatter!(ax3,torus[1,:],torus[2,:],torus[3,:];color=tcoords[j].theta,colormap=:hsv,colorrange=(0,1),markersize=6)
end
save(joinpath(FIGURES,"torus_coordinates.png"),fig;px_per_unit=1.4)
fig=Figure(size=(1100,750),fontsize=16)
aligned=[mod.(c.theta.-c.fit.offset,1) for c in tcoords]
panels=[(aligned[1],uv[1,:],"Inferred 1","Original longitude"),
        (aligned[1],uv[2,:],"Inferred 1","Original meridian"),
        (aligned[1],aligned[2],"Inferred 1","Inferred 2"),
        (aligned[2],uv[1,:],"Inferred 2","Original longitude"),
        (aligned[2],uv[2,:],"Inferred 2","Original meridian"),
        (uv[1,:],uv[2,:],"Original longitude","Original meridian")]
for (k,(x,y,xname,yname)) in enumerate(panels)
    ax=Axis(fig[(k-1)÷3+1,(k-1)%3+1],xlabel=xname,ylabel=yname,aspect=DataAspect())
    scatter!(ax,x,y;markersize=4,color=:dodgerblue)
    xlims!(ax,0,1);ylims!(ax,0,1)
end
save(joinpath(FIGURES,"torus_correlations.png"),fig;px_per_unit=1.3)
save(joinpath(FIGURES,"circle_H1_barcode.png"),barcode_plot(dc;infinity=0.5);px_per_unit=1.3)

csv(joinpath(RESULTS,"checks.csv"),["check","value","comparison_threshold","passed"],CHECKS)
packages=Dict{String,Any}()
for module_ in (TDARipserer,TDAPersistenceDiagrams,MetricSpaces,TDAplots)
    entry=pathof(module_);root=dirname(dirname(entry))
    sourcefiles=sort(vcat([joinpath(root,"Project.toml")],
        [joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(root,"src")) for file in files]))
    treehash=bytes2hex(sha256(join([relpath(file,root)*"="*bytes2hex(sha256(read(file))) for file in sourcefiles],"\n")))
    packages[string(module_)]=Dict("entrypoint"=>entry,"source_tree_sha256"=>treehash,
        "git_head"=>strip(read(`git -C $root rev-parse HEAD`,String)),
        "git_worktree_clean"=>isempty(strip(read(`git -C $root status --porcelain`,String))))
end
project=Base.active_project();manifest=joinpath(dirname(project),"Manifest.toml")
provenance=Dict("julia_version"=>string(VERSION),"active_project"=>project,
    "project_sha256"=>bytes2hex(sha256(read(project))),"manifest_sha256"=>bytes2hex(sha256(read(manifest))),
    "config_sha256"=>bytes2hex(sha256(read(joinpath(HERE,"config.toml")))),
    "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
    "paper_pdf_sha256"=>bytes2hex(sha256(read(joinpath(HERE,"source","paper2011.pdf")))),"packages"=>packages)
open(joinpath(RESULTS,"environment.toml"),"w") do io
    TOML.print(io,provenance)
end
files=sort(vcat([joinpath(DATA,f) for f in readdir(DATA)],
    [joinpath(RESULTS,f) for f in readdir(RESULTS) if f!="sha256.csv"&&!endswith(f,".log")]))
csv(joinpath(RESULTS,"sha256.csv"),["path","sha256"],
    [(relpath(file,HERE),bytes2hex(sha256(read(file)))) for file in files if isfile(file)])
println("Passed ",length(CHECKS)," checks. Degree matrix = ",A)
