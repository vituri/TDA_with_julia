include("common.jl")
using CairoMakie
mkpath(joinpath(ROOT,"figures"))
set_theme!(Theme(fontsize=17,fonts=(;regular="DejaVu Sans")))
local_summary=readdlm(joinpath(ROOT,"results","classification_summary.csv"),',',Float64;header=true)[1]
retrieval=readdlm(joinpath(ROOT,"results","retrieval_refined.csv"),',',Float64;header=true)[1]
grid=readdlm(joinpath(ROOT,"results","retrieval_refined_grid.csv"),',',Float64;header=true)[1]
blue="#2166ac";orange="#d95f02";green="#1b9e77"

fig=Figure(size=(1300,590))
ax=Axis(fig[1,1],title="Classification: nested 10-fold selection",xlabel="HKS index",ylabel="Test accuracy (%)",xticks=1:10,yticks=60:10:100)
errorbars!(ax,local_summary[:,1].-0.1,local_summary[:,3],local_summary[:,4];color=blue,whiskerwidth=10)
scatterlines!(ax,local_summary[:,1].-0.1,local_summary[:,3];color=blue,label="Local 10 splits(raw)",markersize=10)
errorbars!(ax,collect(1:10).+0.1,PAPER_CLASSIFICATION,PAPER_SD;color=orange,whiskerwidth=10)
scatterlines!(ax,collect(1:10).+0.1,PAPER_CLASSIFICATION;color=orange,label="Published Table 1",markersize=10,linestyle=:dash)
ylims!(ax,55,103);xlims!(ax,0.6,10.4)
axislegend(ax;position=:lb,labelsize=14)
ax2=Axis(fig[1,2],title="Retrieval: maximum over labeled queries",xlabel="HKS index",ylabel="Nearest-neighbor accuracy (%)",xticks=1:10,yticks=40:10:100)
scatterlines!(ax2,retrieval[:,1],retrieval[:,5];color=blue,label="Local refined oracle(raw)",markersize=10)
scatterlines!(ax2,1:10,PAPER_RETRIEVAL;color=orange,label="Published Table 3",markersize=10,linestyle=:dash)
scatterlines!(ax2,retrieval[:,1],retrieval[:,8];color=green,label="Local unit-norm sensitivity",markersize=8,linestyle=:dot)
ylims!(ax2,35,100);xlims!(ax2,0.6,10.4)
axislegend(ax2;position=:lb,labelsize=14)
Label(fig[2,1:2],"Classification bars: sample standard deviation over 10 new splits. Retrieval uses all 300 queries.",fontsize=14)
save(joinpath(ROOT,"figures","benchmark.png"),fig;px_per_unit=1.5)

fig2=Figure(size=(1280,580))
for (col,t) in enumerate([2,10])
    local ax=Axis(fig2[1,col],title="HKS$t: scale sensitivity",xlabel="Heat diffusion time σ (log scale)",ylabel="Nearest-neighbor accuracy (%)",xscale=log10,xticks=(2.0 .^[-12,-8,-4,0,4,8,12,16],["2⁻¹²","2⁻⁸","2⁻⁴","1","2⁴","2⁸","2¹²","2¹⁶"]))
    rs=grid[grid[:,1].==t,:]
    scatterlines!(ax,rs[:,2],rs[:,3];color=blue,label="Raw equation 10",markersize=8)
    scatterlines!(ax,rs[:,2],rs[:,4];color=green,label="Unit-norm sensitivity",markersize=8,linestyle=:dash)
    hlines!(ax,[PAPER_RETRIEVAL[t]];color=orange,linestyle=:dot,label="Published oracle")
    ylims!(ax,0,100);xlims!(ax,minimum(SIGMAS)*0.7,maximum(SIGMAS)*1.4)
    axislegend(ax;position=:rt,labelsize=13)
end
Label(fig2[2,1:2],"σ is the heat time in the paper's formula; the author executable takes --time=4σ.",fontsize=14)
save(joinpath(ROOT,"figures","scale-sensitivity.png"),fig2;px_per_unit=1.5)

data=load_data()
fig3=Figure(size=(1220,590))
names=["Male neutral (shape 0)","Female neutral (shape 100)","Child neutral (shape 200)"]
colors=[blue,orange,green]
for (col,t) in enumerate([2,10])
    local ax=Axis(fig3[1,col],title="Preserved H₁ diagrams at HKS$t",xlabel="Birth (stored HKS units)",ylabel="Death (stored HKS units)",aspect=1)
    diagrams=data.diagrams[t][[1,101,201]]
    for (d,name,color) in zip(diagrams,names,colors)
        scatter!(ax,birth.(d),death.(d);color,label=name,markersize=6)
    end
    lo=minimum(birth(p) for d in diagrams for p in d);hi=maximum(death(p) for d in diagrams for p in d)
    lines!(ax,[lo,hi],[lo,hi];color=:gray,linestyle=:dash)
    pad=0.06*(hi-lo);xlims!(ax,lo-pad,hi+pad);ylims!(ax,lo-pad,hi+pad)
    axislegend(ax;position=:rb,labelsize=12)
end
Label(fig3[2,1:2],"No rescaling, essential-point capping, persistence threshold or subsampling is applied.",fontsize=14)
save(joinpath(ROOT,"figures","preserved-diagrams.png"),fig3;px_per_unit=1.5)
println("Wrote 3 precomputed figures.")
