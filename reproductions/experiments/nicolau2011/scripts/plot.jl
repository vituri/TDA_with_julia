using CairoMakie, TDAplots

function draw!(ax,model,data;mode=:filter)
    pos=layout_spring(model.M;seed=2011,iterations=300)
    points=Point2f[]
    for e in Graphs.edges(model.M.g);push!(points,pos[Graphs.src(e)],pos[Graphs.dst(e)]);end
    isempty(points) || linesegments!(ax,points;color="#8695a0",linewidth=.7)
    if mode==:filter
        values=log10.(model.f)
        scatter!(ax,pos;color=node_colors(model.M,values),colormap=:viridis,
                 markersize=4 .+2sqrt.(length.(model.M.C)),strokewidth=.3,strokecolor=:white)
    else
        fractions=[isempty(filter(>(13),c)) ? NaN : mean(data.clinical_er[filter(>(13),c)]) for c in model.M.C]
        # Entirely normal nodes are gray; no invented clinical ER value for controls.
        cmap=CairoMakie.Makie.to_colormap(:viridis)
        colors=[isnan(x) ? RGBAf(.6,.6,.6,1) : cmap[clamp(round(Int,1+x*(length(cmap)-1)),1,length(cmap))] for x in fractions]
        scatter!(ax,pos;color=colors,markersize=4 .+2sqrt.(length.(model.M.C)),strokewidth=.3,strokecolor=:white)
    end
    hidedecorations!(ax);hidespines!(ax)
end

function plot_results(models,sensitivity,data,result,selection)
    CairoMakie.activate!()
    with_theme(Theme(fontsize=15,backgroundcolor="#fcfcfa")) do
        fig=Figure(size=(1320,820))
        Label(fig[0,1:2],"NKI295 + BCN13: historical data, partial PAD reconstruction";fontsize=24,font=:bold)
        for (col,name) in enumerate(("power4","power1"))
            model=models[name];s=summary(model)
            ax=Axis(fig[1,col];title="||Dc||₂$(col==1 ? "⁴ (published power)" : " (sensitivity control)")\n$(s.nodes) nodes · $(s.components) components",aspect=DataAspect())
            draw!(ax,model,data)
            ax2=Axis(fig[2,col];title="Clinical ER fraction, annotated after Mapper",aspect=DataAspect())
            draw!(ax2,model,data;mode=:ER)
        end
        Colorbar(fig[3,1:2];vertical=false,colormap=:viridis,colorrange=(0,1),label="Bottom row: ER-positive tumor fraction (0–1); normal-only nodes are gray",width=Relative(.72),height=14)
        Label(fig[4,1:2],"$(length(selection.selected)) locally selected genes · HSM dimension 10 · 15 equal-width intervals with 80% overlap\nSingle linkage, local first-empty-bin rule (10 bins). Top colors show mean log₁₀ filter, scaled independently.\nNode area grows with membership; graph positions have no biological units. These are not the published c-MYB+ groups.";fontsize=13,tellwidth=false)
        save(joinpath(ROOT,"results","mapper.png"),fig;px_per_unit=1.25)
        save(joinpath(ROOT,"results","mapper.svg"),fig)

        fig2=Figure(size=(1250,610))
        ax=Axis(fig2[1,1];xlabel="PCA dimension of FLAT normal matrix",ylabel="Wold invariant",yscale=log10,title="Healthy State Model diagnostic")
        lines!(ax,1:12,result.wold;color="#2a817b",linewidth=2)
        scatter!(ax,1:12,result.wold;color="#2a817b",markersize=8)
        vlines!(ax,[10];color="#ae536b",linestyle=:dash,label="Published dimension: 10")
        axislegend(ax;position=:lt)
        ax2=Axis(fig2[1,2];xlabel="Gene deviation: max(|q₀.₀₅|, |q₀.₉₅|)",ylabel="Genes",title="Residual gene thresholding")
        hist!(ax2,selection.q;bins=70,color="#2a817b")
        vlines!(ax2,[selection.relaxed,selection.stringent];color=["#ca9844","#ae536b"],linewidth=2)
        Label(fig2[2,1:2],"$(length(data.genes)) unambiguously matched gene symbols, rather than the published 12,237 UniGenes.\nLocal 85th/98th thresholds: $(round(selection.relaxed;digits=3)) / $(round(selection.stringent;digits=3)); $(length(selection.lax)) relaxed and $(length(selection.strict)) stringent genes.\nA gene must correlate > 0.6 with at least three other stringent genes; $(length(selection.selected)) pass. The paper reports 262.";fontsize=14,tellwidth=false)
        save(joinpath(ROOT,"results","dsga_diagnostics.png"),fig2;px_per_unit=1.25)

        fig3=Figure(size=(1240,550))
        limits=extrema([s.nodes for s in sensitivity])
        for (col,k) in enumerate((4,1))
            A=[only(s.nodes for s in sensitivity if s.hsm_dimension==r && s.histogram_bins==b && s.power==k) for r in (8,10,12),b in (5,10,20)]
            ax=Axis(fig3[1,col];title="Mapper nodes: filter power $(k)",xlabel="HSM dimension",ylabel="Histogram bins",xticks=(1:3,["8","10","12"]),yticks=(1:3,["5","10","20"]))
            heatmap!(ax,1:3,1:3,A;colormap=:viridis,colorrange=limits)
            for i in 1:3,j in 1:3;text!(ax,i,j;text=string(A[i,j]),color=A[i,j]>26 ? :black : :white,align=(:center,:center),fontsize=22);end
        end
        Label(fig3[2,1:2],"All 18 configurations retain all 308 original samples before singleton removal.\nChanging the filter power preserves sample order but changes equal-width cover memberships.\nThese settings quantify numerical dependence; survival is never used to select a graph or subgroup.";fontsize=14,tellwidth=false)
        save(joinpath(ROOT,"results","sensitivity.png"),fig3;px_per_unit=1.25)
    end
end
