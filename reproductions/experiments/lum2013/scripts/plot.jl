using CairoMakie, TDAplots

function draw_graph!(ax,model,data; color=:esr1,seed=2034)
    pos=layout_spring(model.M;seed,iterations=300)
    points=Point2f[]
    for e in Graphs.edges(model.M.g)
        push!(points,pos[Graphs.src(e)],pos[Graphs.dst(e)])
    end
    isempty(points) || linesegments!(ax,points;color="#8a949f",linewidth=0.8)
    values=color==:esr1 ? data.esr : data.marker_score
    scatter!(ax,pos;markersize=4 .+ 2sqrt.(length.(model.M.C)),
             color=node_colors(model.M,values),colormap=:viridis,
             colorrange=extrema(values),strokecolor=:white,strokewidth=0.35)
    hidedecorations!(ax);hidespines!(ax)
end

function plot_results(models,summaries,data)
    CairoMakie.activate!()
    with_theme(Theme(fontsize=15,backgroundcolor="#fcfcfa")) do
        fig=Figure(size=(1400,920))
        Label(fig[0,1:3],"GSE2034: a transparent Mapper reconstruction";fontsize=25,font=:bold)
        for (col,name) in enumerate(("log2_r70","log2_r20","outcome_free_r70"))
            model=models[name];s=graph_summary(model.M,data)
            title= name=="outcome_free_r70" ? "Outcome-free control\n70 intervals" : "Relapse + centrality\n$(name=="log2_r70" ? 70 : 20) centrality intervals"
            ax=Axis(fig[1,col];title=title*" · $(s.nodes) nodes / $(s.components) components",aspect=DataAspect())
            draw_graph!(ax,model,data)
            ax2=Axis(fig[2,col];aspect=DataAspect())
            draw_graph!(ax2,model,data;color=:markers)
        end
        Colorbar(fig[3,1:3];vertical=false,colormap=:viridis,colorrange=extrema(data.esr),label="Top row: mean ESR1 log2 intensity (nine annotated probes)",width=Relative(0.70),height=15)
        Colorbar(fig[4,1:3];vertical=false,colormap=:viridis,colorrange=extrema(data.marker_score),label="Bottom row: four named chemokine markers (not the complete KEGG pathway)",width=Relative(0.70),height=15)
        Label(fig[5,1:3],"1,553 highest-variance log2 probes · no gene centering · empirical-rank equalization · gain assumed as width / step = 3\nSingle linkage with 10 histogram bins. Node area increases with membership; patients can appear in several nodes.";fontsize=14,tellwidth=false)
        save(joinpath(ROOT,"results","mapper_comparison.png"),fig;px_per_unit=1.25)
        save(joinpath(ROOT,"results","mapper_comparison.svg"),fig)

        fig2=Figure(size=(1300,650))
        Label(fig2[0,1:3],"Numerical sensitivity: largest non-relapse component";fontsize=24,font=:bold)
        maxsize=maximum(s.largest_nonrelapse_component_patients for s in summaries)
        for (col,nprobes) in enumerate((500,1553,3212)),(row,centered) in enumerate((false,true))
            selected=[s for s in summaries if s.nprobes==nprobes && s.center_genes==centered]
            matrix=[only(s.largest_nonrelapse_component_patients for s in selected if s.resolution==r && s.bins==b) for r in (20,35,70),b in (5,10,20)]
            ax=Axis(fig2[row,col];title="$(nprobes) probes · $(centered ? "gene centered" : "log2 only")",xlabel="Centrality intervals",ylabel="Histogram bins",xticks=(1:3,["20","35","70"]),yticks=(1:3,["5","10","20"]))
            heatmap!(ax,1:3,1:3,matrix;colormap=:viridis,colorrange=(0,maxsize))
            for i in 1:3,j in 1:3;text!(ax,i,j;text=string(matrix[i,j]),align=(:center,:center),color=:white,fontsize=19);end
        end
        Label(fig2[3,1:3],"Every cell covers all 286 patients. Connectivity changes despite the fixed clinical labels.\nThese 54 configurations quantify uncertainty; they are not 54 opportunities to select a favorable biological claim.";fontsize=14,tellwidth=false)
        save(joinpath(ROOT,"results","sensitivity.png"),fig2;px_per_unit=1.25)

        fig3=Figure(size=(1200,590))
        ax=Axis(fig3[1,1];xlabel="L-infinity centrality rank",ylabel="ESR1 mean log2 intensity",title="Expression and centrality, before constructing the graph")
        for er in ("ER+","ER-")
            ids=findall(==(er),data.er)
            scatter!(ax,models["log2_r70"].geometry.rank[ids],data.esr[ids];label=er,markersize=8,color=er=="ER+" ? "#267e75" : "#a84b6d")
        end
        axislegend(ax;position=:rt)
        ax2=Axis(fig3[1,2];xlabel="Deposited overall-relapse group",ylabel="Four-marker mean log2 intensity",title="Clinical ER- patients: descriptive comparison",xticks=(1:2,["Relapse","No relapse"]))
        for (k,r) in enumerate((1,0))
            ids=findall((data.er.=="ER-").&(data.relapse.==r))
            jitter=[0.12sin(i*13.1) for i in eachindex(ids)]
            scatter!(ax2,k .+ jitter,data.marker_score[ids];markersize=8,color=r==1 ? "#a84b6d" : "#267e75")
            scatter!(ax2,[k],[mean(data.marker_score[ids])];marker=:diamond,markersize=18,color=:black)
        end
        Label(fig3[2,1:2],"Clinical ER status defines these groups; it does not reproduce the paper's graph-selected lowERHS membership.\nDiamond = patient mean. Differences are descriptive; no new enrichment or prognostic significance is claimed.";fontsize=14,tellwidth=false)
        save(joinpath(ROOT,"results","patient_diagnostics.png"),fig3;px_per_unit=1.25)
    end
end
