using CairoMakie

const PALETTE = Dict("normal"=>"#29756c", "chemical"=>"#dc9a30", "overt"=>"#aa4563")

function draw_graph!(ax, model; annotation=:density, seed=2007)
    positions = layout_spring(model.M; seed, iterations=300)
    edges = Point2f[]
    for e in Graphs.edges(model.M.g)
        push!(edges, positions[Graphs.src(e)], positions[Graphs.dst(e)])
    end
    !isempty(edges) && linesegments!(ax, edges; color="#718096", linewidth=2)
    sizes = 13 .+ 4sqrt.(length.(model.M.C))
    if annotation == :density
        # Common [0,1] scale after min-max normalization within each pointwise filter.
        values = (node_colors(model.M, model.f) .- minimum(model.f)) ./ (maximum(model.f)-minimum(model.f))
        scatter!(ax, positions; markersize=sizes, color=values, colormap=:jet,
                 colorrange=(0,1), strokecolor=:white, strokewidth=1.5)
    else
        for group in GROUPS
            ids = findall(==(group), node_colors(model.M, model.data_groups))
            !isempty(ids) && scatter!(ax, positions[ids]; markersize=sizes[ids],
                                     color=PALETTE[group], strokecolor=:white, strokewidth=1.5)
        end
    end
    text!(ax, positions; text=string.(length.(model.M.C)), fontsize=12,
          align=(:center,:center), color=:black)
    hidedecorations!(ax); hidespines!(ax)
    autolimits!(ax)
    ax.xautolimitmargin = (0.15,0.15)
    ax.yautolimitmargin = (0.15,0.15)
end

function plot_results(models, selected, rows, data, config)
    CairoMakie.activate!()
    out = joinpath(ROOT,"results")
    theme = Theme(fontsize=15, backgroundcolor="#fcfcfa")
    with_theme(theme) do
        fig = Figure(size=(1450,970))
        Label(fig[0,1:3], "Reaven–Miller Mapper: kernel bandwidth changes the result";
              fontsize=24, font=:bold)
        factors = config["illustration_factors"]
        for (col,factor) in enumerate(factors), (row,k) in enumerate(config["interval_counts"])
            name = "k$(k)_h$(Int(factor))"
            model = models[name]
            s = graph_summary(model.M)
            title = "$(k) intervals | h = $(Int(factor)) × h₀\n$(s.nodes) nodes, $(s.edges) edges | degree ≥ 3 nodes: $(s.branch_nodes)"
            ax = Axis(fig[row,col]; title, aspect=DataAspect())
            draw_graph!(ax,model;seed=config["seed"])
        end
        Colorbar(fig[3,1:3]; colormap=:jet, colorrange=(0,1), vertical=false,
                 label="Relative density within each configuration: blue = low, red = high",
                 height=18, width=Relative(0.70))
        Label(fig[4,1:3], "145 subjects · five standardized measurements · 50% overlap · FirstEmptyBin, 10 bins\n" *
              "h₀ = median nearest-neighbor distance. Number = node size; a subject can occur in multiple nodes.";
              fontsize=14, tellwidth=false)
        save(joinpath(out,"mapper_comparison.png"),fig;px_per_unit=1.25)
        save(joinpath(out,"mapper_comparison.svg"),fig)

        # Explicit use of the ecosystem's public plotting API for all six graphs.
        for row in selected
            model = models[row.name]
            plot = mapper_plot(model.M; node_values=node_colors(model.M,model.f),
                               node_positions=layout_spring(model.M;seed=config["seed"],iterations=300),
                               node_size=13 .+ 4sqrt.(length.(model.M.C)), colormap=:jet)
            Label(plot[0,:], "$(row.intervals) intervals · h/h₀ = $(Int(row.bandwidth_factor))";
                  fontsize=21)
            save(joinpath(out,row.name,"mapper.png"),plot;px_per_unit=1.5)
        end

        fig2 = Figure(size=(1250,680))
        Label(fig2[0,1:2], "Post hoc annotation: classes do not enter the Mapper construction";
              fontsize=22, font=:bold)
        for (col,k) in enumerate(config["interval_counts"])
            model = merge(models["k$(k)_h8"], (; data_groups=data.groups))
            ax = Axis(fig2[1,col];title="$(k) intervals · h = 8h₀ (exploratory configuration)",aspect=DataAspect())
            draw_graph!(ax,model;annotation=:groups,seed=config["seed"])
        end
        elements = [MarkerElement(color=PALETTE[g],marker=:circle,markersize=15) for g in GROUPS]
        Legend(fig2[2,1:2],elements,collect(GROUPS);orientation=:horizontal)
        Label(fig2[3,1:2], "Color = majority class per node; see nodes.csv for composition and membership.csv for subject IDs.\n" *
              "The historical chemical/overt classes are not a direct encoding of diabetes type I/type II.";
              fontsize=14,tellwidth=false)
        save(joinpath(out,"mapper_classes.png"),fig2;px_per_unit=1.25)

        # Sensitivity uses all runs at 50% overlap, not just the six illustrations.
        fig3 = Figure(size=(1330,900))
        Label(fig3[0,1:4], "Sensitivity: nodes of degree ≥ 3 (50% overlap)";
              fontsize=23,font=:bold)
        bins = config["histogram_bins"]
        for (r,normalization) in enumerate(config["normalizations"]),
            (h,histogram_range) in enumerate(config["histogram_ranges"]),
            (q,k) in enumerate(config["interval_counts"])
            col=(h-1)*2+q
            matching = [v for v in rows if v.normalization==normalization &&
                        v.histogram_range==histogram_range && v.intervals==k && v.overlap==0.5]
            matrix = [only(v.branch_nodes for v in matching if v.bandwidth_factor==factor && v.bins==b)
                      for factor in config["bandwidth_factors"], b in bins]
            ax = Axis(fig3[r,col]; title="$(normalization) | $(histogram_range)\n$(k) intervals",
                      xticks=(1:5,string.(config["bandwidth_factors"])),yticks=(1:4,string.(bins)),
                      xlabel="h/h₀",ylabel="bins")
            heatmap!(ax,1:5,1:4,matrix;colormap=:viridis,colorrange=(0,maximum(v.branch_nodes for v in rows)))
            for i in 1:5,j in 1:4
                text!(ax,i,j;text=string(matrix[i,j]),align=(:center,:center),color=:white,fontsize=14)
            end
        end
        Label(fig3[3,1:4], "480 configurations in sensitivity.csv; this figure shows 160 with 50% overlap.\n" *
              "A branch alone does not establish reproduction: check components, cycles and node sizes in the CSV.";
              fontsize=14,tellwidth=false)
        save(joinpath(out,"sensitivity.png"),fig3;px_per_unit=1.25)

        # Raw observables: this is a diagnostic plot, not the paper's projection-pursuit figure.
        fig4 = Figure(size=(1000,650))
        ax = Axis(fig4[1,1];xlabel="Area under the glucose curve",ylabel="Area under the insulin curve",
                  title="Original data: two-measurement diagnostic (without projection pursuit)")
        for group in GROUPS
            ids=findall(==(group),data.groups)
            scatter!(ax,data.measurements[ids,3],data.measurements[ids,4];color=PALETTE[group],label=group,markersize=11)
        end
        Legend(fig4[1,2],ax)
        save(joinpath(out,"data_diagnostic.png"),fig4;px_per_unit=1.25)
    end
end
