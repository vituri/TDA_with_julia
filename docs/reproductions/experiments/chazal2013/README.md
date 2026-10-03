# Chazal et al. (2013): released ToMATo twin-spirals benchmark

This experiment reproduces the two spiral commands in the original authors' [source archive](https://geometrica.saclay.inria.fr/data/ToMATo/ToMATo_code.tar.gz), using the local JuliaTDA `ToMATo.jl` and `MetricSpaces.jl` packages. The [book chapter](../../chazal2013.qmd) contains the analysis and limitations.

The archive supplies **114,562 points with precomputed density**; this is not the exact 10,000-point unit-square realization of the [2013 journal paper](https://doi.org/10.1145/2535927). All source coordinates and density values are preserved. Both original README commands are followed: radius 10 or 25, persistence and peak-height thresholds 0.001. No source class labels exist.

Run from the monorepo root:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no TDA_with_julia/reproductions/experiments/chazal2013/setup.jl
julia --startup-file=no --pkgimages=existing --compiled-modules=existing --threads=2 --project=TDA_with_julia/reproductions/experiments/chazal2013/environment TDA_with_julia/reproductions/experiments/chazal2013/run.jl
julia --startup-file=no --project=TDA_with_julia/reproductions/experiments/chazal2013/environment TDA_with_julia/reproductions/experiments/chazal2013/check.jl
```

The private environment leaves shared environments unchanged. Julia 1.12.5 produced the saved results. A system `tar` command is required. `run.jl` reuses the checked archive, downloading only if absent; hashes are verified before reading it. Source data remain compressed. The archive preserves the authors' GPL license, README and original code; we distribute no extracted original binaries.

- `config.toml`: exact sources, hashes, units, parameters and expected README outcomes.
- `run.jl`: original benchmark using the local Julia implementation, with graph, integrity and result checks.
- `plots.jl`: redraw the figures from saved CSVs, without recomputing clustering.
- `audit.jl`: independent union-find verification; it does not replace plotted library labels.
- `check.jl`: offline result checks and a three-peak saddle with hand-verifiable deaths.
- `results/summary.csv`: graph sizes, cluster sizes, filtering, persistence gap and audit discrepancies.
- `results/labels_radius*.csv`: library and both independent assignments in original row order.
- `results/peaks_radius*.csv` and `death_audit.csv`: raw density persistence and audited deaths.
- `results/threshold_sweep.csv`: joint merging/height-threshold sensitivity.
- `results/provenance.toml` and `runtime.csv`: environment/source fingerprints and observed timing.
- `results/spirals.png` and `persistence.png`: precomputed figures for offline chapter rendering.

The raw library recovers two clusters in both original README settings. An audit exposes equal-density handling and a stale cluster root inside the merge loop: some deaths at infinite threshold differ from exact union-find persistence, and eight final assignments differ at radius 10. At radius 25 final assignments agree exactly. These differences are reported explicitly; no library files are modified.
