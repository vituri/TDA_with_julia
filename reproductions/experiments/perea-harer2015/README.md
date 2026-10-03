# Perea–Harer (2015): exact signals from Figure 3

This experiment regenerates the fully specified synthetic signals in Section 6.3
of [arXiv:1307.6188v2](https://arxiv.org/html/1307.6188v2), the preprint of Perea and
Harer's 2015 paper, DOI [10.1007/s10208-014-9206-z](https://doi.org/10.1007/s10208-014-9206-z).
There is no observational dataset to download for this target. Its data are the
explicit formulas and evaluation times in `config.toml`; `run.jl` saves their raw
and normalized sliding windows as CSV. This is not a reproduction of the paper's
separate gene-expression classification benchmark.

From the JuliaTDA workspace root:

```sh
julia --startup-file=no TDA_with_julia/reproductions/experiments/perea-harer2015/setup.jl
julia --startup-file=no --project=TDA_with_julia/reproductions/experiments/perea-harer2015/environment TDA_with_julia/reproductions/experiments/perea-harer2015/run.jl
```

`setup.jl` changes only this experiment's environment. The saved Manifest uses
relative paths to local JuliaTDA component packages. Setup/instantiation may need
network access on a fresh machine; the analysis itself does not use the network.

- `data/g1_published.csv`, `data/g2_published.csv`: the exact 151 published windows,
  including the periodically duplicated endpoint at `2π`.
- `data/*observations.csv`, `data/*windows.csv`: the 10-seed permutation and Gaussian
  controls, using 271 observations and 151 ordinary, nonwrapping delay windows.
- `results/published_summary.csv`, `results/g*_F*_H1.csv`: full published-target
  diagrams, maximum persistence, normalized scores, and theorem lower bounds.
- `results/cosine_convergence.csv`: an independent regular-polygon reference.
- `results/window_sensitivity.csv`, `results/controls.csv`: explicitly labeled
  methodological extensions; these were not the paper's published experiments.
- `results/checks.csv`, `results/environment.toml`, `results/sha256.csv`: computed
  consistency checks, executable provenance, and numerical-artifact checksums.
- `crosscheck.jl`, `results/algorithm_crosscheck.csv`: matching principal bars from
  TDARipserer's explicit homology and cohomology reduction modes. Run with the same
  private `--project` argument as `run.jl`; this is not an independent PH engine.
- `results/published_summary_first_run.csv`: retained first-run summary;
  a second complete execution produced a byte-identical published summary.
- `figures/`: our generated persistence diagrams, barcode, and sensitivity figure.
- `data/paper_figure3_reference.png`: original Figure 3 downloaded from the primary
  source, used only for visual comparison; source URL below.

Reference image URL:
https://arxiv.org/html/1307.6188v2/FunctionVsDiagramTDA.png

The main book chapter is `../../perea-harer2015.qmd`. It embeds precomputed figures
and static Julia examples, so Quarto can render it without executing Julia or
accessing the network.
