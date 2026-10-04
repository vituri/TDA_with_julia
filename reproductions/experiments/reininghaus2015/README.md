# Reininghaus et al. (CVPR 2015): preserved SHREC14 diagrams

This reproduces the original **synthetic SHREC 2014 empirical benchmark** from
redistributed author-generated HKS persistence diagrams, rather than the unrelated
annuli demonstration or OASIS/NIPS 2015 dataset in `persistence-learning`.

The CSV is pinned to commit `3e217d151d09e5eed213f927960986854331c148` of
[`adaptive_template_systems`](https://github.com/lucho8908/adaptive_template_systems).
The [2019 template-functions paper](https://arxiv.org/html/1902.07190v2),
Section 8.4 and Acknowledgments, identifies the same 300 shapes and 10 HKS settings
and credits Ulrich Bauer for supplying the diagrams. The source notebook maps
IDs 0–299 to 15 identities, 20 poses each. `source-provenance.toml` records sources,
hashes, exclusions, and the evidence for that mapping.

No meshes/HKS are recomputed: original numerical HKS times, mesh preprocessing,
full precision diagrams, 2015 train/test seeds and hyperparameter grids are not
available in this redistribution. This is a reconstruction of the downstream
kernel/classifier/retrieval analysis, with explicit local choices.

## Run from the JuliaTDA monorepo root

```bash
export JULIA_PKG_PRECOMPILE_AUTO=0
# Choose a cache location with adequate free space.
repro_dir=TDA_with_julia/reproductions/experiments/reininghaus2015
export REININGHAUS_CACHE="$repro_dir/.cache"
mkdir -p "$REININGHAUS_CACHE"

julia --compiled-modules=existing --pkgimages=existing "$repro_dir/setup.jl"
julia --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/download.jl"
julia --threads=4 --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/run.jl" --all-classification
julia --threads=4 --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/retrieval_refine.jl"
julia --threads=4 --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/run.jl" --normalized
julia --threads=4 --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/verify.jl"
julia --threads=4 --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/verify.jl" --normalized
julia --compiled-modules=existing --pkgimages=existing \
  --project="$repro_dir/environment" "$repro_dir/figures.jl"
```

`setup.jl` changes only this private environment and develops the local
`TDAPersistenceDiagrams.jl` package. The environment pins LIBSVM 0.8.1 and
CairoMakie 0.14. The commands were executed with Julia 1.12.5. Omit the
`existing` cache flags if using an older Julia release that does not support
them; the workspace flags prevent new native-image precompilation.

`run.jl` defaults to full 10 repetitions at HKS 2 and 10; `--all-classification`
extends classification to all 10 HKS settings. Retrieval always uses all 300 queries
at all 10 settings. `--normalized` writes a separate sensitivity analysis under
`results/normalized/` (default classification targets 2 and 10).

## Protocol and outputs

- All 147,316 finite positive $H_1$ pairs retained, without threshold or subsampling.
  Exact diagonal points (5,576) have zero PSS contribution and are omitted. Negative
  dimension essential records (9,290, including $H_0$) are excluded as in the original
  executable's `--dim 1` selection. The CSV stores capped death coordinates for
  these records; do not mistake them for finite $H_1$ features.
- Heat-time `sigma` follows equation 10. The author executable's `--time T` computes
  `k_(T/4)`; use `T=4sigma` when comparing that executable to our kernel. The
  reflected-point terms and prefactor give exactly the same kernel; there is
  no additional multiplicative constant.
- Local fixed SVM grids: `sigma=2.^(-12:2:16)`, `C=2.^(-5:2:15)`.
  Kernel values are unnormalized in the primary run. Both grids include endpoints;
  some CV winners select the upper C boundary, a documented search-range limit.
- Seed 20150315, 10 new stratified splits, 14 train and 6 test poses per identity.
  The same splits apply to every HKS setting. Inner 10-fold selection uses only
  the 210 training examples, with every class represented in every fold. Ties
  resolve to smallest sigma and C. No HKS setting is selected from test scores.
- Oracle retrieval excludes the query itself. It reports the best score over the
  whole labeled benchmark. This matches the evaluation target of Table 3, but is
  not a held-out generalization estimate. `retrieval_refine.jl` adds half-power
  steps within ± 1.5 units on the log₂ scale of each coarse winner; it never affects SVM selection.
- `cv_grid.csv`, `splits.csv`, and `predictions.csv` permit independent validation
  of hyperparameter selection and test-score accounting. `verify.jl` checks
  equation 10 against independent 256-bit arithmetic, dataset counts, PSD,
  query exclusion and leakage isolation.
- Temporary kernel caches use `REININGHAUS_CACHE`, or the system temporary
  directory if unset. The public commands use an ignored local `.cache/`. They are disposable and are not book resources.
  Our recorded execution instead set `TMPDIR=/mnt/Dados/tmp` to keep temporary
  files on the data disk. Committed CSVs and PNGs allow the chapter to render offline without execution.

The upstream [SHREC dataset page](https://www.cs.cf.ac.uk/shaperetrieval/download.php)
limits original meshes to academic research. We use the publicly redistributed
diagram CSV and preserve its provenance; no original executable or PDF is vendored.
