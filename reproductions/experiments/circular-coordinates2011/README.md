# Persistent Cohomology and Circular Coordinates (2011)

Executed reproduction of the noisy circle (§3.3, Figure 2) and torus (§3.7,
Figure 6) in the **full 2011 author manuscript** by de Silva, Morozov and
Vejdemo-Johansson, DOI [10.1007/s00454-011-9344-x](https://doi.org/10.1007/s00454-011-9344-x).
The primary source is <https://mrzv.org/publications/circular/full-dcg/>;
its SHA256 is recorded in `config.toml`. The older arXiv0905.4887v1 describes
400 noisy-circle points, whereas the full 2011 manuscript describes 200.

The chapter is `../../circular-coordinates2011.qmd`. Rendering it does not
execute Julia or access the network. All displayed figures were generated
by this experiment, rather than copied from the paper.

## Run from the JuliaTDA workspace root

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --compiled-modules=existing --pkgimages=existing --startup-file=no \
  TDA_with_julia/reproductions/experiments/circular-coordinates2011/setup.jl

JULIA_PKG_PRECOMPILE_AUTO=0 julia --compiled-modules=existing --pkgimages=existing --startup-file=no \
  --project=TDA_with_julia/reproductions/experiments/circular-coordinates2011/environment \
  TDA_with_julia/reproductions/experiments/circular-coordinates2011/run.jl
```

`setup.jl` activates this private environment and develops the four local
packages. `TDAmapper` is also a transitive local dependency of the plotting
stack. All five local paths in the lock file are relative to `environment/`.
Setup may download Julia dependencies. The numerical run uses generated
data and installed packages and needs no source PDF or network connection.

## Provenance and choices

- `config.toml` records the source URL/hash, the published sizes, noise laws,
  scale choices, our seeds, and all known differences from the paper.
- The consulted publication and author page did not supply the original
  200-point/400-point realizations or random seeds. We generate fresh samples
  with `MersenneTwister(2011)`. Circle phases are independent uniform variables;
  the paper only says points are distributed along the circle. Torus phases
  are independent uniform variables, as specified by its unit-square sampling.
- Noise is positive **Uniform(0, 0.4)** per Cartesian coordinate for the circle,
  and **Uniform(0, 0.2)** for the torus. It is neither Gaussian nor centered.
- The literal torus surface radii inner=1/outer=3 yield major=2/minor=1.
  The complex-size disagreement motivated a second, explicitly labeled
  terminology audit with major=3/minor=1. This reuses exactly the same phases
  and jitter. It is not a confirmed recovery of the author's generator.
- The paper mentions 47 as an example prime without assigning an explicit
  prime to each experiment. We use 47. The local low-persistence circle class
  is the least-persistent finite class alive at 0.14; the original chosen
  representative was not available.
- TDARipserer supplies persistent cocycles. `run.jl` explicitly checks the
  balanced integer lift on every triangle and minimizes the original
  unweighted harmonic energy with Julia sparse QR. The paper used Dionysus
  and LSQR. One potential is fixed to zero per graph component.
- The public `CircularCoordinates` API implements the later Perea2020 method.
  We compare it separately using all 200 points as landmarks, smoothing at
  delta=0.5 and partition-of-unity extension. The original vertex coordinate
  uses delta=0.4. These are different pipelines.
- The local API's `threshold` keyword shadows its `threshold()` function in
  the infinite-death branch. The experiment avoids that branch by first
  computing a finite full-filtration death and deriving admissible coverage,
  then calling the API without the keyword. No library code was changed.
- A downloaded 2012 author-tutorial file contains 1000 circle points; it is
  **not** treated as the paper's 200-point dataset. Its URL and SHA256 are
  retained under `excluded_download` for the audit trail.

`source/` is an ignored research cache for the PDF, extracted text, and a
rendered protocol page. It is not part of the publishable reproduction and
is not required to run it.

## Outputs and validation

`data/` stores Cartesian observations, clean generating phases, and every
jitter value. Clean phases are used only for evaluation. `results/` contains
persistence intervals, every lifted/smoothed edge cochain, vertex potentials,
actual Rips graph cycles, angular metrics, integer-period degree certificates,
simplex counts, and pass/fail scientific checks. `figures/` contains five
newly computed figures, including the radius audit.

Death `Inf` at a finite filtration cutoff means **right-censored**, not an
essential feature of the eventual complete Rips complex. CSV files mark this
explicitly. Torus signs and ordering are basis/gauge choices; compare integer
periods and the unimodular degree matrix, rather than asking for identical
representative labels.

`results/environment.toml` records Julia version, private environment hashes,
Git heads and source-tree SHA256 for the four TDA packages. A dirty working
tree is reported honestly. `results/sha256.csv` hashes the generated data,
results, five figures, scripts, config and private lock files. After editing
the computation or config, rerun to update these provenance records.

The global circle and both torus coordinate pairs have closed integer lifts,
stationary harmonic representatives, preserved primitive integer periods,
and strong angular associations. The low-persistence circle class has a
localized period and a spiky histogram. Its graph has 17 components, so
between-component histogram alignment depends on the chosen gauge.

The primary 45-check experiment was run twice with a byte-identical
`summary.csv`. The extended run adds the second radius interpretation and
passes 65 scientific checks; the chapter reports that final run. This is a seeded procedural and
qualitative reproduction, not a match to the exact published samples,
timings, persistence endpoints, or simplex counts.
