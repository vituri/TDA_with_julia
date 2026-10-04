# Fasy et al. (2014), Example 13 / Figure 6

Executed qualitative reproduction of the published unit-circle sampling model,
using the local TDARipserer, TDAPersistenceDiagrams and PersistenceInference tools.
The original random realization and unspecified grid/subsample choices are not
claimed to be recovered. The chapter is `../../fasy2014.qmd`.

From the JuliaTDA workspace root:

```bash
julia --startup-file=no TDA_with_julia/reproductions/experiments/fasy2014/setup.jl
julia --startup-file=no --compiled-modules=existing --pkgimages=existing \
  --project=TDA_with_julia/reproductions/experiments/fasy2014 \
  TDA_with_julia/reproductions/experiments/fasy2014/run.jl
julia --startup-file=no --project=TDA_with_julia/reproductions/experiments/fasy2014 \
  TDA_with_julia/reproductions/experiments/fasy2014/provenance.jl
```

Copy `environment.lock.toml` to `Manifest.toml` before setup to restore the
recorded versions. Local package paths in the lock are relative to this folder.
The `existing` flags avoid writing new package precompile caches; they do not
change the numerical protocol.

The density bootstrap finds one significant loop in all 15 seed/grid cases;
the finite-grid Hoeffding bound finds none. Support subsampling finds the loop
at the declared primary b=144 and at b=250, but not b=50 or b=100. There are
18 passing assertions, and all 14 CSV hashes agree across repeated executions.
These comparisons do not establish statistical coverage.

Results include the sample coordinates, both Rips scale conventions, PL density
diagrams, bootstrap/subsampling draws, parameter sensitivity and an independent
population-density comparison. `results/provenance.toml` records manuscript,
environment, numerical-output and local-source checksums.
