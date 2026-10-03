# Mapper 2007: Reaven–Miller

This directory accompanies the book chapter
[Reproducing Mapper: the Reaven–Miller experiment](../../../reproduction-mapper2007.qmd).
It contains the verified data, scripts, full 480-configuration grid, and figures.

The analysis was adapted from `JuliaTDA.jl/reproductions/mapper2007/` into the
book's experiment layout. Dataset bytes and numerical configuration are unchanged;
the setup paths and chart labels were adapted for the book. The book version
loads the Mapper and plotting packages directly from the JuliaTDA ecosystem.

From the book root, after preparing the book environment:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no reproductions/experiments/mapper2007/scripts/setup.jl
julia --project=reproductions/experiments/mapper2007 --startup-file=no reproductions/experiments/mapper2007/scripts/download.jl
julia --project=reproductions/experiments/mapper2007 --startup-file=no reproductions/experiments/mapper2007/scripts/run.jl
```

Use `--no-plots` on the last command to skip chart rendering. Base R is optional:
it is used only to rebuild the CSV and compare the two upstream RData objects.
The source revisions and SHA256 checksums are recorded in `data/provenance.toml`.
`environment.lock.toml` records package versions; `scripts/setup.jl` restores it
when the ignored local `Manifest.toml` is absent. Setup prefers the frozen book
packages under `.book-packages/`; it falls back to sibling checkouts when those
snapshots are unavailable.

The result is a **partial qualitative reproduction**: some parameter choices
recover two low-density flares, but the automatic bandwidth gives a chain.
Age is missing from the public data, and the original bandwidth, preprocessing,
and histogram convention remain unknown.
