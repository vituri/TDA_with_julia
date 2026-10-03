# Lum et al. (2013): GSE2034 Mapper reconstruction

This experiment supports [`../../lum2013.qmd`](../../lum2013.qmd). It uses the
correct GSE2034 expression matrix and **series-level overall-relapse table**,
joined by GSM accession. The sample-level bone-relapse characteristic is not the
outcome used here.

From the JuliaTDA workspace root:

```bash
python3 TDA_with_julia/reproductions/experiments/lum2013/scripts/download.py
julia --project=TDA_with_julia/reproductions/experiments/lum2013/environment \
  -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no \
  --project=TDA_with_julia/reproductions/experiments/lum2013/environment \
  TDA_with_julia/reproductions/experiments/lum2013/scripts/run.jl
```

Add `--no-plots` for a numerical rerun. Python 3.11+ and the `gzip` executable
are required. The copied, privately resolved environment points at local package
directories using relative paths. It imports `TDAmapper` and `TDAplots` directly;
it does not require loading the umbrella package. No shared environment was
modified. `run_metadata.toml` fingerprints the actual local source, including
uncommitted changes, and the private Manifest. The SHA-256 of every original
download is recorded in `data/sources.toml`; changed source bytes cause an error.

The original compressed downloads are present locally and ignored by Git because
they total about 84 MB. The small extracted `data/clinical.tsv` is retained.
`scripts/download.py` redownloads and verifies missing originals and independently
extracts the named clinical table from the SOFT series archive. A deliberate
`--refresh` accepts new upstream bytes; investigate such changes before comparing
them with these recorded results.

The main model is selected before looking at node colors: log2 of the positive
GEO intensities; the top 1,553 probes by sample variance, without gene centering;
Pearson distance; maximal-distance centrality; empirical-rank equalization;
70 centrality intervals and 30 binary-outcome intervals; locally interpreted
gain 3; and single linkage cut at the first empty bin of a ten-bin merge-height
histogram spanning the within-cell distance diameter. The exact GSE2034 probe
list, the original clustering choices, equalization and gain implementation
are not given in sufficient detail by the article/supplement; these substitutions
prevent an exact replication claim. The count 1,553 comes from the NKI example
in Fig. S1 and is an explicitly exploratory choice for GSE2034.

The sweep covers 54 prespecified combinations: 500 / 1,553 / 3,212 probes;
20 / 35 / 70 centrality intervals; 5 / 10 / 20 histogram bins; and no gene
centering / gene centering. The four-marker color averages CCL13, CCL3, CXCL13,
and PF4V1 with equal weight per gene after averaging each gene's probes. It is
not the paper's full KEGG pathway score. ESR1 coloring averages nine probes in
the 2016 GPL96 annotation. Clinical ER- status and the paper's graph-selected
lowERHS patients are distinct definitions.

Outputs include patient-level observables, the exact selected probe list, the
complete sensitivity table, five graph exports (node summaries, membership,
edges and components), figures, and numerical audit metadata. Audits check full
patient coverage, all nerve edges, single linkage against independent threshold
connected components, and Pearson distance against `Statistics.cor`.

## Primary sources

- Lum et al., [Scientific Reports 3, 1236](https://doi.org/10.1038/srep01236).
- [Supplement, Figures S1-S4 and Table S1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fsrep01236/MediaObjects/41598_2013_BFsrep01236_MOESM1_ESM.pdf).
- [GSE2034 deposit](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034),
  expression series matrix, SOFT series archive, and GPL96 annotation URLs in
  `data/sources.toml`.

The GEO summary says 180 non-relapse / 106 relapse, while the attached clinical
table contains 179 / 107. This experiment retains the explicit patient-level
table labels, with no attempt to alter them to match the summary.
