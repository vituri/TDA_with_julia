# Nicolau et al. (2011): executed historical-data PAD reconstruction

This experiment supports [`../../nicolau2011.qmd`](../../nicolau2011.qmd).
It uses the actual NKI295 tumors and a historical BCN13 reference retrieved
from the public repository linked by Reuten et al. (Nature Materials 2021).
It is a **partial reconstruction**, with explicit feature-mapping, imputation,
quality-filter and local-clustering substitutions. It does not recover or
assign the published c-MYB-positive patient group.

From the JuliaTDA workspace root:

```bash
# Optional acquisition. Uses Python standard library, R and readxl.
# Existing derived inputs are checked in and suffice for all Julia runs.
python3 TDA_with_julia/reproductions/experiments/nicolau2011/scripts/prepare.py

# A fresh machine must instantiate this private environment once.
JULIA_PKG_PRECOMPILE_AUTO=0 julia \
  --project=TDA_with_julia/reproductions/experiments/nicolau2011/environment \
  -e 'using Pkg; Pkg.instantiate()'

JULIA_PKG_PRECOMPILE_AUTO=0 julia \
  --compiled-modules=existing --pkgimages=existing \
  --project=TDA_with_julia/reproductions/experiments/nicolau2011/environment \
  TDA_with_julia/reproductions/experiments/nicolau2011/scripts/run.jl
```

Add `--no-plots` for numerical outputs. The `gzip` executable is required.
The Manifest uses relative paths to local JuliaTDA components. No shared
environment or library is modified. The numerical run fixes BLAS to one thread;
the graph drawing fixes its spring-layout seed to 2011.

## Data contract

- `seventyGeneData_1.40.0.tar.gz` preserves the original
  `ZipFiles295Samples.zip`, six expression tables, `nejm_table1.zip`, original
  clinical XLS, and annotated vanDeVijver ExpressionSet. The ExpressionSet is
  read only for probe annotation; expression and clinical values come from
  the original archived files.
- GEO GPL2567 supplies an independent universe of 24,479 assay identifiers.
  The NKI tables include another 17 controls. Treating flag zero as missing
  and requiring 70% valid measurements additionally excludes 26 assay probes,
  leaving 24,453. The paper instead reports 24,479 rows with 70% data.
- The retained tumor matrix contains 36,476 missing values. Deterministic
  gene-oriented KNN (ten complete donor genes, observed-patient Euclidean
  distance, inverse-distance weights) imputes 10,077 affected rows. This is
  an explicit unpartitioned local rule, not a claim of exact historical KNN.
- NKI values are base-ten log ratios and are converted by multiplication by
  `log2(10)`; they are not logged again. Probe rows are mean-collapsed by archived
  HUGO symbol. Empty normal symbols and symbols associated with multiple
  normal UniGenes are rejected before an exact symbol join.
- The pinned BCN13 archive contains complete `BCN.ugc219.pcl`, 18,971 build-219
  clusters, and the original SMD quality-filter report with 32,644 clones.
  Archived identifiers indicate ten `BC.N.*` and three `BC.NRm.*` samples.
  The 2011 text instead describes nine and four. All 13 exact archived sample
  identifiers are retained, with the discrepancy documented.
- The common dataset has 8,940 symbols. The historical NKI-to-UniGene build-219
  mapping needed for the paper's 12,237 common UniGenes was not recovered.
  `results/gene_alignment.csv` records every tumor probe and BCN UniGene used.
- Clinical data are joined by unique integer `SampleID`, never by row position.
  Clinical labels/follow-up do not enter model fitting, gene selection, distances,
  filters or Mapper. They are only exported and used for post-construction ER
  coloring. The table contains 226 ER-positive patients and 79 observed deaths.

`data/sources.json` records URLs, source archive members, full SHA-256 hashes,
the BCN repository commit, and Python/R/readxl versions. Downloads are reused
only if their hashes match a previous manifest. Large original archives are
present locally in ignored `data/raw/`; the derived Julia inputs total about
14 MB and are retained. `data/checksums.toml` verifies those inputs before each
analysis. The acquisition script writes deterministic gzip headers.

## Statistical and topological contract

The common tumor and normal columns are normalized to the mean normal-column
Euclidean magnitude. FLAT fits each normal column to the remaining normals
without an intercept. A thin SVD supplies the ten-dimensional HSM. Tumor
components are orthogonal projections/residuals. Each normal residual comes
from an outer leave-one-out HSM fit, again with ten dimensions. Wold invariants
are exported for all admissible dimensions; dimension 10 is the paper's fixed
choice rather than a locally tuned optimum.

Feature selection uses max(abs(q05),abs(q95)) of each gene's tumor residuals.
A gene must exceed the 85th-percentile threshold and correlate >0.6 (Pearson)
with at least three **other** genes exceeding the 98th-percentile threshold.
Self-correlation is excluded. These correlation conventions are explicit where
the historical text is less specific. The local reference selects 98 genes.

The sample distance is `1-cor` over selected residual genes. The reference
filter is the fourth power of the Euclidean residual norm. Mapper uses 15
equal-width intervals, 80% overlap, closed endpoints, and the exact intersection
nerve. Local clustering uses single linkage cut at the JuliaTDA first-empty-bin
threshold (ten bins from minimum merge height to cell distance diameter, with
the diameter included in the last bin). The original clustering settings were
not given in the recovered publication/supplement.

The 18 prespecified settings cross HSM dimensions 8/10/12, filter powers 1/4,
and clustering histograms 5/10/20. Every setting covers all 308 original samples.
The baseline has 26 nodes, 89 edges, and two components; the largest component
contains 307 samples. Removing singleton nodes leaves 22 nodes, 78 edges and
307 samples in one component; only Sample 282 loses all membership. These 22
graph nodes are not the published 22 c-MYB-positive tumors. No local subgroup
or favorable survival claim is selected by parameter tuning.

## Evidence

The run executes twelve numerical integrity checks plus independent full-nerve,
threshold-connected-component and Pearson-distance audits. It exports aligned
genes, selection statistics, sample observables, both graph membership tables,
nodes and edges, all 18 sensitivity settings, Wold diagnostics, and figures.
`results/run_metadata.toml` fingerprints the private Manifest, inputs, scripts,
and actual local source trees with Git revisions and dirty status.
`results/determinism.toml` records a second complete run's CSV hash comparison.

The paper's `Dataset S1` residual matrix was not found among recovered PMC
attachments. Without that matrix, the historical tumor feature mapping and
subgroup membership, we cannot independently reproduce the exact 262-gene
graph or the published survival result. All these limits are stated in the
chapter.

## Primary sources

- Nicolau, Levine and Carlsson (2011),
  [PNAS 108:7265–7270](https://doi.org/10.1073/pnas.1102826108), and its
  [12-page supporting information](https://pmc.ncbi.nlm.nih.gov/articles/instance/3084136/bin/1102826108_pnas.201102826SI.pdf).
- Nicolau et al. (2007),
  [DSGA](https://doi.org/10.1093/bioinformatics/btm033).
- van de Vijver et al. (2002),
  [NKI295](https://doi.org/10.1056/NEJMoa021967).
- [seventyGeneData maintainer's acquisition vignette](https://bioconductor.posit.co/packages/3.19/data/experiment/vignettes/seventyGeneData/inst/doc/seventyGeneData.html).
- [GPL2567 at GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL2567).
- Reuten et al. (2021),
  [Nature Materials](https://doi.org/10.1038/s41563-020-00894-0), and its
  [public historical normal-reference/DSGA repository](https://github.com/monkgroupie/publication_code/tree/2914487d9138dce2175f3a1e7918277187a11b87).
