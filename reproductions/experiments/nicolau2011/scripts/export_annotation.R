# Run by prepare.py. Only plain slots are read: no Biobase installation needed.
args <- commandArgs(trailingOnly = TRUE)
load(args[1])
f <- attr(attr(vanDeVijver, "featureData"), "data")
write.table(f[c("Substance", "Gene", "HUGO.gene.symbol", "NCBI.gene.symbol", "EntrezGene.ID")],
            args[2], sep = "\t", quote = FALSE, row.names = FALSE, na = "")
clinical <- as.data.frame(readxl::read_xls(args[3], skip = 3))
write.table(clinical, args[4], sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
cat("R", as.character(getRversion()), "readxl", as.character(packageVersion("readxl")), "\n")
