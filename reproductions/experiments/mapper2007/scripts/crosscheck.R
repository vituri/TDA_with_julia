# Optional source-to-source check; needs only base R, no installed packages.
args <- commandArgs(trailingOnly=TRUE)
root <- if (length(args)) args[1] else "."
a <- new.env(); b <- new.env()
load(file.path(root,"data/raw/diabetes.rda"), a)
load(file.path(root,"data/raw/loon_diabetes.rda"), b)
stopifnot(nrow(a$diabetes) == 145L,
          isTRUE(all.equal(as.matrix(a$diabetes[,1:5]),
                           as.matrix(b$diabetes[,1:5]), check.attributes=FALSE)),
          all(as.character(a$diabetes$group) == tolower(as.character(b$diabetes$ClinClass))))
cat("All 725 measurements and 145 labels agree, in the same row order.\n")
