#!/usr/bin/env Rscript
# Export verified signature vectors from the locked RDS into YAML specs.
# Use for buffa51 (and to re-derive any spec from the single source of truth).
#
# Usage: Rscript tools/export_specs_from_rds.R results/signature_gene_vectors.rds
args <- commandArgs(trailingOnly = TRUE)
rds <- if (length(args)) args[1] else "results/signature_gene_vectors.rds"
sigs <- readRDS(rds)
b <- sigs$Buffa
stopifnot(length(b) == 51)
cat("genes:\n"); cat(sprintf("- %s\n", b))
cat("\n# paste into src/hypoxiapipe/signatures/specs/buffa51.yaml under 'genes:',\n")
cat("# then: hypoxiapipe sig hash src/hypoxiapipe/signatures/specs/buffa51.yaml --write\n")
