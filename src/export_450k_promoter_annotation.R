#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (length(script_path) == 0) {
  script_path <- normalizePath("src/export_450k_promoter_annotation.R", mustWork = FALSE)
}

project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
output_path <- file.path(project_root, "data", "raw", "450k_probe_gene_promoter_map.csv")

required_packages <- c(
  "minfi",
  "IlluminaHumanMethylation450kanno.ilmn12.hg19"
)

missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required Bioconductor package(s): ",
      paste(missing_packages, collapse = ", "),
      "\nInstall with:\n",
      "install.packages(\"BiocManager\")\n",
      "BiocManager::install(c(\"minfi\", \"IlluminaHumanMethylation450kanno.ilmn12.hg19\"))\n"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(minfi)
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
})

annotation <- minfi::getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
promoter_groups <- c("TSS200", "TSS1500", "5'UTR", "1stExon")

split_field <- function(value) {
  value <- as.character(value)
  if (is.na(value) || trimws(value) == "") {
    return(character(0))
  }
  trimws(unlist(strsplit(value, ";", fixed = TRUE), use.names = FALSE))
}

rows <- list()
row_index <- 1L
probe_ids <- if ("Name" %in% colnames(annotation)) {
  as.character(annotation$Name)
} else {
  rownames(annotation)
}

for (i in seq_len(nrow(annotation))) {
  probe <- probe_ids[[i]]
  genes <- split_field(annotation$UCSC_RefGene_Name[[i]])
  groups <- split_field(annotation$UCSC_RefGene_Group[[i]])
  genes <- toupper(genes[genes != ""])
  groups <- groups[groups != ""]

  if (length(genes) == 0 || length(groups) == 0) {
    next
  }

  if (length(genes) == length(groups)) {
    for (j in seq_along(genes)) {
      if (groups[[j]] %in% promoter_groups) {
        rows[[row_index]] <- data.frame(Probe = probe, Gene = genes[[j]], Group = groups[[j]])
        row_index <- row_index + 1L
      }
    }
  } else {
    promoter_hits <- unique(groups[groups %in% promoter_groups])
    if (length(promoter_hits) == 0) {
      next
    }
    for (gene in unique(genes)) {
      for (group in promoter_hits) {
        rows[[row_index]] <- data.frame(Probe = probe, Gene = gene, Group = group)
        row_index <- row_index + 1L
      }
    }
  }
}

if (length(rows) == 0) {
  stop("No promoter probe-to-gene rows were generated.", call. = FALSE)
}

result <- unique(do.call(rbind, rows))
result <- result[order(result$Probe, result$Gene, result$Group), ]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(result, output_path, row.names = FALSE, quote = TRUE, fileEncoding = "UTF-8")

cat("Wrote promoter annotation map:\n")
cat(output_path, "\n")
cat("Rows:", nrow(result), "\n")
