#!/usr/bin/env python3
"""Preprocess TCGA-KIRC mutation, expression, and methylation features."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MUTATION_PATH = RAW_DIR / "KIRC_mc3.txt"
EXPRESSION_PATH = RAW_DIR / "HiSeqV2"
METHYLATION_PATH = RAW_DIR / "HumanMethylation450"
PROMOTER_MAP_PATH = RAW_DIR / "450k_probe_gene_promoter_map.csv"

MUTATION_OUT = PROCESSED_DIR / "KIRC_mutation_gene_feature.csv"
EXPRESSION_OUT = PROCESSED_DIR / "KIRC_expression_gene_feature.csv"
METHYLATION_OUT = PROCESSED_DIR / "KIRC_methylation_gene_feature.csv"
MULTIOMICS_OUT = PROCESSED_DIR / "KIRC_multiomics_3omics.csv"
AVAILABILITY_OUT = PROCESSED_DIR / "KIRC_multiomics_availability.csv"
QC_OUT = PROCESSED_DIR / "KIRC_3omics_QC_report.txt"

FUNCTIONAL_EFFECTS = {
    "missense_mutation",
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "in_frame_del",
    "in_frame_ins",
    "translation_start_site",
    "nonstop_mutation",
    "large deletion",
}

INVALID_GENE_SYMBOLS = {
    "",
    "?",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
}


@dataclass
class RunMetrics:
    mutation_total_samples: int = 0
    mutation_type_counts: dict[str, int] = field(default_factory=dict)
    expression_tumor_samples: int = 0
    expression_normal_samples: int = 0
    expression_type_counts: dict[str, int] = field(default_factory=dict)
    methylation_tumor_samples: int = 0
    methylation_normal_samples: int = 0
    methylation_type_counts: dict[str, int] = field(default_factory=dict)
    methylation_raw_probe_count: int = 0
    methylation_matched_probe_count: int = 0


class PreprocessingError(RuntimeError):
    """Raised when source data or processed output fails validation."""


def sample_type(sample: str) -> str | None:
    parts = str(sample).split("-")
    if len(parts) < 4 or len(parts[3]) < 2:
        return None
    return parts[3][:2]


def clean_gene(series: pd.Series) -> pd.Series:
    """
    Normalize gene symbols and remove invalid Xena identifiers.

    Examples:
    - " vhl " -> "VHL"
    - "TP53|7157" -> "TP53"
    - "?|100130426" -> missing
    - "?" -> missing

    LOC-prefixed provisional symbols are retained.
    """
    cleaned = series.astype("string").str.strip().str.upper()
    cleaned = cleaned.str.split("|", n=1, regex=False).str[0].str.strip()
    invalid = cleaned.isna() | cleaned.isin(INVALID_GENE_SYMBOLS)
    return cleaned.mask(invalid, pd.NA)


def min_max(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        raise PreprocessingError("Cannot min-max normalize an empty or all-NaN series.")
    if math.isclose(float(max_value), float(min_value)):
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return handle.readline().rstrip("\n\r").split("\t")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def sample_columns_by_type(header: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
    columns = header[1:]
    type_counts: dict[str, int] = {}
    tumor: list[str] = []
    normal: list[str] = []
    for column in columns:
        stype = sample_type(column)
        if stype is None:
            stype = "UNKNOWN"
        type_counts[stype] = type_counts.get(stype, 0) + 1
        if stype == "01":
            tumor.append(column)
        elif stype == "11":
            normal.append(column)
    return tumor, normal, type_counts


def ensure_raw_structure(metrics: RunMetrics) -> None:
    required = [MUTATION_PATH, EXPRESSION_PATH, METHYLATION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise PreprocessingError(f"Missing raw file(s): {', '.join(missing)}")

    mutation_header = read_header(MUTATION_PATH)
    required_mutation_columns = {"sample", "gene", "effect"}
    if not required_mutation_columns.issubset(set(mutation_header)):
        raise PreprocessingError(
            "KIRC_mc3.txt does not contain required columns: sample, gene, effect"
        )

    mutation_samples = pd.read_csv(MUTATION_PATH, sep="\t", usecols=["sample"], dtype=str)
    mutation_samples["SampleType"] = mutation_samples["sample"].map(sample_type)
    metrics.mutation_type_counts = {
        key: int(value)
        for key, value in mutation_samples.drop_duplicates("sample")["SampleType"]
        .fillna("UNKNOWN")
        .value_counts()
        .sort_index()
        .items()
    }
    metrics.mutation_total_samples = int(
        mutation_samples.loc[mutation_samples["SampleType"] == "01", "sample"].nunique()
    )
    if metrics.mutation_total_samples == 0:
        raise PreprocessingError("No 01 tumor samples were found in mutation data.")

    expression_header = read_header(EXPRESSION_PATH)
    if not expression_header or expression_header[0] != "sample":
        raise PreprocessingError("HiSeqV2 first column must be named 'sample'.")
    expression_tumor, expression_normal, expression_counts = sample_columns_by_type(expression_header)
    metrics.expression_tumor_samples = len(expression_tumor)
    metrics.expression_normal_samples = len(expression_normal)
    metrics.expression_type_counts = expression_counts
    if metrics.expression_tumor_samples == 0:
        raise PreprocessingError("No 01 tumor samples were found in expression data.")
    if metrics.expression_normal_samples == 0:
        raise PreprocessingError("No 11 normal samples were found in expression data.")

    methylation_header = read_header(METHYLATION_PATH)
    if not methylation_header or methylation_header[0] != "sample":
        raise PreprocessingError("HumanMethylation450 first column must be named 'sample'.")
    methylation_tumor, methylation_normal, methylation_counts = sample_columns_by_type(methylation_header)
    metrics.methylation_tumor_samples = len(methylation_tumor)
    metrics.methylation_normal_samples = len(methylation_normal)
    metrics.methylation_type_counts = methylation_counts
    if metrics.methylation_tumor_samples == 0:
        raise PreprocessingError("No 01 tumor samples were found in methylation data.")
    if metrics.methylation_normal_samples == 0:
        raise PreprocessingError("No 11 normal samples were found in methylation data.")


def process_mutation(metrics: RunMetrics) -> pd.DataFrame:
    df = pd.read_csv(MUTATION_PATH, sep="\t", usecols=["sample", "gene", "effect"], dtype=str)
    df["SampleType"] = df["sample"].map(sample_type)
    df = df[df["SampleType"] == "01"].copy()
    df["Gene"] = clean_gene(df["gene"])
    df = df[df["Gene"].notna() & (df["Gene"] != "")]
    df["effect_key"] = df["effect"].astype("string").str.strip().str.lower()
    df = df[df["effect_key"].isin(FUNCTIONAL_EFFECTS)]
    df = df.drop_duplicates(["sample", "Gene"])

    grouped = (
        df.groupby("Gene", as_index=False)["sample"]
        .nunique()
        .rename(columns={"sample": "MutatedSampleCount"})
    )
    grouped["TotalMutationSamples"] = metrics.mutation_total_samples
    grouped["Mutation"] = grouped["MutatedSampleCount"] / metrics.mutation_total_samples
    grouped = grouped.sort_values("Gene").reset_index(drop=True)
    grouped.to_csv(MUTATION_OUT, index=False, encoding="utf-8")

    print(f"Mutation tumor samples: {metrics.mutation_total_samples}")
    print(f"Functional mutated genes: {len(grouped)}")
    print("Top 10 Mutation genes:")
    print(grouped.sort_values("Mutation", ascending=False).head(10).to_string(index=False))
    print(grouped["Mutation"].describe().to_string())
    return grouped


def process_expression(metrics: RunMetrics) -> pd.DataFrame:
    header = read_header(EXPRESSION_PATH)
    tumor_cols, normal_cols, _ = sample_columns_by_type(header)
    usecols = [header[0], *tumor_cols, *normal_cols]
    df = pd.read_csv(EXPRESSION_PATH, sep="\t", usecols=usecols, dtype={header[0]: str})
    df = df.rename(columns={header[0]: "Gene"})
    df["Gene"] = clean_gene(df["Gene"])
    df = df[df["Gene"].notna() & (df["Gene"] != "")].copy()

    numeric_cols = tumor_cols + normal_cols
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.copy()
    df["TumorMean"] = df[tumor_cols].mean(axis=1, skipna=True)
    df["NormalMean"] = df[normal_cols].mean(axis=1, skipna=True)
    df["ExpressionSigned"] = df["TumorMean"] - df["NormalMean"]
    df["ExpressionAbsRaw"] = df["ExpressionSigned"].abs()

    grouped = (
        df.groupby("Gene", as_index=False)[
            ["TumorMean", "NormalMean", "ExpressionSigned", "ExpressionAbsRaw"]
        ]
        .mean()
        .sort_values("Gene")
        .reset_index(drop=True)
    )
    grouped["Expression"] = min_max(grouped["ExpressionAbsRaw"])

    invalid_gene_mask = (
        grouped["Gene"].isna()
        | grouped["Gene"].astype("string").str.startswith("?", na=False)
        | grouped["Gene"].astype("string").str.contains("|", regex=False, na=False)
    )
    if invalid_gene_mask.any():
        examples = grouped.loc[invalid_gene_mask, "Gene"].head(10).tolist()
        raise PreprocessingError(
            f"Invalid Gene values remain after expression cleaning: {examples}"
        )
    if grouped["Gene"].duplicated().any():
        duplicates = grouped.loc[grouped["Gene"].duplicated(), "Gene"].head(10).tolist()
        raise PreprocessingError(
            f"Duplicate Gene values remain after expression cleaning: {duplicates}"
        )

    grouped.to_csv(EXPRESSION_OUT, index=False, encoding="utf-8")

    print(f"Expression tumor samples: {metrics.expression_tumor_samples}")
    print(f"Expression normal samples: {metrics.expression_normal_samples}")
    print(f"Expression genes: {len(grouped)}")
    print("Top 10 ExpressionAbsRaw genes:")
    print(grouped.sort_values("ExpressionAbsRaw", ascending=False).head(10).to_string(index=False))
    print(grouped["Expression"].describe().loc[["min", "max"]].to_string())
    return grouped


def validate_promoter_map() -> pd.DataFrame:
    if not PROMOTER_MAP_PATH.exists():
        raise PreprocessingError(
            "Missing official 450K promoter annotation map: "
            f"{PROMOTER_MAP_PATH}\n"
            "Rscript is required to generate it from Bioconductor package "
            "IlluminaHumanMethylation450kanno.ilmn12.hg19."
        )
    mapping = pd.read_csv(PROMOTER_MAP_PATH, dtype=str)
    required = {"Probe", "Gene", "Group"}
    if not required.issubset(mapping.columns):
        raise PreprocessingError("450K promoter map must contain Probe, Gene, Group columns.")
    mapping = mapping[["Probe", "Gene", "Group"]].copy()
    mapping["Probe"] = mapping["Probe"].astype("string").str.strip()
    mapping["Gene"] = clean_gene(mapping["Gene"])
    mapping["Group"] = mapping["Group"].astype("string").str.strip()
    mapping = mapping[
        mapping["Probe"].notna()
        & (mapping["Probe"] != "")
        & mapping["Gene"].notna()
        & (mapping["Gene"] != "")
    ].drop_duplicates(["Probe", "Gene"])
    if mapping.empty:
        raise PreprocessingError("450K promoter map is empty after validation.")
    return mapping


def try_generate_promoter_map() -> None:
    if PROMOTER_MAP_PATH.exists():
        return
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise PreprocessingError(
            "Rscript is not available, and the promoter map does not exist.\n"
            "Install R and Bioconductor packages, then run:\n"
            "install.packages(\"BiocManager\")\n"
            "BiocManager::install(c(\"minfi\", \"IlluminaHumanMethylation450kanno.ilmn12.hg19\"))\n"
            "Rscript src/export_450k_promoter_annotation.R"
        )
    script = PROJECT_ROOT / "src" / "export_450k_promoter_annotation.R"
    result = subprocess.run([rscript, str(script)], cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise PreprocessingError(
            "Failed to generate 450K promoter map with Rscript.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def process_methylation(metrics: RunMetrics, chunksize: int) -> pd.DataFrame:
    try_generate_promoter_map()
    mapping = validate_promoter_map()

    header = read_header(METHYLATION_PATH)
    tumor_cols, normal_cols, _ = sample_columns_by_type(header)
    usecols = [header[0], *tumor_cols, *normal_cols]
    merged_chunks: list[pd.DataFrame] = []
    raw_probe_count = 0
    matched_probes: set[str] = set()

    for chunk in pd.read_csv(METHYLATION_PATH, sep="\t", usecols=usecols, chunksize=chunksize):
        chunk = chunk.rename(columns={header[0]: "Probe"})
        chunk["Probe"] = chunk["Probe"].astype("string").str.strip()
        raw_probe_count += len(chunk)
        numeric_cols = tumor_cols + normal_cols
        chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        chunk = chunk.copy()
        chunk["TumorBetaMean"] = chunk[tumor_cols].mean(axis=1, skipna=True)
        chunk["NormalBetaMean"] = chunk[normal_cols].mean(axis=1, skipna=True)
        chunk["DeltaBeta"] = chunk["TumorBetaMean"] - chunk["NormalBetaMean"]
        chunk["AbsDeltaBeta"] = chunk["DeltaBeta"].abs()

        probe_features = chunk[["Probe", "DeltaBeta", "AbsDeltaBeta"]]
        merged = probe_features.merge(mapping, on="Probe", how="inner")
        if not merged.empty:
            matched_probes.update(merged["Probe"].dropna().unique().tolist())
            merged_chunks.append(merged[["Gene", "Probe", "DeltaBeta", "AbsDeltaBeta"]])

    metrics.methylation_raw_probe_count = raw_probe_count
    metrics.methylation_matched_probe_count = len(matched_probes)
    if not merged_chunks:
        raise PreprocessingError("Methylation probes could not be matched to promoter annotation.")

    merged_all = pd.concat(merged_chunks, ignore_index=True)
    grouped = (
        merged_all.groupby("Gene", as_index=False)
        .agg(
            MethylationSigned=("DeltaBeta", "median"),
            MethylationAbsRaw=("AbsDeltaBeta", "median"),
            PromoterProbeCount=("Probe", "nunique"),
        )
        .sort_values("Gene")
        .reset_index(drop=True)
    )
    grouped["Methylation"] = min_max(grouped["MethylationAbsRaw"])
    grouped.to_csv(METHYLATION_OUT, index=False, encoding="utf-8")

    print(f"Methylation tumor samples: {metrics.methylation_tumor_samples}")
    print(f"Methylation normal samples: {metrics.methylation_normal_samples}")
    print(f"Raw methylation probes: {metrics.methylation_raw_probe_count}")
    print(f"Matched promoter probes: {metrics.methylation_matched_probe_count}")
    print(f"Methylation genes: {len(grouped)}")
    print("Top 10 MethylationAbsRaw genes:")
    print(grouped.sort_values("MethylationAbsRaw", ascending=False).head(10).to_string(index=False))
    print(grouped["Methylation"].describe().loc[["min", "max"]].to_string())
    return grouped


def merge_multiomics() -> tuple[pd.DataFrame, pd.DataFrame]:
    mutation = pd.read_csv(MUTATION_OUT, usecols=["Gene", "Mutation"])
    expression = pd.read_csv(EXPRESSION_OUT, usecols=["Gene", "Expression"])
    methylation = pd.read_csv(METHYLATION_OUT, usecols=["Gene", "Methylation"])

    for frame in [mutation, expression, methylation]:
        frame["Gene"] = clean_gene(frame["Gene"])
        frame.drop(frame[frame["Gene"].isna() | (frame["Gene"] == "")].index, inplace=True)
        if frame["Gene"].duplicated().any():
            duplicates = frame.loc[frame["Gene"].duplicated(), "Gene"].head(10).tolist()
            raise PreprocessingError(f"Duplicate Gene values found before merge: {duplicates}")

    merged = mutation.merge(expression, on="Gene", how="outer").merge(methylation, on="Gene", how="outer")
    merged["Gene"] = clean_gene(merged["Gene"])
    merged = merged[merged["Gene"].notna()].copy()

    availability = pd.DataFrame(
        {
            "Gene": merged["Gene"],
            "MutationAvailable": merged["Mutation"].notna().astype(int),
            "ExpressionAvailable": merged["Expression"].notna().astype(int),
            "MethylationAvailable": merged["Methylation"].notna().astype(int),
        }
    )
    final = merged[["Gene", "Mutation", "Expression", "Methylation"]].copy()
    final[["Mutation", "Expression", "Methylation"]] = final[
        ["Mutation", "Expression", "Methylation"]
    ].fillna(0)
    final = final.sort_values("Gene").reset_index(drop=True)
    availability = availability.sort_values("Gene").reset_index(drop=True)

    validate_final(final)
    final.to_csv(MULTIOMICS_OUT, index=False, encoding="utf-8")
    availability.to_csv(AVAILABILITY_OUT, index=False, encoding="utf-8")
    return final, availability


def validate_final(final: pd.DataFrame) -> None:
    if final.empty:
        raise PreprocessingError("Final multiomics file has zero genes.")

    cleaned_genes = clean_gene(final["Gene"])
    invalid_gene_mask = (
        cleaned_genes.isna()
        | cleaned_genes.astype("string").str.startswith("?", na=False)
        | cleaned_genes.astype("string").str.contains("|", regex=False, na=False)
    )
    if invalid_gene_mask.any():
        examples = final.loc[invalid_gene_mask, "Gene"].head(10).tolist()
        raise PreprocessingError(
            f"Final multiomics file contains invalid Gene values: {examples}"
        )

    if cleaned_genes.duplicated().any():
        duplicates = cleaned_genes[cleaned_genes.duplicated()].head(10).tolist()
        raise PreprocessingError(
            f"Final multiomics file contains duplicate Gene values: {duplicates}"
        )

    values = final[["Mutation", "Expression", "Methylation"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any():
        raise PreprocessingError("Final features contain NaN or non-numeric values.")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise PreprocessingError("Final features contain infinite values.")
    if not ((values >= 0) & (values <= 1)).all().all():
        raise PreprocessingError("Final features are outside the 0 to 1 range.")


def write_qc_report(metrics: RunMetrics, final: pd.DataFrame, availability: pd.DataFrame) -> None:
    mutation = pd.read_csv(MUTATION_OUT)
    expression = pd.read_csv(EXPRESSION_OUT)
    methylation = pd.read_csv(METHYLATION_OUT)

    features = final[["Mutation", "Expression", "Methylation"]].apply(
        pd.to_numeric, errors="coerce"
    )
    cleaned_genes = clean_gene(final["Gene"])
    invalid_gene_mask = (
        cleaned_genes.isna()
        | cleaned_genes.astype("string").str.startswith("?", na=False)
        | cleaned_genes.astype("string").str.contains("|", regex=False, na=False)
    )
    invalid_gene_count = int(invalid_gene_mask.sum())
    zero_props = (features == 0).mean()
    missing_filled = {
        "Mutation": int((availability["MutationAvailable"] == 0).sum()),
        "Expression": int((availability["ExpressionAvailable"] == 0).sum()),
        "Methylation": int((availability["MethylationAvailable"] == 0).sum()),
    }

    lines: list[str] = []
    lines.append("TCGA-KIRC 3-omics preprocessing QC report")
    lines.append("=" * 60)
    lines.append("Raw files:")
    for path in [MUTATION_PATH, EXPRESSION_PATH, METHYLATION_PATH]:
        lines.append(f"- {path}: {file_size_mb(path):.2f} MB")
    lines.append("")
    lines.append("Sample counts:")
    lines.append(f"- Mutation 01 tumor samples: {metrics.mutation_total_samples}")
    lines.append(f"- Mutation sample types: {metrics.mutation_type_counts}")
    lines.append(f"- Expression 01 tumor samples: {metrics.expression_tumor_samples}")
    lines.append(f"- Expression 11 normal samples: {metrics.expression_normal_samples}")
    lines.append(f"- Expression sample types: {metrics.expression_type_counts}")
    lines.append(f"- Methylation 01 tumor samples: {metrics.methylation_tumor_samples}")
    lines.append(f"- Methylation 11 normal samples: {metrics.methylation_normal_samples}")
    lines.append(f"- Methylation sample types: {metrics.methylation_type_counts}")
    lines.append("")
    lines.append("Intermediate gene counts:")
    lines.append(f"- Mutation genes: {len(mutation)}")
    lines.append(f"- Expression genes: {len(expression)}")
    lines.append(f"- Methylation genes: {len(methylation)}")
    lines.append(f"- Final merged genes: {len(final)}")
    lines.append("")
    lines.append("Missing values filled with 0:")
    for key, value in missing_filled.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Feature statistics:")
    lines.append(features.describe().to_string())
    lines.append("")
    lines.append("Zero proportions:")
    lines.append(zero_props.to_string())
    lines.append("")
    lines.append(f"Contains NaN: {features.isna().any().any()}")
    lines.append(f"Contains infinite values: {np.isinf(features.to_numpy(dtype=float)).any()}")
    lines.append(f"Contains duplicate Gene: {cleaned_genes.duplicated().any()}")
    lines.append(f"Invalid Gene count: {invalid_gene_count}")
    lines.append(f"All features in [0, 1]: {((features >= 0) & (features <= 1)).all().all()}")
    lines.append("")
    lines.append("Methylation probe counts:")
    raw_probe_text = (
        str(metrics.methylation_raw_probe_count)
        if metrics.methylation_raw_probe_count > 0
        else "not recalculated in this run"
    )
    matched_probe_text = (
        str(metrics.methylation_matched_probe_count)
        if metrics.methylation_matched_probe_count > 0
        else "not recalculated in this run"
    )
    lines.append(f"- Raw probes: {raw_probe_text}")
    lines.append(f"- Matched promoter probes: {matched_probe_text}")
    lines.append("")
    lines.append("Final file first 10 rows:")
    lines.append(final.head(10).to_string(index=False))
    lines.append("")
    lines.append("Top 10 Mutation genes:")
    lines.append(final.sort_values("Mutation", ascending=False).head(10).to_string(index=False))
    lines.append("")
    lines.append("Top 10 Expression genes:")
    lines.append(final.sort_values("Expression", ascending=False).head(10).to_string(index=False))
    lines.append("")
    lines.append("Top 10 Methylation genes:")
    lines.append(final.sort_values("Methylation", ascending=False).head(10).to_string(index=False))
    lines.append("")

    QC_OUT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "mutation", "expression", "methylation", "merge", "qc"],
        default=["all"],
        help="Processing steps to run. Default: all.",
    )
    parser.add_argument("--chunksize", type=int, default=1000, help="Methylation chunk size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = set(args.steps)
    if "all" in steps:
        steps = {"mutation", "expression", "methylation", "merge", "qc"}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics = RunMetrics()

    try:
        ensure_raw_structure(metrics)
        if "mutation" in steps:
            process_mutation(metrics)
        if "expression" in steps:
            process_expression(metrics)
        if "methylation" in steps:
            process_methylation(metrics, chunksize=args.chunksize)
        final: pd.DataFrame | None = None
        availability: pd.DataFrame | None = None
        if "merge" in steps:
            final, availability = merge_multiomics()
            print(f"Final merged genes: {len(final)}")
        if "qc" in steps:
            if final is None or availability is None:
                final = pd.read_csv(MULTIOMICS_OUT)
                availability = pd.read_csv(AVAILABILITY_OUT)
                validate_final(final)
            write_qc_report(metrics, final, availability)

        print("Output files:")
        for path in [
            MUTATION_OUT,
            EXPRESSION_OUT,
            METHYLATION_OUT,
            MULTIOMICS_OUT,
            AVAILABILITY_OUT,
            QC_OUT,
        ]:
            if path.exists():
                print(path)
        return 0
    except PreprocessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
