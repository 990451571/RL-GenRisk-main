"""Utilities for frozen low-frequency evidence reward inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mutation_frequency import (
    TOTAL_KIRC_TUMOR_SAMPLES,
    classify_mutation_frequency,
    mutation_frequency_pct,
)


REQUIRED_COLUMNS_V1 = [
    "Gene",
    "MutationPatientCount",
    "MutationFrequency",
    "MutationRarityScore",
    "ExpressionSupport",
    "MethylationSupport",
    "CNVFunctionalSupport",
    "NonMutationOmicsSupport",
    "RawNetworkSupport",
    "Degree",
    "DegreeBin",
    "DegreeCorrectedNetworkSupport",
    "LowFrequencyEvidenceScore",
    "EvidenceWithoutRarity",
    "EvidenceWithoutOmics",
    "EvidenceWithoutNetwork",
    "CNVAvailable",
    "ExpressionDirectionAvailable",
    "MethylationDirectionAvailable",
]

REQUIRED_COLUMNS_V2 = [
    "Gene",
    "MutationPatientCount",
    "MutationFrequency",
    "MutationRarityScore",
    "ExpressionSupport",
    "MethylationSupport",
    "CNVFunctionalSupport",
    "NonMutationOmicsSupport",
    "RawNetworkSupport",
    "Degree",
    "DegreeBinV2",
    "DegreeCorrectedNetworkSupportV2",
    "LowFrequencyEvidenceScoreV2",
    "EvidenceWithoutRarityV2",
    "EvidenceWithoutOmicsV2",
    "EvidenceWithoutNetworkV2",
    "CNVAvailable",
    "ExpressionDirectionAvailable",
    "MethylationDirectionAvailable",
]

REQUIRED_COLUMNS = REQUIRED_COLUMNS_V1

SCORE_COLUMNS_V1 = [
    "MutationRarityScore",
    "ExpressionSupport",
    "MethylationSupport",
    "CNVFunctionalSupport",
    "NonMutationOmicsSupport",
    "RawNetworkSupport",
    "DegreeCorrectedNetworkSupport",
    "LowFrequencyEvidenceScore",
    "EvidenceWithoutRarity",
    "EvidenceWithoutOmics",
    "EvidenceWithoutNetwork",
]

SCORE_COLUMNS_V2 = [
    "MutationRarityScore",
    "ExpressionSupport",
    "MethylationSupport",
    "CNVFunctionalSupport",
    "NonMutationOmicsSupport",
    "RawNetworkSupport",
    "DegreeCorrectedNetworkSupportV2",
    "LowFrequencyEvidenceScoreV2",
    "EvidenceWithoutRarityV2",
    "EvidenceWithoutOmicsV2",
    "EvidenceWithoutNetworkV2",
]

SCORE_COLUMNS = SCORE_COLUMNS_V1


def clean_gene(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    gene = str(value).strip().upper()
    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()
    if gene in {"", "?", "NA", "N/A", "NAN", "NONE", "NULL"}:
        return None
    return gene


def load_evidence_table(path: str | Path) -> pd.DataFrame:
    evidence_path = Path(path)
    raw = pd.read_csv(evidence_path)
    if all(column in raw.columns for column in REQUIRED_COLUMNS_V2):
        required_columns = REQUIRED_COLUMNS_V2
        score_columns = SCORE_COLUMNS_V2
    elif all(column in raw.columns for column in REQUIRED_COLUMNS_V1):
        required_columns = REQUIRED_COLUMNS_V1
        score_columns = SCORE_COLUMNS_V1
    else:
        missing_v2 = [column for column in REQUIRED_COLUMNS_V2 if column not in raw.columns]
        missing_v1 = [column for column in REQUIRED_COLUMNS_V1 if column not in raw.columns]
        raise ValueError(
            "Low-frequency evidence table does not match V1 or V2 schema. "
            f"missing_v2={missing_v2}; missing_v1={missing_v1}"
        )
    df = raw[required_columns].copy()
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Low-frequency evidence table missing columns: {missing}")
    df["Gene"] = df["Gene"].map(clean_gene)
    if df["Gene"].isna().any():
        raise ValueError(f"Low-frequency evidence table contains invalid Gene values: {evidence_path}")
    if df["Gene"].duplicated().any():
        examples = df.loc[df["Gene"].duplicated(), "Gene"].head(10).tolist()
        raise ValueError(f"Low-frequency evidence table has duplicate Gene values: {examples}")

    numeric_columns = [
        column
        for column in required_columns
        if column not in {"Gene", "CNVAvailable", "ExpressionDirectionAvailable", "MethylationDirectionAvailable"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[numeric_columns].isna().any().any():
        raise ValueError(f"Low-frequency evidence table contains NaN numeric values: {evidence_path}")
    if np.isinf(df[numeric_columns].to_numpy(dtype=float)).any():
        raise ValueError(f"Low-frequency evidence table contains Inf numeric values: {evidence_path}")
    for column in score_columns:
        if not ((df[column] >= 0.0) & (df[column] <= 1.0)).all():
            raise ValueError(f"Low-frequency evidence score outside [0,1]: {column}")
    for column in ["CNVAvailable", "ExpressionDirectionAvailable", "MethylationDirectionAvailable"]:
        df[column] = df[column].astype(bool)
    df["MutationPatientCount"] = df["MutationPatientCount"].round().astype(int)
    df["MutationFrequencyPct"] = df["MutationPatientCount"].map(
        lambda count: mutation_frequency_pct(count, TOTAL_KIRC_TUMOR_SAMPLES)
    )
    df["MutationGroup"] = df["MutationPatientCount"].map(classify_mutation_frequency)
    return df


def load_evidence_by_gene(path: str | Path, gene_order: Iterable[str] | None = None) -> dict[str, dict]:
    df = load_evidence_table(path)
    if gene_order is not None:
        cleaned_order = [clean_gene(gene) for gene in gene_order]
        if any(gene is None for gene in cleaned_order):
            raise ValueError("gene_order contains invalid genes.")
        table_order = df["Gene"].tolist()
        if len(cleaned_order) == len(table_order) and cleaned_order != table_order:
            raise ValueError("Low-frequency evidence table Gene order does not match HPRD gene order.")
        missing = sorted(set(cleaned_order) - set(table_order))
        if missing:
            raise ValueError(f"Low-frequency evidence table missing HPRD genes: {missing[:10]}")
    return {row["Gene"]: row.to_dict() for _, row in df.iterrows()}
