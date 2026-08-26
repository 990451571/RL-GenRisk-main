from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import project_root


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = project_root()

# 已完成Gene Symbol归一化和歧义重复处理的正式CNV输入。
THRESHOLDED_PATH = (
    ROOT
    / "data/raw/cnv_kirc/"
      "KIRC_GISTIC2_thresholded.by_genes.normalized_unique_v2.tsv.gz"
)

CONTINUOUS_PATH = (
    ROOT
    / "data/raw/cnv_kirc/"
      "KIRC_GISTIC2_continuous.by_genes.normalized_unique_v2.tsv.gz"
)

MULTIOMICS_3_PATH = (
    ROOT / "data/processed/KIRC_multiomics_3omics.csv"
)

HPRD_PATH = ROOT / "data/HPRD.txt"

OUTPUT_DIR = ROOT / "data/processed/cnv_kirc"
CNV_FEATURE_PATH = OUTPUT_DIR / "KIRC_cnv_gene_feature.csv"
MULTIOMICS_4_PATH = ROOT / "data/processed/KIRC_multiomics_4omics.csv"
REPORT_PATH = OUTPUT_DIR / "cnv_processing_report.json"
FEATURE_STATS_PATH = OUTPUT_DIR / "cnv_feature_statistics.csv"

EXPECTED_THRESHOLD_VALUES = {-2, -1, 0, 1, 2}
INVALID_GENE_VALUES = {"", "NAN", "NONE", "?"}


# ============================================================
# 2. 工具函数
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"路径不是文件：{path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"文件为空：{path}"
        )


def normalize_gene_symbol(value: object) -> str:
    text = str(value).strip()

    if "|" in text:
        text = text.split("|", 1)[0]

    return text.upper()


def get_tcga_sample_type(
    barcode: str,
) -> str | None:
    parts = str(barcode).strip().split("-")

    if len(parts) < 4:
        return None

    if len(parts[3]) < 2:
        return None

    return parts[3][:2]


def get_patient_id(barcode: str) -> str:
    parts = str(barcode).strip().split("-")

    if len(parts) < 3:
        return str(barcode).strip()

    return "-".join(parts[:3])


def detect_gene_column(
    frame: pd.DataFrame,
) -> str:
    candidates = (
        "Gene",
        "gene",
        "Gene Symbol",
        "GeneSymbol",
        "gene_symbol",
        "Hugo_Symbol",
    )

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "未检测到基因列，当前列为："
        f"{frame.columns.tolist()}"
    )


def assert_finite(
    frame: pd.DataFrame,
    name: str,
    allow_nan: bool = False,
) -> None:
    values = frame.to_numpy(dtype=float)

    nan_count = int(
        np.isnan(values).sum()
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    if not allow_nan and nan_count > 0:
        raise ValueError(
            f"{name}存在{nan_count}个NaN。"
        )

    if inf_count > 0:
        raise ValueError(
            f"{name}存在{inf_count}个Inf。"
        )


# ============================================================
# 3. 读取Xena CNV矩阵
# ============================================================


def read_xena_matrix(
    path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    读取gene×sample矩阵，只保留01型原发肿瘤。

    同一患者若存在多个01样本，仅保留字典序第一条，
    并将其他样本写入审计文件。
    """
    require_file(path)

    raw = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    if raw.shape[1] < 2:
        raise ValueError(
            f"矩阵列数异常：{raw.shape}，"
            f"文件={path}"
        )

    gene_col = str(
        raw.columns[0]
    )

    sample_cols = [
        str(column)
        for column in raw.columns[1:]
        if str(column).startswith("TCGA-")
    ]

    unexpected_cols = [
        str(column)
        for column in raw.columns[1:]
        if not str(column).startswith("TCGA-")
    ]

    if unexpected_cols:
        raise ValueError(
            "基因列之后发现非TCGA样本列："
            f"{unexpected_cols[:20]}"
        )

    primary_cols = [
        sample
        for sample in sample_cols
        if get_tcga_sample_type(sample) == "01"
    ]

    if not primary_cols:
        raise ValueError(
            f"未检测到01型原发肿瘤样本：{path}"
        )

    selected_by_patient: dict[
        str,
        str,
    ] = {}

    duplicate_patient_records: list[
        dict[str, str]
    ] = []

    for sample in sorted(primary_cols):
        patient = get_patient_id(sample)

        if patient not in selected_by_patient:
            selected_by_patient[
                patient
            ] = sample

        else:
            duplicate_patient_records.append(
                {
                    "patient_id": patient,
                    "kept_sample": (
                        selected_by_patient[
                            patient
                        ]
                    ),
                    "excluded_duplicate_sample": (
                        sample
                    ),
                }
            )

    selected_cols = list(
        selected_by_patient.values()
    )

    genes = raw[gene_col].map(
        normalize_gene_symbol
    )

    invalid_mask = genes.isin(
        INVALID_GENE_VALUES
    )

    if invalid_mask.any():
        examples = (
            raw.loc[
                invalid_mask,
                gene_col,
            ]
            .head(20)
            .tolist()
        )

        raise ValueError(
            "检测到非法Gene Symbol："
            f"{examples}"
        )

    numeric = raw[selected_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric.insert(
        0,
        "Gene",
        genes,
    )

    duplicate_genes = sorted(
        numeric.loc[
            numeric[
                "Gene"
            ].duplicated(
                keep=False
            ),
            "Gene",
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicate_genes:
        raise ValueError(
            "normalized_unique_v2矩阵仍存在重复"
            "Gene Symbol："
            f"{duplicate_genes[:20]}"
        )

    matrix = numeric.set_index(
        "Gene"
    )

    matrix.index.name = "Gene"

    assert_finite(
        matrix,
        path.name,
    )

    sample_manifest = pd.DataFrame(
        {
            "sample_barcode": selected_cols,
            "patient_id": [
                get_patient_id(sample)
                for sample in selected_cols
            ],
            "sample_type": [
                get_tcga_sample_type(sample)
                for sample in selected_cols
            ],
        }
    )

    duplicate_patient_df = pd.DataFrame(
        duplicate_patient_records,
        columns=[
            "patient_id",
            "kept_sample",
            "excluded_duplicate_sample",
        ],
    )

    audit = {
        "file": str(path),
        "file_sha256": (
            sha256_file(path)
        ),
        "file_size_bytes": int(
            path.stat().st_size
        ),
        "raw_shape": [
            int(raw.shape[0]),
            int(raw.shape[1]),
        ],
        "gene_column": gene_col,
        "all_tcga_samples": int(
            len(sample_cols)
        ),
        "primary_tumor_samples_before_patient_dedup": int(
            len(primary_cols)
        ),
        "primary_tumor_patients_selected": int(
            len(selected_cols)
        ),
        "duplicate_patient_sample_count": int(
            len(
                duplicate_patient_records
            )
        ),
        "final_gene_count": int(
            matrix.shape[0]
        ),
        "final_sample_count": int(
            matrix.shape[1]
        ),
        "duplicate_gene_count": int(
            matrix.index.duplicated().sum()
        ),
        "nan_count": int(
            matrix.isna().sum().sum()
        ),
        "inf_count": int(
            np.isinf(
                matrix.to_numpy(
                    dtype=float
                )
            ).sum()
        ),
    }

    return (
        matrix,
        sample_manifest,
        duplicate_patient_df,
        audit,
    )


# ============================================================
# 4. 矩阵验证
# ============================================================


def validate_thresholded_matrix(
    matrix: pd.DataFrame,
) -> list[int]:
    values = matrix.to_numpy(
        dtype=float
    )

    assert_finite(
        matrix,
        "thresholded GISTIC2矩阵",
    )

    rounded = np.round(values)

    if not np.allclose(
        values,
        rounded,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            "thresholded矩阵存在非整数值，"
            "可能使用了错误文件。"
        )

    unique_values = sorted(
        set(
            rounded
            .astype(int)
            .ravel()
            .tolist()
        )
    )

    unexpected = (
        set(unique_values)
        - EXPECTED_THRESHOLD_VALUES
    )

    if unexpected:
        raise ValueError(
            "检测到非预期阈值："
            f"{sorted(unexpected)}；"
            f"全部值={unique_values}"
        )

    return unique_values


def validate_alignment(
    thresholded: pd.DataFrame,
    continuous: pd.DataFrame,
) -> None:
    if (
        thresholded.index.tolist()
        != continuous.index.tolist()
    ):
        raise ValueError(
            "Thresholded与Continuous"
            "基因顺序不一致。"
        )

    if (
        thresholded.columns.tolist()
        != continuous.columns.tolist()
    ):
        raise ValueError(
            "Thresholded与Continuous"
            "样本顺序不一致。"
        )


# ============================================================
# 5. 构建基因级CNV特征
# ============================================================


def build_cnv_features(
    thresholded: pd.DataFrame,
    continuous: pd.DataFrame,
) -> pd.DataFrame:
    """
    主CNV特征：

        CNV =
        mean(abs(GISTIC2 threshold)) / 2

    范围为[0,1]，同时体现事件频率和事件强度。
    """
    valid_count = (
        thresholded
        .notna()
        .sum(axis=1)
        .astype(int)
    )

    denominator = (
        valid_count
        .replace(0, np.nan)
    )

    abs_thresholded = (
        thresholded.abs()
    )

    feature = pd.DataFrame(
        index=thresholded.index
    )

    feature[
        "CNV_ValidSampleCount"
    ] = valid_count

    feature[
        "CNV_AlteredCount"
    ] = (
        (abs_thresholded > 0)
        .sum(axis=1)
        .astype(int)
    )

    feature[
        "CNV_GainCount"
    ] = (
        (thresholded > 0)
        .sum(axis=1)
        .astype(int)
    )

    feature[
        "CNV_LossCount"
    ] = (
        (thresholded < 0)
        .sum(axis=1)
        .astype(int)
    )

    feature[
        "CNV_DeepGainCount"
    ] = (
        (thresholded == 2)
        .sum(axis=1)
        .astype(int)
    )

    feature[
        "CNV_DeepLossCount"
    ] = (
        (thresholded == -2)
        .sum(axis=1)
        .astype(int)
    )

    feature[
        "CNV_AlteredFraction"
    ] = (
        feature[
            "CNV_AlteredCount"
        ]
        / denominator
    )

    feature[
        "CNV_GainFraction"
    ] = (
        feature[
            "CNV_GainCount"
        ]
        / denominator
    )

    feature[
        "CNV_LossFraction"
    ] = (
        feature[
            "CNV_LossCount"
        ]
        / denominator
    )

    feature[
        "CNV_DeepGainFraction"
    ] = (
        feature[
            "CNV_DeepGainCount"
        ]
        / denominator
    )

    feature[
        "CNV_DeepLossFraction"
    ] = (
        feature[
            "CNV_DeepLossCount"
        ]
        / denominator
    )

    feature[
        "CNV_DeepFraction"
    ] = (
        (abs_thresholded == 2)
        .sum(axis=1)
        / denominator
    )

    # 模型使用的主CNV异常特征。
    feature["CNV"] = (
        abs_thresholded.sum(
            axis=1
        )
        / (
            2.0
            * denominator
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # 正值偏扩增，负值偏缺失。
    feature[
        "CNV_SignedMean"
    ] = (
        thresholded.sum(
            axis=1
        )
        / (
            2.0
            * denominator
        )
    ).clip(
        lower=-1.0,
        upper=1.0,
    )

    feature[
        "CNV_MaxAbsThreshold"
    ] = abs_thresholded.max(
        axis=1
    )

    continuous_valid_count = (
        continuous
        .notna()
        .sum(axis=1)
        .replace(0, np.nan)
    )

    feature[
        "CNV_ContinuousAbsMean"
    ] = continuous.abs().mean(
        axis=1
    )

    feature[
        "CNV_ContinuousSignedMean"
    ] = continuous.mean(
        axis=1
    )

    feature[
        "CNV_ContinuousAbsMedian"
    ] = continuous.abs().median(
        axis=1
    )

    feature[
        "CNV_ContinuousAbsMax"
    ] = continuous.abs().max(
        axis=1
    )

    feature[
        "CNV_ContinuousPositiveFraction"
    ] = (
        (continuous > 0)
        .sum(axis=1)
        / continuous_valid_count
    )

    feature[
        "CNV_ContinuousNegativeFraction"
    ] = (
        (continuous < 0)
        .sum(axis=1)
        / continuous_valid_count
    )

    feature = feature.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if feature.isna().any().any():
        bad_columns = (
            feature.columns[
                feature.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "CNV特征出现NaN："
            f"{bad_columns}"
        )

    if not (
        (
            feature["CNV"] >= 0
        )
        & (
            feature["CNV"] <= 1
        )
    ).all():
        raise ValueError(
            "CNV主特征超出[0,1]。"
        )

    feature.index.name = "Gene"

    return feature.reset_index()


# ============================================================
# 6. 读取三组学并合并CNV
# ============================================================


def read_multiomics_3(
    path: Path,
) -> pd.DataFrame:
    require_file(path)

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    gene_col = detect_gene_column(
        frame
    )

    if gene_col != "Gene":
        frame = frame.rename(
            columns={
                gene_col: "Gene"
            }
        )

    frame["Gene"] = frame[
        "Gene"
    ].map(
        normalize_gene_symbol
    )

    if frame[
        "Gene"
    ].isin(
        INVALID_GENE_VALUES
    ).any():
        raise ValueError(
            "三组学文件存在非法"
            "Gene Symbol。"
        )

    duplicate_genes = sorted(
        frame.loc[
            frame[
                "Gene"
            ].duplicated(
                keep=False
            ),
            "Gene",
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicate_genes:
        raise ValueError(
            "三组学文件存在重复基因："
            f"{duplicate_genes[:20]}"
        )

    if "CNV" in frame.columns:
        raise ValueError(
            "三组学文件已包含CNV列，"
            "拒绝覆盖。"
        )

    numeric_columns = [
        column
        for column in frame.columns
        if column != "Gene"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if frame[
        numeric_columns
    ].isna().any().any():
        bad_columns = (
            frame[
                numeric_columns
            ]
            .columns[
                frame[
                    numeric_columns
                ].isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "三组学文件存在NaN："
            f"{bad_columns}"
        )

    if np.isinf(
        frame[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).any():
        raise ValueError(
            "三组学文件存在Inf。"
        )

    return frame


def merge_cnv(
    multiomics_3: pd.DataFrame,
    cnv_feature: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    merged = multiomics_3.merge(
        cnv_feature[
            [
                "Gene",
                "CNV",
            ]
        ],
        on="Gene",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    missing_mask = (
        merged["CNV"].isna()
    )

    missing = merged.loc[
        missing_mask,
        ["Gene"],
    ].copy()

    missing[
        "CNV_fill_value"
    ] = 0.0

    missing[
        "CNV_fill_reason"
    ] = (
        "No matched gene-level CNV record "
        "in normalized Xena GISTIC2 matrix"
    )

    # 先记录缺失，再补0。
    merged["CNV"] = (
        merged["CNV"].fillna(
            0.0
        )
    )

    if (
        len(merged)
        != len(multiomics_3)
    ):
        raise ValueError(
            "合并CNV后三组学行数发生变化。"
        )

    if (
        merged["Gene"].tolist()
        != multiomics_3[
            "Gene"
        ].tolist()
    ):
        raise ValueError(
            "合并CNV后三组学基因顺序"
            "发生变化。"
        )

    if not merged[
        "Gene"
    ].is_unique:
        raise ValueError(
            "四组学文件Gene不唯一。"
        )

    cnv_values = merged[
        "CNV"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        cnv_values
    ).all():
        raise ValueError(
            "四组学CNV列存在NaN或Inf。"
        )

    if (
        (cnv_values < 0).any()
        or (cnv_values > 1).any()
    ):
        raise ValueError(
            "四组学CNV列超出[0,1]。"
        )

    return merged, missing


# ============================================================
# 7. HPRD审计
# ============================================================


def read_hprd_genes(
    path: Path,
) -> set[str]:
    require_file(path)

    genes: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            if text.startswith("#"):
                continue

            parts = (
                text
                .replace(",", "\t")
                .split()
            )

            if len(parts) < 2:
                continue

            gene1 = normalize_gene_symbol(
                parts[0]
            )

            gene2 = normalize_gene_symbol(
                parts[1]
            )

            if line_number == 1:
                header_values = {
                    "GENE",
                    "GENE1",
                    "GENE2",
                    "SOURCE",
                    "TARGET",
                    "FROM",
                    "TO",
                }

                if (
                    gene1
                    in header_values
                    or gene2
                    in header_values
                ):
                    continue

            if gene1 not in INVALID_GENE_VALUES:
                genes.add(
                    gene1
                )

            if gene2 not in INVALID_GENE_VALUES:
                genes.add(
                    gene2
                )

    if not genes:
        raise ValueError(
            "未从HPRD读取到基因："
            f"{path}"
        )

    return genes


# ============================================================
# 8. 主流程
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MULTIOMICS_4_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========== INPUT =========="
    )

    print(
        "Thresholded:",
        THRESHOLDED_PATH,
    )

    print(
        "Continuous :",
        CONTINUOUS_PATH,
    )

    print(
        "3-omics   :",
        MULTIOMICS_3_PATH,
    )

    print(
        "HPRD      :",
        HPRD_PATH,
    )

    (
        thresholded,
        thresholded_samples,
        thresholded_duplicate_patients,
        thresholded_audit,
    ) = read_xena_matrix(
        THRESHOLDED_PATH
    )

    (
        continuous,
        continuous_samples,
        continuous_duplicate_patients,
        continuous_audit,
    ) = read_xena_matrix(
        CONTINUOUS_PATH
    )

    threshold_values = (
        validate_thresholded_matrix(
            thresholded
        )
    )

    validate_alignment(
        thresholded,
        continuous,
    )

    print(
        "\n========== CNV MATRIX =========="
    )

    print(
        "Gene count  :",
        thresholded.shape[0],
    )

    print(
        "Sample count:",
        thresholded.shape[1],
    )

    print(
        "Values      :",
        threshold_values,
    )

    cnv_feature = build_cnv_features(
        thresholded,
        continuous,
    )

    cnv_feature.to_csv(
        CNV_FEATURE_PATH,
        index=False,
    )

    feature_statistics = (
        cnv_feature
        .drop(
            columns=["Gene"]
        )
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Feature"
            }
        )
    )

    feature_statistics.to_csv(
        FEATURE_STATS_PATH,
        index=False,
    )

    thresholded_samples.to_csv(
        OUTPUT_DIR
        / "cnv_thresholded_sample_manifest.csv",
        index=False,
    )

    continuous_samples.to_csv(
        OUTPUT_DIR
        / "cnv_continuous_sample_manifest.csv",
        index=False,
    )

    thresholded_duplicate_patients.to_csv(
        OUTPUT_DIR
        / "cnv_thresholded_duplicate_patient_samples.csv",
        index=False,
    )

    continuous_duplicate_patients.to_csv(
        OUTPUT_DIR
        / "cnv_continuous_duplicate_patient_samples.csv",
        index=False,
    )

    multiomics_3 = read_multiomics_3(
        MULTIOMICS_3_PATH
    )

    (
        multiomics_4,
        missing_cnv,
    ) = merge_cnv(
        multiomics_3,
        cnv_feature,
    )

    missing_cnv.to_csv(
        OUTPUT_DIR
        / "multiomics_genes_missing_cnv.csv",
        index=False,
    )

    multiomics_4.to_csv(
        MULTIOMICS_4_PATH,
        index=False,
    )

    hprd_genes = read_hprd_genes(
        HPRD_PATH
    )

    cnv_genes = set(
        cnv_feature["Gene"]
    )

    multiomics_genes = set(
        multiomics_4["Gene"]
    )

    missing_cnv_genes = set(
        missing_cnv["Gene"]
    )

    hprd_missing_from_cnv = sorted(
        hprd_genes
        - cnv_genes
    )

    hprd_missing_from_multiomics4 = sorted(
        hprd_genes
        - multiomics_genes
    )

    hprd_zero_filled = sorted(
        hprd_genes
        & missing_cnv_genes
    )

    pd.DataFrame(
        {
            "Gene": (
                hprd_missing_from_cnv
            )
        }
    ).to_csv(
        OUTPUT_DIR
        / "hprd_genes_missing_from_cnv_matrix.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Gene": (
                hprd_missing_from_multiomics4
            )
        }
    ).to_csv(
        OUTPUT_DIR
        / "hprd_genes_missing_from_multiomics4.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Gene": (
                hprd_zero_filled
            ),
            "CNV_fill_value": 0.0,
            "CNV_fill_reason": (
                "No matched gene-level CNV record "
                "in normalized Xena GISTIC2 matrix"
            ),
        }
    ).to_csv(
        OUTPUT_DIR
        / "hprd_genes_zero_filled_cnv.csv",
        index=False,
    )

    # 最终完整性检查。
    if cnv_feature[
        "Gene"
    ].duplicated().any():
        raise ValueError(
            "KIRC_cnv_gene_feature.csv"
            "仍有重复基因。"
        )

    if cnv_feature.isna().any().any():
        raise ValueError(
            "KIRC_cnv_gene_feature.csv"
            "仍有NaN。"
        )

    if multiomics_4.isna().any().any():
        bad_columns = (
            multiomics_4
            .columns[
                multiomics_4
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "KIRC_multiomics_4omics.csv"
            "存在NaN："
            f"{bad_columns}"
        )

    expected_columns = (
        list(
            multiomics_3.columns
        )
        + ["CNV"]
    )

    if (
        list(
            multiomics_4.columns
        )
        != expected_columns
    ):
        raise ValueError(
            "四组学列顺序异常。"
            f"期望={expected_columns}，"
            f"实际={multiomics_4.columns.tolist()}"
        )

    report: dict[
        str,
        Any,
    ] = {
        "processing_timestamp_utc": (
            datetime
            .now(timezone.utc)
            .isoformat()
        ),
        "project_root": str(
            ROOT
        ),
        "main_cnv_feature_definition": (
            "CNV = "
            "mean(abs(thresholded_GISTIC2)) / 2"
        ),
        "main_cnv_feature_range": [
            0.0,
            1.0,
        ],
        "thresholded": (
            thresholded_audit
        ),
        "continuous": (
            continuous_audit
        ),
        "thresholded_unique_values": (
            threshold_values
        ),
        "thresholded_continuous_gene_order_identical": True,
        "thresholded_continuous_sample_order_identical": True,
        "cnv_feature_gene_count": int(
            cnv_feature.shape[0]
        ),
        "cnv_feature_column_count": int(
            cnv_feature.shape[1]
        ),
        "cnv_feature_columns": (
            cnv_feature
            .columns
            .tolist()
        ),
        "multiomics_3_path": str(
            MULTIOMICS_3_PATH
        ),
        "multiomics_3_sha256": (
            sha256_file(
                MULTIOMICS_3_PATH
            )
        ),
        "multiomics_3_shape": [
            int(
                multiomics_3.shape[0]
            ),
            int(
                multiomics_3.shape[1]
            ),
        ],
        "multiomics_4_shape": [
            int(
                multiomics_4.shape[0]
            ),
            int(
                multiomics_4.shape[1]
            ),
        ],
        "multiomics_4_columns": (
            multiomics_4
            .columns
            .tolist()
        ),
        "multiomics_genes_missing_cnv_before_fill": int(
            len(missing_cnv)
        ),
        "multiomics_duplicate_gene_count": int(
            multiomics_4[
                "Gene"
            ].duplicated().sum()
        ),
        "multiomics_cnv_nan_count_after_fill": int(
            multiomics_4[
                "CNV"
            ].isna().sum()
        ),
        "multiomics_cnv_inf_count": int(
            np.isinf(
                multiomics_4[
                    "CNV"
                ].to_numpy(
                    dtype=float
                )
            ).sum()
        ),
        "cnv_min": float(
            multiomics_4[
                "CNV"
            ].min()
        ),
        "cnv_max": float(
            multiomics_4[
                "CNV"
            ].max()
        ),
        "cnv_mean": float(
            multiomics_4[
                "CNV"
            ].mean()
        ),
        "cnv_median": float(
            multiomics_4[
                "CNV"
            ].median()
        ),
        "cnv_nonzero_gene_count": int(
            (
                multiomics_4[
                    "CNV"
                ] > 0
            ).sum()
        ),
        "hprd_path": str(
            HPRD_PATH
        ),
        "hprd_sha256": (
            sha256_file(
                HPRD_PATH
            )
        ),
        "hprd_gene_count": int(
            len(hprd_genes)
        ),
        "hprd_genes_present_in_cnv_matrix": int(
            len(
                hprd_genes
                & cnv_genes
            )
        ),
        "hprd_genes_missing_from_cnv_matrix": int(
            len(
                hprd_missing_from_cnv
            )
        ),
        "hprd_genes_present_in_multiomics4": int(
            len(
                hprd_genes
                & multiomics_genes
            )
        ),
        "hprd_genes_missing_from_multiomics4": int(
            len(
                hprd_missing_from_multiomics4
            )
        ),
        "hprd_genes_zero_filled_cnv": int(
            len(
                hprd_zero_filled
            )
        ),
        "output_cnv_feature": str(
            CNV_FEATURE_PATH
        ),
        "output_cnv_feature_sha256": (
            sha256_file(
                CNV_FEATURE_PATH
            )
        ),
        "output_multiomics_4": str(
            MULTIOMICS_4_PATH
        ),
        "output_multiomics_4_sha256": (
            sha256_file(
                MULTIOMICS_4_PATH
            )
        ),
        "output_feature_statistics": str(
            FEATURE_STATS_PATH
        ),
        "output_feature_statistics_sha256": (
            sha256_file(
                FEATURE_STATS_PATH
            )
        ),
        "final_integrity": {
            "cnv_feature_gene_unique": bool(
                cnv_feature[
                    "Gene"
                ].is_unique
            ),
            "cnv_feature_nan_count": int(
                cnv_feature
                .isna()
                .sum()
                .sum()
            ),
            "multiomics4_gene_unique": bool(
                multiomics_4[
                    "Gene"
                ].is_unique
            ),
            "multiomics4_nan_count": int(
                multiomics_4
                .isna()
                .sum()
                .sum()
            ),
            "multiomics4_gene_order_preserved": bool(
                multiomics_4[
                    "Gene"
                ].tolist()
                == multiomics_3[
                    "Gene"
                ].tolist()
            ),
            "multiomics4_row_count_preserved": bool(
                len(multiomics_4)
                == len(multiomics_3)
            ),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n========== FINAL REPORT =========="
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nCNV processing completed successfully."
    )

    print(
        "CNV feature file :",
        CNV_FEATURE_PATH,
    )

    print(
        "4-omics file     :",
        MULTIOMICS_4_PATH,
    )

    print(
        "Processing report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
