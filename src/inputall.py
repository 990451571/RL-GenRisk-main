import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "data").exists():
    REPO_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "data").exists():
    REPO_DIR = SCRIPT_DIR.parent
else:
    REPO_DIR = SCRIPT_DIR.parent
DATA_DIR = REPO_DIR / "data"
DEFAULT_MULTIOMICS_FEATURE_PATH = DATA_DIR / "processed" / "KIRC_multiomics_3omics.csv"
DEFAULT_MULTIOMICS_4OMICS_FEATURE_PATH = DATA_DIR / "processed" / "KIRC_multiomics_4omics.csv"
DEFAULT_CNV_MISSING_GENE_PATH = DATA_DIR / "processed" / "cnv_kirc" / "multiomics_genes_missing_cnv.csv"
REQUIRED_MULTIOMICS_COLUMNS = ["Mutation", "Expression", "Methylation"]
SUPPORTED_MULTIOMICS_COLUMNS = ["Mutation", "Expression", "Methylation", "CNV"]
MULTIOMICS_COLUMNS = SUPPORTED_MULTIOMICS_COLUMNS
INVALID_GENE_SYMBOLS = {"", "?", "NA", "N/A", "NAN", "NONE", "NULL"}
FEATURE_MODE_COLUMNS = {
    "original3_raw": ["Degree", "WeightValue", "PatientCoverageCount"],
    "original3_zscore": ["Degree", "WeightValue", "PatientCoverageCount"],
    "multiomics3_raw": ["Mutation", "Expression", "Methylation"],
    "hybrid6_raw": ["Degree", "WeightValue", "PatientCoverageCount", "Mutation", "Expression", "Methylation"],
    "hybrid6_zscore": ["Degree", "WeightValue", "PatientCoverageCount", "Mutation", "Expression", "Methylation"],
    "multiomics4_raw": ["Mutation", "Expression", "Methylation", "CNV"],
    "hybrid7_raw": ["Degree", "WeightValue", "PatientCoverageCount", "Mutation", "Expression", "Methylation", "CNV"],
    "multiomics4_zscore": ["Mutation", "Expression", "Methylation", "CNV"],
    "hybrid7_zscore": ["Degree", "WeightValue", "PatientCoverageCount", "Mutation", "Expression", "Methylation", "CNV"],
}
FOUR_OMICS_FEATURE_MODES = {"multiomics4_raw", "hybrid7_raw", "multiomics4_zscore", "hybrid7_zscore"}
Z_SCORE_FEATURE_MODES = {"original3_zscore", "hybrid6_zscore", "multiomics4_zscore", "hybrid7_zscore"}
FEATURE_COLUMN_ALIASES = {"PPI_Degree": "Degree"}


def canonicalize_feature_columns(columns):
    return [FEATURE_COLUMN_ALIASES.get(str(column), str(column)) for column in columns]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def get_default_mutation_path(cancer):
    return DATA_DIR / f"{cancer}.txt"


def clean_gene_symbol(value):
    if value is None or pd.isna(value):
        return None
    gene = str(value).strip().upper()
    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()
    if gene in INVALID_GENE_SYMBOLS:
        return None
    return gene


def _clean_gene_series(series):
    return series.map(clean_gene_symbol)


def _validate_ordered_gene_list(gene_list):
    cleaned = []
    invalid_count = 0
    for gene in gene_list:
        cleaned_gene = clean_gene_symbol(gene)
        if cleaned_gene is None:
            invalid_count += 1
        else:
            cleaned.append(cleaned_gene)

    if invalid_count:
        raise ValueError(f"PPI gene_list contains {invalid_count} invalid Gene values.")

    duplicated = pd.Series(cleaned).duplicated()
    if duplicated.any():
        duplicate_examples = pd.Series(cleaned)[duplicated].head(10).tolist()
        raise ValueError(
            "PPI gene_list contains duplicate Gene values after cleaning: "
            f"{int(duplicated.sum())}; examples={duplicate_examples}"
        )
    return cleaned


def _clean_gene_patient_map(gene_patient_map):
    cleaned = {}
    for gene, patients in gene_patient_map.items():
        cleaned_gene = clean_gene_symbol(gene)
        if cleaned_gene is not None:
            cleaned[cleaned_gene] = patients
    return cleaned


def _reject_duplicate_genes_after_cleaning(df, source_path, context):
    duplicated_mask = df["Gene"].duplicated(keep=False)
    if not duplicated_mask.any():
        return 0
    duplicate_genes = sorted(df.loc[duplicated_mask, "Gene"].dropna().unique().tolist())
    raise ValueError(
        f"{context} found duplicate Gene values after cleaning in {source_path}; "
        f"duplicate_gene_count={len(duplicate_genes)}; examples={duplicate_genes[:10]}; "
        "refusing silent aggregation."
    )


def load_gene_list(path):
    genes = []
    with open(resolve_data_path(path), "r", encoding="utf-8") as file_to_read:
        for line in file_to_read:
            gene_temp = line.split()
            if len(gene_temp) > 0:
                gene = clean_gene_symbol(gene_temp[0])
                if gene is not None:
                    genes.append(gene)
    return list(dict.fromkeys(genes))


def _print_feature_stats(prefix, values, feature_columns):
    for col_idx, col_name in enumerate(feature_columns):
        col = values[:, col_idx]
        print(
            f"{prefix} {col_name}: "
            f"mean={np.mean(col):.4f}, std={np.std(col):.4f}, "
            f"min={np.min(col):.4f}, max={np.max(col):.4f}"
        )


def check_multiomics_ppi_overlap(
    multiomics_dataframe,
    gene_list,
    feature_columns,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR / "processed" / "ppi_overlap_check"
    if not output_dir.is_absolute():
        output_dir = DATA_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_gene_list = _validate_ordered_gene_list(gene_list)
    ppi_genes = set(cleaned_gene_list)
    multiomics_genes = set(multiomics_dataframe["Gene"])
    matched_genes = [gene for gene in cleaned_gene_list if gene in multiomics_genes]
    multiomics_not_in_ppi = sorted(multiomics_genes - ppi_genes)
    ppi_without_multiomics = [gene for gene in cleaned_gene_list if gene not in multiomics_genes]

    indexed = multiomics_dataframe.set_index("Gene", drop=False)
    matched_df = indexed.loc[matched_genes, ["Gene"] + feature_columns].reset_index(drop=True)
    all_zero_mask = (matched_df[feature_columns] == 0).all(axis=1)
    matched_all_zero = matched_df.loc[all_zero_mask].copy()

    summary = {
        "Multiomics_unique_genes": len(multiomics_genes),
        "PPI_unique_genes": len(ppi_genes),
        "Matched_genes": len(matched_genes),
        "Multiomics_match_rate": len(matched_genes) / len(multiomics_genes) if multiomics_genes else 0.0,
        "PPI_coverage_rate": len(matched_genes) / len(ppi_genes) if ppi_genes else 0.0,
        "Multiomics_not_in_PPI": len(multiomics_not_in_ppi),
        "PPI_without_multiomics": len(ppi_without_multiomics),
        "Matched_all_omics_zero": len(matched_all_zero),
    }

    if not matched_df.empty:
        describe = matched_df[feature_columns].describe()
        zero_rates = (matched_df[feature_columns] == 0).mean()
        for column in feature_columns:
            for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                summary[f"{column}_{stat_name}"] = describe.loc[stat_name, column]
            summary[f"{column}_zero_rate"] = zero_rates[column]

    pd.DataFrame([summary]).to_csv(output_dir / "overlap_summary.csv", index=False, encoding="utf-8")
    matched_df.to_csv(output_dir / "matched_multiomics_ppi_genes.csv", index=False, encoding="utf-8")
    pd.DataFrame({"Gene": multiomics_not_in_ppi}).to_csv(
        output_dir / "multiomics_not_in_ppi.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"Gene": ppi_without_multiomics}).to_csv(
        output_dir / "ppi_without_multiomics.csv", index=False, encoding="utf-8"
    )
    matched_all_zero.to_csv(
        output_dir / "matched_genes_all_omics_zero.csv", index=False, encoding="utf-8"
    )

    return summary


def load_multiomics_features(
    feature_path,
    gene_list,
    standardize=False,
    save_overlap_report=True,
    overlap_output_dir=None,
):
    feature_path = resolve_data_path(feature_path)
    df = pd.read_csv(feature_path)

    if "Gene" not in df.columns:
        raise ValueError("Multi-omics feature file missing required column: Gene")

    missing_columns = [col for col in REQUIRED_MULTIOMICS_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Multi-omics feature file missing columns: {missing_columns}")

    feature_columns = [col for col in SUPPORTED_MULTIOMICS_COLUMNS if col in df.columns]
    df = df[["Gene"] + feature_columns].copy()
    df["Gene"] = _clean_gene_series(df["Gene"])
    df = df[df["Gene"].notna()].copy()
    duplicate_count = _reject_duplicate_genes_after_cleaning(
        df,
        feature_path,
        "load_multiomics_features",
    )
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cleaned_gene_list = _validate_ordered_gene_list(gene_list)

    if save_overlap_report:
        overlap_summary = check_multiomics_ppi_overlap(
            df,
            cleaned_gene_list,
            feature_columns,
            output_dir=overlap_output_dir,
        )
    else:
        overlap_summary = None

    indexed = df.set_index("Gene")
    features = np.zeros((len(cleaned_gene_list), len(feature_columns)), dtype=np.float32)
    matched_mask = np.zeros(len(cleaned_gene_list), dtype=bool)
    for idx, gene in enumerate(cleaned_gene_list):
        if gene in indexed.index:
            features[idx, :] = indexed.loc[gene, feature_columns].to_numpy(dtype=np.float32)
            matched_mask[idx] = True

    if standardize:
        matched_values = features[matched_mask, :]
        if matched_values.size:
            means = matched_values.mean(axis=0)
            stds = matched_values.std(axis=0)
            stds[stds == 0] = 1.0
            features[matched_mask, :] = (matched_values - means) / stds

    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Multi-omics node feature matrix contains NaN or infinite values.")

    matched = int(matched_mask.sum())
    missing = len(cleaned_gene_list) - matched
    print("Using multi-omics features")
    print(f"Multi-omics feature file: {feature_path}")
    print(f"Actual omics columns: {feature_columns}")
    print(f"Multi-omics unique genes: {len(df)}")
    print(f"Duplicate genes after cleaning: {duplicate_count}")
    print(f"PPI gene_list genes: {len(cleaned_gene_list)}")
    print(f"Matched genes: {matched}")
    print(f"Missing genes filled with 0: {missing}")
    if overlap_summary:
        print(f"Multiomics match rate: {overlap_summary['Multiomics_match_rate']:.6f}")
        print(f"PPI coverage rate: {overlap_summary['PPI_coverage_rate']:.6f}")
    print(f"node_features.shape: {features.shape}")
    print(f"standardize: {standardize}")
    print("Multi-omics node feature stats:")
    _print_feature_stats("  final", features, feature_columns)
    return features


def build_original_node_features_raw(net, weights, gene_name, gene_final):
    """Build unscaled Degree, WeightValue and PatientCoverageCount features.

    This function is the only valid source for original3_raw / hybrid6_raw.
    It deliberately performs no z-score or min-max scaling.
    """
    cleaned_gene_name = _validate_ordered_gene_list(gene_name)
    cleaned_gene_final = _clean_gene_patient_map(gene_final)
    nodes_size = net.shape[0]
    if nodes_size != len(cleaned_gene_name):
        raise ValueError(
            f"PPI matrix/node-list mismatch: net has {nodes_size} nodes, "
            f"gene_name has {len(cleaned_gene_name)} genes."
        )

    feature = np.zeros((nodes_size, 3), dtype=np.float32)
    for i, gene in enumerate(cleaned_gene_name):
        feature[i, 0] = float(np.sum(net[i]))
        feature[i, 1] = float(weights.get(gene, 0.0))
        feature[i, 2] = float(len(cleaned_gene_final.get(gene, [])))

    if not np.isfinite(feature).all():
        raise ValueError("original3_raw contains NaN or Inf.")
    return feature


def build_original_node_features(net, weights, gene_name, gene_final):
    """Backward-compatible min-max version used by legacy experiments.

    New raw-feature experiments must call build_original_node_features_raw().
    """
    feature = build_original_node_features_raw(net, weights, gene_name, gene_final)
    feature = feature.copy()
    for i in range(feature.shape[1]):
        col = feature[:, [i]]
        min_value = np.min(col)
        max_value = np.max(col)
        if max_value == min_value:
            feature[:, [i]] = 0.0
        else:
            feature[:, [i]] = (col - min_value) / (max_value - min_value)
    return feature.astype(np.float32)


def _zscore_node_features(features, feature_names, normalization_metadata=None):
    features = np.asarray(features, dtype=np.float32).copy()
    feature_names = canonicalize_feature_columns(feature_names)
    if normalization_metadata:
        metadata_feature_names = canonicalize_feature_columns(
            normalization_metadata.get("feature_names", feature_names)
        )
        if metadata_feature_names != feature_names:
            raise ValueError(
                "Normalization metadata feature_names do not match requested features: "
                f"metadata={metadata_feature_names}; requested={feature_names}"
            )
        means = np.asarray(normalization_metadata["mean"], dtype=np.float64)
        stds = np.asarray(normalization_metadata["std"], dtype=np.float64)
    else:
        means = features.mean(axis=0)
        stds = features.std(axis=0)
    safe_stds = stds.copy()
    zero_std = safe_stds == 0
    safe_stds[zero_std] = 1.0
    normalized = (features - means) / safe_stds
    metadata = {
        "method": "zscore",
        "feature_names": list(feature_names),
        "mean": means.astype(float).tolist(),
        "std": stds.astype(float).tolist(),
        "zero_std_columns": [feature_names[i] for i, is_zero in enumerate(zero_std) if is_zero],
    }
    return normalized.astype(np.float32), metadata


def get_feature_columns_for_mode(feature_mode):
    mode = str(feature_mode).lower()
    if mode not in FEATURE_MODE_COLUMNS:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return canonicalize_feature_columns(FEATURE_MODE_COLUMNS[mode])


def default_multiomics_feature_path_for_mode(feature_mode):
    mode = str(feature_mode).lower()
    if mode in FOUR_OMICS_FEATURE_MODES:
        return DEFAULT_MULTIOMICS_4OMICS_FEATURE_PATH
    return DEFAULT_MULTIOMICS_FEATURE_PATH


def _load_cnv_missing_gene_set(cnv_missing_gene_path):
    path = resolve_data_path(cnv_missing_gene_path or DEFAULT_CNV_MISSING_GENE_PATH)
    df = pd.read_csv(path)
    gene_col = "Gene" if "Gene" in df.columns else df.columns[0]
    genes = _clean_gene_series(df[gene_col]).dropna()
    return set(genes), path


def compute_full_multiomics_cnv_normalization_metadata(
    multiomics_feature_path=DEFAULT_MULTIOMICS_4OMICS_FEATURE_PATH,
    cnv_missing_gene_path=DEFAULT_CNV_MISSING_GENE_PATH,
):
    """Compute CNV z-score stats before HPRD alignment.

    CNV zeros introduced for unmatched CNV genes are missing evidence, not
    observed low CNV.  Therefore the CNV mean/std are estimated only from the
    full multi-omics table rows whose genes are not in the missing-CNV list.
    """
    feature_path = resolve_data_path(multiomics_feature_path)
    missing_genes, missing_path = _load_cnv_missing_gene_set(cnv_missing_gene_path)
    df = pd.read_csv(feature_path)
    required = ["Gene", "CNV"]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Multi-omics CNV source missing columns: {missing_columns}")
    df = df[required].copy()
    df["Gene"] = _clean_gene_series(df["Gene"])
    invalid_gene_count = int(df["Gene"].isna().sum())
    if invalid_gene_count:
        raise ValueError(
            f"Multi-omics CNV source contains invalid Gene values: "
            f"invalid_gene_count={invalid_gene_count}; file={feature_path}"
        )
    _reject_duplicate_genes_after_cleaning(
        df,
        feature_path,
        "compute_full_multiomics_cnv_normalization_metadata",
    )
    df["CNV"] = pd.to_numeric(df["CNV"], errors="coerce")
    cnv_values = df["CNV"].to_numpy(dtype=np.float64)
    if np.isnan(cnv_values).any() or np.isinf(cnv_values).any():
        raise ValueError(f"CNV source contains NaN or Inf after numeric conversion: {feature_path}")

    missing_mask = df["Gene"].isin(missing_genes).to_numpy(dtype=bool)
    observed_values = cnv_values[~missing_mask]
    if observed_values.size == 0:
        raise ValueError("CNV z-score requires at least one full multi-omics observed CNV gene.")
    observed_mean = float(observed_values.mean(dtype=np.float64))
    observed_std = float(observed_values.std(ddof=0, dtype=np.float64))

    return {
        "cnv_stat_scope": "full_multiomics_observed_genes_before_hprd_alignment",
        "total_multiomics_gene_count": int(len(df)),
        "cnv_observed_gene_count": int(observed_values.size),
        "cnv_missing_gene_count": int(missing_mask.sum()),
        "cnv_mean": observed_mean,
        "cnv_std": observed_std,
        "cnv_ddof": 0,
        "cnv_missing_zscore_fill_value": 0.0,
        "multiomics_feature_path": str(feature_path),
        "multiomics_feature_sha256": sha256_file(feature_path),
        "cnv_missing_gene_path": str(missing_path),
        "cnv_missing_gene_sha256": sha256_file(missing_path),
    }


def load_multiomics_features_for_columns(
    feature_path,
    gene_list,
    feature_columns,
    cnv_missing_gene_path=DEFAULT_CNV_MISSING_GENE_PATH,
):
    feature_path = resolve_data_path(feature_path)
    df = pd.read_csv(feature_path)

    required = ["Gene"] + list(feature_columns)
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Multi-omics feature file missing columns: {missing_columns}")

    raw_rows = len(df)
    df = df[required].copy()
    df["Gene"] = _clean_gene_series(df["Gene"])
    invalid_gene_count = int(df["Gene"].isna().sum())
    df = df[df["Gene"].notna()].copy()
    duplicate_count = _reject_duplicate_genes_after_cleaning(
        df,
        feature_path,
        "load_multiomics_features_for_columns",
    )
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(feature_columns)].isna().any().any():
        raise ValueError(f"Multi-omics numeric columns contain NaN after conversion: {feature_path}")
    values = df[list(feature_columns)].to_numpy(dtype=np.float32)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(f"Multi-omics numeric columns contain NaN or Inf: {feature_path}")

    cleaned_gene_list = _validate_ordered_gene_list(gene_list)
    indexed = df.set_index("Gene")
    features = np.zeros((len(cleaned_gene_list), len(feature_columns)), dtype=np.float32)
    matched_mask = np.zeros(len(cleaned_gene_list), dtype=bool)
    for idx, gene in enumerate(cleaned_gene_list):
        if gene in indexed.index:
            features[idx, :] = indexed.loc[gene, list(feature_columns)].to_numpy(dtype=np.float32)
            matched_mask[idx] = True

    cnv_missing_mask = np.zeros(len(cleaned_gene_list), dtype=bool)
    cnv_observed_mask = np.zeros(len(cleaned_gene_list), dtype=bool)
    cnv_missing_genes = set()
    cnv_missing_path = None
    if "CNV" in feature_columns:
        cnv_missing_genes, cnv_missing_path = _load_cnv_missing_gene_set(cnv_missing_gene_path)
        for idx, gene in enumerate(cleaned_gene_list):
            if not matched_mask[idx]:
                continue
            if gene in cnv_missing_genes:
                cnv_missing_mask[idx] = True
            else:
                cnv_observed_mask[idx] = True

    report = {
        "path": str(feature_path),
        "raw_rows": int(raw_rows),
        "invalid_gene_count": invalid_gene_count,
        "unique_genes_after_cleaning": int(len(df)),
        "duplicate_gene_count_after_cleaning": duplicate_count,
        "duplicate_gene_count_before_groupby": duplicate_count,
        "ppi_nodes": int(len(cleaned_gene_list)),
        "matched_genes": int(matched_mask.sum()),
        "zero_filled_ppi_genes": int(len(cleaned_gene_list) - matched_mask.sum()),
        "extra_multiomics_genes": int(len(set(df["Gene"]) - set(cleaned_gene_list))),
        "feature_columns": list(feature_columns),
        "cnv_missing_gene_path": str(cnv_missing_path) if cnv_missing_path else None,
        "cnv_missing_hprd_count": int(cnv_missing_mask.sum()),
        "cnv_observed_hprd_count": int(cnv_observed_mask.sum()),
        "_matched_mask": matched_mask,
        "_cnv_missing_mask": cnv_missing_mask,
        "_cnv_observed_mask": cnv_observed_mask,
    }
    return features, report


def _zscore_node_features_with_cnv_missing(
    features,
    feature_names,
    cnv_observed_mask,
    normalization_metadata=None,
    cnv_normalization_metadata=None,
):
    features = np.asarray(features, dtype=np.float32).copy()
    feature_names = canonicalize_feature_columns(feature_names)
    if "CNV" not in feature_names:
        return _zscore_node_features(features, feature_names, normalization_metadata)

    cnv_idx = feature_names.index("CNV")
    normalized = features.copy()
    if normalization_metadata:
        metadata_feature_names = canonicalize_feature_columns(
            normalization_metadata.get("feature_names", feature_names)
        )
        if metadata_feature_names != feature_names:
            raise ValueError(
                "Normalization metadata feature_names do not match requested features: "
                f"metadata={metadata_feature_names}; requested={feature_names}"
            )
        means = np.asarray(normalization_metadata["mean"], dtype=np.float32)
        stds = np.asarray(normalization_metadata["std"], dtype=np.float32)
        cnv_metadata = {
            key: value
            for key, value in normalization_metadata.items()
            if str(key).startswith("cnv_") or key in {
                "total_multiomics_gene_count",
                "multiomics_feature_path",
                "multiomics_feature_sha256",
            }
        }
    else:
        means = features.astype(np.float64).mean(axis=0)
        stds = features.astype(np.float64).std(axis=0)
        if cnv_normalization_metadata is None:
            fallback_observed_mask = np.asarray(cnv_observed_mask, dtype=bool)
            observed_values = features[fallback_observed_mask, cnv_idx]
            if observed_values.size == 0:
                raise ValueError("CNV z-score requires at least one observed CNV gene.")
            means[cnv_idx] = observed_values.mean()
            stds[cnv_idx] = observed_values.std()
            cnv_metadata = {
                "cnv_stat_scope": "hprd_observed_genes_after_alignment_legacy_fallback",
                "cnv_observed_gene_count": int(observed_values.size),
                "cnv_missing_gene_count": int((~fallback_observed_mask).sum()),
                "cnv_mean": float(means[cnv_idx]),
                "cnv_std": float(stds[cnv_idx]),
                "cnv_ddof": 0,
            }
        else:
            means[cnv_idx] = float(cnv_normalization_metadata["cnv_mean"])
            stds[cnv_idx] = float(cnv_normalization_metadata["cnv_std"])
            cnv_metadata = dict(cnv_normalization_metadata)

    safe_stds = stds.copy()
    zero_std = safe_stds == 0
    safe_stds[zero_std] = 1.0
    non_cnv_indices = [idx for idx, name in enumerate(feature_names) if name != "CNV"]
    if non_cnv_indices:
        normalized[:, non_cnv_indices] = (
            features[:, non_cnv_indices] - means[non_cnv_indices]
        ) / safe_stds[non_cnv_indices]
    observed_mask = np.asarray(cnv_observed_mask, dtype=bool)
    normalized[observed_mask, cnv_idx] = (
        features[observed_mask, cnv_idx] - means[cnv_idx]
    ) / safe_stds[cnv_idx]
    # CNV=0 caused by missing evidence is treated as observed mean in z-score space.
    normalized[~observed_mask, cnv_idx] = 0.0

    metadata = {
        "method": "zscore_cnv_missing_aware",
        "ddof": 0,
        "feature_names": feature_names,
        "mean": means.astype(float).tolist(),
        "std": stds.astype(float).tolist(),
        "zero_std_columns": [feature_names[i] for i, is_zero in enumerate(zero_std) if is_zero],
        "cnv_observed_hprd_count": int(observed_mask.sum()),
        "cnv_missing_or_unmatched_hprd_count": int((~observed_mask).sum()),
        "cnv_missing_zscore_fill_value": 0.0,
    }
    metadata.update(cnv_metadata)
    return normalized.astype(np.float32), metadata


def get_node_features_by_mode(
    net,
    weights,
    gene_name,
    gene_final,
    feature_mode="original3_raw",
    multiomics_feature_path=DEFAULT_MULTIOMICS_FEATURE_PATH,
    cnv_missing_gene_path=DEFAULT_CNV_MISSING_GENE_PATH,
    normalization_metadata=None,
    return_report=False,
):
    original_names = ["Degree", "WeightValue", "PatientCoverageCount"]
    multiomics_names = ["Mutation", "Expression", "Methylation"]
    multiomics4_names = ["Mutation", "Expression", "Methylation", "CNV"]
    feature_mode = str(feature_mode).lower()

    if feature_mode in {"original3_raw", "original3_zscore", "hybrid6_raw", "hybrid6_zscore", "hybrid7_raw", "hybrid7_zscore"}:
        original = build_original_node_features_raw(net, weights, gene_name, gene_final)
    else:
        original = None

    if feature_mode in {"multiomics3_raw", "hybrid6_raw", "hybrid6_zscore"}:
        multiomics, multiomics_report = load_multiomics_features_for_columns(
            multiomics_feature_path,
            gene_name,
            multiomics_names,
            cnv_missing_gene_path=cnv_missing_gene_path,
        )
    elif feature_mode in FOUR_OMICS_FEATURE_MODES:
        multiomics, multiomics_report = load_multiomics_features_for_columns(
            multiomics_feature_path,
            gene_name,
            multiomics4_names,
            cnv_missing_gene_path=cnv_missing_gene_path,
        )
    else:
        multiomics = None
        multiomics_report = None

    cnv_normalization_metadata = None
    if feature_mode in {"multiomics4_zscore", "hybrid7_zscore"} and normalization_metadata is None:
        cnv_normalization_metadata = compute_full_multiomics_cnv_normalization_metadata(
            multiomics_feature_path,
            cnv_missing_gene_path,
        )

    if feature_mode == "original3_raw":
        features = original
        feature_names = original_names
        norm = {"method": "none", "feature_names": feature_names}
    elif feature_mode == "original3_zscore":
        feature_names = original_names
        features, norm = _zscore_node_features(original, feature_names, normalization_metadata)
    elif feature_mode == "multiomics3_raw":
        features = multiomics
        feature_names = multiomics_names
        norm = {"method": "none", "feature_names": feature_names}
    elif feature_mode == "hybrid6_raw":
        feature_names = original_names + multiomics_names
        features = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        norm = {"method": "none", "feature_names": feature_names}
    elif feature_mode == "hybrid6_zscore":
        feature_names = original_names + multiomics_names
        raw = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        features, norm = _zscore_node_features(raw, feature_names, normalization_metadata)
    elif feature_mode == "multiomics4_raw":
        features = multiomics
        feature_names = multiomics4_names
        norm = {"method": "none", "feature_names": feature_names}
    elif feature_mode == "hybrid7_raw":
        feature_names = original_names + multiomics4_names
        features = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        norm = {"method": "none", "feature_names": feature_names}
    elif feature_mode == "multiomics4_zscore":
        feature_names = multiomics4_names
        features, norm = _zscore_node_features_with_cnv_missing(
            multiomics,
            feature_names,
            multiomics_report["_cnv_observed_mask"],
            normalization_metadata,
            cnv_normalization_metadata,
        )
    elif feature_mode == "hybrid7_zscore":
        feature_names = original_names + multiomics4_names
        raw = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        features, norm = _zscore_node_features_with_cnv_missing(
            raw,
            feature_names,
            multiomics_report["_cnv_observed_mask"],
            normalization_metadata,
            cnv_normalization_metadata,
        )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError(f"node_features for {feature_mode} contain NaN or Inf.")
    report = {
        "feature_mode": feature_mode,
        "feature_columns": feature_names,
        "node_features_shape": list(features.shape),
        "standardize": feature_mode in Z_SCORE_FEATURE_MODES,
        "normalization_metadata": norm,
        "multiomics_report": {
            key: value
            for key, value in (multiomics_report or {}).items()
            if not key.startswith("_")
        },
    }
    if return_report:
        return features.astype(np.float32), feature_names, feature_mode, norm, report
    return features.astype(np.float32), feature_names, feature_mode, norm




def get_node_features(
    net,
    weights,
    gene_name,
    gene_final,
    use_multiomics=True,
    multiomics_feature_path=DEFAULT_MULTIOMICS_FEATURE_PATH,
    standardize_multiomics=False,
):
    feature_path = resolve_data_path(multiomics_feature_path)
    if use_multiomics and feature_path.exists():
        return load_multiomics_features(feature_path, gene_name, standardize=standardize_multiomics), "multiomics"

    if use_multiomics:
        print(f"Warning: multi-omics feature file not found; using original features fallback: {feature_path}")
    else:
        print("use_multiomics=False; using original features fallback")

    features = build_original_node_features(net, weights, gene_name, gene_final)
    print(f"original node_features.shape: {features.shape}")
    return features, "original"


def getInput(cancer, mutation_path=None, excluded_genes=None):
    filename = resolve_data_path(mutation_path) if mutation_path else get_default_mutation_path(cancer)
    excluded_genes = {
        clean_gene_symbol(gene)
        for gene in (excluded_genes or ['TTN', 'MUC16', 'SYNE1', 'NEB', 'MUC19', 'CCDC168', 'FSIP2', 'OBSCN', 'GPR98'])
    }
    patients = {}
    with open(filename, 'r', encoding="utf-8") as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            if len(gene_temp) == 0:
                continue
            gene = clean_gene_symbol(gene_temp[0])
            if gene is None or gene in excluded_genes:
                continue
            for patient in gene_temp[1:]:
                if patient not in list(patients.keys()):
                    patients[patient] = [gene]
                else:
                    patients[patient].append(gene)
    patient_num = len(patients.keys())
    train_size = int(patient_num * 0.8)
    train_data = {}
    test_data = {}
    patients_name = list(patients.keys())
    for i in range(patient_num):
        if i < train_size:
            patient = patients_name[i]
            train_data[patient] = patients[patient]
        else:
            patient = patients_name[i]
            test_data[patient] = patients[patient]
    return train_data, test_data, patients


def getWeight(gene_name, weight_path=None):
    filename = resolve_data_path(weight_path or 'weights.txt')
    gene_name = {clean_gene_symbol(gene) for gene in gene_name}
    weights = {}
    with open(filename, 'r', encoding="utf-8") as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            if len(gene_temp) < 2:
                continue
            gene = clean_gene_symbol(gene_temp[0])
            if gene in gene_name:
                weight = gene_temp[1]
                weights[gene] = float(weight)

    return weights


def getGene(patients):
    gene_dic = {}
    for patient in list(patients.keys()):
        genes = patients[patient]
        for gene in genes:
            gene = clean_gene_symbol(gene)
            if gene is None:
                continue
            if gene not in list(gene_dic.keys()):
                gene_dic[gene] = [patient]
            else:
                gene_dic[gene].append(patient)
    return gene_dic




def _read_ppi_network(gene, network_path=None):
    set1 = []
    edges = []
    invalid_edges = []
    gene_new = {}
    gene = _clean_gene_patient_map(gene)
    filename = resolve_data_path(network_path or 'HPRD.txt')
    with open(filename, 'r', encoding="utf-8") as file_to_read:
        for line_number, line in enumerate(file_to_read.readlines(), start=1):
            gene_temp = line.split()
            if len(gene_temp) < 2:
                continue
            gene1 = clean_gene_symbol(gene_temp[0])
            gene2 = clean_gene_symbol(gene_temp[1])
            if gene1 is None or gene2 is None:
                invalid_edges.append((line_number, gene_temp[:2]))
                continue
            edges.append((gene1, gene2))
            if gene1 not in list(gene_new.keys()):
                if gene1 in list(gene.keys()):
                    gene_new[gene1] = gene[gene1]
                else:
                    gene_new[gene1] = []
            if gene2 not in list(gene_new.keys()):
                if gene2 in list(gene.keys()):
                    gene_new[gene2] = gene[gene2]
                else:
                    gene_new[gene2] = []
            if gene1 not in set1:
                set1.append(gene1)
            if gene2 not in set1:
                set1.append(gene2)

    if invalid_edges:
        raise ValueError(
            f"PPI network contains {len(invalid_edges)} invalid edge rows; "
            f"examples={invalid_edges[:10]}"
        )

    gen_len = len(set1)
    print(len(set1))
    print(len(gene_new))
    gene_to_index = {gene_name: idx for idx, gene_name in enumerate(set1)}
    network = np.zeros((gen_len, gen_len), dtype=np.float32)
    for gene1, gene2 in edges:
        idx1 = gene_to_index[gene1]
        idx2 = gene_to_index[gene2]
        network[idx1, idx2] = 1
        network[idx2, idx1] = 1

    return network, gene_new, set1


def getNetwork(gene, network_path=None):
    return _read_ppi_network(gene, network_path=network_path)


def getNetworkall(gene, network_path=None):
    return _read_ppi_network(gene, network_path=network_path)
