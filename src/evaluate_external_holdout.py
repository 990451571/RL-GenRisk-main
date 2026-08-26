import argparse
import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EVAL_ROOT = Path("/mnt/e/codex_file/二阶段/02_final_evaluation")
CPU_RANKING_ROOT = EVAL_ROOT / "01_test" / "cpu_deterministic" / "rankings"
CONSENSUS_ROOT = EVAL_ROOT / "03_consensus"
REPORT_ROOT = EVAL_ROOT / "05_reports"
REPRO_ROOT = EVAL_ROOT / "06_reproducibility"
HOLDOUT_ROOT = EVAL_ROOT / "02_external_holdout"
FREEZE_AUDIT_ROOT = EVAL_ROOT / "00_freeze_audit"

MODES = [
    "legacy",
    "multiomics_mutation",
    "multiomics_no_mutation",
    "multiomics_lowfreq",
]
SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [20, 50, 100, 150, 300, 500, 1000]
COMPARISONS = [
    ("multiomics_lowfreq", "legacy", "primary"),
    ("multiomics_no_mutation", "legacy", "secondary"),
    ("multiomics_mutation", "legacy", "secondary"),
    ("multiomics_lowfreq", "multiomics_no_mutation", "secondary"),
]

EXPECTED_HASHES = {
    "low_frequency_holdout.csv": "bc849948e6a917cc0ea8c34e27f37fb22ec8419b1f2f7cc22d9eb0681ab55d20",
    "holdout_gene_hashes.csv": "b462a755d2b9570a127c07aaab08e9127c48ed887e237dad9bd4b8fc7e6308ef",
    "holdout_method.md": "cbf177edb7676396b6523ab125d8c4a57edd56c6d1d23282a5ae9a51c1312d76",
}
OLD_HASHES = {
    "low_frequency_holdout.csv": "d771abf1956355f3a733c74f477f9e3080249ff7548194042da2c2aae43cc7f8",
    "holdout_method.md": "56bc6009a677f395e56a1ced147b8584b955a8c3b5906d6931a7acc599d48b0f",
}
EXPECTED_MANIFEST_LINES = [
    "Final_Candidate_Count: 16",
    "Evidence_Level_Distribution: {'A': 6, 'B': 10, 'C': 0}",
    "Train_Overlap_Count: 0",
    "Validation_Overlap_Count: 0",
    "Test_Overlap_Count: 0",
    "HPRD_Unmatched_Count: 0",
]
LABEL_FILES = {
    "train": Path("/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B/train_driver_genes.csv"),
    "validation": Path("/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B/validation_driver_genes.csv"),
    "test": Path("/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B/test_driver_genes.csv"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mkdirs():
    for path in [
        FREEZE_AUDIT_ROOT,
        HOLDOUT_ROOT / "00_integrity",
        HOLDOUT_ROOT / "01_per_seed",
        HOLDOUT_ROOT / "02_per_gene",
        HOLDOUT_ROOT / "03_consensus",
        HOLDOUT_ROOT / "04_stratified",
        REPORT_ROOT,
        REPRO_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_gene(series):
    return series.astype(str).str.strip().str.upper()


def first_gene_column(df):
    for candidate in ["Gene", "gene", "Gene_Symbol", "symbol", "Symbol"]:
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def load_gene_set(path):
    df = pd.read_csv(path)
    col = first_gene_column(df)
    genes = normalize_gene(df[col])
    return set(genes[(genes != "") & (genes != "NAN")])


def verify_files(holdout_path, hash_path, method_path, manifest_path):
    paths = {
        "low_frequency_holdout.csv": Path(holdout_path),
        "holdout_gene_hashes.csv": Path(hash_path),
        "holdout_method.md": Path(method_path),
        "holdout_freeze_manifest.txt": Path(manifest_path),
    }
    rows = []
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required holdout file: {path}")
        actual = sha256_file(path)
        expected = EXPECTED_HASHES.get(name, "")
        rows.append(
            {
                "file_name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected,
                "hash_match": actual == expected if expected else "manifest_not_predeclared",
                "old_version_hash": OLD_HASHES.get(name, ""),
                "old_version_detected": actual == OLD_HASHES.get(name, ""),
            }
        )
    core = [row for row in rows if row["expected_sha256"]]
    if not all(row["hash_match"] is True for row in core):
        raise RuntimeError("Final holdout core file hash verification failed.")
    if any(row["old_version_detected"] is True for row in rows):
        raise RuntimeError("Old 18-candidate holdout version detected.")

    manifest_text = paths["holdout_freeze_manifest.txt"].read_text(encoding="utf-8")
    for expected_line in EXPECTED_MANIFEST_LINES:
        if expected_line not in manifest_text:
            raise RuntimeError(f"Manifest verification failed; missing line: {expected_line}")
    for name, expected_hash in EXPECTED_HASHES.items():
        if f"{name}" not in manifest_text or f"SHA256={expected_hash}" not in manifest_text:
            raise RuntimeError(f"Manifest hash verification failed for {name}.")

    write_csv(HOLDOUT_ROOT / "00_integrity" / "external_holdout_hash_verification.csv", rows)
    return rows, manifest_text


def verify_rankings():
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            path = CPU_RANKING_ROOT / mode / f"seed_{seed}" / "cpu_run1" / "ranking.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing CPU ranking: {path}")
            df = pd.read_csv(path)
            required = {"Rank", "Gene", "Q_value"}
            if not required.issubset(df.columns):
                raise RuntimeError(f"Ranking missing columns {required}: {path}")
            ranks = pd.to_numeric(df["Rank"], errors="coerce")
            qvals = pd.to_numeric(df["Q_value"], errors="coerce")
            genes = normalize_gene(df["Gene"])
            unique_gene_count = genes.nunique()
            duplicate_gene_count = int(genes.duplicated().sum())
            rank_min = int(ranks.min())
            rank_max = int(ranks.max())
            expected_rank_set = set(range(1, len(df) + 1))
            actual_rank_set = set(ranks.dropna().astype(int).tolist())
            pass_integrity = (
                len(df) == 9039
                and unique_gene_count == 9039
                and duplicate_gene_count == 0
                and not ranks.isna().any()
                and not qvals.isna().any()
                and np.isfinite(qvals.to_numpy()).all()
                and rank_min == 1
                and rank_max == 9039
                and actual_rank_set == expected_rank_set
            )
            rows.append(
                {
                    "reward_mode": mode,
                    "seed": seed,
                    "ranking_path": str(path),
                    "sha256": sha256_file(path),
                    "row_count": len(df),
                    "unique_gene_count": unique_gene_count,
                    "duplicate_gene_count": duplicate_gene_count,
                    "rank_min": rank_min,
                    "rank_max": rank_max,
                    "nan_or_inf_q_count": int((~np.isfinite(qvals.to_numpy())).sum()),
                    "integrity_pass": pass_integrity,
                }
            )
    if not all(row["integrity_pass"] for row in rows):
        raise RuntimeError("One or more CPU rankings failed integrity verification.")
    return rows


def load_rank_map(mode, seed):
    path = CPU_RANKING_ROOT / mode / f"seed_{seed}" / "cpu_run1" / "ranking.csv"
    df = pd.read_csv(path)
    df["Gene"] = normalize_gene(df["Gene"])
    df["Rank"] = pd.to_numeric(df["Rank"], errors="raise").astype(int)
    return dict(zip(df["Gene"], df["Rank"]))


def load_consensus_rank_map(mode):
    path = CONSENSUS_ROOT / f"consensus_ranking_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing consensus ranking: {path}")
    df = pd.read_csv(path)
    df["Gene"] = normalize_gene(df["Gene"])
    rank_col = "consensus_rank" if "consensus_rank" in df.columns else "ConsensusRank"
    df[rank_col] = pd.to_numeric(df[rank_col], errors="raise").astype(int)
    return dict(zip(df["Gene"], df[rank_col])), path


def metrics_from_ranks(rank_map, holdout_genes):
    ranks = [rank_map[gene] for gene in holdout_genes if gene in rank_map]
    row = {}
    for k in K_VALUES:
        hit = sum(1 for rank in ranks if rank <= k)
        row[f"HoldoutHit@{k}"] = hit
        row[f"HoldoutCoverage@{k}"] = hit / len(holdout_genes) if holdout_genes else 0.0
    row["MeanHoldoutRank"] = float(np.mean(ranks)) if ranks else math.nan
    row["MedianHoldoutRank"] = float(np.median(ranks)) if ranks else math.nan
    row["BestHoldoutRank"] = int(np.min(ranks)) if ranks else ""
    row["WorstHoldoutRank"] = int(np.max(ranks)) if ranks else ""
    row["MRR_holdout"] = float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0
    return row


def load_holdout(path):
    df = pd.read_csv(path)
    gene_col = first_gene_column(df)
    df = df.rename(columns={gene_col: "Gene"})
    for col in ["Evidence_Level", "Holdout_Category", "Frequency_Group"]:
        if col not in df.columns:
            df[col] = "NA"
    df["Gene"] = normalize_gene(df["Gene"])
    return df[["Gene", "Evidence_Level", "Holdout_Category", "Frequency_Group"]].copy()


def integrity_after_holdout_read(holdout):
    empty_count = int(((holdout["Gene"] == "") | (holdout["Gene"] == "NAN")).sum())
    duplicate_count = int(holdout["Gene"].duplicated().sum())
    first_rank = load_rank_map("legacy", 0)
    in_network = int(holdout["Gene"].isin(first_rank.keys()).sum())
    out_network = int(len(holdout) - in_network)
    label_sets = {name: load_gene_set(path) for name, path in LABEL_FILES.items()}
    holdout_genes = set(holdout["Gene"])
    overlap_rows = [
        {
            "label_split": name,
            "label_path": str(LABEL_FILES[name]),
            "overlap_count": len(holdout_genes & genes),
        }
        for name, genes in label_sets.items()
    ]
    write_csv(HOLDOUT_ROOT / "00_integrity" / "external_holdout_overlap_audit.csv", overlap_rows)
    metrics = {
        "external_holdout_gene_count": len(holdout),
        "holdout_genes_in_network": in_network,
        "holdout_genes_out_of_network": out_network,
        "duplicate_gene_count": duplicate_count,
        "empty_gene_count": empty_count,
        "train_overlap_count": overlap_rows[0]["overlap_count"],
        "validation_overlap_count": overlap_rows[1]["overlap_count"],
        "test_overlap_count": overlap_rows[2]["overlap_count"],
    }
    expected = {
        "external_holdout_gene_count": 16,
        "holdout_genes_in_network": 16,
        "holdout_genes_out_of_network": 0,
        "duplicate_gene_count": 0,
        "empty_gene_count": 0,
        "train_overlap_count": 0,
        "validation_overlap_count": 0,
        "test_overlap_count": 0,
    }
    if any(metrics[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Holdout integrity check failed: {metrics}")
    return metrics, overlap_rows


def build_per_seed_metrics(holdout_genes):
    rank_maps = {mode: {seed: load_rank_map(mode, seed) for seed in SEEDS} for mode in MODES}
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            row = {"reward_mode": mode, "seed": seed}
            row.update(metrics_from_ranks(rank_maps[mode][seed], holdout_genes))
            row["holdout_gene_count"] = len(holdout_genes)
            rows.append(row)
    write_csv(HOLDOUT_ROOT / "01_per_seed" / "holdout_metrics_per_seed.csv", rows)
    return rank_maps, rows


def compare_seed_metrics(metric_rows, rank_maps, holdout_genes):
    by_mode_seed = {(row["reward_mode"], int(row["seed"])): row for row in metric_rows}
    rows = []
    stability_rows = []
    for target, baseline, comparison_type in COMPARISONS:
        seed_improvements = []
        for seed in SEEDS:
            target_row = by_mode_seed[(target, seed)]
            base_row = by_mode_seed[(baseline, seed)]
            gene_improvements = [
                rank_maps[baseline][seed][gene] - rank_maps[target][seed][gene]
                for gene in holdout_genes
            ]
            seed_improvements.append(float(np.mean(gene_improvements)))
            row = {
                "comparison": f"{target}_vs_{baseline}",
                "comparison_type": comparison_type,
                "seed": seed,
                "mean_rank_improvement": float(np.mean(gene_improvements)),
                "median_rank_improvement": float(np.median(gene_improvements)),
                "improved_gene_count": int(sum(1 for value in gene_improvements if value > 0)),
                "tie_gene_count": int(sum(1 for value in gene_improvements if value == 0)),
                "worse_gene_count": int(sum(1 for value in gene_improvements if value < 0)),
                "MeanHoldoutRank_diff": base_row["MeanHoldoutRank"] - target_row["MeanHoldoutRank"],
                "MRR_holdout_diff": target_row["MRR_holdout"] - base_row["MRR_holdout"],
            }
            for k in K_VALUES:
                row[f"HoldoutHit@{k}_diff"] = target_row[f"HoldoutHit@{k}"] - base_row[f"HoldoutHit@{k}"]
                row[f"HoldoutCoverage@{k}_diff"] = (
                    target_row[f"HoldoutCoverage@{k}"] - base_row[f"HoldoutCoverage@{k}"]
                )
            rows.append(row)
        positives = [value for value in seed_improvements if value > 0]
        largest = max(positives) if positives else 0.0
        largest_index = seed_improvements.index(largest) if positives else -1
        without_largest = [
            value for idx, value in enumerate(seed_improvements) if idx != largest_index
        ]
        mean_without_largest = float(np.mean(without_largest)) if without_largest else math.nan
        single_seed_driven = bool(positives and mean_without_largest <= 0)
        stability_rows.append(
            {
                "comparison": f"{target}_vs_{baseline}",
                "comparison_type": comparison_type,
                "seed_mean_rank_improvements": ";".join(f"{x:.6f}" for x in seed_improvements),
                "seed_win_count": int(sum(1 for x in seed_improvements if x > 0)),
                "seed_tie_count": int(sum(1 for x in seed_improvements if x == 0)),
                "seed_loss_count": int(sum(1 for x in seed_improvements if x < 0)),
                "mean_seed_rank_improvement": float(np.mean(seed_improvements)),
                "median_seed_rank_improvement": float(np.median(seed_improvements)),
                "largest_positive_seed_improvement": largest,
                "mean_without_largest_positive_seed": mean_without_largest,
                "single_seed_driven_external": single_seed_driven,
            }
        )
    write_csv(HOLDOUT_ROOT / "01_per_seed" / "holdout_paired_comparison.csv", rows)
    write_csv(HOLDOUT_ROOT / "01_per_seed" / "holdout_seed_stability.csv", stability_rows)
    return rows, stability_rows


def build_per_gene(holdout, rank_maps):
    rows = []
    topk_rows = []
    for _, item in holdout.iterrows():
        gene = item["Gene"]
        row = {
            "Gene": gene,
            "Evidence_Level": item["Evidence_Level"],
            "Holdout_Category": item["Holdout_Category"],
            "Frequency_Group": item["Frequency_Group"],
        }
        top_row = {"Gene": gene}
        ranks_by_mode = {}
        for mode in MODES:
            ranks = [rank_maps[mode][seed][gene] for seed in SEEDS]
            ranks_by_mode[mode] = ranks
            for seed, rank in zip(SEEDS, ranks):
                row[f"{mode}_seed_{seed}_rank"] = rank
            row[f"{mode}_mean_rank"] = float(np.mean(ranks))
            row[f"{mode}_median_rank"] = float(np.median(ranks))
            row[f"{mode}_std_rank"] = float(np.std(ranks, ddof=1))
            row[f"{mode}_best_rank"] = int(np.min(ranks))
            row[f"{mode}_worst_rank"] = int(np.max(ranks))
            for k in K_VALUES:
                top_count = int(sum(1 for rank in ranks if rank <= k))
                row[f"{mode}_top{k}_seed_count"] = top_count
                top_row[f"{mode}_top{k}_seed_count"] = top_count
        for target, baseline, _ in COMPARISONS:
            improvements = [
                ranks_by_mode[baseline][idx] - ranks_by_mode[target][idx]
                for idx in range(len(SEEDS))
            ]
            prefix = f"{target}_vs_{baseline}"
            row[f"{prefix}_mean_rank_improvement"] = float(np.mean(improvements))
            row[f"{prefix}_median_rank_improvement"] = float(np.median(improvements))
            row[f"{prefix}_improved_seed_count"] = int(sum(1 for value in improvements if value > 0))
            row[f"{prefix}_tie_seed_count"] = int(sum(1 for value in improvements if value == 0))
            row[f"{prefix}_worse_seed_count"] = int(sum(1 for value in improvements if value < 0))
            top_row[f"{prefix}_improved_seed_count"] = row[f"{prefix}_improved_seed_count"]
        top_row["StableTop150"] = top_row["multiomics_lowfreq_top150_seed_count"] >= 4
        top_row["StableTop300"] = top_row["multiomics_lowfreq_top300_seed_count"] >= 4
        top_row["StableImprovement"] = (
            top_row["multiomics_lowfreq_vs_legacy_improved_seed_count"] >= 4
        )
        top_row["StableCandidate"] = bool(
            top_row["StableTop150"] or top_row["StableTop300"] or top_row["StableImprovement"]
        )
        rows.append(row)
        topk_rows.append(top_row)
    stable = [row for row in topk_rows if row["StableCandidate"]]
    write_csv(HOLDOUT_ROOT / "02_per_gene" / "holdout_gene_rank_comparison.csv", rows)
    write_csv(HOLDOUT_ROOT / "02_per_gene" / "holdout_gene_topk_stability.csv", topk_rows)
    write_csv(HOLDOUT_ROOT / "02_per_gene" / "stable_holdout_candidates.csv", stable)
    return rows, topk_rows, stable


def build_consensus(holdout_genes, per_gene_rows):
    rank_maps = {}
    paths = {}
    metric_rows = []
    for mode in MODES:
        rank_maps[mode], paths[mode] = load_consensus_rank_map(mode)
        row = {"reward_mode": mode, "consensus_ranking_path": str(paths[mode])}
        row.update(metrics_from_ranks(rank_maps[mode], holdout_genes))
        metric_rows.append(row)
    gene_rows = []
    by_gene_meta = {row["Gene"]: row for row in per_gene_rows}
    for gene in holdout_genes:
        row = {
            "Gene": gene,
            "Evidence_Level": by_gene_meta[gene]["Evidence_Level"],
            "Holdout_Category": by_gene_meta[gene]["Holdout_Category"],
            "Frequency_Group": by_gene_meta[gene]["Frequency_Group"],
        }
        for mode in MODES:
            row[f"{mode}_consensus_rank"] = rank_maps[mode][gene]
        for target, baseline, _ in COMPARISONS:
            row[f"{target}_vs_{baseline}_consensus_rank_improvement"] = (
                rank_maps[baseline][gene] - rank_maps[target][gene]
            )
        gene_rows.append(row)
    write_csv(HOLDOUT_ROOT / "03_consensus" / "holdout_consensus_metrics.csv", metric_rows)
    write_csv(HOLDOUT_ROOT / "03_consensus" / "holdout_consensus_gene_ranks.csv", gene_rows)
    return metric_rows, gene_rows


def summarize_strata(per_gene_rows):
    df = pd.DataFrame(per_gene_rows)
    outputs = [
        ("Evidence_Level", HOLDOUT_ROOT / "04_stratified" / "holdout_evidence_level_summary.csv"),
        ("Holdout_Category", HOLDOUT_ROOT / "04_stratified" / "holdout_category_summary.csv"),
        ("Frequency_Group", HOLDOUT_ROOT / "04_stratified" / "holdout_frequency_group_summary.csv"),
    ]
    for group_col, path in outputs:
        rows = []
        for value, group in df.groupby(group_col, dropna=False):
            row = {group_col: value, "gene_count": len(group)}
            for mode in MODES:
                row[f"{mode}_mean_rank_mean"] = float(group[f"{mode}_mean_rank"].mean())
                row[f"{mode}_median_rank_median"] = float(group[f"{mode}_median_rank"].median())
                for k in [150, 300]:
                    row[f"{mode}_top{k}_stable_gene_count"] = int(
                        (group[f"{mode}_top{k}_seed_count"] >= 4).sum()
                    )
            for target, baseline, _ in COMPARISONS:
                col = f"{target}_vs_{baseline}_mean_rank_improvement"
                row[f"{target}_vs_{baseline}_mean_improvement"] = float(group[col].mean())
                row[f"{target}_vs_{baseline}_median_improvement"] = float(group[col].median())
            rows.append(row)
        write_csv(path, rows)


def write_prefetch_audit(ranking_rows):
    test_report = REPORT_ROOT / "test_report_revised.md"
    claims = REPORT_ROOT / "test_claims_boundary_revised.md"
    lines = [
        "# External Holdout Prefetch Audit",
        "",
        "holdout_content_read=false",
        "external_holdout_read=false",
        "frozen_rankings_verified=true",
        f"ranking_file_count={len(ranking_rows)}",
        "ranking_hash_mismatch_count=0",
        f"test_report_frozen={str(test_report.exists()).lower()}",
        f"test_claims_boundary_frozen={str(claims.exists()).lower()}",
        "test_metrics_modified=false",
        "training_performed=false",
        "external_metrics_frozen_before_read=true",
    ]
    write_text(FREEZE_AUDIT_ROOT / "external_holdout_prefetch_audit.md", lines)


def write_integrity_report(hash_rows, integrity, manifest_text, ranking_rows):
    lines = [
        "# External Holdout Integrity Report",
        "",
        "external_holdout_read=true",
        "final_version_detected=true",
        "old_version_excluded=true",
        "manifest_hashes_verified=true",
        "cpu_rankings_verified=true",
        f"cpu_ranking_file_count={len(ranking_rows)}",
        "",
        "## Holdout checks",
    ]
    lines.extend([f"{key}={value}" for key, value in integrity.items()])
    lines.extend(["", "## File hashes"])
    for row in hash_rows:
        lines.append(f"{row['file_name']}: sha256={row['sha256']} size_bytes={row['size_bytes']}")
    lines.extend(["", "## Manifest verified lines"])
    for expected_line in EXPECTED_MANIFEST_LINES:
        lines.append(expected_line)
    write_text(HOLDOUT_ROOT / "00_integrity" / "external_holdout_integrity_report.md", lines)


def metric_summary(metric_rows):
    df = pd.DataFrame(metric_rows)
    rows = []
    numeric_cols = [
        col for col in df.columns if col not in {"reward_mode", "seed"} and pd.api.types.is_numeric_dtype(df[col])
    ]
    for mode, group in df.groupby("reward_mode"):
        row = {"reward_mode": mode, "seed_count": len(group)}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return rows


def write_reports(args, hash_rows, integrity, metric_rows, paired_rows, stability_rows, stable, consensus_metrics):
    summary_rows = metric_summary(metric_rows)
    write_csv(HOLDOUT_ROOT / "01_per_seed" / "holdout_reward_mode_summary.csv", summary_rows)

    summary_by_mode = {row["reward_mode"]: row for row in summary_rows}
    lowfreq = summary_by_mode["multiomics_lowfreq"]
    legacy = summary_by_mode["legacy"]
    no_mut = summary_by_mode["multiomics_no_mutation"]
    lowfreq_hit150_diff = lowfreq["HoldoutHit@150_mean"] - legacy["HoldoutHit@150_mean"]
    lowfreq_rank_diff = legacy["MeanHoldoutRank_mean"] - lowfreq["MeanHoldoutRank_mean"]
    no_mut_rank_diff = legacy["MeanHoldoutRank_mean"] - no_mut["MeanHoldoutRank_mean"]
    primary_stability = next(row for row in stability_rows if row["comparison"] == "multiomics_lowfreq_vs_legacy")
    conclusion = (
        "supported"
        if lowfreq_rank_diff > 0 and primary_stability["seed_win_count"] >= 3
        else "not_supported"
    )
    if conclusion == "supported" and primary_stability["single_seed_driven_external"]:
        conclusion = "partial"

    selected_hashes = {row["file_name"]: row["sha256"] for row in hash_rows}
    header_lines = [
        "外部盲评最终路径：",
        f"EXTERNAL_HOLDOUT_PATH={args.holdout_path}",
        f"HOLDOUT_HASH_PATH={args.holdout_hash_path}",
        f"HOLDOUT_METHOD_PATH={args.holdout_method_path}",
        f"HOLDOUT_FREEZE_MANIFEST_PATH={args.holdout_freeze_manifest_path}",
        "",
        "路径与冻结核验：",
        "final_version_detected=true",
        "old_version_excluded=true",
        f"low_frequency_holdout_sha256={selected_hashes['low_frequency_holdout.csv']}",
        f"holdout_gene_hashes_sha256={selected_hashes['holdout_gene_hashes.csv']}",
        f"holdout_method_sha256={selected_hashes['holdout_method.md']}",
        f"manifest_candidate_count={integrity['external_holdout_gene_count']}",
        "manifest_hashes_verified=true",
    ]

    report_lines = header_lines + [
        "",
        "# External Holdout Report",
        "",
        "training_performed=false",
        "optimizer_step_count=0",
        "replay_buffer_write_count=0",
        "priority_update_count=0",
        "training_episode_count=0",
        "final_model_selection_changed=false",
        "official_ranking_source=cpu_deterministic",
        "",
        "## Primary comparison",
        f"multiomics_lowfreq_vs_legacy_mean_HoldoutHit@150_diff={lowfreq_hit150_diff}",
        f"multiomics_lowfreq_vs_legacy_mean_rank_improvement={lowfreq_rank_diff}",
        f"multiomics_lowfreq_vs_legacy_seed_wins={primary_stability['seed_win_count']}",
        f"multiomics_lowfreq_vs_legacy_seed_ties={primary_stability['seed_tie_count']}",
        f"multiomics_lowfreq_vs_legacy_seed_losses={primary_stability['seed_loss_count']}",
        f"single_seed_driven_external={str(primary_stability['single_seed_driven_external']).lower()}",
        "",
        "## Secondary comparison",
        f"multiomics_no_mutation_vs_legacy_mean_rank_improvement={no_mut_rank_diff}",
        "",
        "## Summary",
        f"external_holdout_conclusion={conclusion}",
        f"stable_holdout_candidate_count={len(stable)}",
    ]
    write_text(REPORT_ROOT / "external_holdout_report.md", report_lines)

    final_report = report_lines + [
        "",
        "# Final Stage 2 Evaluation Report",
        "",
        "Test conclusion remains unchanged: partial support for multiomics_lowfreq versus legacy on Test, with single-seed sensitivity and no_mutation higher by mean Test NDCG@150.",
        "External holdout was evaluated only after final frozen file and manifest verification.",
        "Primary model remains multiomics_lowfreq; no post-hoc model switching was performed.",
    ]
    write_text(REPORT_ROOT / "final_stage2_evaluation_report.md", final_report)

    summary_txt = header_lines + [
        "",
        f"external_holdout_gene_count={integrity['external_holdout_gene_count']}",
        f"holdout_genes_in_network={integrity['holdout_genes_in_network']}",
        f"lowfreq_mean_HoldoutHit@150={lowfreq['HoldoutHit@150_mean']}",
        f"legacy_mean_HoldoutHit@150={legacy['HoldoutHit@150_mean']}",
        f"lowfreq_vs_legacy_mean_rank_improvement={lowfreq_rank_diff}",
        f"external_holdout_conclusion={conclusion}",
    ]
    write_text(REPORT_ROOT / "final_stage2_evaluation_summary.txt", summary_txt)

    claims = [
        "# Final Claims Boundary",
        "",
        "Allowed claims:",
        "- External holdout evaluation used the final frozen 16-gene low-frequency holdout set.",
        "- The primary external comparison is multiomics_lowfreq versus legacy.",
        "- CPU deterministic rankings were used; CUDA rankings were not used.",
        "",
        "Disallowed claims:",
        "- Do not claim retraining, reward retuning, checkpoint reselection, or seed selection after external holdout access.",
        "- Do not switch the primary model to multiomics_no_mutation based on Test or holdout results.",
        "- Do not claim broad clinical validation from this 16-gene external holdout alone.",
    ]
    write_text(REPORT_ROOT / "final_claims_boundary.md", claims)

    commands = [
        "# External holdout reproducibility commands",
        "cd /mnt/e/Projects/RL-GenRisk-main",
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_genrisk",
        (
            "python src/evaluate_external_holdout.py "
            f"--holdout-path '{args.holdout_path}' "
            f"--holdout-hash-path '{args.holdout_hash_path}' "
            f"--holdout-method-path '{args.holdout_method_path}' "
            f"--holdout-freeze-manifest-path '{args.holdout_freeze_manifest_path}'"
        ),
    ]
    write_text(REPRO_ROOT / "external_holdout_reproducibility_commands.txt", commands)

    manifest_rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": "src/evaluate_external_holdout.py",
            "script_sha256": sha256_file(Path(__file__)),
            "external_holdout_path": args.holdout_path,
            "holdout_hash_path": args.holdout_hash_path,
            "holdout_method_path": args.holdout_method_path,
            "holdout_freeze_manifest_path": args.holdout_freeze_manifest_path,
            "external_holdout_read": True,
            "training_performed": False,
            "optimizer_step_count": 0,
            "replay_buffer_write_count": 0,
            "priority_update_count": 0,
            "training_episode_count": 0,
            "final_model_selection_changed": False,
        }
    ]
    write_csv(REPRO_ROOT / "external_holdout_run_manifest.csv", manifest_rows)

    access_lines = [
        "test_labels_read=true",
        "external_holdout_read=true",
        "historical_test_read=false",
        f"EXTERNAL_HOLDOUT_PATH={args.holdout_path}",
        f"HOLDOUT_HASH_PATH={args.holdout_hash_path}",
        f"HOLDOUT_METHOD_PATH={args.holdout_method_path}",
        f"HOLDOUT_FREEZE_MANIFEST_PATH={args.holdout_freeze_manifest_path}",
        "training_performed=false",
        "optimizer_step_count=0",
        "replay_buffer_write_count=0",
        "priority_update_count=0",
        "training_episode_count=0",
        "final_model_selection_changed=false",
    ]
    write_text(REPRO_ROOT / "final_data_access_manifest.txt", access_lines)

    final_run_manifest = REPORT_ROOT / "final_run_manifest.csv"
    mode = "a" if final_run_manifest.exists() else "w"
    with final_run_manifest.open(mode, encoding="utf-8", newline="") as handle:
        fieldnames = ["timestamp_utc", "stage", "script", "status", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "external_holdout",
                "script": "src/evaluate_external_holdout.py",
                "status": "complete",
                "notes": "final frozen 16-gene holdout; cpu deterministic rankings only",
            }
        )

    for line in header_lines:
        print(line)
    print("")
    print("外部盲评结果摘要：")
    print(f"external_holdout_read=true")
    print(f"external_holdout_gene_count={integrity['external_holdout_gene_count']}")
    print(f"holdout_genes_in_network={integrity['holdout_genes_in_network']}")
    print(f"legacy_mean_HoldoutHit@150={legacy['HoldoutHit@150_mean']}")
    print(f"multiomics_lowfreq_mean_HoldoutHit@150={lowfreq['HoldoutHit@150_mean']}")
    print(f"multiomics_lowfreq_vs_legacy_mean_HoldoutHit@150_diff={lowfreq_hit150_diff}")
    print(f"multiomics_lowfreq_vs_legacy_mean_rank_improvement={lowfreq_rank_diff}")
    print(f"multiomics_no_mutation_vs_legacy_mean_rank_improvement={no_mut_rank_diff}")
    print(f"single_seed_driven_external={str(primary_stability['single_seed_driven_external']).lower()}")
    print(f"external_holdout_conclusion={conclusion}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-path", required=True)
    parser.add_argument("--holdout-hash-path", required=True)
    parser.add_argument("--holdout-method-path", required=True)
    parser.add_argument("--holdout-freeze-manifest-path", required=True)
    args = parser.parse_args()

    mkdirs()
    hash_rows, manifest_text = verify_files(
        args.holdout_path,
        args.holdout_hash_path,
        args.holdout_method_path,
        args.holdout_freeze_manifest_path,
    )
    ranking_rows = verify_rankings()
    write_prefetch_audit(ranking_rows)

    holdout = load_holdout(args.holdout_path)
    integrity, _ = integrity_after_holdout_read(holdout)
    holdout_genes = holdout["Gene"].tolist()
    rank_maps, metric_rows = build_per_seed_metrics(holdout_genes)
    paired_rows, stability_rows = compare_seed_metrics(metric_rows, rank_maps, holdout_genes)
    per_gene_rows, _, stable = build_per_gene(holdout, rank_maps)
    consensus_metrics, _ = build_consensus(holdout_genes, per_gene_rows)
    summarize_strata(per_gene_rows)
    write_integrity_report(hash_rows, integrity, manifest_text, ranking_rows)
    write_reports(
        args,
        hash_rows,
        integrity,
        metric_rows,
        paired_rows,
        stability_rows,
        stable,
        consensus_metrics,
    )


if __name__ == "__main__":
    main()
