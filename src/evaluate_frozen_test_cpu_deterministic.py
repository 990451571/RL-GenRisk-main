import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/mnt/e/Projects/RL-GenRisk-main")
EVAL_ROOT = Path("/mnt/e/codex_file/二阶段/02_final_evaluation")
AUDIT_DIR = EVAL_ROOT / "00_freeze_audit"
TEST_DIR = EVAL_ROOT / "01_test"
CPU_DIR = TEST_DIR / "cpu_deterministic"
CONSENSUS_DIR = EVAL_ROOT / "03_consensus"
REPORT_DIR = EVAL_ROOT / "05_reports"
REPRO_DIR = EVAL_ROOT / "06_reproducibility"
MODES = ["legacy", "multiomics_mutation", "multiomics_no_mutation", "multiomics_lowfreq"]
K_VALUES = [20, 50, 100, 150]
TIE_TOL = 1e-12
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260717


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_cpu(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def worker_load_features():
    import inputall

    train_data, test_data, patients = inputall.getInput(
        "KIRC", mutation_path=str(PROJECT / "data" / "KIRC.txt")
    )
    gene_num = inputall.getGene(patients)
    net, gene_final, gene_name = inputall.getNetwork(
        gene_num, network_path=str(PROJECT / "data" / "HPRD.txt")
    )
    weights = inputall.getWeight(gene_name, weight_path=str(PROJECT / "data" / "weights.txt"))
    for gene in gene_name:
        weights.setdefault(inputall.clean_gene_symbol(gene), 0.0)
    original = inputall.build_original_node_features_raw(net, weights, gene_name, gene_final)
    multi_path = PROJECT / "data" / "processed" / "KIRC_multiomics_3omics.csv"
    df = pd.read_csv(multi_path)
    cols = ["Gene", "Mutation", "Expression", "Methylation"]
    df = df[cols].copy()
    df["Gene"] = df["Gene"].map(inputall.clean_gene_symbol)
    df = df[df["Gene"].notna()].copy()
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[cols[1:]].isna().any().any():
        raise ValueError("Multiomics file contains NaN after numeric conversion.")
    df = df.groupby("Gene", as_index=False)[cols[1:]].mean().set_index("Gene")
    multi = np.zeros((len(gene_name), 3), dtype=np.float32)
    for idx, gene in enumerate(gene_name):
        if gene in df.index:
            multi[idx] = df.loc[gene, cols[1:]].to_numpy(dtype=np.float32)
    features = np.concatenate([original.astype(np.float32), multi], axis=1)
    if features.shape != (9039, 6) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature matrix {features.shape}")
    return net, features, list(gene_name)


def stable_rank_dataframe(q_values, genes):
    df = pd.DataFrame({"Gene": genes, "Q_value": q_values.astype(float)})
    df = df.sort_values(["Q_value", "Gene"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def run_worker(args):
    torch = setup_cpu(args.seed)
    from qfunction import Q_Fun

    start = time.perf_counter()
    net, features, genes = worker_load_features()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    embedding_size = int(checkpoint.get("args", {}).get("embedding_size", 64))
    model = Q_Fun(6, embedding_size, 3, 1e-4, net)
    model.to(torch.device("cpu"))
    key = "online_net_state_dict" if "online_net_state_dict" in checkpoint else "online_state_dict"
    model.load_state_dict(checkpoint[key])
    model.eval()
    state = torch.as_tensor(features, dtype=torch.float32, device=torch.device("cpu"))
    mask = torch.ones(len(genes), dtype=torch.long, device=torch.device("cpu"))
    with torch.inference_mode():
        q_values, _ = model(None, state, mask)
    q_np = q_values.detach().cpu().numpy()
    if not np.isfinite(q_np).all():
        raise FloatingPointError("CPU inference produced NaN/Inf Q values.")
    ranking = stable_rank_dataframe(q_np, genes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "ranking.csv"
    config_path = output_dir / "run_config.json"
    log_path = output_dir / "runtime_log.txt"
    ranking.to_csv(ranking_path, index=False)
    config = {
        "reward_mode": args.reward_mode,
        "seed": args.seed,
        "run_id": args.run_id,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": sha256_file(args.checkpoint_path),
        "device": "cpu",
        "num_threads": 1,
        "num_interop_threads": 1,
        "deterministic_algorithms": True,
        "ranking_path": str(ranking_path),
        "ranking_sha256": sha256_file(ranking_path),
        "runtime_seconds": time.perf_counter() - start,
        "training_performed": False,
        "optimizer_step_count": 0,
        "replay_buffer_write_count": 0,
        "priority_update_count": 0,
        "training_episode_count": 0,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_path.write_text("CPU deterministic inference completed.\n", encoding="utf-8")


def read_labels(path):
    import inputall

    labels = []
    seen = set()
    invalid = 0
    empty = 0
    duplicate = 0
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "," if "," in sample else ("\t" if "\t" in sample else None)
        rows = csv.reader(handle, delimiter=delimiter) if delimiter else (line.split() for line in handle)
        for row_index, row in enumerate(rows):
            if not row:
                empty += 1
                continue
            gene = inputall.clean_gene_symbol(row[0])
            if row_index == 0 and gene in {"GENE", "GENE_SYMBOL", "GENE SYMBOL"}:
                continue
            if gene is None:
                invalid += 1
                continue
            if gene in seen:
                duplicate += 1
            else:
                seen.add(gene)
                labels.append(gene)
    return labels, invalid, empty, duplicate


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def metrics_for_rows(rows, labels, labels_in_network_count):
    labels = set(labels)
    rank_by_gene = {row["Gene"]: rank for rank, row in enumerate(rows, start=1)}
    metrics = {}
    for k in K_VALUES:
        top = rows[:k]
        hits = sum(1 for row in top if row["Gene"] in labels)
        rel = [1 if row["Gene"] in labels else 0 for row in top]
        ideal_hits = min(k, labels_in_network_count, len(rows))
        idcg = dcg([1] * ideal_hits + [0] * (len(top) - ideal_hits))
        metrics[f"NDCG@{k}"] = dcg(rel) / idcg if idcg > 0 else 0.0
        metrics[f"HitCount@{k}"] = hits
        metrics[f"Precision@{k}"] = hits / k
        metrics[f"Recall@{k}"] = hits / labels_in_network_count if labels_in_network_count else 0.0
    ranks = sorted(rank_by_gene[gene] for gene in labels if gene in rank_by_gene)
    metrics["MRR"] = float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0
    metrics["MeanRank"] = float(np.mean(ranks)) if ranks else None
    metrics["MedianRank"] = float(np.median(ranks)) if ranks else None
    return metrics


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_REPS, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(metric_rows, columns):
    rows = []
    for mode in MODES:
        subset = [row for row in metric_rows if row["reward_mode"] == mode]
        item = {"reward_mode": mode, "run_count": len(subset)}
        for col in columns:
            vals = [float(row[col]) for row in subset]
            lo, hi = bootstrap_ci(vals)
            item[f"{col}_mean"] = float(np.mean(vals))
            item[f"{col}_sample_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            item[f"{col}_median"] = float(np.median(vals))
            item[f"{col}_min"] = float(np.min(vals))
            item[f"{col}_max"] = float(np.max(vals))
            item[f"{col}_bootstrap_95_ci_lower"] = lo
            item[f"{col}_bootstrap_95_ci_upper"] = hi
        rows.append(item)
    return rows


def compare_rankings(run1, run2, manifest_row):
    df1 = pd.read_csv(run1)
    df2 = pd.read_csv(run2)
    rank1 = dict(zip(df1["Gene"], df1["Rank"]))
    rank2 = dict(zip(df2["Gene"], df2["Rank"]))
    q1 = dict(zip(df1["Gene"], df1["Q_value"]))
    q2 = dict(zip(df2["Gene"], df2["Q_value"]))
    genes1 = df1["Gene"].tolist()
    genes2 = df2["Gene"].tolist()
    common = sorted(set(genes1) & set(genes2))
    diffs = np.asarray([abs(q1[g] - q2[g]) for g in common], dtype=float)
    shifts = np.asarray([abs(rank1[g] - rank2[g]) for g in common], dtype=float)
    return {
        "reward_mode": manifest_row["reward_mode"],
        "seed": int(manifest_row["seed"]),
        "checkpoint_sha256": manifest_row["checkpoint_sha256"],
        "run1_ranking_path": str(run1),
        "run2_ranking_path": str(run2),
        "run1_ranking_sha256": sha256_file(run1),
        "run2_ranking_sha256": sha256_file(run2),
        "run1_gene_count": len(df1),
        "run2_gene_count": len(df2),
        "run1_unique_gene_count": df1["Gene"].nunique(),
        "run2_unique_gene_count": df2["Gene"].nunique(),
        "run1_duplicate_gene_count": len(df1) - df1["Gene"].nunique(),
        "run2_duplicate_gene_count": len(df2) - df2["Gene"].nunique(),
        "run1_nan_inf_count": int((~np.isfinite(df1["Q_value"].to_numpy(dtype=float))).sum()),
        "run2_nan_inf_count": int((~np.isfinite(df2["Q_value"].to_numpy(dtype=float))).sum()),
        "gene_set_identical": set(genes1) == set(genes2),
        "gene_order_identical": genes1 == genes2,
        "rank_identical": genes1 == genes2,
        "q_value_exactly_identical": bool(np.array_equal(df1["Q_value"].to_numpy(), df2["Q_value"].to_numpy())),
        "max_abs_q_difference": float(diffs.max()) if diffs.size else None,
        "mean_abs_q_difference": float(diffs.mean()) if diffs.size else None,
        "maximum_absolute_rank_shift": float(shifts.max()) if shifts.size else None,
        "ranking_structural_integrity": len(df1) == 9039 and len(df2) == 9039 and df1["Gene"].nunique() == 9039 and df2["Gene"].nunique() == 9039 and np.isfinite(df1["Q_value"]).all() and np.isfinite(df2["Q_value"]).all(),
    }


def run_parent(args):
    manifest = pd.read_csv(AUDIT_DIR / "final_evaluation_manifest.csv")
    if set(manifest["reward_mode"]) != set(MODES) or sorted(manifest["seed"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Manifest does not cover all modes and seeds.")
    # checkpoint hash verification before running.
    mismatches = []
    for _, row in manifest.iterrows():
        current = sha256_file(row["checkpoint_path"])
        if current != row["checkpoint_sha256"]:
            mismatches.append((row["reward_mode"], int(row["seed"]), current, row["checkpoint_sha256"]))
    if mismatches:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {mismatches}")

    CPU_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["rankings", "run_configs", "logs", "determinism_checks", "per_seed_metrics", "summaries"]:
        (CPU_DIR / sub).mkdir(parents=True, exist_ok=True)

    for _, row in manifest.sort_values(["reward_mode", "seed"]).iterrows():
        for run_id in ["cpu_run1", "cpu_run2"]:
            out = CPU_DIR / "rankings" / row["reward_mode"] / f"seed_{int(row['seed'])}" / run_id
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--reward-mode",
                row["reward_mode"],
                "--seed",
                str(int(row["seed"])),
                "--checkpoint-path",
                row["checkpoint_path"],
                "--run-id",
                run_id,
                "--output-dir",
                str(out),
            ]
            log_path = CPU_DIR / "logs" / f"{row['reward_mode']}_seed{int(row['seed'])}_{run_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"CPU worker failed for {row['reward_mode']} seed {int(row['seed'])} {run_id}; see {log_path}")

    det_rows = []
    int_rows = []
    for _, row in manifest.sort_values(["reward_mode", "seed"]).iterrows():
        base = CPU_DIR / "rankings" / row["reward_mode"] / f"seed_{int(row['seed'])}"
        run1 = base / "cpu_run1" / "ranking.csv"
        run2 = base / "cpu_run2" / "ranking.csv"
        comp = compare_rankings(run1, run2, row)
        passed = bool(comp["gene_order_identical"] and comp["ranking_structural_integrity"])
        comp["deterministic_passed"] = passed
        det_rows.append(comp)
        int_rows.append({
            "reward_mode": row["reward_mode"],
            "seed": int(row["seed"]),
            "ranking_path": str(run1),
            "gene_count": comp["run1_gene_count"],
            "unique_gene_count": comp["run1_unique_gene_count"],
            "duplicate_gene_count": comp["run1_duplicate_gene_count"],
            "missing_gene_count": 9039 - comp["run1_gene_count"],
            "nan_inf_count": comp["run1_nan_inf_count"],
            "rank_min": 1,
            "rank_max": int(pd.read_csv(run1)["Rank"].max()),
            "rank_continuous": pd.read_csv(run1)["Rank"].tolist() == list(range(1, 9040)),
            "ranking_structural_integrity": comp["ranking_structural_integrity"],
        })
    write_csv(CPU_DIR / "cpu_determinism_report.csv", det_rows)
    write_csv(CPU_DIR / "determinism_checks" / "cpu_determinism_report.csv", det_rows)
    write_csv(CPU_DIR / "cpu_integrity_report.csv", int_rows)
    if not all(row["deterministic_passed"] for row in det_rows):
        raise RuntimeError("CPU determinism failed; metrics not generated.")
    if not all(row["ranking_structural_integrity"] for row in int_rows):
        raise RuntimeError("CPU structural integrity failed; metrics not generated.")

    labels, invalid_count, empty_count, duplicate_count = read_labels(args.test_label_path)
    # Use CPU ranking gene set as network set.
    first_ranking = pd.read_csv(CPU_DIR / "rankings" / "legacy" / "seed_0" / "cpu_run1" / "ranking.csv")
    gene_set = set(first_ranking["Gene"])
    labels_in_network = [gene for gene in labels if gene in gene_set]
    labels_out = len(set(labels) - set(labels_in_network))
    test_sha = sha256_file(args.test_label_path)
    access_lines = [
        "test_labels_read=true",
        f"test_first_read_timestamp={datetime.now(timezone.utc).isoformat()}",
        f"test_label_file_path={args.test_label_path}",
        f"test_label_file_sha256={test_sha}",
        f"test_label_count_total={len(labels)}",
        f"test_label_count_in_network={len(labels_in_network)}",
        f"test_label_count_out_of_network={labels_out}",
        f"test_label_invalid_count={invalid_count}",
        f"test_label_empty_count={empty_count}",
        f"test_label_duplicate_count={duplicate_count}",
        "external_holdout_read=false",
        "historical_test_read=false",
        "training_performed=false",
        "optimizer_step_count=0",
        "replay_buffer_write_count=0",
        "priority_update_count=0",
        "training_episode_count=0",
        "primary_evaluation_device=cpu",
    ]
    (CPU_DIR / "logs" / "test_data_access_manifest.txt").write_text("\n".join(access_lines) + "\n", encoding="utf-8")
    (REPRO_DIR / "test_data_access_manifest.txt").write_text("\n".join(access_lines) + "\n", encoding="utf-8")
    label_report = [{
        "test_label_file_path": args.test_label_path,
        "test_label_file_sha256": test_sha,
        "test_labels_total": len(labels),
        "test_labels_in_network": len(labels_in_network),
        "test_labels_out_of_network": labels_out,
        "test_label_invalid_count": invalid_count,
        "test_label_empty_count": empty_count,
        "test_label_duplicate_count": duplicate_count,
    }]
    write_csv(CPU_DIR / "test_label_integrity_report.csv", label_report)

    metric_rows = []
    for _, row in manifest.sort_values(["reward_mode", "seed"]).iterrows():
        ranking_path = CPU_DIR / "rankings" / row["reward_mode"] / f"seed_{int(row['seed'])}" / "cpu_run1" / "ranking.csv"
        ranking = pd.read_csv(ranking_path).sort_values("Rank")
        rows = [{"Gene": gene} for gene in ranking["Gene"]]
        metrics = metrics_for_rows(rows, labels_in_network, len(labels_in_network))
        metrics.update({
            "reward_mode": row["reward_mode"],
            "seed": int(row["seed"]),
            "checkpoint_path": row["checkpoint_path"],
            "ranking_path": str(ranking_path),
            "test_labels_total": len(labels),
            "test_labels_in_network": len(labels_in_network),
            "test_labels_out_of_network": labels_out,
        })
        metric_rows.append(metrics)
    write_csv(CPU_DIR / "test_metrics_per_seed.csv", metric_rows)
    write_csv(CPU_DIR / "per_seed_metrics" / "test_metrics_per_seed.csv", metric_rows)

    value_cols = [f"NDCG@{k}" for k in K_VALUES] + [f"HitCount@{k}" for k in K_VALUES] + [f"Precision@{k}" for k in K_VALUES] + [f"Recall@{k}" for k in K_VALUES] + ["MRR", "MeanRank", "MedianRank"]
    summary_rows = summarize(metric_rows, value_cols)
    write_csv(CPU_DIR / "test_reward_mode_summary.csv", summary_rows)
    write_csv(CPU_DIR / "summaries" / "test_reward_mode_summary.csv", summary_rows)

    legacy = {row["seed"]: row for row in metric_rows if row["reward_mode"] == "legacy"}
    pair_rows = []
    for mode in [m for m in MODES if m != "legacy"]:
        for metric in [f"NDCG@{k}" for k in K_VALUES] + [f"HitCount@{k}" for k in K_VALUES]:
            seed_items = []
            for row in [r for r in metric_rows if r["reward_mode"] == mode]:
                base = legacy[row["seed"]]
                diff = float(row[metric]) - float(base[metric])
                result = "tie" if abs(diff) <= TIE_TOL else ("win" if diff > 0 else "loss")
                seed_items.append((row, base, diff, result))
            diffs = [x[2] for x in seed_items]
            results = [x[3] for x in seed_items]
            wide = {
                "candidate": mode,
                "metric": metric,
                "wins": results.count("win"),
                "ties": results.count("tie"),
                "losses": results.count("loss"),
                "mean_difference": float(np.mean(diffs)),
                "median_difference": float(np.median(diffs)),
                "min_difference": float(np.min(diffs)),
                "max_difference": float(np.max(diffs)),
                "tie_tolerance": TIE_TOL,
            }
            for row, base, diff, result in seed_items:
                seed = int(row["seed"])
                wide[f"seed{seed}_legacy"] = base[metric]
                wide[f"seed{seed}_candidate"] = row[metric]
                wide[f"seed{seed}_difference"] = diff
            pair_rows.append(wide)
    write_csv(CPU_DIR / "test_paired_comparison_vs_legacy.csv", pair_rows)
    write_csv(CPU_DIR / "summaries" / "test_paired_comparison_vs_legacy.csv", pair_rows)

    topk_rows = []
    for mode in MODES:
        subset = [row for row in metric_rows if row["reward_mode"] == mode]
        for k in K_VALUES:
            topk_rows.append({
                "reward_mode": mode,
                "k": k,
                "mean_ndcg": float(np.mean([row[f"NDCG@{k}"] for row in subset])),
                "mean_hit": float(np.mean([row[f"HitCount@{k}"] for row in subset])),
                "mean_precision": float(np.mean([row[f"Precision@{k}"] for row in subset])),
                "mean_recall": float(np.mean([row[f"Recall@{k}"] for row in subset])),
            })
    write_csv(CPU_DIR / "test_topk_comparison.csv", topk_rows)
    write_csv(CPU_DIR / "summaries" / "test_topk_comparison.csv", topk_rows)

    consensus_metrics = []
    for mode in MODES:
        tables = []
        for seed in range(5):
            ranking_path = CPU_DIR / "rankings" / mode / f"seed_{seed}" / "cpu_run1" / "ranking.csv"
            tables.append(pd.read_csv(ranking_path)[["Gene", "Rank"]].rename(columns={"Rank": f"rank_seed{seed}"}))
        merged = tables[0]
        for table in tables[1:]:
            merged = merged.merge(table, on="Gene", how="inner")
        seed_cols = [col for col in merged.columns if col.startswith("rank_seed")]
        merged["median_rank"] = merged[seed_cols].median(axis=1)
        merged["mean_rank"] = merged[seed_cols].mean(axis=1)
        merged["min_rank"] = merged[seed_cols].min(axis=1)
        merged["max_rank"] = merged[seed_cols].max(axis=1)
        merged["rank_std"] = merged[seed_cols].std(axis=1)
        merged = merged.sort_values(["median_rank", "mean_rank", "Gene"]).reset_index(drop=True)
        merged.insert(0, "consensus_rank", np.arange(1, len(merged) + 1))
        out_path = CONSENSUS_DIR / f"consensus_ranking_{mode}.csv"
        merged[["Gene", "median_rank", "mean_rank", "min_rank", "max_rank", "rank_std", "consensus_rank"]].to_csv(out_path, index=False)
        rows = [{"Gene": gene} for gene in merged["Gene"]]
        cm = metrics_for_rows(rows, labels_in_network, len(labels_in_network))
        cm.update({"reward_mode": mode, "consensus_ranking_path": str(out_path)})
        consensus_metrics.append(cm)
    write_csv(CONSENSUS_DIR / "test_consensus_metrics.csv", consensus_metrics)

    low_pair = next(row for row in pair_rows if row["candidate"] == "multiomics_lowfreq" and row["metric"] == "NDCG@150")
    low_hit = next(row for row in pair_rows if row["candidate"] == "multiomics_lowfreq" and row["metric"] == "HitCount@150")
    diffs = [low_pair[f"seed{seed}_difference"] for seed in range(5)]
    positive = [(idx, value) for idx, value in enumerate(diffs) if value > TIE_TOL]
    if positive:
        largest_idx, largest_val = max(positive, key=lambda item: item[1])
        reduced = [value for idx, value in enumerate(diffs) if idx != largest_idx]
        reduced_mean = float(np.mean(reduced))
    else:
        largest_idx, largest_val, reduced_mean = None, 0.0, float(np.mean(diffs))
    single_seed_driven = bool(positive and reduced_mean <= 0.0)

    legacy_summary = next(row for row in summary_rows if row["reward_mode"] == "legacy")
    low_summary = next(row for row in summary_rows if row["reward_mode"] == "multiomics_lowfreq")
    ndcg_top_better = sum(
        1 for k in [50, 100, 150]
        if next(row for row in topk_rows if row["reward_mode"] == "multiomics_lowfreq" and row["k"] == k)["mean_ndcg"] >
        next(row for row in topk_rows if row["reward_mode"] == "legacy" and row["k"] == k)["mean_ndcg"]
    )
    support = (
        low_summary["NDCG@150_mean"] > legacy_summary["NDCG@150_mean"]
        and int(low_pair["wins"]) >= 3
        and low_summary["HitCount@150_mean"] >= legacy_summary["HitCount@150_mean"]
        and ndcg_top_better >= 2
        and not single_seed_driven
    )
    if support:
        conclusion = "支持"
        validation_trend_supported = "true"
    elif low_summary["NDCG@150_mean"] <= legacy_summary["NDCG@150_mean"] or int(low_pair["losses"]) > int(low_pair["wins"]):
        conclusion = "不支持"
        validation_trend_supported = "false"
    else:
        conclusion = "部分支持"
        validation_trend_supported = "partial"

    val_summary = pd.read_csv("/mnt/e/codex_file/二阶段/01_reward改造/output/validation/reward_mode_summary.csv")
    val_order = val_summary.sort_values("NDCG@150_mean", ascending=False)["reward_mode"].tolist()
    test_order = pd.DataFrame(summary_rows).sort_values("NDCG@150_mean", ascending=False)["reward_mode"].tolist()

    report = [
        "# Revised Frozen Test Report",
        "",
        "CUDA/PyG parallel graph aggregation produced tiny numeric nondeterminism before any Test metric summary was generated. The protocol was amended before metric generation to use single-thread CPU deterministic inference for all frozen checkpoints.",
        "",
        "Model, checkpoint set, input features, labels, reward, Top-K values, metric formulas, and consensus method were unchanged.",
        "",
        f"test_labels_total={len(labels)}",
        f"test_labels_in_network={len(labels_in_network)}",
        f"test_labels_out_of_network={labels_out}",
        "Out-of-network labels are reported separately and are not treated as ordinary ranked misses.",
        "",
        f"cpu_evaluated_checkpoints={len(det_rows)}",
        f"cpu_deterministic_pass_count={sum(1 for row in det_rows if row['deterministic_passed'])}",
        f"cpu_structural_integrity_pass_count={sum(1 for row in int_rows if row['ranking_structural_integrity'])}",
        "",
        "## Mean Test NDCG@150",
        *[f"- {row['reward_mode']}: {row['NDCG@150_mean']}" for row in summary_rows],
        "",
        "## Mean Test HitCount@150",
        *[f"- {row['reward_mode']}: {row['HitCount@150_mean']}" for row in summary_rows],
        "",
        "## lowfreq vs legacy",
        f"wins/ties/losses={int(low_pair['wins'])}/{int(low_pair['ties'])}/{int(low_pair['losses'])}",
        f"seed_differences={diffs}",
        f"mean_ndcg_150_difference={low_pair['mean_difference']}",
        f"median_ndcg_150_difference={low_pair['median_difference']}",
        f"mean_hit_150_difference={low_hit['mean_difference']}",
        f"largest_positive_seed_difference={largest_val}",
        f"mean_difference_without_largest_positive_seed={reduced_mean}",
        f"single_seed_driven={single_seed_driven}",
        "",
        "## Mode-order comparison",
        f"validation_order_by_mean_NDCG@150={val_order}",
        f"test_order_by_mean_NDCG@150={test_order}",
        "",
        "## Conclusion",
        f"validation_trend_supported={validation_trend_supported}",
        f"conclusion={conclusion}",
        "",
        "The result is descriptive over n=5 seeds. No statistical-significance claim is made.",
        "",
        "multiomics_no_mutation removes direct mutation reward only; Mutation remains in hybrid6_raw input. multiomics_lowfreq applies a bounded rarity bonus only to Train drivers during training, not to unknown Test genes.",
    ]
    (REPORT_DIR / "test_report_revised.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    claims = [
        "# Revised Test Claims Boundary",
        "",
        "Allowed: describe frozen CPU deterministic Test rankings and fixed-mode comparisons across all five seeds.",
        "Not allowed: claim statistical significance from five seeds alone, new cancer-gene discovery, clinical validity, complete mutation independence, or direct reward of unknown low-frequency candidates.",
        "The CPU protocol amendment was made before Test metrics and was based on label-independent numeric determinism, not performance.",
    ]
    (REPORT_DIR / "test_claims_boundary_revised.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    code_diff = [
        "# CPU Evaluation Code Diff",
        "",
        "Allowed semantic changes relative to the CUDA evaluator: device is CPU, each inference is an independent process, single-thread deterministic PyTorch settings are applied, and CPU run1/run2 determinism audit fields are added.",
        "",
        "Unchanged evaluation semantics: checkpoint set, hybrid6_raw features, PPI graph, Q network architecture, ranking target, Test labels, NDCG/HitCount/Precision/Recall/MRR/MeanRank/MedianRank formulas, Top-K values, and median-rank consensus method.",
    ]
    (REPORT_DIR / "cpu_evaluation_code_diff.md").write_text("\n".join(code_diff) + "\n", encoding="utf-8")

    # CPU vs CUDA rank stability for run1, label-free.
    stability = []
    for _, row in manifest.iterrows():
        mode = row["reward_mode"]; seed = int(row["seed"])
        cpu = pd.read_csv(CPU_DIR / "rankings" / mode / f"seed_{seed}" / "cpu_run1" / "ranking.csv")
        cuda_path = Path("/mnt/e/codex_file/二阶段/02_final_evaluation/01_test/rankings") / f"test_ranking_{mode}_seed{seed}.csv"
        if cuda_path.exists():
            cuda = pd.read_csv(cuda_path)
            cr = dict(zip(cpu["Gene"], cpu["Rank"])); gr = dict(zip(cuda["Gene"], cuda["Rank"]))
            common = sorted(set(cr) & set(gr))
            shifts = np.asarray([abs(cr[g] - gr[g]) for g in common], dtype=float)
            stability.append({
                "reward_mode": mode,
                "seed": seed,
                "top20_overlap": len(set(cpu.head(20)["Gene"]) & set(cuda.head(20)["Gene"])),
                "top50_overlap": len(set(cpu.head(50)["Gene"]) & set(cuda.head(50)["Gene"])),
                "top100_overlap": len(set(cpu.head(100)["Gene"]) & set(cuda.head(100)["Gene"])),
                "top150_overlap": len(set(cpu.head(150)["Gene"]) & set(cuda.head(150)["Gene"])),
                "median_absolute_rank_shift": float(np.median(shifts)),
                "maximum_absolute_rank_shift": float(np.max(shifts)),
                "spearman_rank_correlation": float(pd.Series([cr[g] for g in common]).corr(pd.Series([gr[g] for g in common]), method="spearman")),
            })
    write_csv(CPU_DIR / "cpu_vs_cuda_rank_stability.csv", stability)

    repro = [
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_genrisk",
        "cd /mnt/e/Projects/RL-GenRisk-main",
        f"python src/evaluate_frozen_test_cpu_deterministic.py --test-label-path {args.test_label_path}",
    ]
    (REPRO_DIR / "test_reproducibility_commands.txt").write_text("\n".join(repro) + "\n", encoding="utf-8")
    run_manifest = [{"path": str(path), "size_bytes": path.stat().st_size} for path in sorted(EVAL_ROOT.rglob("*")) if path.is_file()]
    write_csv(REPRO_DIR / "test_run_manifest.csv", run_manifest)
    write_csv(REPORT_DIR / "final_run_manifest.csv", run_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reward-mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--test-label-path")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        if not args.test_label_path:
            raise ValueError("--test-label-path is required in parent mode.")
        run_parent(args)


if __name__ == "__main__":
    main()
