import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import inputall
from qfunction import Q_Fun


PROJECT = Path("/mnt/e/Projects/RL-GenRisk-main")
EVAL_ROOT = Path("/mnt/e/codex_file/二阶段/02_final_evaluation")
AUDIT_DIR = EVAL_ROOT / "00_freeze_audit"
TEST_DIR = EVAL_ROOT / "01_test"
CONSENSUS_DIR = EVAL_ROOT / "03_consensus"
REPORT_DIR = EVAL_ROOT / "05_reports"
REPRO_DIR = EVAL_ROOT / "06_reproducibility"
FIG_DIR = TEST_DIR / "figures"
MODES = ["legacy", "multiomics_mutation", "multiomics_no_mutation", "multiomics_lowfreq"]
K_VALUES = [20, 50, 100, 150]
TIE_TOL = 1e-12
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260717


def sha256_file(path):
    path = normalize_path(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(path):
    text = str(path)
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_labels(path):
    labels = []
    invalid = 0
    seen = set()
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "," if "," in sample else ("\t" if "\t" in sample else None)
        rows = csv.reader(handle, delimiter=delimiter) if delimiter else (line.split() for line in handle)
        for row_index, row in enumerate(rows):
            if not row:
                invalid += 1
                continue
            gene = inputall.clean_gene_symbol(row[0])
            if row_index == 0 and gene in {"GENE", "GENE_SYMBOL", "GENE SYMBOL"}:
                continue
            if gene is None:
                invalid += 1
                continue
            if gene not in seen:
                seen.add(gene)
                labels.append(gene)
    return labels, invalid


def load_features():
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
    multiomics_path = PROJECT / "data" / "processed" / "KIRC_multiomics_3omics.csv"
    df = pd.read_csv(multiomics_path)
    required = ["Gene", "Mutation", "Expression", "Methylation"]
    df = df[required].copy()
    df["Gene"] = df["Gene"].map(inputall.clean_gene_symbol)
    df = df[df["Gene"].notna()].copy()
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[required[1:]].isna().any().any():
        raise ValueError("Multiomics feature file contains NaN after numeric conversion.")
    df = df.groupby("Gene", as_index=False)[required[1:]].mean().set_index("Gene")
    multi = np.zeros((len(gene_name), 3), dtype=np.float32)
    for idx, gene in enumerate(gene_name):
        if gene in df.index:
            multi[idx] = df.loc[gene, required[1:]].to_numpy(dtype=np.float32)
    features = np.concatenate([original.astype(np.float32), multi], axis=1)
    if features.shape != (9039, 6) or not np.isfinite(features).all():
        raise ValueError(f"Invalid hybrid6_raw feature matrix: shape={features.shape}")
    return net, features, list(gene_name)


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
            item[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            item[f"{col}_median"] = float(np.median(vals))
            item[f"{col}_min"] = float(np.min(vals))
            item[f"{col}_max"] = float(np.max(vals))
            item[f"{col}_bootstrap_ci95_low"] = lo
            item[f"{col}_bootstrap_ci95_high"] = hi
        rows.append(item)
    return rows


def assert_code_hashes_current():
    rows = pd.read_csv(AUDIT_DIR / "evaluation_code_hashes.csv")
    mismatches = []
    for _, row in rows.iterrows():
        current = sha256_file(row["file"])
        if current.upper() != str(row["sha256"]).upper():
            mismatches.append((row["file"], row["sha256"], current))
    if mismatches:
        raise RuntimeError(f"Evaluation code hash mismatch before Test read: {mismatches}")


def assert_checkpoint_hashes_current(manifest):
    mismatches = []
    for _, row in manifest.iterrows():
        current = sha256_file(row["checkpoint_path"])
        if current != row["checkpoint_sha256"]:
            mismatches.append((row["reward_mode"], int(row["seed"]), row["checkpoint_sha256"], current))
    if mismatches:
        raise RuntimeError(f"Checkpoint hash mismatch before Test read: {mismatches}")


def infer_checkpoint(checkpoint_path, net, features, genes, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embedding_size = int(payload.get("args", {}).get("embedding_size", 64))
    model = Q_Fun(6, embedding_size, 3, 1e-4, net)
    model.to(device)
    key = "online_net_state_dict" if "online_net_state_dict" in payload else "online_state_dict"
    model.load_state_dict(payload[key])
    model.eval()
    state = torch.as_tensor(features, dtype=torch.float32, device=device)
    mask = torch.ones(len(genes), dtype=torch.long, device=device)
    with torch.no_grad():
        q1, _ = model(None, state, mask)
        q2, _ = model(None, state, mask)
    q1_np = q1.detach().cpu().numpy()
    q2_np = q2.detach().cpu().numpy()
    if not np.isfinite(q1_np).all() or not np.isfinite(q2_np).all():
        raise FloatingPointError(f"Non-finite Q values for checkpoint {checkpoint_path}")
    return q1_np, q2_np


def sorted_ranking(q_values, genes):
    rows = [{"Gene": gene, "Q_value": float(q_values[idx])} for idx, gene in enumerate(genes)]
    rows.sort(key=lambda item: (-item["Q_value"], item["Gene"]))
    return rows


def write_ranking(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Rank", "Gene", "Q_value"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"Rank": rank, **row})


def save_plot(path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(metric_rows, topk_rows, pair_rows):
    df = pd.DataFrame(metric_rows)
    topk = pd.DataFrame(topk_rows)
    pairs = pd.DataFrame(pair_rows)
    val = pd.read_csv("/mnt/e/codex_file/二阶段/01_reward改造/output/validation/reward_mode_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in MODES:
        ax.plot(K_VALUES, [topk[(topk.reward_mode == mode) & (topk.k == k)]["mean_ndcg"].iloc[0] for k in K_VALUES], marker="o", label=mode)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean Test NDCG")
    ax.legend(fontsize=8)
    save_plot(FIG_DIR / "test_ndcg_topk_by_mode.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in MODES:
        ax.plot(K_VALUES, [topk[(topk.reward_mode == mode) & (topk.k == k)]["mean_hit"].iloc[0] for k in K_VALUES], marker="o", label=mode)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean Test HitCount")
    ax.legend(fontsize=8)
    save_plot(FIG_DIR / "test_hit_topk_by_mode.png", fig)

    low = pairs[(pairs.reward_mode == "multiomics_lowfreq") & (pairs.metric == "NDCG@150")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(low["seed"].astype(str), low["difference"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("NDCG@150 difference")
    save_plot(FIG_DIR / "lowfreq_vs_legacy_seed_ndcg150_diff.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODES))
    val_means = [val[val.reward_mode == mode]["NDCG@150_mean"].iloc[0] for mode in MODES]
    test_means = [df[df.reward_mode == mode]["NDCG@150"].mean() for mode in MODES]
    ax.bar(x - 0.18, val_means, width=0.36, label="Validation")
    ax.bar(x + 0.18, test_means, width=0.36, label="Test")
    ax.set_xticks(x, MODES, rotation=20, ha="right")
    ax.set_ylabel("Mean NDCG@150")
    ax.legend()
    save_plot(FIG_DIR / "validation_vs_test_mean_ndcg150.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([df[df.reward_mode == mode]["NDCG@150"].tolist() for mode in MODES], labels=MODES)
    ax.set_xticklabels(MODES, rotation=20, ha="right")
    ax.set_ylabel("Test NDCG@150")
    save_plot(FIG_DIR / "test_ndcg150_seed_distribution.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in [m for m in MODES if m != "legacy"]:
        diffs = [pairs[(pairs.reward_mode == mode) & (pairs.metric == f"NDCG@{k}")]["difference"].mean() for k in K_VALUES]
        ax.plot(K_VALUES, diffs, marker="o", label=mode)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean NDCG difference vs legacy")
    ax.legend(fontsize=8)
    save_plot(FIG_DIR / "candidate_mean_ndcg_diff_vs_legacy_by_topk.png", fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-label-path", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    manifest = pd.read_csv(AUDIT_DIR / "final_evaluation_manifest.csv")
    if not bool(manifest["freeze_verified"].all()):
        raise RuntimeError("Freeze audit did not pass.")
    if set(manifest["reward_mode"]) != set(MODES) or sorted(manifest["seed"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Manifest does not cover all modes and seeds 0-4.")
    assert_checkpoint_hashes_current(manifest)
    assert_code_hashes_current()

    test_labels, invalid_count = read_labels(args.test_label_path)
    test_sha = sha256_file(args.test_label_path)
    first_read = datetime.now(timezone.utc).isoformat()
    net, features, genes = load_features()
    gene_set = set(genes)
    labels_in_network = [gene for gene in test_labels if gene in gene_set]
    labels_out_of_network = len(set(test_labels) - set(labels_in_network))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    access_lines = [
        "test_labels_read=true",
        f"test_first_read_timestamp={first_read}",
        f"test_label_file_path={args.test_label_path}",
        f"test_label_file_sha256={test_sha}",
        f"test_label_count_total={len(test_labels)}",
        f"test_label_count_in_network={len(labels_in_network)}",
        f"test_label_count_out_of_network={labels_out_of_network}",
        f"test_label_invalid_count={invalid_count}",
        "external_holdout_read=false",
        "historical_test_read=false",
        "training_performed=false",
        "optimizer_step_count=0",
        "replay_buffer_write_count=0",
        "priority_update_count=0",
        "training_episode_count=0",
    ]
    (TEST_DIR / "logs" / "test_data_access_manifest.txt").write_text("\n".join(access_lines) + "\n", encoding="utf-8")
    (REPRO_DIR / "test_data_access_manifest.txt").write_text("\n".join(access_lines) + "\n", encoding="utf-8")

    metric_rows = []
    integrity_rows = []
    determinism_rows = []
    for _, item in manifest.sort_values(["reward_mode", "seed"]).iterrows():
        mode = item["reward_mode"]
        seed = int(item["seed"])
        checkpoint = Path(item["checkpoint_path"])
        q1, q2 = infer_checkpoint(checkpoint, net, features, genes, device)
        ranking1 = sorted_ranking(q1, genes)
        ranking2 = sorted_ranking(q2, genes)
        same_order = [row["Gene"] for row in ranking1] == [row["Gene"] for row in ranking2]
        same_scores = bool(np.array_equal(q1, q2))
        determinism_rows.append({
            "reward_mode": mode,
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "gene_order_identical": same_order,
            "scores_identical": same_scores,
            "max_abs_score_difference": float(np.max(np.abs(q1 - q2))),
            "passed": same_order and same_scores,
        })
        ranking_path = TEST_DIR / "rankings" / f"test_ranking_{mode}_seed{seed}.csv"
        write_ranking(ranking_path, ranking1)
        ranks = list(range(1, len(ranking1) + 1))
        integrity = {
            "reward_mode": mode,
            "seed": seed,
            "ranking_path": str(ranking_path),
            "gene_count": len(ranking1),
            "unique_gene_count": len({row["Gene"] for row in ranking1}),
            "duplicate_gene_count": len(ranking1) - len({row["Gene"] for row in ranking1}),
            "missing_gene_count": 9039 - len(ranking1),
            "nan_count": int(np.isnan(q1).sum()),
            "inf_count": int(np.isinf(q1).sum()),
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_continuous": ranks == list(range(1, 9040)),
            "deterministic": same_order and same_scores,
            "passed": len(ranking1) == 9039 and len({row["Gene"] for row in ranking1}) == 9039 and np.isfinite(q1).all() and same_order and same_scores,
        }
        integrity_rows.append(integrity)
        metrics = metrics_for_rows(ranking1, labels_in_network, len(labels_in_network))
        metrics.update({
            "reward_mode": mode,
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "ranking_path": str(ranking_path),
            "test_labels_total": len(test_labels),
            "test_labels_in_network": len(labels_in_network),
            "test_labels_out_of_network": labels_out_of_network,
            "optimizer_step_count": 0,
            "replay_buffer_write_count": 0,
            "priority_update_count": 0,
            "training_episode_count": 0,
        })
        metric_rows.append(metrics)

    write_csv(TEST_DIR / "test_integrity_report.csv", integrity_rows)
    write_csv(TEST_DIR / "test_determinism_report.csv", determinism_rows)
    write_csv(TEST_DIR / "deterministic_checks" / "test_determinism_report.csv", determinism_rows)
    if not all(row["passed"] for row in integrity_rows):
        raise RuntimeError("Ranking integrity failed; stopping before conclusion generation.")

    write_csv(TEST_DIR / "test_metrics_per_seed.csv", metric_rows)
    write_csv(TEST_DIR / "per_seed_metrics" / "test_metrics_per_seed.csv", metric_rows)
    value_cols = [f"NDCG@{k}" for k in K_VALUES] + [f"HitCount@{k}" for k in K_VALUES] + [f"Precision@{k}" for k in K_VALUES] + [f"Recall@{k}" for k in K_VALUES] + ["MRR", "MeanRank", "MedianRank"]
    summary_rows = summarize(metric_rows, value_cols)
    write_csv(TEST_DIR / "test_reward_mode_summary.csv", summary_rows)
    write_csv(TEST_DIR / "summaries" / "test_reward_mode_summary.csv", summary_rows)

    legacy = {row["seed"]: row for row in metric_rows if row["reward_mode"] == "legacy"}
    pair_rows = []
    for mode in [m for m in MODES if m != "legacy"]:
        for metric in [f"NDCG@{k}" for k in K_VALUES] + [f"HitCount@{k}" for k in K_VALUES]:
            seed_items = []
            for row in [r for r in metric_rows if r["reward_mode"] == mode]:
                base = legacy[row["seed"]]
                diff = float(row[metric]) - float(base[metric])
                if abs(diff) <= TIE_TOL:
                    result = "tie"
                elif diff > 0:
                    result = "win"
                else:
                    result = "loss"
                seed_items.append((row, base, diff, result))
            diffs = [x[2] for x in seed_items]
            results = [x[3] for x in seed_items]
            for row, base, diff, result in seed_items:
                pair_rows.append({
                    "reward_mode": mode,
                    "metric": metric,
                    "seed": row["seed"],
                    "candidate_value": row[metric],
                    "legacy_value": base[metric],
                    "difference": diff,
                    "result": result,
                    "wins": results.count("win"),
                    "ties": results.count("tie"),
                    "losses": results.count("loss"),
                    "mean_difference": float(np.mean(diffs)),
                    "median_difference": float(np.median(diffs)),
                    "min_difference": float(np.min(diffs)),
                    "max_difference": float(np.max(diffs)),
                    "tie_tolerance": TIE_TOL,
                })
    write_csv(TEST_DIR / "test_paired_comparison_vs_legacy.csv", pair_rows)
    write_csv(TEST_DIR / "summaries" / "test_paired_comparison_vs_legacy.csv", pair_rows)

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
    write_csv(TEST_DIR / "test_topk_comparison.csv", topk_rows)
    write_csv(TEST_DIR / "summaries" / "test_topk_comparison.csv", topk_rows)

    consensus_metrics = []
    for mode in MODES:
        tables = []
        for seed in range(5):
            path = TEST_DIR / "rankings" / f"test_ranking_{mode}_seed{seed}.csv"
            tables.append(pd.read_csv(path)[["Gene", "Rank"]].rename(columns={"Rank": f"rank_seed{seed}"}))
        merged = tables[0]
        for table in tables[1:]:
            merged = merged.merge(table, on="Gene", how="inner")
        seed_cols = [col for col in merged.columns if col.startswith("rank_seed")]
        merged["median_rank"] = merged[seed_cols].median(axis=1)
        merged["mean_rank"] = merged[seed_cols].mean(axis=1)
        merged = merged.sort_values(["median_rank", "mean_rank", "Gene"]).reset_index(drop=True)
        merged.insert(0, "ConsensusRank", np.arange(1, len(merged) + 1))
        out_path = CONSENSUS_DIR / f"consensus_ranking_{mode}.csv"
        merged.to_csv(out_path, index=False)
        rows = [{"Gene": gene} for gene in merged["Gene"]]
        cm = metrics_for_rows(rows, labels_in_network, len(labels_in_network))
        cm.update({"reward_mode": mode, "consensus_ranking_path": str(out_path)})
        consensus_metrics.append(cm)
    write_csv(CONSENSUS_DIR / "test_consensus_metrics.csv", consensus_metrics)

    make_figures(metric_rows, topk_rows, pair_rows)

    low_summary = next(row for row in summary_rows if row["reward_mode"] == "multiomics_lowfreq")
    legacy_summary = next(row for row in summary_rows if row["reward_mode"] == "legacy")
    low_ndcg150_pairs = [row for row in pair_rows if row["reward_mode"] == "multiomics_lowfreq" and row["metric"] == "NDCG@150"]
    low_hit150_pairs = [row for row in pair_rows if row["reward_mode"] == "multiomics_lowfreq" and row["metric"] == "HitCount@150"]
    wins, ties, losses = int(low_ndcg150_pairs[0]["wins"]), int(low_ndcg150_pairs[0]["ties"]), int(low_ndcg150_pairs[0]["losses"])
    ndcg_top_better = sum(
        1 for k in [50, 100, 150]
        if next(row for row in topk_rows if row["reward_mode"] == "multiomics_lowfreq" and row["k"] == k)["mean_ndcg"]
        > next(row for row in topk_rows if row["reward_mode"] == "legacy" and row["k"] == k)["mean_ndcg"]
    )
    abs_diffs = sorted([abs(float(row["difference"])) for row in low_ndcg150_pairs], reverse=True)
    single_seed_driven = bool(abs_diffs and abs_diffs[0] > sum(abs_diffs[1:]))
    support_conditions = [
        low_summary["NDCG@150_mean"] > legacy_summary["NDCG@150_mean"],
        wins >= 3,
        low_summary["HitCount@150_mean"] >= legacy_summary["HitCount@150_mean"],
        ndcg_top_better >= 2,
        not single_seed_driven,
        all(row["passed"] for row in integrity_rows),
    ]
    if sum(support_conditions) >= 5:
        conclusion = "support"
    elif low_summary["NDCG@150_mean"] <= legacy_summary["NDCG@150_mean"] or losses > wins:
        conclusion = "not_support"
    else:
        conclusion = "partial_support"

    report_lines = [
        "# Test Report",
        "",
        "One-time Test evaluation used frozen Validation-best checkpoints only. No training, optimizer step, replay-buffer write, PER priority update, reward change, feature change, or checkpoint reselection occurred.",
        "",
        f"test_label_file_sha256={test_sha}",
        f"test_labels_total={len(test_labels)}",
        f"test_labels_in_network={len(labels_in_network)}",
        f"test_labels_out_of_network={labels_out_of_network}",
        "Recall and NDCG denominators use Test labels present in the 9039-gene HPRD network. Out-of-network labels are reported separately and are not treated as ordinary ranked misses.",
        f"bootstrap_repetitions={BOOTSTRAP_REPS}",
        f"bootstrap_seed={BOOTSTRAP_SEED}",
        "",
        "## Mean Test NDCG@150",
        *[f"- {row['reward_mode']}: {row['NDCG@150_mean']}" for row in summary_rows],
        "",
        "## multiomics_lowfreq vs legacy",
        f"wins/ties/losses={wins}/{ties}/{losses}",
        f"mean_ndcg_150_difference={low_ndcg150_pairs[0]['mean_difference']}",
        f"median_ndcg_150_difference={low_ndcg150_pairs[0]['median_difference']}",
        f"mean_hit_150_difference={low_hit150_pairs[0]['mean_difference']}",
        f"single_seed_driven={single_seed_driven}",
        f"test_conclusion={conclusion}",
        "",
        "This is a descriptive n=5 seed evaluation. No statistical-significance claim is made.",
    ]
    (TEST_DIR / "summaries" / "test_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "test_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    claims = [
        "# Test Claims Boundary",
        "",
        "Allowed: report frozen Test metrics for all five seeds and compare fixed reward modes.",
        "Not allowed: claim statistical significance from five seeds alone, new cancer-gene discovery, clinical validity, complete mutation independence, or direct reward of unknown low-frequency candidates.",
        "multiomics_no_mutation removes direct mutation reward only; Mutation remains in hybrid6_raw input.",
        "multiomics_lowfreq gives bounded rarity bonus only to Train drivers during training, not to unknown Test genes.",
    ]
    (REPORT_DIR / "test_claims_boundary.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    repro = [
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_genrisk",
        "cd /mnt/e/Projects/RL-GenRisk-main",
        f"python src/evaluate_frozen_test.py --test-label-path {args.test_label_path} --device {args.device}",
    ]
    (REPRO_DIR / "test_reproducibility_commands.txt").write_text("\n".join(repro) + "\n", encoding="utf-8")
    run_manifest = [{"path": str(path), "size_bytes": path.stat().st_size} for path in sorted(EVAL_ROOT.rglob("*")) if path.is_file()]
    write_csv(REPRO_DIR / "test_run_manifest.csv", run_manifest)
    write_csv(REPORT_DIR / "final_run_manifest.csv", run_manifest)


if __name__ == "__main__":
    main()
