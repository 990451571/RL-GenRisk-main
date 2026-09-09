#!/usr/bin/env python3
"""Validation-only comparison for shared preference-conditioned bandit.

Primary comparison uses final-checkpoint greedy rollouts for the shared model
and ``last_Q_rollout`` for scalar DDQN/contextual-bandit, avoiding asymmetric
Validation checkpoint selection.  Static and supervised final rankers are
included as non-RL references.  Test labels are never read.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r".*_r(?P<rec>\d{3})_d(?P<disc>\d{3})_seed(?P<seed>\d+)$")
PREFERENCES = ("r1.00_d0.00", "r0.80_d0.20", "r0.50_d0.50", "r0.20_d0.80", "r0.00_d1.00")
METRICS = ("NDCG@150", "Recall@150", "DiscoveryPrecision@150", "DiscoveryFoldEnrichment@150")
SCHEDULE_POLICY = "balanced_latin_blocks_v1"


def parse_scalar(name):
    match = RUN_RE.match(str(name))
    if not match:
        raise ValueError(f"Cannot parse scalar run name: {name}")
    return int(match.group("seed")), f"r{int(match.group('rec')) / 100:.2f}_d{int(match.group('disc')) / 100:.2f}"


def number(value):
    return pd.to_numeric(value, errors="coerce")


def expand(paths):
    out = []
    for item in paths:
        out.extend(glob.glob(item) if "*" in item else [item])
    return [Path(path) for path in sorted(set(out))]


def read_shared(run_paths, expected_seeds):
    rows = []
    for run in expand(run_paths):
        cfg = json.loads((run / "preference_bandit_config.json").read_text(encoding="utf-8"))
        if cfg.get("learning_mode") != "contextual_bandit" or cfg.get("td_target") != "immediate_reward_only":
            raise ValueError(f"{run} is not a preference-conditioned contextual bandit.")
        if cfg.get("evaluation_policy") != "final_checkpoint_greedy_rollout":
            raise ValueError(f"{run} does not use the frozen final-checkpoint policy.")
        if cfg.get("schedule_policy") != SCHEDULE_POLICY:
            raise ValueError(
                f"{run} uses an unbalanced or unknown preference schedule; "
                "it cannot be used for the shared-bandit primary comparison."
            )
        expected_phase_counts = [2] * len(PREFERENCES)
        if any(values != expected_phase_counts for values in cfg.get("trained_preference_phase_counts", {}).values()):
            raise ValueError(f"{run} does not have phase-balanced preference exposure.")
        frame = pd.read_csv(run / "preference_bandit_final_metrics.csv")
        for _, row in frame.iterrows():
            item = {"Method": "shared_preference_bandit", "Seed": int(row["seed"]), "Preference": row["preference"],
                    "Run": str(run), "Checkpoint": "final", "RankingPath": row["ranking_path"]}
            for metric in METRICS:
                item[metric] = number(row[metric])
            rows.append(item)
    required = {(seed, preference) for seed in expected_seeds for preference in PREFERENCES}
    if {(row["Seed"], row["Preference"]) for row in rows} != required:
        raise ValueError(
            "Shared bandit runs do not match the requested seed × 5 preference grid."
        )
    return rows


def read_scalar(eval_dir, method, source="last_Q_rollout"):
    root = Path(eval_dir)
    frame = pd.read_csv(root / "summary_metrics.csv")
    frame = frame[frame.Method.eq(source)]
    rows = []
    for _, row in frame.iterrows():
        seed, preference = parse_scalar(row["Run"])
        checkpoint = str(row["CheckpointSource"])
        item = {"Method": method, "Seed": seed, "Preference": preference, "Run": row["Run"],
                "Checkpoint": checkpoint, "RankingPath": str(root / "rankings" / f"{row['Run']}_{checkpoint}_Q_rollout.csv")}
        for metric in METRICS:
            item[metric] = number(row[metric])
        rows.append(item)
    return rows


def read_static(eval_dir):
    root = Path(eval_dir)
    frame = pd.read_csv(root / "summary_metrics.csv")
    frame = frame[frame.Method.eq("static_mutation")]
    rows = []
    for _, row in frame.iterrows():
        seed, _ = parse_scalar(row["Run"])
        if any(item["Seed"] == seed for item in rows):
            continue
        item = {"Method": "static_mutation", "Seed": seed, "Preference": "static", "Run": row["Run"],
                "Checkpoint": "static", "RankingPath": str(root / "rankings" / f"{row['Run']}_static_mutation.csv")}
        for metric in METRICS:
            item[metric] = number(row[metric])
        rows.append(item)
    return rows


def read_supervised(directories):
    rows = []
    for directory in map(Path, directories):
        match = re.search(r"seed(?P<seed>\d+)$", directory.name)
        if not match:
            raise ValueError(f"Supervised directory must end in seedNN: {directory}")
        seed = int(match.group("seed"))
        frame = pd.read_csv(directory / "supervised_summary.csv")
        for original, method, filename in (
            ("supervised_GCN_final", "supervised_GCN", "supervised_GCN_final.csv"),
            ("supervised_MLP_no_graph_final", "supervised_MLP", "supervised_MLP_final.csv"),
        ):
            item = frame[frame.Method.eq(original)]
            if len(item) != 1:
                raise ValueError(f"Missing {original} in {directory}")
            row = item.iloc[0]
            result = {"Method": method, "Seed": seed, "Preference": "static", "Run": original,
                      "Checkpoint": "final", "RankingPath": str(directory / "rankings" / filename)}
            for metric in METRICS:
                result[metric] = number(row.get(metric, np.nan))
            rows.append(result)
    return rows


def top150(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["Gene"] for row in csv.DictReader(handle) if int(row["Rank"]) <= 150}


def aggregate(rows):
    output = []
    frame = pd.DataFrame(rows)
    for (method, preference), group in frame.groupby(["Method", "Preference"]):
        item = {"Method": method, "Preference": preference, "SeedCount": int(group.Seed.nunique())}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        output.append(item)
    return output


def stability(rows):
    output = []
    frame = pd.DataFrame(rows)
    for (method, preference), group in frame.groupby(["Method", "Preference"]):
        gene_sets = [(int(row.Seed), top150(row.RankingPath)) for _, row in group.iterrows()]
        values = [len(left & right) / len(left | right) for (_, left), (_, right) in combinations(gene_sets, 2)]
        output.append({"Method": method, "Preference": preference, "SeedCount": len(gene_sets), "PairCount": len(values),
                       "Top150JaccardMean": float(np.mean(values)) if values else np.nan,
                       "Top150JaccardSD": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan})
    return output


def paired(shared_rows, baseline_rows, label):
    shared = pd.DataFrame(shared_rows).set_index(["Seed", "Preference"])
    base = pd.DataFrame(baseline_rows).set_index(["Seed", "Preference"])
    output = []
    for key in sorted(shared.index):
        row = {"Seed": key[0], "Preference": key[1]}
        for metric in METRICS:
            row[f"shared_minus_{label}_{metric}"] = float(shared.loc[key, metric] - base.loc[key, metric])
        output.append(row)
    return output


def trend(shared_rows):
    frame = pd.DataFrame(shared_rows).copy()
    frame["w_discovery"] = frame.Preference.str.extract(r"_d(\d+\.\d+)$").astype(float)
    output = []
    for seed, group in frame.groupby("Seed"):
        group = group.sort_values("w_discovery")
        output.append({
            "Seed": int(seed),
            "RecoveryNDCG_vs_w_discovery_spearman": float(group["w_discovery"].corr(group["NDCG@150"], method="spearman")),
            "RecoveryRecall_vs_w_discovery_spearman": float(group["w_discovery"].corr(group["Recall@150"], method="spearman")),
            "DiscoveryPrecision_vs_w_discovery_spearman": float(group["w_discovery"].corr(group["DiscoveryPrecision@150"], method="spearman")),
            "DiscoveryFold_vs_w_discovery_spearman": float(group["w_discovery"].corr(group["DiscoveryFoldEnrichment@150"], method="spearman")),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-runs", nargs="+", required=True)
    parser.add_argument("--expected-seeds", default="42,45,46",
                        help="Comma-separated seeds expected in --shared-runs.")
    parser.add_argument("--ddqn-eval")
    parser.add_argument("--scalar-bandit-eval")
    parser.add_argument("--supervised-dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected_seeds = tuple(int(token) for token in args.expected_seeds.split(",") if token.strip())
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("--expected-seeds must contain one or more distinct integer seeds.")
    baseline_args = (args.ddqn_eval, args.scalar_bandit_eval, args.supervised_dirs)
    if any(value is not None for value in baseline_args) and not all(value is not None for value in baseline_args):
        raise ValueError("Provide all baseline arguments together, or omit all for shared-only reproducibility analysis.")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    shared = read_shared(args.shared_runs, expected_seeds)
    rows = list(shared)
    if args.ddqn_eval is not None:
        # Existing scalar baselines are intentionally limited to the original
        # three matched seeds.  A five-seed shared-only extension assesses
        # reproducibility, not a newly unmatched cross-method comparison.
        baseline_seeds = (42, 45, 46)
        ddqn = read_scalar(args.ddqn_eval, "scalar_ddqn")
        scalar_bandit = read_scalar(args.scalar_bandit_eval, "scalar_contextual_bandit")
        required = {(seed, preference) for seed in baseline_seeds for preference in PREFERENCES}
        for name, baseline_rows in (("DDQN", ddqn), ("scalar contextual bandit", scalar_bandit)):
            if {(row["Seed"], row["Preference"]) for row in baseline_rows} != required:
                raise ValueError(f"{name} is not exactly the required 3 seed × 5 preference grid.")
        if tuple(expected_seeds) != baseline_seeds:
            raise ValueError("Baseline comparison requires exactly expected seeds 42,45,46.")
        rows += ddqn + scalar_bandit + read_static(args.ddqn_eval) + read_supervised(args.supervised_dirs)
    frame = pd.DataFrame(rows); frame["test_labels_read"] = False
    frame.to_csv(output / "preference_bandit_per_seed.csv", index=False)
    pd.DataFrame(aggregate(rows)).to_csv(output / "preference_bandit_summary.csv", index=False)
    pd.DataFrame(stability(rows)).to_csv(output / "top150_stability.csv", index=False)
    if args.ddqn_eval is not None:
        pd.DataFrame(paired(shared, ddqn, "ddqn")).to_csv(output / "shared_minus_ddqn_paired.csv", index=False)
        pd.DataFrame(paired(shared, scalar_bandit, "scalar_bandit")).to_csv(output / "shared_minus_scalar_bandit_paired.csv", index=False)
    pd.DataFrame(trend(shared)).to_csv(output / "shared_bandit_trend_by_seed.csv", index=False)
    (output / "analysis_metadata.json").write_text(json.dumps({
        "primary_policy": "shared bandit final checkpoint rollout; scalar RL final (last_Q_rollout) checkpoint rollout",
        "comparison": (
            "same frozen data/reward/features and seeds; shared model has 50 total episodes = 10 per preference"
            if args.ddqn_eval is not None else
            "shared-bandit reproducibility extension only; no unmatched baseline comparison is made"
        ),
        "expected_seeds": list(expected_seeds),
        "baseline_comparison_included": args.ddqn_eval is not None,
        "shared_schedule_policy": SCHEDULE_POLICY,
        "grn_feature_active": False,
        "test_labels_read": False,
        "caveat": (
            f"n={len(expected_seeds)} supports descriptive mean±SD and seed-wise direction checks; "
            "formal significance claims require a pre-specified inferential design."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote preference-conditioned bandit comparison to {output}")


if __name__ == "__main__":
    main()
