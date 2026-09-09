#!/usr/bin/env python3
"""Validation-only comparison: scalar DDQN, contextual bandit and rankers.

No Test label is accepted or read.  DDQN and bandit are paired by the same
seed and the same frozen scalar Recovery/Discovery reward weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r".*_r(?P<rec>\d{3})_d(?P<disc>\d{3})_seed(?P<seed>\d+)$")
METRICS = [
    "NDCG@150", "Recall@150", "DiscoveryPrecision@150", "DiscoveryFoldEnrichment@150",
    "CumulativeRecoveryRawReward@150", "CumulativeDiscoveryRawReward@150",
    "FinalPatientCoverage@150",
]


def parse_run(name):
    match = RUN_RE.match(str(name))
    if not match:
        raise ValueError(f"Cannot parse frozen scalarization and seed from run name: {name}")
    return int(match.group("seed")), f"r{int(match.group('rec')) / 100:.2f}_d{int(match.group('disc')) / 100:.2f}"


def q(value):
    return pd.to_numeric(value, errors="coerce")


def read_rl(eval_dir, method):
    root = Path(eval_dir)
    frame = pd.read_csv(root / "summary_metrics.csv")
    primary = frame[frame["Method"] == "rollout_primary_available"].copy()
    rows = []
    for _, row in primary.iterrows():
        seed, preference = parse_run(row["Run"])
        item = {"Method": method, "Seed": seed, "Preference": preference, "Run": row["Run"],
                "CheckpointSource": row["CheckpointSource"], "RankingPath": str(
                    root / "rankings" / f"{row['Run']}_{row['CheckpointSource']}_Q_rollout.csv"
                )}
        for metric in METRICS:
            item[metric] = q(row.get(metric, np.nan))
        # The same selected checkpoint also has a one-pass ranking.  Replaying
        # each ordering through the unchanged reward environment permits a
        # direct, validation-label-free check of whether sequential rollout
        # adds history-dependent reward or patient coverage.
        onepass_method = f"{row['CheckpointSource']}_Q_onepass"
        onepass = frame[(frame["Run"] == row["Run"]) & (frame["Method"] == onepass_method)]
        if len(onepass) != 1:
            raise ValueError(f"Missing {onepass_method} metrics for {row['Run']}")
        onepass_row = onepass.iloc[0]
        for metric in (
            "CumulativeRecoveryRawReward@150",
            "CumulativeDiscoveryRawReward@150",
            "FinalPatientCoverage@150",
        ):
            item[f"RolloutMinusOnepass_{metric}"] = (
                q(row.get(metric, np.nan)) - q(onepass_row.get(metric, np.nan))
            )
        rows.append(item)
    return rows


def read_static(eval_dir):
    root = Path(eval_dir)
    frame = pd.read_csv(root / "summary_metrics.csv")
    frame = frame[frame["Method"] == "static_mutation"].copy()
    rows = []
    # static mutation is weight-independent; retain exactly one copy per seed.
    for _, row in frame.iterrows():
        seed, _ = parse_run(row["Run"])
        if any(item["Seed"] == seed for item in rows):
            continue
        item = {"Method": "static_mutation", "Seed": seed, "Preference": "static", "Run": row["Run"],
                "CheckpointSource": "static", "RankingPath": str(root / "rankings" / f"{row['Run']}_static_mutation.csv")}
        for metric in METRICS:
            item[metric] = q(row.get(metric, np.nan))
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
        for source, method, ranking in (
            ("supervised_GCN_final", "supervised_GCN", "supervised_GCN_final.csv"),
            ("supervised_MLP_no_graph_final", "supervised_MLP", "supervised_MLP_final.csv"),
        ):
            sub = frame[frame["Method"] == source]
            if len(sub) != 1:
                raise ValueError(f"Missing {source} in {directory / 'supervised_summary.csv'}")
            row = sub.iloc[0]
            item = {"Method": method, "Seed": seed, "Preference": "static", "Run": source,
                    "CheckpointSource": "fixed_epoch_200", "RankingPath": str(directory / "rankings" / ranking)}
            for metric in METRICS[:4]:
                item[metric] = q(row.get(metric, np.nan))
            for metric in METRICS[4:]:
                item[metric] = np.nan
            rows.append(item)
    return rows


def top150(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["Gene"] for row in csv.DictReader(handle) if int(row["Rank"]) <= 150}


def stability(rows):
    output = []
    for (method, preference), group in pd.DataFrame(rows).groupby(["Method", "Preference"], dropna=False):
        members = [(int(row.Seed), top150(row.RankingPath)) for _, row in group.iterrows()]
        values = [len(a & b) / len(a | b) for (_, a), (_, b) in combinations(members, 2)]
        output.append({"Method": method, "Preference": preference, "SeedCount": len(members),
                       "PairCount": len(values), "Top150JaccardMean": float(np.mean(values)) if values else np.nan,
                       "Top150JaccardSD": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan})
    return output


def aggregate(frame):
    rows = []
    for (method, preference), group in frame.groupby(["Method", "Preference"], dropna=False):
        item = {"Method": method, "Preference": preference, "SeedCount": int(group["Seed"].nunique())}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddqn-eval", required=True)
    parser.add_argument("--bandit-eval", required=True)
    parser.add_argument("--supervised-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ddqn = read_rl(args.ddqn_eval, "ddqn")
    bandit = read_rl(args.bandit_eval, "contextual_bandit")
    static = read_static(args.bandit_eval)
    supervised = read_supervised(args.supervised_dirs)
    required = {(seed, pref) for seed in (42, 45, 46) for pref in (
        "r1.00_d0.00", "r0.80_d0.20", "r0.50_d0.50", "r0.20_d0.80", "r0.00_d1.00"
    )}
    if {(row["Seed"], row["Preference"]) for row in ddqn} != required:
        raise ValueError("DDQN comparison set is not the required 3 seed × 5 scalarization grid.")
    if {(row["Seed"], row["Preference"]) for row in bandit} != required:
        raise ValueError("Bandit comparison set is not the required 3 seed × 5 scalarization grid.")

    rows = ddqn + bandit + static + supervised
    frame = pd.DataFrame(rows)
    frame["test_labels_read"] = False
    frame.to_csv(output / "rl_necessity_per_seed.csv", index=False)
    summary = pd.DataFrame(aggregate(frame))
    summary.to_csv(output / "rl_necessity_summary.csv", index=False)
    pd.DataFrame(stability(rows)).to_csv(output / "top150_stability.csv", index=False)

    ddf = pd.DataFrame(ddqn).set_index(["Seed", "Preference"])
    bdf = pd.DataFrame(bandit).set_index(["Seed", "Preference"])
    paired = []
    for key in sorted(required):
        row = {"Seed": key[0], "Preference": key[1]}
        for metric in METRICS:
            row[f"ddqn_minus_bandit_{metric}"] = float(ddf.loc[key, metric] - bdf.loc[key, metric])
        paired.append(row)
    pd.DataFrame(paired).to_csv(output / "ddqn_minus_bandit_paired.csv", index=False)
    lift_metrics = [
        "RolloutMinusOnepass_CumulativeRecoveryRawReward@150",
        "RolloutMinusOnepass_CumulativeDiscoveryRawReward@150",
        "RolloutMinusOnepass_FinalPatientCoverage@150",
    ]
    lift_rows = []
    for method_rows, method in ((ddqn, "ddqn"), (bandit, "contextual_bandit")):
        for row in method_rows:
            lift_rows.append({
                "Method": method, "Seed": row["Seed"], "Preference": row["Preference"],
                **{metric: row[metric] for metric in lift_metrics},
            })
    lift_frame = pd.DataFrame(lift_rows)
    lift_frame.to_csv(output / "stateful_rollout_lift_per_seed.csv", index=False)
    lift_summary = lift_frame.groupby(["Method", "Preference"], as_index=False)[lift_metrics].agg(["mean", "std"])
    lift_summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in lift_summary.columns
    ]
    lift_summary.to_csv(output / "stateful_rollout_lift_summary.csv", index=False)
    (output / "analysis_metadata.json").write_text(json.dumps({
        "comparison": "same seed and frozen scalar rd_scan reward; bandit removes only TD bootstrap",
        "test_labels_read": False,
        "stateful_check": "greedy rollout minus one-pass replay on raw reward and patient coverage",
        "caveat": "n=3 supports descriptive mean±SD only, not significance claims.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RL-necessity comparison to {output}")


if __name__ == "__main__":
    main()
