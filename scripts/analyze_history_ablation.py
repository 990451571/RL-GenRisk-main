#!/usr/bin/env python3
"""Summarize the validation-only DDQN history-ablation experiment.

The three inputs must be greedy-rollout evaluations of identical scalar DDQN
grids.  They differ only in the history context supplied to the Q network.
No Test label is accepted or read.
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


RUN_RE = re.compile(
    r".*_(?P<mode>full|no_history|shuffled_history)_r(?P<rec>\d{3})_d(?P<disc>\d{3})_seed(?P<seed>\d+)$"
)
MODES = ("full", "no_history", "shuffled_history")
PREFERENCES = ("r1.00_d0.00", "r0.80_d0.20", "r0.50_d0.50", "r0.20_d0.80", "r0.00_d1.00")
SEEDS = (42, 45, 46)
METRICS = (
    "NDCG@150",
    "Recall@150",
    "DiscoveryPrecision@150",
    "DiscoveryFoldEnrichment@150",
    "CumulativeRecoveryRawReward@150",
    "CumulativeDiscoveryRawReward@150",
    "FinalPatientCoverage@150",
)
STATEFUL_LIFT_METRICS = (
    "CumulativeRecoveryRawReward@150",
    "CumulativeDiscoveryRawReward@150",
    "FinalPatientCoverage@150",
)


def parse_run(name: str):
    match = RUN_RE.match(str(name))
    if not match:
        raise ValueError(f"Cannot parse history-ablation run name: {name}")
    return (
        match.group("mode"),
        int(match.group("seed")),
        f"r{int(match.group('rec')) / 100:.2f}_d{int(match.group('disc')) / 100:.2f}",
    )


def numeric(value):
    return pd.to_numeric(value, errors="coerce")


def read_eval(eval_dir: str, expected_mode: str):
    root = Path(eval_dir)
    frame = pd.read_csv(root / "summary_metrics.csv")
    primary = frame[frame["Method"].eq("rollout_primary_available")].copy()
    rows = []
    for _, row in primary.iterrows():
        mode, seed, preference = parse_run(row["Run"])
        if mode != expected_mode:
            raise ValueError(f"{row['Run']} has mode={mode}, expected {expected_mode}.")
        source = row["CheckpointSource"]
        onepass = frame[(frame["Run"].eq(row["Run"])) & (frame["Method"].eq(f"{source}_Q_onepass"))]
        if len(onepass) != 1:
            raise ValueError(f"Missing matching one-pass metrics for {row['Run']} ({source}).")
        onepass = onepass.iloc[0]
        item = {
            "HistoryMode": mode,
            "Seed": seed,
            "Preference": preference,
            "Run": row["Run"],
            "CheckpointSource": source,
            "RankingPath": str(root / "rankings" / f"{row['Run']}_{source}_Q_rollout.csv"),
        }
        for metric in METRICS:
            item[metric] = numeric(row.get(metric, np.nan))
        for metric in STATEFUL_LIFT_METRICS:
            item[f"RolloutMinusOnepass_{metric}"] = numeric(row.get(metric, np.nan)) - numeric(onepass.get(metric, np.nan))
        rows.append(item)
    expected = {(seed, preference) for seed in SEEDS for preference in PREFERENCES}
    actual = {(row["Seed"], row["Preference"]) for row in rows}
    if actual != expected:
        raise ValueError(f"{expected_mode} does not contain exactly the required 3 seed × 5 preference grid.")
    return rows


def top150(path: str):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["Gene"] for row in csv.DictReader(handle) if int(row["Rank"]) <= 150}


def stability(rows):
    output = []
    frame = pd.DataFrame(rows)
    for (mode, preference), group in frame.groupby(["HistoryMode", "Preference"]):
        sets = [(int(row.Seed), top150(row.RankingPath)) for _, row in group.iterrows()]
        values = [len(a & b) / len(a | b) for (_, a), (_, b) in combinations(sets, 2)]
        output.append({
            "HistoryMode": mode,
            "Preference": preference,
            "SeedCount": len(sets),
            "PairCount": len(values),
            "Top150JaccardMean": float(np.mean(values)) if values else np.nan,
            "Top150JaccardSD": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        })
    return output


def aggregate(rows):
    frame = pd.DataFrame(rows)
    output = []
    for (mode, preference), group in frame.groupby(["HistoryMode", "Preference"]):
        item = {"HistoryMode": mode, "Preference": preference, "SeedCount": int(group["Seed"].nunique())}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        output.append(item)
    return output


def paired_differences(rows, ablated_mode):
    frame = pd.DataFrame(rows)
    full = frame[frame.HistoryMode.eq("full")].set_index(["Seed", "Preference"])
    ablated = frame[frame.HistoryMode.eq(ablated_mode)].set_index(["Seed", "Preference"])
    out = []
    for key in sorted(full.index):
        row = {"Seed": key[0], "Preference": key[1]}
        for metric in METRICS:
            row[f"full_minus_{ablated_mode}_{metric}"] = float(full.loc[key, metric] - ablated.loc[key, metric])
        for metric in STATEFUL_LIFT_METRICS:
            lift = f"RolloutMinusOnepass_{metric}"
            row[f"full_minus_{ablated_mode}_{lift}"] = float(full.loc[key, lift] - ablated.loc[key, lift])
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-eval", required=True)
    parser.add_argument("--no-history-eval", required=True)
    parser.add_argument("--shuffled-history-eval", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = (
        read_eval(args.full_eval, "full")
        + read_eval(args.no_history_eval, "no_history")
        + read_eval(args.shuffled_history_eval, "shuffled_history")
    )
    per_seed = pd.DataFrame(rows)
    per_seed["test_labels_read"] = False
    per_seed.to_csv(output / "history_ablation_per_seed.csv", index=False)
    pd.DataFrame(aggregate(rows)).to_csv(output / "history_ablation_summary.csv", index=False)
    pd.DataFrame(stability(rows)).to_csv(output / "top150_stability.csv", index=False)
    for mode in ("no_history", "shuffled_history"):
        pd.DataFrame(paired_differences(rows, mode)).to_csv(
            output / f"full_minus_{mode}_paired.csv", index=False
        )
    (output / "analysis_metadata.json").write_text(json.dumps({
        "comparison": "scalar DDQN with identical environment/reward/action legality/PER; only Q-visible history context differs",
        "full": "true selected-gene mask",
        "no_history": "all-available mask supplied to Q at every step",
        "shuffled_history": "same selected count but selected-gene identities deterministically permuted per seed/episode/step",
        "test_labels_read": False,
        "caveat": "n=3 supports descriptive mean±SD and consistency checks, not significance claims.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote DDQN history-ablation comparison to {output}")


if __name__ == "__main__":
    main()
