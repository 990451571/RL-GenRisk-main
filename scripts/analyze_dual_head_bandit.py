#!/usr/bin/env python3
"""Compare dual-head immediate bandit with frozen shared/scalar controls.

All model evaluation uses Validation labels only.  Scalar-bandit paired
comparisons are restricted to seeds with an existing matched run.
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


PREFERENCES = (
    "r1.00_d0.00", "r0.90_d0.10", "r0.80_d0.20",
    "r0.65_d0.35", "r0.50_d0.50", "r0.35_d0.65",
    "r0.20_d0.80", "r0.10_d0.90", "r0.00_d1.00",
)
SEEN = {"r1.00_d0.00", "r0.80_d0.20", "r0.50_d0.50", "r0.20_d0.80", "r0.00_d1.00"}
METRICS = (
    "NDCG@150", "Recall@150",
    "DiscoveryPrecision@150", "DiscoveryFoldEnrichment@150",
)
RUN_RE = re.compile(r".*_r(?P<rec>\d{3})_d(?P<disc>\d{3})_seed(?P<seed>\d+)$")


def top150(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["Gene"] for row in csv.DictReader(handle) if int(row["Rank"]) <= 150}


def jaccard(left, right):
    return len(left & right) / len(left | right)


def aggregate(frame, method):
    rows = []
    for (scope, preference, w_disc), group in frame.groupby(
        ["Scope", "Preference", "w_discovery"], sort=False
    ):
        row = {
            "Method": method,
            "Scope": scope,
            "Preference": preference,
            "w_discovery": float(w_disc),
            "SeedCount": int(group.Seed.nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return rows


def interpolation_diagnostics(frame, sets, method):
    adjacent, bracket, trend = [], [], []
    for seed, group in frame.groupby("Seed"):
        ordered = group.sort_values("w_discovery")
        records = list(ordered.to_dict("records"))
        for left, right in zip(records, records[1:]):
            overlap = jaccard(sets[(seed, left["Preference"])], sets[(seed, right["Preference"])])
            adjacent.append({
                "Method": method, "Seed": int(seed),
                "LeftPreference": left["Preference"], "RightPreference": right["Preference"],
                "Top150Jaccard": overlap,
                "Top150SharedGeneCount": int(round(300 * overlap / (1 + overlap))),
                "test_labels_read": False,
            })
        seen = [row for row in records if row["Scope"] == "seen"]
        for row in (item for item in records if item["Scope"] == "unseen"):
            lower = max((item for item in seen if item["w_discovery"] < row["w_discovery"]), key=lambda x: x["w_discovery"])
            upper = min((item for item in seen if item["w_discovery"] > row["w_discovery"]), key=lambda x: x["w_discovery"])
            item = {
                "Method": method, "Seed": int(seed), "UnseenPreference": row["Preference"],
                "LowerSeenPreference": lower["Preference"], "UpperSeenPreference": upper["Preference"],
                "JaccardToLower": jaccard(sets[(seed, row["Preference"])], sets[(seed, lower["Preference"])]),
                "JaccardToUpper": jaccard(sets[(seed, row["Preference"])], sets[(seed, upper["Preference"])]),
                "test_labels_read": False,
            }
            for metric in METRICS:
                low, high = sorted((lower[metric], upper[metric]))
                item[f"{metric}_within_seen_bracket"] = bool(low - 1e-12 <= row[metric] <= high + 1e-12)
            bracket.append(item)
        trend.append({
            "Method": method, "Seed": int(seed),
            "UniqueTop150Policies": len({frozenset(sets[(seed, row["Preference"])]) for row in records}),
            "RecoveryNDCGSpearman": float(ordered.w_discovery.corr(ordered["NDCG@150"], method="spearman")),
            "RecoveryRecallSpearman": float(ordered.w_discovery.corr(ordered["Recall@150"], method="spearman")),
            "DiscoveryPrecisionSpearman": float(ordered.w_discovery.corr(ordered["DiscoveryPrecision@150"], method="spearman")),
            "DiscoveryFoldSpearman": float(ordered.w_discovery.corr(ordered["DiscoveryFoldEnrichment@150"], method="spearman")),
            "test_labels_read": False,
        })
    return adjacent, bracket, trend


def paired(left, right, left_name, right_name):
    keys = ["Seed", "Preference"]
    merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        item = {"Seed": int(row.Seed), "Preference": row.Preference}
        for metric in METRICS:
            item[f"{left_name}_minus_{right_name}_{metric}"] = float(row[f"{metric}_left"] - row[f"{metric}_right"])
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-runs", nargs="+", required=True)
    parser.add_argument("--single-shared-interpolation", required=True)
    parser.add_argument("--scalar-bandit-eval", required=True)
    parser.add_argument("--expected-seeds", default="42,45,48")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = tuple(int(token) for token in args.expected_seeds.split(",") if token.strip())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    sets = {}
    for run in map(Path, args.dual_runs):
        config = json.loads((run / "dual_head_bandit_config.json").read_text(encoding="utf-8"))
        required = {
            "model": "dual_head_preference_contextual_bandit",
            "objective_dim": 2,
            "q_preference_dim": 0,
            "learning_mode": "contextual_bandit",
            "bootstrap": False,
            "action_scalarization": "raw_linear",
            "schedule_policy": "balanced_latin_blocks_v1",
            "test_labels_read": False,
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError(f"Dual-head run contract failed: {run}")
        frame = pd.read_csv(run / "dual_head_bandit_final_metrics.csv")
        if len(frame) != len(PREFERENCES) or set(frame.preference) != set(PREFERENCES):
            raise ValueError(f"Missing seen/unseen preference rows: {run}")
        for _, row in frame.iterrows():
            item = row.to_dict()
            item.update({"Seed": int(row.seed), "Scope": row.scope, "Preference": row.preference})
            rows.append(item)
            sets[(int(row.seed), row.preference)] = top150(row.ranking_path)
    dual = pd.DataFrame(rows)
    if tuple(sorted(dual.Seed.unique())) != tuple(sorted(expected)):
        raise ValueError("Dual-head seeds do not match --expected-seeds.")
    dual.to_csv(output / "dual_head_per_seed.csv", index=False)
    pd.DataFrame(aggregate(dual, "dual_head_bandit")).sort_values("w_discovery").to_csv(
        output / "dual_head_summary.csv", index=False
    )

    adjacent, bracket, trend = interpolation_diagnostics(dual, sets, "dual_head_bandit")
    pd.DataFrame(adjacent).to_csv(output / "dual_head_adjacent_jaccard.csv", index=False)
    pd.DataFrame(bracket).to_csv(output / "dual_head_unseen_brackets.csv", index=False)
    pd.DataFrame(trend).to_csv(output / "dual_head_trend_by_seed.csv", index=False)

    stability_rows = []
    for preference, group in dual.groupby("Preference"):
        seed_sets = [sets[(int(seed), preference)] for seed in group.Seed]
        values = [jaccard(left, right) for left, right in combinations(seed_sets, 2)]
        stability_rows.append({
            "Preference": preference,
            "Scope": group.Scope.iloc[0],
            "w_discovery": float(group.w_discovery.iloc[0]),
            "Top150JaccardMean": float(np.mean(values)),
            "Top150JaccardSD": float(np.std(values, ddof=1)),
        })
    pd.DataFrame(stability_rows).sort_values("w_discovery").to_csv(
        output / "dual_head_across_seed_stability.csv", index=False
    )

    single_root = Path(args.single_shared_interpolation)
    single = pd.read_csv(single_root / "interpolation_metrics_per_seed.csv")
    single = single[single.Seed.isin(expected)].copy()
    pd.DataFrame(paired(dual, single, "dual", "single_shared")).to_csv(
        output / "dual_minus_single_shared_paired.csv", index=False
    )

    scalar_raw = pd.read_csv(Path(args.scalar_bandit_eval) / "summary_metrics.csv")
    scalar_raw = scalar_raw[scalar_raw.Method.eq("last_Q_rollout")]
    scalar_rows = []
    scalar_seeds = sorted(set(expected) & {42, 45, 46})
    for _, row in scalar_raw.iterrows():
        match = RUN_RE.match(str(row.Run))
        if not match:
            continue
        seed = int(match.group("seed"))
        if seed not in scalar_seeds:
            continue
        preference = f"r{int(match.group('rec')) / 100:.2f}_d{int(match.group('disc')) / 100:.2f}"
        item = {"Seed": seed, "Preference": preference}
        item.update({metric: float(row[metric]) for metric in METRICS})
        scalar_rows.append(item)
    scalar = pd.DataFrame(scalar_rows)
    dual_seen_scalar_seeds = dual[dual.Seed.isin(scalar_seeds) & dual.Preference.isin(SEEN)]
    pd.DataFrame(paired(dual_seen_scalar_seeds, scalar, "dual", "scalar_bandit")).to_csv(
        output / "dual_minus_scalar_bandit_paired.csv", index=False
    )

    single_adj = pd.read_csv(single_root / "interpolation_adjacent_jaccard.csv")
    single_adj = single_adj[single_adj.Seed.isin(expected)]
    single_bracket = pd.read_csv(single_root / "interpolation_unseen_brackets.csv")
    single_bracket = single_bracket[single_bracket.Seed.isin(expected)]
    smooth_rows = []
    for method, adjacent_frame, bracket_frame in (
        ("dual_head_bandit", pd.DataFrame(adjacent), pd.DataFrame(bracket)),
        ("single_head_shared_bandit", single_adj, single_bracket),
    ):
        item = {
            "Method": method,
            "SeedCount": int(adjacent_frame.Seed.nunique()),
            "AdjacentTop150JaccardMean": float(adjacent_frame.Top150Jaccard.mean()),
            "AdjacentTop150JaccardMin": float(adjacent_frame.Top150Jaccard.min()),
        }
        for metric in METRICS:
            column = f"{metric}_within_seen_bracket"
            item[f"{metric}_within_bracket_count"] = int(bracket_frame[column].sum())
            item[f"{metric}_within_bracket_total"] = int(len(bracket_frame))
            item[f"{metric}_within_bracket_rate"] = float(bracket_frame[column].mean())
        smooth_rows.append(item)
    pd.DataFrame(smooth_rows).to_csv(output / "interpolation_smoothness_comparison.csv", index=False)

    (output / "analysis_metadata.json").write_text(json.dumps({
        "expected_dual_head_seeds": list(expected),
        "single_shared_matched_seeds": list(expected),
        "scalar_bandit_matched_seeds": scalar_seeds,
        "scalar_bandit_limitation": (
            "No existing scalar contextual-bandit run exists for seed48; paired scalar comparison is n=2."
        ),
        "training_performed_by_analysis": False,
        "test_labels_read": False,
        "decision_note": (
            "With n=3, 'significant Recovery loss' is not inferentially testable; report paired directions "
            "and effect sizes descriptively."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dual-head contextual-bandit comparison to {output}")


if __name__ == "__main__":
    main()
