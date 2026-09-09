#!/usr/bin/env python3
"""Validation-only interpolation audit for frozen shared-bandit checkpoints.

The script evaluates seen and held-out intermediate preferences with greedy
rollout.  It never trains, selects a checkpoint, or reads Test labels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import tempfile
from argparse import Namespace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import rd_definitions  # noqa: E402
import train  # noqa: E402
from train_preference_bandit import greedy_rollout, parse_preferences, pref_name  # noqa: E402


DEFAULT_UNSEEN = "[[0.9,0.1],[0.65,0.35],[0.35,0.65],[0.1,0.9]]"
METRICS = (
    "NDCG@150",
    "Recall@150",
    "DiscoveryPrecision@150",
    "DiscoveryFoldEnrichment@150",
)


def resolve_runs(patterns):
    runs = []
    for pattern in patterns:
        matches = glob.glob(pattern) if any(char in pattern for char in "*?[") else [pattern]
        runs.extend(Path(item) for item in matches)
    return sorted(set(runs))


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandit-runs", nargs="+", required=True)
    parser.add_argument("--unseen-preferences", default=DEFAULT_UNSEEN)
    parser.add_argument("--expected-seeds", default="42,45,46,47,48")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_seeds = tuple(int(token) for token in args.expected_seeds.split(",") if token.strip())
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("--expected-seeds must contain distinct integer seeds.")
    unseen = parse_preferences(args.unseen_preferences)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    top_rows = []
    policy_sets = {}
    seen_by_seed = {}
    observed_seeds = []

    for run in resolve_runs(args.bandit_runs):
        config = json.loads((run / "preference_bandit_config.json").read_text(encoding="utf-8"))
        if config.get("model") != "shared_preference_conditioned_contextual_bandit":
            raise ValueError(f"Not a shared preference bandit run: {run}")
        if config.get("td_target") != "immediate_reward_only":
            raise ValueError(f"Run is not an immediate-reward bandit: {run}")
        if config.get("schedule_policy") != "balanced_latin_blocks_v1":
            raise ValueError(f"Run does not use the phase-balanced schedule: {run}")
        base_args = Namespace(**config["base_args"])
        base_args.device = "cpu"
        train.validate_no_test_path(base_args)
        seed = int(base_args.seed)
        observed_seeds.append(seed)
        seen = parse_preferences(json.dumps(config["seen_preferences"]))
        if {tuple(item) for item in seen} & {tuple(item) for item in unseen}:
            raise ValueError("Seen and unseen preferences overlap.")
        seen_by_seed[seed] = {pref_name(item) for item in seen}

        with tempfile.TemporaryDirectory(prefix="preference_bandit_interpolation_") as temp:
            env = train.build_environment(base_args, Path(temp))
            agent = train.build_agent(
                base_args,
                env,
                torch.device("cpu"),
                preference_dim=2,
                learning_mode="contextual_bandit",
            )
            train.load_checkpoint(agent, run / "checkpoint_final.pt", base_args, env)
            evidence = pd.read_csv(
                train.resolve_path(base_args.lowfreq_evidence_path, base=train.REPO_DIR)
            )
            known = set(env["train_driver_genes"]) | set(env["validation_driver_genes"])
            discovery_sets = rd_definitions.discovery_sets_from_evidence(
                evidence, known, base_args.rd_evidence_min
            )
            for scope, preferences in (("seen", seen), ("unseen", unseen)):
                for preference in preferences:
                    result = greedy_rollout(agent, env, base_args, preference, discovery_sets)
                    order = result.pop("order")
                    result.pop("scores")
                    key = pref_name(preference)
                    genes = [env["gene_name"][index] for index in order[:agent.selection_budget]]
                    policy_sets[(seed, key)] = set(genes)
                    metric_rows.append({
                        "Seed": seed,
                        "Scope": scope,
                        "Preference": key,
                        "w_recovery": float(preference[0]),
                        "w_discovery": float(preference[1]),
                        **result,
                        "Run": str(run),
                        "Checkpoint": "final",
                        "test_labels_read": False,
                    })
                    top_rows.extend({
                        "Seed": seed,
                        "Scope": scope,
                        "Preference": key,
                        "Rank": rank,
                        "Gene": gene,
                        "test_labels_read": False,
                    } for rank, gene in enumerate(genes, start=1))

    if tuple(sorted(observed_seeds)) != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"Observed seeds {sorted(observed_seeds)} do not match expected {sorted(expected_seeds)}."
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["Seed", "w_discovery"])
    metrics.to_csv(output / "interpolation_metrics_per_seed.csv", index=False)
    write_csv(output / "interpolation_top150.csv", top_rows)

    aggregate_rows = []
    for (scope, preference, w_disc), group in metrics.groupby(
        ["Scope", "Preference", "w_discovery"], sort=False
    ):
        row = {
            "Scope": scope,
            "Preference": preference,
            "w_discovery": float(w_disc),
            "SeedCount": int(group.Seed.nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).sort_values("w_discovery").to_csv(
        output / "interpolation_metrics_summary.csv", index=False
    )

    adjacent_rows = []
    bracket_rows = []
    trend_rows = []
    for seed, group in metrics.groupby("Seed"):
        ordered = group.sort_values("w_discovery")
        records = list(ordered.to_dict("records"))
        for left, right in zip(records, records[1:]):
            left_set = policy_sets[(seed, left["Preference"])]
            right_set = policy_sets[(seed, right["Preference"])]
            adjacent_rows.append({
                "Seed": int(seed),
                "LeftPreference": left["Preference"],
                "RightPreference": right["Preference"],
                "LeftScope": left["Scope"],
                "RightScope": right["Scope"],
                "Top150Jaccard": jaccard(left_set, right_set),
                "Top150SharedGeneCount": len(left_set & right_set),
                "test_labels_read": False,
            })

        seen_records = [row for row in records if row["Scope"] == "seen"]
        for row in (item for item in records if item["Scope"] == "unseen"):
            lower = max(
                (item for item in seen_records if item["w_discovery"] < row["w_discovery"]),
                key=lambda item: item["w_discovery"],
            )
            upper = min(
                (item for item in seen_records if item["w_discovery"] > row["w_discovery"]),
                key=lambda item: item["w_discovery"],
            )
            output_row = {
                "Seed": int(seed),
                "UnseenPreference": row["Preference"],
                "LowerSeenPreference": lower["Preference"],
                "UpperSeenPreference": upper["Preference"],
                "JaccardToLower": jaccard(
                    policy_sets[(seed, row["Preference"])],
                    policy_sets[(seed, lower["Preference"])],
                ),
                "JaccardToUpper": jaccard(
                    policy_sets[(seed, row["Preference"])],
                    policy_sets[(seed, upper["Preference"])],
                ),
                "test_labels_read": False,
            }
            fraction = (
                (row["w_discovery"] - lower["w_discovery"])
                / (upper["w_discovery"] - lower["w_discovery"])
            )
            for metric in METRICS:
                expected = lower[metric] + fraction * (upper[metric] - lower[metric])
                low, high = sorted((lower[metric], upper[metric]))
                output_row[f"{metric}_linear_expected"] = expected
                output_row[f"{metric}_minus_linear_expected"] = row[metric] - expected
                output_row[f"{metric}_within_seen_bracket"] = bool(
                    low - 1e-12 <= row[metric] <= high + 1e-12
                )
            bracket_rows.append(output_row)

        trend_rows.append({
            "Seed": int(seed),
            "PreferenceCount": int(len(ordered)),
            "UniqueTop150Policies": int(len({
                frozenset(policy_sets[(seed, row["Preference"])]) for row in records
            })),
            "RecoveryNDCGSpearman": float(
                ordered.w_discovery.corr(ordered["NDCG@150"], method="spearman")
            ),
            "RecoveryRecallSpearman": float(
                ordered.w_discovery.corr(ordered["Recall@150"], method="spearman")
            ),
            "DiscoveryPrecisionSpearman": float(
                ordered.w_discovery.corr(ordered["DiscoveryPrecision@150"], method="spearman")
            ),
            "DiscoveryFoldSpearman": float(
                ordered.w_discovery.corr(ordered["DiscoveryFoldEnrichment@150"], method="spearman")
            ),
            "test_labels_read": False,
        })

    pd.DataFrame(adjacent_rows).to_csv(output / "interpolation_adjacent_jaccard.csv", index=False)
    pd.DataFrame(bracket_rows).to_csv(output / "interpolation_unseen_brackets.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(output / "interpolation_trend_by_seed.csv", index=False)

    across_seed_rows = []
    for preference, group in metrics.groupby("Preference"):
        sets = [(int(row.Seed), policy_sets[(int(row.Seed), preference)]) for _, row in group.iterrows()]
        overlaps = [jaccard(left, right) for (_, left), (_, right) in combinations(sets, 2)]
        across_seed_rows.append({
            "Scope": group.Scope.iloc[0],
            "Preference": preference,
            "w_discovery": float(group.w_discovery.iloc[0]),
            "SeedCount": len(sets),
            "PairCount": len(overlaps),
            "Top150JaccardMean": float(np.mean(overlaps)),
            "Top150JaccardSD": float(np.std(overlaps, ddof=1)),
            "test_labels_read": False,
        })
    pd.DataFrame(across_seed_rows).sort_values("w_discovery").to_csv(
        output / "interpolation_across_seed_stability.csv", index=False
    )

    (output / "analysis_metadata.json").write_text(json.dumps({
        "analysis": "frozen final-checkpoint greedy-rollout preference interpolation",
        "seen_preferences": sorted(next(iter(seen_by_seed.values()))),
        "unseen_preferences": [pref_name(item) for item in unseen],
        "expected_seeds": list(expected_seeds),
        "checkpoint_selection": "none; frozen final checkpoints only",
        "training_performed": False,
        "test_labels_read": False,
        "caveat": (
            "Validation metrics are discrete at Top-150; linear interpolation residuals are diagnostics, "
            "not a requirement that biological utility be exactly linear."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote shared-bandit interpolation audit to {output}")


if __name__ == "__main__":
    main()
