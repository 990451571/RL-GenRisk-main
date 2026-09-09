#!/usr/bin/env python3
"""Audit Top-k policy-set interpolation for retained MORL checkpoints.

For the latest retained checkpoint of each supplied MORL run, this replays the
same greedy policy for seen and unseen preferences, then records adjacent
Top-k gene-set Jaccard overlaps.  It is a policy-level diagnostic only; it
does not read Test labels or participate in checkpoint/model selection.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import rd_definitions  # noqa: E402
import train  # noqa: E402
from train_preference_morl import greedy_rollout, pref_name  # noqa: E402


def resolve_runs(patterns):
    runs = []
    for pattern in patterns:
        matches = glob.glob(pattern) if any(char in pattern for char in "*?[") else [pattern]
        runs.extend(Path(item) for item in matches)
    return sorted(set(runs))


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morl-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    adjacent_rows, bracket_rows = [], []
    for run in resolve_runs(args.morl_runs):
        config = json.loads((run / "morl_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run / "pareto_manifest.json").read_text(encoding="utf-8"))
        base_args = Namespace(**config["base_args"])
        base_args.device = "cpu"  # deterministic diagnostic; no update is performed.
        train.validate_no_test_path(base_args)
        episode = max(int(item) for item in manifest["retained_union_episodes"])
        checkpoint = run / "pareto_checkpoints" / f"episode_{episode:03d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Retained checkpoint missing: {checkpoint}")

        with tempfile.TemporaryDirectory(prefix="morl_policy_audit_") as temp:
            env = train.build_environment(base_args, Path(temp))
            vector_q = bool(config.get("vector_q", False))
            agent = train.build_agent(
                base_args,
                env,
                torch.device("cpu"),
                preference_dim=2,
                q_preference_dim=0 if vector_q else None,
                objective_dim=2 if vector_q else 1,
                vector_value_normalization=(
                    config.get("vector_value_normalization", "legacy_loss_scale")
                    if vector_q else "legacy_loss_scale"
                ),
            )
            train.load_checkpoint(agent, checkpoint, base_args, env)
            evidence = pd.read_csv(train.resolve_path(base_args.lowfreq_evidence_path, base=train.REPO_DIR))
            known = set(env["train_driver_genes"]) | set(env["validation_driver_genes"])
            disc_sets = rd_definitions.discovery_sets_from_evidence(
                evidence, known, base_args.rd_evidence_min
            )
            values = [("seen", np.asarray(item, dtype=np.float32)) for item in config["seen_preferences"]]
            values += [("unseen", np.asarray(item, dtype=np.float32)) for item in config["unseen_preferences"]]
            policies = {}
            for scope, preference in values:
                key = pref_name(preference)
                result = greedy_rollout(agent, env, preference, disc_sets)
                policies[key] = {
                    "scope": scope,
                    "preference": preference,
                    "genes": set(result["selected_genes"]),
                }

        ordered = sorted(policies.values(), key=lambda item: float(item["preference"][1]))
        seed = int(config["base_args"]["seed"])
        for left, right in zip(ordered, ordered[1:]):
            adjacent_rows.append({
                "seed": seed,
                "checkpoint_episode": episode,
                "left_preference": pref_name(left["preference"]),
                "right_preference": pref_name(right["preference"]),
                "left_scope": left["scope"],
                "right_scope": right["scope"],
                "top150_jaccard": jaccard(left["genes"], right["genes"]),
                "top150_shared_gene_count": len(left["genes"] & right["genes"]),
                "test_labels_read": False,
            })
        seen = sorted(
            (item for item in policies.values() if item["scope"] == "seen"),
            key=lambda item: float(item["preference"][1]),
        )
        for unseen in (item for item in policies.values() if item["scope"] == "unseen"):
            weight = float(unseen["preference"][1])
            left = max((item for item in seen if float(item["preference"][1]) < weight), key=lambda item: float(item["preference"][1]))
            right = min((item for item in seen if float(item["preference"][1]) > weight), key=lambda item: float(item["preference"][1]))
            bracket_rows.append({
                "seed": seed,
                "checkpoint_episode": episode,
                "unseen_preference": pref_name(unseen["preference"]),
                "lower_seen_preference": pref_name(left["preference"]),
                "upper_seen_preference": pref_name(right["preference"]),
                "jaccard_to_lower_seen": jaccard(unseen["genes"], left["genes"]),
                "jaccard_to_upper_seen": jaccard(unseen["genes"], right["genes"]),
                "test_labels_read": False,
            })

    pd.DataFrame(adjacent_rows).to_csv(output / "policy_adjacent_jaccard.csv", index=False)
    pd.DataFrame(bracket_rows).to_csv(output / "policy_unseen_brackets.csv", index=False)
    print(f"Wrote policy interpolation diagnostics to {output}")


if __name__ == "__main__":
    main()
