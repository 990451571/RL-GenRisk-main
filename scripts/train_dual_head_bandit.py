#!/usr/bin/env python3
"""Dual-head immediate-value contextual bandit (Validation only).

The shared trunk outputs two action values, one for the frozen Recovery reward
and one for the frozen Discovery reward.  Each head regresses its own immediate
reward without bootstrap.  Preference is used only at action selection:
score = w_recovery * Q_recovery + w_discovery * Q_discovery.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import rd_definitions  # noqa: E402
import train  # noqa: E402
from train_preference_bandit import (  # noqa: E402
    DEFAULT_SEEN,
    SCHEDULE_POLICY,
    balanced_latin_schedule,
    greedy_rollout,
    parse_preferences,
    pref_name,
    write_ranking,
    write_rows,
)


DEFAULT_UNSEEN = "[[0.9,0.1],[0.65,0.35],[0.35,0.65],[0.1,0.9]]"


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--dual-train-preferences", default=DEFAULT_SEEN)
    custom.add_argument("--dual-unseen-preferences", default=DEFAULT_UNSEEN)
    custom_args, base_argv = custom.parse_known_args()
    sys.argv = [sys.argv[0], *base_argv]
    args = train.parse_args()
    if args.resume:
        raise ValueError("Resume is disabled: replay transitions are not serialized.")
    args.reward_mode = "rd_scan"
    args.learning_mode = "contextual_bandit"
    args.history_ablation = "full"
    train.validate_no_test_path(args)
    train.validate_training_args(args)

    seen = parse_preferences(custom_args.dual_train_preferences)
    unseen = parse_preferences(custom_args.dual_unseen_preferences)
    if {tuple(item) for item in seen} & {tuple(item) for item in unseen}:
        raise ValueError("Seen and unseen preferences must be disjoint.")

    run_dir = train.make_run_dir(args.output_dir, args.seed, args.feature_mode)
    train.setup_logger(run_dir)
    train.set_seed(args.seed)
    device = train.choose_device(args.device)
    env = train.build_environment(args, run_dir)
    agent = train.build_agent(
        args,
        env,
        device,
        preference_dim=2,
        q_preference_dim=0,
        objective_dim=2,
        vector_value_normalization="legacy_loss_scale",
        vector_action_scalarization="raw",
        learning_mode="contextual_bandit",
    )
    if (
        agent.learning_mode != "contextual_bandit"
        or agent.objective_dim != 2
        or agent.q_preference_dim != 0
        or agent.vector_action_scalarization != "raw"
    ):
        raise RuntimeError("Dual-head contextual bandit construction contract failed.")

    evidence = pd.read_csv(train.resolve_path(args.lowfreq_evidence_path, base=train.REPO_DIR))
    known = set(env["train_driver_genes"]) | set(env["validation_driver_genes"])
    discovery_sets = rd_definitions.discovery_sets_from_evidence(
        evidence, known, args.rd_evidence_min
    )
    schedule = balanced_latin_schedule(seen, args.max_episodes, np.random.default_rng(args.seed))
    counts = {
        pref_name(pref): int(sum(np.array_equal(pref, item) for item in schedule))
        for pref in seen
    }
    phase_counts = {
        pref_name(pref): [
            int(sum(np.array_equal(pref, item) for item in schedule[position::len(seen)]))
            for position in range(len(seen))
        ]
        for pref in seen
    }

    train.write_json(run_dir / "dual_head_bandit_config.json", {
        "model": "dual_head_preference_contextual_bandit",
        "heads": ["Q_recovery_immediate", "Q_discovery_immediate"],
        "shared_trunk": True,
        "objective_dim": 2,
        "q_preference_dim": 0,
        "learning_mode": "contextual_bandit",
        "td_targets": ["reward_recovery_raw", "reward_discovery_raw"],
        "bootstrap": False,
        "action_scalarization": "raw_linear",
        "action_score_formula": "w_recovery * Q_recovery + w_discovery * Q_discovery",
        "loss_scale_handling": (
            "Per-head EMA target scale is used only to balance the two Huber losses; "
            "action scores use raw predicted values."
        ),
        "history_ablation": "full",
        "seen_preferences": [item.tolist() for item in seen],
        "unseen_preferences": [item.tolist() for item in unseen],
        "schedule_policy": SCHEDULE_POLICY,
        "training_schedule": [pref_name(item) for item in schedule],
        "trained_preference_counts": counts,
        "trained_preference_phase_counts": phase_counts,
        "evaluation_policy": "final_checkpoint_greedy_rollout",
        "checkpoint_selection": "none; final checkpoint only",
        "candidate_pool": {
            key: discovery_sets[key]
            for key in ("n_lowfreq_novel", "n_evidence_supported")
        },
        "test_labels_read": False,
        "base_args": vars(args),
    })

    training_rows = []
    for episode, preference in enumerate(schedule, start=1):
        started = time.perf_counter()
        agent.set_preference(preference)
        result = train.run_episode(
            agent, env, args, episode, run_dir=run_dir, preference=preference
        )
        training_rows.append({
            "episode": episode,
            "train_preference": pref_name(preference),
            "w_recovery": float(preference[0]),
            "w_discovery": float(preference[1]),
            "episode_reward_scalar_diagnostic": float(result["episode_reward"]),
            "mean_loss": result["mean_loss"],
            "td_error_abs_mean": result["td_error_abs_mean"],
            "gradient_norm_mean": result["gradient_norm_mean"],
            "loss_scale_recovery": result["objective_q_scale_recovery"],
            "loss_scale_discovery": result["objective_q_scale_discovery"],
            "epsilon": result["epsilon"],
            "learn_count": result["learn_count"],
            "test_labels_read": False,
        })
        write_rows(run_dir / "dual_head_bandit_training_metrics.csv", training_rows)
        print(
            f"[dual-head bandit seed={args.seed}] {episode}/{args.max_episodes} "
            f"w={pref_name(preference)} steps={result['steps']} "
            f"reward(diagnostic)={result['episode_reward']:.4f} "
            f"learn={result['learn_count']} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    checkpoint = run_dir / "checkpoint_final.pt"
    train.save_checkpoint(agent, args, env, checkpoint, args.max_episodes, 0.0)
    ranking_dir = run_dir / "final_rollout_rankings"
    ranking_dir.mkdir(exist_ok=True)
    final_rows = []
    for scope, preferences in (("seen", seen), ("unseen", unseen)):
        for preference in preferences:
            result = greedy_rollout(agent, env, args, preference, discovery_sets)
            key = pref_name(preference)
            ranking_path = ranking_dir / f"{key}.csv"
            write_ranking(
                ranking_path,
                result.pop("order"),
                result.pop("scores"),
                env["gene_name"],
            )
            result.update({
                "seed": int(args.seed),
                "scope": scope,
                "preference": key,
                "w_recovery": float(preference[0]),
                "w_discovery": float(preference[1]),
                "checkpoint": "final",
                "ranking_path": str(ranking_path),
                "test_labels_read": False,
            })
            final_rows.append(result)
            print(
                f"[dual-head bandit seed={args.seed}] final {scope} {key}: "
                f"NDCG={result['NDCG@150']:.4f} Recall={result['Recall@150']:.4f} "
                f"Precision={result['DiscoveryPrecision@150']:.4f} "
                f"Fold={result['DiscoveryFoldEnrichment@150']:.4f}",
                flush=True,
            )
    write_rows(run_dir / "dual_head_bandit_final_metrics.csv", final_rows)
    summary = {
        "status": "COMPLETED",
        "run_dir": str(run_dir),
        "checkpoint_final": str(checkpoint),
        "trained_preference_counts": counts,
        "evaluated_seen_preferences": [item.tolist() for item in seen],
        "evaluated_unseen_preferences": [item.tolist() for item in unseen],
        "bootstrap": False,
        "action_scalarization": "raw_linear",
        "test_labels_read": False,
    }
    train.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
