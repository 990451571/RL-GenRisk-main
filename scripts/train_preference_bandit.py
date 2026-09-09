#!/usr/bin/env python3
"""Shared preference-conditioned contextual bandit (validation-only).

One Q network receives ``w=(w_recovery,w_discovery)`` and directly regresses
the immediate scalar ``rd_scan`` reward for the action.  It deliberately does
not bootstrap from Q(s', a).  Each of five frozen preferences receives equal
episode exposure; the final checkpoint is the sole primary evaluation point,
so Validation metrics are never used to select a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import train  # noqa: E402
import rd_definitions  # noqa: E402


DEFAULT_SEEN = "[[1,0],[0.8,0.2],[0.5,0.5],[0.2,0.8],[0,1]]"
SCHEDULE_POLICY = "balanced_latin_blocks_v1"


def parse_preferences(raw):
    try:
        values = np.asarray(json.loads(raw), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("--bandit-train-preferences must be JSON [[recovery,discovery], ...].") from exc
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError(f"Preferences must have shape (n,2), got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Each preference must be finite, non-negative, and sum to 1.")
    return [row.copy() for row in values]


def pref_name(preference):
    return f"r{preference[0]:.2f}_d{preference[1]:.2f}"


def balanced_latin_schedule(preferences, max_episodes, rng):
    """Create equal-exposure, phase-balanced preference blocks.

    A repeated single permutation makes one preference always occur at the
    same position in a block.  That aliases preference with global training
    time, epsilon and replay age.  Here each set of ``n`` blocks is a Latin
    square: every preference occupies every within-block position once.
    """
    n_preferences = len(preferences)
    if max_episodes % n_preferences:
        raise ValueError("max_episodes must be divisible by the number of preferences.")
    n_blocks = max_episodes // n_preferences
    if n_blocks % n_preferences:
        raise ValueError(
            "Phase-balanced scheduling requires (max_episodes / n_preferences) "
            "to be divisible by n_preferences."
        )
    schedule = []
    for _ in range(n_blocks // n_preferences):
        base = rng.permutation(n_preferences)
        # Randomize the order of rotations, while retaining the Latin-square
        # guarantee that each preference fills every block position once.
        for shift in rng.permutation(n_preferences):
            schedule.extend(preferences[index] for index in np.roll(base, shift))
    if len(schedule) != max_episodes:
        raise RuntimeError("Internal error: preference schedule length mismatch.")
    return schedule


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def greedy_rollout(agent, env, args, preference, disc_sets):
    """Roll out the final policy under one preference and return full ranking."""
    import torch

    agent.Q.eval()
    n = agent.n_actions
    true_mask = np.ones(n, dtype=np.int64)
    selected, embedding = [], None
    q_at_select = np.full(n, -np.inf, dtype=np.float32)
    state = torch.as_tensor(env["node_features"], dtype=torch.float32, device=agent.Q.device)
    pref = torch.as_tensor(preference, dtype=torch.float32, device=agent.Q.device)
    final_q = None
    with torch.no_grad():
        for step in range(agent.selection_budget):
            history_mask = train.model_history_mask(true_mask, args, episode=0, step=step)
            context = torch.as_tensor(history_mask, dtype=torch.long, device=agent.Q.device)
            q_values, embedding = agent.Q(embedding, state, context, preference=pref)
            action_values = (
                agent.scalarize_q(q_values, pref)
                if agent.objective_dim == 2 else q_values
            )
            q = action_values.detach().cpu().numpy()
            valid = np.flatnonzero(true_mask)
            action = int(valid[np.argmax(q[valid])])
            selected.append(action)
            q_at_select[action] = q[action]
            true_mask[action] = 0
            final_q = q
    tail = sorted(np.flatnonzero(true_mask).tolist(), key=lambda i: (-final_q[i], env["gene_name"][i]))
    order = selected + tail
    top_genes = [env["gene_name"][i] for i in order[:agent.selection_budget]]
    recovery = train.metrics_at_k([{"Gene": gene} for gene in top_genes], set(env["validation_driver_genes"]), agent.selection_budget)
    lowfreq = sum(gene in disc_sets["lowfreq_novel"] for gene in top_genes)
    supported = sum(gene in disc_sets["evidence_supported"] for gene in top_genes)
    pool_precision = disc_sets["n_evidence_supported"] / disc_sets["n_lowfreq_novel"]
    precision = supported / lowfreq if lowfreq else 0.0
    return {
        "NDCG@150": recovery["NDCG"],
        "Recall@150": recovery["Recall"],
        "HitCount@150": recovery["HitCount"],
        "LowFreqNovel@150": int(lowfreq),
        "HighEvidenceLowFreqNovel@150": int(supported),
        "DiscoveryPrecision@150": precision,
        "DiscoveryFoldEnrichment@150": precision / pool_precision if pool_precision else 0.0,
        "order": order,
        "scores": np.where(q_at_select > -np.inf, q_at_select, final_q),
    }


def write_ranking(path, order, scores, gene_names):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Rank", "Gene", "Score"])
        for rank, index in enumerate(order, start=1):
            writer.writerow([rank, gene_names[index], float(scores[index])])


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--bandit-train-preferences", default=DEFAULT_SEEN)
    custom_args, base_argv = custom.parse_known_args()
    sys.argv = [sys.argv[0], *base_argv]
    args = train.parse_args()
    if args.resume:
        raise ValueError("Resume is disabled: replay transitions carry preferences and are not serialized.")
    args.reward_mode = "rd_scan"
    args.learning_mode = "contextual_bandit"
    # This experiment tests bootstrap removal plus preference conditioning. It
    # retains the ordinary, true selected-gene context; no history ablation is
    # mixed into the new mainline validation.
    args.history_ablation = "full"
    train.validate_no_test_path(args)
    train.validate_training_args(args)
    seen = parse_preferences(custom_args.bandit_train_preferences)
    if args.max_episodes % len(seen):
        raise ValueError("--max_episodes must be divisible by the number of seen preferences for equal exposure.")

    run_dir = train.make_run_dir(args.output_dir, args.seed, args.feature_mode)
    train.setup_logger(run_dir)
    train.set_seed(args.seed)
    device = train.choose_device(args.device)
    env = train.build_environment(args, run_dir)
    agent = train.build_agent(
        args, env, device, preference_dim=2, learning_mode="contextual_bandit"
    )
    if agent.learning_mode != "contextual_bandit" or agent.preference_dim != 2:
        raise RuntimeError("Shared preference bandit was not constructed as contextual_bandit with preference_dim=2.")

    import pandas as pd
    evidence = pd.read_csv(train.resolve_path(args.lowfreq_evidence_path, base=train.REPO_DIR))
    known = set(env["train_driver_genes"]) | set(env["validation_driver_genes"])
    disc_sets = rd_definitions.discovery_sets_from_evidence(evidence, known, args.rd_evidence_min)
    rng = np.random.default_rng(args.seed)
    schedule = balanced_latin_schedule(seen, args.max_episodes, rng)
    counts = {pref_name(pref): int(sum(np.array_equal(pref, item) for item in schedule)) for pref in seen}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Unequal preference exposure: {counts}")
    phase_counts = {
        pref_name(pref): [
            int(sum(np.array_equal(pref, item) for item in schedule[position::len(seen)]))
            for position in range(len(seen))
        ]
        for pref in seen
    }
    expected_per_phase = args.max_episodes // (len(seen) * len(seen))
    if any(values != [expected_per_phase] * len(seen) for values in phase_counts.values()):
        raise RuntimeError(f"Preference schedule is not phase balanced: {phase_counts}")

    train.write_json(run_dir / "preference_bandit_config.json", {
        "model": "shared_preference_conditioned_contextual_bandit",
        "learning_mode": "contextual_bandit",
        "td_target": "immediate_reward_only",
        "history_ablation": "full",
        "seen_preferences": [item.tolist() for item in seen],
        "schedule_policy": SCHEDULE_POLICY,
        "training_schedule": [pref_name(item) for item in schedule],
        "trained_preference_counts": counts,
        "trained_preference_phase_counts": phase_counts,
        "evaluation_policy": "final_checkpoint_greedy_rollout",
        "checkpoint_selection": "none; final checkpoint only",
        "candidate_pool": {key: disc_sets[key] for key in ("n_lowfreq_novel", "n_evidence_supported")},
        "test_labels_read": False,
        "base_args": vars(args),
    })

    training_rows = []
    for episode, preference in enumerate(schedule, start=1):
        started = time.perf_counter()
        agent.set_preference(preference)
        result = train.run_episode(agent, env, args, episode, run_dir=run_dir, preference=preference)
        training_rows.append({
            "episode": episode,
            "train_preference": pref_name(preference),
            "w_recovery": float(preference[0]),
            "w_discovery": float(preference[1]),
            "episode_reward_raw_diagnostic": float(result["episode_reward"]),
            "mean_loss": result["mean_loss"],
            "td_error_abs_mean": result["td_error_abs_mean"],
            "gradient_norm_mean": result["gradient_norm_mean"],
            "epsilon": result["epsilon"],
            "learn_count": result["learn_count"],
            "test_labels_read": False,
        })
        write_rows(run_dir / "preference_bandit_training_metrics.csv", training_rows)
        print(
            f"[preference-bandit seed={args.seed}] {episode}/{args.max_episodes} "
            f"w={pref_name(preference)} steps={result['steps']} "
            f"reward(diagnostic)={result['episode_reward']:.4f} "
            f"learn={result['learn_count']} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    checkpoint = run_dir / "checkpoint_final.pt"
    train.save_checkpoint(agent, args, env, checkpoint, args.max_episodes, 0.0)
    rankings = run_dir / "final_rollout_rankings"
    rankings.mkdir(exist_ok=True)
    final_rows = []
    for preference in seen:
        metric = greedy_rollout(agent, env, args, preference, disc_sets)
        key = pref_name(preference)
        ranking_path = rankings / f"{key}.csv"
        write_ranking(ranking_path, metric.pop("order"), metric.pop("scores"), env["gene_name"])
        metric.update({
            "seed": int(args.seed), "preference": key,
            "w_recovery": float(preference[0]), "w_discovery": float(preference[1]),
            "checkpoint": "final", "ranking_path": str(ranking_path), "test_labels_read": False,
        })
        final_rows.append(metric)
        print(
            f"[preference-bandit seed={args.seed}] final {key}: "
            f"NDCG={metric['NDCG@150']:.4f} Recall={metric['Recall@150']:.4f} "
            f"Precision={metric['DiscoveryPrecision@150']:.4f} "
            f"Fold={metric['DiscoveryFoldEnrichment@150']:.4f}", flush=True,
        )
    write_rows(run_dir / "preference_bandit_final_metrics.csv", final_rows)
    summary = {
        "status": "COMPLETED", "run_dir": str(run_dir), "checkpoint_final": str(checkpoint),
        "trained_preferences": [item.tolist() for item in seen], "trained_preference_counts": counts,
        "evaluation_policy": "final_checkpoint_greedy_rollout", "test_labels_read": False,
    }
    train.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
