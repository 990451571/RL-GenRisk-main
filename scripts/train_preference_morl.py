#!/usr/bin/env python3
"""Preference-conditioned Recovery–Discovery DDQN training (validation only).

One shared Q network receives w=(w_recovery,w_discovery), samples only the
declared seen preferences during training, and is evaluated by greedy rollout
for seen and held-out interpolation preferences. Test labels are never read.
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
DEFAULT_UNSEEN = "[[0.9,0.1],[0.65,0.35],[0.35,0.65],[0.1,0.9]]"


def parse_preferences(raw, label):
    try:
        values = np.asarray(json.loads(raw), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a JSON list of [recovery, discovery] pairs.") from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"{label} must have shape (n,2), got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError(f"{label} entries must be finite, non-negative and sum to 1.")
    return [row.copy() for row in values]


def pref_name(pref):
    return f"r{pref[0]:.2f}_d{pref[1]:.2f}"


def greedy_rollout(agent, env, preference, disc_sets):
    """Primary evaluation policy: greedily select 150 actions conditioned on w."""
    import torch

    agent.Q.eval()
    n = agent.n_actions
    mask = np.ones(n, dtype=np.int64)
    selected, embedding = [], None
    state = torch.as_tensor(env["node_features"], dtype=torch.float32, device=agent.Q.device)
    pref = torch.as_tensor(preference, dtype=torch.float32, device=agent.Q.device)
    with torch.no_grad():
        for _ in range(agent.selection_budget):
            action_mask = torch.as_tensor(mask, dtype=torch.long, device=agent.Q.device)
            q_values, embedding = agent.Q(embedding, state, action_mask, preference=pref)
            q = (
                agent.scalarize_q(q_values, pref) if agent.objective_dim == 2 else q_values
            ).detach().cpu().numpy()
            valid = np.flatnonzero(mask)
            action = int(valid[np.argmax(q[valid])])
            selected.append(action)
            mask[action] = 0
    genes = [env["gene_name"][i] for i in selected]
    labels = set(env["validation_driver_genes"])
    recovery = train.metrics_at_k([{"Gene": gene} for gene in genes], labels, agent.selection_budget)
    lowfreq = sum(gene in disc_sets["lowfreq_novel"] for gene in genes)
    supported = sum(gene in disc_sets["evidence_supported"] for gene in genes)
    precision = supported / lowfreq if lowfreq else 0.0
    pool_precision = disc_sets["n_evidence_supported"] / disc_sets["n_lowfreq_novel"]
    return {
        "NDCG@150": recovery["NDCG"],
        "Recall@150": recovery["Recall"],
        "HitCount@150": recovery["HitCount"],
        "LowFreqNovel@150": int(lowfreq),
        "HighEvidenceLowFreqNovel@150": int(supported),
        "DiscoveryPrecision@150": precision,
        "DiscoveryFoldEnrichment@150": precision / pool_precision if pool_precision else 0.0,
        "selected_genes": genes,
    }


def dominates(a, b):
    # Fold enrichment is a constant multiple of precision for a fixed candidate pool.
    keys = ("NDCG@150", "Recall@150", "DiscoveryPrecision@150")
    return all(a[key] >= b[key] for key in keys) and any(a[key] > b[key] for key in keys)


def pareto_front(rows):
    return [row for row in rows if not any(other is not row and dominates(other, row) for other in rows)]


def write_rows(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--morl-train-preferences", default=DEFAULT_SEEN)
    custom.add_argument("--morl-unseen-preferences", default=DEFAULT_UNSEEN)
    custom.add_argument("--morl-eval-interval", type=int, default=1)
    custom.add_argument("--morl-checkpoint-dirname", default="pareto_checkpoints")
    custom.add_argument("--vector-morl", action="store_true")
    custom.add_argument(
        "--vector-value-normalization",
        choices=("popart", "legacy_loss_scale"),
        default="popart",
        help="Vector-Q only: PopArt is the default; legacy_loss_scale exists only to replay old checkpoints.",
    )
    custom_args, base_argv = custom.parse_known_args()
    if custom_args.morl_eval_interval <= 0:
        raise ValueError("--morl-eval-interval must be positive.")
    sys.argv = [sys.argv[0], *base_argv]
    args = train.parse_args()
    args.reward_mode = "rd_scan"
    if args.resume:
        raise ValueError("MORL resume is disabled: replay preferences are intentionally not checkpointed.")
    train.validate_no_test_path(args)
    train.validate_training_args(args)
    seen = parse_preferences(custom_args.morl_train_preferences, "--morl-train-preferences")
    unseen = parse_preferences(custom_args.morl_unseen_preferences, "--morl-unseen-preferences")
    if {tuple(x) for x in seen} & {tuple(x) for x in unseen}:
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
        q_preference_dim=0 if custom_args.vector_morl else None,
        objective_dim=2 if custom_args.vector_morl else 1,
        vector_value_normalization=(
            custom_args.vector_value_normalization if custom_args.vector_morl else "legacy_loss_scale"
        ),
    )

    import pandas as pd
    ev_df = pd.read_csv(train.resolve_path(args.lowfreq_evidence_path, base=train.REPO_DIR))
    known = set(env["train_driver_genes"]) | set(env["validation_driver_genes"])
    disc_sets = rd_definitions.discovery_sets_from_evidence(ev_df, known, args.rd_evidence_min)
    checkpoint_dir = run_dir / custom_args.morl_checkpoint_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A shuffled five-preference cycle is repeated, giving exactly equal exposure
    # whenever max_episodes is a multiple of five (50 -> ten episodes each).
    rng = np.random.default_rng(args.seed)
    cycle = [seen[i] for i in rng.permutation(len(seen))]
    schedule = [cycle[i % len(cycle)] for i in range(args.max_episodes)]
    schedule_counts = {
        pref_name(pref): int(sum(np.array_equal(pref, item) for item in schedule))
        for pref in seen
    }
    config = {
        "morl": True,
        "vector_q": bool(custom_args.vector_morl),
        "objective_dim": int(agent.objective_dim),
        "q_preference_dim": int(agent.q_preference_dim),
        "vector_value_normalization": agent.vector_value_normalization,
        "objective_scale_normalization": (
            (
                "PopArt EMA mean/std per head; raw-Q-preserving output rescaling; "
                "normalized TD loss, PER priority, and action scalarization."
                if agent.vector_value_normalization == "popart" else
                "Legacy EMA abs TD-target scale per head; used for loss, PER priority and action scalarization."
            ) if custom_args.vector_morl else None
        ),
        "seen_preferences": [x.tolist() for x in seen],
        "unseen_preferences": [x.tolist() for x in unseen],
        "evaluation_policy": "greedy_rollout",
        "pareto_objectives": ["NDCG@150", "Recall@150", "DiscoveryPrecision@150"],
        "fold_enrichment_note": "Recorded but excluded from dominance because it is precision / fixed pool prevalence.",
        "candidate_pool": {k: disc_sets[k] for k in ("n_lowfreq_novel", "n_evidence_supported")},
        "training_schedule": [pref_name(item) for item in schedule],
        "trained_preference_counts": schedule_counts,
        "test_labels_read": False,
        "base_args": vars(args),
    }
    train.write_json(run_dir / "morl_config.json", config)

    rollout_rows = []
    training_rows = []
    retained = set()
    for episode, preference in enumerate(schedule, start=1):
        episode_started = time.perf_counter()
        print(
            f"[MORL seed={args.seed}] episode {episode}/{args.max_episodes} "
            f"train_preference={pref_name(preference)}: collecting transitions...",
            flush=True,
        )
        agent.set_preference(preference)
        train_result = train.run_episode(
            agent, env, args, episode, run_dir=run_dir, preference=preference
        )
        training_rows.append({
            "episode": episode,
            "train_preference": pref_name(preference),
            "w_recovery": float(preference[0]),
            "w_discovery": float(preference[1]),
            "episode_reward_legacy_scalar": float(train_result["episode_reward"]),
            "mean_loss": train_result["mean_loss"],
            "td_error_abs_mean": train_result["td_error_abs_mean"],
            "gradient_norm_mean": train_result["gradient_norm_mean"],
            "objective_q_mean_recovery": train_result["objective_q_mean_recovery"],
            "objective_q_mean_discovery": train_result["objective_q_mean_discovery"],
            "objective_q_scale_recovery": train_result["objective_q_scale_recovery"],
            "objective_q_scale_discovery": train_result["objective_q_scale_discovery"],
            "epsilon": train_result["epsilon"],
            "learn_count": train_result["learn_count"],
            "test_labels_read": False,
        })
        write_rows(run_dir / "morl_training_metrics.csv", training_rows)
        print(
            f"[MORL seed={args.seed}] episode {episode}/{args.max_episodes}: "
            f"steps={train_result['steps']}, reward={train_result['episode_reward']:.4f}, "
            f"epsilon={train_result['epsilon']:.4f}, learn_count={train_result['learn_count']}; "
            "greedy rollout evaluation...",
            flush=True,
        )
        if episode % custom_args.morl_eval_interval:
            continue
        checkpoint = checkpoint_dir / f"episode_{episode:03d}.pt"
        train.save_checkpoint(agent, args, env, checkpoint, episode, 0.0)
        for scope, preferences in (("seen", seen), ("unseen", unseen)):
            for pref_index, eval_pref in enumerate(preferences, start=1):
                metric = greedy_rollout(agent, env, eval_pref, disc_sets)
                metric.update({
                    "episode": episode,
                    "scope": scope,
                    "preference": pref_name(eval_pref),
                    "w_recovery": float(eval_pref[0]),
                    "w_discovery": float(eval_pref[1]),
                })
                metric.pop("selected_genes")
                rollout_rows.append(metric)
                print(
                    f"[MORL seed={args.seed}] episode {episode}/{args.max_episodes} "
                    f"{scope} {pref_index}/{len(preferences)} {pref_name(eval_pref)}: "
                    f"NDCG={metric['NDCG@150']:.4f}, Recall={metric['Recall@150']:.4f}, "
                    f"DiscoveryPrecision={metric['DiscoveryPrecision@150']:.4f}, "
                    f"Fold={metric['DiscoveryFoldEnrichment@150']:.4f}",
                    flush=True,
                )

        # Pareto selection never consults unseen preferences.
        retained = set()
        fronts = {}
        for pref in seen:
            key = pref_name(pref)
            rows = [r for r in rollout_rows if r["scope"] == "seen" and r["preference"] == key]
            front = pareto_front(rows)
            fronts[key] = sorted({int(r["episode"]) for r in front})
            retained.update(fronts[key])
        for path in checkpoint_dir.glob("episode_*.pt"):
            if int(path.stem.split("_")[-1]) not in retained:
                path.unlink()
        train.write_json(run_dir / "pareto_manifest.json", {
            "selection_scope": "seen preferences only",
            "front_episodes_by_preference": fronts,
            "retained_union_episodes": sorted(retained),
            "test_labels_read": False,
        })
        write_rows(run_dir / "morl_rollout_metrics.csv", rollout_rows)
        print(
            f"[MORL seed={args.seed}] episode {episode}/{args.max_episodes}: "
            f"Pareto checkpoints retained={len(retained)}; "
            f"elapsed={time.perf_counter() - episode_started:.1f}s",
            flush=True,
        )

    summary = {
        "status": "COMPLETED",
        "run_dir": str(run_dir),
        "trained_preferences": [x.tolist() for x in seen],
        "trained_preference_counts": schedule_counts,
        "unseen_evaluation_preferences": [x.tolist() for x in unseen],
        "pareto_checkpoint_count": len(retained),
        "test_labels_read": False,
    }
    train.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
