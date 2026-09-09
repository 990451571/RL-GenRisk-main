#!/usr/bin/env python3
"""Validation-only diagnostics for decomposed two-head vector-Q MORL.

This reconstructs logged transitions from finished vector-Q MORL runs.  It
never reads Test labels, changes a checkpoint, or claims historical PER
sampling statistics (the replay buffer is deliberately not serialized).
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import train  # noqa: E402


OBJECTIVES = ("recovery", "discovery")


def pref_name(values):
    return f"r{float(values[0]):.2f}_d{float(values[1]):.2f}"


def resolve_runs(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern) if any(char in pattern for char in "*?[") else [pattern])
    return sorted({Path(path) for path in paths})


def latest_checkpoint(run_dir):
    manifest = json.loads((run_dir / "pareto_manifest.json").read_text(encoding="utf-8"))
    episode = max(int(value) for value in manifest["retained_union_episodes"])
    path = run_dir / "pareto_checkpoints" / f"episode_{episode:03d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing retained checkpoint: {path}")
    return episode, path


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "p50": float(np.quantile(values, 0.5)), "p90": float(np.quantile(values, 0.9)),
    }


def cosine(left, right):
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else float("nan")


def gradient_vector(model, include):
    values = []
    for name, parameter in model.named_parameters():
        if include(name):
            grad = parameter.grad
            values.append((torch.zeros_like(parameter) if grad is None else grad).detach().reshape(-1))
    return torch.cat(values).cpu().numpy()


def reconstruct_records(log, schedule, n_actions, per_preference, rng):
    log = log.copy()
    log["preference"] = log["episode"].map(lambda ep: schedule[int(ep) - 1])
    required = {"reward_recovery_raw", "reward_discovery_raw"}
    missing = required - set(log.columns)
    if missing:
        raise ValueError(f"Vector-Q action log lacks raw reward columns: {sorted(missing)}")
    selected = set()
    counts = {}
    for preference, group in log.groupby("preference", sort=True):
        counts[preference] = int(len(group))
        take = min(per_preference, len(group))
        selected.update(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())
        # Always retain the true terminal transition.  A uniform 64/1500 probe
        # can otherwise miss it, making a terminal-bootstrap audit meaningless.
        selected.update(group.loc[group["done"].astype(bool)].index.to_list())

    records = []
    for _, episode_rows in log.groupby("episode", sort=True):
        mask = np.ones(n_actions, dtype=np.int64)
        for row_index, row in episode_rows.sort_values("step").iterrows():
            action = int(row["action_index"])
            if mask[action] != 1:
                raise ValueError(f"Invalid/repeated action in episode {row['episode']}")
            next_mask = mask.copy()
            next_mask[action] = 0
            if row_index in selected:
                records.append({
                    "preference": row["preference"], "action": action,
                    "reward": np.asarray([row["reward_recovery_raw"], row["reward_discovery_raw"]], dtype=np.float32),
                    "done": float(bool(row["done"])), "mask": mask.copy(), "next_mask": next_mask,
                })
            mask = next_mask
    return records, counts


def td_for_record(agent, state, online_embedding, target_embedding, record):
    """Return prediction, target, reward and bootstrap for one logged transition."""
    preference = preference_vector(record["preference"])
    mask = torch.as_tensor(record["mask"], dtype=torch.long)
    next_mask = torch.as_tensor(record["next_mask"], dtype=torch.long)
    current, _ = agent.Q(online_embedding, state, mask)
    prediction_normalized = current[record["action"]]
    prediction = agent.denormalize_objective_q(prediction_normalized)
    with torch.no_grad():
        online_next, _ = agent.Q(online_embedding, state, next_mask)
        scores = agent.scalarize_q(online_next, preference)
        next_action = int(scores.masked_fill(next_mask == 0, -1e9).argmax().item())
        target_next, _ = agent.Q_target(target_embedding, state, next_mask)
        target_next = agent.denormalize_objective_q(target_next)
        reward = torch.as_tensor(record["reward"], dtype=torch.float32)
        bootstrap = agent.gamma * (1.0 - record["done"]) * target_next[next_action]
        target = reward + bootstrap
    return prediction, target, reward, bootstrap, prediction_normalized


def preference_vector(name):
    # Names originate internally from pref_name and avoid ambiguous float parsing.
    recovery, discovery = name[1:].split("_d")
    return torch.tensor([float(recovery), float(discovery)], dtype=torch.float32)


def diagnose_one(run_dir, samples_per_preference, n_fixed_states, topk):
    config = json.loads((run_dir / "morl_config.json").read_text(encoding="utf-8"))
    if not config.get("vector_q", False):
        raise ValueError(f"Not a vector-Q MORL run: {run_dir}")
    args = Namespace(**config["base_args"])
    args.device = "cpu"
    train.validate_no_test_path(args)
    episode, checkpoint = latest_checkpoint(run_dir)
    with tempfile.TemporaryDirectory(prefix="vector_morl_diagnosis_") as temp:
        env = train.build_environment(args, Path(temp))
        agent = train.build_agent(
            args,
            env,
            torch.device("cpu"),
            preference_dim=2,
            q_preference_dim=0,
            objective_dim=2,
            vector_value_normalization=config.get("vector_value_normalization", "legacy_loss_scale"),
        )
        train.load_checkpoint(agent, checkpoint, args, env)
        agent.Q.eval()
        agent.Q_target.eval()
        schedule = list(config["training_schedule"])
        log = pd.read_csv(run_dir / "action_reward_log.csv")
        records, counts = reconstruct_records(
            log, schedule, agent.n_actions, samples_per_preference, np.random.default_rng(args.seed + 1103)
        )
        by_preference = {}
        for record in records:
            by_preference.setdefault(record["preference"], []).append(record)
        seen_names = [pref_name(item) for item in config["seen_preferences"]]
        if set(by_preference) != set(seen_names):
            raise RuntimeError(f"Reconstructed preference groups differ from schedule: {sorted(by_preference)}")

        state = torch.as_tensor(env["node_features"], dtype=torch.float32)
        full_mask = torch.ones(agent.n_actions, dtype=torch.long)
        with torch.no_grad():
            _, online_embedding = agent.Q(None, state, full_mask)
            _, target_embedding = agent.Q_target(None, state, full_mask)

        response_rows = []
        probe = records[:max(1, min(n_fixed_states, len(records)))]
        for left, right in itertools.combinations(seen_names, 2):
            left_pref, right_pref = preference_vector(left), preference_vector(right)
            deltas, correlations, jaccards = [], [], []
            for record in probe:
                mask = torch.as_tensor(record["mask"], dtype=torch.long)
                with torch.no_grad():
                    heads, _ = agent.Q(online_embedding, state, mask)
                    left_q = agent.scalarize_q(heads, left_pref).cpu().numpy()
                    right_q = agent.scalarize_q(heads, right_pref).cpu().numpy()
                left_top = set(np.argsort(-left_q, kind="stable")[:topk])
                right_top = set(np.argsort(-right_q, kind="stable")[:topk])
                deltas.append(float(np.mean(np.abs(left_q - right_q))))
                correlations.append(float(spearmanr(left_q, right_q).statistic))
                jaccards.append(float(len(left_top & right_top) / len(left_top | right_top)))
            response_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode,
                "left_preference": left, "right_preference": right,
                "n_fixed_states": len(probe), "mean_abs_scalarized_q_delta": float(np.mean(deltas)),
                "rank_spearman_mean": float(np.mean(correlations)), f"top{topk}_jaccard_mean": float(np.mean(jaccards)),
                "test_labels_read": False,
            })

        td_rows, target_audit_rows = [], []
        for name in seen_names:
            rewards, targets, errors = [], [[], []], [[], []]
            terminal_residuals, nonterminal_bootstrap, nonterminal_rewards, nonterminal_targets = (
                [[], []], [[], []], [[], []], [[], []]
            )
            for record in by_preference[name]:
                prediction, target, reward, bootstrap, _prediction_normalized = td_for_record(
                    agent, state, online_embedding, target_embedding, record
                )
                rewards.append(record["reward"])
                for head in range(2):
                    targets[head].append(float(target[head]))
                    errors[head].append(float((target[head] - prediction[head]).detach().abs()))
                    if record["done"]:
                        terminal_residuals[head].append(
                            float((target[head] - reward[head]).detach().abs())
                        )
                    else:
                        nonterminal_bootstrap[head].append(float(bootstrap[head].abs()))
                        nonterminal_rewards[head].append(float(reward[head].abs()))
                        nonterminal_targets[head].append(float(target[head].abs()))
            rewards = np.asarray(rewards)
            for head, objective in enumerate(OBJECTIVES):
                td_rows.append({
                    "seed": int(args.seed), "checkpoint_episode": episode, "preference": name, "objective": objective,
                    "n_transitions": len(by_preference[name]),
                    **{f"reward_{key}": value for key, value in quantiles(rewards[:, head]).items()},
                    **{f"td_target_{key}": value for key, value in quantiles(targets[head]).items()},
                    **{f"abs_td_error_{key}": value for key, value in quantiles(errors[head]).items()},
                    "test_labels_read": False,
                })
                boot = np.asarray(nonterminal_bootstrap[head], dtype=np.float64)
                immediate = np.asarray(nonterminal_rewards[head], dtype=np.float64)
                target_abs = np.asarray(nonterminal_targets[head], dtype=np.float64)
                terminal = np.asarray(terminal_residuals[head], dtype=np.float64)
                target_audit_rows.append({
                    "seed": int(args.seed), "checkpoint_episode": episode,
                    "preference": name, "objective": objective,
                    "terminal_transition_count": int(len(terminal)),
                    "terminal_target_minus_reward_abs_max": float(terminal.max()) if len(terminal) else float("nan"),
                    "nonterminal_transition_count": int(len(boot)),
                    "nonterminal_abs_bootstrap_mean": float(boot.mean()) if len(boot) else float("nan"),
                    "nonterminal_abs_bootstrap_p90": float(np.quantile(boot, 0.9)) if len(boot) else float("nan"),
                    "nonterminal_abs_reward_mean": float(immediate.mean()) if len(immediate) else float("nan"),
                    "nonterminal_abs_target_mean": float(target_abs.mean()) if len(target_abs) else float("nan"),
                    "bootstrap_to_reward_mean_ratio": float(boot.mean() / immediate.mean()) if len(boot) and immediate.mean() > 0 else float("inf"),
                    "test_labels_read": False,
                })

        # Compare raw objective gradients on an exactly preference-balanced probe.
        head_gradients, gradient_rows = {}, []
        all_records = [record for name in seen_names for record in by_preference[name]]
        scale = torch.as_tensor(agent.objective_value_scale, dtype=torch.float32)
        for head, objective in enumerate(OBJECTIVES):
            agent.Q.zero_grad(set_to_none=True)
            losses = []
            for record in all_records:
                prediction, target, _reward, _bootstrap, prediction_normalized = td_for_record(
                    agent, state, online_embedding, target_embedding, record
                )
                if agent.vector_value_normalization == "popart":
                    loss_prediction = prediction_normalized[head]
                    loss_target = agent.normalize_objective_q(target)[head]
                else:
                    loss_prediction = prediction[head] / scale[head]
                    loss_target = target[head] / scale[head]
                losses.append(torch.nn.functional.smooth_l1_loss(loss_prediction, loss_target))
            loss = torch.stack(losses).mean()
            loss.backward()
            all_vector = gradient_vector(agent.Q, lambda _name: True)
            trunk_vector = gradient_vector(agent.Q, lambda name: not name.startswith("lin8"))
            head_gradients[objective] = {"all": all_vector, "trunk": trunk_vector}
            gradient_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode, "objective": objective,
                "n_balanced_transitions": len(all_records), "normalized_probe_loss": float(loss.item()),
                "all_parameter_gradient_norm": float(np.linalg.norm(all_vector)),
                "shared_trunk_gradient_norm": float(np.linalg.norm(trunk_vector)), "test_labels_read": False,
            })
        cosine_rows = [{
            "seed": int(args.seed), "checkpoint_episode": episode,
            "recovery_discovery_all_parameter_cosine": cosine(head_gradients["recovery"]["all"], head_gradients["discovery"]["all"]),
            "recovery_discovery_shared_trunk_cosine": cosine(head_gradients["recovery"]["trunk"], head_gradients["discovery"]["trunk"]),
            "test_labels_read": False,
        }]
        scale_rows = [{
            "seed": int(args.seed), "checkpoint_episode": episode,
            "recovery_q_scale": float(agent.objective_value_scale[0]),
            "discovery_q_scale": float(agent.objective_value_scale[1]),
            "scale_ratio_max_over_min": float(np.max(agent.objective_value_scale) / np.min(agent.objective_value_scale)),
            "test_labels_read": False,
        }]
        terminal = log.sort_values(["episode", "step"]).tail(agent.memory.mem_size).copy()
        terminal["preference"] = terminal["episode"].map(lambda ep: schedule[int(ep) - 1])
        replay_rows = [{
            "seed": int(args.seed), "checkpoint_episode": episode, "preference": name,
            "recorded_transition_count": int(counts[name]),
            "reconstructed_terminal_fifo_count": int((terminal["preference"] == name).sum()),
            "per_sampling_ratio_available": False, "per_mean_priority_available": False,
            "test_labels_read": False,
        } for name in seen_names]

    summary = {
        "seed": int(args.seed), "run_dir": str(run_dir), "checkpoint": str(checkpoint),
        "checkpoint_episode": episode, "objective_value_scale": [float(x) for x in agent.objective_value_scale],
        "recorded_transition_counts": counts, "replay_buffer_serialized": False,
        "per_sampling_ratio_available": False, "per_priority_available": False,
        "posthoc_probe_note": "TD and gradient probes use the final retained checkpoint plus logged transitions, not historical training-time gradients.",
        "test_labels_read": False,
    }
    return summary, response_rows, td_rows, target_audit_rows, gradient_rows, cosine_rows, scale_rows, replay_rows


def write_rows(path, rows, keys):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(row[key] for key in keys)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morl-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-preference", type=int, default=64)
    parser.add_argument("--fixed-states", type=int, default=8)
    parser.add_argument("--topk", type=int, default=150)
    args = parser.parse_args()
    if min(args.samples_per_preference, args.fixed_states, args.topk) <= 0:
        raise ValueError("Diagnostic sample sizes must be positive.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [[], [], [], [], [], [], []]
    summaries = []
    runs = resolve_runs(args.morl_runs)
    for index, run in enumerate(runs, start=1):
        config = json.loads((run / "morl_config.json").read_text(encoding="utf-8"))
        print(
            f"[vector-Q audit] {index}/{len(runs)} seed={config['base_args']['seed']}: "
            "loading checkpoint and probing TD targets...",
            flush=True,
        )
        result = diagnose_one(run, args.samples_per_preference, args.fixed_states, args.topk)
        summaries.append(result[0])
        for bucket, rows in zip(all_rows, result[1:]):
            bucket.extend(rows)
        print(f"[vector-Q audit] {index}/{len(runs)} seed={config['base_args']['seed']}: done.", flush=True)
    (output / "vector_diagnosis_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, rows, keys in (
        ("q_response.csv", all_rows[0], ["seed", "left_preference", "right_preference"]),
        ("td_probe_distributions.csv", all_rows[1], ["seed", "preference", "objective"]),
        ("td_target_bootstrap_audit.csv", all_rows[2], ["seed", "preference", "objective"]),
        ("objective_gradient_norms.csv", all_rows[3], ["seed", "objective"]),
        ("objective_gradient_cosine.csv", all_rows[4], ["seed"]),
        ("objective_scales.csv", all_rows[5], ["seed"]),
        ("reconstructed_replay_composition.csv", all_rows[6], ["seed", "preference"]),
    ):
        write_rows(output / filename, rows, keys)
    print(f"Wrote vector-Q diagnostics for {len(summaries)} run(s) to {output}")


if __name__ == "__main__":
    main()
