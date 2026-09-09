#!/usr/bin/env python3
"""Post-hoc, validation-only diagnosis of preference-conditioned MORL runs.

This script does not train or modify a model.  It reconstructs transitions
from action_reward_log.csv and the recorded training schedule, then probes the
latest retained checkpoint.  Replay priorities/sampling are explicitly marked
unavailable because completed runs intentionally do not serialize replay.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import train  # noqa: E402


def pref_name(values):
    return f"r{float(values[0]):.2f}_d{float(values[1]):.2f}"


def resolve_runs(patterns):
    items = []
    for pattern in patterns:
        items.extend(glob.glob(pattern) if any(char in pattern for char in "*?[") else [pattern])
    return sorted({Path(item) for item in items})


def latest_retained_checkpoint(run_dir):
    manifest = json.loads((run_dir / "pareto_manifest.json").read_text(encoding="utf-8"))
    episode = max(int(value) for value in manifest["retained_union_episodes"])
    checkpoint = run_dir / "pareto_checkpoints" / f"episode_{episode:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing retained checkpoint: {checkpoint}")
    return episode, checkpoint


def reconstruct_records(action_log, schedule, n_actions, samples_per_preference, rng):
    """Rebuild sampled (mask, next_mask, reward, done, w) records from logs."""
    action_log = action_log.copy()
    action_log["preference"] = action_log["episode"].map(
        lambda episode: schedule[int(episode) - 1]
    )
    wanted = set()
    counts = {}
    for preference, group in action_log.groupby("preference", sort=True):
        counts[preference] = int(len(group))
        amount = min(samples_per_preference, len(group))
        wanted.update(rng.choice(group.index.to_numpy(), size=amount, replace=False).tolist())

    records = []
    for _, group in action_log.groupby("episode", sort=True):
        mask = np.ones(n_actions, dtype=np.int64)
        for row_index, row in group.sort_values("step").iterrows():
            action = int(row["action_index"])
            if mask[action] != 1:
                raise ValueError(f"Invalid/repeated action while reconstructing episode {row['episode']}")
            next_mask = mask.copy()
            next_mask[action] = 0
            if row_index in wanted:
                records.append({
                    "preference": row["preference"],
                    "action": action,
                    "reward": float(row["final_reward"]),
                    "done": float(bool(row["done"])),
                    "mask": mask.copy(),
                    "next_mask": next_mask,
                })
            mask = next_mask
    return records, counts


def vector_from_grads(model, include):
    pieces = []
    for name, parameter in model.named_parameters():
        if include(name):
            value = parameter.grad
            pieces.append(
                torch.zeros_like(parameter).reshape(-1) if value is None else value.detach().reshape(-1)
            )
    return torch.cat(pieces).cpu().numpy()


def cosine(left, right):
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else float("nan")


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "p50": float(np.quantile(values, 0.5)), "p90": float(np.quantile(values, 0.9)),
    }


def diagnose_one(run_dir, samples_per_preference, n_probe_states, topk):
    config = json.loads((run_dir / "morl_config.json").read_text(encoding="utf-8"))
    args = Namespace(**config["base_args"])
    args.device = "cpu"
    train.validate_no_test_path(args)
    episode, checkpoint = latest_retained_checkpoint(run_dir)
    train.set_seed(args.seed)
    with tempfile.TemporaryDirectory(prefix="morl_condition_diagnosis_") as temp:
        env = train.build_environment(args, Path(temp))
        agent = train.build_agent(args, env, torch.device("cpu"), preference_dim=2)
        initial_preference = {
            name: parameter.detach().clone()
            for name, parameter in agent.Q.named_parameters()
            if name.startswith("preference_encoder")
        }
        train.load_checkpoint(agent, checkpoint, args, env)
        agent.Q.eval()
        agent.Q_target.eval()

        schedule = list(config["training_schedule"])
        log = pd.read_csv(run_dir / "action_reward_log.csv")
        records, transition_counts = reconstruct_records(
            log, schedule, agent.n_actions, samples_per_preference,
            np.random.default_rng(args.seed + 1009),
        )
        terminal_fifo = log.sort_values(["episode", "step"]).copy()
        terminal_fifo["preference"] = terminal_fifo["episode"].map(
            lambda item: schedule[int(item) - 1]
        )
        terminal_fifo = terminal_fifo.tail(agent.memory.mem_size)
        terminal_fifo_counts = terminal_fifo["preference"].value_counts().to_dict()
        replay_rows = [
            {
                "seed": int(args.seed), "checkpoint_episode": episode,
                "preference": preference,
                "recorded_transition_count": int(transition_counts[preference]),
                "reconstructed_terminal_fifo_count": int(terminal_fifo_counts.get(preference, 0)),
                "reconstructed_terminal_fifo_fraction": float(
                    terminal_fifo_counts.get(preference, 0) / agent.memory.mem_size
                ),
                "per_sampling_ratio_available": False,
                "per_mean_priority_available": False,
                "test_labels_read": False,
            }
            for preference in sorted(transition_counts)
        ]
        by_preference = {}
        for record in records:
            by_preference.setdefault(record["preference"], []).append(record)
        seen_names = [pref_name(values) for values in config["seen_preferences"]]
        if set(by_preference) != set(seen_names):
            raise RuntimeError(f"Unexpected reconstructed preferences: {sorted(by_preference)}")

        state = torch.as_tensor(env["node_features"], dtype=torch.float32)
        full_mask = torch.ones(agent.n_actions, dtype=torch.long)
        ref_pref = torch.as_tensor(config["seen_preferences"][0], dtype=torch.float32)
        with torch.no_grad():
            _, online_embedding = agent.Q(None, state, full_mask, preference=ref_pref)
            _, target_embedding = agent.Q_target(None, state, full_mask, preference=ref_pref)

        preference_values = {
            pref_name(values): torch.as_tensor(values, dtype=torch.float32)
            for values in config["seen_preferences"]
        }
        # Fixed state batch: masks are held fixed while only w changes.
        probe_records = records[: max(1, min(n_probe_states, len(records)))]
        response_rows = []
        for left_name, right_name in itertools.combinations(seen_names, 2):
            q_deltas, rank_corr, jaccards = [], [], []
            for record in probe_records:
                mask = torch.as_tensor(record["mask"], dtype=torch.long)
                with torch.no_grad():
                    left_q, _ = agent.Q(online_embedding, state, mask, preference=preference_values[left_name])
                    right_q, _ = agent.Q(online_embedding, state, mask, preference=preference_values[right_name])
                left = left_q.cpu().numpy()
                right = right_q.cpu().numpy()
                left_top = set(np.argsort(-left, kind="stable")[:topk])
                right_top = set(np.argsort(-right, kind="stable")[:topk])
                q_deltas.append(float(np.mean(np.abs(left - right))))
                rank_corr.append(float(spearmanr(left, right).statistic))
                jaccards.append(float(len(left_top & right_top) / len(left_top | right_top)))
            response_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode,
                "left_preference": left_name, "right_preference": right_name,
                "n_fixed_states": len(probe_records),
                "mean_abs_q_delta": float(np.mean(q_deltas)),
                "rank_spearman_mean": float(np.mean(rank_corr)),
                f"top{topk}_jaccard_mean": float(np.mean(jaccards)),
                "test_labels_read": False,
            })

        gradient_rows, gradient_vectors, td_rows = [], {}, []
        for name in seen_names:
            preference = preference_values[name]
            current = by_preference[name]
            losses, rewards, targets, errors = [], [], [], []
            agent.Q.zero_grad(set_to_none=True)
            for record in current:
                mask = torch.as_tensor(record["mask"], dtype=torch.long)
                next_mask = torch.as_tensor(record["next_mask"], dtype=torch.long)
                q_current, _ = agent.Q(online_embedding, state, mask, preference=preference)
                prediction = q_current[record["action"]]
                with torch.no_grad():
                    q_online_next, _ = agent.Q(online_embedding, state, next_mask, preference=preference)
                    next_action = int(torch.argmax(q_online_next.masked_fill(next_mask == 0, -1e9)).item())
                    q_target_next, _ = agent.Q_target(
                        target_embedding, state, next_mask, preference=preference
                    )
                    target = float(record["reward"]) + agent.gamma * (1.0 - record["done"]) * q_target_next[next_action]
                losses.append(torch.nn.functional.smooth_l1_loss(prediction, target))
                rewards.append(float(record["reward"]))
                targets.append(float(target.item()))
                errors.append(float((target - prediction).detach().abs().item()))
            loss = torch.stack(losses).mean()
            loss.backward()
            encoder_vector = vector_from_grads(agent.Q, lambda parameter_name: parameter_name.startswith("preference_encoder"))
            shared_vector = vector_from_grads(agent.Q, lambda parameter_name: True)
            gradient_vectors[name] = {"encoder": encoder_vector, "all": shared_vector}
            gradient_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode, "preference": name,
                "n_transitions": len(current), "probe_td_loss": float(loss.item()),
                "preference_encoder_gradient_norm": float(np.linalg.norm(encoder_vector)),
                "all_parameter_gradient_norm": float(np.linalg.norm(shared_vector)),
                "test_labels_read": False,
            })
            td_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode, "preference": name,
                "n_transitions": len(current),
                **{f"reward_{key}": value for key, value in quantiles(rewards).items()},
                **{f"td_target_{key}": value for key, value in quantiles(targets).items()},
                **{f"abs_td_error_{key}": value for key, value in quantiles(errors).items()},
                "test_labels_read": False,
            })

        cosine_rows = []
        for left_name, right_name in itertools.combinations(seen_names, 2):
            cosine_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode,
                "left_preference": left_name, "right_preference": right_name,
                "encoder_gradient_cosine": cosine(
                    gradient_vectors[left_name]["encoder"], gradient_vectors[right_name]["encoder"]
                ),
                "all_parameter_gradient_cosine": cosine(
                    gradient_vectors[left_name]["all"], gradient_vectors[right_name]["all"]
                ),
                "test_labels_read": False,
            })

        update_rows = []
        optimizer_state = agent.Q.optimizer.state
        for name, parameter in agent.Q.named_parameters():
            if not name.startswith("preference_encoder"):
                continue
            initial = initial_preference[name]
            state_entry = optimizer_state.get(parameter, {})
            exp_avg = state_entry.get("exp_avg")
            update_rows.append({
                "seed": int(args.seed), "checkpoint_episode": episode, "parameter": name,
                "initial_norm": float(initial.norm().item()),
                "final_norm": float(parameter.detach().norm().item()),
                "update_l2_norm": float((parameter.detach() - initial).norm().item()),
                "adam_exp_avg_norm": float(exp_avg.norm().item()) if exp_avg is not None else float("nan"),
                "test_labels_read": False,
            })

    run_summary = {
        "seed": int(args.seed), "run_dir": str(run_dir), "checkpoint": str(checkpoint),
        "checkpoint_episode": episode,
        "recorded_transition_counts": transition_counts,
        "reconstructed_terminal_fifo_counts": terminal_fifo_counts,
        "replay_buffer_serialized": False,
        "per_sampling_ratio_available": False,
        "per_priority_available": False,
        "posthoc_probe_note": "TD targets/errors/gradients use final retained checkpoint and logged transitions; they are not historical training-time values.",
        "test_labels_read": False,
    }
    return run_summary, response_rows, gradient_rows, cosine_rows, td_rows, update_rows, replay_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morl-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-preference", type=int, default=64)
    parser.add_argument("--fixed-states", type=int, default=8)
    parser.add_argument("--topk", type=int, default=150)
    args = parser.parse_args()
    if min(args.samples_per_preference, args.fixed_states, args.topk) <= 0:
        raise ValueError("All diagnostic sample sizes must be positive.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, responses, gradients, cosines, td, updates, replay = [], [], [], [], [], [], []
    for run_dir in resolve_runs(args.morl_runs):
        result = diagnose_one(run_dir, args.samples_per_preference, args.fixed_states, args.topk)
        summaries.append(result[0])
        responses.extend(result[1])
        gradients.extend(result[2])
        cosines.extend(result[3])
        td.extend(result[4])
        updates.extend(result[5])
        replay.extend(result[6])
    summary_path = output / "diagnosis_summary.json"
    previous_summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    merged_summaries = {int(item["seed"]): item for item in previous_summaries}
    merged_summaries.update({int(item["seed"]): item for item in summaries})
    summary_path.write_text(
        json.dumps([merged_summaries[key] for key in sorted(merged_summaries)], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, rows, keys in (
        ("q_response.csv", responses, ["seed", "left_preference", "right_preference"]),
        ("gradient_norms.csv", gradients, ["seed", "preference"]),
        ("gradient_cosine.csv", cosines, ["seed", "left_preference", "right_preference"]),
        ("td_probe_distributions.csv", td, ["seed", "preference"]),
        ("preference_encoder_updates.csv", updates, ["seed", "parameter"]),
        ("reconstructed_replay_composition.csv", replay, ["seed", "preference"]),
    ):
        path = output / name
        new_frame = pd.DataFrame(rows)
        old_frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new_frame.columns)
        merged = pd.concat([old_frame, new_frame], ignore_index=True)
        merged = merged.drop_duplicates(keys, keep="last").sort_values(keys)
        merged.to_csv(path, index=False)
    print(f"Wrote conditioning diagnostics for {len(summaries)} runs to {output}")


if __name__ == "__main__":
    main()
