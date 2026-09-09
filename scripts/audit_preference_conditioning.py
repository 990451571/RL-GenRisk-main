#!/usr/bin/env python3
"""Executable audit for the preference-conditioned DDQN path.

It first verifies the complete synthetic path
Q input -> replay -> sample -> online Q -> target Q -> TD update, then can
probe a saved MORL checkpoint with one fixed real state and several preferences.
No labels from the Test split are read and the script writes no result files.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from DQN import DeepQNetwork  # noqa: E402
import train  # noqa: E402


PREFERENCES = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)


def _force_cpu(agent):
    device = torch.device("cpu")
    agent.Q.to(device)
    agent.Q_target.to(device)
    agent.Q.device = device
    agent.Q_target.device = device


def _make_synthetic_agent():
    n_actions, feature_dim = 8, 3
    agent = DeepQNetwork(
        n_actions=n_actions,
        net_ori=np.eye(n_actions, dtype=np.float32),
        fea_ori=np.arange(n_actions * feature_dim, dtype=np.float32).reshape(n_actions, feature_dim),
        embedding_size=4,
        train_patient_data=np.empty((0, 0), dtype=np.float32),
        test_patient_data=np.empty((0, 0), dtype=np.float32),
        gene_sta=set(),
        weights={},
        score_alpha=0.5,
        memory_size=8,
        batch_size=4,
        selection_budget=4,
        preference_dim=2,
    )
    _force_cpu(agent)
    return agent


def synthetic_chain_audit():
    """Assert that sampled w reaches all three DDQN evaluations and gradients."""
    np.random.seed(7)
    torch.manual_seed(7)
    agent = _make_synthetic_agent()
    state = agent.fea_ori.astype(np.float32)
    current_mask = np.ones(agent.n_actions, dtype=np.float32)
    for index, preference in enumerate(PREFERENCES):
        next_mask = current_mask.copy()
        next_mask[index] = 0.0
        agent.remember(
            state, index, 1.0 + index, current_mask, next_mask, False, preference=preference
        )
    # Add a fourth transition so a full batch is sampled without replacement.
    next_mask = current_mask.copy()
    next_mask[3] = 0.0
    agent.remember(state, 3, 1.0, current_mask, next_mask, True, preference=PREFERENCES[1])

    sampled = {}
    original_sample = agent.memory.sample_buffer

    def sample_with_capture(batch_size):
        batch = original_sample(batch_size)
        sampled["preference"] = batch[-1].copy()
        return batch

    agent.memory.sample_buffer = sample_with_capture
    seen = {"online": [], "target": []}

    def capture_forward(module, name):
        original_forward = module.forward

        def wrapped(*args, **kwargs):
            preference = kwargs.get("preference")
            seen[name].append(None if preference is None else preference.detach().cpu().numpy().copy())
            return original_forward(*args, **kwargs)

        module.forward = wrapped

    capture_forward(agent.Q, "online")
    capture_forward(agent.Q_target, "target")
    agent.learn()

    expected = sampled["preference"]
    online_ok = len(seen["online"]) == 2 and all(
        item.shape == expected.shape and np.array_equal(item, expected) for item in seen["online"]
    )
    target_ok = len(seen["target"]) == 1 and np.array_equal(seen["target"][0], expected)
    online_td_calls = len(seen["online"])
    target_td_calls = len(seen["target"])
    grad = agent.Q.preference_encoder.weight.grad
    gradient_norm = float(grad.norm().item()) if grad is not None else 0.0
    if not online_ok or not target_ok or gradient_norm <= 0.0:
        raise AssertionError(
            f"Preference chain failed: online_ok={online_ok}, target_ok={target_ok}, "
            f"preference_gradient_norm={gradient_norm}."
        )

    agent.Q.eval()
    state_t = torch.as_tensor(state, dtype=torch.float32)
    mask_t = torch.ones(agent.n_actions, dtype=torch.long)
    with torch.no_grad():
        q_values = [
            agent.Q(None, state_t, mask_t, preference=torch.as_tensor(pref))[0].numpy()
            for pref in PREFERENCES
        ]
    deltas = [float(np.max(np.abs(q_values[i] - q_values[0]))) for i in range(1, len(q_values))]
    if not any(delta > 1e-8 for delta in deltas):
        raise AssertionError("Different preferences produced identical Q values on a fixed state.")
    return {
        "synthetic_chain_passed": True,
        "sampled_preference_shape": list(expected.shape),
        "online_q_calls_with_sampled_w": online_td_calls,
        "target_q_calls_with_sampled_w": target_td_calls,
        "preference_encoder_gradient_norm": gradient_norm,
        "fixed_state_q_max_abs_delta_vs_w10": deltas,
    }


def checkpoint_probe(run_dir, checkpoint_name, topk):
    """Measure actual checkpoint sensitivity to w at a single fixed state."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "morl_config.json").read_text(encoding="utf-8"))
    args = Namespace(**config["base_args"])
    args.device = "cpu"  # diagnostic reproducibility; no model update occurs.
    train.validate_no_test_path(args)
    with tempfile.TemporaryDirectory(prefix="morl_condition_audit_") as tmp:
        tmp_dir = Path(tmp)
        env = train.build_environment(args, tmp_dir)
        agent = train.build_agent(args, env, torch.device("cpu"), preference_dim=2)
        checkpoint = run_dir / "pareto_checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        train.load_checkpoint(agent, checkpoint, args, env)
        agent.Q.eval()
        state = torch.as_tensor(env["node_features"], dtype=torch.float32)
        mask = torch.ones(agent.n_actions, dtype=torch.long)
        q_by_pref = {}
        with torch.no_grad():
            for pref in PREFERENCES:
                key = f"r{pref[0]:.2f}_d{pref[1]:.2f}"
                q_by_pref[key] = agent.Q(
                    None, state, mask, preference=torch.as_tensor(pref)
                )[0].cpu().numpy()
    reference = q_by_pref["r1.00_d0.00"]
    summary = {}
    reference_top = np.argsort(-reference, kind="stable")[:topk]
    for key, values in q_by_pref.items():
        rank = np.argsort(-values, kind="stable")[:topk]
        summary[key] = {
            "top1_action": int(rank[0]),
            "max_abs_q_delta_vs_w10": float(np.max(np.abs(values - reference))),
            "mean_abs_q_delta_vs_w10": float(np.mean(np.abs(values - reference))),
            "topk_overlap_vs_w10": float(len(set(rank) & set(reference_top)) / topk),
        }
    return {
        "checkpoint": str(checkpoint),
        "topk": int(topk),
        "fixed_state_response": summary,
        "test_labels_read": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morl-run", help="MORL run directory for an actual checkpoint probe.")
    parser.add_argument("--checkpoint", default="episode_010.pt")
    parser.add_argument("--topk", type=int, default=150)
    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    result = {"chain": synthetic_chain_audit()}
    if args.morl_run:
        result["checkpoint_probe"] = checkpoint_probe(args.morl_run, args.checkpoint, args.topk)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
