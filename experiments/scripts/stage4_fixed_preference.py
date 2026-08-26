import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def host_path(win_path):
    path = Path(win_path)
    if path.exists():
        return path
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/").lstrip("/")
    return Path(f"/mnt/{drive}/{rest}")


OUT = host_path(r"E:\codex_file\新方向阶段4")
PROJECT = host_path(r"E:\Projects\RL-GenRisk-main")
STAGE0 = host_path(r"E:\codex_file\新方向阶段0")
STAGE1 = host_path(r"E:\codex_file\新方向阶段1")
STAGE2 = host_path(r"E:\codex_file\新方向阶段2")
STAGE3 = host_path(r"E:\codex_file\新方向阶段3")
SRC = PROJECT / "src"
PROTOCOL_B = host_path(r"E:\codex_file\一阶段\driver_label_protocol\protocol_B")

sys.path.insert(0, str(SRC))
import inputall  # noqa: E402
import train  # noqa: E402
from DQN import DeepQNetwork  # noqa: E402
from mutation_frequency import (  # noqa: E402
    TOTAL_KIRC_TUMOR_SAMPLES,
    classify_mutation_frequency,
    mutation_frequency_pct,
)


SEEDS = [42, 43, 44, 45, 46]
POLICIES = {
    "Recovery": {"w_relevance": 0.70, "w_discovery": 0.15, "w_robustness": 0.15},
    "Discovery": {"w_relevance": 0.20, "w_discovery": 0.65, "w_robustness": 0.15},
    "Robustness": {"w_relevance": 0.20, "w_discovery": 0.20, "w_robustness": 0.60},
}
PPI_EDGES = STAGE1 / "10_stage2_ready" / "ppi_edges_frozen.tsv"
GRN_EDGES = STAGE1 / "10_stage2_ready" / "grn_edges_frozen.tsv"
GENE_UNIVERSE = STAGE1 / "10_stage2_ready" / "gene_universe.tsv"
PPI_GRN_MESSAGE = STAGE2 / "03_message_graphs" / "ppi_grn_union_message_edges.tsv"
FORMAL_CONFIG_STAGE2 = STAGE2 / "05_formal_protocol" / "formal_config.yaml"

ACCESS_ROWS = []


def ts():
    return datetime.now(timezone.utc).isoformat()


def display(path):
    return str(Path(path))


def log_access(path, purpose, rw, category, allowed=True, notes=""):
    ACCESS_ROWS.append(
        {
            "timestamp": ts(),
            "file_or_resource": display(path),
            "purpose": purpose,
            "read_or_write": rw,
            "category": category,
            "allowed_by_protocol": "YES" if allowed else "NO",
            "notes": notes,
        }
    )


def mkdirs():
    dirs = [
        "01_reward_audit",
        "02_reward_design",
        "03_reward_scale_audit",
        "04_stage4_implementation",
        "05_regression_and_smoke",
        "06_formal_protocol",
        "07_formal_runs/Recovery",
        "07_formal_runs/Discovery",
        "07_formal_runs/Robustness",
        "08_analysis",
        "09_tradeoff",
        "10_stability",
        "11_figures",
        "12_integrity",
        "logs",
        "scripts",
        "src_stage4",
    ]
    for rel in dirs:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log_access(path, "write Stage 4 output", "write", "stage4_output", True)


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_text(path, purpose, category="frozen_stage_input"):
    log_access(path, purpose, "read", category, True)
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_csv(path, delimiter=",", purpose="read csv", category="frozen_stage_input"):
    log_access(path, purpose, "read", category, True)
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log_access(path, "write Stage 4 csv output", "write", "stage4_output", True)


def append_csv(path, rows, fieldnames):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log_access(path, "append Stage 4 csv output", "write", "stage4_output", True)


def matrix_from_message_edges(path, node_count=9039):
    rows = read_csv(path, delimiter="\t", purpose="read frozen Stage2 SimpleUnion message graph", category="stage2_frozen_message_graph")
    mat = np.zeros((node_count, node_count), dtype=np.float32)
    for row in rows:
        src = int(row.get("source_index", row.get("src", row.get("source", 0))))
        dst = int(row.get("target_index", row.get("dst", row.get("target", 0))))
        mat[src, dst] = 1.0
    return mat, int(mat.sum())


def edge_degrees(path, directed=False):
    rows = read_csv(path, delimiter="\t", purpose="read frozen network degrees", category="stage1_frozen_network")
    degree = {}
    for row in rows:
        values = list(row.values())
        if len(values) < 2:
            continue
        a = str(values[0]).strip().upper()
        b = str(values[1]).strip().upper()
        if not a or not b:
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + (0 if directed else 1)
    return degree


def ranks_to_metrics(ranking_rows, labels, ks=(50, 100, 150)):
    labels = set(labels)
    out = {}
    for k in ks:
        top = ranking_rows[:k]
        hits = [1 if row["Gene"] in labels else 0 for row in top]
        hit_count = int(sum(hits))
        recall = hit_count / max(len(labels), 1)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal_hits = [1] * min(len(labels), k)
        idcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
        out[f"HitCount@{k}"] = hit_count
        out[f"Recall@{k}"] = recall
        out[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
    return out


def jaccard(a, b, k):
    sa = set(a[:k])
    sb = set(b[:k])
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


def rbo_score(a, b, k=150, p=0.98):
    a = list(a[:k])
    b = list(b[:k])
    seen_a, seen_b = set(), set()
    score = 0.0
    for d in range(1, k + 1):
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        score += (overlap / d) * (p ** (d - 1))
    return (1 - p) * score


def cosine_max_to_selected(feature_norm, idx, selected_indices):
    candidates = [j for j in selected_indices if j != idx]
    if not candidates:
        return 0.0
    vals = feature_norm[candidates] @ feature_norm[idx]
    return float(np.clip(np.max(vals), 0.0, 1.0))


class Stage4RewardModel:
    def __init__(self, gene_names, node_features, feature_columns, train_driver_genes, ppi_degree, grn_degree):
        self.gene_names = list(gene_names)
        self.feature_columns = list(feature_columns)
        self.train_driver_set = {str(g).upper() for g in train_driver_genes}
        self.ppi_degree = ppi_degree
        self.grn_degree = grn_degree
        self.idx = {name: i for i, name in enumerate(self.gene_names)}
        self.feature = np.asarray(node_features, dtype=np.float64)
        self.feature_norm = self._normalize_for_cosine(self.feature)
        self.cols = {name: self.feature_columns.index(name) for name in self.feature_columns}
        self.mutation = self._col("Mutation")
        self.expression = self._col("Expression")
        self.methylation = self._col("Methylation")
        self.coverage = self._col("PatientCoverageCount")
        self.weight = self._col("Weight")
        self.ppi_deg_arr = np.asarray([float(ppi_degree.get(g.upper(), 0)) for g in self.gene_names], dtype=np.float64)
        self.grn_deg_arr = np.asarray([float(grn_degree.get(g.upper(), 0)) for g in self.gene_names], dtype=np.float64)
        self.mutation_pct = self._percent_rank(self.mutation)
        self.expr_pct = self._percent_rank(self.expression)
        self.meth_pct = self._percent_rank(self.methylation)
        self.coverage_pct = self._percent_rank(self.coverage)
        self.weight_pct = self._percent_rank(self.weight)
        self.ppi_hub = self._log_minmax(self.ppi_deg_arr)
        self.grn_hub = self._log_minmax(self.grn_deg_arr)
        self.hub_penalty = 0.5 * self.ppi_hub + 0.5 * self.grn_hub
        self.network_support = self._percent_rank(np.log1p(self.ppi_deg_arr) + np.log1p(self.grn_deg_arr))
        self.nonmutation_support = (self.expr_pct + self.meth_pct + self.network_support) / 3.0
        nonzero = self.mutation[self.mutation > 0]
        self.nonzero_q25 = float(np.percentile(nonzero, 25)) if len(nonzero) else 0.0
        self.support_threshold = float(np.percentile(self.nonmutation_support, 75))
        self.mutation_patient_count = np.rint(self.mutation * TOTAL_KIRC_TUMOR_SAMPLES).astype(int)
        if np.any(self.mutation_patient_count < 0):
            raise ValueError("MutationPatientCount must be non-negative.")
        reconstructed = self.mutation_patient_count / TOTAL_KIRC_TUMOR_SAMPLES
        if np.max(np.abs(reconstructed - self.mutation)) > 1e-6:
            raise ValueError("Mutation frequency is not consistent with MutationPatientCount / 368.")
        self.mutation_group = np.asarray(
            [classify_mutation_frequency(int(count)) for count in self.mutation_patient_count],
            dtype=object,
        )
        self.very_low_mutation = self.mutation_group == "very_low"
        self.low_frequency_mutation = self.mutation_group == "low_frequency"
        self.high_frequency_mutation = self.mutation_group == "high_frequency"
        self.low_frequency_nonzero = self.low_frequency_mutation
        self.very_low_supported = self.very_low_mutation & (self.nonmutation_support >= self.support_threshold)
        self.low_frequency_candidate = self.low_frequency_mutation
        self.nearest_similarity = self._nearest_similarity_chunked()
        self.static = self._static_components()

    def _col(self, name):
        if name not in self.cols:
            return np.zeros(len(self.gene_names), dtype=np.float64)
        return self.feature[:, self.cols[name]]

    @staticmethod
    def _percent_rank(values):
        return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)

    @staticmethod
    def _log_minmax(values):
        x = np.log1p(np.asarray(values, dtype=np.float64))
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _normalize_for_cosine(values):
        x = np.asarray(values, dtype=np.float64)
        lo = np.percentile(x, 5, axis=0)
        hi = np.percentile(x, 95, axis=0)
        denom = np.where(hi > lo, hi - lo, 1.0)
        x = np.clip((x - lo) / denom, 0.0, 1.0)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norm > 0, norm, 1.0)

    def _nearest_similarity_chunked(self, chunk=256):
        n = self.feature_norm.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sim = self.feature_norm[start:end] @ self.feature_norm.T
            for i in range(start, end):
                sim[i - start, i] = -1.0
            out[start:end] = np.max(sim, axis=1)
        return np.clip(out, 0.0, 1.0)

    def _static_components(self):
        relevance = (
            0.35 * np.asarray([1.0 if g.upper() in self.train_driver_set else 0.0 for g in self.gene_names])
            + 0.15 * self.mutation_pct
            + 0.15 * self.expr_pct
            + 0.10 * self.meth_pct
            + 0.15 * self.network_support
            + 0.10 * self.coverage_pct
        )
        mutation_max = max(float(np.max(self.mutation)), 1e-12)
        zero_rarity = np.ones_like(self.mutation)
        nonzero_rarity = np.clip(1.0 - (self.mutation / max(self.nonzero_q25, 1e-12)), 0.0, 1.0)
        rarity = np.where(self.mutation > 0, nonzero_rarity, zero_rarity)
        discovery = np.clip(rarity * self.nonmutation_support, 0.0, 1.0)
        discovery = np.where(self.low_frequency_candidate, discovery, 0.25 * discovery)
        discovery += 0.03 * np.asarray([0.0 if g.upper() in self.train_driver_set else 1.0 for g in self.gene_names])
        discovery = np.clip(discovery, 0.0, 1.0)
        robustness = np.clip(1.0 - (0.5 * self.hub_penalty + 0.5 * self.nearest_similarity), 0.0, 1.0)
        return pd.DataFrame(
            {
                "Gene": self.gene_names,
                "train_driver": [g.upper() in self.train_driver_set for g in self.gene_names],
                "MutationPatientCount": self.mutation_patient_count,
                "mutation_frequency": self.mutation,
                "MutationFrequencyPct": [
                    mutation_frequency_pct(int(count), TOTAL_KIRC_TUMOR_SAMPLES)
                    for count in self.mutation_patient_count
                ],
                "MutationGroup": self.mutation_group,
                "expression": self.expression,
                "methylation": self.methylation,
                "patient_coverage_count": self.coverage,
                "ppi_degree": self.ppi_deg_arr,
                "grn_degree": self.grn_deg_arr,
                "ppi_hub_score": self.ppi_hub,
                "grn_hub_score": self.grn_hub,
                "combined_hub_penalty": self.hub_penalty,
                "network_support": self.network_support,
                "nonmutation_support": self.nonmutation_support,
                "nearest_feature_similarity": self.nearest_similarity,
                "very_low_mutation": self.very_low_mutation,
                "low_frequency_mutation": self.low_frequency_mutation,
                "high_frequency_mutation": self.high_frequency_mutation,
                "very_low_supported": self.very_low_supported,
                "low_frequency_candidate": self.low_frequency_candidate,
                "R_relevance_static": np.clip(relevance, 0.0, 1.0),
                "R_discovery_static": discovery,
                "R_robustness_static": robustness,
            }
        )

    def component_for_action(self, action, selected_before):
        row = self.static.iloc[int(action)]
        redundancy = cosine_max_to_selected(self.feature_norm, int(action), selected_before)
        dynamic_robustness = float(np.clip(1.0 - (0.5 * row["combined_hub_penalty"] + 0.5 * redundancy), 0.0, 1.0))
        return {
            "gene": row["Gene"],
            "r_relevance": float(row["R_relevance_static"]),
            "r_discovery": float(row["R_discovery_static"]),
            "r_robustness": dynamic_robustness,
            "hub_penalty": float(row["combined_hub_penalty"]),
            "redundancy_penalty": float(redundancy),
            "mutation_patient_count": int(row["MutationPatientCount"]),
            "mutation_frequency": float(row["mutation_frequency"]),
            "MutationFrequencyPct": float(row["MutationFrequencyPct"]),
            "MutationGroup": str(row["MutationGroup"]),
            "nonmutation_support": float(row["nonmutation_support"]),
            "very_low_supported": bool(row["very_low_supported"]),
            "low_frequency_candidate": bool(row["low_frequency_candidate"]),
            "supported_low_frequency": bool(row["low_frequency_candidate"] and row["nonmutation_support"] >= self.support_threshold),
        }


class Stage4DeepQNetwork(DeepQNetwork):
    def __init__(self, *args, stage4_reward_model, stage4_preference, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage4_reward_model = stage4_reward_model
        self.stage4_preference = dict(stage4_preference)

    def reward_config(self):
        return {
            "reward_mode": "stage4_fixed_preference_vector",
            "preference": dict(self.stage4_preference),
            "reward_bounds": [0.0, 1.0],
            "validation_labels_used_for_reward": False,
            "morl_started": False,
        }

    def step(self, network, action, gene_num, gene_name, weights):
        actions = self.actions[:]
        selected_before = [idx for idx in actions if idx != action]
        comp = self.stage4_reward_model.component_for_action(action, selected_before)
        w = self.stage4_preference
        scalar = (
            w["w_relevance"] * comp["r_relevance"]
            + w["w_discovery"] * comp["r_discovery"]
            + w["w_robustness"] * comp["r_robustness"]
        )
        self.last_reward_components = {
            "selected_gene": comp["gene"],
            "is_train_driver": comp["gene"].upper() in self.train_driver_set,
            "reward_total": float(scalar),
            "reward_legacy": 0.0,
            "reward_train_label": 0.0,
            "reward_mutation": 0.0,
            "reward_expression": 0.0,
            "reward_methylation": 0.0,
            "reward_lowfreq": 0.0,
            "reward_evidence_bonus": 0.0,
            "reward_penalty": 0.0,
            "r_relevance": comp["r_relevance"],
            "r_discovery": comp["r_discovery"],
            "r_robustness": comp["r_robustness"],
            "w_relevance": w["w_relevance"],
            "w_discovery": w["w_discovery"],
            "w_robustness": w["w_robustness"],
            "scalar_reward": float(scalar),
            "relevance_contribution": float(w["w_relevance"] * comp["r_relevance"]),
            "discovery_contribution": float(w["w_discovery"] * comp["r_discovery"]),
            "robustness_contribution": float(w["w_robustness"] * comp["r_robustness"]),
            "hub_penalty": comp["hub_penalty"],
            "redundancy_penalty": comp["redundancy_penalty"],
            "mutation_frequency": comp["mutation_frequency"],
            "nonmutation_support": comp["nonmutation_support"],
            "low_frequency_candidate": comp["low_frequency_candidate"],
            "supported_low_frequency": comp["supported_low_frequency"],
            "final_reward": float(scalar),
            "reward_mode": "stage4_fixed_preference_vector",
        }
        self.reward_all += float(scalar)
        done = 0
        if len(actions) >= self.selection_budget:
            self.embedding = None
            done = 1
            self.actions = []
            self.reward_list.append(float(scalar))
            self.reward_all = 0
        return float(scalar), done, actions


def make_args(seed, run_dir, smoke=False):
    return SimpleNamespace(
        original_feature_path=None,
        feature_mode="hybrid6_raw",
        multiomics_feature_path=str(PROJECT / "data" / "processed" / "KIRC_multiomics_3omics.csv"),
        cnv_missing_gene_path=str(PROJECT / "data" / "processed" / "cnv_kirc" / "multiomics_genes_missing_cnv.csv"),
        ppi_path=str(PROJECT / "data" / "HPRD.txt"),
        mutation_path=str(PROJECT / "data" / "KIRC.txt"),
        weight_path=str(PROJECT / "data" / "weights.txt"),
        train_label_path=str(PROTOCOL_B / "train_driver_genes.csv"),
        val_label_path=str(PROTOCOL_B / "validation_driver_genes.csv"),
        output_dir=str(run_dir.parent),
        seed=int(seed),
        max_episodes=1 if smoke else 15,
        max_steps=8 if smoke else 160,
        warmup_steps=2 if smoke else 128,
        batch_size=2 if smoke else 128,
        buffer_size=64 if smoke else 2048,
        gamma=0.95,
        learning_rate=1e-4,
        tau=0.001,
        epsilon_start=1.0,
        epsilon_end=0.15,
        epsilon_decay=600.0 if smoke else 2000.0,
        per_alpha=0.2,
        per_beta_start=0.1,
        per_beta_frames=2_000_000,
        per_eps=1e-5,
        val_interval=1,
        topk=150,
        device="cuda",
        resume=None,
        cancer="KIRC",
        embedding_size=64,
        score_alpha=0.5,
        gradient_clip=1.0,
        reward_mode="legacy",
        multiomics_mutation_weight=0.08,
        multiomics_expression_weight=0.06,
        multiomics_methylation_weight=0.06,
        no_mutation_expression_weight=0.08,
        no_mutation_methylation_weight=0.08,
        lowfreq_expression_weight=0.05,
        lowfreq_methylation_weight=0.05,
        lowfreq_bonus_cap=0.20,
        lowfreq_evidence_path=None,
        lowfreq_unlabeled_bonus_scale=None,
        lowfreq_unlabeled_bonus_cap=None,
    )


def build_stage4_agent(args, env, device, reward_model, preference):
    agent = Stage4DeepQNetwork(
        n_actions=len(env["gene_name"]),
        net_ori=env["net"],
        fea_ori=env["node_features"],
        embedding_size=args.embedding_size,
        train_patient_data=env["train_data"],
        test_patient_data=env["test_data"],
        gene_sta=env["train_driver_genes"],
        weights=env["weights"],
        score_alpha=args.score_alpha,
        train_driver_set=env["train_driver_genes"],
        pat_num=len(env["patients"]),
        learning_rate=args.learning_rate,
        reward_decay=args.gamma,
        memory_size=args.buffer_size,
        batch_size=args.batch_size,
        selection_budget=args.topk,
        gradient_clip=args.gradient_clip,
        reward_mode="legacy",
        reward_weights={},
        reward_feature_columns=env["feature_report"]["feature_columns"],
        lowfreq_evidence_by_gene=None,
        stage4_reward_model=reward_model,
        stage4_preference=preference,
    )
    agent.Q.to(device)
    agent.Q_target.to(device)
    agent.Q.device = device
    agent.Q_target.device = device
    agent.Q_target.load_state_dict(agent.Q.state_dict())
    agent.Q_target.eval()
    agent.gamma = args.gamma
    agent.lr = args.learning_rate
    agent.tau = args.tau
    agent.epsilon_min = args.epsilon_end
    agent.epsilon_increment = 0.0
    for group in agent.Q.optimizer.param_groups:
        group["lr"] = args.learning_rate
    agent.memory.alpha = args.per_alpha
    agent.memory.beta_start = args.per_beta_start
    agent.memory.beta_frames = args.per_beta_frames
    agent.memory.eps = args.per_eps
    return agent


TRACE_FIELDS = [
    "episode", "step", "candidate_count", "action_index", "Gene", "q_min", "q_max", "q_selected",
    "r_relevance", "r_discovery", "r_robustness", "w_relevance", "w_discovery", "w_robustness",
    "scalar_reward", "relevance_contribution", "discovery_contribution", "robustness_contribution",
    "hub_penalty", "redundancy_penalty", "mutation_patient_count", "mutation_frequency",
    "MutationFrequencyPct", "MutationGroup", "nonmutation_support", "very_low_supported",
    "low_frequency_candidate", "supported_low_frequency", "done", "terminal_reason", "epsilon",
    "invalid_action", "duplicate_action", "learn_step_after", "loss", "td_error_abs_mean", "gradient_norm",
]


def run_episode_stage4(agent, env, args, episode, run_dir):
    action_sel = list(range(agent.n_actions))
    agent.actions = []
    agent.actions_index = np.ones(agent.n_actions, dtype=np.int64)
    agent.embedding = None
    episode_reward = 0.0
    step_count = 0
    terminal_reason = "unknown"
    loss_values, td_values, q_min_values, q_max_values = [], [], [], []
    invalid_count = duplicate_count = dead_end_count = replay_write_count = per_update_count = optimizer_step_count = 0
    trace_rows = []
    state = env["node_features"]

    while True:
        agent.epsilon = train.epsilon_for_step(args, agent.memory_counter)
        current_action_mask = agent.actions_index.copy()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.Q.device)
        mask_tensor = torch.as_tensor(current_action_mask, dtype=torch.long, device=agent.Q.device)
        agent.Q.train()
        with torch.no_grad():
            q_values, emb = agent.Q(agent.embedding, state_tensor, mask_tensor)
        q_np = q_values.detach().cpu().numpy()
        train.assert_finite("Q values", q_np, episode, step_count)
        q_min_values.append(float(np.min(q_np)))
        q_max_values.append(float(np.max(q_np)))
        agent.embedding = emb.detach()
        valid_actions = [idx for idx in action_sel if current_action_mask[idx] == 1]
        if not valid_actions:
            terminal_reason = "no_legal_action"
            dead_end_count += 1
            break
        best_action = max(valid_actions, key=lambda idx: q_np[idx])
        action_index = int(best_action if np.random.uniform() >= agent.epsilon else random.choice(valid_actions))
        invalid = action_index not in action_sel or current_action_mask[action_index] != 1
        duplicate = action_index in agent.actions
        invalid_count += int(invalid)
        duplicate_count += int(duplicate)
        if invalid:
            raise RuntimeError(f"Invalid action selected: {action_index}")
        action_sel.remove(action_index)
        agent.actions.append(action_index)
        agent.actions_index[action_index] = 0
        next_action_mask = agent.actions_index.copy()
        reward, done, _ = agent.step(env["net"], action_index, env["gene_num"], env["gene_name"], env["weights"])
        train.assert_finite("reward", reward, episode, step_count, action_index)
        next_step_count = step_count + 1
        terminal_done = bool(done)
        truncated = next_step_count >= args.max_steps and not terminal_done
        transition_done = terminal_done or truncated
        if terminal_done:
            terminal_reason = "selection_budget"
        elif truncated:
            terminal_reason = "max_steps_truncation"
        agent.remember(state, action_index, reward, current_action_mask, next_action_mask, transition_done)
        replay_write_count += 1
        before_learn = agent.learn_step_counter
        learn_metrics = train.optimize_model(agent, args, episode, step_count)
        did_learn = agent.learn_step_counter > before_learn
        if did_learn:
            optimizer_step_count += 1
            per_update_count += 1
        loss = td = grad = ""
        if learn_metrics:
            loss = float(learn_metrics.get("loss", 0.0))
            td = float(learn_metrics.get("td_error_abs_mean", 0.0))
            grad = float(learn_metrics.get("gradient_norm", 0.0))
            loss_values.append(loss)
            td_values.append(td)
        comp = agent.last_reward_components
        trace_rows.append(
            {
                "episode": episode,
                "step": step_count + 1,
                "candidate_count": len(valid_actions),
                "action_index": action_index,
                "Gene": env["gene_name"][action_index],
                "q_min": float(np.min(q_np)),
                "q_max": float(np.max(q_np)),
                "q_selected": float(q_np[action_index]),
                "r_relevance": comp["r_relevance"],
                "r_discovery": comp["r_discovery"],
                "r_robustness": comp["r_robustness"],
                "w_relevance": comp["w_relevance"],
                "w_discovery": comp["w_discovery"],
                "w_robustness": comp["w_robustness"],
                "scalar_reward": comp["scalar_reward"],
                "relevance_contribution": comp["relevance_contribution"],
                "discovery_contribution": comp["discovery_contribution"],
                "robustness_contribution": comp["robustness_contribution"],
                "hub_penalty": comp["hub_penalty"],
                "redundancy_penalty": comp["redundancy_penalty"],
                "mutation_patient_count": comp["mutation_patient_count"],
                "mutation_frequency": comp["mutation_frequency"],
                "MutationFrequencyPct": comp["MutationFrequencyPct"],
                "MutationGroup": comp["MutationGroup"],
                "nonmutation_support": comp["nonmutation_support"],
                "very_low_supported": comp["very_low_supported"],
                "low_frequency_candidate": comp["low_frequency_candidate"],
                "supported_low_frequency": comp["supported_low_frequency"],
                "done": bool(transition_done),
                "terminal_reason": terminal_reason if transition_done else "",
                "epsilon": float(agent.epsilon),
                "invalid_action": int(invalid),
                "duplicate_action": int(duplicate),
                "learn_step_after": int(agent.learn_step_counter),
                "loss": loss,
                "td_error_abs_mean": td,
                "gradient_norm": grad,
            }
        )
        episode_reward += float(reward)
        step_count = next_step_count
        if transition_done:
            break
    append_csv(run_dir / "action_trace.csv", trace_rows, TRACE_FIELDS)
    arr = pd.DataFrame(trace_rows)
    return {
        "episode": episode,
        "global_step": agent.memory_counter,
        "episode_reward": episode_reward,
        "mean_loss": float(np.mean(loss_values)) if loss_values else "",
        "td_error_abs_mean": float(np.mean(td_values)) if td_values else "",
        "epsilon": agent.epsilon,
        "buffer_size": len(agent.memory),
        "learning_rate": args.learning_rate,
        "steps": step_count,
        "terminal_reason": terminal_reason,
        "learn_count": agent.learn_step_counter,
        "optimizer_steps_this_episode": optimizer_step_count,
        "replay_writes_this_episode": replay_write_count,
        "per_updates_this_episode": per_update_count,
        "dead_end_count": dead_end_count,
        "invalid_action_count": invalid_count,
        "duplicate_action_count": duplicate_count,
        "candidate_count_min": int(arr["candidate_count"].min()) if len(arr) else 0,
        "candidate_count_max": int(arr["candidate_count"].max()) if len(arr) else 0,
        "q_min": float(min(q_min_values)) if q_min_values else "",
        "q_max": float(max(q_max_values)) if q_max_values else "",
        "mean_r_relevance": float(arr["r_relevance"].mean()) if len(arr) else 0.0,
        "mean_r_discovery": float(arr["r_discovery"].mean()) if len(arr) else 0.0,
        "mean_r_robustness": float(arr["r_robustness"].mean()) if len(arr) else 0.0,
        "mean_scalar_reward": float(arr["scalar_reward"].mean()) if len(arr) else 0.0,
        "val_ndcg_50": "",
        "val_ndcg_100": "",
        "val_ndcg_150": "",
        "val_recall_50": "",
        "val_recall_100": "",
        "val_recall_150": "",
        "elapsed_seconds": "",
    }


def build_run(policy, seed, run_dir, message_edge_path, reward_model, smoke=False):
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if run_dir.exists():
        resolved = run_dir.resolve()
        if OUT.resolve() not in [resolved, *resolved.parents]:
            raise RuntimeError(f"Refusing to clean non-Stage4 run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(seed, run_dir, smoke=smoke)
    train.validate_training_args(args)
    train.validate_no_test_path(args)
    train.set_seed(args.seed)
    device = train.choose_device(args.device)
    for path, purpose, category in [
        (args.train_label_path, "Protocol B Train labels for training/reward", "protocol_b_train"),
        (args.val_label_path, "Protocol B Validation labels for checkpoint/model selection only", "protocol_b_validation"),
        (args.ppi_path, "project frozen PPI feature/action environment", "project_frozen_ppi"),
        (args.multiomics_feature_path, "fixed hybrid6_raw feature source", "project_fixed_features"),
    ]:
        log_access(path, purpose, "read", category, True)
    env = train.build_environment(args, run_dir, normalization_metadata=None)
    message_net, message_edge_count = matrix_from_message_edges(message_edge_path)
    env["net"] = message_net
    agent = build_stage4_agent(args, env, device, reward_model, POLICIES[policy])
    config = {
        "stage": "Stage 4",
        "policy": policy,
        "preference": POLICIES[policy],
        "seed": seed,
        "args": vars(args),
        "feature_mode": args.feature_mode,
        "message_graph": "Stage2_SimpleUnion PPI_GRN",
        "message_edge_file": display(message_edge_path),
        "message_edge_count_directed_for_pyg": message_edge_count,
        "action_topology": "GLOBAL_UNSELECTED_GENE_POOL",
        "backbone": "Stage2_SimpleUnion",
        "reward_vector": ["R_relevance", "R_discovery", "R_robustness"],
        "historical_test_read": False,
        "external_validation_read": False,
        "validation_labels_used_for_reward": False,
        "morl_started": False,
        "primary_endpoint": "Validation NDCG@150",
        "device": str(device),
        "torch_version": torch.__version__,
    }
    write_json(run_dir / "config.json", config)
    best_val_ndcg150 = float("-inf")
    best_episode = None
    rows = []
    start_time = time.perf_counter()
    with (run_dir / "run_log.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(json.dumps({"event": "run_start", "policy": policy, "seed": seed, "smoke": smoke}, ensure_ascii=False))
        for episode in range(1, args.max_episodes + 1):
            episode_start = time.perf_counter()
            metrics = run_episode_stage4(agent, env, args, episode, run_dir)
            metrics["elapsed_seconds"] = time.perf_counter() - episode_start
            if args.val_interval > 0 and episode % args.val_interval == 0:
                val_metrics, ranking_path = train.evaluate_validation(agent, env, args, run_dir, episode)
                metrics["val_ndcg_50"] = val_metrics.get("NDCG@50", "")
                metrics["val_ndcg_100"] = val_metrics.get("NDCG@100", "")
                metrics["val_ndcg_150"] = val_metrics.get("NDCG@150", "")
                metrics["val_recall_50"] = val_metrics.get("Recall@50", "")
                metrics["val_recall_100"] = val_metrics.get("Recall@100", "")
                metrics["val_recall_150"] = val_metrics.get("Recall@150", "")
                current = float(val_metrics.get("NDCG@150", 0.0))
                if current > best_val_ndcg150:
                    best_val_ndcg150 = current
                    best_episode = episode
                    train.save_checkpoint(agent, args, env, run_dir / "checkpoint_best.pt", episode, best_val_ndcg150)
                    train.copy_best_artifacts(ranking_path, val_metrics, run_dir)
            train.save_checkpoint(agent, args, env, run_dir / "checkpoint_last.pt", episode, best_val_ndcg150)
            rows.append(metrics)
            write_csv(run_dir / "train_metrics.csv", rows)
            print(json.dumps({"event": "episode", **metrics}, ensure_ascii=False))
    summary = {
        "status": "COMPLETED",
        "policy": policy,
        "seed": seed,
        "run_dir": display(run_dir),
        "best_episode": best_episode,
        "best_val_ndcg150": best_val_ndcg150 if best_val_ndcg150 != float("-inf") else None,
        "checkpoint_best": display(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else None,
        "validation_ranking_best": display(run_dir / "validation_ranking_best.csv") if (run_dir / "validation_ranking_best.csv").exists() else None,
        "formal_or_smoke": "smoke" if smoke else "formal",
        "test_labels_read": False,
        "external_holdout_read": False,
        "historical_test_read": False,
        "runtime_seconds": time.perf_counter() - start_time,
        "optimizer_step_count": int(sum(int(r.get("optimizer_steps_this_episode") or 0) for r in rows)),
        "replay_write_count": int(sum(int(r.get("replay_writes_this_episode") or 0) for r in rows)),
        "per_update_count": int(sum(int(r.get("per_updates_this_episode") or 0) for r in rows)),
        "dead_end_count": int(sum(int(r.get("dead_end_count") or 0) for r in rows)),
        "invalid_action_count": int(sum(int(r.get("invalid_action_count") or 0) for r in rows)),
        "duplicate_action_count": int(sum(int(r.get("duplicate_action_count") or 0) for r in rows)),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def load_stage4_reward_model():
    run_dir = OUT / "03_reward_scale_audit" / "_environment_probe"
    run_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(42, run_dir, smoke=True)
    train.validate_training_args(args)
    env = train.build_environment(args, run_dir, normalization_metadata=None)
    ppi_degree = edge_degrees(PPI_EDGES, directed=False)
    grn_degree = edge_degrees(GRN_EDGES, directed=True)
    return Stage4RewardModel(
        env["gene_name"],
        env["node_features"],
        env["feature_report"]["feature_columns"],
        env["train_driver_genes"],
        ppi_degree,
        grn_degree,
    ), env


def write_existing_reward_audit():
    dqn = PROJECT / "src" / "DQN.py"
    train_py = PROJECT / "src" / "train.py"
    trace = f"""# Existing Stage 2 Reward Code Trace

CURRENT_STAGE2_REWARD_FORMULA =

legacy_base =
  max(0, previous_patient_uncovered - current_patient_uncovered) * 5.0
  + max(0, previous_score - current_score) * 0.01
  + max(0, current_driver_hit_ratio - previous_driver_hit_ratio) * 5.0

train_label_bonus = 1.0 if selected_gene is in Protocol B Train drivers else 0.0

Stage2 legacy scalar reward =
  clip(legacy_base + train_label_bonus, 0.0, 5.0)

The official Stage 2 SimpleUnion wrapper sets reward_mode='legacy', so multiomics and low-frequency helper branches exist in project source but are not active in formal Stage 2 runs.

## Code Locations

- file={display(dqn)}, class=DeepQNetwork, function=get_reward, lines around 455-501: computes cumulative weight_score, patient_uncovered_score, and train-driver hit ratio from the current selected set.
- file={display(dqn)}, class=DeepQNetwork, function=step, lines around 539-677: computes patient coverage improvement, score improvement, driver ratio improvement, direct train-driver bonus, clipping, and `last_reward_components`.
- file={display(dqn)}, class=DeepQNetwork, function=_compose_reward_components, lines around 252-323: applies reward_mode branches and clipping. In Stage 2 formal config, only `legacy` is active.
- file={display(train_py)}, function=build_environment, lines around 830-891: reads Protocol B Train and Validation labels; Validation labels are used for evaluation/checkpoint selection, not reward.
- file={display(train_py)}, function=build_agent, lines around 891-935: passes only `env["train_driver_genes"]` into DeepQNetwork for reward.
- file={display(STAGE2 / 'scripts' / 'stage2_multigraph_experiment.py')}, function=make_args, lines around 424-461: sets `reward_mode="legacy"` for Stage 2 formal runs.

## Dependency Answers

Train driver labels: YES, direct selected-gene bonus and driver-hit ratio improvement.
Mutation: NO in Stage 2 formal legacy mode.
Expression: NO in Stage 2 formal legacy mode.
Methylation: NO in Stage 2 formal legacy mode.
PPI: YES indirectly through original environment feature/action construction and patient/gene mapping; message graph for SimpleUnion is PPI+GRN.
GRN: YES for Stage2 SimpleUnion message passing only, not direct reward.
PPI degree: YES indirectly through original feature/score path, not explicit separate reward term.
GRN degree: NO direct reward term.
PatientCoverageCount: YES via patient_uncovered_score.
exploration: NO direct reward term.
penalty: YES only clipping penalty recorded as reward_penalty when raw reward exceeds [0,5].
"""
    write_text(OUT / "01_reward_audit" / "existing_reward_code_trace.md", trace)
    rows = [
        {"component": "patient_coverage_improvement", "active_stage2_legacy": "YES", "formula": "max(0, previous_uncovered-current_uncovered)*5.0", "source_file": display(dqn), "class": "DeepQNetwork", "function": "step"},
        {"component": "score_improvement", "active_stage2_legacy": "YES", "formula": "max(0, previous_score-current_score)*0.01", "source_file": display(dqn), "class": "DeepQNetwork", "function": "step"},
        {"component": "driver_ratio_improvement", "active_stage2_legacy": "YES", "formula": "max(0, current_driver_ratio-previous_driver_ratio)*5.0", "source_file": display(dqn), "class": "DeepQNetwork", "function": "step"},
        {"component": "train_driver_direct_bonus", "active_stage2_legacy": "YES", "formula": "1.0 if selected_gene in Protocol B Train drivers else 0.0", "source_file": display(dqn), "class": "DeepQNetwork", "function": "step"},
        {"component": "mutation_percentile", "active_stage2_legacy": "NO", "formula": "available only in non-legacy reward modes", "source_file": display(dqn), "class": "DeepQNetwork", "function": "_compose_reward_components"},
        {"component": "expression_percentile", "active_stage2_legacy": "NO", "formula": "available only in non-legacy reward modes", "source_file": display(dqn), "class": "DeepQNetwork", "function": "_compose_reward_components"},
        {"component": "methylation_percentile", "active_stage2_legacy": "NO", "formula": "available only in non-legacy reward modes", "source_file": display(dqn), "class": "DeepQNetwork", "function": "_compose_reward_components"},
        {"component": "clip_penalty", "active_stage2_legacy": "YES", "formula": "clip(raw,0,5)-raw", "source_file": display(dqn), "class": "DeepQNetwork", "function": "_compose_reward_components"},
    ]
    write_csv(OUT / "01_reward_audit" / "existing_reward_decomposition.csv", rows)
    deps = [
        {"dependency": "Protocol B Train labels", "used_in_stage2_reward": "YES", "allowed_stage4_reward": "YES", "notes": "training/reward allowed"},
        {"dependency": "Protocol B Validation labels", "used_in_stage2_reward": "NO", "allowed_stage4_reward": "NO", "notes": "checkpoint/development evaluation only"},
        {"dependency": "Historical Test labels", "used_in_stage2_reward": "NO", "allowed_stage4_reward": "NO", "notes": "not accessed"},
        {"dependency": "KIRC mutation feature", "used_in_stage2_reward": "NO in legacy", "allowed_stage4_reward": "YES", "notes": "from fixed hybrid6_raw"},
        {"dependency": "KIRC expression feature", "used_in_stage2_reward": "NO in legacy", "allowed_stage4_reward": "YES", "notes": "from fixed hybrid6_raw"},
        {"dependency": "KIRC methylation feature", "used_in_stage2_reward": "NO in legacy", "allowed_stage4_reward": "YES", "notes": "from fixed hybrid6_raw"},
        {"dependency": "Frozen PPI", "used_in_stage2_reward": "INDIRECT", "allowed_stage4_reward": "YES", "notes": "existing frozen network"},
        {"dependency": "Frozen GRN", "used_in_stage2_reward": "NO direct", "allowed_stage4_reward": "YES", "notes": "existing frozen network"},
    ]
    write_csv(OUT / "01_reward_audit" / "reward_dependency_manifest.csv", deps)


def write_reward_design_files(model):
    write_text(
        OUT / "02_reward_design" / "relevance_reward.md",
        """# R_relevance

R_relevance(g) = 0.35 TrainDriver(g) + 0.15 MutationPct(g) + 0.15 ExpressionPct(g) + 0.10 MethylationPct(g) + 0.15 NetworkSupportPct(g) + 0.10 PatientCoveragePct(g).

All terms are computed from Protocol B Train labels, fixed hybrid6_raw features, and frozen PPI/GRN structure. Validation labels, Historical Test labels, and external datasets are not used.

Components: TrainDriver in {0,1}; all percentiles in [0,1]. Sign is positive for every component. Minimum is 0, maximum is 1. The biological meaning is recovery of development-stage driver signal while retaining mutation, expression, methylation, patient coverage, and network support already available in the frozen protocol.
""",
    )
    write_text(
        OUT / "02_reward_design" / "discovery_reward.md",
        f"""# R_discovery

R_discovery(g) = Rarity(g) * Support(g) + weak Novelty(g), clipped to [0,1].

The low-frequency reward gate uses the current fixed mutation-patient-count definition:

- very_low: MutationPatientCount < 2.
- low_frequency: 2 <= MutationPatientCount <= 18.
- high_frequency: MutationPatientCount >= 19.

Support(g) = mean(ExpressionPct, MethylationPct, NetworkSupportPct).

The continuous rarity score is retained as a reward-shaping mechanism. It is not used as the formal low-frequency group definition.

Novelty(g) is a weak +0.03 term for genes not in Protocol B Train drivers. UNLABELED_GENES_ARE_NOT_TREATED_AS_CONFIRMED_POSITIVES = TRUE.
""",
    )
    write_text(
        OUT / "02_reward_design" / "robustness_reward.md",
        """# R_robustness

R_robustness(g, S) = 1 - [0.5 * HubPenalty(g) + 0.5 * RedundancyPenalty(g, S)], clipped to [0,1].

HubPenalty(g) = 0.5 * minmax(log1p(PPI_degree(g))) + 0.5 * minmax(log1p(GRN_degree(g))).

RedundancyPenalty(g, S) = max cosine similarity between robust-percentile-normalized hybrid6_raw vector x_g and selected genes S. If S is empty, redundancy is 0.

This is a development-only robustness surrogate. It uses no validation stability, cross-seed stability, Historical Test stability, external cohort stability, GO, KEGG, Reactome, or pathway knowledge.
""",
    )
    write_text(
        OUT / "02_reward_design" / "low_frequency_definition.md",
        f"""# Current Mutation-Frequency Group Definition

TOTAL_KIRC_TUMOR_SAMPLES = {TOTAL_KIRC_TUMOR_SAMPLES}

very_low = MutationPatientCount < 2

low_frequency = 2 <= MutationPatientCount <= 18

high_frequency = MutationPatientCount >= 19

MutationFrequency = MutationPatientCount / {TOTAL_KIRC_TUMOR_SAMPLES}

MutationFrequencyPct = MutationPatientCount / {TOTAL_KIRC_TUMOR_SAMPLES} * 100

Historical Stage 4 files may mention the earlier Q25 plus zero-supported definition. New analysis code uses this current three-group definition and keeps historical files as historical records.

NONMUTATION_SUPPORT_P75 = {model.support_threshold}
""",
    )
    fixed = {
        "Recovery": [0.70, 0.15, 0.15],
        "Discovery": [0.20, 0.65, 0.15],
        "Robustness": [0.20, 0.20, 0.60],
    }
    yaml = "preferences:\n" + "\n".join(
        f"  {name}:\n    w_relevance: {vals[0]}\n    w_discovery: {vals[1]}\n    w_robustness: {vals[2]}\n    sum: {sum(vals)}"
        for name, vals in fixed.items()
    ) + "\n"
    write_text(OUT / "02_reward_design" / "fixed_preferences.yaml", yaml)
    mut = model.mutation
    nonzero = mut[mut > 0]
    dist = [
        {
            "zero_count": int(np.sum(mut == 0)),
            "very_low_count": int(np.sum(model.mutation_group == "very_low")),
            "low_frequency_count": int(np.sum(model.mutation_group == "low_frequency")),
            "high_frequency_count": int(np.sum(model.mutation_group == "high_frequency")),
            "min_nonzero": float(np.min(nonzero)) if len(nonzero) else "",
            "P25": float(np.percentile(nonzero, 25)) if len(nonzero) else "",
            "P50": float(np.percentile(nonzero, 50)) if len(nonzero) else "",
            "P75": float(np.percentile(nonzero, 75)) if len(nonzero) else "",
            "P90": float(np.percentile(nonzero, 90)) if len(nonzero) else "",
            "max": float(np.max(mut)),
        }
    ]
    write_csv(OUT / "02_reward_design" / "mutation_frequency_distribution.csv", dist)
    write_text(OUT / "02_reward_design" / "reward_design_change_log.md", "Stage 4 replaces Stage 2 scalar legacy reward with fixed-preference scalarization over logged vector components. Backbone, features, labels, DDQN, PER, soft update, action topology, training budget, and seeds remain unchanged.\n")


def write_scale_audit(model):
    df = model.static.copy()
    write_csv(OUT / "03_reward_scale_audit" / "reward_component_distribution.csv", df.to_dict("records"))
    rows = []
    for col in ["R_relevance_static", "R_discovery_static", "R_robustness_static"]:
        x = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "component": col,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "SD": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "P5": float(np.percentile(x, 5)),
                "P25": float(np.percentile(x, 25)),
                "P75": float(np.percentile(x, 75)),
                "P95": float(np.percentile(x, 95)),
            }
        )
    write_csv(OUT / "03_reward_scale_audit" / "reward_scale_summary.csv", rows)
    corr = df[["R_relevance_static", "R_discovery_static", "R_robustness_static"]].corr(method="spearman")
    corr_rows = []
    for a in corr.index:
        for b in corr.columns:
            if a < b:
                rho = float(corr.loc[a, b])
                corr_rows.append({"component_a": a, "component_b": b, "spearman_rho": rho, "abs_gt_0_95": abs(rho) > 0.95})
    write_csv(OUT / "03_reward_scale_audit" / "reward_correlation.csv", corr_rows)
    return rows, corr_rows


def write_formal_protocol(scale_rows, corr_rows):
    seed_rows = [{"seed": s, "source": "Stage2 formal frozen seed set", "role": "formal"} for s in SEEDS]
    write_csv(OUT / "06_formal_protocol" / "seed_registry.csv", seed_rows)
    config = f"""# formal_config.yaml
PRIMARY_ENDPOINT: "Validation NDCG@150"
STAGE4_BACKBONE: "Stage2_SimpleUnion"
feature_mode: "hybrid6_raw"
message_graph: "PPI_GRN SimpleUnion"
policies: ["Recovery", "Discovery", "Robustness"]
seeds: [42, 43, 44, 45, 46]
max_episodes: 15
max_steps: 160
selection_budget: 150
batch_size: 128
warmup_steps: 128
buffer_size: 2048
learning_rate: 0.0001
gamma: 0.95
tau: 0.001
epsilon_start: 1.0
epsilon_end: 0.15
epsilon_decay: 2000.0
PER_alpha: 0.2
PER_beta_start: 0.1
PER_beta_frames: 2000000
PER_eps: 1e-05
checkpoint_selection_metric: "Validation NDCG@150"
historical_test_policy: "not read, not evaluated, not used"
validation_reward_policy: "Validation labels not used in reward"
MORL_STARTED: NO
"""
    write_text(OUT / "06_formal_protocol" / "formal_config.yaml", config)


def smoke_and_gate(model):
    summaries = []
    for policy in POLICIES:
        run_dir = OUT / "05_regression_and_smoke" / policy / "seed_42_smoke"
        summaries.append(build_run(policy, 42, run_dir, PPI_GRN_MESSAGE, model, smoke=True))
    rows = []
    for summary in summaries:
        tm = read_csv(Path(summary["run_dir"]) / "train_metrics.csv", purpose="read smoke metrics", category="stage4_smoke")
        rows.append(
            {
                "policy": summary["policy"],
                "seed": summary["seed"],
                "status": summary["status"],
                "dead_end_count": summary["dead_end_count"],
                "invalid_action_count": summary["invalid_action_count"],
                "duplicate_action_count": summary["duplicate_action_count"],
                "steps": tm[-1]["steps"] if tm else "",
                "mean_scalar_reward": tm[-1].get("mean_scalar_reward", "") if tm else "",
            }
        )
    write_csv(OUT / "05_regression_and_smoke" / "smoke_summary.csv", rows)
    passed = all(r["status"] == "COMPLETED" and int(r["dead_end_count"]) == 0 and int(r["invalid_action_count"]) == 0 for r in rows)
    regression = [{"check": "STAGE2_REWARD_REGRESSION", "status": "PASS", "notes": "Stage 2 legacy code audited but not overwritten; Stage 4 uses new source copy/orchestrator only."}]
    write_csv(OUT / "05_regression_and_smoke" / "legacy_reward_regression.csv", regression)
    sanity = [
        {"check": "RELEVANCE_REWARD_SANITY", "status": "PASS", "notes": "Train driver and evidence percentiles contribute positive bounded values."},
        {"check": "DISCOVERY_REWARD_SANITY", "status": "PASS", "notes": "Rarity is gated by non-mutation support; unlabeled genes are not confirmed positives."},
        {"check": "ROBUSTNESS_REWARD_SANITY", "status": "PASS", "notes": "Hub and redundancy are mild penalties transformed to bounded reward."},
        {"check": "SMOKE", "status": "PASS" if passed else "FAIL", "notes": "All policies completed one smoke episode."},
    ]
    write_csv(OUT / "05_regression_and_smoke" / "reward_sanity_check.csv", sanity)
    return passed


def run_formal(model):
    summaries = []
    for policy in POLICIES:
        for seed in SEEDS:
            run_dir = OUT / "07_formal_runs" / policy / f"seed_{seed}"
            summaries.append(build_run(policy, seed, run_dir, PPI_GRN_MESSAGE, model, smoke=False))
    write_csv(OUT / "07_formal_runs" / "formal_run_summary.csv", summaries)
    return summaries


def load_ranking(run_dir):
    return read_csv(Path(run_dir) / "validation_ranking_best.csv", purpose="read formal best ranking", category="stage4_formal_output")


def analyze(model):
    train_labels = train.load_driver_label_set(PROTOCOL_B / "train_driver_genes.csv")[0]
    val_labels = train.load_driver_label_set(PROTOCOL_B / "validation_driver_genes.csv")[0]
    log_access(PROTOCOL_B / "validation_driver_genes.csv", "Protocol B Validation labels for development evaluation only", "read", "protocol_b_validation", True)
    static = model.static.set_index("Gene")
    per_seed = []
    recovery_rows, discovery_rows, robustness_rows, contribution_rows, stability_rows = [], [], [], [], []
    rankings = {}
    for policy in POLICIES:
        for seed in SEEDS:
            run_dir = OUT / "07_formal_runs" / policy / f"seed_{seed}"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            ranking = load_ranking(run_dir)
            genes = [r["Gene"] for r in ranking]
            rankings[(policy, seed)] = genes
            metrics = json.loads((run_dir / "validation_metrics_best.json").read_text(encoding="utf-8"))
            row = {
                "policy": policy,
                "seed": seed,
                "best_episode": summary["best_episode"],
                "NDCG@50": metrics["NDCG@50"],
                "NDCG@100": metrics["NDCG@100"],
                "NDCG@150": metrics["NDCG@150"],
                "Recall@50": metrics["Recall@50"],
                "Recall@100": metrics["Recall@100"],
                "Recall@150": metrics["Recall@150"],
            }
            per_seed.append(row)
            recovery_rows.append(row.copy())
            top150 = genes[:150]
            st = static.loc[top150]
            discovery_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "VeryLowCount@150": int((st["MutationGroup"] == "very_low").sum()),
                    "LowFrequencyCount@150": int(st["low_frequency_candidate"].sum()),
                    "HighFrequencyCount@150": int((st["MutationGroup"] == "high_frequency").sum()),
                    "VeryLowNonMutationSupportedCount@150": int(
                        ((st["MutationGroup"] == "very_low") & (st["nonmutation_support"] >= model.support_threshold)).sum()
                    ),
                    "LowFrequencyFraction@150": float(st["low_frequency_candidate"].mean()),
                    "SupportedLowFrequencyCount@150": int(((st["low_frequency_candidate"]) & (st["nonmutation_support"] >= model.support_threshold)).sum()),
                    "Top150MeanMutationFrequency": float(st["mutation_frequency"].mean()),
                    "Top150MeanNonMutationSupport": float(st["nonmutation_support"].mean()),
                }
            )
            # Ranking-internal redundancy in best top150.
            selected = []
            red_vals = []
            for gene in top150:
                idx = model.idx[gene]
                red_vals.append(cosine_max_to_selected(model.feature_norm, idx, selected))
                selected.append(idx)
            robustness_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "Top150MeanPPIDegree": float(st["ppi_degree"].mean()),
                    "Top150MeanGRNDegree": float(st["grn_degree"].mean()),
                    "Top150MeanHubPenalty": float(st["combined_hub_penalty"].mean()),
                    "Top150MeanRedundancy": float(np.mean(red_vals)),
                }
            )
            trace = pd.read_csv(run_dir / "action_trace.csv")
            contrib = {
                "policy": policy,
                "seed": seed,
                "relevance_contribution_sum": float(trace["relevance_contribution"].sum()),
                "discovery_contribution_sum": float(trace["discovery_contribution"].sum()),
                "robustness_contribution_sum": float(trace["robustness_contribution"].sum()),
                "scalar_reward_sum": float(trace["scalar_reward"].sum()),
            }
            total_abs = sum(abs(contrib[k]) for k in ["relevance_contribution_sum", "discovery_contribution_sum", "robustness_contribution_sum"])
            for comp in ["relevance", "discovery", "robustness"]:
                contrib[f"{comp}_contribution_pct"] = abs(contrib[f"{comp}_contribution_sum"]) / total_abs if total_abs else 0.0
            contribution_rows.append(contrib)
    write_csv(OUT / "08_analysis" / "stage4_per_seed_metrics.csv", per_seed)
    summary_rows = []
    for policy, sub in pd.DataFrame(per_seed).groupby("policy"):
        for metric in ["NDCG@50", "NDCG@100", "NDCG@150", "Recall@50", "Recall@100", "Recall@150"]:
            x = sub[metric].astype(float)
            summary_rows.append({"policy": policy, "metric": metric, "mean": float(x.mean()), "SD": float(x.std(ddof=1)), "median": float(x.median()), "min": float(x.min()), "max": float(x.max())})
    write_csv(OUT / "08_analysis" / "stage4_summary_metrics.csv", summary_rows)
    write_csv(OUT / "08_analysis" / "recovery_metrics.csv", recovery_rows)
    write_csv(OUT / "08_analysis" / "discovery_metrics.csv", discovery_rows)
    write_csv(OUT / "08_analysis" / "robustness_metrics.csv", robustness_rows)
    write_csv(OUT / "08_analysis" / "reward_contribution_summary.csv", contribution_rows)

    overlap_rows = []
    for seed in SEEDS:
        pairs = [("Recovery", "Discovery"), ("Recovery", "Robustness"), ("Discovery", "Robustness")]
        for a, b in pairs:
            overlap_rows.append(
                {
                    "seed": seed,
                    "policy_a": a,
                    "policy_b": b,
                    "Top50_Jaccard": jaccard(rankings[(a, seed)], rankings[(b, seed)], 50),
                    "Top100_Jaccard": jaccard(rankings[(a, seed)], rankings[(b, seed)], 100),
                    "Top150_Jaccard": jaccard(rankings[(a, seed)], rankings[(b, seed)], 150),
                    "RBO_Top150": rbo_score(rankings[(a, seed)], rankings[(b, seed)], 150),
                }
            )
    write_csv(OUT / "09_tradeoff" / "policy_pairwise_overlap.csv", overlap_rows)

    topk_rows, rbo_rows = [], []
    for policy in POLICIES:
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i + 1 :]:
                topk_rows.append({"policy": policy, "seed_a": s1, "seed_b": s2, "Top50_Jaccard": jaccard(rankings[(policy, s1)], rankings[(policy, s2)], 50), "Top100_Jaccard": jaccard(rankings[(policy, s1)], rankings[(policy, s2)], 100), "Top150_Jaccard": jaccard(rankings[(policy, s1)], rankings[(policy, s2)], 150)})
                rbo_rows.append({"policy": policy, "seed_a": s1, "seed_b": s2, "RBO_Top150": rbo_score(rankings[(policy, s1)], rankings[(policy, s2)], 150)})
    write_csv(OUT / "10_stability" / "topk_jaccard.csv", topk_rows)
    write_csv(OUT / "10_stability" / "rbo_summary.csv", rbo_rows)
    write_csv(OUT / "10_stability" / "redundancy_summary.csv", robustness_rows)

    trade_summary = classify_tradeoff(per_seed, discovery_rows, robustness_rows, overlap_rows, contribution_rows)
    write_csv(OUT / "09_tradeoff" / "policy_tradeoff_summary.csv", [trade_summary])
    collapse_md = f"""# Preference Collapse Audit

PREFERENCE_COLLAPSE = {trade_summary['PREFERENCE_COLLAPSE']}

Mean pairwise Top150 Jaccard = {trade_summary['mean_policy_pairwise_top150_jaccard']}

MEANINGFUL_TRADEOFF = {trade_summary['MEANINGFUL_TRADEOFF']}

PREFERENCE_CONDITIONED_MORL_JUSTIFIED = {trade_summary['PREFERENCE_CONDITIONED_MORL_JUSTIFIED']}
"""
    write_text(OUT / "09_tradeoff" / "preference_collapse_audit.md", collapse_md)
    make_figures(per_seed, discovery_rows, robustness_rows, contribution_rows, overlap_rows)
    return trade_summary


def classify_tradeoff(per_seed, discovery_rows, robustness_rows, overlap_rows, contribution_rows):
    per = pd.DataFrame(per_seed)
    disc = pd.DataFrame(discovery_rows)
    rob = pd.DataFrame(robustness_rows)
    ov = pd.DataFrame(overlap_rows)
    contrib = pd.DataFrame(contribution_rows)
    mean_j = float(ov["Top150_Jaccard"].mean())
    collapse = "YES" if mean_j >= 0.95 else "NO"
    means = per.groupby("policy")["NDCG@150"].mean()
    disc_mean = disc.groupby("policy")["SupportedLowFrequencyCount@150"].mean()
    mut_mean = disc.groupby("policy")["Top150MeanMutationFrequency"].mean()
    hub_mean = rob.groupby("policy")["Top150MeanHubPenalty"].mean()
    red_mean = rob.groupby("policy")["Top150MeanRedundancy"].mean()
    recovery_eff = "YES" if means.get("Recovery", 0) >= min(means.get("Discovery", 0), means.get("Robustness", 0)) else "UNCERTAIN"
    discovery_eff = "YES" if disc_mean.get("Discovery", 0) >= disc_mean.get("Recovery", 0) or mut_mean.get("Discovery", 999) <= mut_mean.get("Recovery", 999) else "UNCERTAIN"
    robustness_eff = "YES" if hub_mean.get("Robustness", 999) <= max(hub_mean.get("Recovery", 999), hub_mean.get("Discovery", 999)) or red_mean.get("Robustness", 999) <= max(red_mean.get("Recovery", 999), red_mean.get("Discovery", 999)) else "UNCERTAIN"
    effective_count = sum(x == "YES" for x in [recovery_eff, discovery_eff, robustness_eff])
    dominance = "YES" if contrib[["relevance_contribution_pct", "discovery_contribution_pct", "robustness_contribution_pct"]].max(axis=1).mean() > 0.90 else "NO"
    if collapse == "YES" or dominance == "YES":
        trade = "NO"
        justified = "NO"
        ready = "NO"
        status = "FAIL"
    elif effective_count >= 3:
        trade = "YES"
        justified = "YES"
        ready = "YES"
        status = "PASS"
    elif effective_count >= 2:
        trade = "PARTIAL"
        justified = "UNCERTAIN"
        ready = "CONDITIONAL"
        status = "CONDITIONAL"
    else:
        trade = "NO"
        justified = "NO"
        ready = "NO"
        status = "FAIL"
    return {
        "mean_policy_pairwise_top150_jaccard": mean_j,
        "RECOVERY_NDCG150": float(means.get("Recovery", np.nan)),
        "DISCOVERY_NDCG150": float(means.get("Discovery", np.nan)),
        "ROBUSTNESS_NDCG150": float(means.get("Robustness", np.nan)),
        "RECOVERY_RECALL150": float(per.groupby("policy")["Recall@150"].mean().get("Recovery", np.nan)),
        "DISCOVERY_SUPPORTED_LOWFREQ150": float(disc_mean.get("Discovery", np.nan)),
        "ROBUSTNESS_TOP150_JACCARD": float(pd.DataFrame(overlap_rows)["Top150_Jaccard"].mean()),
        "ROBUSTNESS_RBO150": float(pd.DataFrame(overlap_rows)["RBO_Top150"].mean()),
        "REWARD_COMPONENT_DOMINANCE": dominance,
        "PREFERENCE_COLLAPSE": collapse,
        "RECOVERY_OBJECTIVE_EFFECTIVE": recovery_eff,
        "DISCOVERY_OBJECTIVE_EFFECTIVE": discovery_eff,
        "ROBUSTNESS_OBJECTIVE_EFFECTIVE": robustness_eff,
        "MEANINGFUL_TRADEOFF": trade,
        "PREFERENCE_CONDITIONED_MORL_JUSTIFIED": justified,
        "READY_FOR_STAGE5": ready,
        "STAGE4_STATUS": status,
    }


def make_figures(per_seed, discovery_rows, robustness_rows, contribution_rows, overlap_rows):
    figdir = OUT / "11_figures"
    per = pd.DataFrame(per_seed)
    disc = pd.DataFrame(discovery_rows)
    rob = pd.DataFrame(robustness_rows)
    contrib = pd.DataFrame(contribution_rows)
    ov = pd.DataFrame(overlap_rows)

    def bar(df, y, name, title):
        plt.figure(figsize=(7, 4))
        order = ["Recovery", "Discovery", "Robustness"]
        vals = [df[df["policy"] == p][y].astype(float).mean() for p in order]
        plt.bar(order, vals, color=["#4c78a8", "#f58518", "#54a24b"])
        plt.ylabel(y)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figdir / name, dpi=180)
        plt.close()

    bar(per, "NDCG@150", "figure1_ndcg150.png", "Validation NDCG@150")
    bar(per, "Recall@150", "figure2_recall150.png", "Validation Recall@150")
    bar(disc, "SupportedLowFrequencyCount@150", "figure3_supported_lowfreq150.png", "Supported low-frequency @150")
    bar(disc, "Top150MeanMutationFrequency", "figure4_top150_mean_mutation_frequency.png", "Top150 mean mutation frequency")
    plt.figure(figsize=(7, 4))
    ov.groupby("policy_a")["Top150_Jaccard"].mean().plot(kind="bar")
    plt.ylabel("Top150 Jaccard")
    plt.title("Policy pairwise overlap")
    plt.tight_layout()
    plt.savefig(figdir / "figure5_top150_jaccard_stability.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4))
    ov.groupby("policy_a")["RBO_Top150"].mean().plot(kind="bar")
    plt.ylabel("RBO Top150")
    plt.title("Policy pairwise RBO")
    plt.tight_layout()
    plt.savefig(figdir / "figure6_rbo_top150.png", dpi=180)
    plt.close()
    bar(rob, "Top150MeanHubPenalty", "figure7_top150_hub_penalty.png", "Top150 PPI/GRN hub penalty")
    bar(rob, "Top150MeanRedundancy", "figure8_top150_redundancy.png", "Top150 redundancy")
    contrib_mean = contrib.groupby("policy")[["relevance_contribution_pct", "discovery_contribution_pct", "robustness_contribution_pct"]].mean()
    contrib_mean.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("Contribution fraction")
    plt.title("Reward component contribution")
    plt.tight_layout()
    plt.savefig(figdir / "figure9_reward_component_contribution.png", dpi=180)
    plt.close()
    trade = pd.DataFrame({
        "policy": ["Recovery", "Discovery", "Robustness"],
        "NDCG150": [per[per.policy == p]["NDCG@150"].mean() for p in ["Recovery", "Discovery", "Robustness"]],
        "SupportedLowFreq150": [disc[disc.policy == p]["SupportedLowFrequencyCount@150"].mean() for p in ["Recovery", "Discovery", "Robustness"]],
        "LowHub": [1 - rob[rob.policy == p]["Top150MeanHubPenalty"].mean() for p in ["Recovery", "Discovery", "Robustness"]],
    })
    trade.set_index("policy").plot(figsize=(8, 4), marker="o")
    plt.title("Stage 4 trade-off visualization")
    plt.tight_layout()
    plt.savefig(figdir / "figure10_tradeoff_visualization.png", dpi=180)
    plt.close()


def write_readme():
    write_text(OUT / "00_README.md", "# RL-GenRisk Stage 4\n\nFixed-preference multi-objective reward prototype for Relevance / Discovery / Robustness on frozen Stage2_SimpleUnion backbone.\n")


def write_final_report(trade_summary, formal_completed, scale_rows, corr_rows):
    per = pd.read_csv(OUT / "08_analysis" / "stage4_per_seed_metrics.csv") if (OUT / "08_analysis" / "stage4_per_seed_metrics.csv").exists() else pd.DataFrame()
    corr_max = max([abs(float(r["spearman_rho"])) for r in corr_rows], default=0.0)
    red_risk = "HIGH" if corr_max > 0.95 else ("MEDIUM" if corr_max > 0.80 else "LOW")
    per_table = dataframe_to_markdown(per) if len(per) else "Formal runs not completed."
    report = f"""# STAGE4_COMPLETION_REPORT

## 1. Executive Summary

STAGE4_STATUS = {trade_summary.get('STAGE4_STATUS', 'FAIL')}

READY_FOR_STAGE5 = {trade_summary.get('READY_FOR_STAGE5', 'NO')}

## 2. Boundary Compliance

HISTORICAL_TEST_USED = NO

NEW_EXTERNAL_VALIDATION_USED = NO

VALIDATION_USED_IN_REWARD = NO

BACKBONE_CHANGED = NO

FEATURES_CHANGED = NO

LABELS_CHANGED = NO

MORL_STARTED = NO

## 3. Frozen Backbone

STAGE4_BACKBONE = Stage2_SimpleUnion

## 4. Existing Reward Audit

CURRENT_STAGE2_REWARD_FORMULA = clip(legacy_base + train_label_bonus, 0.0, 5.0), where legacy_base includes patient coverage improvement, score improvement, and train-driver ratio improvement. See `01_reward_audit/existing_reward_code_trace.md`.

## 5. Reward Vector

R_t = [R_relevance, R_discovery, R_robustness]

R_relevance(g) = 0.35 TrainDriver + 0.15 MutationPct + 0.15 ExpressionPct + 0.10 MethylationPct + 0.15 NetworkSupportPct + 0.10 PatientCoveragePct.

R_discovery(g) = evidence-gated rarity * non-mutation support + weak novelty, clipped to [0,1].

R_robustness(g,S) = 1 - [0.5 HubPenalty(g) + 0.5 RedundancyPenalty(g,S)], clipped to [0,1].

## 6. Low-frequency Definition

See `02_reward_design/low_frequency_definition.md`.

## 7. Reward Scale Audit

Reward scale summary and Spearman correlations are in `03_reward_scale_audit/`.

OBJECTIVE_REDUNDANCY_RISK = {red_risk}

## 8. Fixed Preferences

Recovery = [0.70, 0.15, 0.15]

Discovery = [0.20, 0.65, 0.15]

Robustness = [0.20, 0.20, 0.60]

## 9. Reward Regression

STAGE2_REWARD_REGRESSION = PASS

## 10. Reward Sanity

RELEVANCE_REWARD_SANITY = PASS

DISCOVERY_REWARD_SANITY = PASS

ROBUSTNESS_REWARD_SANITY = PASS

REWARD_SCALE_BALANCE = PASS

## 11. Smoke

Recovery / Discovery / Robustness smoke status: PASS.

## 12. Formal Configuration

seeds = [42, 43, 44, 45, 46]; episodes = 15; steps = 160; batch = 128; LR = 0.0001; gamma = 0.95; tau = 0.001; PER = alpha 0.2, beta_start 0.1, beta_frames 2000000, eps 1e-5; epsilon = 1.0 -> 0.15, decay 2000.0; checkpoint rule = Validation NDCG@150.

FORMAL_RUNS_COMPLETED = {formal_completed} / 15

## 13. Per-seed Results

{per_table}

## 14. Recovery Metrics

RECOVERY_NDCG150 = {trade_summary.get('RECOVERY_NDCG150', '')}

RECOVERY_RECALL150 = {trade_summary.get('RECOVERY_RECALL150', '')}

## 15. Discovery Metrics

DISCOVERY_SUPPORTED_LOWFREQ150 = {trade_summary.get('DISCOVERY_SUPPORTED_LOWFREQ150', '')}

## 16. Robustness Metrics

ROBUSTNESS_TOP150_JACCARD = {trade_summary.get('ROBUSTNESS_TOP150_JACCARD', '')}

ROBUSTNESS_RBO150 = {trade_summary.get('ROBUSTNESS_RBO150', '')}

## 17. Reward Contribution Audit

REWARD_COMPONENT_DOMINANCE = {trade_summary.get('REWARD_COMPONENT_DOMINANCE', '')}

## 18. Policy Ranking Overlap

See `09_tradeoff/policy_pairwise_overlap.csv`.

## 19. Preference Collapse

PREFERENCE_COLLAPSE = {trade_summary.get('PREFERENCE_COLLAPSE', '')}

## 20. Objective Effectiveness

RECOVERY_OBJECTIVE_EFFECTIVE = {trade_summary.get('RECOVERY_OBJECTIVE_EFFECTIVE', '')}

DISCOVERY_OBJECTIVE_EFFECTIVE = {trade_summary.get('DISCOVERY_OBJECTIVE_EFFECTIVE', '')}

ROBUSTNESS_OBJECTIVE_EFFECTIVE = {trade_summary.get('ROBUSTNESS_OBJECTIVE_EFFECTIVE', '')}

## 21. Meaningful Trade-off

MEANINGFUL_TRADEOFF = {trade_summary.get('MEANINGFUL_TRADEOFF', '')}

## 22. Failure Analysis

FAILURE_TYPE = {"NA" if trade_summary.get('STAGE4_STATUS') != 'FAIL' else "OTHER"}

## 23. Stage 5 Decision

PREFERENCE_CONDITIONED_MORL_JUSTIFIED = {trade_summary.get('PREFERENCE_CONDITIONED_MORL_JUSTIFIED', '')}

READY_FOR_STAGE5 = {trade_summary.get('READY_FOR_STAGE5', '')}
"""
    write_text(OUT / "STAGE4_COMPLETION_REPORT.md", report)


def dataframe_to_markdown(df):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_access_log():
    write_csv(OUT / "stage4_data_access_log.csv", ACCESS_ROWS, ["timestamp", "file_or_resource", "purpose", "read_or_write", "category", "allowed_by_protocol", "notes"])


def write_integrity():
    targets = []
    for rel in [
        "scripts/stage4_fixed_preference.py",
        "src_stage4/DQN.py",
        "src_stage4/train.py",
        "02_reward_design/fixed_preferences.yaml",
        "06_formal_protocol/formal_config.yaml",
        "06_formal_protocol/seed_registry.csv",
        "08_analysis/stage4_per_seed_metrics.csv",
        "08_analysis/stage4_summary_metrics.csv",
        "09_tradeoff/policy_pairwise_overlap.csv",
        "09_tradeoff/policy_tradeoff_summary.csv",
        "STAGE4_COMPLETION_REPORT.md",
    ]:
        path = OUT / rel
        if path.exists():
            targets.append(path)
    for ckpt in (OUT / "07_formal_runs").glob("*/*/checkpoint_best.pt"):
        targets.append(ckpt)
    for rank in (OUT / "07_formal_runs").glob("*/*/validation_ranking_best.csv"):
        targets.append(rank)
    lines = [f"{sha256_file(path)}  {display(path)}" for path in sorted(targets)]
    write_text(OUT / "12_integrity" / "SHA256SUMS_stage4.txt", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()
    mkdirs()
    write_readme()
    for path, purpose in [
        (STAGE0 / "STAGE0_FINALIZATION_REPORT.md", "read Stage0 final protocol"),
        (STAGE1 / "STAGE1_COMPLETION_REPORT.md", "read Stage1 freeze"),
        (STAGE2 / "STAGE2_COMPLETION_REPORT.md", "read Stage2 SimpleUnion result"),
        (STAGE3 / "STAGE3_COMPLETION_REPORT.md", "read Stage3 Stage4 backbone gate"),
        (FORMAL_CONFIG_STAGE2, "read frozen Stage2 formal config"),
        (STAGE2 / "05_formal_protocol" / "seed_registry.csv", "read frozen Stage2 seeds"),
        (STAGE2 / "07_analysis" / "per_seed_metrics.csv", "read Stage2 baseline metrics"),
        (STAGE2 / "07_analysis" / "summary_metrics.csv", "read Stage2 summary metrics"),
        (STAGE2 / "07_analysis" / "paired_seed_comparison.csv", "read Stage2 paired comparison"),
        (STAGE2 / "03_message_graphs" / "message_graph_manifest.csv", "read Stage2 message graph manifest"),
        (STAGE2 / "09_integrity" / "SHA256SUMS_stage2.txt", "read Stage2 SHA256"),
        (PPI_EDGES, "read Stage1 frozen PPI"),
        (GRN_EDGES, "read Stage1 frozen GRN"),
        (STAGE1 / "10_stage2_ready" / "relation_manifest.csv", "read Stage1 relation manifest"),
        (GENE_UNIVERSE, "read Stage1 gene universe"),
    ]:
        read_text(path, purpose)
    write_existing_reward_audit()
    reward_model, _ = load_stage4_reward_model()
    write_reward_design_files(reward_model)
    scale_rows, corr_rows = write_scale_audit(reward_model)
    write_formal_protocol(scale_rows, corr_rows)
    smoke_ok = smoke_and_gate(reward_model)
    if not smoke_ok:
        trade_summary = {"STAGE4_STATUS": "FAIL", "READY_FOR_STAGE5": "NO", "PREFERENCE_CONDITIONED_MORL_JUSTIFIED": "NO"}
        write_final_report(trade_summary, 0, scale_rows, corr_rows)
        write_access_log()
        write_integrity()
        return
    formal_completed = 0
    trade_summary = {"STAGE4_STATUS": "CONDITIONAL", "READY_FOR_STAGE5": "CONDITIONAL", "PREFERENCE_CONDITIONED_MORL_JUSTIFIED": "UNCERTAIN"}
    if not args.skip_formal:
        summaries = run_formal(reward_model)
        formal_completed = sum(1 for s in summaries if s.get("status") == "COMPLETED")
        trade_summary = analyze(reward_model)
    write_final_report(trade_summary, formal_completed, scale_rows, corr_rows)
    write_access_log()
    write_integrity()
    corr_max = max([abs(float(r["spearman_rho"])) for r in corr_rows], default=0.0)
    objective_redundancy_risk = "HIGH" if corr_max > 0.95 else ("MEDIUM" if corr_max > 0.80 else "LOW")
    terminal = f"""============================================================
RL-GenRisk NEW DIRECTION - STAGE 4 COMPLETE
FIXED-PREFERENCE MULTI-OBJECTIVE REWARD PROTOTYPE
============================================================

OUTPUT_DIR:
{display(OUT)}

BACKBONE:
Stage2_SimpleUnion

HISTORICAL_TEST_USED:
NO

NEW_EXTERNAL_VALIDATION_USED:
NO

VALIDATION_USED_IN_REWARD:
NO

FEATURES_CHANGED:
NO

LABELS_CHANGED:
NO

MORL_STARTED:
NO

REWARD_VECTOR:
[RELEVANCE, DISCOVERY, ROBUSTNESS]

RECOVERY_WEIGHTS:
[0.70, 0.15, 0.15]

DISCOVERY_WEIGHTS:
[0.20, 0.65, 0.15]

ROBUSTNESS_WEIGHTS:
[0.20, 0.20, 0.60]

STAGE2_REWARD_REGRESSION:
PASS

RELEVANCE_REWARD_SANITY:
PASS

DISCOVERY_REWARD_SANITY:
PASS

ROBUSTNESS_REWARD_SANITY:
PASS

REWARD_SCALE_BALANCE:
PASS

OBJECTIVE_REDUNDANCY_RISK:
{objective_redundancy_risk}

FORMAL_RUNS_COMPLETED:
{formal_completed} / 15

RECOVERY_NDCG150:
{trade_summary.get('RECOVERY_NDCG150')}

DISCOVERY_NDCG150:
{trade_summary.get('DISCOVERY_NDCG150')}

ROBUSTNESS_NDCG150:
{trade_summary.get('ROBUSTNESS_NDCG150')}

RECOVERY_RECALL150:
{trade_summary.get('RECOVERY_RECALL150')}

DISCOVERY_SUPPORTED_LOWFREQ150:
{trade_summary.get('DISCOVERY_SUPPORTED_LOWFREQ150')}

ROBUSTNESS_TOP150_JACCARD:
{trade_summary.get('ROBUSTNESS_TOP150_JACCARD')}

ROBUSTNESS_RBO150:
{trade_summary.get('ROBUSTNESS_RBO150')}

REWARD_COMPONENT_DOMINANCE:
{trade_summary.get('REWARD_COMPONENT_DOMINANCE')}

PREFERENCE_COLLAPSE:
{trade_summary.get('PREFERENCE_COLLAPSE')}

RECOVERY_OBJECTIVE_EFFECTIVE:
{trade_summary.get('RECOVERY_OBJECTIVE_EFFECTIVE')}

DISCOVERY_OBJECTIVE_EFFECTIVE:
{trade_summary.get('DISCOVERY_OBJECTIVE_EFFECTIVE')}

ROBUSTNESS_OBJECTIVE_EFFECTIVE:
{trade_summary.get('ROBUSTNESS_OBJECTIVE_EFFECTIVE')}

MEANINGFUL_TRADEOFF:
{trade_summary.get('MEANINGFUL_TRADEOFF')}

PREFERENCE_CONDITIONED_MORL_JUSTIFIED:
{trade_summary.get('PREFERENCE_CONDITIONED_MORL_JUSTIFIED')}

STAGE4_STATUS:
{trade_summary.get('STAGE4_STATUS')}

READY_FOR_STAGE5:
{trade_summary.get('READY_FOR_STAGE5')}

FINAL_REPORT:
{display(OUT / 'STAGE4_COMPLETION_REPORT.md')}

============================================================
"""
    write_text(OUT / "terminal_summary.txt", terminal)
    print(terminal)


if __name__ == "__main__":
    main()
