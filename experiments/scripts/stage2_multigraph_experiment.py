import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


if os.name == "nt":
    PROJECT = Path(r"E:\Projects\RL-GenRisk-main")
    STAGE0 = Path(r"E:\codex_file\新方向阶段0")
    STAGE1 = Path(r"E:\codex_file\新方向阶段1")
    STAGE1_READY = STAGE1 / "10_stage2_ready"
    OUT = Path(r"E:\codex_file\新方向阶段2")
    PROTOCOL_B = Path(r"E:\codex_file\一阶段\driver_label_protocol\protocol_B")
else:
    PROJECT = Path("/mnt/e/Projects/RL-GenRisk-main")
    STAGE0 = Path("/mnt/e/codex_file/新方向阶段0")
    STAGE1 = Path("/mnt/e/codex_file/新方向阶段1")
    STAGE1_READY = STAGE1 / "10_stage2_ready"
    OUT = Path("/mnt/e/codex_file/新方向阶段2")
    PROTOCOL_B = Path("/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B")

SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train  # noqa: E402


CONDITIONS = {
    "PPI": "PPI-only",
    "GRN": "GRN-message model under fixed original action topology",
    "PPI_GRN": "PPI+GRN edge-union message model under fixed original action topology",
}
FORMAL_SEEDS = [42, 43, 44, 45, 46]
PRIMARY_ENDPOINT = "Validation NDCG@150"
FEATURE_MODE = "hybrid6_raw"
K_VALUES = [50, 100, 150]
DATA_ACCESS = []


def ts():
    return datetime.now(timezone.utc).isoformat()


def current_path(value):
    s = str(value)
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":\\":
        return Path(f"/mnt/{s[0].lower()}/" + s[3:].replace("\\", "/"))
    return Path(s)


def display_path(value):
    s = str(value)
    if os.name != "nt" and s.startswith("/mnt/") and len(s) > 7 and s[6] == "/":
        return f"{s[5].upper()}:\\" + s[7:].replace("/", "\\")
    return s


def mkdirs():
    for name in [
        "01_action_topology_audit",
        "02_stage2_implementation",
        "03_message_graphs",
        "04_smoke_test/PPI",
        "04_smoke_test/GRN",
        "04_smoke_test/PPI_GRN",
        "05_formal_protocol",
        "06_formal_runs/PPI",
        "06_formal_runs/GRN",
        "06_formal_runs/PPI_GRN",
        "07_analysis",
        "08_figures",
        "09_integrity",
        "scripts",
        "src_stage2",
        "logs",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def log_access(path, purpose, rw, category, allowed=True, notes=""):
    DATA_ACCESS.append(
        {
            "timestamp": ts(),
            "file": display_path(path),
            "purpose": purpose,
            "read_write": rw,
            "category": category,
            "protocol_allowed": "YES" if allowed else "NO",
            "notes": notes,
        }
    )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log_access(path, "write Stage 2 output", "write", "stage2_output")


def read_text(path, purpose="read frozen evidence"):
    log_access(path, purpose, "read", "stage0_stage1_frozen")
    return Path(path).read_text(encoding="utf-8")


def read_csv(path, delimiter=",", purpose="read frozen evidence"):
    log_access(path, purpose, "read", "stage0_stage1_or_project")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, rows, fieldnames=None, delimiter=","):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log_access(path, "write Stage 2 output", "write", "stage2_output")


def append_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    log_access(path, "append Stage 2 run log", "write", "stage2_output")


def load_frozen_inputs():
    required = [
        STAGE0 / "STAGE0_FINALIZATION_REPORT.md",
        STAGE1 / "STAGE1_COMPLETION_REPORT.md",
        STAGE1_READY / "gene_universe.tsv",
        STAGE1_READY / "ppi_edges_frozen.tsv",
        STAGE1_READY / "grn_edges_frozen.tsv",
        STAGE1_READY / "relation_manifest.csv",
        STAGE1 / "09_stage2_input_freeze" / "fixed_feature_manifest.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    stage0_report = read_text(required[0])
    stage1_report = read_text(required[1])
    if "STAGE0_FINAL_STATUS = PASS" not in stage0_report:
        raise RuntimeError("Stage 0 final status is not PASS.")
    if "READY_FOR_STAGE2 = YES" not in stage1_report:
        raise RuntimeError("Stage 1 does not declare READY_FOR_STAGE2 = YES.")
    gene_rows = read_csv(required[2], delimiter="\t")
    ppi_rows = read_csv(required[3], delimiter="\t")
    grn_rows = read_csv(required[4], delimiter="\t")
    relation_manifest = read_csv(required[5])
    feature_manifest = read_csv(required[6])
    genes = [r["gene_symbol"] for r in sorted(gene_rows, key=lambda x: int(x["node_index"]))]
    if len(genes) != 9039 or len(set(genes)) != 9039:
        raise RuntimeError("Frozen gene universe is not 9039 unique genes.")
    return {
        "stage0_report": stage0_report,
        "stage1_report": stage1_report,
        "genes": genes,
        "ppi_rows": ppi_rows,
        "grn_rows": grn_rows,
        "relation_manifest": relation_manifest,
        "feature_manifest": feature_manifest,
    }


def stage2a_action_audit():
    dqn = SRC / "DQN.py"
    train_py = SRC / "train.py"
    qfun = SRC / "qfunction.py"
    inputall = SRC / "inputall.py"
    for path in [dqn, train_py, qfun, inputall]:
        log_access(path, "read current RL source for action topology audit", "read", "project_source")

    audit = f"""# Stage 2A Action / Transition Fairness Audit

## Gate Result

ACTION_FAIRNESS = PASS

ACTION_SPACE_TYPE = GLOBAL_UNSELECTED_GENE_POOL

MESSAGE_GRAPH_AND_ACTION_GRAPH_COUPLED = NO in the active `src/train.py` training path.

FAIRNESS_FIX_REQUIRED = YES for message graph injection, because `inputall.getNetwork()` symmetrizes every edge file and `hybrid6_raw` original features are built from the same loaded PPI matrix. Directly passing GRN through `--ppi_path` would change features and directionality.

FAIRNESS_FIX_APPLIED = YES in isolated Stage 2 code: build the original PPI environment first, then replace only the dense matrix passed to `Q_Fun`/`GCNConv` as the message graph.

## Required Questions

Q1. Action is selected from all currently unselected genes, not only graph neighbors, in the active training loop. At [train.py]({display_path(train_py)}:1006), `action_sel = list(range(agent.n_actions))`; at [train.py]({display_path(train_py)}:1039), valid actions are filtered only by the current unselected mask.

Q2. PPI adjacency does not directly participate in active candidate action generation. It is used by the Q encoder, while candidates come from `action_sel` and `actions_index`.

Q3. In the active path, PPI is used for GCN message passing through `Q_Fun`. [DQN.py]({display_path(dqn)}:117) and [DQN.py]({display_path(dqn)}:118) instantiate online/target Q networks with `self.net_ori`; [qfunction.py]({display_path(qfun)}:45) converts nonzero adjacency entries to `edge_index`; [qfunction.py]({display_path(qfun)}:56) and [qfunction.py]({display_path(qfun)}:60) call `GCNConv`.

Q4. State transition does not depend on current node -> neighboring node in the active loop. Each step removes the chosen global action from `action_sel` at [train.py]({display_path(train_py)}:1053) and marks it unavailable at [train.py]({display_path(train_py)}:1055).

Q5. The active action mask only prevents repeated selection. At [train.py]({display_path(train_py)}:1025) the mask is `agent.actions_index`; at [train.py]({display_path(train_py)}:1055), only the selected action index is set to 0. No network-neighbor condition is applied.

Q6. If one naively replaced `--ppi_path` with GRN, it would change message passing, edge directionality, and fixed original-degree features because [inputall.py]({display_path(inputall)}:903)-[inputall.py]({display_path(inputall)}:904) symmetrize every loaded edge, and [inputall.py]({display_path(inputall)}:619) builds original features from that network. It would not change active action candidate generation, but it would violate Stage 2 feature and GRN-direction rules. The Stage 2 wrapper avoids this by using original PPI for environment/features and replacing only message graph input.

## Definitions

STATE_DEFINITION = full fixed node-feature matrix `(9039, 6)` with `hybrid6_raw` columns.

ACTION_DEFINITION = an integer gene index in the 9039-gene frozen universe.

ACTION_MASK_DEFINITION = 1 means unselected/available, 0 means already selected.

CANDIDATE_GENERATION = global `list(range(agent.n_actions))`, then remove selected genes.

TRANSITION_DEFINITION = update selected action list and unselected mask; terminate by selection budget or max-step truncation.

MESSAGE_PASSING_GRAPH = dense adjacency converted to PyG `edge_index` inside `Q_Fun`; Stage 2 varies only this graph.

ACTION_TOPOLOGY = original active global unselected-gene topology, fixed across PPI, GRN, and PPI+GRN.
"""
    trace = f"""# Action Graph Code Trace

## Active Training Loop

- [train.py]({display_path(train_py)}:1003): `run_episode(...)` defines the active environment transition loop.
- [train.py]({display_path(train_py)}:1006): `action_sel = list(range(agent.n_actions))`.
- [train.py]({display_path(train_py)}:1025): current mask is copied from `agent.actions_index`.
- [train.py]({display_path(train_py)}:1039): `valid_actions = [idx for idx in action_sel if current_action_mask[idx] == 1]`.
- [train.py]({display_path(train_py)}:1053): selected action is removed from `action_sel`.
- [train.py]({display_path(train_py)}:1055): selected action mask is set to zero.
- [train.py]({display_path(train_py)}:1108)-[train.py]({display_path(train_py)}:1117): replay stores state, action, reward, pre-action mask, post-action mask, and done flag.

## Inactive / Legacy Neighbor Helper

- [DQN.py]({display_path(dqn)}:389)-[DQN.py]({display_path(dqn)}:406): `getAction()` expands candidates from `self.net_ori[i][j]`, but active `train.py` explicitly does not call it.
- [train.py]({display_path(train_py)}:1003)-[train.py]({display_path(train_py)}:1006): code comment says the training entry keeps a global action space and avoids `DQN.getAction()`.

## Message Passing

- [DQN.py]({display_path(dqn)}:117)-[DQN.py]({display_path(dqn)}:118): online and target Q networks receive the adjacency matrix.
- [qfunction.py]({display_path(qfun)}:45)-[qfunction.py]({display_path(qfun)}:46): nonzero matrix entries are frozen as `edge_index` buffer.
- [qfunction.py]({display_path(qfun)}:56)-[qfunction.py]({display_path(qfun)}:60): two `GCNConv` calls use that message graph.

## Loader Risk

- [inputall.py]({display_path(inputall)}:903)-[inputall.py]({display_path(inputall)}:904): any network loaded through `getNetwork()` is made bidirectional.
- [inputall.py]({display_path(inputall)}:619)-[inputall.py]({display_path(inputall)}:661): `hybrid6_raw` includes original PPI-degree/weight/patient features plus omics, so Stage 2 must not build features from GRN.
"""
    write_text(OUT / "01_action_topology_audit" / "action_topology_audit.md", audit)
    write_text(OUT / "01_action_topology_audit" / "action_graph_code_trace.md", trace)
    return {"ACTION_FAIRNESS": "PASS", "ACTION_SPACE_TYPE": "GLOBAL_UNSELECTED_GENE_POOL", "MESSAGE_ACTION_GRAPH_COUPLED": "NO"}


def pyg_runtime_smoke():
    result = {
        "PYG_RUNTIME": "FAIL",
        "GCN_FORWARD": "FAIL",
        "BACKWARD": "FAIL",
        "OPTIMIZER_STEP_SMOKE": "FAIL",
        "CUDA": "FAIL",
        "PYG_LIB_WARNING": "UNKNOWN",
    }
    warn_messages = []
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        import torch_geometric
        from torch_geometric.nn import GCNConv
        warn_messages = [str(w.message) for w in seen]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result["torch"] = torch.__version__
    result["torch_geometric"] = torch_geometric.__version__
    result["CUDA"] = "PASS" if torch.cuda.is_available() and device.type == "cuda" else "FAIL"
    x = torch.randn(5, 3, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 2, 3, 4, 0]], dtype=torch.long, device=device)
    model = GCNConv(3, 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    opt.zero_grad()
    out = model(x, edge_index)
    if torch.isfinite(out).all() and tuple(out.shape) == (5, 4):
        result["GCN_FORWARD"] = "PASS"
    loss = out.pow(2).mean()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    if grad_ok:
        result["BACKWARD"] = "PASS"
    opt.step()
    result["OPTIMIZER_STEP_SMOKE"] = "PASS"
    result["loss"] = float(loss.detach().cpu())
    warning_text = "\n".join(warn_messages)
    try:
        import pyg_lib  # noqa: F401
        result["pyg_lib_import"] = "PASS"
    except Exception as exc:
        result["pyg_lib_import"] = f"FAIL: {type(exc).__name__}: {exc}"
    result["warnings"] = warn_messages
    if "pyg" in result["pyg_lib_import"].lower() or "pyg-lib" in warning_text.lower():
        result["PYG_LIB_WARNING"] = "ACCEPTED_NON_BLOCKING"
    else:
        result["PYG_LIB_WARNING"] = "NONE_OR_NOT_OBSERVED"
    if all(result[k] == "PASS" for k in ["GCN_FORWARD", "BACKWARD", "OPTIMIZER_STEP_SMOKE", "CUDA"]):
        result["PYG_RUNTIME"] = "PASS"
    write_text(OUT / "02_stage2_implementation" / "pyg_runtime_smoke.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def load_edge_indices(rows):
    edges = []
    for r in rows:
        edges.append((int(r["source_index"]), int(r["target_index"]), r["source"], r["target"]))
    return edges


def generate_message_graphs(frozen):
    genes = frozen["genes"]
    index_to_gene = {idx: gene for idx, gene in enumerate(genes)}
    ppi = load_edge_indices(frozen["ppi_rows"])
    grn = load_edge_indices(frozen["grn_rows"])
    ppi_directed = set()
    for s, t, _, _ in ppi:
        ppi_directed.add((s, t))
        ppi_directed.add((t, s))
    grn_directed = {(s, t) for s, t, _, _ in grn}
    union = sorted(ppi_directed | grn_directed)
    graph_defs = {
        "ppi_message_edges.tsv": (sorted(ppi_directed), "PPI", "false_as_bidirectional_for_GCN"),
        "grn_message_edges.tsv": (sorted(grn_directed), "GRN", "true_TF_to_target"),
        "ppi_grn_union_message_edges.tsv": (union, "PPI_GRN", "PPI_bidirectional_union_GRN_directed"),
    }
    manifest = []
    for filename, (edges, relation_name, direction_rule) in graph_defs.items():
        rows = [
            {
                "source": index_to_gene[s],
                "target": index_to_gene[t],
                "source_index": s,
                "target_index": t,
                "relation_name": relation_name,
                "direction_rule": direction_rule,
            }
            for s, t in edges
        ]
        path = OUT / "03_message_graphs" / filename
        write_csv(path, rows, delimiter="\t")
        manifest.append(
            {
                "condition": relation_name,
                "file": display_path(path),
                "edge_count_directed_for_pyg": len(edges),
                "node_count": len(genes),
                "sha256": sha256_file(path),
                "ppi_bidirectional": "YES" if relation_name in {"PPI", "PPI_GRN"} else "NO",
                "grn_directed": "YES" if relation_name in {"GRN", "PPI_GRN"} else "NO",
                "duplicate_rule": "exact directed duplicate edges removed",
                "self_loop_rule": "not written here; PyG GCNConv default add_self_loops applies equally",
            }
        )
    write_csv(OUT / "03_message_graphs" / "message_graph_manifest.csv", manifest)
    prereg = """# Message Graph Preregistration

PPI_MESSAGE_GRAPH = frozen Stage 1 PPI edges expanded to both directions, matching the current dense symmetric PPI baseline behavior.

GRN_MESSAGE_GRAPH = frozen Stage 1 DoRothEA A+B TF -> target directed edges, direction preserved.

PPI_GRN_UNION = exact directed edge union of PPI bidirectional edges and GRN directed edges.

Duplicate handling = exact `(source_index, target_index)` duplicates removed.

Overlapping PPI/GRN pairs = PPI contributes both directions; GRN contributes its TF -> target direction. If that directed edge already exists from PPI expansion, it is not duplicated.

Self-loops = not serialized in message edge TSVs; `torch_geometric.nn.GCNConv` default self-loop behavior is allowed to apply equally to all three models.

Relation-aware parameters = NO. Stage 2 uses the same `Q_Fun`/GCN architecture and varies only `message_edge_index`.
"""
    write_text(OUT / "03_message_graphs" / "MESSAGE_GRAPH_PREREGISTRATION.md", prereg)
    return {row["condition"]: current_path(row["file"]) for row in manifest}


def matrix_from_message_edges(path, node_count=9039):
    rows = read_csv(path, delimiter="\t", purpose="read Stage 2 frozen message graph")
    mat = np.zeros((node_count, node_count), dtype=np.float32)
    for r in rows:
        mat[int(r["source_index"]), int(r["target_index"])] = 1.0
    return mat, len(rows)


def make_args(seed, run_dir, smoke=False):
    return SimpleNamespace(
        original_feature_path=None,
        feature_mode=FEATURE_MODE,
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


ACTION_TRACE_FIELDS = [
    "episode",
    "step",
    "candidate_count",
    "action_index",
    "Gene",
    "q_min",
    "q_max",
    "q_selected",
    "reward",
    "done",
    "terminal_reason",
    "epsilon",
    "invalid_action",
    "duplicate_action",
    "learn_step_after",
    "loss",
    "td_error_abs_mean",
    "gradient_norm",
]


def run_episode_stage2(agent, env, args, episode, run_dir):
    action_sel = list(range(agent.n_actions))
    agent.actions = []
    agent.actions_index = np.ones(agent.n_actions, dtype=np.int64)
    agent.score_be = 0
    agent.score_sta = 0
    agent.score_pat = 0
    agent.embedding = None
    episode_reward = 0.0
    step_count = 0
    terminal_reason = "unknown"
    loss_values = []
    td_values = []
    q_min_values = []
    q_max_values = []
    invalid_count = 0
    duplicate_count = 0
    dead_end_count = 0
    replay_write_count = 0
    per_update_count = 0
    optimizer_step_count = 0
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
        candidate_count = len(valid_actions)
        if not valid_actions:
            terminal_reason = "no_legal_action"
            dead_end_count += 1
            break
        best_action = max(valid_actions, key=lambda idx: q_np[idx])
        if np.random.uniform() >= agent.epsilon:
            action_index = int(best_action)
        else:
            action_index = int(random.choice(valid_actions))
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
        loss = ""
        td = ""
        grad = ""
        if learn_metrics:
            loss = float(learn_metrics.get("loss", 0.0))
            td = float(learn_metrics.get("td_error_abs_mean", 0.0))
            grad = float(learn_metrics.get("gradient_norm", 0.0))
            loss_values.append(loss)
            td_values.append(td)
        trace_rows.append(
            {
                "episode": episode,
                "step": step_count + 1,
                "candidate_count": candidate_count,
                "action_index": action_index,
                "Gene": env["gene_name"][action_index],
                "q_min": float(np.min(q_np)),
                "q_max": float(np.max(q_np)),
                "q_selected": float(q_np[action_index]),
                "reward": float(reward),
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
    append_csv(run_dir / "action_trace.csv", trace_rows, ACTION_TRACE_FIELDS)
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
        "candidate_count_min": int(min([r["candidate_count"] for r in trace_rows], default=0)),
        "candidate_count_max": int(max([r["candidate_count"] for r in trace_rows], default=0)),
        "q_min": float(min(q_min_values)) if q_min_values else "",
        "q_max": float(max(q_max_values)) if q_max_values else "",
        "val_ndcg_150": "",
        "val_precision_k": "",
        "val_recall_k": "",
        "elapsed_seconds": "",
    }


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_run(condition, seed, run_dir, message_edge_path, smoke=False):
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if run_dir.exists():
        resolved = run_dir.resolve()
        if OUT.resolve() not in [resolved, *resolved.parents]:
            raise RuntimeError(f"Refusing to clean non-Stage2 run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(seed, run_dir, smoke=smoke)
    train.validate_training_args(args)
    train.validate_no_test_path(args)
    start_time = time.perf_counter()
    train.set_seed(args.seed)
    device = train.choose_device(args.device)
    log_access(args.train_label_path, "Protocol B Train labels for training/reward", "read", "protocol_b_train", True)
    log_access(args.val_label_path, "Protocol B Validation labels for checkpoint/model selection", "read", "protocol_b_validation", True)
    log_access(args.ppi_path, "original PPI action/feature environment", "read", "project_frozen_ppi", True)
    log_access(args.multiomics_feature_path, "fixed hybrid6_raw multiomics features", "read", "project_fixed_features", True)
    env = train.build_environment(args, run_dir, normalization_metadata=None)
    message_net, message_edge_count = matrix_from_message_edges(message_edge_path)
    env["net"] = message_net
    env["stage2_message_condition"] = condition
    env["stage2_message_edge_count"] = message_edge_count
    agent = train.build_agent(args, env, device)
    config = {
        "stage": "Stage 2",
        "condition": condition,
        "condition_definition": CONDITIONS[condition],
        "seed": seed,
        "args": vars(args),
        "run_dir": display_path(run_dir),
        "feature_mode": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "message_edge_file": display_path(message_edge_path),
        "message_edge_count_directed_for_pyg": message_edge_count,
        "action_topology": "GLOBAL_UNSELECTED_GENE_POOL",
        "action_graph_reference": "original active train.py global action behavior",
        "features_built_from": "original frozen PPI environment before message graph swap",
        "device": str(device),
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "historical_test_read": False,
        "external_validation_read": False,
        "reward_modified": False,
        "morl_started": False,
        "primary_endpoint": PRIMARY_ENDPOINT,
    }
    write_json(run_dir / "config.json", config)
    best_val_ndcg150 = float("-inf")
    best_episode = None
    rows = []
    with (run_dir / "run_log.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(json.dumps({"event": "run_start", "condition": condition, "seed": seed, "smoke": smoke}, ensure_ascii=False))
        for episode in range(1, args.max_episodes + 1):
            episode_start = time.perf_counter()
            metrics = run_episode_stage2(agent, env, args, episode, run_dir)
            metrics["elapsed_seconds"] = time.perf_counter() - episode_start
            if args.val_interval > 0 and episode % args.val_interval == 0:
                val_metrics, ranking_path = train.evaluate_validation(agent, env, args, run_dir, episode)
                metrics["val_ndcg_150"] = val_metrics.get("NDCG@150", "")
                metrics["val_precision_k"] = val_metrics.get(f"Precision@{args.topk}", "")
                metrics["val_recall_k"] = val_metrics.get(f"Recall@{args.topk}", "")
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
        "condition": condition,
        "seed": seed,
        "run_dir": display_path(run_dir),
        "feature_mode": args.feature_mode,
        "node_features_shape": list(env["node_features"].shape),
        "message_edge_count_directed_for_pyg": message_edge_count,
        "best_episode": best_episode,
        "best_val_ndcg150": best_val_ndcg150 if best_val_ndcg150 != float("-inf") else None,
        "checkpoint_best": display_path(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else None,
        "checkpoint_last": display_path(run_dir / "checkpoint_last.pt"),
        "validation_ranking_best": display_path(run_dir / "validation_ranking_best.csv") if (run_dir / "validation_ranking_best.csv").exists() else None,
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


def run_smoke(message_paths):
    summaries = []
    for condition in CONDITIONS:
        summary = build_run(condition, 42, OUT / "04_smoke_test" / condition / "seed_42_smoke", message_paths[condition], smoke=True)
        summaries.append(summary)
    candidate_sequences = {}
    for condition in CONDITIONS:
        rows = read_csv(OUT / "04_smoke_test" / condition / "seed_42_smoke" / "action_trace.csv", purpose="read Stage 2 smoke action trace")
        candidate_sequences[condition] = [int(r["candidate_count"]) for r in rows]
    fairness = "PASS" if len({tuple(v) for v in candidate_sequences.values()}) == 1 else "FAIL"
    checks = {}
    for summary in summaries:
        c = summary["condition"]
        run_dir = current_path(summary["run_dir"])
        best_metrics = json.loads((run_dir / "validation_metrics_best.json").read_text(encoding="utf-8"))
        ranking = read_csv(run_dir / "validation_ranking_best.csv", purpose="read smoke ranking")
        q_values = [float(r["Q_value"]) for r in ranking]
        checks[c] = {
            "startup": "PASS",
            "training_forward": "PASS",
            "optimizer_step": "PASS" if summary["optimizer_step_count"] > 0 else "FAIL",
            "replay_write": "PASS" if summary["replay_write_count"] > 0 else "FAIL",
            "PER_update": "PASS" if summary["per_update_count"] > 0 else "FAIL",
            "checkpoint_save": "PASS" if summary["checkpoint_best"] else "FAIL",
            "validation_inference": "PASS" if best_metrics else "FAIL",
            "ranking_generation": "PASS" if len(ranking) == 9039 and len({r["Gene"] for r in ranking}) == 9039 else "FAIL",
            "NaN": int(np.isnan(np.asarray(q_values)).sum()),
            "Inf": int(np.isinf(np.asarray(q_values)).sum()),
            "node_count": len(ranking),
            "candidate_count_distribution": {
                "min": min(candidate_sequences[c]),
                "max": max(candidate_sequences[c]),
                "sequence": candidate_sequences[c],
            },
            "episode_termination_reason": read_csv(run_dir / "train_metrics.csv", purpose="read smoke metrics")[0]["terminal_reason"],
            "dead_end_count": summary["dead_end_count"],
            "invalid_action_count": summary["invalid_action_count"],
        }
    smoke_summary = {
        "ACTION_FAIRNESS": fairness,
        "PYG_RUNTIME": "PASS",
        "smoke_training_note": "1 seed, 1 episode, 8 max steps, reduced warmup/batch for optimizer smoke only; not formal evidence.",
        "condition_checks": checks,
        "historical_test_read": False,
        "external_validation_read": False,
    }
    write_json(OUT / "04_smoke_test" / "smoke_summary.json", smoke_summary)
    return smoke_summary


def write_formal_protocol():
    seed_rows = [
        {
            "seed": seed,
            "source": "Stage 0 baseline_registry formal seeds 42-46",
            "included": "YES",
            "change_after_start_allowed": "NO",
        }
        for seed in FORMAL_SEEDS
    ]
    write_csv(OUT / "05_formal_protocol" / "seed_registry.csv", seed_rows)
    config = {
        "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT,
        "feature_mode": FEATURE_MODE,
        "message_network_conditions": list(CONDITIONS),
        "seeds": FORMAL_SEEDS,
        "max_episodes": 15,
        "max_steps": 160,
        "selection_budget": 150,
        "batch_size": 128,
        "warmup_steps": 128,
        "buffer_size": 2048,
        "learning_rate": 0.0001,
        "gamma": 0.95,
        "tau": 0.001,
        "epsilon_start": 1.0,
        "epsilon_end": 0.15,
        "epsilon_decay": 2000.0,
        "PER_alpha": 0.2,
        "PER_beta_start": 0.1,
        "PER_beta_frames": 2000000,
        "PER_eps": 1e-5,
        "target_update": "soft update every learn step, tau=0.001",
        "checkpoint_interval": "every episode",
        "checkpoint_selection_metric": "Validation NDCG@150",
        "historical_test_policy": "not read, not evaluated, not used for model/network/checkpoint selection",
        "low_frequency_dev_eval": "DEFERRED",
    }
    lines = ["# formal_config.yaml"]
    for key, value in config.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    write_text(OUT / "05_formal_protocol" / "formal_config.yaml", "\n".join(lines) + "\n")
    protocol = f"""# STAGE2_FORMAL_PROTOCOL

PRIMARY_ENDPOINT = {PRIMARY_ENDPOINT}

Formal matrix = 3 network conditions x 5 seeds = 15 runs.

Conditions:

- PPI-only: message passing = frozen PPI expanded bidirectionally; action topology = active original global unselected-gene pool.
- GRN-only: message passing = frozen directed DoRothEA A+B GRN; action topology = active original global unselected-gene pool.
- PPI+GRN: message passing = exact directed union of PPI bidirectional edges and GRN directed edges; action topology = active original global unselected-gene pool.

Feature mode = hybrid6_raw. Node features, labels, reward, DDQN, PER, soft update, training budget, seed set, and action rules are fixed. The only primary variable is MESSAGE_PASSING_NETWORK.

PPI_BASELINE_REUSED = NO. Reason: Stage 2 introduces an isolated wrapper that decouples message graph injection from the original PPI feature/action environment. The PPI-only path is behaviorally intended as the regression control but is a new execution path, so the 5 PPI seeds are rerun.

LOW_FREQUENCY_DEV_EVAL = DEFERRED because no Stage 2-start frozen, separately audited low-frequency development subset was available.
"""
    write_text(OUT / "05_formal_protocol" / "STAGE2_FORMAL_PROTOCOL.md", protocol)
    return config


def run_formal(message_paths):
    summaries = []
    for condition in CONDITIONS:
        for seed in FORMAL_SEEDS:
            run_dir = OUT / "06_formal_runs" / condition / f"seed_{seed}"
            summaries.append(build_run(condition, seed, run_dir, message_paths[condition], smoke=False))
    return summaries


def metrics_from_best(run_dir):
    metrics = json.loads((run_dir / "validation_metrics_best.json").read_text(encoding="utf-8"))
    return metrics


def load_ranking(run_dir):
    return read_csv(run_dir / "validation_ranking_best.csv", purpose="read formal ranking")


def degrees_from_edges(path, directed=False):
    rows = read_csv(path, delimiter="\t", purpose="read frozen graph for hub analysis")
    deg = {}
    for r in rows:
        s = r["source"]
        t = r["target"]
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    return deg


def spearman_rank_degree(ranking, degree):
    rank_values = []
    degree_values = []
    for row in ranking:
        rank_values.append(int(row["Rank"]))
        degree_values.append(float(degree.get(row["Gene"], 0.0)))
    return float(pd.Series(rank_values).corr(pd.Series(degree_values), method="spearman"))


def analyze_results():
    ppi_degree = degrees_from_edges(STAGE1_READY / "ppi_edges_frozen.tsv")
    grn_degree = degrees_from_edges(STAGE1_READY / "grn_edges_frozen.tsv", directed=True)
    per_seed = []
    checkpoint_rows = []
    hub_rows = []
    stability_rows = []
    ranking_sets = {}
    for condition in CONDITIONS:
        for seed in FORMAL_SEEDS:
            run_dir = OUT / "06_formal_runs" / condition / f"seed_{seed}"
            if not (run_dir / "summary.json").exists():
                continue
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "COMPLETED" or not (run_dir / "validation_metrics_best.json").exists():
                continue
            metrics = metrics_from_best(run_dir)
            ranking = load_ranking(run_dir)
            q_values = np.asarray([float(r["Q_value"]) for r in ranking], dtype=np.float64)
            if len(ranking) != 9039 or len({r["Gene"] for r in ranking}) != 9039 or np.isnan(q_values).any() or np.isinf(q_values).any():
                raise RuntimeError(f"Invalid ranking integrity for {condition} seed {seed}")
            row = {"model": condition, "seed": seed, "best_episode": metrics["episode"]}
            for k in K_VALUES:
                row[f"NDCG@{k}"] = float(metrics[f"NDCG@{k}"])
                row[f"Recall@{k}"] = float(metrics[f"Recall@{k}"])
                row[f"HitCount@{k}"] = int(metrics[f"HitCount@{k}"])
            per_seed.append(row)
            checkpoint_rows.append(
                {
                    "model": condition,
                    "seed": seed,
                    "checkpoint": display_path(run_dir / "checkpoint_best.pt"),
                    "validation_ndcg150": float(metrics["NDCG@150"]),
                    "selection_reason": "max Validation NDCG@150; earlier episode retained implicitly by strict greater-than update",
                }
            )
            top150 = ranking[:150]
            hub_rows.append(
                {
                    "model": condition,
                    "seed": seed,
                    "mean_top150_ppi_degree": float(np.mean([ppi_degree.get(r["Gene"], 0) for r in top150])),
                    "median_top150_ppi_degree": float(np.median([ppi_degree.get(r["Gene"], 0) for r in top150])),
                    "mean_top150_grn_degree": float(np.mean([grn_degree.get(r["Gene"], 0) for r in top150])),
                    "median_top150_grn_degree": float(np.median([grn_degree.get(r["Gene"], 0) for r in top150])),
                    "rank_vs_ppi_degree_spearman": spearman_rank_degree(ranking, ppi_degree),
                    "rank_vs_grn_degree_spearman": spearman_rank_degree(ranking, grn_degree),
                }
            )
            train_rows = read_csv(run_dir / "train_metrics.csv", purpose="read formal training stability metrics")
            action_rows = read_csv(run_dir / "action_trace.csv", purpose="read formal action trace")
            stability_rows.append(
                {
                    "model": condition,
                    "seed": seed,
                    "episode_reward_mean": float(np.mean([float(r["episode_reward"]) for r in train_rows])),
                    "loss_mean": float(np.mean([float(r["mean_loss"]) for r in train_rows if r["mean_loss"] != ""])) if any(r["mean_loss"] != "" for r in train_rows) else "",
                    "q_min": float(np.min([float(r["q_min"]) for r in train_rows if r["q_min"] != ""])),
                    "q_max": float(np.max([float(r["q_max"]) for r in train_rows if r["q_max"] != ""])),
                    "dead_end_count": int(sum(int(r["dead_end_count"]) for r in train_rows)),
                    "invalid_action_count": int(sum(int(r["invalid_action_count"]) for r in train_rows)),
                    "candidate_count_min": int(min(int(r["candidate_count"]) for r in action_rows)),
                    "candidate_count_max": int(max(int(r["candidate_count"]) for r in action_rows)),
                }
            )
            ranking_sets[(condition, seed)] = {r["Gene"] for r in top150}
    write_csv(OUT / "06_formal_runs" / "checkpoint_selection_log.csv", checkpoint_rows)
    write_csv(OUT / "07_analysis" / "per_seed_metrics.csv", per_seed)
    write_csv(OUT / "07_analysis" / "hub_bias_metrics.csv", hub_rows)
    write_csv(OUT / "07_analysis" / "training_stability.csv", stability_rows)
    df = pd.DataFrame(per_seed)
    summary_rows = []
    for condition in CONDITIONS:
        sub = df[df["model"] == condition]
        for metric in ["NDCG@50", "NDCG@100", "NDCG@150", "Recall@50", "Recall@100", "Recall@150"]:
            values = sub[metric].astype(float).to_numpy()
            summary_rows.append(
                {
                    "model": condition,
                    "metric": metric,
                    "mean": float(np.mean(values)) if len(values) else "",
                    "SD": float(np.std(values, ddof=1)) if len(values) > 1 else "",
                    "median": float(np.median(values)) if len(values) else "",
                    "min": float(np.min(values)) if len(values) else "",
                    "max": float(np.max(values)) if len(values) else "",
                }
            )
    write_csv(OUT / "07_analysis" / "summary_metrics.csv", summary_rows)
    paired = []
    for metric in ["NDCG@150", "Recall@150", "NDCG@100", "Recall@100", "NDCG@50", "Recall@50"]:
        for seed in FORMAL_SEEDS:
            vals = {cond: float(df[(df["model"] == cond) & (df["seed"] == seed)][metric].iloc[0]) for cond in CONDITIONS}
            paired.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "PPI": vals["PPI"],
                    "GRN": vals["GRN"],
                    "PPI_GRN": vals["PPI_GRN"],
                    "PPI_GRN_minus_PPI": vals["PPI_GRN"] - vals["PPI"],
                    "GRN_minus_PPI": vals["GRN"] - vals["PPI"],
                    "PPI_GRN_vs_PPI": "win" if vals["PPI_GRN"] > vals["PPI"] else ("tie" if vals["PPI_GRN"] == vals["PPI"] else "loss"),
                }
            )
    write_csv(OUT / "07_analysis" / "paired_seed_comparison.csv", paired)
    names = list(ranking_sets)
    heat_rows = []
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = len(ranking_sets[a] & ranking_sets[b])
            union = len(ranking_sets[a] | ranking_sets[b])
            val = inter / union if union else 0.0
            mat[i, j] = val
            heat_rows.append({"a": f"{a[0]}_{a[1]}", "b": f"{b[0]}_{b[1]}", "top150_jaccard": val})
    write_csv(OUT / "07_analysis" / "ranking_top150_jaccard.csv", heat_rows)
    make_figures(df, pd.DataFrame(hub_rows), pd.DataFrame(stability_rows), mat, [f"{c}_{s}" for c, s in names])
    return {
        "per_seed": per_seed,
        "summary": summary_rows,
        "paired": paired,
        "hub": hub_rows,
        "stability": stability_rows,
    }


def make_figures(df, hub_df, stability_df, jaccard_mat, jaccard_labels):
    def save(name):
        plt.tight_layout()
        plt.savefig(OUT / "08_figures" / name, dpi=160)
        plt.close()
        log_access(OUT / "08_figures" / name, "write Stage 2 figure", "write", "stage2_figure")

    x = np.arange(len(FORMAL_SEEDS))
    width = 0.25
    for metric, filename in [("NDCG@150", "figure1_validation_ndcg150_per_seed.png"), ("Recall@150", "figure2_recall150_per_seed.png")]:
        plt.figure(figsize=(8, 4.5))
        for idx, condition in enumerate(CONDITIONS):
            vals = [float(df[(df["model"] == condition) & (df["seed"] == seed)][metric].iloc[0]) for seed in FORMAL_SEEDS]
            plt.bar(x + (idx - 1) * width, vals, width, label=condition)
        plt.xticks(x, FORMAL_SEEDS)
        plt.xlabel("Seed")
        plt.ylabel(metric)
        plt.legend()
        save(filename)

    plt.figure(figsize=(7, 4.5))
    labels = ["NDCG@50", "NDCG@100", "NDCG@150"]
    x2 = np.arange(len(labels))
    for idx, condition in enumerate(CONDITIONS):
        means = [float(df[df["model"] == condition][m].mean()) for m in labels]
        plt.bar(x2 + (idx - 1) * width, means, width, label=condition)
    plt.xticks(x2, labels)
    plt.ylabel("Mean validation NDCG")
    plt.legend()
    save("figure3_ndcg_summary.png")

    for metric, filename, ylabel in [
        ("episode_reward_mean", "figure4_reward_curve_summary.png", "Mean episode reward"),
        ("loss_mean", "figure5_loss_curve_summary.png", "Mean loss"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for condition in CONDITIONS:
            sub = stability_df[stability_df["model"] == condition]
            plt.plot(sub["seed"], sub[metric], marker="o", label=condition)
        plt.xlabel("Seed")
        plt.ylabel(ylabel)
        plt.legend()
        save(filename)

    for metric, filename, ylabel in [
        ("mean_top150_ppi_degree", "figure6_topk_ppi_degree_comparison.png", "Mean Top150 PPI degree"),
        ("mean_top150_grn_degree", "figure7_topk_grn_degree_comparison.png", "Mean Top150 GRN degree"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for condition in CONDITIONS:
            sub = hub_df[hub_df["model"] == condition]
            plt.plot(sub["seed"], sub[metric], marker="o", label=condition)
        plt.xlabel("Seed")
        plt.ylabel(ylabel)
        plt.legend()
        save(filename)

    plt.figure(figsize=(8, 7))
    plt.imshow(jaccard_mat, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Top150 Jaccard")
    plt.xticks(range(len(jaccard_labels)), jaccard_labels, rotation=90, fontsize=6)
    plt.yticks(range(len(jaccard_labels)), jaccard_labels, fontsize=6)
    save("figure8_ranking_overlap_jaccard_heatmap.png")


def classify(analysis):
    df = pd.DataFrame(analysis["per_seed"])
    ppi = df[df["model"] == "PPI"].set_index("seed")
    grn = df[df["model"] == "GRN"].set_index("seed")
    union = df[df["model"] == "PPI_GRN"].set_index("seed")
    ppi_mean = float(ppi["NDCG@150"].mean())
    grn_mean = float(grn["NDCG@150"].mean())
    union_mean = float(union["NDCG@150"].mean())
    ppi_recall = float(ppi["Recall@150"].mean())
    grn_recall = float(grn["Recall@150"].mean())
    union_recall = float(union["Recall@150"].mean())
    diff = union["NDCG@150"] - ppi["NDCG@150"]
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    grn_signal = "YES" if grn_mean > 0 and float(grn["HitCount@150"].mean()) > 0 else "NO"
    if union_mean > ppi_mean and wins >= 3:
        adds = "YES"
        relation = "YES"
        ready = "YES"
        status = "PASS"
    elif abs(union_mean - ppi_mean) <= 0.01 and (union_recall >= ppi_recall or wins >= 2):
        adds = "UNCERTAIN"
        relation = "UNCERTAIN"
        ready = "CONDITIONAL"
        status = "CONDITIONAL"
    elif grn_signal == "YES" and union_mean < ppi_mean:
        adds = "NO"
        relation = "UNCERTAIN"
        ready = "CONDITIONAL"
        status = "CONDITIONAL"
    else:
        adds = "NO"
        relation = "NO"
        ready = "NO"
        status = "FAIL"
    return {
        "PPI_NDCG150_MEAN": ppi_mean,
        "GRN_NDCG150_MEAN": grn_mean,
        "PPI_GRN_NDCG150_MEAN": union_mean,
        "PPI_RECALL150_MEAN": ppi_recall,
        "GRN_RECALL150_MEAN": grn_recall,
        "PPI_GRN_RECALL150_MEAN": union_recall,
        "PPI_GRN_vs_PPI_wins": wins,
        "PPI_GRN_vs_PPI_losses": losses,
        "GRN_HAS_INDEPENDENT_SIGNAL": grn_signal,
        "GRN_ADDS_VALUE_TO_PPI": adds,
        "RELATION_AWARE_ENCODER_JUSTIFIED": relation,
        "READY_FOR_STAGE3": ready,
        "STAGE2_STATUS": status,
    }


def markdown_table(rows):
    rows = list(rows)
    if not rows:
        return ""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def implementation_manifest():
    source_rows = []
    files = [
        OUT / "scripts" / "stage2_multigraph_experiment.py",
        PROJECT / "src" / "train.py",
        PROJECT / "src" / "DQN.py",
        PROJECT / "src" / "qfunction.py",
        PROJECT / "src" / "inputall.py",
        PROJECT / "src" / "replay_buffer.py",
    ]
    for path in files:
        if path.exists():
            source_rows.append(
                {
                    "file": display_path(path),
                    "role": "stage2_new_code" if str(path).startswith(str(OUT)) else "project_source_read_only_import",
                    "sha256": sha256_file(path),
                    "modified_by_stage2": "YES" if str(path).startswith(str(OUT)) else "NO",
                }
            )
    write_csv(OUT / "02_stage2_implementation" / "source_manifest.csv", source_rows)
    write_text(
        OUT / "02_stage2_implementation" / "implementation_change_log.md",
        """# Stage 2 Implementation Change Log

CORE_PROJECT_SOURCE_MODIFIED = NO

Stage 2 uses a new isolated wrapper in `E:\\codex_file\\新方向阶段2\\scripts\\stage2_multigraph_experiment.py`.

Reason for wrapper: direct use of `--ppi_path` cannot fairly test GRN because project loader symmetrizes all edges and `hybrid6_raw` original features are built from the loaded PPI matrix. The wrapper builds the original PPI environment and swaps only the message graph matrix passed into the Q encoder.

Reward, DDQN, PER, soft update, checkpoint rule, labels, feature mode, and active action mask semantics are unchanged.
""",
    )
    code_hash_lines = [f"{row['sha256']}  {row['file']}" for row in source_rows]
    write_text(OUT / "02_stage2_implementation" / "SHA256SUMS_stage2_code.txt", "\n".join(code_hash_lines) + "\n")


def write_readme():
    write_text(
        OUT / "00_README.md",
        """# RL-GenRisk New Direction Stage 2

Minimal multi-relational feasibility experiment for PPI-only, GRN-only, and PPI+GRN message passing under fixed `hybrid6_raw` features and fixed active global action topology.

Historical Test and future external validation data are not used.
""",
    )


def write_final_report(action, pyg, smoke, formal_summaries, analysis, decision):
    formal_completed = sum(1 for s in formal_summaries if s.get("status") == "COMPLETED")
    per_seed_table = markdown_table(analysis["per_seed"])
    summary_table = markdown_table(analysis["summary"])
    hub_table = markdown_table(analysis["hub"])
    stability_table = markdown_table(analysis["stability"])
    report = f"""# STAGE2_COMPLETION_REPORT

## 1. Executive Summary

STAGE2_STATUS = {decision['STAGE2_STATUS']}

READY_FOR_STAGE3 = {decision['READY_FOR_STAGE3']}

Stage 2 completed the gated action-topology audit, PyG runtime smoke, formal-data smoke, and formal PPI/GRN/PPI+GRN feasibility matrix.

## 2. Boundary Compliance

HISTORICAL_TEST_USED = NO

NEW_EXTERNAL_VALIDATION_USED = NO

REWARD_MODIFIED = NO

MORL_STARTED = NO

NODE_FEATURES_CHANGED = NO

LABEL_SPLIT_CHANGED = NO

## 3. Action Topology Audit

ACTION_SPACE_TYPE = {action['ACTION_SPACE_TYPE']}

MESSAGE_ACTION_GRAPH_COUPLED = {action['MESSAGE_ACTION_GRAPH_COUPLED']}

FAIRNESS_FIX_REQUIRED = YES

FAIRNESS_FIX_APPLIED = YES

The active training path selects from the global unselected 9039-gene pool. The Stage 2 wrapper keeps this action rule fixed and changes only the message graph passed to the Q encoder.

## 4. PyG Runtime

torch = {pyg.get('torch')}

torch_geometric = {pyg.get('torch_geometric')}

CUDA = {pyg.get('CUDA')}

GCN forward = {pyg.get('GCN_FORWARD')}

backward = {pyg.get('BACKWARD')}

optimizer smoke = {pyg.get('OPTIMIZER_STEP_SMOKE')}

pyg-lib warning status = {pyg.get('PYG_LIB_WARNING')}

## 5. Model Definitions

PPI-only = PPI bidirectional message graph; fixed active global action topology.

GRN-only = directed TF -> target DoRothEA A+B message graph; fixed active global action topology.

PPI+GRN = exact directed union of PPI bidirectional edges and directed GRN edges; fixed active global action topology.

## 6. Formal Protocol

seeds = {FORMAL_SEEDS}

episodes = 15

steps = 160

batch = 128

LR = 0.0001

gamma = 0.95

tau = 0.001

PER = alpha 0.2, beta_start 0.1, beta_frames 2000000, eps 1e-5

epsilon = start 1.0, end 0.15, decay 2000.0

checkpoint rule = Validation NDCG@150

## 7. PPI Regression Control

PPI_BASELINE_REUSED = NO

PPI_REGRESSION_CONTROL = rerun under Stage 2 wrapper for seeds 42-46

HISTORICAL_VALIDATION_CONSISTENCY = compared descriptively only; new execution path requires fair rerun.

## 8. Per-seed Results

{per_seed_table}

## 9. Summary Results

{summary_table}

## 10. Primary Endpoint

PRIMARY_ENDPOINT = Validation NDCG@150

PPI_NDCG150_MEAN = {decision['PPI_NDCG150_MEAN']}

GRN_NDCG150_MEAN = {decision['GRN_NDCG150_MEAN']}

PPI_GRN_NDCG150_MEAN = {decision['PPI_GRN_NDCG150_MEAN']}

## 11. GRN Signal Assessment

GRN_HAS_INDEPENDENT_SIGNAL = {decision['GRN_HAS_INDEPENDENT_SIGNAL']}

Evidence: GRN-only Validation NDCG@150 mean = {decision['GRN_NDCG150_MEAN']}; GRN Recall@150 mean = {decision['GRN_RECALL150_MEAN']}.

## 12. PPI+GRN Incremental Value

GRN_ADDS_VALUE_TO_PPI = {decision['GRN_ADDS_VALUE_TO_PPI']}

Evidence: PPI+GRN minus PPI paired wins = {decision['PPI_GRN_vs_PPI_wins']}, losses = {decision['PPI_GRN_vs_PPI_losses']}.

## 13. Hub Bias

{hub_table}

## 14. Training Stability

{stability_table}

## 15. Failures / Unexpected Results

Smoke ACTION_FAIRNESS = {smoke['ACTION_FAIRNESS']}

Formal runs completed = {formal_completed} / 15

LOW_FREQUENCY_DEV_EVAL = DEFERRED

No Historical Test or future external validation result was used to tune or reinterpret the protocol.

## 16. Go / No-Go

GRN_HAS_INDEPENDENT_SIGNAL = {decision['GRN_HAS_INDEPENDENT_SIGNAL']}

GRN_ADDS_VALUE_TO_PPI = {decision['GRN_ADDS_VALUE_TO_PPI']}

RELATION_AWARE_ENCODER_JUSTIFIED = {decision['RELATION_AWARE_ENCODER_JUSTIFIED']}

READY_FOR_STAGE3 = {decision['READY_FOR_STAGE3']}
"""
    write_text(OUT / "STAGE2_COMPLETION_REPORT.md", report)


def write_integrity():
    paths = [
        OUT / "scripts" / "stage2_multigraph_experiment.py",
        OUT / "01_action_topology_audit" / "action_topology_audit.md",
        OUT / "01_action_topology_audit" / "action_graph_code_trace.md",
        OUT / "02_stage2_implementation" / "source_manifest.csv",
        OUT / "03_message_graphs" / "message_graph_manifest.csv",
        OUT / "05_formal_protocol" / "STAGE2_FORMAL_PROTOCOL.md",
        OUT / "05_formal_protocol" / "formal_config.yaml",
        OUT / "05_formal_protocol" / "seed_registry.csv",
        OUT / "06_formal_runs" / "checkpoint_selection_log.csv",
        OUT / "07_analysis" / "per_seed_metrics.csv",
        OUT / "07_analysis" / "summary_metrics.csv",
        OUT / "07_analysis" / "paired_seed_comparison.csv",
        OUT / "STAGE2_COMPLETION_REPORT.md",
        OUT / "terminal_summary.txt",
    ]
    for condition in CONDITIONS:
        paths.append(OUT / "03_message_graphs" / {"PPI": "ppi_message_edges.tsv", "GRN": "grn_message_edges.tsv", "PPI_GRN": "ppi_grn_union_message_edges.tsv"}[condition])
        for seed in FORMAL_SEEDS:
            run_dir = OUT / "06_formal_runs" / condition / f"seed_{seed}"
            paths.extend([run_dir / "checkpoint_best.pt", run_dir / "validation_ranking_best.csv", run_dir / "summary.json"])
    lines = []
    for path in paths:
        if path.exists():
            lines.append(f"{sha256_file(path)}  {display_path(path)}")
    write_text(OUT / "09_integrity" / "SHA256SUMS_stage2.txt", "\n".join(lines) + "\n")


def write_terminal_summary(action, pyg, smoke, formal_summaries, decision):
    formal_completed = sum(1 for s in formal_summaries if s.get("status") == "COMPLETED")
    summary = f"""============================================================
RL-GenRisk NEW DIRECTION - STAGE 2 COMPLETE
MINIMAL MULTI-RELATIONAL FEASIBILITY EXPERIMENT
============================================================

OUTPUT_DIR:
{display_path(OUT)}

HISTORICAL_TEST_USED:
NO

NEW_EXTERNAL_VALIDATION_USED:
NO

FEATURES_CHANGED:
NO

REWARD_CHANGED:
NO

MORL_STARTED:
NO

ACTION_SPACE_TYPE:
{action['ACTION_SPACE_TYPE']}

MESSAGE_ACTION_GRAPH_COUPLED:
{action['MESSAGE_ACTION_GRAPH_COUPLED']}

ACTION_FAIRNESS:
{smoke['ACTION_FAIRNESS']}

PYG_RUNTIME:
{pyg['PYG_RUNTIME']}

PPI_BASELINE_REUSED:
NO

FORMAL_SEEDS:
{','.join(str(s) for s in FORMAL_SEEDS)}

FORMAL_RUNS_COMPLETED:
{formal_completed} / 15

PPI_NDCG150_MEAN:
{decision['PPI_NDCG150_MEAN']}

GRN_NDCG150_MEAN:
{decision['GRN_NDCG150_MEAN']}

PPI_GRN_NDCG150_MEAN:
{decision['PPI_GRN_NDCG150_MEAN']}

PPI_RECALL150_MEAN:
{decision['PPI_RECALL150_MEAN']}

GRN_RECALL150_MEAN:
{decision['GRN_RECALL150_MEAN']}

PPI_GRN_RECALL150_MEAN:
{decision['PPI_GRN_RECALL150_MEAN']}

GRN_HAS_INDEPENDENT_SIGNAL:
{decision['GRN_HAS_INDEPENDENT_SIGNAL']}

GRN_ADDS_VALUE_TO_PPI:
{decision['GRN_ADDS_VALUE_TO_PPI']}

RELATION_AWARE_ENCODER_JUSTIFIED:
{decision['RELATION_AWARE_ENCODER_JUSTIFIED']}

STAGE2_STATUS:
{decision['STAGE2_STATUS']}

READY_FOR_STAGE3:
{decision['READY_FOR_STAGE3']}

FINAL_REPORT:
{display_path(OUT / 'STAGE2_COMPLETION_REPORT.md')}
============================================================
"""
    write_text(OUT / "terminal_summary.txt", summary)
    print(summary)


def write_access_log():
    write_csv(OUT / "stage2_data_access_log.csv", DATA_ACCESS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()
    mkdirs()
    write_readme()
    frozen = load_frozen_inputs()
    action = stage2a_action_audit()
    implementation_manifest()
    pyg = pyg_runtime_smoke()
    if pyg["PYG_RUNTIME"] != "PASS":
        write_access_log()
        raise RuntimeError("PYG_RUNTIME failed; stopping before Stage 2B.")
    message_paths = generate_message_graphs(frozen)
    smoke = run_smoke(message_paths)
    if smoke["ACTION_FAIRNESS"] != "PASS":
        write_access_log()
        raise RuntimeError("ACTION_FAIRNESS failed; stopping before formal runs.")
    write_formal_protocol()
    if args.skip_formal:
        write_access_log()
        return
    formal_summaries = run_formal(message_paths)
    analysis = analyze_results()
    decision = classify(analysis)
    write_final_report(action, pyg, smoke, formal_summaries, analysis, decision)
    write_terminal_summary(action, pyg, smoke, formal_summaries, decision)
    write_access_log()
    write_integrity()


if __name__ == "__main__":
    main()
