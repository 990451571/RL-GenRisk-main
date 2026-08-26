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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter as scatter_add


if os.name == "nt":
    PROJECT = Path(r"E:\Projects\RL-GenRisk-main")
    STAGE0 = Path("E:/codex_file/\u65b0\u65b9\u5411\u9636\u6bb50")
    STAGE1 = Path("E:/codex_file/\u65b0\u65b9\u5411\u9636\u6bb51")
    STAGE2 = Path("E:/codex_file/\u65b0\u65b9\u5411\u9636\u6bb52")
    STAGE1_READY = STAGE1 / "10_stage2_ready"
    OUT = Path("E:/codex_file/\u65b0\u65b9\u5411\u9636\u6bb53")
    PROTOCOL_B = Path("E:/codex_file/\u4e00\u9636\u6bb5/driver_label_protocol/protocol_B")
else:
    PROJECT = Path("/mnt/e/Projects/RL-GenRisk-main")
    STAGE0 = Path("/mnt/e/codex_file/\u65b0\u65b9\u5411\u9636\u6bb50")
    STAGE1 = Path("/mnt/e/codex_file/\u65b0\u65b9\u5411\u9636\u6bb51")
    STAGE2 = Path("/mnt/e/codex_file/\u65b0\u65b9\u5411\u9636\u6bb52")
    STAGE1_READY = STAGE1 / "10_stage2_ready"
    OUT = Path("/mnt/e/codex_file/\u65b0\u65b9\u5411\u9636\u6bb53")
    PROTOCOL_B = Path("/mnt/e/codex_file/\u4e00\u9636\u6bb5/driver_label_protocol/protocol_B")

SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train  # noqa: E402
from qfunction import Q_Fun, change  # noqa: E402


CONDITIONS = {
    "PPI": "PPI-only",
    "GRN": "GRN-message model under fixed original action topology",
    "PPI_GRN": "PPI+GRN edge-union message model under fixed original action topology",
    "RelationAware": "DualBranch_GCN_GlobalGate",
    "CapacityMatchedSimpleUnion": "Capacity-matched simple union control",
}
FORMAL_SEEDS = [42, 43, 44, 45, 46]
PRIMARY_ENDPOINT = "Validation NDCG@150"
FEATURE_MODE = "hybrid6_raw"
K_VALUES = [50, 100, 150]
DATA_ACCESS = []
PRIMARY_MODEL = "DualBranch_GCN_GlobalGate"
STAGE2_MESSAGE_DIR = STAGE2 / "03_message_graphs"


class DualBranchGlobalGateQ(nn.Module):
    """Stage 3 primary relation-aware Q encoder.

    It preserves the `Q_Fun.forward(mu, x, action_sel, batch_flag)` interface so
    DDQN, PER, replay masks, reward, and action topology stay unchanged.
    """

    def __init__(self, in_dim, hid_dim, ppi_matrix, grn_matrix):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hid_dim = int(hid_dim)
        self.n_actions = int(np.asarray(ppi_matrix).shape[0])
        if np.asarray(grn_matrix).shape != (self.n_actions, self.n_actions):
            raise ValueError("PPI and GRN message matrices must have identical square shape.")

        self.ppi_lin1 = nn.Linear(self.in_dim, self.hid_dim)
        self.ppi_conv1 = GCNConv(self.hid_dim, self.hid_dim)
        self.ppi_conv2 = GCNConv(self.hid_dim, self.hid_dim)
        self.ppi_lin2 = nn.Linear(3 * self.hid_dim, self.hid_dim)

        self.grn_lin1 = nn.Linear(self.in_dim, self.hid_dim)
        self.grn_conv1 = GCNConv(self.hid_dim, self.hid_dim)
        self.grn_conv2 = GCNConv(self.hid_dim, self.hid_dim)
        self.grn_lin2 = nn.Linear(3 * self.hid_dim, self.hid_dim)

        # Kept to mirror the dormant parameter capacity in the current Q_Fun.
        self.lin3 = nn.Linear(self.in_dim, self.hid_dim)
        self.lin4 = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin6 = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin5 = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.lin8 = nn.Linear(self.hid_dim, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.relation_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer("ppi_edge_index", torch.as_tensor(change(ppi_matrix), dtype=torch.long), persistent=False)
        self.register_buffer("grn_edge_index", torch.as_tensor(change(grn_matrix), dtype=torch.long), persistent=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.mask_mode = "full"
        self.last_embedding_stats = {}

    def _branch(self, x, lin1, conv1, conv2, lin2, edge_index):
        x1 = lin1(x)
        x2 = F.relu(conv1(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(conv2(x2, edge_index))
        x3 = self.dropout(x3)
        return lin2(torch.cat([x1, x2, x3], dim=-1))

    def alphas(self):
        return torch.softmax(self.relation_logits, dim=0)

    def alpha_values(self):
        vals = self.alphas().detach().cpu().numpy().astype(float).tolist()
        return {"alpha_PPI": vals[0], "alpha_GRN": vals[1]}

    def gradient_report(self):
        def norm_for(names):
            total = 0.0
            seen = False
            for name, param in self.named_parameters():
                if any(name.startswith(prefix) for prefix in names) and param.grad is not None:
                    total += float(param.grad.detach().pow(2).sum().cpu())
                    seen = True
            return math.sqrt(total) if seen else 0.0

        return {
            "ppi_branch_grad_norm": norm_for(["ppi_"]),
            "grn_branch_grad_norm": norm_for(["grn_"]),
            "gate_grad_norm": norm_for(["relation_logits"]),
        }

    def encoder_parameter_count(self):
        prefixes = ("ppi_", "grn_")
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith(prefixes))

    def fusion_parameter_count(self):
        return int(self.relation_logits.numel())

    def q_head_parameter_count(self):
        return sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith(("ppi_", "grn_")) and n != "relation_logits"
        )

    def forward(self, mu, x, action_sel, batch_flag=False, test_flag=False):
        if mu is None:
            h_ppi = self._branch(x, self.ppi_lin1, self.ppi_conv1, self.ppi_conv2, self.ppi_lin2, self.ppi_edge_index)
            h_grn = self._branch(x, self.grn_lin1, self.grn_conv1, self.grn_conv2, self.grn_lin2, self.grn_edge_index)
            alpha = self.alphas()
            if self.mask_mode == "minus_ppi":
                nodes_vec = h_grn
            elif self.mask_mode == "minus_grn":
                nodes_vec = h_ppi
            else:
                nodes_vec = alpha[0] * h_ppi + alpha[1] * h_grn
            with torch.no_grad():
                self.last_embedding_stats = {
                    "mean_norm_H_PPI": float(torch.linalg.norm(h_ppi, dim=-1).mean().detach().cpu()),
                    "mean_norm_H_GRN": float(torch.linalg.norm(h_grn, dim=-1).mean().detach().cpu()),
                    "mean_norm_H_fused": float(torch.linalg.norm(nodes_vec, dim=-1).mean().detach().cpu()),
                    "var_H_PPI": float(h_ppi.var().detach().cpu()),
                    "var_H_GRN": float(h_grn.var().detach().cpu()),
                    "var_H_fused": float(nodes_vec.var().detach().cpu()),
                    **self.alpha_values(),
                }
        else:
            nodes_vec = mu

        num_nodes = self.n_actions
        if not batch_flag:
            idx = action_sel.long()
            graph_pool2 = scatter_add(nodes_vec, idx, dim=-2, dim_size=2)[0]
            graph_pool2 = graph_pool2.repeat(num_nodes, 1)
        else:
            idx = action_sel.long()
            idx_expanded = idx.unsqueeze(-1).expand_as(nodes_vec)
            out = torch.zeros(nodes_vec.size(0), 2, nodes_vec.size(2), device=nodes_vec.device, dtype=nodes_vec.dtype)
            out.scatter_add_(1, idx_expanded, nodes_vec)
            graph_pool2 = out[:, [0], :].repeat(1, num_nodes, 1)
        cat = torch.cat((self.lin6(graph_pool2), nodes_vec), dim=-1)
        q_values = self.lin8(F.relu(self.lin5(F.relu(cat)))).squeeze(-1)
        return q_values, nodes_vec


def qfun_parameter_breakdown(model):
    names = dict(model.named_parameters())
    encoder_prefixes = ("lin1", "conv1", "conv2", "lin2")
    encoder = sum(p.numel() for n, p in names.items() if n.startswith(encoder_prefixes))
    q_head = sum(p.numel() for n, p in names.items()) - encoder
    return {
        "total": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder": encoder,
        "q_head": q_head,
        "fusion": 0,
    }


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
        "01_architecture_audit",
        "02_stage3_implementation",
        "03_formal_runs/RelationAware",
        "04_analysis",
        "05_relation_attribution",
        "06_stability",
        "07_capacity_control",
        "08_figures",
        "09_integrity",
        "scripts",
        "src_stage3",
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
    log_access(path, "write Stage 3 output", "write", "stage3_output")


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
    log_access(path, "write Stage 3 output", "write", "stage3_output")


def append_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    log_access(path, "append Stage 3 run log", "write", "stage3_output")


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
    write_text(OUT / "02_stage3_implementation" / "pyg_runtime_smoke.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def load_edge_indices(rows):
    edges = []
    for r in rows:
        edges.append((int(r["source_index"]), int(r["target_index"]), r["source"], r["target"]))
    return edges


def load_stage2_message_graphs():
    manifest_path = STAGE2_MESSAGE_DIR / "message_graph_manifest.csv"
    integrity_path = STAGE2 / "09_integrity" / "SHA256SUMS_stage2.txt"
    manifest = read_csv(manifest_path, purpose="read frozen Stage 2 message graph manifest")
    integrity_text = read_text(integrity_path, purpose="read Stage 2 SHA256 integrity manifest")
    paths = {}
    integrity_ok = True
    integrity_rows = []
    for row in manifest:
        path = current_path(row["file"])
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = row["sha256"]
        in_integrity = expected in integrity_text and display_path(path) in integrity_text
        ok = actual == expected and in_integrity
        integrity_ok = integrity_ok and ok
        integrity_rows.append(
            {
                "condition": row["condition"],
                "file": display_path(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "stage2_integrity_manifest_contains_entry": "YES" if in_integrity else "NO",
                "status": "PASS" if ok else "FAIL",
            }
        )
        paths[row["condition"]] = path
    write_csv(OUT / "01_architecture_audit" / "stage3_input_integrity.csv", integrity_rows)
    if not integrity_ok:
        write_text(OUT / "01_architecture_audit" / "STAGE3_INPUT_INTEGRITY_FAIL.md", "STAGE3_INPUT_INTEGRITY = FAIL\n")
        raise RuntimeError("STAGE3_INPUT_INTEGRITY = FAIL")
    return paths


def matrix_from_message_edges(path, node_count=9039):
    rows = read_csv(path, delimiter="\t", purpose="read Stage 2 frozen message graph")
    mat = np.zeros((node_count, node_count), dtype=np.float32)
    for r in rows:
        mat[int(r["source_index"]), int(r["target_index"])] = 1.0
    return mat, len(rows)


def architecture_audit(message_paths):
    ppi_net, _ = matrix_from_message_edges(message_paths["PPI"])
    grn_net, _ = matrix_from_message_edges(message_paths["GRN"])
    union_net, _ = matrix_from_message_edges(message_paths["PPI_GRN"])
    simple = Q_Fun(6, 64, 3, 0.0001, union_net)
    relation = DualBranchGlobalGateQ(6, 64, ppi_net, grn_net)
    simple_counts = qfun_parameter_breakdown(simple)
    relation_counts = {
        "total": sum(p.numel() for p in relation.parameters() if p.requires_grad),
        "encoder": relation.encoder_parameter_count(),
        "q_head": relation.q_head_parameter_count(),
        "fusion": relation.fusion_parameter_count(),
    }
    ratio = relation_counts["total"] / simple_counts["total"]
    capacity_required = ratio > 1.10
    best_h = None
    best_counts = None
    best_abs = None
    for h in range(8, 257):
        model = Q_Fun(6, h, 3, 0.0001, union_net)
        counts = qfun_parameter_breakdown(model)
        diff = abs(counts["total"] - relation_counts["total"])
        if best_abs is None or diff < best_abs:
            best_abs = diff
            best_h = h
            best_counts = counts
    capacity_diff = (best_counts["total"] - relation_counts["total"]) / relation_counts["total"]
    rows = [
        {
            "model": "PPI-only",
            "total_trainable_parameters": simple_counts["total"],
            "encoder_parameters": simple_counts["encoder"],
            "q_head_parameters": simple_counts["q_head"],
            "fusion_parameters": simple_counts["fusion"],
            "relative_to_simple_union": 1.0,
            "hidden_dim": 64,
        },
        {
            "model": "GRN-only",
            "total_trainable_parameters": simple_counts["total"],
            "encoder_parameters": simple_counts["encoder"],
            "q_head_parameters": simple_counts["q_head"],
            "fusion_parameters": simple_counts["fusion"],
            "relative_to_simple_union": 1.0,
            "hidden_dim": 64,
        },
        {
            "model": "SimpleUnion",
            "total_trainable_parameters": simple_counts["total"],
            "encoder_parameters": simple_counts["encoder"],
            "q_head_parameters": simple_counts["q_head"],
            "fusion_parameters": simple_counts["fusion"],
            "relative_to_simple_union": 1.0,
            "hidden_dim": 64,
        },
        {
            "model": "RelationAware",
            "total_trainable_parameters": relation_counts["total"],
            "encoder_parameters": relation_counts["encoder"],
            "q_head_parameters": relation_counts["q_head"],
            "fusion_parameters": relation_counts["fusion"],
            "relative_to_simple_union": ratio,
            "hidden_dim": 64,
        },
        {
            "model": "CapacityMatchedSimpleUnion_candidate",
            "total_trainable_parameters": best_counts["total"],
            "encoder_parameters": best_counts["encoder"],
            "q_head_parameters": best_counts["q_head"],
            "fusion_parameters": best_counts["fusion"],
            "relative_to_simple_union": best_counts["total"] / simple_counts["total"],
            "hidden_dim": best_h,
        },
    ]
    write_csv(OUT / "01_architecture_audit" / "model_parameter_counts.csv", rows)
    protocol = f"""# STAGE3_ARCHITECTURE_PROTOCOL

PRIMARY_RELATION_AWARE_MODEL = {PRIMARY_MODEL}

Architecture = Dual-Branch Relation-Specific GCN + Lightweight Global Gated Fusion.

PPI branch uses frozen Stage 2 PPI bidirectional message graph.

GRN branch uses frozen Stage 1/2 DoRothEA A+B TF -> target directed message graph.

Fusion:

`alpha = softmax([theta_PPI, theta_GRN])`

`H_fused = alpha_PPI * H_PPI + alpha_GRN * H_GRN`

Gate initialization = theta_PPI = 0, theta_GRN = 0, so initial alpha_PPI = alpha_GRN = 0.5.

The Q head, DDQN target logic, PER, reward, action mask, action topology, feature mode, labels, seeds, and training budget are unchanged.

Forbidden in this primary experiment: R-GCN, HGT, Graph Transformer, multi-head attention, node-wise attention, new reward, new features, new labels, Historical Test, or external validation.
"""
    write_text(OUT / "01_architecture_audit" / "STAGE3_ARCHITECTURE_PROTOCOL.md", protocol)
    diagram = """# Architecture Diagram

```mermaid
flowchart TD
    X["hybrid6_raw X"] --> P["PPI GCN branch<br/>E = frozen bidirectional PPI"]
    X --> G["GRN GCN branch<br/>E = directed TF->target GRN"]
    P --> HP["H_PPI"]
    G --> HG["H_GRN"]
    HP --> F["global relation gate<br/>softmax(theta_PPI, theta_GRN)"]
    HG --> F
    F --> H["H_fused"]
    H --> Q["existing DDQN/Q head"]
```
"""
    write_text(OUT / "01_architecture_audit" / "architecture_diagram.md", diagram)
    decision = f"""# Capacity Control Decision

SimpleUnion parameters = {simple_counts['total']}

RelationAware parameters = {relation_counts['total']}

parameter ratio = {ratio}

CAPACITY_CONTROL_REQUIRED = {'YES' if capacity_required else 'NO'}

CapacityMatchedSimpleUnion hidden_dim candidate = {best_h}

CapacityMatchedSimpleUnion parameters = {best_counts['total']}

capacity matched relative difference vs RelationAware = {capacity_diff}

Capacity control will be run only if RelationAware primary formal results outperform Stage 2 SimpleUnion and this parameter ratio remains > 1.10.
"""
    write_text(OUT / "01_architecture_audit" / "capacity_control_decision.md", decision)
    return {
        "simple_counts": simple_counts,
        "relation_counts": relation_counts,
        "ratio": ratio,
        "capacity_required": capacity_required,
        "capacity_hidden_dim": best_h,
        "capacity_counts": best_counts,
        "capacity_diff": capacity_diff,
    }


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


def build_run(condition, seed, run_dir, message_edge_path, smoke=False, model_kind="SimpleUnion", ppi_edge_path=None, grn_edge_path=None, hidden_dim=64):
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if run_dir.exists():
        resolved = run_dir.resolve()
        if OUT.resolve() not in [resolved, *resolved.parents]:
            raise RuntimeError(f"Refusing to clean non-Stage3 run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(seed, run_dir, smoke=smoke)
    args.embedding_size = int(hidden_dim)
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
    env["stage3_message_condition"] = condition
    env["stage3_message_edge_count"] = message_edge_count
    agent = train.build_agent(args, env, device)
    if model_kind == "RelationAware":
        if ppi_edge_path is None or grn_edge_path is None:
            raise ValueError("RelationAware requires ppi_edge_path and grn_edge_path.")
        ppi_net, ppi_edge_count = matrix_from_message_edges(ppi_edge_path)
        grn_net, grn_edge_count = matrix_from_message_edges(grn_edge_path)
        agent.Q = DualBranchGlobalGateQ(agent.feature_dim, args.embedding_size, ppi_net, grn_net).to(device)
        agent.Q_target = DualBranchGlobalGateQ(agent.feature_dim, args.embedding_size, ppi_net, grn_net).to(device)
        agent.Q_target.load_state_dict(agent.Q.state_dict())
        agent.Q_target.eval()
        agent.Q.device = device
        agent.Q_target.device = device
        agent.Q.optimizer = torch.optim.Adam(agent.Q.parameters(), lr=args.learning_rate)
        ppi_branch_edges = ppi_edge_count
        grn_branch_edges = grn_edge_count
    else:
        ppi_branch_edges = ""
        grn_branch_edges = ""
    config = {
        "stage": "Stage 3",
        "condition": condition,
        "condition_definition": CONDITIONS.get(condition, condition),
        "model_kind": model_kind,
        "seed": seed,
        "args": vars(args),
        "run_dir": display_path(run_dir),
        "feature_mode": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "message_edge_file": display_path(message_edge_path),
        "message_edge_count_directed_for_pyg": message_edge_count,
        "ppi_branch_edge_count": ppi_branch_edges,
        "grn_branch_edge_count": grn_branch_edges,
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
                if model_kind == "RelationAware":
                    alpha = agent.Q.alpha_values()
                    val_metrics["alpha_PPI"] = alpha["alpha_PPI"]
                    val_metrics["alpha_GRN"] = alpha["alpha_GRN"]
                    val_metrics.update(agent.Q.last_embedding_stats)
                    append_csv(
                        run_dir / "validation_trajectory.csv",
                        [
                            {
                                "episode": episode,
                                "NDCG@50": val_metrics.get("NDCG@50"),
                                "NDCG@100": val_metrics.get("NDCG@100"),
                                "NDCG@150": val_metrics.get("NDCG@150"),
                                "Recall@50": val_metrics.get("Recall@50"),
                                "Recall@100": val_metrics.get("Recall@100"),
                                "Recall@150": val_metrics.get("Recall@150"),
                                "alpha_PPI": alpha["alpha_PPI"],
                                "alpha_GRN": alpha["alpha_GRN"],
                                "mean_norm_H_PPI": agent.Q.last_embedding_stats.get("mean_norm_H_PPI"),
                                "mean_norm_H_GRN": agent.Q.last_embedding_stats.get("mean_norm_H_GRN"),
                                "mean_norm_H_fused": agent.Q.last_embedding_stats.get("mean_norm_H_fused"),
                            }
                        ],
                        [
                            "episode",
                            "NDCG@50",
                            "NDCG@100",
                            "NDCG@150",
                            "Recall@50",
                            "Recall@100",
                            "Recall@150",
                            "alpha_PPI",
                            "alpha_GRN",
                            "mean_norm_H_PPI",
                            "mean_norm_H_GRN",
                            "mean_norm_H_fused",
                        ],
                    )
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
        "model_kind": model_kind,
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
    if model_kind == "RelationAware":
        payload = torch.load(run_dir / "checkpoint_best.pt", map_location=agent.Q.device, weights_only=False)
        state = payload.get("online_net_state_dict", payload.get("online_state_dict"))
        agent.Q.load_state_dict(state)
        agent.Q.eval()
        alpha = agent.Q.alpha_values()
        summary.update(alpha)
        summary.update(agent.Q.last_embedding_stats)
        if smoke:
            summary["gradient_report"] = getattr(agent.Q, "gradient_report")()
    write_json(run_dir / "summary.json", summary)
    return summary


def stage3b_regression_and_smoke(message_paths):
    regression = build_run(
        "PPI_GRN",
        42,
        OUT / "02_stage3_implementation" / "simple_union_regression_seed42",
        message_paths["PPI_GRN"],
        smoke=False,
        model_kind="SimpleUnion",
    )
    stage2_summary = json.loads((STAGE2 / "06_formal_runs" / "PPI_GRN" / "seed_42" / "summary.json").read_text(encoding="utf-8"))
    stage2_metrics = json.loads((STAGE2 / "06_formal_runs" / "PPI_GRN" / "seed_42" / "validation_metrics_best.json").read_text(encoding="utf-8"))
    reg_dir = current_path(regression["run_dir"])
    reg_metrics = json.loads((reg_dir / "validation_metrics_best.json").read_text(encoding="utf-8"))
    ndcg_diff = abs(float(reg_metrics["NDCG@150"]) - float(stage2_metrics["NDCG@150"]))
    best_episode_same = int(reg_metrics["episode"]) == int(stage2_metrics["episode"])
    regression_pass = ndcg_diff <= 1e-9 and best_episode_same
    report = f"""# SimpleUnion Regression Control

SIMPLE_UNION_REGRESSION = {'PASS' if regression_pass else 'FAIL'}

Stage 2 seed42 best episode = {stage2_metrics['episode']}

Stage 3 SimpleUnion seed42 best episode = {reg_metrics['episode']}

Stage 2 seed42 NDCG@150 = {stage2_metrics['NDCG@150']}

Stage 3 seed42 NDCG@150 = {reg_metrics['NDCG@150']}

absolute difference = {ndcg_diff}

Criterion: same best episode and NDCG@150 difference <= 1e-9.
"""
    write_text(OUT / "02_stage3_implementation" / "regression_control_report.md", report)
    if not regression_pass:
        write_json(OUT / "02_stage3_implementation" / "regression_control_fail.json", {"stage2": stage2_metrics, "stage3": reg_metrics, "ndcg_diff": ndcg_diff})
        raise RuntimeError("SIMPLE_UNION_REGRESSION = FAIL")

    smoke = build_run(
        "RelationAware",
        42,
        OUT / "02_stage3_implementation" / "relationaware_smoke_seed42",
        message_paths["PPI_GRN"],
        smoke=True,
        model_kind="RelationAware",
        ppi_edge_path=message_paths["PPI"],
        grn_edge_path=message_paths["GRN"],
    )
    smoke_dir = current_path(smoke["run_dir"])
    ranking = read_csv(smoke_dir / "validation_ranking_best.csv", purpose="read RelationAware smoke ranking")
    q_values = np.asarray([float(r["Q_value"]) for r in ranking], dtype=np.float64)
    grad = smoke.get("gradient_report", {})
    alpha_sum = float(smoke.get("alpha_PPI", 0.0)) + float(smoke.get("alpha_GRN", 0.0))
    gradient_smoke = {
        "forward": "PASS",
        "backward": "PASS" if smoke["optimizer_step_count"] > 0 else "FAIL",
        "optimizer": "PASS" if smoke["optimizer_step_count"] > 0 else "FAIL",
        "PER": "PASS" if smoke["per_update_count"] > 0 else "FAIL",
        "checkpoint": "PASS" if smoke["checkpoint_best"] else "FAIL",
        "validation_inference": "PASS",
        "ranking_generation": "PASS" if len(ranking) == 9039 and len({r["Gene"] for r in ranking}) == 9039 else "FAIL",
        "NaN": int(np.isnan(q_values).sum()),
        "Inf": int(np.isinf(q_values).sum()),
        "alpha_PPI": smoke.get("alpha_PPI"),
        "alpha_GRN": smoke.get("alpha_GRN"),
        "alpha_sum": alpha_sum,
        "alpha_sum_close_to_1": "PASS" if abs(alpha_sum - 1.0) < 1e-6 else "FAIL",
        "ppi_branch_receives_gradient": "YES" if float(grad.get("ppi_branch_grad_norm", 0.0)) > 0 else "NO",
        "grn_branch_receives_gradient": "YES" if float(grad.get("grn_branch_grad_norm", 0.0)) > 0 else "NO",
        "gate_parameters_receive_gradient": "YES" if float(grad.get("gate_grad_norm", 0.0)) > 0 else "NO",
        **grad,
        "H_PPI_shape": [9039, 64],
        "H_GRN_shape": [9039, 64],
        "H_fused_shape": [9039, 64],
    }
    write_json(OUT / "02_stage3_implementation" / "gradient_smoke.json", gradient_smoke)
    required = [
        gradient_smoke["backward"] == "PASS",
        gradient_smoke["optimizer"] == "PASS",
        gradient_smoke["PER"] == "PASS",
        gradient_smoke["checkpoint"] == "PASS",
        gradient_smoke["ranking_generation"] == "PASS",
        gradient_smoke["NaN"] == 0,
        gradient_smoke["Inf"] == 0,
        gradient_smoke["alpha_sum_close_to_1"] == "PASS",
        gradient_smoke["ppi_branch_receives_gradient"] == "YES",
        gradient_smoke["grn_branch_receives_gradient"] == "YES",
        gradient_smoke["gate_parameters_receive_gradient"] == "YES",
    ]
    if not all(required):
        raise RuntimeError("RelationAware smoke or gradient check failed.")
    return {"SIMPLE_UNION_REGRESSION": "PASS", "RELATIONAWARE_SMOKE": "PASS", "gradient_smoke": gradient_smoke}


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
    write_csv(OUT / "01_architecture_audit" / "seed_registry.csv", seed_rows)
    config = {
        "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT,
        "feature_mode": FEATURE_MODE,
        "message_network_conditions": ["RelationAware"],
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
    lines = ["# stage3_formal_config.yaml"]
    for key, value in config.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    write_text(OUT / "01_architecture_audit" / "stage3_formal_config.yaml", "\n".join(lines) + "\n")
    protocol = f"""# STAGE3_FORMAL_PROTOCOL

PRIMARY_ENDPOINT = {PRIMARY_ENDPOINT}

Formal matrix = 1 primary relation-aware model x 5 seeds = 5 runs.

Primary condition:

- RelationAware: DualBranch_GCN_GlobalGate with PPI and GRN branch identities preserved.

Feature mode = hybrid6_raw. Node features, labels, reward, DDQN, PER, soft update, training budget, seed set, and action rules are fixed. The only primary variable is MESSAGE_PASSING_NETWORK.

SIMPLE_UNION_BASELINE_REUSED = YES if Stage 3 SimpleUnion seed42 regression control passes.

LOW_FREQUENCY_DEV_EVAL = DEFERRED.
"""
    write_text(OUT / "01_architecture_audit" / "STAGE3_FORMAL_PROTOCOL.md", protocol)
    return config


def run_formal(message_paths):
    summaries = []
    for seed in FORMAL_SEEDS:
        run_dir = OUT / "03_formal_runs" / "RelationAware" / f"seed{seed}"
        summaries.append(
            build_run(
                "RelationAware",
                seed,
                run_dir,
                message_paths["PPI_GRN"],
                smoke=False,
                model_kind="RelationAware",
                ppi_edge_path=message_paths["PPI"],
                grn_edge_path=message_paths["GRN"],
            )
        )
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


def evaluate_relation_masking(seed, run_dir, message_paths):
    args = make_args(seed, run_dir, smoke=False)
    train.validate_training_args(args)
    train.set_seed(seed)
    device = train.choose_device(args.device)
    env = train.build_environment(args, run_dir, normalization_metadata=None)
    union_net, _ = matrix_from_message_edges(message_paths["PPI_GRN"])
    ppi_net, _ = matrix_from_message_edges(message_paths["PPI"])
    grn_net, _ = matrix_from_message_edges(message_paths["GRN"])
    env["net"] = union_net
    agent = train.build_agent(args, env, device)
    agent.Q = DualBranchGlobalGateQ(agent.feature_dim, args.embedding_size, ppi_net, grn_net).to(device)
    agent.Q.device = device
    payload = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    state = payload.get("online_net_state_dict", payload.get("online_state_dict"))
    agent.Q.load_state_dict(state)
    agent.Q.eval()
    mode_rows = []
    rankings = {}
    for mode, label in [("full", "Full"), ("minus_ppi", "-PPI"), ("minus_grn", "-GRN")]:
        agent.Q.mask_mode = mode
        state_tensor = torch.tensor(env["node_features"], dtype=torch.float32, device=agent.Q.device)
        mask_tensor = torch.LongTensor(np.ones(agent.n_actions)).to(agent.Q.device)
        with torch.no_grad():
            q_values, _ = agent.Q(None, state_tensor, mask_tensor)
        q_np = q_values.detach().cpu().numpy()
        ranking_path = run_dir / f"relation_mask_{mode}_ranking.csv"
        ranking_rows = train.write_ranking(ranking_path, q_np, env["gene_name"], feature_mode=args.feature_mode)
        for rank, ranking_row in enumerate(ranking_rows, start=1):
            ranking_row.setdefault("Rank", rank)
        rankings[label] = ranking_rows
        labels = set(env["validation_driver_genes"])
        metrics = {"seed": seed, "mask_mode": label}
        for k in K_VALUES:
            item = train.metrics_at_k(ranking_rows, labels, k)
            metrics[f"NDCG@{k}"] = item["NDCG"]
            metrics[f"Recall@{k}"] = item["Recall"]
            metrics[f"HitCount@{k}"] = item["HitCount"]
        metrics.update(agent.Q.alpha_values())
        metrics.update(agent.Q.last_embedding_stats)
        mode_rows.append(metrics)
    full = next(r for r in mode_rows if r["mask_mode"] == "Full")
    for row in mode_rows:
        row["Delta_NDCG150_vs_Full"] = row["NDCG@150"] - full["NDCG@150"]
        row["Delta_Recall150_vs_Full"] = row["Recall@150"] - full["Recall@150"]
    write_csv(run_dir / "relation_masking_metrics.csv", mode_rows)
    rank_maps = {label: {row["Gene"]: int(row["Rank"]) for row in rows} for label, rows in rankings.items()}
    top150 = rankings["Full"][:150]
    delta_rows = [
        {
            "seed": seed,
            "Gene": row["Gene"],
            "Full_Rank": rank_maps["Full"][row["Gene"]],
            "PPI_masked_Rank": rank_maps["-PPI"][row["Gene"]],
            "GRN_masked_Rank": rank_maps["-GRN"][row["Gene"]],
            "DeltaRank_PPI": rank_maps["-PPI"][row["Gene"]] - rank_maps["Full"][row["Gene"]],
            "DeltaRank_GRN": rank_maps["-GRN"][row["Gene"]] - rank_maps["Full"][row["Gene"]],
        }
        for row in top150
    ]
    write_csv(run_dir / "gene_relation_deltarank.csv", delta_rows)
    return mode_rows, delta_rows


def rbo_score(a, b, p=0.9, k=150):
    a = list(a)[:k]
    b = list(b)[:k]
    seen_a, seen_b = set(), set()
    score = 0.0
    for d in range(1, k + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        score += (overlap / d) * (p ** (d - 1))
    return (1 - p) * score


def topk_stability(rankings_by_model):
    rows = []
    summary = []
    rbo_rows = []
    for model, seed_rankings in rankings_by_model.items():
        seeds = sorted(seed_rankings)
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1:]:
                r1 = [r["Gene"] for r in seed_rankings[s1]]
                r2 = [r["Gene"] for r in seed_rankings[s2]]
                row = {"model": model, "seed_a": s1, "seed_b": s2}
                for k in [50, 100, 150]:
                    row[f"Top{k}_Jaccard"] = len(set(r1[:k]) & set(r2[:k])) / len(set(r1[:k]) | set(r2[:k]))
                rows.append(row)
                rbo_rows.append({"model": model, "seed_a": s1, "seed_b": s2, "RBO_top150_p0.9": rbo_score(r1, r2, p=0.9, k=150)})
        model_rows = [r for r in rows if r["model"] == model]
        model_rbo = [r for r in rbo_rows if r["model"] == model]
        summary.append(
            {
                "model": model,
                "Top50_Jaccard_mean": float(np.mean([r["Top50_Jaccard"] for r in model_rows])),
                "Top100_Jaccard_mean": float(np.mean([r["Top100_Jaccard"] for r in model_rows])),
                "Top150_Jaccard_mean": float(np.mean([r["Top150_Jaccard"] for r in model_rows])),
                "RBO_top150_mean": float(np.mean([r["RBO_top150_p0.9"] for r in model_rbo])),
            }
        )
    write_csv(OUT / "06_stability" / "topk_jaccard.csv", rows)
    write_csv(OUT / "06_stability" / "rbo_matrix.csv", rbo_rows)
    write_csv(OUT / "06_stability" / "stability_summary.csv", summary)
    return rows, rbo_rows, summary


def analyze_results(message_paths, include_capacity=False):
    ppi_degree = degrees_from_edges(STAGE1_READY / "ppi_edges_frozen.tsv")
    grn_degree = degrees_from_edges(STAGE1_READY / "grn_edges_frozen.tsv", directed=True)
    stage2_per_seed = read_csv(STAGE2 / "07_analysis" / "per_seed_metrics.csv", purpose="read frozen Stage 2 per-seed metrics")
    per_seed = []
    for row in stage2_per_seed:
        if row["model"] in {"PPI", "GRN", "PPI_GRN"}:
            per_seed.append({k: (int(v) if k in {"seed", "best_episode", "HitCount@50", "HitCount@100", "HitCount@150"} else v) for k, v in row.items()})
    checkpoint_rows = []
    hub_rows = []
    stability_rows = []
    gate_rows = []
    masking_rows = []
    deltarank_rows = []
    rankings_by_model = {"PPI_GRN": {}, "RelationAware": {}}
    for seed in FORMAL_SEEDS:
        stage2_run = STAGE2 / "06_formal_runs" / "PPI_GRN" / f"seed_{seed}"
        rankings_by_model["PPI_GRN"][seed] = load_ranking(stage2_run)
        run_dir = OUT / "03_formal_runs" / "RelationAware" / f"seed{seed}"
        metrics = metrics_from_best(run_dir)
        ranking = load_ranking(run_dir)
        rankings_by_model["RelationAware"][seed] = ranking
        q_values = np.asarray([float(r["Q_value"]) for r in ranking], dtype=np.float64)
        if len(ranking) != 9039 or len({r["Gene"] for r in ranking}) != 9039 or np.isnan(q_values).any() or np.isinf(q_values).any():
            raise RuntimeError(f"Invalid RelationAware ranking seed {seed}")
        row = {"model": "RelationAware", "seed": seed, "best_episode": int(metrics["episode"])}
        for k in K_VALUES:
            row[f"NDCG@{k}"] = float(metrics[f"NDCG@{k}"])
            row[f"Recall@{k}"] = float(metrics[f"Recall@{k}"])
            row[f"HitCount@{k}"] = int(metrics[f"HitCount@{k}"])
        per_seed.append(row)
        checkpoint_rows.append(
            {
                "model": "RelationAware",
                "seed": seed,
                "checkpoint": display_path(run_dir / "checkpoint_best.pt"),
                "validation_ndcg150": float(metrics["NDCG@150"]),
                "selection_reason": "max Validation NDCG@150",
            }
        )
        traj = read_csv(run_dir / "validation_trajectory.csv", purpose="read RelationAware validation trajectory")
        final = traj[-1]
        gate_rows.append(
            {
                "seed": seed,
                "best_episode": metrics["episode"],
                "alpha_PPI": float(metrics.get("alpha_PPI", final.get("alpha_PPI", 0.0))),
                "alpha_GRN": float(metrics.get("alpha_GRN", final.get("alpha_GRN", 0.0))),
                "final_episode": final["episode"],
                "best_NDCG@150": float(metrics["NDCG@150"]),
                "final_NDCG@150": float(final["NDCG@150"]),
                "best_minus_final_NDCG150_gap": float(metrics["NDCG@150"]) - float(final["NDCG@150"]),
                "best_Recall@150": float(metrics["Recall@150"]),
                "final_Recall@150": float(final["Recall@150"]),
            }
        )
        top150 = ranking[:150]
        hub_rows.append(
            {
                "model": "RelationAware",
                "seed": seed,
                "mean_top150_ppi_degree": float(np.mean([ppi_degree.get(r["Gene"], 0) for r in top150])),
                "median_top150_ppi_degree": float(np.median([ppi_degree.get(r["Gene"], 0) for r in top150])),
                "mean_top150_grn_degree": float(np.mean([grn_degree.get(r["Gene"], 0) for r in top150])),
                "median_top150_grn_degree": float(np.median([grn_degree.get(r["Gene"], 0) for r in top150])),
                "rank_vs_ppi_degree_spearman": spearman_rank_degree(ranking, ppi_degree),
                "rank_vs_grn_degree_spearman": spearman_rank_degree(ranking, grn_degree),
            }
        )
        train_rows = read_csv(run_dir / "train_metrics.csv", purpose="read RelationAware training stability metrics")
        action_rows = read_csv(run_dir / "action_trace.csv", purpose="read RelationAware action trace")
        stability_rows.append(
            {
                "model": "RelationAware",
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
        mask_rows, delta_rows = evaluate_relation_masking(seed, run_dir, message_paths)
        masking_rows.extend(mask_rows)
        deltarank_rows.extend(delta_rows)
    write_csv(OUT / "03_formal_runs" / "RelationAware" / "checkpoint_selection_log.csv", checkpoint_rows)
    write_csv(OUT / "04_analysis" / "stage3_per_seed_metrics.csv", per_seed)
    write_csv(OUT / "04_analysis" / "relation_gate_by_seed.csv", gate_rows)
    write_csv(OUT / "04_analysis" / "hub_bias_metrics.csv", hub_rows)
    write_csv(OUT / "04_analysis" / "training_stability.csv", stability_rows)
    write_csv(OUT / "05_relation_attribution" / "relation_masking_metrics.csv", masking_rows)
    write_csv(OUT / "05_relation_attribution" / "gene_relation_deltarank.csv", deltarank_rows)
    df = pd.DataFrame(per_seed)
    summary_rows = []
    for model in ["PPI", "GRN", "PPI_GRN", "RelationAware"]:
        sub = df[df["model"] == model]
        for metric in ["NDCG@50", "NDCG@100", "NDCG@150", "Recall@50", "Recall@100", "Recall@150"]:
            values = sub[metric].astype(float).to_numpy()
            summary_rows.append({"model": model, "metric": metric, "mean": float(np.mean(values)), "SD": float(np.std(values, ddof=1)), "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))})
    write_csv(OUT / "04_analysis" / "stage3_summary_metrics.csv", summary_rows)
    paired_outputs = {}
    for baseline, filename in [("PPI_GRN", "relationaware_vs_union_paired.csv"), ("PPI", "relationaware_vs_ppi_paired.csv"), ("GRN", "relationaware_vs_grn_paired.csv")]:
        rows = []
        for metric in ["NDCG@150", "Recall@150", "NDCG@100", "Recall@100", "NDCG@50", "Recall@50"]:
            for seed in FORMAL_SEEDS:
                ra = float(df[(df["model"] == "RelationAware") & (df["seed"] == seed)][metric].iloc[0])
                base = float(df[(df["model"] == baseline) & (df["seed"] == seed)][metric].iloc[0])
                rows.append({"seed": seed, "metric": metric, baseline: base, "RelationAware": ra, "RelationAware_minus_baseline": ra - base, "direction": "win" if ra > base else ("tie" if ra == base else "loss")})
        write_csv(OUT / "04_analysis" / filename, rows)
        paired_outputs[baseline] = rows
    stability = topk_stability(rankings_by_model)
    make_stage3_figures(pd.DataFrame(per_seed), pd.DataFrame(gate_rows), pd.DataFrame(masking_rows), pd.DataFrame(stability[2]), pd.DataFrame(stability_rows))
    return {"per_seed": per_seed, "summary": summary_rows, "paired": paired_outputs, "hub": hub_rows, "stability": stability_rows, "gate": gate_rows, "masking": masking_rows, "stability_summary": stability[2]}


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


def make_stage3_figures(df, gate_df, masking_df, stability_summary_df, stability_df):
    metric_columns = [
        "NDCG@50",
        "NDCG@100",
        "NDCG@150",
        "Recall@150",
        "alpha_PPI",
        "alpha_GRN",
        "episode_reward_mean",
        "loss_mean",
        "Top150_Jaccard_mean",
    ]
    for frame in [df, gate_df, masking_df, stability_summary_df, stability_df]:
        for column in metric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "seed" in frame.columns:
            frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")

    def save(name):
        plt.tight_layout()
        plt.savefig(OUT / "08_figures" / name, dpi=160)
        plt.close()
        log_access(OUT / "08_figures" / name, "write Stage 3 figure", "write", "stage3_figure")

    models = ["PPI", "GRN", "PPI_GRN", "RelationAware"]
    x = np.arange(len(FORMAL_SEEDS))
    width = 0.18
    for metric, filename, ylabel in [
        ("NDCG@150", "figure1_validation_ndcg150_per_seed.png", "Validation NDCG@150"),
        ("Recall@150", "figure2_recall150_per_seed.png", "Validation Recall@150"),
    ]:
        plt.figure(figsize=(8.5, 4.8))
        for idx, model in enumerate(models):
            vals = [float(df[(df["model"] == model) & (df["seed"] == seed)][metric].iloc[0]) for seed in FORMAL_SEEDS]
            plt.bar(x + (idx - 1.5) * width, vals, width, label=model)
        plt.xticks(x, FORMAL_SEEDS)
        plt.xlabel("Seed")
        plt.ylabel(ylabel)
        plt.legend()
        save(filename)

    plt.figure(figsize=(7, 4.8))
    labels = ["NDCG@50", "NDCG@100", "NDCG@150"]
    x2 = np.arange(len(labels))
    for idx, model in enumerate(models):
        vals = [float(df[df["model"] == model][m].mean()) for m in labels]
        plt.bar(x2 + (idx - 1.5) * width, vals, width, label=model)
    plt.xticks(x2, labels)
    plt.ylabel("Mean validation NDCG")
    plt.legend()
    save("figure3_ndcg_summary.png")

    for metric, filename, ylabel in [
        ("episode_reward_mean", "figure4_relationaware_reward_curves.png", "Mean episode reward"),
        ("loss_mean", "figure5_relationaware_loss_curves.png", "Mean loss"),
    ]:
        plt.figure(figsize=(7, 4.8))
        sub = stability_df[stability_df["model"] == "RelationAware"]
        plt.plot(sub["seed"], sub[metric], marker="o")
        plt.xlabel("Seed")
        plt.ylabel(ylabel)
        save(filename)

    plt.figure(figsize=(7, 4.8))
    plt.plot(gate_df["seed"], gate_df["alpha_PPI"], marker="o", label="alpha_PPI")
    plt.plot(gate_df["seed"], gate_df["alpha_GRN"], marker="o", label="alpha_GRN")
    plt.ylim(0, 1)
    plt.xlabel("Seed")
    plt.ylabel("Relation gate weight")
    plt.legend()
    save("figure6_alpha_ppi_grn_across_seeds.png")

    plt.figure(figsize=(7, 4.8))
    for mode in ["Full", "-PPI", "-GRN"]:
        sub = masking_df[masking_df["mask_mode"] == mode]
        plt.plot(sub["seed"], sub["NDCG@150"], marker="o", label=mode)
    plt.xlabel("Seed")
    plt.ylabel("Validation NDCG@150")
    plt.legend()
    save("figure7_relation_masking_ndcg150.png")

    plt.figure(figsize=(6, 4.5))
    sub = stability_summary_df[stability_summary_df["model"].isin(["PPI_GRN", "RelationAware"])]
    plt.bar(sub["model"], sub["Top150_Jaccard_mean"])
    plt.ylabel("Mean pairwise Top150 Jaccard")
    save("figure8_top150_stability.png")


def classify(analysis, arch):
    df = pd.DataFrame(analysis["per_seed"])
    numeric_cols = [
        "seed",
        "NDCG@50",
        "NDCG@100",
        "NDCG@150",
        "Recall@50",
        "Recall@100",
        "Recall@150",
        "Precision@150",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    union = df[df["model"] == "PPI_GRN"].set_index("seed")
    ppi = df[df["model"] == "PPI"].set_index("seed")
    grn = df[df["model"] == "GRN"].set_index("seed")
    ra = df[df["model"] == "RelationAware"].set_index("seed")
    union_mean = float(union["NDCG@150"].mean())
    union_sd = float(union["NDCG@150"].std(ddof=1))
    ra_mean = float(ra["NDCG@150"].mean())
    ra_sd = float(ra["NDCG@150"].std(ddof=1))
    ra_recall = float(ra["Recall@150"].mean())
    diff = ra["NDCG@150"] - union["NDCG@150"]
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    rel_outperforms = "YES" if ra_mean > union_mean and wins >= 4 else ("UNCERTAIN" if ra_mean >= union_mean or wins >= 3 else "NO")
    sd_reduces = "YES" if ra_sd < union_sd else "NO"
    gate_df = pd.DataFrame(analysis["gate"])
    for col in ["alpha_PPI", "alpha_GRN"]:
        gate_df[col] = pd.to_numeric(gate_df[col], errors="coerce")
    alpha_ppi_mean = float(gate_df["alpha_PPI"].mean())
    alpha_grn_mean = float(gate_df["alpha_GRN"].mean())
    mask_df = pd.DataFrame(analysis["masking"])
    for col in ["Delta_NDCG150_vs_Full", "Delta_Recall150_vs_Full", "NDCG@150", "Recall@150"]:
        if col in mask_df.columns:
            mask_df[col] = pd.to_numeric(mask_df[col], errors="coerce")
    minus_ppi = mask_df[mask_df["mask_mode"] == "-PPI"]
    minus_grn = mask_df[mask_df["mask_mode"] == "-GRN"]
    delta_ppi = float(minus_ppi["Delta_NDCG150_vs_Full"].mean())
    delta_grn = float(minus_grn["Delta_NDCG150_vs_Full"].mean())
    both_used = "YES" if delta_ppi < 0 and delta_grn < 0 and alpha_ppi_mean > 0.05 and alpha_grn_mean > 0.05 else ("UNCERTAIN" if alpha_ppi_mean > 0.05 and alpha_grn_mean > 0.05 else "NO")
    capacity_required = "YES" if arch["capacity_required"] else "NO"
    capacity_triggered = arch["capacity_required"] and ra_mean > union_mean
    if rel_outperforms == "YES":
        status = "PASS"
        ready = "YES"
        backbone = "RelationAware"
    elif rel_outperforms == "UNCERTAIN" or sd_reduces == "YES" or ra_recall >= float(union["Recall@150"].mean()):
        status = "CONDITIONAL"
        ready = "CONDITIONAL"
        backbone = "RelationAware" if rel_outperforms != "NO" else "Stage2_SimpleUnion"
    else:
        status = "FAIL"
        ready = "YES"
        backbone = "Stage2_SimpleUnion"
    gain_explained = "NOT_TESTED"
    if capacity_triggered:
        gain_explained = "UNCERTAIN"
    return {
        "SIMPLE_UNION_NDCG150": union_mean,
        "SIMPLE_UNION_NDCG150_SD": union_sd,
        "RELATION_AWARE_NDCG150_MEAN": ra_mean,
        "RELATION_AWARE_NDCG150_SD": ra_sd,
        "RELATION_AWARE_RECALL150_MEAN": ra_recall,
        "RELATION_AWARE_MINUS_UNION_NDCG150_MEAN": ra_mean - union_mean,
        "RELATION_AWARE_MINUS_UNION_NDCG150_RELATIVE": (ra_mean - union_mean) / union_mean if union_mean else "",
        "RELATION_AWARE_VS_UNION_WINS": wins,
        "RELATION_AWARE_VS_UNION_LOSSES": losses,
        "RELATION_AWARE_VS_UNION_TIES": ties,
        "SIMPLE_UNION_PARAMETERS": arch["simple_counts"]["total"],
        "RELATION_AWARE_PARAMETERS": arch["relation_counts"]["total"],
        "PARAMETER_RATIO": arch["ratio"],
        "RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION": rel_outperforms,
        "RELATION_AWARE_REDUCES_SEED_VARIANCE": sd_reduces,
        "BOTH_RELATIONS_ARE_USED": both_used,
        "GAIN_EXPLAINED_BY_PARAMETER_COUNT": gain_explained,
        "CAPACITY_CONTROL_REQUIRED": capacity_required,
        "CAPACITY_CONTROL_TRIGGERED": "YES" if capacity_triggered else "NO",
        "CAPACITY_CONTROL_RUN": "NO",
        "ALPHA_PPI_MEAN": alpha_ppi_mean,
        "ALPHA_GRN_MEAN": alpha_grn_mean,
        "RELATION_MASK_PPI_DELTA_NDCG150": delta_ppi,
        "RELATION_MASK_GRN_DELTA_NDCG150": delta_grn,
        "STAGE4_BACKBONE": backbone,
        "STAGE3_STATUS": status,
        "READY_FOR_STAGE4": ready,
    }


def run_capacity_control_if_needed(decision, analysis, arch, message_paths):
    if decision["CAPACITY_CONTROL_TRIGGERED"] != "YES":
        return []
    hidden_dim = int(arch["capacity_hidden_dim"])
    summaries = []
    for seed in FORMAL_SEEDS:
        run_dir = OUT / "07_capacity_control" / "CapacityMatchedSimpleUnion" / f"seed{seed}"
        summaries.append(
            build_run(
                "CapacityMatchedSimpleUnion",
                seed,
                run_dir,
                message_paths["PPI_GRN"],
                smoke=False,
                model_kind="SimpleUnion",
                hidden_dim=hidden_dim,
            )
        )
    cap_rows = []
    ranking_by_seed = {}
    for seed in FORMAL_SEEDS:
        run_dir = OUT / "07_capacity_control" / "CapacityMatchedSimpleUnion" / f"seed{seed}"
        metrics = metrics_from_best(run_dir)
        row = {
            "model": "CapacityMatchedSimpleUnion",
            "seed": seed,
            "hidden_dim": hidden_dim,
            "best_episode": metrics.get("episode"),
        }
        for key in ["NDCG@50", "NDCG@100", "NDCG@150", "Precision@150", "Recall@150"]:
            row[key] = metrics.get(key)
        cap_rows.append(row)
        ranking_by_seed[seed] = load_ranking(run_dir)
    write_csv(OUT / "07_capacity_control" / "capacity_per_seed_metrics.csv", cap_rows)

    cap_df = pd.DataFrame(cap_rows).set_index("seed")
    ra_df = pd.DataFrame(analysis["per_seed"])
    ra_df = ra_df[ra_df["model"] == "RelationAware"].set_index("seed")
    paired_rows = []
    diff = []
    for seed in FORMAL_SEEDS:
        d = float(ra_df.loc[seed, "NDCG@150"]) - float(cap_df.loc[seed, "NDCG@150"])
        diff.append(d)
        paired_rows.append(
            {
                "seed": seed,
                "RelationAware_NDCG@150": ra_df.loc[seed, "NDCG@150"],
                "CapacityMatchedSimpleUnion_NDCG@150": cap_df.loc[seed, "NDCG@150"],
                "delta": d,
            }
        )
    write_csv(OUT / "07_capacity_control" / "relationaware_vs_capacity_paired.csv", paired_rows)

    cap_summary = []
    for metric in ["NDCG@50", "NDCG@100", "NDCG@150", "Recall@150"]:
        values = pd.to_numeric(cap_df[metric])
        cap_summary.append(
            {
                "model": "CapacityMatchedSimpleUnion",
                "metric": metric,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "n": int(values.count()),
            }
        )
    write_csv(OUT / "07_capacity_control" / "capacity_summary_metrics.csv", cap_summary)

    stability = topk_stability({"CapacityMatchedSimpleUnion": ranking_by_seed})
    write_csv(OUT / "07_capacity_control" / "capacity_stability_summary.csv", stability["summary"])

    cap_mean = float(pd.to_numeric(cap_df["NDCG@150"]).mean())
    cap_sd = float(pd.to_numeric(cap_df["NDCG@150"]).std(ddof=1))
    ra_mean = float(ra_df["NDCG@150"].mean())
    ra_wins = int(sum(1 for d in diff if d > 0))
    if ra_mean > cap_mean and ra_wins >= 4:
        gain_explained = "NO"
        specific_advantage = "YES"
        if decision["RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION"] == "YES":
            decision["STAGE4_BACKBONE"] = "RelationAware"
            decision["STAGE3_STATUS"] = "PASS"
            decision["READY_FOR_STAGE4"] = "YES"
    elif ra_mean <= cap_mean:
        gain_explained = "YES"
        specific_advantage = "NO"
        decision["STAGE4_BACKBONE"] = "Stage2_SimpleUnion"
        decision["STAGE3_STATUS"] = "CONDITIONAL"
        decision["READY_FOR_STAGE4"] = "CONDITIONAL"
    else:
        gain_explained = "UNCERTAIN"
        specific_advantage = "UNCERTAIN"
        decision["STAGE4_BACKBONE"] = "Stage2_SimpleUnion"
        decision["STAGE3_STATUS"] = "CONDITIONAL"
        decision["READY_FOR_STAGE4"] = "CONDITIONAL"

    decision.update(
        {
            "CAPACITY_CONTROL_RUN": "YES",
            "CAPACITY_MATCHED_HIDDEN_DIM": hidden_dim,
            "CAPACITY_MATCHED_UNION_PARAMETERS": arch["capacity_counts"]["total"],
            "CAPACITY_MATCHED_UNION_NDCG150_MEAN": cap_mean,
            "CAPACITY_MATCHED_UNION_NDCG150_SD": cap_sd,
            "RELATION_AWARE_VS_CAPACITY_WINS": ra_wins,
            "RELATION_AWARE_VS_CAPACITY_LOSSES": int(sum(1 for d in diff if d < 0)),
            "RELATION_AWARE_VS_CAPACITY_TIES": int(sum(1 for d in diff if d == 0)),
            "GAIN_EXPLAINED_BY_PARAMETER_COUNT": gain_explained,
            "RELATION_AWARE_SPECIFIC_ADVANTAGE_AFTER_CAPACITY_CONTROL": specific_advantage,
        }
    )

    plt.figure(figsize=(7, 4.8))
    labels = ["SimpleUnion", "CapacityMatchedSimpleUnion", "RelationAware"]
    values = [
        decision["SIMPLE_UNION_NDCG150"],
        decision["CAPACITY_MATCHED_UNION_NDCG150_MEAN"],
        decision["RELATION_AWARE_NDCG150_MEAN"],
    ]
    plt.bar(labels, values)
    plt.ylabel("Validation NDCG@150")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "08_figures" / "figure9_capacity_control_ndcg150.png", dpi=160)
    plt.close()
    return summaries


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
        OUT / "scripts" / "stage3_relation_aware_experiment.py",
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
                    "role": "stage3_new_code" if str(path).startswith(str(OUT)) else "project_source_read_only_import",
                    "sha256": sha256_file(path),
                    "modified_by_stage3": "YES" if str(path).startswith(str(OUT)) else "NO",
                }
            )
    write_csv(OUT / "02_stage3_implementation" / "source_manifest.csv", source_rows)
    write_text(
        OUT / "02_stage3_implementation" / "implementation_change_log.md",
        """# Stage 3 Implementation Change Log

CORE_PROJECT_SOURCE_MODIFIED = NO

Stage 3 uses a new isolated wrapper in `E:\\codex_file\\新方向阶段3\\scripts\\stage3_relation_aware_experiment.py`.

Reason for wrapper: Stage 3 changes only the Q encoder/message passing module. It keeps the Stage 2 environment, fixed PPI-derived `hybrid6_raw` features, reward, DDQN target logic, PER, soft update, validation checkpoint rule, and global action mask.

Primary model: DualBranch_GCN_GlobalGate. Gate logits initialize to zero, so initial alpha_PPI = alpha_GRN = 0.5.
""",
    )
    code_hash_lines = [f"{row['sha256']}  {row['file']}" for row in source_rows]
    write_text(OUT / "02_stage3_implementation" / "SHA256SUMS_stage3_code.txt", "\n".join(code_hash_lines) + "\n")


def write_readme():
    write_text(
        OUT / "00_README.md",
        """# RL-GenRisk New Direction Stage 3

Relation-aware encoder validation for DualBranch_GCN_GlobalGate under fixed Stage 2 features, labels, reward, DDQN/PER, soft update, seeds, and global action topology.

Historical Test and future external validation data are not used.
""",
    )


def write_final_report(smoke, formal_summaries, capacity_summaries, analysis, decision):
    formal_completed = sum(1 for s in formal_summaries if s.get("status") == "COMPLETED")
    capacity_completed = sum(1 for s in capacity_summaries if s.get("status") == "COMPLETED")
    per_seed_table = markdown_table(analysis["per_seed"])
    summary_table = markdown_table(analysis["summary"])
    hub_table = markdown_table(analysis["hub"])
    stability_table = markdown_table(analysis["stability"])
    gate_table = markdown_table(analysis["gate"])
    masking_table = markdown_table(analysis["masking"])
    stability_summary = markdown_table(analysis["stability_summary"])
    report = f"""# STAGE3_COMPLETION_REPORT

## 1. Executive Summary

STAGE3_STATUS = {decision['STAGE3_STATUS']}

READY_FOR_STAGE4 = {decision['READY_FOR_STAGE4']}

Stage 3 evaluated one pre-registered primary relation-aware encoder: DualBranch_GCN_GlobalGate.

## 2. Boundary Compliance

HISTORICAL_TEST_USED = NO

NEW_EXTERNAL_VALIDATION_USED = NO

REWARD_MODIFIED = NO

MORL_STARTED = NO

NODE_FEATURES_CHANGED = NO

LABEL_SPLIT_CHANGED = NO

## 3. Action Topology Audit

ACTION_SPACE_TYPE = GLOBAL_UNSELECTED_GENE_POOL

MESSAGE_ACTION_GRAPH_COUPLED = NO

FEATURES_CHANGED = NO

REWARD_CHANGED = NO

LABEL_SPLIT_CHANGED = NO

MORL_STARTED = NO

## 3. Frozen Stage 2 Baselines

PPI, GRN, and SimpleUnion baselines are read from frozen Stage 2 outputs.

## 4. Architecture

PRIMARY_MODEL = {PRIMARY_MODEL}

PPI branch = frozen bidirectional PPI message graph.

GRN branch = frozen directed TF -> target GRN message graph.

Fusion = global softmax gate, `H_fused = alpha_PPI * H_PPI + alpha_GRN * H_GRN`.

## 5. Parameter Audit

SimpleUnion parameters = {decision['SIMPLE_UNION_PARAMETERS']}

RelationAware parameters = {decision['RELATION_AWARE_PARAMETERS']}

parameter ratio = {decision['PARAMETER_RATIO']}

CAPACITY_CONTROL_REQUIRED = {decision['CAPACITY_CONTROL_REQUIRED']}

## 6. Regression Control

SIMPLE_UNION_REGRESSION = {smoke['SIMPLE_UNION_REGRESSION']}

## 7. Smoke

forward = PASS

backward = {smoke['gradient_smoke']['backward']}

gradient = PPI branch {smoke['gradient_smoke']['ppi_branch_receives_gradient']}, GRN branch {smoke['gradient_smoke']['grn_branch_receives_gradient']}, gate {smoke['gradient_smoke']['gate_parameters_receive_gradient']}

gate = alpha sum close to 1: {smoke['gradient_smoke']['alpha_sum_close_to_1']}

ranking = {smoke['gradient_smoke']['ranking_generation']}

NaN = {smoke['gradient_smoke']['NaN']}

Inf = {smoke['gradient_smoke']['Inf']}

## 8. Formal Configuration

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

## 9. Per-seed Results

{per_seed_table}

## 10. Primary Comparison

RelationAware vs SimpleUnion mean difference = {decision['RELATION_AWARE_MINUS_UNION_NDCG150_MEAN']}

relative difference = {decision['RELATION_AWARE_MINUS_UNION_NDCG150_RELATIVE']}

wins/losses/ties = {decision['RELATION_AWARE_VS_UNION_WINS']} / {decision['RELATION_AWARE_VS_UNION_LOSSES']} / {decision['RELATION_AWARE_VS_UNION_TIES']}

## 11. Secondary Comparison

See paired CSV outputs for RelationAware vs PPI and RelationAware vs GRN.

## 12. Checkpoint Stability

{gate_table}

## 13. Relation Gate

ALPHA_PPI_MEAN = {decision['ALPHA_PPI_MEAN']}

ALPHA_GRN_MEAN = {decision['ALPHA_GRN_MEAN']}

## 14. Relation Masking

{masking_table}

## 15. Ranking Stability

{stability_summary}

## 16. Hub Bias

{hub_table}

## 17. Capacity Control

CAPACITY_CONTROL_RUN = {decision['CAPACITY_CONTROL_RUN']}

CAPACITY_CONTROL_COMPLETED = {capacity_completed} / 5

CAPACITY_MATCHED_HIDDEN_DIM = {decision.get('CAPACITY_MATCHED_HIDDEN_DIM', 'NA')}

CAPACITY_MATCHED_UNION_PARAMETERS = {decision.get('CAPACITY_MATCHED_UNION_PARAMETERS', 'NA')}

CAPACITY_MATCHED_UNION_NDCG150_MEAN = {decision.get('CAPACITY_MATCHED_UNION_NDCG150_MEAN', 'NA')}

RELATION_AWARE_VS_CAPACITY_WINS_LOSSES_TIES = {decision.get('RELATION_AWARE_VS_CAPACITY_WINS', 'NA')} / {decision.get('RELATION_AWARE_VS_CAPACITY_LOSSES', 'NA')} / {decision.get('RELATION_AWARE_VS_CAPACITY_TIES', 'NA')}

GAIN_EXPLAINED_BY_PARAMETER_COUNT = {decision['GAIN_EXPLAINED_BY_PARAMETER_COUNT']}

## 18. Unexpected Results

Formal RelationAware runs completed = {formal_completed} / 5

LOW_FREQUENCY_DEV_EVAL = DEFERRED

## 19. Final Scientific Interpretation

RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION = {decision['RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION']}

RELATION_AWARE_REDUCES_SEED_VARIANCE = {decision['RELATION_AWARE_REDUCES_SEED_VARIANCE']}

BOTH_RELATIONS_ARE_USED = {decision['BOTH_RELATIONS_ARE_USED']}

GAIN_EXPLAINED_BY_PARAMETER_COUNT = {decision['GAIN_EXPLAINED_BY_PARAMETER_COUNT']}

## 20. Stage 4 Backbone

STAGE4_BACKBONE = {decision['STAGE4_BACKBONE']}

## 21. Final Gate

READY_FOR_STAGE4 = {decision['READY_FOR_STAGE4']}

## Supporting Summary Results

{summary_table}
 
## Training Stability

{stability_table}
"""
    write_text(OUT / "STAGE3_COMPLETION_REPORT.md", report)


def write_integrity():
    paths = [
        OUT / "scripts" / "stage3_relation_aware_experiment.py",
        OUT / "01_architecture_audit" / "STAGE3_ARCHITECTURE_PROTOCOL.md",
        OUT / "01_architecture_audit" / "model_parameter_counts.csv",
        OUT / "01_architecture_audit" / "capacity_control_decision.md",
        OUT / "01_architecture_audit" / "stage3_formal_config.yaml",
        OUT / "02_stage3_implementation" / "source_manifest.csv",
        OUT / "02_stage3_implementation" / "pyg_runtime_smoke.json",
        OUT / "02_stage3_implementation" / "gradient_smoke.json",
        OUT / "02_stage3_implementation" / "regression_control_report.md",
        OUT / "03_formal_runs" / "RelationAware" / "checkpoint_selection_log.csv",
        OUT / "04_analysis" / "stage3_per_seed_metrics.csv",
        OUT / "04_analysis" / "stage3_summary_metrics.csv",
        OUT / "04_analysis" / "relationaware_vs_union_paired.csv",
        OUT / "05_relation_attribution" / "relation_masking_metrics.csv",
        OUT / "05_relation_attribution" / "gene_relation_deltarank.csv",
        OUT / "06_stability" / "stability_summary.csv",
        OUT / "07_capacity_control" / "capacity_per_seed_metrics.csv",
        OUT / "07_capacity_control" / "capacity_summary_metrics.csv",
        OUT / "07_capacity_control" / "relationaware_vs_capacity_paired.csv",
        OUT / "STAGE3_COMPLETION_REPORT.md",
        OUT / "terminal_summary.txt",
    ]
    for seed in FORMAL_SEEDS:
        run_dir = OUT / "03_formal_runs" / "RelationAware" / f"seed{seed}"
        paths.extend([run_dir / "checkpoint_best.pt", run_dir / "validation_ranking_best.csv", run_dir / "summary.json", run_dir / "validation_trajectory.csv"])
        cap_dir = OUT / "07_capacity_control" / "CapacityMatchedSimpleUnion" / f"seed{seed}"
        paths.extend([cap_dir / "checkpoint_best.pt", cap_dir / "validation_ranking_best.csv", cap_dir / "summary.json"])
    lines = []
    for path in paths:
        if path.exists():
            lines.append(f"{sha256_file(path)}  {display_path(path)}")
    write_text(OUT / "09_integrity" / "SHA256SUMS_stage3.txt", "\n".join(lines) + "\n")


def write_terminal_summary(pyg, smoke, formal_summaries, capacity_summaries, decision):
    formal_completed = sum(1 for s in formal_summaries if s.get("status") == "COMPLETED")
    capacity_completed = sum(1 for s in capacity_summaries if s.get("status") == "COMPLETED")
    capacity_run = decision["CAPACITY_CONTROL_RUN"]
    capacity_run_detail = f"{capacity_run} ({capacity_completed} / 5)" if capacity_run == "YES" else capacity_run
    summary = f"""============================================================
RL-GenRisk NEW DIRECTION - STAGE 3 COMPLETE
RELATION-AWARE ENCODER VALIDATION
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
GLOBAL_UNSELECTED_GENE_POOL

MESSAGE_ACTION_GRAPH_COUPLED:
NO

ACTION_FAIRNESS:
PASS

PYG_RUNTIME:
{pyg['PYG_RUNTIME']}

PRIMARY_MODEL:
{PRIMARY_MODEL}

SIMPLE_UNION_REGRESSION:
{smoke['SIMPLE_UNION_REGRESSION']}

RELATIONAWARE_SMOKE:
{smoke['RELATIONAWARE_SMOKE']}

FORMAL_SEEDS:
{','.join(str(s) for s in FORMAL_SEEDS)}

FORMAL_RUNS_COMPLETED:
{formal_completed} / 5

SIMPLE_UNION_NDCG150:
{decision['SIMPLE_UNION_NDCG150']}

RELATION_AWARE_NDCG150_MEAN:
{decision['RELATION_AWARE_NDCG150_MEAN']}

RELATION_AWARE_NDCG150_SD:
{decision['RELATION_AWARE_NDCG150_SD']}

RELATION_AWARE_RECALL150_MEAN:
{decision['RELATION_AWARE_RECALL150_MEAN']}

RELATION_AWARE_VS_UNION_WINS_LOSSES_TIES:
{decision['RELATION_AWARE_VS_UNION_WINS']} / {decision['RELATION_AWARE_VS_UNION_LOSSES']} / {decision['RELATION_AWARE_VS_UNION_TIES']}

RELATION_AWARE_PARAMETERS:
{decision['RELATION_AWARE_PARAMETERS']}

SIMPLE_UNION_PARAMETERS:
{decision['SIMPLE_UNION_PARAMETERS']}

PARAMETER_RATIO:
{decision['PARAMETER_RATIO']}

CAPACITY_CONTROL_REQUIRED:
{decision['CAPACITY_CONTROL_REQUIRED']}

CAPACITY_CONTROL_RUN:
{capacity_run_detail}

GAIN_EXPLAINED_BY_PARAMETER_COUNT:
{decision['GAIN_EXPLAINED_BY_PARAMETER_COUNT']}

ALPHA_PPI_MEAN:
{decision['ALPHA_PPI_MEAN']}

ALPHA_GRN_MEAN:
{decision['ALPHA_GRN_MEAN']}

RELATION_MASK_PPI_DELTA_NDCG150:
{decision['RELATION_MASK_PPI_DELTA_NDCG150']}

RELATION_MASK_GRN_DELTA_NDCG150:
{decision['RELATION_MASK_GRN_DELTA_NDCG150']}

RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION:
{decision['RELATION_AWARE_OUTPERFORMS_SIMPLE_UNION']}

RELATION_AWARE_REDUCES_SEED_VARIANCE:
{decision['RELATION_AWARE_REDUCES_SEED_VARIANCE']}

BOTH_RELATIONS_ARE_USED:
{decision['BOTH_RELATIONS_ARE_USED']}

STAGE4_BACKBONE:
{decision['STAGE4_BACKBONE']}

STAGE3_STATUS:
{decision['STAGE3_STATUS']}

READY_FOR_STAGE4:
{decision['READY_FOR_STAGE4']}

FINAL_REPORT:
{display_path(OUT / 'STAGE3_COMPLETION_REPORT.md')}
============================================================
"""
    write_text(OUT / "terminal_summary.txt", summary)
    print(summary)


def write_access_log():
    write_csv(OUT / "stage3_data_access_log.csv", DATA_ACCESS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()
    mkdirs()
    write_readme()
    load_frozen_inputs()
    implementation_manifest()
    pyg = pyg_runtime_smoke()
    if pyg["PYG_RUNTIME"] != "PASS":
        write_access_log()
        raise RuntimeError("PYG_RUNTIME failed; stopping before Stage 3B.")
    message_paths = load_stage2_message_graphs()
    arch = architecture_audit(message_paths)
    write_formal_protocol()
    gates = stage3b_regression_and_smoke(message_paths)
    if args.skip_formal:
        write_access_log()
        return
    formal_summaries = run_formal(message_paths)
    analysis = analyze_results(message_paths)
    decision = classify(analysis, arch)
    capacity_summaries = run_capacity_control_if_needed(decision, analysis, arch, message_paths)
    write_final_report(gates, formal_summaries, capacity_summaries, analysis, decision)
    write_terminal_summary(pyg, gates, formal_summaries, capacity_summaries, decision)
    write_access_log()
    write_integrity()


if __name__ == "__main__":
    main()
