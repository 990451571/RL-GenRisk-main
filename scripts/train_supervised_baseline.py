#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态监督基线：把「RL 用奖励学」换成「用已知 driver 标签学」。

要回答的问题：
  RL 学不进去（loss 不降、训练前≈训练后），到底是特征/图没有信号，
  还是 RL 这个学习机制本身用不上信号？

做法：
  复用与 RL 完全相同的 Q 网络（同一 GCN 图编码器 + 打分头，同一维度、同一初始化种子），
  把损失从「TD 奖励误差」换成「16 个已知 train driver = 正样本」的二分类 BCE，
  训练后在同一 25 个验证 driver 上评估 —— 与 RL 的评估方式完全一致。

严格防泄漏：
  - 正样本只来自 train_driver_genes（16 个），validation_driver_genes（25 个）完全不参与训练；
  - 无验证集早停：默认固定训练 N 轮，用最后一轮模型评估（避免把验证集信息泄漏进选择过程）。

对比模型：
  1. 监督 GCN  —— 复用 Q_Fun（图编码器 + 打分头），BCE 训练
  2. 监督 MLP  —— 同样的原始特征、同样的打分思路，但去掉图卷积（检验 PPI 图是否贡献信号）
  3. （对照）静态基线 —— 突变频率 / 度数 / 低频证据分 V2（完全不训练）
  4. （对照）RL 结果 —— 已由 evaluate_greedy_rollout.py 产出，脚本末尾读入汇总对比

用法：
  python scripts/train_supervised_baseline.py \
      --run-dir outputs/exp1_legacy_seed42/hybrid6_raw/<run> \
      --output outputs/supervised_eval \
      --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent

V2_EVIDENCE_PATH = Path(
    "/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="任一 run 目录（用于取 config/seed/特征，不加载 checkpoint）")
    parser.add_argument("--output", default=str(REPO_DIR / "outputs" / "supervised_eval"))
    parser.add_argument("--epochs", type=int, default=200, help="监督训练轮数（全图 batch，很快）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "config.json").exists():
        sys.exit(f"找不到 config.json：{run_dir}")

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cargs = SimpleNamespace(**cfg["args"])

    os.environ.setdefault("PYTHONHASHSEED", str(cargs.seed))
    sys.argv = [sys.argv[0], "--seed", str(cargs.seed), "--device", args.device]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(REPO_DIR / "src"))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from qfunction import Q_Fun
    import train  # noqa: E402

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备：{device}，seed={cargs.seed}，epochs={args.epochs}")

    # ---------- 重建环境（与训练同源：同 seed、同 build 顺序） ----------
    with tempfile.TemporaryDirectory(prefix="supervised_") as tmp:
        train.set_seed(cargs.seed)
        env = train.build_environment(cargs, Path(tmp), normalization_metadata=None)

    stored_report = cfg["feature_report"]
    new_report = env["feature_report"]
    checks = {
        "feature_dim": stored_report["feature_dim"] == new_report["feature_dim"],
        "shape": stored_report["node_features_shape"] == new_report["node_features_shape"],
    }
    if not all(checks.values()):
        print(f"  [警告] 重建特征与训练记录不一致：{checks}")
    else:
        print("  [校验] 重建特征与训练记录一致 ✓")

    features = env["node_features"]  # (n, 6) numpy
    gene_name = list(env["gene_name"])
    n = len(gene_name)
    net = env["net"]  # 稠密邻接矩阵 → Q_Fun 内部转 edge_index
    train_drivers = set(env["train_driver_genes"])
    val_drivers = set(env["validation_driver_genes"])
    topk = int(cargs.topk)
    print(f"节点数={n}，train driver={len(train_drivers)}，val driver={len(val_drivers)}，topk={topk}")

    # 标签：1 = train driver，0 = 其余（验证 driver 不参与）
    y_np = np.array([1.0 if g in train_drivers else 0.0 for g in gene_name], dtype=np.float32)
    n_pos = int(y_np.sum())
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"正样本={n_pos}，负样本={n_neg}，pos_weight={n_neg / n_pos:.1f}")

    x_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_np, device=device)
    mask_ones = torch.ones(n, dtype=torch.long, device=device)

    # ---------- 静态基线（对照，不训练） ----------
    import pandas as pd
    ev_df = pd.read_csv(V2_EVIDENCE_PATH)
    ev_score = dict(zip(ev_df["Gene"], ev_df["LowFrequencyEvidenceScoreV2"]))
    static_orders = {
        "static_degree": sorted(range(n), key=lambda i: (-features[i, 0], gene_name[i])),
        "static_mutation": sorted(range(n), key=lambda i: (-features[i, 3], gene_name[i])),
        "static_evidenceV2": sorted(
            range(n), key=lambda i: (-ev_score.get(gene_name[i], -1.0), gene_name[i])
        ),
    }

    # ---------- 监督模型 ----------
    def build_supervised_gcn():
        # 复用 Q_Fun = 与 RL 完全相同的网络（图编码器 + 打分头），仅换损失
        m = Q_Fun(env["feature_report"]["feature_dim"], cargs.embedding_size, 3, cargs.learning_rate, net)
        return m.to(device)

    class MLPNoGraph(nn.Module):
        """同样的原始特征 + 同样深度的打分，但没有图卷积。"""
        def __init__(self, in_dim, hid_dim):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hid_dim)
            self.lin2 = nn.Linear(hid_dim, hid_dim)
            self.head = nn.Linear(hid_dim, 1)
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            h = F.relu(self.lin1(x))
            h = self.dropout(h)
            h = F.relu(self.lin2(h))
            return self.head(h).squeeze(-1)

    def evaluate(model, tag):
        model.eval()
        with torch.no_grad():
            logits, _ = model(None, x_t, mask_ones) if isinstance(model, Q_Fun) else (model(x_t), None)
        scores = logits.detach().cpu().numpy()
        order = sorted(range(n), key=lambda i: (-scores[i], gene_name[i]))
        rows = [{"Gene": gene_name[i]} for i in order]
        item = train.metrics_at_k(rows, val_drivers, topk)
        rank_by_gene = {gene_name[i]: r for r, i in enumerate(order, 1)}
        present = [rank_by_gene[g] for g in val_drivers if g in rank_by_gene]
        mrr = float(np.mean([1.0 / r for r in present])) if present else 0.0
        mean_rank = float(np.mean(present)) if present else None
        # 训练集记忆程度
        train_found = [g for g in train_drivers if rank_by_gene.get(g, 1e9) <= topk]
        print(f"    [{tag}] NDCG@150={item['NDCG']:.4f} Recall={item['Recall']:.3f} "
              f"Hits={item['HitCount']} MRR={mrr:.4f} "
              f"MeanRank={mean_rank if mean_rank else 'n/a':>7}"
              f" | 训练集命中 {len(train_found)}/{len(train_drivers)}")
        return {
            "Method": tag, "NDCG@150": item["NDCG"], "Recall@150": item["Recall"],
            "Precision@150": item["Precision"], "HitCount@150": item["HitCount"],
            "MRR": mrr, "MeanRank": mean_rank,
        }, order, scores

    def train_supervised(model, tag, optimizer):
        print(f"\n=== 训练监督模型：{tag} ===")
        summary = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(None, x_t, mask_ones) if isinstance(model, Q_Fun) else (model(x_t), None)
            loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cargs.gradient_clip)
            optimizer.step()
            if epoch % 25 == 0 or epoch == 1:
                m, order, scores = evaluate(model, f"{tag}_ep{epoch}")
                m["Epoch"] = epoch
                m["TrainBCE"] = float(loss.item())
                summary.append(m)
        # 最终模型（固定轮数，无验证集早停）
        final_m, final_order, final_scores = evaluate(model, f"{tag}_final")
        final_m["Epoch"] = args.epochs
        summary.append(final_m)
        return summary, final_order, final_scores

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rankings_dir = output_root / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # 静态基线指标
    for tag, order in static_orders.items():
        m, _, _ = evaluate_manual(tag, order, val_drivers, train.metrics_at_k, gene_name, topk)
        m["Epoch"] = "-"
        m["TrainBCE"] = ""
        all_rows.append(m)

    # 监督 GCN —— 直接紧跟 build_environment 之后建网，不重置随机数，
    # 使初始化与 RL 训练（同 seed）的网络完全同源，只有「学习方式」不同。
    gcn = build_supervised_gcn()
    gcn_summary, gcn_order, gcn_scores = train_supervised(
        gcn, "supervised_GCN",
        optim.Adam(gcn.parameters(), lr=cargs.learning_rate, weight_decay=1e-4),
    )
    all_rows.extend(gcn_summary)
    _write_csv(rankings_dir, f"supervised_GCN_final.csv", gcn_order, gcn_scores, gene_name)

    # 监督 MLP（无图）—— 独立固定种子，与 GCN 无关
    torch.manual_seed(0)
    np.random.seed(0)
    mlp = MLPNoGraph(env["feature_report"]["feature_dim"], cargs.embedding_size).to(device)
    mlp_summary, mlp_order, mlp_scores = train_supervised(
        mlp, "supervised_MLP_no_graph",
        optim.Adam(mlp.parameters(), lr=cargs.learning_rate, weight_decay=1e-4),
    )
    all_rows.extend(mlp_summary)
    _write_csv(rankings_dir, f"supervised_MLP_final.csv", mlp_order, mlp_scores, gene_name)

    # 汇总表
    cols = ["Method", "Epoch", "TrainBCE", "NDCG@150", "Recall@150", "Precision@150",
            "HitCount@150", "MRR", "MeanRank"]
    table_path = output_root / "supervised_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n监督模型汇总已写入：{table_path}")

    # 读入 RL 结果做三方对比
    rl_path = REPO_DIR / "outputs" / "rollout_eval" / "summary_metrics.csv"
    if rl_path.exists():
        print("\n=== 三方对比：RL / 静态 / 监督（验证集指标）===")
        rl = pd.read_csv(rl_path)
        rl_best = rl[rl["Method"].isin(["best_Q_onepass", "best_Q_rollout"])]
        rl_rows = rl_best.groupby("Run")["HitCount@150"].max().reset_index()
        print(f"  RL 三个 run 各自的最高命中数：{dict(zip(rl_rows.Run, rl_rows['HitCount@150']))}")
        gcn_final = next(r for r in gcn_summary if r["Epoch"] == args.epochs)
        mut = next(r for r in all_rows if r["Method"] == "static_mutation")
        print(f"  RL 最好   ：{rl_best['HitCount@150'].max()} 个（3 run 中最高）")
        print(f"  静态突变  ：{mut['HitCount@150']} 个")
        print(f"  监督 GCN  ：{gcn_final['HitCount@150']} 个")


def evaluate_manual(tag, order, val_drivers, metrics_at_k, gene_name, topk):
    rows = [{"Gene": gene_name[i]} for i in order]
    item = metrics_at_k(rows, val_drivers, topk)
    rank_by_gene = {gene_name[i]: r for r, i in enumerate(order, 1)}
    present = [rank_by_gene[g] for g in val_drivers if g in rank_by_gene]
    mrr = float(np.mean([1.0 / r for r in present])) if present else 0.0
    mean_rank = float(np.mean(present)) if present else None
    print(f"    [{tag}] NDCG@150={item['NDCG']:.4f} Recall={item['Recall']:.3f} "
          f"Hits={item['HitCount']} MRR={mrr:.4f}")
    return {"Method": tag, "NDCG@150": item["NDCG"], "Recall@150": item["Recall"],
            "Precision@150": item["Precision"], "HitCount@150": item["HitCount"],
            "MRR": mrr, "MeanRank": mean_rank}, order, None


def _write_csv(rankings_dir, filename, order, scores, gene_name):
    path = rankings_dir / filename
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Rank", "Gene", "Score"])
        for r, i in enumerate(order, 1):
            val = "" if scores is None else float(scores[i])
            w.writerow([r, gene_name[i], val])
    return path


if __name__ == "__main__":
    main()
