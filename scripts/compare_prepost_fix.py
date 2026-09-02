#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 修复验证：训练前 vs 训练后的排名对比（2026-09-01）。

回答：修完机制后，训练到底有没有让排名变好，还是「学了跟没学一样」。

输入（由 scripts/run_mechanism_fix_validation.sh 的评估步骤产出）：
  <eval-dir>/summary_metrics.csv        —— pretrain/best 的 NDCG@150 / Recall@150 / HitCount@150 / MRR
  <eval-dir>/rankings/<run>_pretrain_Q_onepass.csv
  <eval-dir>/rankings/<run>_best_Q_onepass.csv

输出：
  <output>：每个 run 一张表 + 三种子汇总：
    - NDCG/Recall/HitCount/MRR：训练前(pretrain) → 训练后(best)，以及绝对/相对变化
    - Top-150 重合：两者 top-150 的交集个数 + Jaccard（0=完全不同，1=完全一样）
    - rank correlation：对两者 top-150 并集基因的 Spearman 秩相关（+1=排序完全一致，0=无关，-1=反序）

判读口径：
  - 修复有效：NDCG/Recall 明显上升（如 +0.05 以上），同时 top-150 重合率下降、
    rank corr 明显低于 1 → 训练真的在改排序，且改的方向和评价一致。
  - 修复无效：前后 NDCG/Recall 几乎不动，top-150 重合≈150、rank corr≈1 → 学到=没学。
  - 病态：NDCG 下降但 reward 上升 → 训练目标仍与评价目标冲突（见 README）。

用法：
  python scripts/compare_prepost_fix.py \
      --eval-dir outputs/fix_rollout_eval \
      --output outputs/fix_rollout_eval/prepost_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent


def load_rankings(csv_path: Path) -> list[str]:
    """读取 Rank,Gene,Score 排名文件，返回按 rank 升序的基因名列表。"""
    genes = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            genes.append(row["Gene"])
    return genes


def spearman_ranks(genes_a: list[str], genes_b: list[str], topk: int) -> float:
    """在 A、B 的 top-k 并集基因上计算 Spearman 秩相关（用各自全排名的名次）。

    只选两边都考虑过的基因（top-k 并集），避免大量「双方都排很后」的
    平庸一致项把相关系数顶到接近 1。
    """
    rank_a = {g: r for r, g in enumerate(genes_a, 1)}
    rank_b = {g: r for r, g in enumerate(genes_b, 1)}
    union = list({g for g in genes_a[:topk]} | {g for g in genes_b[:topk]})
    if len(union) < 2:
        return 0.0
    ra = np.array([rank_a[g] for g in union], dtype=float)
    rb = np.array([rank_b[g] for g in union], dtype=float)
    return float(pd.Series(ra).corr(pd.Series(rb), method="spearman"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True,
                        help="evaluate_greedy_rollout.py 的输出目录（含 summary_metrics.csv 与 rankings/）")
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    parser.add_argument("--topk", type=int, default=150, help="评价 K，默认 150")
    parser.add_argument("--run-prefix", default="fix_legacy_seed",
                        help="只看以此开头的 run；默认 fix_legacy_seed。"
                             "对旧输出用 exp1_legacy_seed 可得到「修复前」的同样对比")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    summ_path = eval_dir / "summary_metrics.csv"
    rank_dir = eval_dir / "rankings"
    if not summ_path.exists() or not rank_dir.exists():
        sys.exit(f"找不到评估输出：{summ_path} 或 {rank_dir}")

    df = pd.read_csv(summ_path)
    runs = sorted({r for r in df["Run"] if r.startswith(args.run_prefix)})
    if not runs:
        sys.exit(f"{eval_dir} 里没有 fix_legacy_seed* 的 run")

    rows = []
    for run in runs:
        pre = df[(df["Run"] == run) & (df["Method"] == "pretrain_Q_onepass")]
        post = df[(df["Run"] == run) & (df["Method"] == "best_Q_onepass")]
        if pre.empty or post.empty:
            print(f"  [跳过] {run}：缺 pretrain_Q_onepass 或 best_Q_onepass")
            continue
        pre = pre.iloc[0]
        post = post.iloc[0]

        genes_pre = load_rankings(rank_dir / f"{run}_pretrain_Q_onepass.csv")
        genes_post = load_rankings(rank_dir / f"{run}_best_Q_onepass.csv")
        top_pre = set(genes_pre[: args.topk])
        top_post = set(genes_post[: args.topk])
        inter = len(top_pre & top_post)
        union = len(top_pre | top_post)

        def delta(field):
            a, b = float(pre[field]), float(post[field])
            rel = (b - a) / a if a > 1e-9 else float("nan")
            return b - a, rel

        for label, field in [("NDCG@150", "NDCG@150"), ("Recall@150", "Recall@150"),
                             ("HitCount@150", "HitCount@150"), ("MRR", "MRR")]:
            abs_d, rel_d = delta(field)
            rows.append({
                "Run": run, "Metric": label,
                "Pretrain": pre[field], "PostTrain": post[field],
                "Delta": round(abs_d, 4), "RelDeltaPct": round(rel_d * 100, 1),
            })
        # 排名位移
        rows.append({"Run": run, "Metric": "Top150_Intersection", "Pretrain": inter,
                     "PostTrain": args.topk, "Delta": inter, "RelDeltaPct": ""})
        rows.append({"Run": run, "Metric": "Top150_Jaccard", "Pretrain": round(inter / union, 3),
                     "PostTrain": round(inter / union, 3), "Delta": "", "RelDeltaPct": ""})
        rows.append({"Run": run, "Metric": "RankCorr_Spearman", "Pretrain": "",
                     "PostTrain": "", "Delta": round(spearman_ranks(genes_pre, genes_post, args.topk), 3),
                     "RelDeltaPct": ""})

    if not rows:
        sys.exit("没有任何可比对的 run")
    out = pd.DataFrame(rows)

    # 三种子汇总（均值）
    numeric = out[pd.to_numeric(out["Delta"], errors="coerce").notna()].copy()
    numeric["Delta"] = pd.to_numeric(numeric["Delta"])
    mean_delta = numeric.groupby("Metric")["Delta"].mean().round(3).rename("MeanDelta3Seed")
    mean_delta = mean_delta.reset_index()
    mean_delta["Run"] = "__MEAN__"
    summary = pd.concat([out, mean_delta], ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"前后对比已写入：{args.output}\n")

    # 控制台摘要
    ndcg = out[(out["Run"] != "__MEAN__") & (out["Metric"] == "NDCG@150")]
    rec = out[(out["Metric"] == "Recall@150") & (out["Run"] != "__MEAN__")]
    hits = out[(out["Metric"] == "HitCount@150") & (out["Run"] != "__MEAN__")]
    inter = out[(out["Metric"] == "Top150_Intersection") & (out["Run"] != "__MEAN__")]
    corr = out[(out["Metric"] == "RankCorr_Spearman") & (out["Run"] != "__MEAN__")]
    print("=== 修复验证：训练前 → 训练后（每种子）===")
    for i, run in enumerate([r for r in runs if r in set(ndcg["Run"])]):
        print(f"  {run}: NDCG {ndcg['Pretrain'].iloc[i]:.3f}→{ndcg['PostTrain'].iloc[i]:.3f} "
              f"| Recall {rec['Pretrain'].iloc[i]:.3f}→{rec['PostTrain'].iloc[i]:.3f} "
              f"| Hits {hits['Pretrain'].iloc[i]:.0f}→{hits['PostTrain'].iloc[i]:.0f} "
              f"| top150 重合 {inter['Pretrain'].iloc[i]:.0f}/150 "
              f"| Spearman {corr['Delta'].iloc[i]:.3f}")
    print("\n=== 三种子均值变化 ===")
    for _, r in mean_delta.iterrows():
        print(f"  {r['Metric']:24s} Δ = {r['MeanDelta3Seed']}")
    print("\n判读：NDCG/Recall 上升 & top150 重合下降 & rank corr<0.9 → 修复有效；"
          "前后几乎不动 → 仍需暂停 MORL。")


if __name__ == "__main__":
    main()
