#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""贪心 rollout 评估 + 无 RL 基线，作用于已训练好的 checkpoint（无需重新训练）。

对比四类排名：
  1. 纯静态基线：Degree / Mutation / 低频证据分 V2（完全不用网络）
  2. 训练前 Q（同一 seed 重建随机初始化网络，空上下文一次性打分）
  3. 训练后 Q · 一次性打分（当前评估方式，空上下文）
  4. 训练后 Q · 贪心 rollout（序列决策：逐次选当前 Q 最大且未选的基因，更新上下文，选满 topk）

用法：
  python scripts/evaluate_greedy_rollout.py \
      --run-dirs 'outputs/exp*_seed*/hybrid6_raw/*' \
      --output outputs/rollout_eval

说明：
  - 通过 run_dir/config.json 重建 args 与环境，与训练完全同源。
  - set_seed(seed) → build_environment → build_agent 的顺序与训练 main() 一致，
    因此“训练前 Q”精确复现训练开始时（未学任何东西）的随机初始化网络排名。
  - checkpoint_best.pt 与 checkpoint_last.pt 均评估。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_run_seeds(run_dirs):
    seeds = []
    for rd in run_dirs:
        cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
        seeds.append(int(cfg["args"]["seed"]))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="run 目录（含 checkpoint_best.pt 与 config.json）")
    parser.add_argument("--output", default=str(REPO_DIR / "outputs" / "rollout_eval"),
                        help="输出目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import glob as globlib
    run_dirs = []
    for pattern in args.run_dirs:
        matches = sorted(globlib.glob(pattern)) if "*" in pattern else [pattern]
        for m in matches:
            mp = Path(m)
            if (mp / "checkpoint_best.pt").exists() and (mp / "config.json").exists():
                run_dirs.append(mp)
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        sys.exit("未找到任何含 checkpoint_best.pt 的 run 目录。")
    print(f"发现 {len(run_dirs)} 个 run：")
    for rd in run_dirs:
        print(f"  - {rd}")

    # 在导入 train 前设置进程级 hash seed，与训练保持同源（train 模块导入时会读 sys.argv）。
    seeds = read_run_seeds(run_dirs)
    os.environ.setdefault("PYTHONHASHSEED", str(seeds[0]))
    sys.argv = [sys.argv[0], "--seed", str(seeds[0]), "--device", args.device]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(REPO_DIR / "src"))
    import torch
    import train  # noqa: E402
    import rd_definitions  # noqa: E402  (Recovery–Discovery 探针共享定义)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n评估设备：{device}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rankings_dir = output_root / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    # 静态基线：V2 证据分（直接读表，不经过网络）
    import pandas as pd
    ev_df = pd.read_csv(V2_EVIDENCE_PATH)
    ev_score = dict(zip(ev_df["Gene"], ev_df["LowFrequencyEvidenceScoreV2"]))

    summary_rows = []
    overlap_rows = []
    found_rows = []

    for rd in run_dirs:
        name = rd.parts[-3]
        print(f"\n{'=' * 70}\n== {name}  {rd}\n{'=' * 70}")

        cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
        cargs = SimpleNamespace(**cfg["args"])

        with tempfile.TemporaryDirectory(prefix="rollout_eval_") as tmp:
            tmp_dir = Path(tmp)
            train.set_seed(cargs.seed)
            env = train.build_environment(cargs, tmp_dir, normalization_metadata=None)

            # 校验重建的特征与训练时记录一致（防止 hash 种子等原因导致特征漂移）
            stored_report = cfg["feature_report"]
            new_report = env["feature_report"]
            checks = {
                "feature_mode": stored_report["feature_mode"] == new_report["feature_mode"],
                "feature_dim": stored_report["feature_dim"] == new_report["feature_dim"],
                "shape": stored_report["node_features_shape"] == new_report["node_features_shape"],
                "multiomics_match_count": stored_report["multiomics_report"].get("matched_genes")
                == new_report["multiomics_report"].get("matched_genes"),
            }
            if not all(checks.values()):
                print(f"  [警告] 重建特征与训练记录不一致：{checks}")
            else:
                print(f"  [校验] 重建特征与训练记录一致 ✓")

            agent = train.build_agent(cargs, env, device)
            agent.Q.eval()

            val_labels = set(env["validation_driver_genes"])
            gene_name = list(env["gene_name"])
            n_actions = agent.n_actions
            feature_cols = env["feature_report"]["feature_columns"]
            features = env["node_features"]

            # Discovery 候选集（低频新候选 / 证据支持的候选），known= train∪val driver
            known_drivers = set(env["train_driver_genes"]) | val_labels
            disc_sets = rd_definitions.discovery_sets_from_evidence(ev_df, known_drivers)
            print(
                f"  Discovery 候选集：低频新候选 {disc_sets['n_lowfreq_novel']} 个，"
                f"证据支持(证据分≥{rd_definitions.RD_EVIDENCE_MIN_DEFAULT}) "
                f"{disc_sets['n_evidence_supported']} 个"
            )

            topk = int(cargs.topk)
            print(f"  验证集 driver 数：{len(val_labels)}，topk={topk}")

            # ---------- 1. 静态基线 ----------
            static_orders = {
                "static_degree": sorted(range(n_actions), key=lambda i: (-features[i, 0], gene_name[i])),
                "static_mutation": sorted(range(n_actions), key=lambda i: (-features[i, 3], gene_name[i])),
                "static_evidenceV2": sorted(
                    range(n_actions),
                    key=lambda i: (-ev_score.get(gene_name[i], -1.0), gene_name[i]),
                ),
            }

            # ---------- 2/3/4. 网络排名 ----------
            def one_pass_order():
                state_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
                mask_tensor = torch.as_tensor(np.ones(n_actions, dtype=np.int64), dtype=torch.long, device=device)
                with torch.no_grad():
                    q_values, _ = agent.Q(None, state_tensor, mask_tensor)
                q_np = q_values.detach().cpu().numpy()
                return sorted(range(n_actions), key=lambda i: (-q_np[i], gene_name[i])), q_np

            def greedy_rollout_order():
                # 返回完整 9039 长度排名：已选 150 个按选择顺序在前，未选基因按最后一轮 Q 降序排后。
                # score 数组长度 n_actions：已选基因记选择时 Q，未选基因记最后一轮 Q。
                state_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
                mask = np.ones(n_actions, dtype=np.int64)
                emb = None
                selected = []
                q_at_select = np.full(n_actions, -np.inf)
                final_q = None
                with torch.no_grad():
                    for _ in range(topk):
                        mask_tensor = torch.as_tensor(mask, dtype=torch.long, device=device)
                        q_values, emb = agent.Q(emb, state_tensor, mask_tensor)
                        q_np = q_values.detach().cpu().numpy()
                        valid = [i for i in range(n_actions) if mask[i] == 1]
                        best = max(valid, key=lambda i: q_np[i])
                        selected.append(best)
                        q_at_select[best] = q_np[best]
                        mask[best] = 0
                        final_q = q_np
                unselected = [i for i in range(n_actions) if mask[i] == 1]
                tail = sorted(unselected, key=lambda i: (-final_q[i], gene_name[i]))
                order = selected + tail
                score = np.where(q_at_select > -np.inf, q_at_select, final_q)
                return order, score

            def metrics_for(order, tag):
                rows = [{"Gene": gene_name[i]} for i in order]
                rank_by_gene = {gene_name[i]: r for r, i in enumerate(order, 1)}
                present = [rank_by_gene[g] for g in val_labels if g in rank_by_gene]
                k = 150
                item = train.metrics_at_k(rows, val_labels, k)
                topk_genes = [gene_name[i] for i in order[:k]]
                lf_novel_count = sum(1 for g in topk_genes if g in disc_sets["lowfreq_novel"])
                ev_supported_count = sum(1 for g in topk_genes if g in disc_sets["evidence_supported"])
                return {
                    "Method": tag,
                    "Run": name,
                    "NDCG@150": item["NDCG"],
                    "Recall@150": item["Recall"],
                    "Precision@150": item["Precision"],
                    "HitCount@150": item["HitCount"],
                    "LowFreqNovel@150": lf_novel_count,
                    "EvidenceSupportedLowFreqNovel@150": ev_supported_count,
                    "MRR": float(np.mean([1.0 / r for r in present])) if present else 0.0,
                    "MeanRank": float(np.mean(present)) if present else None,
                    "MedianRank": float(np.median(present)) if present else None,
                }

            def write_csv(order, scores, tag):
                # tag 已含 run 名；scores: None（静态基线，无分数）或长度 n_actions 的数组
                path = rankings_dir / f"{tag}.csv"
                with path.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["Rank", "Gene", "Score"])
                    for r, i in enumerate(order, 1):
                        val = ""
                        if scores is not None:
                            val = float(scores[i])
                        w.writerow([r, gene_name[i], val])
                return path

            # 训练前 Q（未加载 checkpoint）
            pretrain_order, pretrain_q = one_pass_order()
            pm = metrics_for(pretrain_order, "pretrain_Q_onepass")
            summary_rows.append(pm)
            write_csv(pretrain_order, pretrain_q, f"{name}_pretrain_Q_onepass")
            print("  训练前Q(一次性)     ", _fmt(pm))

            # 训练后 best / last
            best_orders = {}
            for ckpt_tag, ckpt_name in [("best", "checkpoint_best.pt"), ("last", "checkpoint_last.pt")]:
                train.load_checkpoint(agent, rd / ckpt_name, cargs, env)
                agent.Q.eval()
                one_order, one_q = one_pass_order()
                roll_order, roll_q = greedy_rollout_order()
                best_orders[ckpt_tag] = (one_order, roll_order)
                om = metrics_for(one_order, f"{ckpt_tag}_Q_onepass")
                rm = metrics_for(roll_order, f"{ckpt_tag}_Q_rollout")
                summary_rows.extend([om, rm])
                write_csv(one_order, one_q, f"{name}_{ckpt_tag}_Q_onepass")
                write_csv(roll_order, roll_q, f"{name}_{ckpt_tag}_Q_rollout")
                print(f"  {ckpt_tag}·一次性  ", _fmt(om))
                print(f"  {ckpt_tag}·rollout ", _fmt(rm))

                # 一次性 vs rollout 的 top-150 差异
                one_top150 = {gene_name[i] for i in one_order[:150]}
                roll_top150 = {gene_name[i] for i in roll_order[:150]}
                inter = len(one_top150 & roll_top150)
                union = len(one_top150 | roll_top150)
                print(f"    → top-150 重合率：{inter}/{union} (Jaccard={inter / union:.3f})")
                overlap_rows.append({
                    "Run": name, "Checkpoint": ckpt_tag,
                    "onepass_top150": len(one_top150), "rollout_top150": len(roll_top150),
                    "intersection": inter, "jaccard": inter / union if union else 0.0,
                })

            # 静态基线指标
            for tag, order in static_orders.items():
                sm = metrics_for(order, tag)
                summary_rows.append(sm)
                write_csv(order, None, f"{name}_{tag}")
                print(f"  {tag:22s}", _fmt(sm))

            # 找到的验证 driver（best·rollout / best·一次性）
            best_one, best_roll = best_orders["best"]
            for method_key, tag, order in [
                ("best_Q_onepass", "一次性", best_one),
                ("best_Q_rollout", "rollout", best_roll),
            ]:
                found = sorted(g for g in val_labels if g in {gene_name[i] for i in order[:150]})
                found_rows.append({"Run": name, "Method": method_key, "FoundDrivers": "|".join(found),
                                   "Count": len(found)})
                print(f"    {tag} 找到的 driver：{found}")

    # ---------- 汇总输出 ----------
    out_csv = output_root / "summary_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n指标汇总已写入：{out_csv}")

    ov_csv = output_root / "summary_onepass_vs_rollout_top150.csv"
    with ov_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overlap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overlap_rows)
    print(f"一次性 vs rollout 重合率已写入：{ov_csv}")

    fd_csv = output_root / "summary_found_drivers.csv"
    with fd_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(found_rows[0].keys()))
        writer.writeheader()
        writer.writerows(found_rows)
    print(f"命中的验证 driver 已写入：{fd_csv}")

    # 跨 run 的 rollout 稳定性：top-150 Jaccard 矩阵
    print("\n=== 跨 run 的 best·rollout top-150 重合（Jaccard）===")
    roll_runs = {}
    for rd in run_dirs:
        name = rd.parts[-3]
        tag = f"{name}_best_Q_rollout"
        path = rankings_dir / f"{tag}.csv"
        genes = []
        with path.open(encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                if int(row["Rank"]) <= 150:
                    genes.append(row["Gene"])
        roll_runs[name] = set(genes)
    names = list(roll_runs.keys())
    header = "          " + " ".join(f"{n[-9:]:>10}" for n in names)
    print(header)
    for a in names:
        row_str = f"{a[-9:]:>10}"
        for b in names:
            s = len(roll_runs[a] & roll_runs[b])
            u = len(roll_runs[a] | roll_runs[b])
            row_str += f"{s / u if u else 0.0:>10.3f}"
        print(row_str)


def _fmt(m):
    mean_rank = f"{m['MeanRank']:>7.0f}" if m["MeanRank"] is not None else "    n/a"
    return (f"NDCG={m['NDCG@150']:.4f} Recall={m['Recall@150']:.3f} "
            f"Hits={m['HitCount@150']:>2} MRR={m['MRR']:.4f} MeanRank={mean_rank}")


if __name__ == "__main__":
    main()
