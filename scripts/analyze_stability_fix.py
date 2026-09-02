#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复第二阶段：验证「前 1-3 轮最好、后续持续下降」是否稳定现象（2026-09-02）。

输入：
  --metrics-runs  训练 run 目录(每个含 train_metrics.csv)，可给 glob
  --eval-dir      evaluate_greedy_rollout.py 的输出目录(含 summary_metrics.csv 与 rankings/)

输出（控制台 + CSV）：
  A. 训练曲线稳定性（per-run 读 train_metrics.csv）：
     - best_episode / best NDCG / Recall@best
     - 第 1-3 轮见顶是否成立；第 50 轮 vs 峰值（跌多少）；reward 是否持续走高
  B. 排名三时点对比（pretrain → best → last，来自 eval 的一次性打分）：
     NDCG / Recall / HitCount + Top-150 重合 + Spearman 秩相关

判读：
  - "前 1-3 轮见顶"在多 seed 下稳定复现 → 训练目标与评价目标部分冲突仍在（见 README 第 5 条），
    说明继续无脑训练会自己把好排序毁掉；需要用 best-checkpoint 或调整稳定参数。
  - reward 全程走高而 NDCG 见顶后回落 → reward 驱动的方向与评价不完全一致。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def spearman_rank_corr(genes_a: list[str], genes_b: list[str], topk: int) -> float:
    rank_a = {g: r for r, g in enumerate(genes_a, 1)}
    rank_b = {g: r for r, g in enumerate(genes_b, 1)}
    union = list({g for g in genes_a[:topk]} | {g for g in genes_b[:topk]})
    if len(union) < 2:
        return 0.0
    ra = np.array([rank_a[g] for g in union], dtype=float)
    rb = np.array([rank_b[g] for g in union], dtype=float)
    return float(pd.Series(ra).corr(pd.Series(rb), method="spearman"))


def load_rankings(path: Path) -> list[str]:
    genes = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            genes.append(row["Gene"])
    return genes


def expand_braces(pattern: str):
    """把 {a,b,c} 单层展开为多个 glob 串（glob 本身不支持大括号）。"""
    import re as _re
    m = _re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    choices = m.group(1).split(",")
    head, tail = pattern[: m.start()], pattern[m.end():]
    return [expand_braces(head + c + tail) for c in choices]  # noqa: 递归到无括号


def flatten(xs):
    out = []
    for x in xs:
        out.extend(x if isinstance(x, list) else [x])
    return out


def ep_num(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def curve_stats(df: pd.DataFrame):
    """per-run 训练曲线统计。df 需含 episode/episode_reward/val_ndcg_150/val_recall_k。"""
    df = df.dropna(subset=["val_ndcg_150"]).copy()
    df["ep"] = df["episode"].map(ep_num)
    df["nd"] = pd.to_numeric(df["val_ndcg_150"], errors="coerce")
    df["rw"] = pd.to_numeric(df["episode_reward"], errors="coerce")
    df["rc"] = pd.to_numeric(df["val_recall_k"], errors="coerce")
    df = df.sort_values("ep").reset_index(drop=True)
    if df.empty or df["nd"].isna().all():
        return None
    peak_idx = int(df["nd"].idxmax())
    peak_ep = int(df.loc[peak_idx, "ep"])
    peak_nd = float(df["nd"].max())
    peak_rc = float(df.loc[peak_idx, "rc"])
    peak_rw = float(df.loc[peak_idx, "rw"])
    n_ep = len(df)
    last_ep = int(df["ep"].iloc[-1])
    last_nd = float(df["nd"].iloc[-1])
    last_rc = float(df["rc"].iloc[-1])
    last_rw = float(df["rw"].iloc[-1])
    first3 = df[df["ep"] <= 3]
    first3_max = float(first3["nd"].max()) if not first3.empty else np.nan
    first3_max_ep = int(first3.loc[first3["nd"].idxmax(), "ep"]) if not first3.empty else None
    reward_first3 = float(first3["rw"].mean()) if not first3.empty else np.nan
    reward_last10 = float(df["rw"].tail(10).mean())
    rw_rising = reward_last10 > reward_first3 * 1.2
    # 峰值后 5 轮均值（若存在）
    post = df[df["ep"] > peak_ep]
    post5 = float(post["nd"].head(5).mean()) if not post.empty else np.nan
    decline_frac = (peak_nd - last_nd) / peak_nd if peak_nd > 0 else np.nan
    return {
        "episodes": n_ep, "last_ep": last_ep,
        "peak_episode": peak_ep, "best_ndcg": peak_nd, "recall_at_best": peak_rc,
        "reward_at_best": peak_rw, "ndcg_ep3_max": first3_max, "peak_in_first3": peak_ep <= 3,
        "ndcg_last": last_nd, "recall_last": last_rc, "reward_last": last_rw,
        "post_peak_5ep_mean": post5, "decline_frac": decline_frac,
        "reward_rising_to_end": rw_rising,
        "reward_first3_mean": reward_first3, "reward_last10_mean": reward_last10,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-runs", nargs="+", default=None,
                    help="训练 run 目录或 glob（train_metrics.csv 所在），如 'outputs/fix_legacy_seed*/hybrid6_raw/*'")
    ap.add_argument("--eval-dir", default=None, help="evaluate_greedy_rollout.py 输出目录")
    ap.add_argument("--topk", type=int, default=150)
    ap.add_argument("--output-dir", default="outputs/fix5_rollout_eval")
    args = ap.parse_args()

    run_dirs = []
    if args.metrics_runs:
        for g in args.metrics_runs:
            for pat in flatten(expand_braces(g)):
                run_dirs += glob.glob(pat)
        run_dirs = sorted({Path(d) for d in run_dirs if (Path(d) / "train_metrics.csv").exists()})
    if not run_dirs:
        sys.exit("找不到含 train_metrics.csv 的 run 目录")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- A. 训练曲线稳定性 ----------
    stab_rows = []
    for rd in run_dirs:
        name = rd.parts[-3]
        df = pd.read_csv(rd / "train_metrics.csv")
        cs = curve_stats(df)
        if cs is None:
            print(f"  [跳过] {name}: 无有效验证曲线")
            continue
        cs["Run"] = name
        stab_rows.append(cs)
        pk = cs["peak_in_first3"]
        print(f"  {name}: 峰值在第 {cs['peak_episode']:>2} 轮(NDCG {cs['best_ndcg']:.4f}, "
              f"Recall {cs['recall_at_best']:.2f}) | 末轮 NDCG {cs['ndcg_last']:.4f} "
              f"(跌 {cs['decline_frac'] * 100:.0f}%) | 峰值后5轮均值 {cs['post_peak_5ep_mean']:.4f} | "
              f"reward{'持续走高' if cs['reward_rising_to_end'] else '未走高'} "
              f"({'前3轮均值 {:.2f} → 末10轮均值 {:.2f}'.format(cs['reward_first3_mean'], cs['reward_last10_mean'])})"
              f"{'  ← 前1-3轮见顶' if pk else ''}")
    stab = pd.DataFrame(stab_rows)
    n = len(stab)
    n_early = int(stab["peak_in_first3"].sum())
    med_peak = int(np.median(stab["peak_episode"])) if n else None
    avg_best = stab["best_ndcg"].mean() if n else float("nan")
    avg_last = stab["ndcg_last"].mean() if n else float("nan")
    avg_decline = stab["decline_frac"].mean() if n else float("nan")
    n_reward_up = int(stab["reward_rising_to_end"].sum())
    stab.to_csv(out_dir / "stability_curves.csv", index=False, encoding="utf-8-sig")

    # ---------- B. pretrain → best → last 排名对比（eval 产物） ----------
    pre_best_last = []
    if args.eval_dir and Path(args.eval_dir).exists():
        ev = Path(args.eval_dir)
        summ = pd.read_csv(ev / "summary_metrics.csv")
        rk = ev / "rankings"
        for _, row in stab.iterrows():
            name = row["Run"]
            sub = summ[(summ["Run"] == name) & (summ["Method"].isin(
                ["pretrain_Q_onepass", "best_Q_onepass", "last_Q_onepass"]))]
            if sub.empty:
                print(f"  [提示] {name}: eval 缺该 run 的三时点指标")
                continue
            sub = sub.set_index("Method")
            try:
                pre, best, last = (sub.loc[m] for m in
                                   ["pretrain_Q_onepass", "best_Q_onepass", "last_Q_onepass"])
            except KeyError as exc:
                print(f"  [提示] {name}: 缺 {exc} 方法行")
                continue
            g_pre = load_rankings(rk / f"{name}_pretrain_Q_onepass.csv")
            g_best = load_rankings(rk / f"{name}_best_Q_onepass.csv")
            g_last = load_rankings(rk / f"{name}_last_Q_onepass.csv")
            s_pre_best = spearman_rank_corr(g_pre, g_best, args.topk)
            s_best_last = spearman_rank_corr(g_best, g_last, args.topk)
            def ov(a, b):
                ia = set(a[: args.topk]); ib = set(b[: args.topk])
                return len(ia & ib)
            pre_best_last.append({
                "Run": name,
                "NDCG_pre": pre["NDCG@150"], "NDCG_best": best["NDCG@150"], "NDCG_last": last["NDCG@150"],
                "Recall_pre": pre["Recall@150"], "Recall_best": best["Recall@150"], "Recall_last": last["Recall@150"],
                "Hits_pre": pre["HitCount@150"], "Hits_best": best["HitCount@150"], "Hits_last": last["HitCount@150"],
                "Top150_overlap_pre_best": ov(g_pre, g_best), "Top150_overlap_best_last": ov(g_best, g_last),
                "Spearman_pre_best": round(s_pre_best, 3), "Spearman_best_last": round(s_best_last, 3),
            })
    if pre_best_last:
        tbl = pd.DataFrame(pre_best_last)
        tbl.to_csv(out_dir / "pre_best_last_compare.csv", index=False, encoding="utf-8-sig")
        print("\n=== 三时点对比（一次性打分）===")
        print(tbl.to_string(index=False))
        print("\n平均：NDCG pre {:.3f} → best {:.3f} → last {:.3f}".format(
            tbl["NDCG_pre"].mean(), tbl["NDCG_best"].mean(), tbl["NDCG_last"].mean()))
        print("命中：pre {:.1f} → best {:.1f} → last {:.1f} | "
              "top150 重合 best∩last 平均 {:.0f}/150 | "
              "Spearman best↔last 平均 {:.2f}".format(
            tbl["Hits_pre"].mean(), tbl["Hits_best"].mean(), tbl["Hits_last"].mean(),
            tbl["Top150_overlap_best_last"].mean(), tbl["Spearman_best_last"].mean()))

    # ---------- 汇总判定 ----------
    print("\n" + "=" * 70)
    print(f"稳定性判定（{n} 个 seed）")
    print(f"  前1-3轮见顶的 seed：{n_early}/{n}  |  峰值轮中位数：{med_peak}")
    print(f"  best NDCG 平均 {avg_best:.4f} → 末轮 NDCG 平均 {avg_last:.4f}（峰值后平均跌 {avg_decline * 100:.0f}%）")
    print(f"  reward 末段仍走高的 seed：{n_reward_up}/{n}")
    if n_early == n:
        print("  ⇒ 「前 1-3 轮最好、后续下降」在本批 seed 中稳定复现：训练目标与评价目标部分冲突仍在。")
        print("    下一步：best-checkpoint 继续使用；若要缓解，只能在 lr/epsilon 衰减/更新频率里找（不加奖励）。")
    elif n_early >= max(1, n // 2):
        print("  ⇒ 「前 1-3 轮见顶」是主要但不完全稳定的模式，多数 seed 复现。")
    else:
        print("  ⇒ 「前 1-3 轮见顶」在本批 seed 中不是稳定现象（best 轮分布更散）。")
    if pre_best_last:
        avg_gain = float(tbl["NDCG_best"].mean()) - float(tbl["NDCG_pre"].mean())
        print(f"\n修复可重复性：NDCG 训练前→best 平均 {avg_gain:+.3f}" +
              ("（明显为正 → 修复效果在更多 seed 下可重复，可进 Recovery–Discovery 权重扫描）"
               if avg_gain > 0.05 else
               "（仍不显著 → 继续定位训练稳定性/泛化问题，先不进扫描）"))
    stab.to_csv(out_dir / "stability_curves.csv", index=False, encoding="utf-8-sig")
    print(f"\n详细表已写入 {out_dir}/stability_curves.csv 与 pre_best_last_compare.csv")


if __name__ == "__main__":
    main()
