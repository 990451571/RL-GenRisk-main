#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery–Discovery 权重探针分析：mean±SD + 方向性判定（2026-09-02）。

输入：evaluate_greedy_rollout.py 的 summary_metrics.csv（含两条 Discovery 指标列）。
输出（控制台 + CSV）：
  A. 每组权重(w_rec,w_disc)×3 seed 的四指标 mean±SD；
  B. 方向性：w_disc 从 0→1，Recovery(NDCG@150/Recall@150) 是否一致下降、
     Discovery(LowFreqNovel@150/EvidenceSupportedLowFreqNovel@150) 是否一致上升。
     逐 seed 看 Spearman 符号一致率，避免只靠 15 个点整体相关。

判读（用户裁定）：
  - 若权重升高能稳定、同向改变这些指标 → 进 preference-conditioned MORL；
  - 否则暂停 MORL，重新定义 Recovery/Discovery 目标。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RECOVERY_METRICS = ["NDCG@150", "Recall@150", "HitCount@150"]
DISCOVERY_METRICS = ["LowFreqNovel@150", "EvidenceSupportedLowFreqNovel@150"]
ALL_METRICS = RECOVERY_METRICS + DISCOVERY_METRICS

RUN_RE = re.compile(r".*_r(?P<rec>\d+(?:\.\d+)?)_d(?P<disc>\d+(?:\.\d+)?)_seed(?P<seed>\d+)$")


def spearman(xs, ys):
    x = pd.Series(np.asarray(xs, dtype=float))
    y = pd.Series(np.asarray(ys, dtype=float))
    return float(x.corr(y, method="spearman")) if len(x) >= 3 else float("nan")


def fmt_sd(mean, sd, nd=4):
    return f"{mean:.{nd}f}±{sd:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, help="summary_metrics.csv 所在目录")
    ap.add_argument("--output-dir", default=None, help="输出目录（默认同 eval-dir）")
    ap.add_argument("--methods", nargs="+",
                    default=["pretrain_Q_onepass", "best_Q_onepass", "best_Q_rollout",
                             "last_Q_onepass"],
                    help="要汇总的 Method 子集（须是 summary_metrics.csv 中出现的方法名）")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir) if args.output_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "summary_metrics.csv"
    if not summary_path.exists():
        sys.exit(f"找不到 {summary_path}")

    df = pd.read_csv(summary_path)
    missing = [c for c in ALL_METRICS if c not in df.columns]
    if missing:
        sys.exit(f"summary_metrics.csv 缺 Discovery 指标列 {missing}；请用升级后的 "
                 f"evaluate_greedy_rollout.py 重新评估。")

    parsed = []
    for _, row in df.iterrows():
        m = RUN_RE.match(str(row["Run"]))
        if not m:
            continue
        parsed.append({
            **row.to_dict(),
            "w_rec": float(m.group("rec")) / 100.0 if "." not in m.group("rec") else float(m.group("rec")),
            "w_disc": float(m.group("disc")) / 100.0 if "." not in m.group("disc") else float(m.group("disc")),
            "seed": int(m.group("seed")),
        })
    if not parsed:
        sys.exit(f"Run 名与期望格式不匹配（期望 rdprobe_r{disc}_d{disc}_seed{seed}）。"
                 f"前几条 Run 值：{df['Run'].unique()[:5]}")
    long = pd.DataFrame(parsed)
    avail_methods = sorted(long["Method"].unique())
    methods = [m for m in args.methods if m in set(avail_methods)]
    if not methods:
        sys.exit(f"方法 {args.methods} 都不在 {avail_methods} 中。")

    group_rows, dir_rows = [], []
    print("=" * 78)
    for method in methods:
        sub = long[long["Method"] == method].copy()
        print(f"\n### 方法 {method}（w_disc 0→1 扫描，每组 3 seed）")
        # 按组排序：w_disc 升序
        groups = sorted(sub["w_disc"].unique())
        table = {m: [] for m in ALL_METRICS}
        header = "  权重(rec,disc)".ljust(16) + "  ".join(f"{m:>28}" for m in ALL_METRICS)
        print(header)
        for wd in groups:
            g = sub[sub["w_disc"] == wd]
            wr = float(g["w_rec"].iloc[0])
            cells = []
            for m in ALL_METRICS:
                vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    cells.append(f"{'n/a':>28}")
                    table[m].append(np.nan)
                    group_rows.append({"Method": method, "w_rec": wr, "w_disc": wd,
                                       "metric": m, "mean": np.nan, "sd": np.nan, "n": 0})
                    continue
                mean, sd = float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
                cells.append(f"{fmt_sd(mean, sd):>28}")
                table[m].append(mean)
                group_rows.append({"Method": method, "w_rec": wr, "w_disc": wd,
                                   "metric": m, "mean": mean, "sd": sd, "n": int(len(vals))})
            print(f"  ({wr:.1f},{wd:.1f})".ljust(16) + "  ".join(cells))

        # ---- 方向性 ----
        print("\n  方向性判定（Spearman：w_disc 与指标的相关，+表示随 discovery 权重上升而上升）")
        agg_corrs = {}
        for m in ALL_METRICS:
            pts = [(float(r["w_disc"]), float(r[m])) for _, r in sub.iterrows() if pd.notna(r[m])]
            rho_all = spearman([p[0] for p in pts], [p[1] for p in pts])
            # 逐 seed：5 个权重组内的相关
            per_seed = {}
            for seed in sorted(sub["seed"].unique()):
                sg = sub[sub["seed"] == seed]
                pts_s = [(float(r["w_disc"]), float(r[m])) for _, r in sg.iterrows() if pd.notna(r[m])]
                per_seed[seed] = spearman([p[0] for p in pts_s], [p[1] for p in pts_s])
            agg_corrs[m] = (rho_all, per_seed)
            # 端点组均值
            g0 = sub[sub["w_disc"] == min(groups)][m]
            g1 = sub[sub["w_disc"] == max(groups)][m]
            m0 = float(g0.mean()) if len(g0) else float("nan")
            m1 = float(g1.mean()) if len(g1) else float("nan")
            seed_signs = [s for s in per_seed.values() if not np.isnan(s)]
            n_agree = sum(1 for s in seed_signs if (s > 0) == (rho_all > 0))
            label = "↑Discovery" if m in DISCOVERY_METRICS else "↓随Discovery? (Recovery)"
            print(f"    {m:<32} rho={rho_all:+.3f} 逐seed={ {k: round(v, 2) for k, v in per_seed.items()} }"
                  f"  端点({min(groups):.1f}→{max(groups):.1f})均值 {m0:.3f}→{m1:.3f}"
                  f"  符号一致 {n_agree}/{len(seed_signs)}")
            dir_rows.append({
                "Method": method, "metric": m, "kind": "recovery" if m in RECOVERY_METRICS else "discovery",
                "rho_all": rho_all, "seed_rhos": per_seed, "seeds_agree_sign": n_agree,
                "n_seeds": len(seed_signs), "endpoint_min_wdisc": min(groups),
                "endpoint_max_wdisc": max(groups),
                "mean_at_min_wdisc": m0, "mean_at_max_wdisc": m1,
                "mean_delta": m1 - m0,
            })

    pd.DataFrame(group_rows).to_csv(out_dir / "rd_probe_group_summary.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(dir_rows).to_csv(out_dir / "rd_probe_direction.csv",
                                  index=False, encoding="utf-8-sig")

    # ================= 汇总判定 =================
    print("\n" + "=" * 78)
    print("汇总判定（以 best_Q_onepass / best_Q_rollout 为准，用户裁定口径）")
    for method in ["best_Q_onepass", "best_Q_rollout"]:
        dr = pd.DataFrame([r for r in dir_rows if r["Method"] == method])
        if dr.empty:
            continue
        rec = dr[dr["kind"] == "recovery"]
        dis = dr[dr["kind"] == "discovery"]
        rec_down = rec.apply(lambda r: r["rho_all"] < -0.3 and r["seeds_agree_sign"] >= max(1, r["n_seeds"] - 1), axis=1)
        dis_up = dis.apply(lambda r: r["rho_all"] > 0.3 and r["seeds_agree_sign"] >= max(1, r["n_seeds"] - 1), axis=1)
        ok = bool(rec_down.all() if len(rec) else False) and bool(dis_up.all() if len(dis) else False)
        print(f"\n  {method}:")
        print(f"    Recovery 指标随 w_disc 上升而一致下降：{'✓' if rec_down.all() else '✗'}")
        print(f"    Discovery 指标随 w_disc 上升而一致上升：{'✓' if dis_up.all() else '✗'}")
        if ok:
            print("    ⇒ 权重变化稳定、同向改变指标 → 可进入 preference-conditioned MORL。")
        else:
            print("    ⇒ 权重变化不能稳定/同向改变指标 → 暂停 MORL，重新定义 Recovery/Discovery。")
    print(f"\n详细表已写入 {out_dir}/rd_probe_group_summary.csv 与 {out_dir}/rd_probe_direction.csv")


if __name__ == "__main__":
    main()
