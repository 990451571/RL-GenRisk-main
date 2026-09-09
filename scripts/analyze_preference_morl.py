#!/usr/bin/env python3
"""Compare a shared preference-conditioned MORL run with scalarized RD runs.

For each seed and seen preference, reports whether any retained MORL Pareto
checkpoint weakly covers (or dominates) the matching scalarized rollout point.
No test labels are read.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

SCALAR_RE = re.compile(r".*_r(?P<rec>\d+)_d(?P<disc>\d+)_seed(?P<seed>\d+)$")


def dominates(a, b):
    keys = ["NDCG@150", "Recall@150", "DiscoveryPrecision@150"]
    return all(float(a[k]) >= float(b[k]) for k in keys) and any(float(a[k]) > float(b[k]) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--morl-runs", nargs="+", required=True)
    ap.add_argument("--scalar-summary", default="outputs/rdprobe_rollout_primary_eval/summary_metrics.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scalar = pd.read_csv(args.scalar_summary)
    scalar = scalar[scalar["Method"] == "rollout_primary_available"].copy()
    scalar_rows = {}
    for _, row in scalar.iterrows():
        m = SCALAR_RE.match(str(row["Run"]))
        if m:
            scalar_rows[(int(m.group("seed")), f"r{int(m.group('rec')) / 100:.2f}_d{int(m.group('disc')) / 100:.2f}")] = row

    rows = []
    run_paths = []
    for pattern in args.morl_runs:
        run_paths.extend(Path(path) for path in (glob.glob(pattern) if "*" in pattern else [pattern]))
    for run in sorted(set(run_paths)):
        cfg = json.loads((run / "morl_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((run / "pareto_manifest.json").read_text(encoding="utf-8"))
        seed = int(cfg["base_args"]["seed"])
        metrics = pd.read_csv(run / "morl_rollout_metrics.csv")
        retained = set(manifest["retained_union_episodes"])
        for pref in cfg["seen_preferences"]:
            key = f"r{pref[0]:.2f}_d{pref[1]:.2f}"
            target = scalar_rows.get((seed, key))
            sub = metrics[(metrics["scope"] == "seen") & (metrics["preference"] == key) & metrics["episode"].isin(retained)]
            if target is None or sub.empty:
                continue
            coverage = any(dominates(row, target) or all(float(row[k]) >= float(target[k]) for k in ("NDCG@150", "Recall@150", "DiscoveryPrecision@150")) for _, row in sub.iterrows())
            rows.append({
                "seed": seed, "preference": key, "retained_morl_points": len(sub),
                "morl_covers_scalar": bool(coverage),
                "scalar_ndcg": float(target["NDCG@150"]),
                "scalar_recall": float(target["Recall@150"]),
                "scalar_precision": float(target["DiscoveryPrecision@150"]),
                "morl_max_ndcg": float(sub["NDCG@150"].max()),
                "morl_max_recall": float(sub["Recall@150"].max()),
                "morl_max_precision": float(sub["DiscoveryPrecision@150"].max()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "morl_vs_scalar_frontier_coverage.csv", index=False)
    print(frame.to_string(index=False) if len(frame) else "No comparable MORL rows found.")


if __name__ == "__main__":
    main()
