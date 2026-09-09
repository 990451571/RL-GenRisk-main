#!/usr/bin/env python3
"""Freeze the three dual-head bandit candidate profiles before external audit.

This script intentionally reads no independent evidence and no Test labels. It
copies the already-produced Top-150 rankings, records hashes of source
rankings/configs/checkpoints, and builds a seed-consensus table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/discovery_validity_audit_20260909/frozen_candidates"
RUNS = {
    42: ROOT / "outputs/dual_head_bandit_20260908_210039_seed42/hybrid6_raw/seed_42_20260908_210041",
    45: ROOT / "outputs/dual_head_bandit_20260908_210039_seed45/hybrid6_raw/seed_45_20260908_212658",
    48: ROOT / "outputs/dual_head_bandit_20260908_210039_seed48/hybrid6_raw/seed_48_20260908_215312",
}
PREFERENCES = {
    "recovery_heavy": "r0.80_d0.20",
    "compromise": "r0.50_d0.50",
    "discovery_heavy": "r0.20_d0.80",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "freeze_version": "discovery_candidates_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_scope": "latest dual-head contextual bandit",
        "seeds": sorted(RUNS),
        "preferences": PREFERENCES,
        "top_k": 150,
        "candidate_selection_after_freeze_allowed": False,
        "external_evidence_read_by_freeze": False,
        "test_labels_read": False,
        "files": [],
    }
    frozen_frames = []

    for seed, run_dir in RUNS.items():
        config = run_dir / "dual_head_bandit_config.json"
        checkpoint = run_dir / "checkpoint_final.pt"
        if not config.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"Incomplete frozen run: {run_dir}")
        config_data = json.loads(config.read_text(encoding="utf-8"))
        if config_data.get("test_labels_read") is not False:
            raise RuntimeError(f"Test-isolation contract failed: {config}")
        if config_data.get("bootstrap") is not False:
            raise RuntimeError(f"Expected contextual-bandit no-bootstrap config: {config}")

        for profile, preference in PREFERENCES.items():
            source = run_dir / "final_rollout_rankings" / f"{preference}.csv"
            ranking = pd.read_csv(source)
            required = {"Rank", "Gene", "Score"}
            if not required.issubset(ranking.columns):
                raise ValueError(f"Missing ranking columns in {source}: {required - set(ranking.columns)}")
            if len(ranking) != 9039 or ranking.Gene.astype(str).nunique() != 9039:
                raise ValueError(f"Expected a unique 9,039-gene ranking: {source}")
            ranking = ranking.sort_values("Rank")
            if ranking.Rank.tolist() != list(range(1, 9040)):
                raise ValueError(f"Non-contiguous rank column: {source}")

            frozen = ranking.head(150).copy()
            frozen.insert(0, "Profile", profile)
            frozen.insert(1, "Preference", preference)
            frozen.insert(2, "Seed", seed)
            frozen = frozen.rename(columns={"Score": "BanditRolloutScore"})
            frozen["BanditRankScore"] = -frozen["Rank"].astype(float)
            target = out / f"{profile}_{preference}_seed{seed}_top150.csv"
            frozen.to_csv(target, index=False)
            frozen_frames.append(frozen)
            manifest["files"].append({
                "profile": profile,
                "preference": preference,
                "seed": seed,
                "source_ranking": str(source),
                "source_ranking_sha256": sha256(source),
                "frozen_top150": str(target),
                "frozen_top150_sha256": sha256(target),
                "config": str(config),
                "config_sha256": sha256(config),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            })

    all_frozen = pd.concat(frozen_frames, ignore_index=True)
    consensus = (
        all_frozen.groupby(["Profile", "Preference", "Gene"], as_index=False)
        .agg(
            SeedSelectionCount=("Seed", "nunique"),
            MeanRank=("Rank", "mean"),
            MedianRank=("Rank", "median"),
            MeanBanditRolloutScore=("BanditRolloutScore", "mean"),
        )
        .sort_values(["Profile", "SeedSelectionCount", "MeanRank"], ascending=[True, False, True])
    )
    consensus.to_csv(out / "frozen_three_profile_consensus.csv", index=False)

    manifest_path = out / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "FROZEN",
        "output_dir": str(out),
        "manifest": str(manifest_path),
        "frozen_rankings": len(manifest["files"]),
        "test_labels_read": False,
        "external_evidence_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
