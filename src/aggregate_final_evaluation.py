import argparse
import csv
from pathlib import Path

import pandas as pd


EVAL_ROOT = Path("/mnt/e/codex_file/二阶段/02_final_evaluation")
TEST_DIR = EVAL_ROOT / "01_test"
HOLDOUT_DIR = EVAL_ROOT / "02_external_holdout"
REPORT_DIR = EVAL_ROOT / "05_reports"
REPRO_DIR = EVAL_ROOT / "06_reproducibility"


def read_optional_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def write_manifest():
    rows = []
    for path in sorted(EVAL_ROOT.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })
    with (REPORT_DIR / "final_run_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-label-path", default="")
    parser.add_argument("--holdout-path", default="")
    args = parser.parse_args()

    test_summary = read_optional_csv(TEST_DIR / "summaries" / "test_reward_mode_summary.csv")
    test_pairs = read_optional_csv(TEST_DIR / "summaries" / "test_paired_comparison_vs_legacy.csv")
    holdout_summary = read_optional_csv(HOLDOUT_DIR / "summaries" / "holdout_reward_mode_summary.csv")
    holdout_gene = read_optional_csv(HOLDOUT_DIR / "per_gene" / "holdout_gene_rank_comparison.csv")

    lines = ["# Final Stage 2 Evaluation Report", ""]
    if test_summary is None:
        lines.append("Test evaluation has not been run because no Test label path was provided.")
    else:
        low = test_summary[test_summary["reward_mode"] == "multiomics_lowfreq"].iloc[0]
        leg = test_summary[test_summary["reward_mode"] == "legacy"].iloc[0]
        lines += [
            "## Test",
            "",
            f"legacy mean NDCG@150 = {leg['NDCG@150_mean']}",
            f"multiomics_lowfreq mean NDCG@150 = {low['NDCG@150_mean']}",
        ]
        if test_pairs is not None:
            pair = test_pairs[test_pairs["reward_mode"] == "multiomics_lowfreq"].iloc[0]
            lines.append(f"lowfreq vs legacy paired wins/ties/losses = {int(pair['wins'])}/{int(pair['ties'])}/{int(pair['losses'])}")
    if holdout_summary is None:
        lines += ["", "## External Holdout", "", "External holdout evaluation has not been run because no user-provided holdout path was supplied."]
    else:
        low = holdout_summary[holdout_summary["reward_mode"] == "multiomics_lowfreq"].iloc[0]
        leg = holdout_summary[holdout_summary["reward_mode"] == "legacy"].iloc[0]
        lines += [
            "",
            "## External Holdout",
            "",
            f"legacy mean HoldoutHit@150 = {leg['HoldoutHit@150_mean']}",
            f"multiomics_lowfreq mean HoldoutHit@150 = {low['HoldoutHit@150_mean']}",
        ]
    lines += [
        "",
        "## Boundaries",
        "",
        "No final evaluation script trains the model, changes reward, writes replay-buffer transitions, updates PER priority, or performs checkpoint selection from Test or external holdout results.",
    ]
    (REPORT_DIR / "final_stage2_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    claims = [
        "# Final Claims Boundary",
        "",
        "Allowed: describe frozen Test and external holdout ranking behavior across all five seeds.",
        "Not allowed: claim new cancer-gene discovery, clinical utility, statistical significance from five seeds alone, complete mutation independence, or direct reward of unknown low-frequency genes.",
    ]
    (REPORT_DIR / "final_claims_boundary.md").write_text("\n".join(claims) + "\n", encoding="utf-8")

    summary = [
        "Final Stage 2 Evaluation Summary",
        f"test_label_path={args.test_label_path}",
        f"holdout_path={args.holdout_path}",
        f"test_completed={test_summary is not None}",
        f"external_holdout_completed={holdout_summary is not None}",
    ]
    (REPORT_DIR / "final_stage2_evaluation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    commands = [
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_genrisk",
        "cd /mnt/e/Projects/RL-GenRisk-main",
    ]
    if args.test_label_path:
        commands.append(f"python src/evaluate_frozen_test.py --test-label-path {args.test_label_path} --device cuda")
    if args.holdout_path:
        commands.append(f"python src/evaluate_external_holdout.py --holdout-path {args.holdout_path}")
    commands.append("python src/aggregate_final_evaluation.py")
    (REPRO_DIR / "reproducibility_commands_final.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_manifest()


if __name__ == "__main__":
    main()
