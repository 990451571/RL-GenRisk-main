#!/usr/bin/env python3
"""Audit and regenerate mutation-frequency group annotations for RL-GenRisk."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r"E:\Projects\RL-GenRisk-main")
OUT = Path(r"E:\codex_file\mutation_frequency_redefinition")
STAGE1 = Path(r"E:\codex_file\新方向阶段1")
STAGE2 = Path(r"E:\codex_file\新方向阶段2")
STAGE4 = Path(r"E:\codex_file\新方向阶段4")
PROTOCOL_B = Path(r"E:\codex_file\一阶段\driver_label_protocol\protocol_B")

sys.path.insert(0, str(PROJECT / "src"))
from mutation_frequency import (  # noqa: E402
    HIGH_FREQUENCY_MUTATION_MIN_COUNT,
    LOW_FREQUENCY_MUTATION_MAX_COUNT,
    LOW_FREQUENCY_MUTATION_MIN_COUNT,
    MUTATION_GROUPS,
    TOTAL_KIRC_TUMOR_SAMPLES,
    classify_mutation_frequency,
    mutation_frequency,
    mutation_frequency_pct,
)

LOG_ROWS: list[dict[str, str]] = []


def log(path: Path | str, purpose: str, action: str = "read", test: bool = False) -> None:
    LOG_ROWS.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
            "purpose": purpose,
            "action": action,
            "test_used": "YES" if test else "NO",
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        log(path, "write empty output table", "write")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(path, "write mutation-frequency redefinition output", "write")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log(path, "write mutation-frequency redefinition report", "write")


def clean_gene(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    gene = str(value).strip().upper()
    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()
    if gene in {"", "?", "NA", "N/A", "NAN", "NONE", "NULL"}:
        return None
    return gene


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    log(path, "read source table")
    return pd.read_csv(path, **kwargs)


def load_gene_universe() -> tuple[list[str], str]:
    frozen = STAGE1 / "10_stage2_ready" / "gene_universe.tsv"
    if frozen.exists():
        df = read_csv(frozen, sep="\t")
        genes = [clean_gene(g) for g in df["gene_symbol"]]
        return [g for g in genes if g], str(frozen)

    hprd = PROJECT / "data" / "HPRD.txt"
    log(hprd, "read fallback HPRD gene universe")
    genes: list[str] = []
    seen: set[str] = set()
    with hprd.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            for raw in parts[:2]:
                gene = clean_gene(raw)
                if gene and gene not in seen:
                    seen.add(gene)
                    genes.append(gene)
    return genes, str(hprd)


def load_mutation_annotation(genes: list[str]) -> tuple[pd.DataFrame, dict]:
    mutation_path = PROJECT / "data" / "processed" / "KIRC_mutation_gene_feature.csv"
    raw = read_csv(mutation_path)
    raw["Gene"] = raw["Gene"].map(clean_gene)
    raw = raw[raw["Gene"].notna()].copy()
    if "TotalMutationSamples" in raw.columns:
        totals = set(pd.to_numeric(raw["TotalMutationSamples"], errors="coerce").dropna().astype(int))
        if totals != {TOTAL_KIRC_TUMOR_SAMPLES}:
            raise ValueError(f"Unexpected TotalMutationSamples values: {sorted(totals)}")
    raw["MutatedSampleCount"] = pd.to_numeric(raw["MutatedSampleCount"], errors="raise").astype(int)
    if (raw["MutatedSampleCount"] < 0).any():
        raise ValueError("Negative MutatedSampleCount found.")
    by_gene = raw.set_index("Gene")

    rows = []
    missing_from_mutation_table = 0
    for gene in genes:
        if gene in by_gene.index:
            count = int(by_gene.loc[gene, "MutatedSampleCount"])
            existing_freq = float(by_gene.loc[gene, "Mutation"])
        else:
            count = 0
            existing_freq = 0.0
            missing_from_mutation_table += 1
        freq = mutation_frequency(count)
        if not math.isclose(existing_freq, freq, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Mutation frequency mismatch for {gene}: {existing_freq} != {freq}")
        rows.append(
            {
                "Gene": gene,
                "MutationPatientCount": count,
                "MutationFrequency": freq,
                "MutationFrequencyPct": mutation_frequency_pct(count),
                "MutationGroup": classify_mutation_frequency(count),
            }
        )
    df = pd.DataFrame(rows)
    info = {
        "gene_universe_count": len(genes),
        "mutation_table_gene_count": int(raw["Gene"].nunique()),
        "missing_universe_genes_treated_as_zero": int(missing_from_mutation_table),
    }
    return df, info


def summarize_groups(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    rows = []
    for group in MUTATION_GROUPS:
        count = int((df["MutationGroup"] == group).sum())
        rows.append(
            {
                "MutationGroup": group,
                "GeneCount": count,
                "PercentageOfGenes": count / total * 100.0 if total else 0.0,
            }
        )
    return rows


def load_driver_genes(path: Path) -> list[str]:
    df = read_csv(path)
    return [g for g in df["Gene"].map(clean_gene).dropna().tolist()]


def driver_group_summary(annotation: pd.DataFrame) -> list[dict]:
    by_gene = annotation.set_index("Gene")
    rows = []
    for split, path in [
        ("train", PROTOCOL_B / "train_driver_genes.csv"),
        ("validation", PROTOCOL_B / "validation_driver_genes.csv"),
    ]:
        genes = load_driver_genes(path)
        present = [g for g in genes if g in by_gene.index]
        split_df = by_gene.loc[present].reset_index()
        for group in MUTATION_GROUPS:
            rows.append(
                {
                    "Split": split,
                    "MutationGroup": group,
                    "DriverGeneCount": int((split_df["MutationGroup"] == group).sum()),
                    "DriverGenesInUniverse": len(present),
                }
            )
    return rows


def percent_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def edge_degrees(path: Path, directed: bool) -> dict[str, int]:
    if not path.exists():
        return {}
    df = read_csv(path, sep="\t")
    degree: dict[str, int] = {}
    for _, row in df.iterrows():
        a = clean_gene(row.iloc[0])
        b = clean_gene(row.iloc[1])
        if not a or not b:
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + (0 if directed else 1)
    return degree


def build_support_table(annotation: pd.DataFrame) -> pd.DataFrame:
    features = read_csv(PROJECT / "data" / "processed" / "KIRC_multiomics_3omics.csv")
    features["Gene"] = features["Gene"].map(clean_gene)
    features = features.dropna(subset=["Gene"]).drop_duplicates("Gene").set_index("Gene")
    ppi_degree = edge_degrees(STAGE1 / "10_stage2_ready" / "ppi_edges_frozen.tsv", directed=False)
    grn_degree = edge_degrees(STAGE1 / "10_stage2_ready" / "grn_edges_frozen.tsv", directed=True)

    genes = annotation["Gene"].tolist()
    expr = np.asarray([float(features.loc[g, "Expression"]) if g in features.index else 0.0 for g in genes])
    meth = np.asarray([float(features.loc[g, "Methylation"]) if g in features.index else 0.0 for g in genes])
    ppi = np.asarray([float(ppi_degree.get(g, 0)) for g in genes])
    grn = np.asarray([float(grn_degree.get(g, 0)) for g in genes])
    network_support = percent_rank(np.log1p(ppi) + np.log1p(grn))
    nonmutation_support = (percent_rank(expr) + percent_rank(meth) + network_support) / 3.0
    out = annotation.copy()
    out["NonMutationSupport"] = nonmutation_support
    out["NetworkSupport"] = network_support
    out["HasNonMutationSupport"] = out["NonMutationSupport"] >= float(np.percentile(nonmutation_support, 75))
    out["HasNetworkSupport"] = out["NetworkSupport"] >= float(np.percentile(network_support, 75))
    return out


def stage4_top150_summary(support: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    by_gene = support.set_index("Gene")
    summary_rows: list[dict] = []
    annotated_rows: list[dict] = []
    for ranking_path in sorted((STAGE4 / "07_formal_runs").glob("*/*/validation_ranking_best.csv")):
        policy = ranking_path.parent.parent.name
        seed = ranking_path.parent.name
        ranking = read_csv(ranking_path)
        top = ranking.head(150).copy()
        genes = [clean_gene(g) for g in top["Gene"]]
        st = by_gene.loc[[g for g in genes if g in by_gene.index]].copy()
        row = {
            "Policy": policy,
            "Seed": seed,
            "TopK": 150,
            "VeryLowCandidateCount": int((st["MutationGroup"] == "very_low").sum()),
            "LowFrequencyCandidateCount": int((st["MutationGroup"] == "low_frequency").sum()),
            "HighFrequencyCandidateCount": int((st["MutationGroup"] == "high_frequency").sum()),
            "VeryLowNonMutationSupportedCount": int(
                ((st["MutationGroup"] == "very_low") & st["HasNonMutationSupport"]).sum()
            ),
            "LowFrequencyNonMutationSupportedCount": int(
                ((st["MutationGroup"] == "low_frequency") & st["HasNonMutationSupport"]).sum()
            ),
            "VeryLowNetworkSupportedCount": int(((st["MutationGroup"] == "very_low") & st["HasNetworkSupport"]).sum()),
            "LowFrequencyNetworkSupportedCount": int(
                ((st["MutationGroup"] == "low_frequency") & st["HasNetworkSupport"]).sum()
            ),
        }
        summary_rows.append(row)
        for _, rank_row in top.iterrows():
            gene = clean_gene(rank_row["Gene"])
            if gene not in by_gene.index:
                continue
            ann = by_gene.loc[gene]
            annotated_rows.append(
                {
                    "Policy": policy,
                    "Seed": seed,
                    "Rank": int(rank_row["Rank"]),
                    "Gene": gene,
                    "Q_value": float(rank_row["Q_value"]),
                    "MutationPatientCount": int(ann["MutationPatientCount"]),
                    "MutationFrequency": float(ann["MutationFrequency"]),
                    "MutationFrequencyPct": float(ann["MutationFrequencyPct"]),
                    "MutationGroup": ann["MutationGroup"],
                    "HasNonMutationSupport": bool(ann["HasNonMutationSupport"]),
                    "HasNetworkSupport": bool(ann["HasNetworkSupport"]),
                }
            )
    return summary_rows, annotated_rows


def audit_markdown() -> str:
    return f"""# Low-Frequency Definition Audit

Scope: targeted mutation-frequency grouping audit for RL-GenRisk.

Current definition:

- very_low: MutationPatientCount < {LOW_FREQUENCY_MUTATION_MIN_COUNT}
- low_frequency: {LOW_FREQUENCY_MUTATION_MIN_COUNT} <= MutationPatientCount <= {LOW_FREQUENCY_MUTATION_MAX_COUNT}
- high_frequency: MutationPatientCount >= {HIGH_FREQUENCY_MUTATION_MIN_COUNT}

Audit checklist:

| File | Location / function | Current definition before this task | Changed? | Reason |
|---|---|---|---|---|
| E:\\Projects\\RL-GenRisk-main\\src\\process_kirc_3omics.py | process_mutation, lines 191-209 | Patient-level count via drop_duplicates([sample, Gene]) then nunique(sample) | NO | This already matches MutationPatientCount semantics. |
| E:\\Projects\\RL-GenRisk-main\\src\\DQN.py | _compose_reward_components, lines 290-303 | Continuous rarity = 1 - Mutation percentile for train-driver lowfreq reward branch | NO reward change | Continuous reward shaping, not a discrete grouping threshold. |
| E:\\Projects\\RL-GenRisk-main\\src\\DQN.py | last_reward_components, lines around 610-632 | Logged count/frequency but no MutationGroup | YES | Add new group/pct fields without changing reward. |
| E:\\Projects\\RL-GenRisk-main\\src\\train.py | ACTION_REWARD_LOG_FIELDNAMES and action_reward_rows | Logged count/frequency only | YES | New action logs can carry MutationGroup. |
| E:\\Projects\\RL-GenRisk-main\\src\\lowfreq_evidence.py | load_evidence_table | Loaded evidence schema without MutationGroup | YES | Add derived MutationFrequencyPct and MutationGroup. |
| E:\\codex_file\\新方向阶段4\\scripts\\stage4_fixed_preference.py | Stage4RewardModel.__init__, lines around 270-275 | old gate: mutation > 0 and <= nonzero Q25, plus zero-mutation supported | YES | Future Stage4 low-frequency candidate gate must use 2-18 patient-count definition and keep very_low separate. |
| E:\\codex_file\\新方向阶段4\\scripts\\stage4_fixed_preference.py | discovery/reporting outputs | LowFrequencyCount@150 was based on old gate | YES | Future analysis adds very_low/low_frequency/high_frequency Top150 counts. |
| E:\\codex_file\\新方向阶段4\\02_reward_design\\*.md and STAGE4_COMPLETION_REPORT.md | historical generated outputs | Historical Q25/zero-supported definition | NO | Historical records are preserved and not overwritten. |
| E:\\codex_file\\新方向阶段2\\06_formal_runs\\*\\config.json | historical formal run configs | reward_mode=legacy; lowfreq weights present but inactive | NO | Historical result/config files are not modified. |
| E:\\Projects\\RL-GenRisk-main\\src\\evaluate_frozen_test*.py | frozen Test evaluation scripts | Mentions multiomics_lowfreq mode, no patient-count grouping threshold found | NO | Test scripts are not used in this task. |
| E:\\Projects\\RL-GenRisk-main\\src\\evaluate_external_holdout.py | external holdout analysis | Uses historical low_frequency_holdout files | NO | Historical/reference analysis; not used for threshold selection. |
"""


def report_text(group_rows, driver_rows, top150_rows, info, hashes) -> str:
    by_group = {r["MutationGroup"]: r["GeneCount"] for r in group_rows}
    driver = {(r["Split"], r["MutationGroup"]): r["DriverGeneCount"] for r in driver_rows}
    top150 = pd.DataFrame(top150_rows)
    stage4_lines = []
    if not top150.empty:
        for policy in sorted(top150["Policy"].unique()):
            sub = top150[top150["Policy"] == policy]
            stage4_lines.append(
                f"- {policy}: mean Top150 very_low={sub['VeryLowCandidateCount'].mean():.2f}, "
                f"low_frequency={sub['LowFrequencyCandidateCount'].mean():.2f}, "
                f"high_frequency={sub['HighFrequencyCandidateCount'].mean():.2f}"
            )
    return f"""# Mutation-Frequency Redefinition Report

Run time UTC: {datetime.now(timezone.utc).isoformat()}

1. Before this task, active Stage4 low-frequency candidate logic used mutation > 0 and <= non-zero Q25 plus zero-mutation-supported candidates. Main DQN lowfreq reward used continuous rarity, not a patient-count class.
2. <=5 or <=10 were found as historical concepts in prior experiment documentation/search targets; active future code modified here did not use literal <=5/<=10 gates. Historical records were not edited.
3. Historical records left unchanged: Stage2 formal configs/results, Stage4 generated reports/results/checkpoints, frozen Test/external holdout artifacts.
4. Future analysis code changed: main evidence/action logging and Stage4 low-frequency candidate/statistics generation.
5. Current unified definition is exactly: very_low N < 2; low_frequency 2 <= N <= 18; high_frequency N >= 19.
6. With {TOTAL_KIRC_TUMOR_SAMPLES} samples: 0=0.000%, 1={mutation_frequency_pct(1):.3f}%, 2={mutation_frequency_pct(2):.3f}%, 18={mutation_frequency_pct(18):.3f}%, 19={mutation_frequency_pct(19):.3f}%.
7. Current 9039-gene universe: very_low={by_group.get('very_low', 0)}, low_frequency={by_group.get('low_frequency', 0)}, high_frequency={by_group.get('high_frequency', 0)}.
8. Train drivers: very_low={driver.get(('train','very_low'),0)}, low_frequency={driver.get(('train','low_frequency'),0)}, high_frequency={driver.get(('train','high_frequency'),0)}.
9. Validation drivers: very_low={driver.get(('validation','very_low'),0)}, low_frequency={driver.get(('validation','low_frequency'),0)}, high_frequency={driver.get(('validation','high_frequency'),0)}.
10. Stage4 Top150 reannotation:
{chr(10).join(stage4_lines) if stage4_lines else '- No Stage4 ranking files found.'}
11. Training performed: NO.
12. Test read or used: NO.
13. Core model architecture/DDQN/PER/Soft Update modified: NO.
14. Continuous MutationRarityScore/reward rarity retained: YES.
15. Historical experiment definitions/results preserved: YES.

Input notes:

- gene_universe_count={info['gene_universe_count']}
- mutation_table_gene_count={info['mutation_table_gene_count']}
- missing_universe_genes_treated_as_zero={info['missing_universe_genes_treated_as_zero']}

Modified/generated source and output file SHA256:

{chr(10).join(f'- {path}: {digest}' for path, digest in hashes.items())}

MUTATION_GROUP_DEFINITION =
very_low: N < 2
low_frequency: 2 <= N <= 18
high_frequency: N >= 19

TOTAL_KIRC_TUMOR_SAMPLES = {TOTAL_KIRC_TUMOR_SAMPLES}

TRAINING_PERFORMED = NO

TEST_USED_FOR_THRESHOLD_SELECTION = NO

HISTORICAL_RESULTS_OVERWRITTEN = NO
"""


def run_boundary_tests(annotation: pd.DataFrame) -> dict:
    expected = {
        0: "very_low",
        1: "very_low",
        2: "low_frequency",
        18: "low_frequency",
        19: "high_frequency",
        100: "high_frequency",
    }
    actual = {count: classify_mutation_frequency(count) for count in expected}
    if actual != expected:
        raise AssertionError(f"Boundary test failed: {actual}")
    if set(annotation["MutationGroup"].unique()) - set(MUTATION_GROUPS):
        raise AssertionError("Unexpected MutationGroup value found.")
    if (annotation["MutationPatientCount"] < 0).any():
        raise AssertionError("Negative MutationPatientCount found.")
    reconstructed = annotation["MutationPatientCount"] / TOTAL_KIRC_TUMOR_SAMPLES
    if not np.allclose(reconstructed, annotation["MutationFrequency"], rtol=0.0, atol=1e-12):
        raise AssertionError("MutationFrequency mismatch.")
    group_sum = sum((annotation["MutationGroup"] == group).sum() for group in MUTATION_GROUPS)
    if int(group_sum) != len(annotation):
        raise AssertionError("Mutation groups do not cover all genes exactly once.")
    return {"boundary_actual": actual, "status": "PASS"}


def main() -> None:
    for name in ["audit", "tables", "reports", "logs"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)

    genes, universe_source = load_gene_universe()
    annotation, info = load_mutation_annotation(genes)
    info["gene_universe_source"] = universe_source
    support = build_support_table(annotation)
    group_rows = summarize_groups(annotation)
    driver_rows = driver_group_summary(annotation)
    top150_rows, top150_annotated = stage4_top150_summary(support)
    tests = run_boundary_tests(annotation)

    write_csv(OUT / "tables" / "mutation_frequency_group_summary.csv", group_rows)
    write_csv(OUT / "tables" / "mutation_frequency_gene_annotation.csv", annotation.to_dict("records"))
    write_csv(OUT / "tables" / "driver_mutation_group_summary.csv", driver_rows)
    write_csv(OUT / "tables" / "stage4_top150_mutation_group_summary.csv", top150_rows)
    write_csv(OUT / "tables" / "stage4_top150_gene_annotation.csv", top150_annotated)
    write_text(OUT / "audit" / "low_frequency_definition_audit.md", audit_markdown())
    write_text(OUT / "reports" / "boundary_tests.json", json.dumps(tests, indent=2, ensure_ascii=False) + "\n")

    log(OUT / "logs" / "execution_log.txt", "write execution log", "write")
    pd.DataFrame(LOG_ROWS).to_csv(OUT / "logs" / "execution_log.txt", index=False)

    generated = [
        OUT / "audit" / "low_frequency_definition_audit.md",
        OUT / "tables" / "mutation_frequency_group_summary.csv",
        OUT / "tables" / "mutation_frequency_gene_annotation.csv",
        OUT / "tables" / "driver_mutation_group_summary.csv",
        OUT / "tables" / "stage4_top150_mutation_group_summary.csv",
        OUT / "tables" / "stage4_top150_gene_annotation.csv",
        OUT / "reports" / "boundary_tests.json",
        OUT / "logs" / "execution_log.txt",
    ]
    modified_sources = [
        PROJECT / "src" / "mutation_frequency.py",
        PROJECT / "src" / "lowfreq_evidence.py",
        PROJECT / "src" / "DQN.py",
        PROJECT / "src" / "train.py",
        PROJECT / "scripts" / "audit_mutation_frequency_redefinition.py",
        STAGE4 / "scripts" / "stage4_fixed_preference.py",
    ]
    hashes = {str(path): sha256(path) for path in [*modified_sources, *generated]}
    write_text(OUT / "reports" / "mutation_frequency_redefinition_report.md", report_text(group_rows, driver_rows, top150_rows, info, hashes))
    hash_rows = [
        {
            "Path": str(path),
            "SHA256": sha256(path),
            "Category": "source_modified" if path in modified_sources else "generated_output",
        }
        for path in [*modified_sources, *generated, OUT / "reports" / "mutation_frequency_redefinition_report.md"]
    ]
    write_csv(OUT / "tables" / "modified_files_sha256.csv", hash_rows)
    print(json.dumps({"status": "PASS", "output_dir": str(OUT), "group_counts": group_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
