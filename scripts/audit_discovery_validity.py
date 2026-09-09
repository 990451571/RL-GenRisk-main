#!/usr/bin/env python3
"""Discovery validity audit using frozen rankings and reward-independent evidence.

This script never reads Test labels, trains a model, updates a checkpoint, or
changes the reward. External raw files are downloaded into a temporary folder;
only compact derived audit artifacts are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "discovery_validity_audit_20260909"
FREEZE = OUT / "frozen_candidates" / "freeze_manifest.json"
PROTOCOL = OUT / "audit_protocol.json"
PROTOCOL_AMENDMENT = OUT / "audit_protocol_amendment_01.json"
EVIDENCE = Path(
    "/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/"
    "low_frequency_evidence_table_internal_v2.csv"
)
HOLDOUT = Path(
    "/mnt/e/codex_file/二阶段/00_低频基因盲评集/output/"
    "low_frequency_holdout.csv"
)
GRN_ROOT = Path("/mnt/e/codex_file/新方向阶段1/06_grn_audit")
TRAIN_LABEL = REPO / "experiments/protocol_B/train_driver_genes.csv"
VAL_LABEL = REPO / "experiments/protocol_B/validation_driver_genes.csv"
MULTIOMICS = REPO / "data/processed/KIRC_multiomics_3omics.csv"

DEPMAP_MODEL_URL = "https://ndownloader.figshare.com/files/51065297"
DEPMAP_EFFECT_URL = "https://ndownloader.figshare.com/files/51064667"
CPTAC_MAP_URL = (
    "https://zenodo.org/api/records/8212665/files/"
    "bcm-ccrcc-mapping-gencode.v34.basic.annotation-mapping.txt.gz/content"
)
CPTAC_TUMOR_URL = (
    "https://zenodo.org/api/records/8212665/files/"
    "bcm-ccrcc-proteomics-CCRCC_proteomics_gene_abundance_log2_reference_"
    "intensity_normalized_Tumor.txt.gz/content"
)
CPTAC_NORMAL_URL = (
    "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/"
    "data_freeze_v1.2_reorganized/CCRCC/"
    "CCRCC_proteomics_gene_abundance_log2_reference_intensity_normalized_Normal.txt"
)
CPTAC_MUTATION_URL = (
    "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/"
    "data_freeze_v1.2_reorganized/CCRCC/CCRCC_somatic_mutation_gene_level_binary.txt"
)


def clean_gene(value) -> str:
    return str(value).strip().upper()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> dict:
    print(f"[download] {url}", flush=True)
    h = hashlib.sha256()
    size = 0
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RL-GenRisk reproducibility audit/1.0)"}
    with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                h.update(chunk)
                size += len(chunk)
                if size // (64 * 1024 * 1024) != (size - len(chunk)) // (64 * 1024 * 1024):
                    suffix = f"/{total / 1e6:.1f} MB" if total else ""
                    print(f"  {size / 1e6:.1f} MB{suffix}", flush=True)
    return {"url": url, "bytes": size, "sha256": h.hexdigest()}


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if not len(vals):
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[ok] = restored
    return out


def load_gene_set(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    return {clean_gene(x) for x in frame.iloc[:, 0].dropna()}


def load_base_table() -> tuple[pd.DataFrame, set[str]]:
    ev = pd.read_csv(EVIDENCE)
    ev["Gene"] = ev["Gene"].map(clean_gene)
    train = load_gene_set(TRAIN_LABEL)
    val = load_gene_set(VAL_LABEL)
    known = train | val
    lowfreq = ev["MutationPatientCount"].between(2, 18) & ~ev["Gene"].isin(known)
    pool = set(ev.loc[lowfreq, "Gene"])
    if len(ev) != 9039 or len(pool) != 2419:
        raise RuntimeError(f"Unexpected universe/pool size: {len(ev)}/{len(pool)}")

    omics = pd.read_csv(MULTIOMICS)
    omics["Gene"] = omics["Gene"].map(clean_gene)
    omics = omics.rename(columns={
        "Mutation": "Omics_Mutation",
        "Expression": "Omics_Expression",
        "Methylation": "Omics_Methylation",
    })
    frame = ev.merge(omics, on="Gene", how="left")
    for filename, column in [
        ("grn_in_degree.csv", "GRN_in_degree"),
        ("grn_out_degree.csv", "GRN_out_degree"),
        ("grn_total_degree.csv", "GRN_total_degree"),
    ]:
        part = pd.read_csv(GRN_ROOT / filename)
        part["Gene"] = part["Gene"].map(clean_gene)
        source_col = next(c for c in part.columns if c != "Gene")
        frame = frame.merge(part[["Gene", source_col]].rename(columns={source_col: column}), on="Gene", how="left")
    frame = frame.rename(columns={"Degree": "PPI_Degree"})
    return frame, pool


def ensembl_map(mapping: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    mapping = mapping.dropna(subset=["gene", "gene_name"]).copy()
    mapping["gene_name"] = mapping["gene_name"].map(clean_gene)
    full = dict(zip(mapping["gene"].astype(str), mapping["gene_name"]))
    base = dict(zip(mapping["gene"].astype(str).str.split(".").str[0], mapping["gene_name"]))
    return full, base


def map_ensembl(value, full: dict[str, str], base: dict[str, str]) -> str | None:
    key = str(value)
    return full.get(key) or base.get(key.split(".")[0])


def derive_cptac(tmp: Path, universe: set[str], source_records: list[dict], cache: Path | None = None) -> pd.DataFrame:
    paths = {
        "mapping": tmp / "cptac_mapping.txt.gz",
        "tumor": tmp / "cptac_protein_tumor.txt.gz",
        "normal": tmp / "cptac_protein_normal.txt",
        "mutation": tmp / "cptac_mutation.txt",
    }
    cache_names = {"mapping": "map.txt.gz", "tumor": "tumor.txt.gz", "normal": "normal.txt", "mutation": "mut.txt"}
    for key, url in [
        ("mapping", CPTAC_MAP_URL), ("tumor", CPTAC_TUMOR_URL),
        ("normal", CPTAC_NORMAL_URL), ("mutation", CPTAC_MUTATION_URL),
    ]:
        cached = cache / cache_names[key] if cache else None
        if cached is not None and cached.exists():
            paths[key] = cached
            rec = {"url": url, "retrieval": "verified temporary cache", "bytes": cached.stat().st_size, "sha256": sha256(cached)}
            print(f"[cache] CPTAC {key}: {cached}", flush=True)
        else:
            rec = download(url, paths[key])
        rec["name"] = f"CPTAC_{key}"; source_records.append(rec)

    mapping = pd.read_csv(paths["mapping"], sep="\t")
    full, base = ensembl_map(mapping)
    mutation = pd.read_csv(paths["mutation"], sep="\t")
    mutation["Gene"] = mutation["idx"].map(lambda x: map_ensembl(x, full, base))
    mutation = mutation.dropna(subset=["Gene"]).drop(columns="idx")
    mut_cols = [c for c in mutation.columns if c != "Gene"]
    mutation[mut_cols] = mutation[mut_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    mut_gene = mutation.groupby("Gene", sort=False)[mut_cols].max()
    mut_count = mut_gene.sum(axis=1).rename("CPTACMutationCount")

    tumor = pd.read_csv(paths["tumor"], sep="\t")
    normal = pd.read_csv(paths["normal"], sep="\t")
    for table in (tumor, normal):
        table["Gene"] = table["idx"].map(lambda x: map_ensembl(x, full, base))
    tumor = tumor.dropna(subset=["Gene"]).drop(columns="idx").groupby("Gene", sort=False).median(numeric_only=True)
    normal = normal.dropna(subset=["Gene"]).drop(columns="idx").groupby("Gene", sort=False).median(numeric_only=True)
    paired = sorted(set(tumor.columns) & set(normal.columns))
    rows = []
    for gene in sorted(universe & set(tumor.index) & set(normal.index)):
        delta = (tumor.loc[gene, paired] - normal.loc[gene, paired]).dropna().to_numpy(float)
        if len(delta) >= 20:
            effect = float(np.median(delta))
            if np.allclose(delta, 0):
                pvalue = 1.0
            else:
                try:
                    pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
                except ValueError:
                    pvalue = 1.0
        else:
            effect, pvalue = np.nan, np.nan
        rows.append((gene, len(delta), effect, pvalue))
    result = pd.DataFrame(rows, columns=["Gene", "CPTACProteinPairs", "CPTACProteinMedianDelta", "CPTACProteinP"])
    result["CPTACProteinQ"] = bh_fdr(result["CPTACProteinP"].to_numpy())
    result = result.merge(mut_count, left_on="Gene", right_index=True, how="outer")
    result.index.name = None
    if "Gene" not in result.columns:
        result = result.reset_index().rename(columns={"index": "Gene"})
    result["Gene"] = result["Gene"].map(clean_gene)
    result["CPTACMutationCount"] = result["CPTACMutationCount"].fillna(0).astype(int)
    result["CPTACRecurrence"] = result["CPTACMutationCount"] >= 2
    result["CPTACProteinSupport"] = (
        (result["CPTACProteinPairs"] >= 20)
        & (result["CPTACProteinMedianDelta"].abs() >= 1.0)
        & (result["CPTACProteinQ"] < 0.05)
    )
    print(f"[CPTAC] mapped genes={result.Gene.nunique()}, matched pairs={len(paired)}", flush=True)
    return result.drop_duplicates("Gene")


def derive_depmap(tmp: Path, universe: set[str], source_records: list[dict]) -> pd.DataFrame:
    model_path = tmp / "depmap_24q4_model.csv"
    effect_path = tmp / "depmap_24q4_crispr_gene_effect.csv"
    for name, url, path in [
        ("DepMap24Q4_Model", DEPMAP_MODEL_URL, model_path),
        ("DepMap24Q4_CRISPRGeneEffect", DEPMAP_EFFECT_URL, effect_path),
    ]:
        rec = download(url, path); rec["name"] = name; source_records.append(rec)
    models = pd.read_csv(model_path)
    ccrcc = set(models.loc[models["OncotreeSubtype"].eq("Renal Clear Cell Carcinoma"), "ModelID"].astype(str))
    nonkidney = set(models.loc[~models["OncotreeLineage"].eq("Kidney"), "ModelID"].astype(str))

    header = pd.read_csv(effect_path, nrows=0).columns.tolist()
    gene_by_col = {}
    for col in header[1:]:
        gene = clean_gene(re.sub(r"\s+\([^)]*\)$", "", str(col)))
        if gene in universe and gene not in gene_by_col.values():
            gene_by_col[col] = gene
    usecols = [header[0], *gene_by_col]
    data = pd.read_csv(effect_path, usecols=usecols)
    data = data.rename(columns={header[0]: "ModelID", **gene_by_col}).set_index("ModelID")
    ccrcc_ids = sorted(ccrcc & set(data.index.astype(str)))
    nonkidney_ids = sorted(nonkidney & set(data.index.astype(str)))
    ccrcc_values = data.loc[ccrcc_ids]
    nonkidney_values = data.loc[nonkidney_ids]
    result = pd.DataFrame({
        "Gene": data.columns,
        "DepMapCCRCCN": ccrcc_values.notna().sum(axis=0).to_numpy(),
        "DepMapCCRCCMedian": ccrcc_values.median(axis=0).to_numpy(),
        "DepMapNonKidneyN": nonkidney_values.notna().sum(axis=0).to_numpy(),
        "DepMapNonKidneyMedian": nonkidney_values.median(axis=0).to_numpy(),
    })
    result["DepMapSelectivityDelta"] = result["DepMapNonKidneyMedian"] - result["DepMapCCRCCMedian"]
    result["DepMapCCRCCDependency"] = (result["DepMapCCRCCN"] >= 5) & (result["DepMapCCRCCMedian"] <= -0.5)
    result["DepMapCCRCCSelective"] = result["DepMapCCRCCDependency"] & (result["DepMapSelectivityDelta"] >= 0.25)
    print(
        f"[DepMap] ccRCC models={len(ccrcc_ids)}, non-kidney models={len(nonkidney_ids)}, genes={len(result)}",
        flush=True,
    )
    return result


def load_rankings(base: pd.DataFrame, freeze: dict) -> list[dict]:
    rankings = []
    for method, column in [
        ("EvidenceScore", "LowFrequencyEvidenceScoreV2"),
        ("mutation", "MutationFrequency"),
        ("PPI_degree", "PPI_Degree"),
        ("GRN_degree", "GRN_total_degree"),
    ]:
        rank = base[["Gene", column]].rename(columns={column: "Score"}).copy()
        rank = rank.sort_values(["Score", "Gene"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        rank["Rank"] = np.arange(1, len(rank) + 1)
        rankings.append({"Method": method, "Family": method, "Seed": np.nan, "Profile": "static", "data": rank})

    for seed in (42, 45, 46):
        for family, file_tag in [("MLP", "MLP"), ("GCN", "GCN")]:
            path = REPO / f"outputs/rlnecessity_supervised_20260907_152513/seed{seed}/rankings/supervised_{file_tag}_final.csv"
            rank = pd.read_csv(path)
            rank["Gene"] = rank["Gene"].map(clean_gene)
            rankings.append({"Method": family, "Family": family, "Seed": seed, "Profile": "supervised", "data": rank})

    seen = set()
    for entry in freeze["files"]:
        path = Path(entry["source_ranking"])
        if sha256(path) != entry["source_ranking_sha256"]:
            raise RuntimeError(f"Frozen source hash changed: {path}")
        key = (entry["profile"], int(entry["seed"]))
        if key in seen:
            continue
        seen.add(key)
        rank = pd.read_csv(path)
        rank["Gene"] = rank["Gene"].map(clean_gene)
        rankings.append({
            "Method": f"bandit_{entry['profile']}", "Family": "bandit",
            "Seed": int(entry["seed"]), "Profile": entry["profile"], "data": rank,
        })
    return rankings


def correlations(base: pd.DataFrame, pool: set[str], rankings: list[dict]) -> pd.DataFrame:
    feature_cols = [
        "MutationFrequency", "PPI_Degree", "GRN_in_degree", "GRN_out_degree", "GRN_total_degree",
        "Omics_Mutation", "Omics_Expression", "Omics_Methylation",
    ]
    score_sources = [{
        "Method": "EvidenceScore", "Seed": np.nan, "Profile": "static",
        "ScoreType": "EvidenceScore", "data": base[["Gene", "LowFrequencyEvidenceScoreV2"]].rename(columns={"LowFrequencyEvidenceScoreV2": "AuditScore"}),
    }]
    for item in rankings:
        if item["Family"] != "bandit":
            continue
        rank = item["data"][["Gene", "Rank", "Score"]].copy()
        score_sources.extend([
            {**{k: item[k] for k in ("Method", "Seed", "Profile")}, "ScoreType": "BanditRolloutScore", "data": rank.rename(columns={"Score": "AuditScore"})[["Gene", "AuditScore"]]},
            {**{k: item[k] for k in ("Method", "Seed", "Profile")}, "ScoreType": "BanditRankScore", "data": rank.assign(AuditScore=-rank["Rank"])[["Gene", "AuditScore"]]},
        ])
    rows = []
    for source in score_sources:
        merged = base[["Gene", *feature_cols]].merge(source["data"], on="Gene", how="inner")
        for scope, scoped in [("all_9039", merged), ("lowfreq_novel", merged[merged.Gene.isin(pool)])]:
            for feature in feature_cols:
                valid = scoped[["AuditScore", feature]].dropna()
                rho, pvalue = spearmanr(valid["AuditScore"], valid[feature]) if len(valid) >= 3 else (np.nan, np.nan)
                rows.append({
                    **{k: source[k] for k in ("Method", "Seed", "Profile", "ScoreType")},
                    "Scope": scope, "Feature": feature, "N": len(valid),
                    "SpearmanRho": rho, "PValue": pvalue,
                })
    return pd.DataFrame(rows)


def pairwise_jaccard(sets: list[set[str]]) -> tuple[float, float, int]:
    values = [len(a & b) / len(a | b) if a | b else np.nan for a, b in combinations(sets, 2)]
    return (float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan, len(values))


def evaluate_rankings(rankings: list[dict], evidence: pd.DataFrame, pool: set[str], ks: list[int]):
    lookup = evidence.set_index("Gene")
    signals = [
        "IndependentAnyHit", "IndependentStrongHit", "FrozenLiteratureHoldout",
        "CPTACRecurrence", "CPTACProteinSupport", "DepMapCCRCCDependency", "DepMapCCRCCSelective",
        "PostHocNonMutationAnyHit", "PostHocNonMutationStrongHit",
    ]
    pool_rates = {s: float(lookup.reindex(sorted(pool))[s].fillna(False).mean()) for s in signals}
    rows, selected_rows = [], []
    selection_sets = {}
    for item in rankings:
        order = item["data"].sort_values("Rank")["Gene"].tolist()
        for scope in ("candidate_pool", "global_then_filter"):
            for k in ks:
                selected = [g for g in order if g in pool][:k] if scope == "candidate_pool" else [g for g in order[:k] if g in pool]
                key = (item["Method"], item["Seed"], item["Profile"], scope, k)
                selection_sets[key] = set(selected)
                for rank_within, gene in enumerate(selected, 1):
                    selected_rows.append({
                        "Method": item["Method"], "Seed": item["Seed"], "Profile": item["Profile"],
                        "Scope": scope, "K": k, "SelectedRank": rank_within, "Gene": gene,
                    })
                sub = lookup.reindex(selected)
                for signal in signals:
                    hit_n = int(sub[signal].fillna(False).sum()) if len(sub) else 0
                    rate = hit_n / len(selected) if selected else np.nan
                    baseline = pool_rates[signal]
                    rows.append({
                        "Method": item["Method"], "Family": item["Family"], "Seed": item["Seed"],
                        "Profile": item["Profile"], "Scope": scope, "K": k,
                        "Evidence": signal, "SelectedN": len(selected), "HitN": hit_n,
                        "HitRate": rate, "PoolHitRate": baseline,
                        "FoldEnrichment": rate / baseline if baseline > 0 and np.isfinite(rate) else np.nan,
                    })
    metrics = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows).merge(evidence, on="Gene", how="left")

    stability = []
    for (method, profile, scope, k), group in metrics.groupby(["Method", "Profile", "Scope", "K"], dropna=False):
        seeds = sorted(group["Seed"].dropna().unique())
        if len(seeds) >= 2:
            sets = [selection_sets[(method, seed, profile, scope, k)] for seed in seeds]
            mean, sd, pairs = pairwise_jaccard(sets)
        else:
            mean, sd, pairs = 1.0, 0.0, 0
        stability.append({"Method": method, "Profile": profile, "Scope": scope, "K": k, "SeedN": len(seeds), "PairN": pairs, "MeanPairwiseJaccard": mean, "SDPairwiseJaccard": sd})
    return metrics, selected, pd.DataFrame(stability)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["Method", "Family", "Profile", "Scope", "K", "Evidence"], dropna=False)
    return grouped.agg(
        RunN=("HitRate", "count"), SelectedNMean=("SelectedN", "mean"),
        HitRateMean=("HitRate", "mean"), HitRateSD=("HitRate", "std"),
        FoldMean=("FoldEnrichment", "mean"), FoldSD=("FoldEnrichment", "std"),
    ).reset_index()


def paired_supervised_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Strictly paired Bandit-vs-MLP/GCN comparison on the shared seeds 42/45."""
    sub = metrics[
        (metrics["Scope"] == "candidate_pool")
        & metrics["Evidence"].isin(["IndependentAnyHit", "IndependentStrongHit"])
        & metrics["Seed"].isin([42, 45])
    ].copy()
    rows = []
    for bandit_method in sorted(x for x in sub.Method.unique() if str(x).startswith("bandit_")):
        for comparator in ("MLP", "GCN"):
            left = sub[sub.Method.eq(bandit_method)]
            right = sub[sub.Method.eq(comparator)]
            paired = left.merge(right, on=["Seed", "K", "Evidence"], suffixes=("_Bandit", "_Comparator"))
            for _, row in paired.iterrows():
                rows.append({
                    "BanditMethod": bandit_method, "Comparator": comparator,
                    "Seed": int(row.Seed), "K": int(row.K), "Evidence": row.Evidence,
                    "BanditHitRate": row.HitRate_Bandit, "ComparatorHitRate": row.HitRate_Comparator,
                    "HitRateDifference": row.HitRate_Bandit - row.HitRate_Comparator,
                    "BanditFold": row.FoldEnrichment_Bandit, "ComparatorFold": row.FoldEnrichment_Comparator,
                    "FoldDifference": row.FoldEnrichment_Bandit - row.FoldEnrichment_Comparator,
                })
    return pd.DataFrame(rows)


def reordering_audit(base: pd.DataFrame, rankings: list[dict]) -> pd.DataFrame:
    baseline_orders = {}
    for item in rankings:
        if item["Method"] in {"EvidenceScore", "PPI_degree", "GRN_degree", "mutation"}:
            baseline_orders[item["Method"]] = item["data"].sort_values("Rank")["Gene"].tolist()
    base_index = base.set_index("Gene")
    percentile = {
        "EvidenceScore": base_index["LowFrequencyEvidenceScoreV2"].rank(pct=True),
        "PPI_degree": base_index["PPI_Degree"].rank(pct=True),
        "GRN_degree": base_index["GRN_total_degree"].rank(pct=True),
    }
    rows = []
    for item in rankings:
        if item["Family"] != "bandit":
            continue
        order = item["data"].sort_values("Rank")["Gene"].tolist()
        bandit_top = set(order[:150])
        bandit_rank = pd.Series(np.arange(1, len(order) + 1), index=order)
        for baseline, base_order in baseline_orders.items():
            base_top = set(base_order[:150])
            base_rank = pd.Series(np.arange(1, len(base_order) + 1), index=base_order)
            common = bandit_rank.index.intersection(base_rank.index)
            rho = float(spearmanr(-bandit_rank.loc[common], -base_rank.loc[common]).statistic)
            jac = len(bandit_top & base_top) / len(bandit_top | base_top)
            row = {
                "Method": item["Method"], "Seed": item["Seed"], "Profile": item["Profile"],
                "Baseline": baseline, "FullRankSpearman": rho, "Top150OverlapN": len(bandit_top & base_top),
                "Top150Jaccard": jac, "HighRankSimilarity": abs(rho) >= 0.80,
                "HighTop150Overlap": jac >= 0.67,
            }
            if baseline in percentile:
                vals = percentile[baseline].reindex(sorted(bandit_top)).dropna()
                row["BanditTop150MedianBaselinePercentile"] = float(vals.median())
                row["BanditTop150FractionBaselineTop10Pct"] = float((vals >= 0.90).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def make_plots(corr: pd.DataFrame, summary: pd.DataFrame, reorder: pd.DataFrame, stability: pd.DataFrame):
    plot_dir = OUT / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    c = corr[(corr.Scope == "all_9039") & (corr.ScoreType.isin(["EvidenceScore", "BanditRankScore"]))].copy()
    c["Series"] = np.where(c.ScoreType.eq("EvidenceScore"), "EvidenceScore", c.Method + "_s" + c.Seed.fillna(0).astype(int).astype(str))
    pivot = c.pivot_table(index="Series", columns="Feature", values="SpearmanRho", aggfunc="mean")
    plt.figure(figsize=(13, max(5, 0.45 * len(pivot))))
    sns.heatmap(pivot, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "Spearman rho"})
    plt.title("Score/rank correlations with mutation, network degree and omics")
    plt.tight_layout(); plt.savefig(plot_dir / "01_score_feature_spearman.png", dpi=220); plt.close()

    s = summary[(summary.Scope == "candidate_pool") & (summary.K == 150) & (summary.Evidence == "IndependentAnyHit")].copy()
    s = s.sort_values("FoldMean", ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=s, y="Method", x="FoldMean", hue="Method", legend=False, color="#3973AC")
    plt.axvline(1, color="black", ls="--", lw=1)
    plt.xlabel("Fold enrichment vs low-frequency novel pool"); plt.ylabel("")
    plt.title("Independent-evidence enrichment among candidate-pool Top-150")
    plt.tight_layout(); plt.savefig(plot_dir / "02_external_enrichment_top150.png", dpi=220); plt.close()

    r = reorder[reorder.Baseline.isin(["EvidenceScore", "PPI_degree", "GRN_degree"])].copy()
    plt.figure(figsize=(12, 7))
    sns.barplot(data=r, x="Profile", y="Top150Jaccard", hue="Baseline", errorbar="sd")
    plt.axhline(0.67, color="black", ls="--", lw=1, label="high-overlap heuristic")
    plt.ylim(0, 1); plt.title("Bandit Top-150 overlap with simple rankings")
    plt.tight_layout(); plt.savefig(plot_dir / "03_bandit_baseline_top150_overlap.png", dpi=220); plt.close()

    st = stability[(stability.Scope == "candidate_pool") & (stability.K == 150) & stability.Method.str.startswith(("bandit", "MLP", "GCN"))]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=st, y="Method", x="MeanPairwiseJaccard", hue="Method", legend=False, color="#4C956C")
    plt.xlim(0, 1); plt.title("Cross-seed stability of low-frequency candidate Top-150")
    plt.tight_layout(); plt.savefig(plot_dir / "04_cross_seed_stability.png", dpi=220); plt.close()


def write_report(base: pd.DataFrame, pool: set[str], external: pd.DataFrame, metrics: pd.DataFrame,
                 summary: pd.DataFrame, reorder: pd.DataFrame, stability: pd.DataFrame,
                 paired: pd.DataFrame):
    primary = summary[(summary.Scope == "candidate_pool") & (summary.K == 150) & (summary.Evidence == "IndependentAnyHit")].copy()
    simple = primary[primary.Method.isin(["EvidenceScore", "mutation", "PPI_degree", "GRN_degree"])]
    simple_best = float(simple.HitRateMean.max())
    bandit_runs = metrics[(metrics.Scope == "candidate_pool") & (metrics.K == 150) & (metrics.Evidence == "IndependentAnyHit") & metrics.Method.str.startswith("bandit_")]
    decisions = []
    for method, group in bandit_runs.groupby("Method"):
        rates = group.HitRate.dropna()
        passes = len(rates) == 3 and bool((rates > simple_best).all())
        decisions.append((method, float(rates.mean()), float(rates.std(ddof=1)), passes))
    retain = any(x[3] for x in decisions)

    lines = [
        "# Discovery 有效性审计",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 结论",
        "",
        ("按预先冻结的严格规则，至少一个 Bandit 三档策略在 3 个 seed 上均超过最强简单基线，当前证据支持保留 Bandit 主线。" if retain else
         "按预先冻结并经逻辑勘误的严格规则，没有 Bandit 策略在 3 个 seed 上均超过最强 mutation/EvidenceScore/degree 简单基线；当前证据不支持继续把 Bandit 作为主线，应转向静态融合/监督排序。"),
        "",
        "这是方法路线结论，不是新癌基因或临床有效性结论。",
        "",
        "## 事实边界",
        "",
        f"- 全基因宇宙：{len(base)}；低频新候选池：{len(pool)}。",
        f"- 低频候选池内冻结文献盲评阳性：{int(external.FrozenLiteratureHoldout.sum())}/16（其余盲评基因为 N<2，不属于本轮固定低频定义）。",
        f"- 低频候选池内 CPTAC 独立复现阳性：{int(external.CPTACRecurrence.sum())}；CPTAC 蛋白支持：{int(external.CPTACProteinSupport.sum())}。",
        f"- 低频候选池内 DepMap ccRCC 依赖：{int(external.DepMapCCRCCDependency.sum())}；其中选择性依赖：{int(external.DepMapCCRCCSelective.sum())}。",
        "- Test 标签未读取；没有训练、checkpoint 重选、reward 调整、模型修改或新增 seed。",
        "- GRN degree 只用于事后混杂审计，不是最新 Bandit 的输入特征。",
        "",
        "## 主要比较（共同低频候选池 Top-150）",
        "",
        "| 方法 | n | 独立证据命中率 mean±SD | Fold mean±SD |",
        "|---|---:|---:|---:|",
    ]
    for _, row in primary.sort_values("HitRateMean", ascending=False).iterrows():
        hsd = 0.0 if pd.isna(row.HitRateSD) else row.HitRateSD
        fsd = 0.0 if pd.isna(row.FoldSD) else row.FoldSD
        lines.append(f"| {row.Method} | {int(row.RunN)} | {row.HitRateMean:.3f}±{hsd:.3f} | {row.FoldMean:.2f}±{fsd:.2f} |")
    lines.extend(["", "严格 Bandit 判定："])
    for method, mean, sd, passed in decisions:
        lines.append(f"- {method}: {mean:.3f}±{sd:.3f}; 三个 seed 均高于最强简单基线={str(passed).lower()}。")

    sensitivity = summary[(summary.Scope == "candidate_pool") & (summary.K == 150) & (summary.Evidence == "PostHocNonMutationAnyHit")]
    lines.extend([
        "", "## 事后敏感性分析：剔除 CPTAC 突变复现", "",
        "该分析只检查主结果是否被独立队列突变频率主导，不用于重新选择模型。",
        "", "| 方法 | 非突变外部证据命中率 | Fold |", "|---|---:|---:|",
    ])
    for _, row in sensitivity.sort_values("HitRateMean", ascending=False).iterrows():
        lines.append(f"| {row.Method} | {row.HitRateMean:.3f} | {row.FoldMean:.2f} |")

    paired150 = paired[(paired.K == 150) & (paired.Evidence == "IndependentAnyHit")]
    lines.extend(["", "## 与监督模型的严格配对比较（仅 seed42/45）", ""])
    for (method, comparator), group in paired150.groupby(["BanditMethod", "Comparator"]):
        lines.append(
            f"- {method} vs {comparator}: 命中率差 mean={group.HitRateDifference.mean():+.3f}，"
            f"两 seed 胜/负={int((group.HitRateDifference > 0).sum())}/{int((group.HitRateDifference < 0).sum())}。"
        )

    high = reorder[reorder.HighRankSimilarity | reorder.HighTop150Overlap]
    lines.extend([
        "", "## 是否只是简单分数重排", "",
        f"- 触发高相似启发式的 Bandit–baseline 配对：{len(high)}/{len(reorder)}。",
        "- 阈值为 |Spearman|≥0.80 或 Top-150 Jaccard≥0.67；它只是诊断阈值，不是统计学等价证明。",
    ])
    if len(high):
        for _, row in high.iterrows():
            lines.append(f"- {row.Method}/seed{int(row.Seed)} vs {row.Baseline}: rho={row.FullRankSpearman:.3f}, Jaccard={row.Top150Jaccard:.3f}。")
    else:
        lines.append("- 没有配对达到高相似阈值；但这不自动证明 Bandit 学到了新的生物机制。")

    st = stability[(stability.Scope == "candidate_pool") & (stability.K == 150)]
    lines.extend(["", "## 跨 seed 稳定性", ""])
    for _, row in st.iterrows():
        if row.SeedN >= 2:
            lines.append(f"- {row.Method}: mean pairwise Jaccard={row.MeanPairwiseJaccard:.3f}（{int(row.SeedN)} seeds）。")

    lines.extend([
        "", "## 限制与风险", "",
        "- 16 基因盲评集很小且为人工阳性集合；它可检验召回，不能估计完整假阳性率。",
        "- DepMap 是细胞系 CRISPR 必需性，偏向可增殖细胞和必需基因；不等同于患者肿瘤驱动作用。",
        "- CPTAC 蛋白差异说明肿瘤相关表达改变，不等同于致癌因果性；复现突变也受约 100 例队列功效限制。",
        "- MLP/GCN 为 seed42/45/46，Bandit 为 seed42/45/48；三 seed 均值只能描述。严格配对只能使用 seed42/45。",
        "- EvidenceScore 同时参与 Discovery reward，因此只能作为直接排序基线和混杂诊断，绝不能作为 Bandit 的独立有效性终点。",
        "", "## 输出索引", "",
        "- `audit_protocol.json`：结果前冻结的口径。",
        "- `audit_protocol_amendment_01.json`：将 mutation 纳入最强简单基线的逻辑勘误；没有改动任何结果阈值。",
        "- `frozen_candidates/freeze_manifest.json`：三档候选、源 ranking/config/checkpoint 哈希。",
        "- `score_feature_spearman.csv`：分数与突变、PPI/GRN degree、多组学相关。",
        "- `top150_reordering_audit.csv`：Bandit 与简单排序的重合/重排诊断。",
        "- `independent_evidence_metrics_per_run.csv` / `independent_evidence_summary.csv`：命中率与富集。",
        "- `paired_seed42_45_comparison.csv`：Bandit 与 MLP/GCN 的同 seed 严格配对。",
        "- `topk_stability.csv`：跨 seed Jaccard。",
        "- `independent_evidence_gene_table.csv`：低频候选的外部证据明细。",
    ])
    (OUT / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return retain, decisions


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cptac-cache", type=Path, default=None)
    parser.add_argument("--reuse-derived-external", action="store_true")
    args = parser.parse_args()
    OUT = args.output.resolve(); OUT.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("test_labels_read") or freeze.get("external_evidence_read_by_freeze"):
        raise RuntimeError("Candidate freeze integrity flags failed")
    base, pool = load_base_table()
    universe = set(base.Gene)
    derived_path = OUT / "independent_evidence_gene_table.csv"
    if args.reuse_derived_external and derived_path.exists():
        external_pool = pd.read_csv(derived_path)
        external_pool["Gene"] = external_pool["Gene"].map(clean_gene)
        external = external_pool.copy()
        previous_manifest = OUT / "audit_manifest.json"
        source_records = json.loads(previous_manifest.read_text(encoding="utf-8")).get("external_sources", []) if previous_manifest.exists() else []
        for record in source_records:
            record.pop("cache_path", None)
        print(f"[reuse] independent evidence: {derived_path}", flush=True)
    else:
        source_records = []
        with tempfile.TemporaryDirectory(prefix="discovery_audit_external_") as tmp_name:
            tmp = Path(tmp_name)
            cptac = derive_cptac(tmp, universe, source_records, args.cptac_cache)
            depmap = derive_depmap(tmp, universe, source_records)

        holdout = pd.read_csv(HOLDOUT, usecols=["Gene", "Evidence_Level", "Evidence_Type_Count", "Holdout_Category"])
        holdout["Gene"] = holdout["Gene"].map(clean_gene)
        external = base[["Gene"]].merge(cptac, on="Gene", how="left").merge(depmap, on="Gene", how="left")
        external = external.merge(holdout.assign(FrozenLiteratureHoldout=True), on="Gene", how="left")
        bool_cols = ["CPTACRecurrence", "CPTACProteinSupport", "DepMapCCRCCDependency", "DepMapCCRCCSelective", "FrozenLiteratureHoldout"]
        for col in bool_cols:
            external[col] = external[col].fillna(False).astype(bool)
        classes = external[["FrozenLiteratureHoldout", "CPTACRecurrence", "CPTACProteinSupport", "DepMapCCRCCSelective"]].sum(axis=1)
        external["IndependentEvidenceClassCount"] = classes.astype(int)
        external["IndependentAnyHit"] = classes >= 1
        external["IndependentStrongHit"] = classes >= 2
        external["LowFrequencyNovel"] = external.Gene.isin(pool)
        external_pool = external[external.LowFrequencyNovel].copy()
        external_pool.to_csv(derived_path, index=False)
    for col in ["CPTACRecurrence", "CPTACProteinSupport", "DepMapCCRCCDependency", "DepMapCCRCCSelective", "FrozenLiteratureHoldout", "IndependentAnyHit", "IndependentStrongHit", "LowFrequencyNovel"]:
        external[col] = external[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)
    nonmutation_classes = external[["FrozenLiteratureHoldout", "CPTACProteinSupport", "DepMapCCRCCSelective"]].sum(axis=1)
    external["PostHocNonMutationAnyHit"] = nonmutation_classes >= 1
    external["PostHocNonMutationStrongHit"] = nonmutation_classes >= 2
    external_pool = external[external.Gene.isin(pool)].copy()
    external_pool.to_csv(derived_path, index=False)

    rankings = load_rankings(base, freeze)
    corr = correlations(base, pool, rankings)
    corr.to_csv(OUT / "score_feature_spearman.csv", index=False)
    reorder = reordering_audit(base, rankings)
    reorder.to_csv(OUT / "top150_reordering_audit.csv", index=False)
    metrics, selected, stability = evaluate_rankings(rankings, external, pool, protocol["top_k_values"])
    summary = summarize(metrics)
    paired = paired_supervised_comparison(metrics)
    metrics.to_csv(OUT / "independent_evidence_metrics_per_run.csv", index=False)
    summary.to_csv(OUT / "independent_evidence_summary.csv", index=False)
    stability.to_csv(OUT / "topk_stability.csv", index=False)
    paired.to_csv(OUT / "paired_seed42_45_comparison.csv", index=False)
    selected[(selected.K == 150)].to_csv(OUT / "method_top150_candidate_evidence.csv", index=False)
    make_plots(corr, summary, reorder, stability)
    retain, decisions = write_report(base, pool, external_pool, metrics, summary, reorder, stability, paired)

    manifest = {
        "audit_version": "discovery_validity_audit_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_amendment_sha256": sha256(PROTOCOL_AMENDMENT),
        "freeze_manifest_sha256": sha256(FREEZE),
        "local_inputs": {str(p): sha256(p) for p in [EVIDENCE, HOLDOUT, TRAIN_LABEL, VAL_LABEL, MULTIOMICS]},
        "external_sources": source_records,
        "outputs": {p.name: sha256(p) for p in sorted(OUT.glob("*.csv"))},
        "decision_retain_bandit": retain,
        "decision_details": [{"method": m, "mean": mean, "sd": sd, "passes": passed} for m, mean, sd, passed in decisions],
        "training_performed": False, "checkpoint_selection_changed": False, "reward_changed": False,
        "model_changed": False, "new_seed_added": False, "test_labels_read": False,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE", "output": str(OUT), "lowfreq_pool": len(pool),
        "retain_bandit": retain, "test_labels_read": False, "training_performed": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
