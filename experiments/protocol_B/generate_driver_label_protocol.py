import csv
import hashlib
import json
import random
from pathlib import Path


BASE = Path("/mnt/e/codex_file/driver_label_protocol")
REPO = Path("/mnt/e/Projects/RL-GenRisk-main")
DATA = REPO / "data"
INVALID = {"", "?", "NA", "N/A", "NAN", "NONE", "NULL"}


def clean_gene(value):
    if value is None:
        return None
    gene = str(value).strip().upper()
    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()
    if gene in INVALID:
        return None
    return gene


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_genes(genes):
    h = hashlib.sha256()
    for gene in genes:
        h.update(gene.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_ppi_gene_list(path):
    genes = []
    seen = set()
    invalid_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            g1 = clean_gene(parts[0])
            g2 = clean_gene(parts[1])
            if g1 is None or g2 is None:
                invalid_rows += 1
                continue
            for gene in (g1, g2):
                if gene not in seen:
                    seen.add(gene)
                    genes.append(gene)
    return genes, invalid_rows


def read_label_file(path):
    raw_values = []
    raw_lines = 0
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                raw_lines += 1
                if not row:
                    raw_values.append("")
                else:
                    raw_values.append(row[0])
    else:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw_lines += 1
                parts = line.split()
                raw_values.append(parts[0] if parts else "")

    cleaned = []
    invalid_count = 0
    seen = set()
    duplicate_count = 0
    for idx, value in enumerate(raw_values):
        gene = clean_gene(value)
        if idx == 0 and gene in {"GENE", "GENE SYMBOL", "GENE_SYMBOL", "SYMBOL"}:
            continue
        if gene is None:
            invalid_count += 1
            continue
        if gene in seen:
            duplicate_count += 1
            continue
        seen.add(gene)
        cleaned.append(gene)
    return {
        "raw_lines": raw_lines,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "genes": cleaned,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gene_split(path, genes, split, source):
    rows = [{"Gene": gene, "Split": split, "Source": source, "In_PPI": True} for gene in genes]
    write_csv(path, rows, ["Gene", "Split", "Source", "In_PPI"])


def overlap_check(train, val, test, extra=None):
    sets = {"train": set(train), "validation": set(val), "test": set(test)}
    if extra:
        sets.update({k: set(v) for k, v in extra.items()})
    lines = []
    ok = True
    base_names = ["train", "validation", "test"]
    for i, a in enumerate(base_names):
        for b in base_names[i + 1:]:
            inter = sets[a] & sets[b]
            lines.append(f"{a} ∩ {b}: {len(inter)}")
            if inter:
                ok = False
                lines.append(f"  examples: {sorted(inter)[:20]}")
    for name, genes in sets.items():
        dup_count = len(list(genes)) - len(set(genes))
        invalid = [g for g in genes if clean_gene(g) is None]
        lines.append(f"{name}: count={len(genes)}, duplicate_count={dup_count}, invalid_count={len(invalid)}")
    lines.append(f"strict_train_validation_test_disjoint: {ok}")
    return ok, "\n".join(lines) + "\n"


def save_protocol(name, train, val, test, sources, sensitivity=None, notes=None):
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    write_gene_split(out / "train_driver_genes.csv", train, "train", sources["train"])
    write_gene_split(out / "validation_driver_genes.csv", val, "validation", sources["validation"])
    write_gene_split(out / "test_driver_genes.csv", test, "test", sources["test"])
    extra = {}
    if sensitivity is not None:
        write_gene_split(
            out / "sensitivity_shared_external.csv",
            sensitivity,
            "sensitivity_shared_external",
            sources.get("sensitivity", "IntOGen∩NCG not in Train"),
        )
        extra["sensitivity_shared_external"] = sensitivity
    ok, check_text = overlap_check(train, val, test, extra=extra)
    (out / "split_overlap_check.txt").write_text(check_text, encoding="utf-8")

    file_hashes = {}
    for file_name in ["train_driver_genes.csv", "validation_driver_genes.csv", "test_driver_genes.csv"]:
        file_hashes[file_name] = sha256_file(out / file_name)
    if sensitivity is not None:
        file_hashes["sensitivity_shared_external.csv"] = sha256_file(out / "sensitivity_shared_external.csv")

    summary = {
        "protocol": name,
        "train_count": len(train),
        "validation_count": len(val),
        "test_count": len(test),
        "strict_train_validation_test_disjoint": ok,
        "train_gene_sha256": sha256_genes(train),
        "validation_gene_sha256": sha256_genes(val),
        "test_gene_sha256": sha256_genes(test),
        "file_sha256": file_hashes,
        "sources": sources,
        "notes": notes or [],
    }
    if sensitivity is not None:
        summary["sensitivity_shared_external_count"] = len(sensitivity)
        summary["sensitivity_gene_sha256"] = sha256_genes(sensitivity)
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    ppi_gene_list, invalid_ppi_rows = read_ppi_gene_list(DATA / "HPRD.txt")
    ppi_set = set(ppi_gene_list)
    if len(ppi_gene_list) != 9039:
        raise RuntimeError(f"Expected 9039 PPI genes, got {len(ppi_gene_list)}")

    label_specs = {
        "KnownDriver": DATA / "ccRCC_known_driver_genes.txt",
        "IntOGen": DATA / "sta_ccRCC_IntOGen.txt",
        "NCG": DATA / "sta_ccRCC_NCG.txt",
        "Merged": DATA / "sta_ccRCC_Merged.txt",
        "GeneID": DATA / "GeneID.csv",
    }
    label_data = {}
    inventory_rows = []
    for source, path in label_specs.items():
        data = read_label_file(path)
        genes = data["genes"]
        ppi_genes = sorted(set(genes) & ppi_set)
        not_ppi = sorted(set(genes) - ppi_set)
        label_data[source] = {
            **data,
            "set": set(genes),
            "ppi_set": set(ppi_genes),
            "ppi_genes": ppi_genes,
            "not_ppi": not_ppi,
        }
        inventory_rows.append(
            {
                "Source": source,
                "File_path": str(path),
                "Raw_lines": data["raw_lines"],
                "Clean_unique_genes": len(genes),
                "Invalid_gene_count": data["invalid_count"],
                "Duplicate_gene_count": data["duplicate_count"],
                "PPI_intersection_count": len(ppi_genes),
                "Not_in_PPI_count": len(not_ppi),
                "Participates_current_reward": source == "KnownDriver",
                "Participates_current_identify_or_utils_draw_evaluation": source in {"KnownDriver", "IntOGen", "NCG", "Merged"},
                "Source_confirmed_from_README_or_comments": False,
                "Source_note": (
                    "Default reward label and identify.py default label path"
                    if source == "KnownDriver"
                    else "Used by utils_draw.py label file"
                    if source in {"IntOGen", "NCG", "Merged"}
                    else "GeneID.csv content/source not confirmed as high-confidence gold standard; excluded from splits"
                ),
            }
        )
    write_csv(
        BASE / "label_inventory.csv",
        inventory_rows,
        [
            "Source",
            "File_path",
            "Raw_lines",
            "Clean_unique_genes",
            "Invalid_gene_count",
            "Duplicate_gene_count",
            "PPI_intersection_count",
            "Not_in_PPI_count",
            "Participates_current_reward",
            "Participates_current_identify_or_utils_draw_evaluation",
            "Source_confirmed_from_README_or_comments",
            "Source_note",
        ],
    )

    sources = ["KnownDriver", "IntOGen", "NCG", "Merged"]
    pair_rows = []
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            A = label_data[a]["ppi_set"]
            B = label_data[b]["ppi_set"]
            inter = A & B
            union = A | B
            pair_rows.append(
                {
                    "Source_A": a,
                    "Source_B": b,
                    "Size_A": len(A),
                    "Size_B": len(B),
                    "Intersection": len(inter),
                    "A_only": len(A - B),
                    "B_only": len(B - A),
                    "Jaccard": len(inter) / len(union) if union else 0.0,
                }
            )
    write_csv(
        BASE / "label_pairwise_overlap.csv",
        pair_rows,
        ["Source_A", "Source_B", "Size_A", "Size_B", "Intersection", "A_only", "B_only", "Jaccard"],
    )

    all_label_union = set().union(*(label_data[s]["set"] for s in sources))
    membership_genes = [g for g in ppi_gene_list if g in all_label_union]
    matrix_rows = []
    for gene in membership_genes:
        flags = {source: gene in label_data[source]["ppi_set"] for source in sources}
        matrix_rows.append(
            {
                "Gene": gene,
                "In_PPI": True,
                "KnownDriver": flags["KnownDriver"],
                "IntOGen": flags["IntOGen"],
                "NCG": flags["NCG"],
                "Merged": flags["Merged"],
                "Source_count": sum(flags.values()),
            }
        )
    write_csv(
        BASE / "label_membership_matrix.csv",
        matrix_rows,
        ["Gene", "In_PPI", "KnownDriver", "IntOGen", "NCG", "Merged", "Source_count"],
    )

    K = label_data["KnownDriver"]["ppi_set"]
    I = label_data["IntOGen"]["ppi_set"]
    N = label_data["NCG"]["ppi_set"]
    M = label_data["Merged"]["ppi_set"]
    external_union = I | N
    overlap_stats = {
        "KnownDriver∩IntOGen": sorted(K & I),
        "KnownDriver∩NCG": sorted(K & N),
        "IntOGen∩NCG": sorted(I & N),
        "KnownDriver∩IntOGen∩NCG": sorted(K & I & N),
        "IntOGen-only": sorted(I - K - N),
        "NCG-only": sorted(N - K - I),
        "IntOGen-NCG_shared_not_KnownDriver": sorted((I & N) - K),
        "External_union_minus_KnownDriver": sorted(external_union - K),
    }

    # Protocol A.
    train_A = sorted(K)
    val_A = sorted(I - K)
    test_A = sorted(N - K - set(val_A))
    summary_A = save_protocol(
        "protocol_A",
        train_A,
        val_A,
        test_A,
        {
            "train": "KnownDriver∩PPI",
            "validation": "IntOGen∩PPI minus Train",
            "test": "NCG∩PPI minus Train and Validation",
        },
        notes=[
            "NCG genes overlapping IntOGen are assigned to Validation before Test.",
            "Test therefore contains NCG-specific genes after removing KnownDriver and IntOGen overlap.",
        ],
    )

    # Protocol B.
    train_B = sorted(K)
    shared_external = (I & N) - K
    val_B = sorted(I - N - K)
    test_B = sorted(N - I - K)
    sensitivity_B = sorted(shared_external)
    summary_B = save_protocol(
        "protocol_B",
        train_B,
        val_B,
        test_B,
        {
            "train": "KnownDriver∩PPI",
            "validation": "IntOGen-only minus Train",
            "test": "NCG-only minus Train",
            "sensitivity": "IntOGen∩NCG shared external genes not in Train",
        },
        sensitivity=sensitivity_B,
        notes=[
            "Shared external IntOGen/NCG labels are excluded from primary Validation/Test.",
            "Shared external labels are saved as Sensitivity_shared_external.",
        ],
    )

    # Protocol C.
    train_C = sorted(K)
    pool_C = sorted(external_union - K)
    rng = random.Random(42)
    shuffled_C = pool_C[:]
    rng.shuffle(shuffled_C)
    val_count = int(len(shuffled_C) * 0.4)
    val_C = sorted(shuffled_C[:val_count])
    test_C = sorted(shuffled_C[val_count:])
    summary_C = save_protocol(
        "protocol_C",
        train_C,
        val_C,
        test_C,
        {
            "train": "KnownDriver∩PPI",
            "validation": "40% of sorted ExternalPool shuffled with seed=42",
            "test": "60% of sorted ExternalPool shuffled with seed=42",
        },
        notes=[
            "ExternalPool=(IntOGen∪NCG)∩PPI minus Train.",
            "Source independence is weaker than source-separated protocols, but counts are more balanced.",
            f"ExternalPool gene SHA256 before shuffle: {sha256_genes(pool_C)}",
        ],
    )
    summary_C["external_pool_count"] = len(pool_C)
    summary_C["external_pool_gene_sha256"] = sha256_genes(pool_C)
    (BASE / "protocol_C" / "split_summary.json").write_text(
        json.dumps(summary_C, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    protocol_summaries = {"protocol_A": summary_A, "protocol_B": summary_B, "protocol_C": summary_C}
    comparison_rows = []
    for protocol, summary in protocol_summaries.items():
        test_count = summary["test_count"]
        val_count = summary["validation_count"]
        source_independent = protocol in {"protocol_A", "protocol_B"}
        db_overlap = (
            "validation/test source-separated but shared genes assigned away from test"
            if protocol == "protocol_A"
            else "shared IntOGen/NCG genes excluded to sensitivity"
            if protocol == "protocol_B"
            else "validation/test drawn from pooled external labels"
        )
        too_small = test_count < 10
        comparison_rows.append(
            {
                "Protocol": protocol,
                "Train_count": summary["train_count"],
                "Validation_count": val_count,
                "Test_count": test_count,
                "Strictly_disjoint": summary["strict_train_validation_test_disjoint"],
                "Source_independent": source_independent,
                "Database_overlap_handling": db_overlap,
                "Test_too_small_risk": too_small,
                "Suitable_Precision_at_K": test_count > 0,
                "Suitable_Recall_at_K": test_count >= 10,
                "Suitable_NDCG_at_K": test_count >= 10,
                "Paper_explainability": "high" if protocol in {"protocol_A", "protocol_B"} else "medium",
                "Main_bias_risk": (
                    "Test may contain only NCG-specific genes"
                    if protocol == "protocol_A"
                    else "External shared labels removed; validation/test may be small"
                    if protocol == "protocol_B"
                    else "Source independence weaker because IntOGen and NCG are pooled then randomly split"
                ),
            }
        )
    write_csv(
        BASE / "protocol_comparison.csv",
        comparison_rows,
        [
            "Protocol",
            "Train_count",
            "Validation_count",
            "Test_count",
            "Strictly_disjoint",
            "Source_independent",
            "Database_overlap_handling",
            "Test_too_small_risk",
            "Suitable_Precision_at_K",
            "Suitable_Recall_at_K",
            "Suitable_NDCG_at_K",
            "Paper_explainability",
            "Main_bias_risk",
        ],
    )

    # Recommendation by actual counts: prefer B if test large enough; else A; else C.
    if summary_B["test_count"] >= 10:
        recommended = "protocol_B"
        reason = (
            "Protocol B keeps reward labels out of validation/test, makes Train/Validation/Test strictly disjoint, "
            "and removes IntOGen∩NCG shared external genes into a sensitivity set. "
            "The observed Test count is sufficient for Recall@K/NDCG@K interpretation."
        )
    elif summary_A["test_count"] >= 10:
        recommended = "protocol_A"
        reason = (
            "Protocol A preserves source separation and has an interpretable Test count, but assigns IntOGen∩NCG "
            "shared labels to Validation, leaving Test as NCG-specific after exclusions."
        )
    else:
        recommended = "protocol_C"
        reason = (
            "Source-separated tests are too small, so Protocol C provides more balanced validation/test counts at "
            "the cost of weaker source independence."
        )

    rec_summary = protocol_summaries[recommended]
    recommended_text = f"""Recommended protocol: {recommended}

Reason:
{reason}

Recommended Train labels:
- Source: {rec_summary['sources']['train']}
- Count: {rec_summary['train_count']}

Recommended Validation labels:
- Source: {rec_summary['sources']['validation']}
- Count: {rec_summary['validation_count']}

Recommended Test labels:
- Source: {rec_summary['sources']['test']}
- Count: {rec_summary['test_count']}

Leakage control:
- Train/Validation/Test overlap check is strict and zero for the recommended protocol.
- Test labels do not include ccRCC_known_driver_genes used by current reward.
- GeneID.csv is excluded because the project does not confirm it as a high-confidence independent cancer gene gold standard.
- Unlabeled PPI genes are not treated as confirmed negatives.

Sensitivity set:
- Protocol B provides sensitivity_shared_external.csv for IntOGen∩NCG shared external labels not in Train.
- Use this only as a secondary robustness/sensitivity analysis, not as the primary Test set.

Task framing:
- Current full KIRC multiomics features are acceptable for cohort-level gene prioritization because they summarize the KIRC cohort at gene level.
- Do not claim patient-level generalization: patient split is file-order, and full-cohort features/patient coverage can include held-out patient information.

Metrics:
- Prioritize Precision@K, Recall@K, NDCG@K, Mean Rank, Median Rank for K=20,50,100,150.
- AUROC/AUPRC are not primary because there is no verified negative label set and unlabeled genes are not true negatives.

Experiment protocol:
- Use at least 5 random seeds.
- Use validation only for model/episode selection.
- Use test once after final model selection.
- Original and Multiomics must use identical label splits, seeds, PPI node universe, and ranking evaluator.
"""
    (BASE / "recommended_protocol.txt").write_text(recommended_text, encoding="utf-8")

    identify_changes = """identify.py required changes for a future task; not applied here.

1. Align default multiomics feature path with train.py or require explicit --multiomics-feature-path.
2. Align standardize_multiomics with training config or require explicit run metadata.
3. Save model configuration metadata during training: feature_source, feature path, standardization, gene_name hash, selection_budget, label split hashes, seed.
4. Make identify.py read metadata and refuse to rank if the current feature configuration/gene_name hash mismatches training.
5. Keep Ranking output at all 9039 PPI nodes; do not truncate to selection_budget.
6. Separate reward label path from evaluation label paths in downstream scripts.
7. Avoid manual invocation with inconsistent feature_source or standardization settings.
"""
    (BASE / "identify_required_changes.txt").write_text(identify_changes, encoding="utf-8")

    stats_summary = {
        "ppi_gene_count": len(ppi_gene_list),
        "invalid_ppi_rows": invalid_ppi_rows,
        "label_counts": {
            s: {
                "clean_unique": len(label_data[s]["genes"]),
                "ppi_intersection": len(label_data[s]["ppi_genes"]),
                "not_in_ppi": len(label_data[s]["not_ppi"]),
            }
            for s in label_specs
        },
        "overlap_stats": {k: {"count": len(v), "genes": v} for k, v in overlap_stats.items()},
        "protocols": protocol_summaries,
        "recommended_protocol": recommended,
        "recommendation_reason": reason,
        "GeneID_excluded": True,
        "GeneID_exclusion_reason": "Project does not confirm GeneID.csv as a high-confidence independent cancer gene gold standard.",
    }
    (BASE / "protocol_summary.json").write_text(json.dumps(stats_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log_lines = [
        "Driver label protocol generation complete.",
        f"PPI genes: {len(ppi_gene_list)}",
        f"KnownDriver∩PPI: {len(K)}",
        f"IntOGen∩PPI: {len(I)}",
        f"NCG∩PPI: {len(N)}",
        f"Merged∩PPI: {len(M)}",
        f"Recommended: {recommended}",
    ]
    (BASE / "protocol_generation_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
