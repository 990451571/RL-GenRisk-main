#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


INVALID = {"", "?", "NA", "N/A", "NAN", "NONE", "NULL"}
DEFAULT_PPI = Path("/mnt/e/Projects/RL-GenRisk-main/data/HPRD.txt")


def clean_gene(value):
    if value is None:
        return None
    gene = str(value).strip().upper()
    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()
    if gene in INVALID:
        return None
    return gene


def read_ppi_genes(path=DEFAULT_PPI):
    genes = []
    seen = set()
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            for value in parts[:2]:
                gene = clean_gene(value)
                if gene is not None and gene not in seen:
                    seen.add(gene)
                    genes.append(gene)
    return genes, set(genes)


def sniff_delimiter(path):
    sample = Path(path).read_text(encoding="utf-8", errors="replace")[:4096]
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return None


def normalize_header(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def read_table(path):
    path = Path(path)
    delim = sniff_delimiter(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty file: {path}")
    if delim is None:
        rows = [[line.split()[0]] for line in lines if line.split()]
    else:
        rows = list(csv.reader(lines, delimiter=delim))
    return rows


def read_labels(path):
    rows = read_table(path)
    genes = []
    for idx, row in enumerate(rows):
        if not row:
            continue
        gene = clean_gene(row[0])
        if idx == 0 and gene in {"GENE", "GENE_SYMBOL", "GENE SYMBOL"}:
            continue
        if gene is not None:
            genes.append(gene)
    return set(dict.fromkeys(genes))


def read_ranking(path, ppi_set):
    rows = read_table(path)
    if not rows:
        raise ValueError("Ranking is empty.")

    first = [normalize_header(x) for x in rows[0]]
    known_headers = {"gene", "rank", "q_value", "qvalue", "score"}
    has_header = any(x in known_headers for x in first)
    data_rows = rows[1:] if has_header else rows

    if has_header:
        header = first
        gene_idx = None
        for candidate in ("gene", "gene_symbol", "symbol"):
            if candidate in header:
                gene_idx = header.index(candidate)
                break
        if gene_idx is None:
            gene_idx = 0
        rank_idx = header.index("rank") if "rank" in header else None
        q_idx = None
        for candidate in ("q_value", "qvalue", "score"):
            if candidate in header:
                q_idx = header.index(candidate)
                break
    else:
        gene_idx = 0
        rank_idx = None
        q_idx = 1 if rows and len(rows[0]) > 1 else None

    parsed = []
    for order, row in enumerate(data_rows):
        if gene_idx >= len(row):
            continue
        gene = clean_gene(row[gene_idx])
        if gene is None or gene not in ppi_set:
            continue
        rank = None
        q_value = None
        if rank_idx is not None and rank_idx < len(row):
            try:
                rank = float(row[rank_idx])
            except ValueError:
                rank = None
        if q_idx is not None and q_idx < len(row):
            try:
                q_value = float(row[q_idx])
            except ValueError:
                q_value = None
        parsed.append({"Gene": gene, "Rank": rank, "Q_value": q_value, "Input_order": order})

    if not parsed:
        raise ValueError("No valid PPI genes found in ranking.")

    if any(item["Rank"] is not None for item in parsed):
        parsed.sort(key=lambda x: (float("inf") if x["Rank"] is None else x["Rank"], x["Input_order"]))
        sort_mode = "Rank_ascending"
    elif any(item["Q_value"] is not None for item in parsed):
        parsed.sort(key=lambda x: (float("-inf") if x["Q_value"] is None else x["Q_value"], -x["Input_order"]), reverse=True)
        sort_mode = "Q_value_descending"
    else:
        parsed.sort(key=lambda x: x["Input_order"])
        sort_mode = "file_order"

    deduped = []
    seen = set()
    duplicate_count = 0
    for item in parsed:
        if item["Gene"] in seen:
            duplicate_count += 1
            continue
        seen.add(item["Gene"])
        item = dict(item)
        item["Final_rank"] = len(deduped) + 1
        deduped.append(item)

    return deduped, {"sort_mode": sort_mode, "duplicate_genes_removed": duplicate_count}


def dcg(relevances):
    return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(relevances))


def metric_for_labels(ranking, labels, k_values, label_name):
    if not labels:
        raise ValueError(f"{label_name} label set is empty.")
    rank_by_gene = {item["Gene"]: item["Final_rank"] for item in ranking}
    present_ranks = sorted(rank_by_gene[g] for g in labels if g in rank_by_gene)
    missing = sorted(labels - set(rank_by_gene))
    rows = []
    for k in k_values:
        top = ranking[: min(k, len(ranking))]
        top_genes = {item["Gene"] for item in top}
        hits = len(top_genes & labels)
        rel = [1 if item["Gene"] in labels else 0 for item in top]
        ideal_hits = min(k, len(labels), len(ranking))
        ideal = [1] * ideal_hits + [0] * (len(top) - ideal_hits)
        idcg = dcg(ideal)
        rows.append(
            {
                "Label_set": label_name,
                "K": k,
                "Effective_K": len(top),
                "HitCount": hits,
                "Precision": hits / k,
                "Recall": hits / len(labels),
                "NDCG": dcg(rel) / idcg if idcg > 0 else 0.0,
            }
        )
    ranks_for_stats = present_ranks
    summary = {
        "Label_set": label_name,
        "Label_count": len(labels),
        "Labels_in_ranking": len(ranks_for_stats),
        "Labels_missing_from_ranking": len(missing),
        "Mean_rank": statistics.mean(ranks_for_stats) if ranks_for_stats else None,
        "Median_rank": statistics.median(ranks_for_stats) if ranks_for_stats else None,
        "Min_rank": min(ranks_for_stats) if ranks_for_stats else None,
        "Max_rank": max(ranks_for_stats) if ranks_for_stats else None,
        "MRR": statistics.mean([1.0 / r for r in ranks_for_stats]) if ranks_for_stats else 0.0,
    }
    return rows, summary, missing


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_k_values(value):
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a full gene Ranking against held-out driver labels.")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--test-labels")
    parser.add_argument("--validation-labels")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--train-labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default="20,50,100,150")
    parser.add_argument("--ppi-path", default=str(DEFAULT_PPI))
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k_values = parse_k_values(args.k_values)
    _, ppi_set = read_ppi_genes(args.ppi_path)
    ranking, ranking_meta = read_ranking(args.ranking, ppi_set)

    if not args.test_labels and not args.validation_only:
        raise ValueError("--test-labels is required unless --validation-only is set.")
    test_labels = (read_labels(args.test_labels) & ppi_set) if args.test_labels else set()
    validation_labels = (read_labels(args.validation_labels) & ppi_set) if args.validation_labels else set()
    label_for_default_train = args.test_labels or args.validation_labels
    train_label_path = Path(args.train_labels) if args.train_labels else Path(label_for_default_train).parent / "train_driver_genes.csv"
    train_labels = (read_labels(train_label_path) & ppi_set) if train_label_path.exists() else set()
    if args.validation_only and not validation_labels:
        raise ValueError("Validation label set is empty after cleaning/PPI filtering.")
    if not args.validation_only and not test_labels:
        raise ValueError("Test label set is empty after cleaning/PPI filtering.")
    overlap = train_labels & test_labels
    if overlap:
        raise ValueError(f"Train/Test label overlap detected: {len(overlap)} genes; examples={sorted(overlap)[:20]}")
    validation_overlap = train_labels & validation_labels
    if validation_overlap:
        raise ValueError(
            f"Train/Validation label overlap detected: {len(validation_overlap)} genes; "
            f"examples={sorted(validation_overlap)[:20]}"
        )

    metric_rows = []
    summaries = []
    missing_by_set = {}
    if not args.validation_only:
        test_rows, test_summary, test_missing = metric_for_labels(ranking, test_labels, k_values, "test")
        metric_rows.extend(test_rows)
        summaries.append(test_summary)
        missing_by_set["test"] = test_missing
    if validation_labels:
        validation_rows, validation_summary, validation_missing = metric_for_labels(
            ranking, validation_labels, k_values, "validation"
        )
        metric_rows.extend(validation_rows)
        summaries.append(validation_summary)
        missing_by_set["validation"] = validation_missing

    write_csv(
        out / "ranking_metrics_by_k.csv",
        metric_rows,
        ["Label_set", "K", "Effective_K", "HitCount", "Precision", "Recall", "NDCG"],
    )
    write_csv(
        out / "ranking_label_rank_summary.csv",
        summaries,
        [
            "Label_set",
            "Label_count",
            "Labels_in_ranking",
            "Labels_missing_from_ranking",
            "Mean_rank",
            "Median_rank",
            "Min_rank",
            "Max_rank",
            "MRR",
        ],
    )
    metadata = {
        "ranking_gene_count": len(ranking),
        "ranking_ppi_intersection_count": len(ranking),
        "ppi_gene_count": len(ppi_set),
        "sort_mode": ranking_meta["sort_mode"],
        "duplicate_genes_removed": ranking_meta["duplicate_genes_removed"],
        "k_values": k_values,
        "train_label_count": len(train_labels),
        "train_label_path_used_for_overlap_check": str(train_label_path) if train_label_path.exists() else None,
        "validation_label_count": len(validation_labels),
        "test_label_count": len(test_labels),
        "missing_labels": missing_by_set,
        "notes": [
            "AUROC/AUPRC are intentionally not implemented as primary metrics.",
            "Unlabeled genes are not treated as confirmed negative genes.",
        ],
    }
    (out / "ranking_evaluation_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    main()
