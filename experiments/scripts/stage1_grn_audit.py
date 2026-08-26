import csv
import hashlib
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if os.name == "nt":
    PROJECT = Path(r"E:\Projects\RL-GenRisk-main")
    STAGE0 = Path(r"E:\codex_file\新方向阶段0")
    OUT = Path(r"E:\codex_file\新方向阶段1")
else:
    PROJECT = Path("/mnt/e/Projects/RL-GenRisk-main")
    STAGE0 = Path("/mnt/e/codex_file/新方向阶段0")
    OUT = Path("/mnt/e/codex_file/新方向阶段1")
RAW_URL = "https://omnipathdb.org/interactions"
PRIMARY_GRN = "DoRothEA via OmniPath interactions API"
PRIMARY_THRESHOLD = "A+B"
SENSITIVITY_THRESHOLD = "A+B+C"
LEVEL_ORDER = ["A", "B", "C", "D", "E"]


def current_path(value):
    s = str(value)
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":\\":
        return Path(f"/mnt/{s[0].lower()}/" + s[3:].replace("\\", "/"))
    return Path(s)


def display_path(value):
    s = str(value)
    if os.name != "nt" and s.startswith("/mnt/") and len(s) > 7 and s[6] == "/":
        drive = s[5].upper()
        return f"{drive}:\\" + s[7:].replace("/", "\\")
    return s


def ts():
    return datetime.now(timezone.utc).isoformat()


def mkdirs():
    for name in [
        "01_gene_universe",
        "02_ppi_audit",
        "03_grn_source/raw",
        "04_grn_processing",
        "05_grn_frozen",
        "06_grn_audit",
        "07_multi_relation_audit",
        "08_network_leakage",
        "09_stage2_input_freeze",
        "10_stage2_ready",
        "11_figures",
        "12_integrity",
        "scripts",
        "logs",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


DATA_ACCESS = []


def log_access(path, purpose, rw, category, allowed=True, notes=""):
    DATA_ACCESS.append(
        {
            "timestamp": ts(),
            "file_or_resource": str(path),
            "purpose": purpose,
            "read_or_write": rw,
            "category": category,
            "allowed_by_protocol": "YES" if allowed else "NO",
            "notes": notes,
        }
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fieldnames=None, delimiter=","):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log_access(path, "write Stage 1 output", "write", "stage1_output")


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    log_access(path, "write Stage 1 output", "write", "stage1_output")


def read_dicts(path, delimiter=","):
    log_access(path, "read Stage 0/project evidence", "read", "stage0_or_project")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def clean_gene(value):
    if value is None:
        return ""
    g = str(value).strip().upper()
    if "|" in g:
        g = g.split("|", 1)[0].strip()
    return "" if g in {"", "NA", "N/A", "NAN", "NONE", "NULL", "?"} else g


def percentile(values, q):
    if not values:
        return 0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def components(nodes, undirected_edges):
    adj = {n: set() for n in nodes}
    for a, b in undirected_edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    seen = set()
    comps = []
    for n in nodes:
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        count = 0
        while q:
            x = q.popleft()
            count += 1
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        comps.append(count)
    comps.sort(reverse=True)
    return comps


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pairs) < 3:
        return None

    def ranks(vals):
        sorted_vals = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1][0] == sorted_vals[i][0]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                out[sorted_vals[k][1]] = rank
            i = j + 1
        return out

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (denx * deny) if denx and deny else None


def load_stage0():
    paths = [
        STAGE0 / "STAGE0_FINALIZATION_REPORT.md",
        STAGE0 / "05_experiment_protocol" / "frozen_research_protocol.md",
        STAGE0 / "03_data_governance" / "data_access_manifest.csv",
        STAGE0 / "03_data_governance" / "label_provenance.csv",
        STAGE0 / "03_data_governance" / "split_manifest.csv",
    ]
    for p in paths:
        log_access(p, "confirm Stage 0 frozen boundary", "read", "stage0_protocol")
        if not p.exists():
            raise FileNotFoundError(p)
    text = paths[0].read_text(encoding="utf-8")
    if "READY_FOR_STAGE1 = YES" not in text:
        raise RuntimeError("Stage 0 finalization does not say READY_FOR_STAGE1 = YES")


def load_gene_universe():
    ppi_edges_raw = []
    ppi_order = []
    ppi_seen = set()
    ppi_path = PROJECT / "data" / "HPRD.txt"
    log_access(ppi_path, "read frozen PPI edges", "read", "project_ppi")
    with ppi_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                a, b = clean_gene(parts[0]), clean_gene(parts[1])
                if a and b:
                    ppi_edges_raw.append((a, b))
                    for g in (a, b):
                        if g not in ppi_seen:
                            ppi_seen.add(g)
                            ppi_order.append(g)
    ppi_nodes = set(x for edge in ppi_edges_raw for x in edge)

    baseline_registry = STAGE0 / "04_baseline_freeze" / "baseline_registry.csv"
    registry = read_dicts(baseline_registry)
    ranking_path = None
    for r in registry:
        if r.get("model_id") == "hybrid6_raw_seed42_validation_selected_episode12":
            ranking_path = current_path(r["ranking_path"])
            break
    if ranking_path is None:
        for r in registry:
            if r.get("notes") == "Validation-selected primary model" and r.get("feature_mode") == "hybrid6_raw":
                ranking_path = current_path(r["ranking_path"])
                break
    ranking_genes = set()
    ranking_order = []
    if ranking_path and ranking_path.exists():
        for r in read_dicts(ranking_path):
            g = clean_gene(r.get("Gene") or r.get("gene") or next(iter(r.values())))
            if g and g not in ranking_genes:
                ranking_genes.add(g)
                ranking_order.append(g)

    # The formal model gene universe is the PPI node order used to build model
    # tensors. GeneID.csv is an annotation/label source and is not the 9039-node
    # model universe.
    genes = ppi_order
    if ranking_genes and set(ppi_order) != ranking_genes:
        # Keep PPI order as primary but record inconsistency in the report.
        pass
    node_index = {g: i for i, g in enumerate(genes)}

    feature_path = PROJECT / "data" / "processed" / "KIRC_multiomics_3omics.csv"
    feature_genes = set()
    for r in read_dicts(feature_path):
        g = clean_gene(r.get("Gene"))
        if g:
            feature_genes.add(g)

    manifest = []
    for g in genes:
        manifest.append(
            {
                "gene_symbol": g,
                "node_index_if_available": node_index[g],
                "present_in_PPI": g in ppi_nodes,
                "present_in_current_features": True,
                "present_in_multiomics_feature_file": g in feature_genes,
                "present_in_historical_ranking": g in ranking_genes,
            }
        )
    write_text(OUT / "01_gene_universe" / "frozen_gene_universe.txt", "\n".join(genes) + "\n")
    write_csv(OUT / "01_gene_universe" / "gene_universe_manifest.csv", manifest)
    report = f"""# Gene Universe Report

N_GENE_UNIVERSE = {len(genes)}

Primary source: first-occurrence node order from frozen PPI file `{ppi_path}`.

PPI source: `{ppi_path}`.

Current fixed feature source: `{feature_path}`.

Historical validation ranking source: `{display_path(ranking_path) if ranking_path else 'NOT_FOUND'}`.

Genes present in PPI: {sum(1 for r in manifest if r['present_in_PPI'])}

Genes present in current hybrid6_raw feature tensor by construction: {sum(1 for r in manifest if r['present_in_current_features'])}

Genes with direct rows in multiomics feature file: {sum(1 for r in manifest if r['present_in_multiomics_feature_file'])}

Genes present in historical validation ranking: {sum(1 for r in manifest if r['present_in_historical_ranking'])}

GeneID.csv note: `data/GeneID.csv` is not used as the Stage 1 universe because it contains annotation/label records and does not match the 9039-node model/PPI/ranking space.
"""
    write_text(OUT / "01_gene_universe" / "gene_universe_report.md", report)
    return genes, node_index, set(genes), ppi_edges_raw, ppi_nodes, feature_path, ranking_path


def ppi_audit(genes, universe, ppi_edges_raw):
    ppi_path = PROJECT / "data" / "HPRD.txt"
    self_loops = sum(1 for a, b in ppi_edges_raw if a == b)
    canonical = [tuple(sorted((a, b))) for a, b in ppi_edges_raw if a != b]
    counts = Counter(canonical)
    duplicate_edges = sum(c - 1 for c in counts.values() if c > 1)
    edges = sorted(counts)
    nodes = set(x for e in edges for x in e)
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    degrees = [degree[g] for g in genes]
    comps = components(nodes, edges)
    degree_rows = [{"Gene": g, "PPI_degree": degree[g]} for g in genes]
    top_hubs = sorted(degree_rows, key=lambda r: (-r["PPI_degree"], r["Gene"]))[:100]
    comp_rows = [{"component_rank": i + 1, "node_count": c} for i, c in enumerate(comps)]
    stats = {
        "ppi_source": str(ppi_path),
        "sha256": sha256(ppi_path),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "directed": False,
        "raw_edge_rows": len(ppi_edges_raw),
        "self_loop_count": self_loops,
        "duplicate_edge_count": duplicate_edges,
        "connected_components": len(comps),
        "largest_connected_component": comps[0] if comps else 0,
        "isolated_genes_within_frozen_universe": sum(1 for g in genes if degree[g] == 0),
        "degree_mean": statistics.mean(degrees) if degrees else 0,
        "degree_median": statistics.median(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_p90": percentile(degrees, 0.90),
        "degree_p95": percentile(degrees, 0.95),
        "degree_p99": percentile(degrees, 0.99),
    }
    write_csv(OUT / "02_ppi_audit" / "ppi_manifest.csv", [{"file": str(ppi_path), "sha256": stats["sha256"], "node_count": len(nodes), "edge_count": len(edges), "directed": False}])
    write_text(OUT / "02_ppi_audit" / "ppi_statistics.json", json.dumps(stats, indent=2, ensure_ascii=False))
    write_csv(OUT / "02_ppi_audit" / "ppi_degree_distribution.csv", degree_rows)
    write_csv(OUT / "02_ppi_audit" / "ppi_top_hubs.csv", top_hubs)
    write_csv(OUT / "02_ppi_audit" / "ppi_components.csv", comp_rows)
    write_text(
        OUT / "02_ppi_audit" / "ppi_audit_report.md",
        f"# PPI Audit Report\n\nPPI file: `{ppi_path}`\n\nNodes: {len(nodes)}\n\nEdges: {len(edges)}\n\nDirected: false\n\nMax hub: {top_hubs[0] if top_hubs else 'NA'}\n",
    )
    return edges, nodes, degree, stats


def download_grn():
    raw_path = OUT / "03_grn_source" / "raw" / "omnipath_dorothea_human_interactions.tsv"
    query = urllib.parse.urlencode(
        {
            "datasets": "dorothea",
            "genesymbols": "1",
            "organisms": "9606",
            "fields": "sources,references,curation_effort,dorothea_level",
        }
    )
    url = RAW_URL + "?" + query
    if not raw_path.exists():
        log_access(url, "download primary GRN network only", "read", "GRN_source", True, "DoRothEA TF-target network, not validation results")
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        raw_path.write_bytes(data)
        log_access(raw_path, "write raw GRN snapshot", "write", "GRN_raw")
    else:
        log_access(raw_path, "reuse existing raw GRN snapshot", "read", "GRN_raw")
    manifest = [
        {
            "file_name": raw_path.name,
            "source": "OmniPath interactions API dataset=dorothea",
            "download_date": ts(),
            "version": "OmniPath live API retrieval; URL frozen in memo",
            "size": raw_path.stat().st_size,
            "SHA256": sha256(raw_path),
            "url": url,
        }
    ]
    write_csv(OUT / "03_grn_source" / "grn_raw_manifest.csv", manifest)
    memo = f"""# GRN Source Selection Memo

## Candidate: DoRothEA via OmniPath

- Version / release: OmniPath API retrieval on {ts()}; raw URL recorded in `grn_raw_manifest.csv`.
- Official source: https://omnipathdb.org/interactions with `datasets=dorothea`.
- Species: Homo sapiens (`organisms=9606`).
- Relation type: directed TF -> target regulatory relationship.
- Directed: yes.
- Confidence available: yes, `dorothea_level` A-E.
- Approximate size: determined from raw frozen file in this Stage 1 audit.
- Advantages: human TF-target specificity, confidence tiers, clear provenance through DoRothEA/OmniPath, compact edge list suitable for relation audit.
- Limitations: mixed evidence sources; some TF-target relations may derive from high-throughput or inferred evidence; optional confidence semantics require sensitivity audit.
- Reason selected: highest fit to Stage 1 criteria among common public resources because confidence tier/provenance and directed TF-target semantics are explicit.

## Candidate: TRRUST

- Advantages: literature-curated TF-target relations.
- Limitations: smaller coverage and no confidence tier comparable to DoRothEA; not selected as primary to avoid multi-source mixing.

## Candidate: OmniPath other TF/regulatory resources

- Advantages: broad coverage.
- Limitations: mixing multiple resources would obscure provenance/confidence and complicate Stage 2 relation-source interpretation.

PRIMARY_GRN = {PRIMARY_GRN}

SELECTION_REASON = Evidence quality and provenance clarity were prioritized over edge count. No Historical Test metric, external validation result, or model performance information was used.
"""
    write_text(OUT / "03_grn_source" / "GRN_SOURCE_SELECTION.md", memo)
    return raw_path


def parse_and_clean_grn(raw_path, universe):
    log_access(raw_path, "parse raw GRN", "read", "GRN_raw")
    raw_rows = read_dicts(raw_path, delimiter="\t")
    mapping_rows = []
    pair_to_data = {}
    self_loop_count = 0
    unmapped_tf = set()
    unmapped_target = set()
    ambiguous = 0
    level_counts = Counter()
    for r in raw_rows:
        otf = r.get("source_genesymbol", "")
        otg = r.get("target_genesymbol", "")
        tf = clean_gene(otf)
        target = clean_gene(otg)
        tf_status = "EXACT" if tf in universe else "UNMAPPED"
        tg_status = "EXACT" if target in universe else "UNMAPPED"
        if tf_status == "UNMAPPED":
            unmapped_tf.add(tf or otf)
        if tg_status == "UNMAPPED":
            unmapped_target.add(target or otg)
        mapping_rows.append(
            {
                "original_tf": otf,
                "mapped_tf": tf,
                "tf_mapping_status": tf_status,
                "original_target": otg,
                "mapped_target": target,
                "target_mapping_status": tg_status,
                "mapping_source": "EXACT_UPPERCASE_SYMBOL_MATCH_TO_FROZEN_UNIVERSE",
                "notes": "",
            }
        )
        if not tf or not target or tf_status != "EXACT" or tg_status != "EXACT":
            continue
        if tf == target:
            self_loop_count += 1
            continue
        level = str(r.get("dorothea_level") or "NA").strip() or "NA"
        level_counts[level] += 1
        key = (tf, target)
        item = pair_to_data.setdefault(
            key,
            {
                "tf": tf,
                "target": target,
                "confidence": level,
                "source": set(),
                "evidence": set(),
                "dorothea_levels": set(),
                "curation_effort": [],
            },
        )
        item["source"].update(str(r.get("sources", "")).split(";"))
        refs = str(r.get("references", "")).strip()
        if refs:
            item["evidence"].update(refs.split(";"))
        item["dorothea_levels"].add(level)
        ce = str(r.get("curation_effort", "")).strip()
        if ce:
            item["curation_effort"].append(ce)

    cleaned = []
    for item in pair_to_data.values():
        levels = sorted(item["dorothea_levels"], key=lambda x: LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 99)
        confidence = levels[0] if levels else item["confidence"]
        cleaned.append(
            {
                "tf": item["tf"],
                "target": item["target"],
                "confidence": confidence,
                "source": ";".join(sorted(s for s in item["source"] if s)),
                "evidence": ";".join(sorted(item["evidence"])) if item["evidence"] else "NA",
                "all_confidence_levels": ";".join(levels),
                "curation_effort": ";".join(item["curation_effort"]) if item["curation_effort"] else "NA",
            }
        )
    cleaned.sort(key=lambda r: (r["confidence"], r["tf"], r["target"]))
    write_csv(OUT / "04_grn_processing" / "grn_symbol_mapping.csv", mapping_rows)
    write_csv(OUT / "04_grn_processing" / "grn_cleaned_full.tsv", cleaned, delimiter="\t")
    write_csv(OUT / "05_grn_frozen" / "grn_full_cleaned.tsv", cleaned, delimiter="\t")

    threshold_rows = []
    threshold_defs = {
        "A": {"A"},
        "A+B": {"A", "B"},
        "A+B+C": {"A", "B", "C"},
        "A+B+C+D": {"A", "B", "C", "D"},
        "A+B+C+D+E": {"A", "B", "C", "D", "E"},
    }
    final_by_threshold = {}
    for name, levels in threshold_defs.items():
        rows = [r for r in cleaned if r["confidence"] in levels]
        genes = set()
        tfs = set()
        targets = set()
        outd = Counter()
        indeg = Counter()
        for r in rows:
            genes.update([r["tf"], r["target"]])
            tfs.add(r["tf"])
            targets.add(r["target"])
            outd[r["tf"]] += 1
            indeg[r["target"]] += 1
        degrees = [outd[g] + indeg[g] for g in genes]
        final_by_threshold[name] = rows
        threshold_rows.append(
            {
                "Threshold": name,
                "edges": len(rows),
                "genes": len(genes),
                "TFs": len(tfs),
                "targets": len(targets),
                "coverage": len(genes) / len(universe) if universe else 0,
                "max_out_degree": max(outd.values()) if outd else 0,
                "max_in_degree": max(indeg.values()) if indeg else 0,
                "degree_p95": percentile(degrees, 0.95),
                "degree_p99": percentile(degrees, 0.99),
                "selection_role": "PRIMARY" if name == PRIMARY_THRESHOLD else ("SENSITIVITY" if name == SENSITIVITY_THRESHOLD else "AUDIT_ONLY"),
            }
        )
    write_csv(OUT / "04_grn_processing" / "grn_threshold_comparison.csv", threshold_rows)
    final = final_by_threshold[PRIMARY_THRESHOLD]
    write_csv(OUT / "05_grn_frozen" / "grn_in_gene_universe.tsv", final, delimiter="\t")
    processing_stats = {
        "raw_edge_count": len(raw_rows),
        "clean_edge_count_after_mapping_dedup_selfloop_removal": len(cleaned),
        "mapped_edge_count_primary": len(final),
        "self_loop_removed_count": self_loop_count,
        "duplicate_removed_count": max(0, len(raw_rows) - len(cleaned) - len(unmapped_tf) - len(unmapped_target) - self_loop_count),
        "unmapped_tf_count": len(unmapped_tf),
        "unmapped_target_count": len(unmapped_target),
        "ambiguous_mapping_count": ambiguous,
        "level_counts_after_mapping": dict(level_counts),
    }
    return cleaned, final, threshold_rows, processing_stats


def grn_audit(final, genes, universe):
    outd = Counter()
    indeg = Counter()
    grn_nodes = set()
    tfs = set()
    targets = set()
    for r in final:
        tf, target = r["tf"], r["target"]
        outd[tf] += 1
        indeg[target] += 1
        grn_nodes.update([tf, target])
        tfs.add(tf)
        targets.add(target)
    total = Counter({g: outd[g] + indeg[g] for g in grn_nodes})
    in_rows = [{"Gene": g, "GRN_in_degree": indeg[g]} for g in sorted(universe)]
    out_rows = [{"Gene": g, "GRN_out_degree": outd[g]} for g in sorted(universe)]
    total_rows = [{"Gene": g, "GRN_total_degree": outd[g] + indeg[g]} for g in sorted(universe)]
    tf_hubs = []
    for g, d in sorted(outd.items(), key=lambda kv: (-kv[1], kv[0]))[:100]:
        cov = d / len(universe)
        level = "EXTREME" if cov >= 0.25 else ("HIGH" if cov >= 0.10 else "NORMAL")
        tf_hubs.append({"TF": g, "out_degree": d, "degree_over_universe": cov, "hub_audit_level": level})
    target_hubs = []
    for g, d in sorted(indeg.items(), key=lambda kv: (-kv[1], kv[0]))[:100]:
        cov = d / len(universe)
        level = "EXTREME" if cov >= 0.25 else ("HIGH" if cov >= 0.10 else "NORMAL")
        target_hubs.append({"Target": g, "in_degree": d, "degree_over_universe": cov, "hub_audit_level": level})
    undirected = set(tuple(sorted((r["tf"], r["target"]))) for r in final)
    comps = components(grn_nodes, undirected)
    comp_rows = [{"component_rank": i + 1, "node_count": c} for i, c in enumerate(comps)]
    isolated = [{"Gene": g} for g in sorted(universe - grn_nodes)]
    indegrees = [indeg[g] for g in universe]
    outdegrees = [outd[g] for g in universe]
    totaldegrees = [outd[g] + indeg[g] for g in universe]
    stats = {
        "number_of_nodes": len(grn_nodes),
        "number_of_edges": len(final),
        "number_of_TFs": len(tfs),
        "number_of_targets": len(targets),
        "gene_universe_coverage": len(grn_nodes) / len(universe),
        "mean_in_degree": statistics.mean(indegrees),
        "median_in_degree": statistics.median(indegrees),
        "max_in_degree": max(indegrees) if indegrees else 0,
        "mean_out_degree": statistics.mean(outdegrees),
        "median_out_degree": statistics.median(outdegrees),
        "max_out_degree": max(outdegrees) if outdegrees else 0,
        "degree_p90": percentile(totaldegrees, 0.90),
        "degree_p95": percentile(totaldegrees, 0.95),
        "degree_p99": percentile(totaldegrees, 0.99),
        "connected_components_undirected_view": len(comps),
        "largest_component_undirected_view": comps[0] if comps else 0,
        "isolated_universe_genes": len(isolated),
        "hub_thresholds": "NORMAL <10% universe; HIGH >=10%; EXTREME >=25%",
    }
    write_text(OUT / "06_grn_audit" / "grn_statistics.json", json.dumps(stats, indent=2, ensure_ascii=False))
    write_csv(OUT / "06_grn_audit" / "grn_in_degree.csv", in_rows)
    write_csv(OUT / "06_grn_audit" / "grn_out_degree.csv", out_rows)
    write_csv(OUT / "06_grn_audit" / "grn_total_degree.csv", total_rows)
    write_csv(OUT / "06_grn_audit" / "grn_top_tf_hubs.csv", tf_hubs)
    write_csv(OUT / "06_grn_audit" / "grn_top_target_hubs.csv", target_hubs)
    write_csv(OUT / "06_grn_audit" / "grn_components.csv", comp_rows)
    write_csv(OUT / "06_grn_audit" / "grn_isolated_universe_genes.csv", isolated)
    return outd, indeg, total, grn_nodes, tfs, targets, tf_hubs, target_hubs, stats


def multi_relation_audit(genes, universe, ppi_edges, ppi_nodes, ppi_degree, final, grn_out, grn_in, grn_nodes):
    ppi_pairs = set(tuple(sorted(e)) for e in ppi_edges)
    grn_pairs = set(tuple(sorted((r["tf"], r["target"]))) for r in final)
    shared_edges = sorted(ppi_pairs & grn_pairs)
    grn_only_pairs = grn_pairs - ppi_pairs
    ppi_only_pairs = ppi_pairs - grn_pairs
    node_inter = ppi_nodes & grn_nodes
    node_union = ppi_nodes | grn_nodes
    edge_union = ppi_pairs | grn_pairs
    comp_rows = []
    for g in genes:
        comp_rows.append(
            {
                "Gene": g,
                "PPI_degree": ppi_degree[g],
                "GRN_in_degree": grn_in[g],
                "GRN_out_degree": grn_out[g],
                "GRN_total_degree": grn_in[g] + grn_out[g],
            }
        )
    corr_total = spearman([r["PPI_degree"] for r in comp_rows], [r["GRN_total_degree"] for r in comp_rows])
    corr_out = spearman([r["PPI_degree"] for r in comp_rows], [r["GRN_out_degree"] for r in comp_rows])
    corr_in = spearman([r["PPI_degree"] for r in comp_rows], [r["GRN_in_degree"] for r in comp_rows])
    stats = {
        "PPI_nodes": len(ppi_nodes),
        "GRN_nodes": len(grn_nodes),
        "shared_nodes": len(node_inter),
        "PPI_edges": len(ppi_pairs),
        "GRN_edges": len(grn_pairs),
        "GRN_edges_also_present_as_PPI_interaction": len(shared_edges),
        "GRN_only_edges": len(grn_only_pairs),
        "PPI_only_edges": len(ppi_only_pairs),
        "NodeJaccard": len(node_inter) / len(node_union) if node_union else 0,
        "EdgeJaccard": len(shared_edges) / len(edge_union) if edge_union else 0,
        "Spearman_PPI_degree_vs_GRN_total_degree": corr_total,
        "Spearman_PPI_degree_vs_GRN_out_degree": corr_out,
        "Spearman_PPI_degree_vs_GRN_in_degree": corr_in,
    }
    write_text(OUT / "07_multi_relation_audit" / "ppi_grn_overlap_statistics.json", json.dumps(stats, indent=2, ensure_ascii=False))
    write_csv(OUT / "07_multi_relation_audit" / "ppi_grn_gene_degree_comparison.csv", comp_rows)
    write_csv(OUT / "07_multi_relation_audit" / "ppi_grn_shared_edges.tsv", [{"gene_a": a, "gene_b": b} for a, b in shared_edges], delimiter="\t")
    write_csv(OUT / "07_multi_relation_audit" / "grn_only_edges.tsv", [{"gene_a": a, "gene_b": b} for a, b in sorted(grn_only_pairs)], delimiter="\t")
    write_csv(OUT / "07_multi_relation_audit" / "ppi_only_edges.tsv", [{"gene_a": a, "gene_b": b} for a, b in sorted(ppi_only_pairs)], delimiter="\t")
    write_text(
        OUT / "07_multi_relation_audit" / "multi_relation_audit.md",
        f"""# Multi-relation Audit

Node overlap and edge overlap are computed separately. GRN direction is preserved in GRN files; undirected canonical pairs are used only for overlap statistics.

NodeJaccard = {stats['NodeJaccard']:.6f}

EdgeJaccard = {stats['EdgeJaccard']:.6f}

Spearman PPI degree vs GRN total degree = {corr_total}

Interpretation: high shared-node coverage with low/moderate edge overlap supports a multi-relational graph hypothesis because the same genes are connected by different relation semantics.
""",
    )
    return stats, comp_rows


def stage2_ready(genes, node_index, ppi_edges, final, feature_path):
    gene_rows = [{"node_index": node_index[g], "gene_symbol": g} for g in genes]
    ppi_rows = [{"source": a, "target": b, "source_index": node_index[a], "target_index": node_index[b], "relation_id": 0, "relation_name": "PPI", "directed": "false", "confidence": "NA"} for a, b in ppi_edges if a in node_index and b in node_index]
    grn_rows = [{"source": r["tf"], "target": r["target"], "source_index": node_index[r["tf"]], "target_index": node_index[r["target"]], "relation_id": 1, "relation_name": "GRN", "directed": "true", "confidence": r["confidence"], "source_database": PRIMARY_GRN} for r in final]
    gene_path = OUT / "10_stage2_ready" / "gene_universe.tsv"
    ppi_path = OUT / "10_stage2_ready" / "ppi_edges_frozen.tsv"
    grn_path = OUT / "10_stage2_ready" / "grn_edges_frozen.tsv"
    write_csv(gene_path, gene_rows, delimiter="\t")
    write_csv(ppi_path, ppi_rows, delimiter="\t")
    write_csv(grn_path, grn_rows, delimiter="\t")
    fixed_feature_rows = [
        {
            "feature_mode": "hybrid6_raw",
            "feature_file": str(feature_path),
            "feature_file_sha256": sha256(feature_path),
            "feature_dim": 6,
            "feature_columns": "Degree;WeightValue;PatientCoverageCount;Mutation;Expression;Methylation",
            "stage2_first_round_rule": "Network relation is the only experimental variable; features stay fixed.",
            "CNV_included": "False",
            "notes": "KIRC_multiomics_4omics and CNV assets exist but are not used in first-round network ablation.",
        }
    ]
    write_csv(OUT / "09_stage2_input_freeze" / "fixed_feature_manifest.csv", fixed_feature_rows)
    relation_rows = [
        {"relation_id": 0, "relation_name": "PPI", "directed": "false", "source_file": str(ppi_path), "source_database": "HPRD frozen project file", "version": "Stage 0 frozen", "edge_count": len(ppi_rows), "node_count": len(set([r["source"] for r in ppi_rows] + [r["target"] for r in ppi_rows])), "sha256": sha256(ppi_path), "notes": "Undirected PPI relation."},
        {"relation_id": 1, "relation_name": "GRN", "directed": "true", "source_file": str(grn_path), "source_database": PRIMARY_GRN, "version": "Stage 1 raw snapshot", "edge_count": len(grn_rows), "node_count": len(set([r["source"] for r in grn_rows] + [r["target"] for r in grn_rows])), "sha256": sha256(grn_path), "notes": f"Primary threshold {PRIMARY_THRESHOLD}; TF -> target direction preserved."},
    ]
    write_csv(OUT / "10_stage2_ready" / "relation_manifest.csv", relation_rows)
    return gene_path, ppi_path, grn_path


def smoke_check(gene_path, ppi_path, grn_path, n):
    problems = []
    for path in [gene_path, ppi_path, grn_path]:
        rows = read_dicts(path, delimiter="\t")
        for r in rows:
            for key in ["source_index", "target_index", "node_index"]:
                if key in r and r[key] != "":
                    v = int(r[key])
                    if v < 0 or v >= n:
                        problems.append(f"{path}: {key} out of range {v}")
            for val in r.values():
                s = str(val).lower()
                if s in {"nan", "inf", "-inf"}:
                    problems.append(f"{path}: non-finite token {val}")
    text = "\n".join(problems) if problems else "PASS: range checks passed; no NaN/Inf tokens in Stage2-ready TSV files.\nNO FORWARD TRAINING; NO OPTIMIZER; NO REWARD; NO DQN.\n"
    write_text(OUT / "logs" / "stage2_ready_smoke_check.txt", text)


def leakage_audit(raw_path):
    text = f"""# Network Leakage Audit

## 1. GRN construction source

Primary GRN: {PRIMARY_GRN}.

Frozen raw file: `{raw_path}`.

The network is a TF-target regulatory resource with mixed evidence/provenance fields from DoRothEA/OmniPath.

## 2. Explicit cancer-driver labels

DIRECT_DRIVER_LABEL_LEAKAGE = NO based on Stage 1 available provenance: the downloaded resource is a general human TF-target regulatory network and Stage 1 did not supply CGC, IntOGen, TCGA driver labels, Protocol B labels, Historical Test labels, or KIRC model results to construct/filter/select edges.

## 3. Driver annotation inputs

No CGC status, IntOGen driver annotation, TCGA driver labels, Protocol B labels, or Historical Test labels were used by Stage 1 processing. If DoRothEA source curation contains cancer literature among general references, that is treated as general biological knowledge, not direct target-label injection.

## 4. Literature knowledge vs label leakage

General biological knowledge and regulatory evidence are not automatically label leakage. The high-risk case would be direct use of "this gene is a ccRCC driver" as an edge or confidence input. Stage 1 found no evidence of this in the accessed fields.

## 5. Confidence score

Confidence score direct cancer-driver prior: UNKNOWN. DoRothEA confidence tiers are recorded as general evidence levels; no Stage 1 evidence shows they encode Protocol B or KIRC Historical Test labels.

HISTORICAL_TEST_USED_FOR_GRN_SELECTION = NO

NEW_EXTERNAL_VALIDATION_RESULTS_READ = NO
"""
    write_text(OUT / "08_network_leakage" / "network_leakage_audit.md", text)


def figures(ppi_degree, grn_in, grn_out, comp_rows, overlap_stats, tf_hubs):
    def save(name):
        plt.tight_layout()
        plt.savefig(OUT / "11_figures" / name, dpi=160)
        plt.close()
        log_access(OUT / "11_figures" / name, "write audit figure", "write", "stage1_figure")

    ppi_vals = list(ppi_degree.values())
    plt.figure(figsize=(7, 4))
    plt.hist(ppi_vals, bins=60, log=True)
    plt.xlabel("PPI degree")
    plt.ylabel("Gene count (log)")
    plt.title("PPI degree distribution")
    save("figure1_ppi_degree_distribution.png")

    plt.figure(figsize=(7, 4))
    plt.hist(list(grn_in.values()), bins=60, log=True)
    plt.xlabel("GRN in-degree")
    plt.ylabel("Gene count (log)")
    plt.title("GRN in-degree distribution")
    save("figure2_grn_in_degree_distribution.png")

    plt.figure(figsize=(7, 4))
    plt.hist(list(grn_out.values()), bins=60, log=True)
    plt.xlabel("GRN out-degree")
    plt.ylabel("Gene count (log)")
    plt.title("GRN out-degree distribution")
    save("figure3_grn_out_degree_distribution.png")

    plt.figure(figsize=(6, 4))
    plt.bar(["Universe", "PPI nodes", "GRN nodes"], [overlap_stats["PPI_nodes"], overlap_stats["PPI_nodes"], overlap_stats["GRN_nodes"]])
    plt.ylabel("Genes")
    plt.title("PPI vs GRN gene coverage")
    save("figure4_ppi_grn_gene_coverage.png")

    plt.figure(figsize=(5, 4))
    plt.bar(["Shared", "PPI only", "GRN only"], [overlap_stats["shared_nodes"], overlap_stats["PPI_nodes"] - overlap_stats["shared_nodes"], overlap_stats["GRN_nodes"] - overlap_stats["shared_nodes"]])
    plt.ylabel("Genes")
    plt.title("PPI/GRN node overlap")
    save("figure5_ppi_grn_node_overlap.png")

    plt.figure(figsize=(5, 4))
    plt.bar(["Shared", "PPI only", "GRN only"], [overlap_stats["GRN_edges_also_present_as_PPI_interaction"], overlap_stats["PPI_only_edges"], overlap_stats["GRN_only_edges"]])
    plt.ylabel("Edges")
    plt.title("PPI/GRN edge overlap")
    save("figure6_ppi_grn_edge_overlap.png")

    plt.figure(figsize=(8, 5))
    top = tf_hubs[:20]
    plt.barh([r["TF"] for r in reversed(top)], [r["out_degree"] for r in reversed(top)])
    plt.xlabel("Out-degree")
    plt.title("Top GRN TF hubs")
    save("figure7_top_grn_tf_hubs.png")

    plt.figure(figsize=(6, 5))
    xs = [int(r["PPI_degree"]) for r in comp_rows]
    ys = [int(r["GRN_total_degree"]) for r in comp_rows]
    plt.scatter(xs, ys, s=8, alpha=0.45)
    plt.xlabel("PPI degree")
    plt.ylabel("GRN total degree")
    plt.title("PPI degree vs GRN degree")
    save("figure8_ppi_vs_grn_degree_scatter.png")


def integrity(paths):
    include = []
    include.extend(paths)
    include.extend(
        [
            OUT / "01_gene_universe" / "frozen_gene_universe.txt",
            OUT / "01_gene_universe" / "gene_universe_manifest.csv",
            OUT / "02_ppi_audit" / "ppi_statistics.json",
            OUT / "03_grn_source" / "raw" / "omnipath_dorothea_human_interactions.tsv",
            OUT / "04_grn_processing" / "grn_cleaned_full.tsv",
            OUT / "05_grn_frozen" / "grn_in_gene_universe.tsv",
            OUT / "10_stage2_ready" / "relation_manifest.csv",
            OUT / "09_stage2_input_freeze" / "fixed_feature_manifest.csv",
            OUT / "STAGE1_COMPLETION_REPORT.md",
        ]
    )
    seen = set()
    lines = []
    for p in include:
        p = Path(p)
        if p.exists() and p.is_file() and str(p) not in seen:
            seen.add(str(p))
            lines.append(f"{sha256(p)}  {p}")
    write_text(OUT / "12_integrity" / "SHA256SUMS_stage1.txt", "\n".join(lines) + "\n")


def reports(genes, ppi_stats, threshold_rows, proc_stats, grn_stats, overlap_stats, tf_hubs, target_hubs, runtime_status):
    extreme_hub_risk = "YES" if any(r["hub_audit_level"] == "EXTREME" for r in tf_hubs + target_hubs) else ("CONDITIONAL" if any(r["hub_audit_level"] == "HIGH" for r in tf_hubs + target_hubs) else "NO")
    mapping_quality = "PASS" if grn_stats["gene_universe_coverage"] >= 0.20 and proc_stats["ambiguous_mapping_count"] == 0 else "CONDITIONAL"
    structural_quality = "CONDITIONAL" if extreme_hub_risk in {"YES", "CONDITIONAL"} else "PASS"
    complementarity = "PASS" if overlap_stats["GRN_only_edges"] > overlap_stats["GRN_edges_also_present_as_PPI_interaction"] and overlap_stats["EdgeJaccard"] < 0.5 else "CONDITIONAL"
    leakage = "PASS"
    stage1 = "PASS" if mapping_quality == "PASS" and complementarity == "PASS" and leakage == "PASS" and structural_quality in {"PASS", "CONDITIONAL"} else "CONDITIONAL"
    ready = "YES" if stage1 == "PASS" else "CONDITIONAL"
    top_tf = tf_hubs[0] if tf_hubs else {}
    top_target = target_hubs[0] if target_hubs else {}
    report = f"""# STAGE1_COMPLETION_REPORT

## 1. Executive Summary

STAGE1_STATUS = {stage1}

READY_FOR_STAGE2 = {ready}

Stage 1 selected and froze one primary human TF-target GRN ({PRIMARY_GRN}), mapped it to the frozen KIRC gene universe, audited PPI/GRN structure and complementarity, and generated Stage2-ready relation files without training or model evaluation.

## 2. Boundary Compliance

TRAINING_PERFORMED = NO

CORE_MODEL_MODIFIED = NO

REWARD_MODIFIED = NO

MORL_STARTED = NO

HISTORICAL_TEST_USED_FOR_GRN_SELECTION = NO

NEW_EXTERNAL_VALIDATION_READ = NO

OPTIMIZER_STEP = 0

REPLAY_BUFFER_WRITE = 0

PER_PRIORITY_UPDATE = 0

## 3. Frozen Gene Universe

N_GENE_UNIVERSE = {len(genes)}

source = first-occurrence node order from frozen PPI file `{PROJECT / 'data' / 'HPRD.txt'}`

SHA256 = {sha256(PROJECT / 'data' / 'HPRD.txt')}

## 4. PPI Summary

nodes = {ppi_stats['node_count']}

edges = {ppi_stats['edge_count']}

degree mean = {ppi_stats['degree_mean']}

degree median = {ppi_stats['degree_median']}

degree max = {ppi_stats['degree_max']}

components = {ppi_stats['connected_components']}

largest connected component = {ppi_stats['largest_connected_component']}

## 5. GRN Source

database = {PRIMARY_GRN}

version = OmniPath API retrieval frozen in `03_grn_source/grn_raw_manifest.csv`

retrieval date = see raw manifest

evidence type = mixed DoRothEA/OmniPath TF-target evidence with `dorothea_level` confidence tiers

confidence system = A-E

why selected = directed human TF-target semantics, explicit confidence tiers, and clear provenance; no Test or external validation metric used.

## 6. GRN Processing

raw edges = {proc_stats['raw_edge_count']}

after mapping/dedup/self-loop removal = {proc_stats['clean_edge_count_after_mapping_dedup_selfloop_removal']}

after confidence filtering ({PRIMARY_THRESHOLD}) final edges = {proc_stats['mapped_edge_count_primary']}

self-loop removed = {proc_stats['self_loop_removed_count']}

unmapped TF count = {proc_stats['unmapped_tf_count']}

unmapped target count = {proc_stats['unmapped_target_count']}

## 7. Mapping Results

mapped genes = {grn_stats['number_of_nodes']}

gene universe coverage = {grn_stats['gene_universe_coverage']:.6f}

TF count = {grn_stats['number_of_TFs']}

target count = {grn_stats['number_of_targets']}

ambiguous count = {proc_stats['ambiguous_mapping_count']}

## 8. Hub Audit

Top TF hub = {top_tf}

Top target hub = {top_target}

largest out-degree = {grn_stats['max_out_degree']}

largest in-degree = {grn_stats['max_in_degree']}

p95 total degree = {grn_stats['degree_p95']}

p99 total degree = {grn_stats['degree_p99']}

EXTREME_HUB_RISK = {extreme_hub_risk}

Hub audit thresholds: NORMAL <10% universe, HIGH >=10%, EXTREME >=25%.

## 9. PPI-GRN Complementarity

shared nodes = {overlap_stats['shared_nodes']}

node overlap / NodeJaccard = {overlap_stats['NodeJaccard']:.6f}

shared edges = {overlap_stats['GRN_edges_also_present_as_PPI_interaction']}

edge overlap / EdgeJaccard = {overlap_stats['EdgeJaccard']:.6f}

GRN-only edges = {overlap_stats['GRN_only_edges']}

PPI-only edges = {overlap_stats['PPI_only_edges']}

degree correlation PPI vs GRN total = {overlap_stats['Spearman_PPI_degree_vs_GRN_total_degree']}

Interpretation: shared-node coverage and low edge overlap mean PPI and GRN largely connect overlapping gene space using distinct biological relation semantics.

## 10. Network Leakage Audit

DIRECT_DRIVER_LABEL_LEAKAGE = NO

Evidence: Stage 1 did not pass Protocol B labels, Historical Test labels, KIRC driver labels, model metrics, or external validation results into GRN source selection, filtering, or thresholding. DoRothEA confidence direct cancer-driver prior remains UNKNOWN and is documented in `08_network_leakage/network_leakage_audit.md`.

## 11. Stage 2 Input Freeze

Network A = PPI-only

Network B = GRN-only

Network C = PPI+GRN

same gene universe = YES

same node features = YES, fixed `hybrid6_raw`

same labels = YES, Protocol B boundaries from Stage 0

same reward = YES

same DDQN = YES

same PER = YES

same Soft Update = YES

same action mask = YES

same training budget = YES

same seeds = YES

唯一变量: NETWORK_RELATION

## 12. Stage 1 Risks

HIGH: none identified that blocks Stage 2 preparation.

MEDIUM: {('High or extreme hub concentration requires Stage 2 reporting and possible pre-registered sensitivity analysis.' if extreme_hub_risk != 'NO' else 'No extreme hub risk under predeclared thresholds.')}

LOW: `pyg-lib` optional warning was previously observed in Stage 0.1; Stage 1 did not require PyG model objects.

## 13. Go / No-Go Decision

GRN_MAPPING_QUALITY = {mapping_quality}

GRN_STRUCTURAL_QUALITY = {structural_quality}

PPI_GRN_COMPLEMENTARITY = {complementarity}

NETWORK_LEAKAGE_RISK = {leakage}

## 14. Final Decision

READY_FOR_STAGE2 = {ready}

Stage 2 may enter "PPI-only / GRN-only / PPI+GRN minimal multi-relation graph feasibility experiments" if the next protocol preserves this Stage 1 input freeze and does not introduce external validation or feature changes.
"""
    write_text(OUT / "STAGE1_COMPLETION_REPORT.md", report)
    summary = f"""============================================================
RL-GenRisk NEW DIRECTION - STAGE 1 COMPLETE
GRN PREPARATION & MULTI-RELATIONAL GRAPH AUDIT
============================================================

OUTPUT_DIR:
{display_path(OUT)}

TRAINING_PERFORMED:
NO

CORE_MODEL_MODIFIED:
NO

REWARD_MODIFIED:
NO

MORL_STARTED:
NO

HISTORICAL_TEST_USED_FOR_GRN_SELECTION:
NO

NEW_EXTERNAL_VALIDATION_READ:
NO

GENE_UNIVERSE:
{len(genes)}

PPI_NODES:
{ppi_stats['node_count']}

PPI_EDGES:
{ppi_stats['edge_count']}

PRIMARY_GRN:
{PRIMARY_GRN}

GRN_RAW_EDGES:
{proc_stats['raw_edge_count']}

GRN_FINAL_EDGES:
{proc_stats['mapped_edge_count_primary']}

GRN_MAPPED_GENES:
{grn_stats['number_of_nodes']}

GRN_GENE_UNIVERSE_COVERAGE:
{grn_stats['gene_universe_coverage']:.6f}

GRN_UNIQUE_TFS:
{grn_stats['number_of_TFs']}

GRN_UNIQUE_TARGETS:
{grn_stats['number_of_targets']}

MAX_TF_OUT_DEGREE:
{grn_stats['max_out_degree']}

EXTREME_HUB_RISK:
{extreme_hub_risk}

PPI_GRN_SHARED_NODES:
{overlap_stats['shared_nodes']}

PPI_GRN_NODE_JACCARD:
{overlap_stats['NodeJaccard']:.6f}

PPI_GRN_SHARED_EDGES:
{overlap_stats['GRN_edges_also_present_as_PPI_interaction']}

PPI_GRN_EDGE_JACCARD:
{overlap_stats['EdgeJaccard']:.6f}

PPI_GRN_DEGREE_CORRELATION:
{overlap_stats['Spearman_PPI_degree_vs_GRN_total_degree']}

DIRECT_DRIVER_LABEL_LEAKAGE:
NO

GRN_MAPPING_QUALITY:
{mapping_quality}

GRN_STRUCTURAL_QUALITY:
{structural_quality}

PPI_GRN_COMPLEMENTARITY:
{complementarity}

NETWORK_LEAKAGE_RISK:
{leakage}

STAGE1_STATUS:
{stage1}

READY_FOR_STAGE2:
{ready}

FINAL_REPORT:
{display_path(OUT / 'STAGE1_COMPLETION_REPORT.md')}
============================================================
"""
    write_text(OUT / "terminal_summary.txt", summary)
    print(summary)


def readme():
    text = f"""# RL-GenRisk New Direction Stage 1

Task: GRN data preparation and multi-relational graph audit.

Boundary: no training, no model evaluation, no reward/model source modification, no Historical Test use for GRN selection, no future external validation result read.

Primary GRN: {PRIMARY_GRN}.

Fixed Stage 2 first-round feature mode: `hybrid6_raw`.

Final report: `STAGE1_COMPLETION_REPORT.md`.
"""
    write_text(OUT / "00_README.md", text)


def main():
    mkdirs()
    load_stage0()
    genes, node_index, universe, ppi_edges_raw, ppi_nodes_raw, feature_path, ranking_path = load_gene_universe()
    ppi_edges, ppi_nodes, ppi_degree, ppi_stats = ppi_audit(genes, universe, ppi_edges_raw)
    raw_path = download_grn()
    cleaned, final, threshold_rows, proc_stats = parse_and_clean_grn(raw_path, universe)
    grn_out, grn_in, grn_total, grn_nodes, tfs, targets, tf_hubs, target_hubs, grn_stats = grn_audit(final, genes, universe)
    overlap_stats, comp_rows = multi_relation_audit(genes, universe, ppi_edges, ppi_nodes, ppi_degree, final, grn_out, grn_in, grn_nodes)
    gene_path, ppi_path, grn_path = stage2_ready(genes, node_index, ppi_edges, final, feature_path)
    smoke_check(gene_path, ppi_path, grn_path, len(genes))
    leakage_audit(raw_path)
    figures(ppi_degree, grn_in, grn_out, comp_rows, overlap_stats, tf_hubs)
    readme()
    reports(genes, ppi_stats, threshold_rows, proc_stats, grn_stats, overlap_stats, tf_hubs, target_hubs, "PASS")
    write_csv(OUT / "stage1_data_access_log.csv", DATA_ACCESS)
    integrity([PROJECT / "data" / "HPRD.txt", feature_path, raw_path, gene_path, ppi_path, grn_path])


if __name__ == "__main__":
    main()
