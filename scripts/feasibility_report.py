"""
feasibility_report.py

Label-aware feasibility report for the split-graph components build_split_graph.py
produces. Written as a standalone, persisted step -- BEFORE designing any k-fold
split -- because component *count* alone is a misleading feasibility signal: a
"158 components, largest 286" summary looked healthy, but the 286-member
component turned out to be entirely one negative family, and the true positive
sample size was 6 components (2 of which hold 142/146 positives, overlapping
5/7 and 2/7 of the reference bank respectively). This report makes that kind
of composition visible up front for every regime.

Also runs the sequence-identity threshold sensitivity sweep (20/30/40/50%,
fixed 80% bidirectional coverage) requested alongside -- report-only, reusing
the already-computed exhaustive all-vs-all sequence_pairs.tsv rather than
re-running mmseqs, and explicitly NOT used to pick a threshold that happens to
produce more folds.

CLI:
    python feasibility_report.py --split-graph data/split_graph.json \
        --pockets-dir data/pockets --catalog full_structure_catalog.csv \
        --out data/feasibility_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from utils import get_logger, PocketSubgraph
from build_split_graph import components_from_edges, sequence_edges

log = get_logger(__name__)

REFERENCE_BANK_IDS = {"NDM-1", "VIM-2", "IMP-1", "CphA", "Sfh-I", "L1", "FEZ-1"}
SENSITIVITY_IDENTITY_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
SENSITIVITY_COVERAGE_THRESHOLD = 0.8


def load_metadata(pockets_dir: Path, catalog_path: Path) -> dict[str, dict]:
    meta = {}
    for f in pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        meta[pocket.metadata.source_structure_id] = {
            "label": pocket.metadata.label,
            "subclass": pocket.metadata.subclass,
            "confidence_tier": pocket.metadata.confidence_tier,
        }
    catalog = {r["accession"]: r for r in csv.DictReader(open(catalog_path))}
    for sid, row in catalog.items():
        if sid in meta:
            meta[sid]["source"] = row.get("source", "")
            meta[sid]["neg_family"] = row.get("neg_family", "")
    for sid in REFERENCE_BANK_IDS:
        if sid in meta:
            meta[sid].setdefault("source", "reference_bank")
            meta[sid].setdefault("neg_family", "")
    return meta


def component_report(components: dict[str, list[str]], meta: dict[str, dict]) -> list[dict]:
    rows = []
    for root, members in components.items():
        labels = Counter(meta.get(s, {}).get("label", "?") for s in members)
        subclasses = Counter(meta.get(s, {}).get("subclass") for s in members if meta.get(s, {}).get("label") == "positive")
        sources = Counter(meta.get(s, {}).get("source", "?") for s in members)
        tiers = Counter(meta.get(s, {}).get("confidence_tier", "?") for s in members)
        neg_families = Counter(meta.get(s, {}).get("neg_family", "") for s in members if meta.get(s, {}).get("label") != "positive")
        refs = sorted(s for s in members if s in REFERENCE_BANK_IDS)
        rows.append({
            "component_id": root,
            "size": len(members),
            "n_positive": labels.get("positive", 0),
            "label_counts": dict(labels),
            "positive_subclass_counts": dict(subclasses),
            "source_counts": dict(sources),
            "tier_counts": {str(k): v for k, v in tiers.items()},
            "neg_family_counts": {k: v for k, v in neg_families.items() if k},
            "reference_bank_members": refs,
        })
    rows.sort(key=lambda r: -r["n_positive"])
    return rows


def summarize_positive_feasibility(rows: list[dict]) -> dict:
    pos_rows = [r for r in rows if r["n_positive"] > 0]
    total_pos = sum(r["n_positive"] for r in rows)
    ref_containing = [r for r in pos_rows if r["reference_bank_members"]]
    pos_in_ref_components = sum(r["n_positive"] for r in ref_containing)
    return {
        "total_positives": total_pos,
        "n_components_with_positives": len(pos_rows),
        "positives_per_component": [r["n_positive"] for r in pos_rows],
        "positives_lost_if_reference_components_excluded": pos_in_ref_components,
        "positives_remaining_if_reference_components_excluded": total_pos - pos_in_ref_components,
    }


def sensitivity_sweep(seq_pairs_tsv: Path, all_ids: list[str], meta: dict[str, dict]) -> list[dict]:
    results = []
    for thr in SENSITIVITY_IDENTITY_THRESHOLDS:
        edges = list(sequence_edges(seq_pairs_tsv, thr, SENSITIVITY_COVERAGE_THRESHOLD))
        comps = components_from_edges(all_ids, edges)
        rows = component_report(comps, meta)
        feas = summarize_positive_feasibility(rows)
        results.append({
            "identity_threshold": thr,
            "coverage_threshold": SENSITIVITY_COVERAGE_THRESHOLD,
            "n_components": len(comps),
            "largest_component": max(len(m) for m in comps.values()),
            **feas,
        })
        log.info(f"identity>={thr}: {len(comps)} components, "
                 f"{feas['n_components_with_positives']} contain positives, "
                 f"{feas['positives_remaining_if_reference_components_excluded']}/{feas['total_positives']} "
                 f"positives survive reference-component exclusion.")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = json.loads(args.split_graph.read_text())
    meta = load_metadata(args.pockets_dir, args.catalog)
    all_ids = sorted({sid for members in d["sequence_components"].values() for sid in members})

    report = {"regimes": {}}
    for regime_key in ["sequence_components", "structure_components_foldremote", "structure_components_redundancy"]:
        rows = component_report(d[regime_key], meta)
        report["regimes"][regime_key] = {
            "feasibility": summarize_positive_feasibility(rows),
            "components": rows,
        }
        log.info(f"{regime_key}: {report['regimes'][regime_key]['feasibility']}")

    seq_pairs_tsv = Path(d["raw_pair_files"]["sequence_pairs_tsv"])
    report["sequence_identity_sensitivity_sweep"] = sensitivity_sweep(seq_pairs_tsv, all_ids, meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote feasibility report -> {args.out}")


if __name__ == "__main__":
    main()
