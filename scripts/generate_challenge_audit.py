"""
generate_challenge_audit.py

Compact, TRACKED audit of component_challenge_split.py's output (which
lives under gitignored data/, so isn't itself in the repo): input hashes,
panel/LONO counts, label and family composition per panel, and an explicit
statement of the reference-bank policy -- so a future reader (or a
downstream evaluation script) can see at a glance what these splits are
and are not, without re-deriving it from the full challenge_splits.json.

CLI:
    python generate_challenge_audit.py --challenge-splits data/challenge_splits.json \
        --split-graph data/split_graph.json --pockets-dir data/pockets \
        --catalog full_structure_catalog.csv --out reports/challenge_split_audit.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

REFERENCE_BANK_IDS = {"NDM-1", "VIM-2", "IMP-1", "CphA", "Sfh-I", "L1", "FEZ-1"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metadata(pockets_dir: Path, catalog_path: Path) -> dict[str, dict]:
    meta = {}
    for f in pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        meta[pocket.metadata.source_structure_id] = {"label": pocket.metadata.label, "subclass": pocket.metadata.subclass}
    catalog = {r["accession"]: r for r in csv.DictReader(open(catalog_path))}
    for sid, row in catalog.items():
        if sid in meta:
            meta[sid]["neg_family"] = row.get("neg_family", "")
    return meta


def composition(ids: list[str], meta: dict[str, dict]) -> dict:
    labels = Counter(meta.get(s, {}).get("label", "?") for s in ids)
    subclasses = Counter(meta.get(s, {}).get("subclass") for s in ids if meta.get(s, {}).get("label") == "positive")
    families = Counter(meta.get(s, {}).get("neg_family", "") for s in ids if meta.get(s, {}).get("label") != "positive")
    return {
        "n": len(ids),
        "label_counts": dict(labels),
        "positive_subclass_counts": dict(subclasses),
        "neg_family_counts": {k: v for k, v in families.items() if k},
        "reference_bank_present": sorted(set(ids) & REFERENCE_BANK_IDS),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    challenge = json.loads(args.challenge_splits.read_text())
    meta = load_metadata(args.pockets_dir, args.catalog)

    report = {
        "input_hashes": {
            "challenge_splits_json": file_hash(args.challenge_splits),
            "split_graph_json": file_hash(args.split_graph),
        },
        "regime": challenge["regime"],
        "reference_bank_policy": (
            "The reference bank is a deployment-time retrieval anchor set, NOT an independent validation set: "
            "5/7 (NDM-1, VIM-2, IMP-1, CphA, Sfh-I) sit inside the 116-member B1/B2 sequence component, 2/7 "
            "(FEZ-1, L1) inside the 26-member B3 component. In the 3 main panels, references travel with their "
            "natural component and become unavailable as anchors whenever that component is held out (mechanism/"
            "generalization protocol). operational_reference_anchored_retrieval is a SEPARATE, explicitly-labeled "
            "protocol where the reference bank stays in train as anchors and only the 4 reference-free singleton "
            "positives are evaluated -- report those as individual discovery case studies, not a population "
            "sensitivity estimate. evaluate.py's reference_bank_leave_one_out_ranks measures near-duplicate "
            "retrieval, not external generalization, on this dataset -- see its docstring."
        ),
        "panels": {
            name: {
                "test_positive_component_ids": panel["test_positive_component_ids"],
                "test": composition(panel["test_ids"], meta),
                "train": composition(panel["train_ids"], meta),
            }
            for name, panel in challenge["panels"].items()
        },
        "leave_one_negative_family_out": {
            fam: {
                "n_target_family_in_test": cfg["n_target_family_in_test"],
                "n_target_family_in_train": cfg["n_target_family_in_train"],
                "collateral_family_counts_in_test": cfg["collateral_family_counts_in_test"],
                "test": composition(cfg["test_ids"], meta),
            }
            for fam, cfg in challenge["leave_one_negative_family_out"].items()
        },
        "operational_reference_anchored_retrieval": {
            "test": composition(challenge["operational_reference_anchored_retrieval"]["test_ids"], meta),
        },
        "audit": challenge["audit"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote challenge-split audit -> {args.out}")


if __name__ == "__main__":
    main()
