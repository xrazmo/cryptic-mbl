"""
pocket_report.py

Aggregates every PocketSubgraph under a pockets directory into one CSV, so
the composition of the dataset that actually enters training/eval can be
inspected before spending compute on training: label/tier/pocket_source
balance, Metal3D vs fpocket-cavity-fallback rate, pLDDT distribution, pocket
size (residues/atoms), and (optionally) which split fold each structure
lands in.

CLI:
    python pocket_report.py --pockets-dir data/pockets \
        --splits-json data/splits.json --out data/pocket_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils import PocketSubgraph, get_logger

log = get_logger(__name__)


def build_fold_lookup(splits_json: Path) -> dict[str, dict]:
    """Returns {structure_id: {"external_holdout": bool, "fold_<i>": "train"|"val"|"test"}}."""
    splits = json.loads(splits_json.read_text())
    lookup: dict[str, dict] = {}
    for sid in splits.get("external_holdout", []):
        lookup.setdefault(sid, {})["external_holdout"] = True
    for fold in splits["folds"]:
        fid = fold["fold_id"]
        for part in ("train", "val", "test"):
            for sid in fold[part]:
                lookup.setdefault(sid, {})[f"fold_{fid}"] = part
    return lookup


def build_report_rows(pockets_dir: Path, fold_lookup: dict[str, dict]) -> list[dict]:
    rows = []
    for npz_path in sorted(pockets_dir.glob("*.npz")):
        pocket = PocketSubgraph.load(npz_path)
        meta = pocket.metadata
        n_atoms = len(pocket.res_ids)
        n_residues = len(set(pocket.res_ids.tolist()))
        fold_info = fold_lookup.get(meta.source_structure_id, {})
        row = {
            "structure_id": meta.source_structure_id,
            "label": meta.label,
            "confidence_tier": meta.confidence_tier,
            "subclass": meta.subclass or "",
            "pocket_source": meta.pocket_source,
            "metal_confidence": meta.metal_confidence if meta.metal_confidence is not None else "",
            "mean_pocket_plddt": meta.mean_pocket_plddt if meta.mean_pocket_plddt is not None else "",
            "has_metal_coord": pocket.metal_coord is not None,
            "n_atoms": n_atoms,
            "n_residues": n_residues,
            "external_holdout": fold_info.get("external_holdout", False),
        }
        for k, v in fold_info.items():
            if k.startswith("fold_"):
                row[k] = v
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path):
    fold_cols = sorted({k for r in rows for k in r if k.startswith("fold_")})
    fieldnames = [
        "structure_id", "label", "confidence_tier", "subclass", "pocket_source",
        "metal_confidence", "mean_pocket_plddt", "has_metal_coord",
        "n_atoms", "n_residues", "external_holdout",
    ] + fold_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--splits-json", type=Path, default=None,
                    help="Optional; annotates each row with its fold/train-val-test assignment.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    fold_lookup = build_fold_lookup(args.splits_json) if args.splits_json else {}
    rows = build_report_rows(args.pockets_dir, fold_lookup)
    write_csv(rows, args.out)
    log.info(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
