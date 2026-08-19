"""Sensitivity analysis for the B1 donor-triad enumeration tolerance.

This is not a threshold-selection procedure.  The primary detector remains
frozen at 1.50 A; the sweep reports whether conclusions depend on the
candidate-enumeration prefilter.  The final six-donor RMSD gate remains fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from b1_structural_model import load_b1_template
from metal_independent_b1 import score_without_predicted_metals
from utils import get_logger

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(path: str, template_path: str, tolerance: float) -> bool:
    result = score_without_predicted_metals(
        Path(path), load_b1_template(Path(template_path)),
        pair_distance_tolerance=tolerance,
    )
    return bool(result["architecture_call"])


def confusion(ids: list[str], truth: dict[str, bool], calls: dict[str, bool]) -> dict:
    tp = sum(truth[sid] and calls[sid] for sid in ids)
    fn = sum(truth[sid] and not calls[sid] for sid in ids)
    fp = sum(not truth[sid] and calls[sid] for sid in ids)
    tn = sum(not truth[sid] and not calls[sid] for sid in ids)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(ids), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None
        ),
    }


def score_paths(
    paths: dict[str, Path], template: Path, tolerance: float, workers: int,
) -> dict[str, bool]:
    calls = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score, str(path), str(template), tolerance): sid
            for sid, path in paths.items()
        }
        for future in as_completed(futures):
            calls[futures[future]] = future.result()
    if calls.keys() != paths.keys():
        raise AssertionError("threshold sweep returned an incomplete ID set")
    return calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--external-config", required=True, type=Path)
    parser.add_argument("--external-chains-dir", required=True, type=Path)
    parser.add_argument("--tolerances", nargs="+", type=float,
                        default=[1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--primary-tolerance", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.primary_tolerance not in args.tolerances:
        raise ValueError("primary tolerance must be included in the sensitivity sweep")

    manifest = list(csv.DictReader(args.manifest.open()))
    labels = {row["structure_id"]: row["label"] for row in manifest}
    subclasses = {row["structure_id"]: row["subclass"] for row in manifest}
    internal_truth = {
        sid: labels[sid] == "positive" and subclasses[sid] == "B1" for sid in labels
    }
    internal_paths = {sid: args.structures_dir / f"{sid}.pdb" for sid in labels}
    missing = [str(path) for path in internal_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} internal structures; first: {missing[0]}")

    splits = json.loads(args.splits.read_text())
    panel = splits["panels"]["B1_B2_transfer"]["test_ids"]
    panel_ids = [sid for sid in panel if internal_truth[sid] or labels[sid] != "positive"]
    all_b1_and_negatives = [
        sid for sid in labels if internal_truth[sid] or labels[sid] != "positive"
    ]

    external_config = json.loads(args.external_config.read_text())
    external_paths = {
        row["pdb_id"].upper(): args.external_chains_dir / f"{row['pdb_id'].upper()}.pdb"
        for row in external_config["entries"]
    }
    external_truth = {
        row["pdb_id"].upper(): row["group"] == "canonical_B1"
        for row in external_config["entries"]
    }
    external_ids = [
        row["pdb_id"].upper() for row in external_config["entries"]
        if row["group"] in {"canonical_B1", "B2_control", "B3_control"}
    ]

    sweep = {}
    for tolerance in args.tolerances:
        log.info("Scoring threshold sensitivity at %.2f A", tolerance)
        internal_calls = score_paths(
            internal_paths, args.template, tolerance, args.workers
        )
        external_calls = score_paths(
            external_paths, args.template, tolerance, args.workers
        )
        sweep[f"{tolerance:.2f}"] = {
            "internal_B1_panel_vs_panel_negatives": confusion(
                panel_ids, internal_truth, internal_calls
            ),
            "internal_all_B1_vs_all_labeled_negatives": confusion(
                all_b1_and_negatives, internal_truth, internal_calls
            ),
            "external_canonical_B1_vs_B2_B3_controls": confusion(
                external_ids, external_truth, external_calls
            ),
            "external_positive_ids": [sid for sid in external_ids if external_calls[sid]],
        }

    output = {
        "schema_version": 1,
        "purpose": "candidate-enumeration sensitivity analysis, not threshold fitting",
        "primary_pair_distance_tolerance_angstrom": args.primary_tolerance,
        "fixed_pharmacophore_rmsd_gate_angstrom": 1.25,
        "input_hashes": {
            "template": sha256(args.template), "manifest": sha256(args.manifest),
            "splits": sha256(args.splits),
            "external_config": sha256(args.external_config),
        },
        "sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    log.info("Wrote B1 threshold sensitivity report -> %s", args.out)


if __name__ == "__main__":
    main()
