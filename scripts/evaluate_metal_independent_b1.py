"""Frozen evaluation of the full-chain, metal-independent B1 detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from b1_structural_model import load_b1_template
from metal_independent_b1 import extract_donors, score_donor_roles
from utils import get_logger

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_one(
    structure_id: str, structure_path: str, template_path: str, seed: int,
    pair_distance_tolerance: float,
) -> tuple[str, dict]:
    template = load_b1_template(Path(template_path))
    protein, roles = extract_donors(Path(structure_path))
    native = score_donor_roles(
        protein, roles, template, pair_distance_tolerance=pair_distance_tolerance
    )

    donor_atoms = [atom for role in ("HIS_N", "ASP_O", "CYS_S") for atom in roles[role]]
    rng = np.random.default_rng(seed)
    permuted_coords = [donor_atoms[i].coord.copy() for i in rng.permutation(len(donor_atoms))]
    cursor = 0
    scrambled_roles = {}
    for role in ("HIS_N", "ASP_O", "CYS_S"):
        scrambled_roles[role] = []
        for atom in roles[role]:
            scrambled_roles[role].append(replace(atom, coord=permuted_coords[cursor]))
            cursor += 1
    scrambled = score_donor_roles(
        protein, scrambled_roles, template,
        pair_distance_tolerance=pair_distance_tolerance,
    )

    distinct = {
        role: len({atom.residue_key for atom in atoms}) for role, atoms in roles.items()
    }
    inventory_call = (
        distinct["HIS_N"] >= 4 and distinct["ASP_O"] >= 1 and distinct["CYS_S"] >= 1
    )
    pharmacophore_call = bool(native["architecture_call"])
    return structure_id, {
        "native": native,
        "scrambled": scrambled,
        "distinct_donor_residue_counts": distinct,
        "donor_inventory_call": inventory_call,
        "within_site_geometry_call": (
            native.get("n_dch_triads", 0) > 0 and native.get("n_three_his_triads", 0) > 0
        ),
        "pharmacophore_call": pharmacophore_call,
        "scrambled_pharmacophore_call": bool(
            scrambled["architecture_call"]
        ),
    }


def metrics(ids: list[str], truth: dict[str, bool], calls: dict[str, bool]) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--similarity-audit", required=True, type=Path)
    parser.add_argument("--esm2-baseline", required=True, type=Path)
    parser.add_argument("--metal-anchored-results", required=True, type=Path)
    parser.add_argument("--fargene-results", required=True, type=Path)
    parser.add_argument("--fargene-b1-results", required=True, type=Path)
    parser.add_argument("--plm-arg-results", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--pair-distance-tolerance", type=float, default=1.5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest_rows = list(csv.DictReader(args.manifest.open()))
    expected_ids = [row["structure_id"] for row in manifest_rows]
    labels = {row["structure_id"]: row["label"] for row in manifest_rows}
    subclasses = {row["structure_id"]: row["subclass"] or None for row in manifest_rows}
    truth = {sid: labels[sid] == "positive" and subclasses[sid] == "B1" for sid in expected_ids}
    catalog = {row["accession"]: row for row in csv.DictReader(args.catalog.open())}
    split = json.loads(args.splits.read_text())
    similarity = json.loads(args.similarity_audit.read_text())
    esm2 = json.loads(args.esm2_baseline.read_text())
    anchored = json.loads(args.metal_anchored_results.read_text())
    fargene = json.loads(args.fargene_results.read_text())
    fargene_b1 = json.loads(args.fargene_b1_results.read_text())
    plm_arg = json.loads(args.plm_arg_results.read_text())

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index, sid in enumerate(expected_ids):
            path = args.structures_dir / f"{sid}.pdb"
            if not path.exists():
                raise FileNotFoundError(path)
            future = pool.submit(
                _score_one, sid, str(path), str(args.template), args.seed + index,
                args.pair_distance_tolerance,
            )
            futures[future] = sid
        for completed, future in enumerate(as_completed(futures), start=1):
            sid, result = future.result()
            results[sid] = result
            if completed % 100 == 0:
                log.info("Scored %d/%d", completed, len(expected_ids))
    if set(results) != set(expected_ids):
        raise AssertionError("metal-independent scorer returned an incomplete ID set")

    calls = {
        "donor_inventory_only": {sid: row["donor_inventory_call"] for sid, row in results.items()},
        "within_site_geometry": {sid: row["within_site_geometry_call"] for sid, row in results.items()},
        "six_donor_pharmacophore": {sid: row["pharmacophore_call"] for sid, row in results.items()},
        "full_geometry_and_product_pose": {
            sid: row["native"]["positive_call"] for sid, row in results.items()
        },
        "donor_role_coordinate_permutation": {
            sid: row["scrambled"]["positive_call"] for sid, row in results.items()
        },
        "donor_role_permutation_pharmacophore": {
            sid: row["scrambled_pharmacophore_call"] for sid, row in results.items()
        },
        "fargene_B1_B2_HMM": {
            sid: fargene["per_example"][sid]["predicted_positive"] for sid in expected_ids
        },
        "fargene_B1_specific_HMM": {
            sid: fargene_b1["per_example"][sid]["predicted_positive"] for sid in expected_ids
        },
        "PLM_ARG_beta_lactam": {
            sid: plm_arg["per_example"][sid]["predicted_beta_lactam"] for sid in expected_ids
        },
    }

    panel_ids = split["panels"]["B1_B2_transfer"]["test_ids"]
    b1_panel = [sid for sid in panel_ids if truth[sid]]
    panel_negatives = [sid for sid in panel_ids if labels[sid] != "positive"]
    all_b1 = [sid for sid in expected_ids if truth[sid]]
    all_negatives = [sid for sid in expected_ids if labels[sid] != "positive"]
    known_b2_b3 = [
        sid for sid in expected_ids
        if labels[sid] == "positive" and subclasses[sid] in {"B2", "B3"}
    ]
    unclassified_positives = [
        sid for sid in expected_ids
        if labels[sid] == "positive" and subclasses[sid] not in {"B1", "B2", "B3"}
    ]
    zero_identity = [
        sid for sid in b1_panel
        if similarity["panel:B1_B2_transfer"]["examples"][sid]["max_identity_at_80cov"] == 0.0
    ]
    esm2_misses = [
        sid for sid in b1_panel
        if esm2["panel:B1_B2_transfer"]["per_example"][sid]["pred"] == "negative"
    ]
    fargene_misses = [sid for sid in all_b1 if not calls["fargene_B1_specific_HMM"][sid]]
    populations = {
        "B1_panel_vs_panel_negatives": b1_panel + panel_negatives,
        "B1_panel_only": b1_panel,
        "zero_80cov_identity_B1_panel": zero_identity,
        "mean_ESM2_missed_B1_panel": esm2_misses,
        "fargene_missed_all_B1": fargene_misses,
        "all_B1": all_b1,
        "all_B1_vs_known_B2_B3": all_b1 + known_b2_b3,
        "all_B1_vs_known_B2_B3_and_labeled_negatives": (
            all_b1 + known_b2_b3 + all_negatives
        ),
        "all_labeled_negatives": all_negatives,
    }
    evaluation = {
        method: {population: metrics(ids, truth, method_calls)
                 for population, ids in populations.items()}
        for method, method_calls in calls.items()
    }

    full_calls = calls["full_geometry_and_product_pose"]
    false_positives = [sid for sid in all_negatives if full_calls[sid]]
    false_positives_by_family = defaultdict(list)
    for sid in false_positives:
        false_positives_by_family[catalog.get(sid, {}).get("neg_family") or "unknown"].append(sid)
    recovered_vs_anchored = [
        sid for sid in all_b1
        if full_calls[sid] and not anchored["per_example"][sid]["positive_call"]
    ]

    output = {
        "schema_version": 1,
        "model": "metal_independent_b1_pharmacophore_v1",
        "scientific_scope": {
            "uses_sequence": False,
            "uses_predicted_metal_coordinates": False,
            "uses_labeled_reference_panel": False,
            "uses_experimental_reaction_state_template": True,
            "claim": "canonical B1 catalytic-architecture support, not catalytic proof",
        },
        "input_hashes": {
            "template": sha256(args.template), "manifest": sha256(args.manifest),
            "splits": sha256(args.splits), "similarity_audit": sha256(args.similarity_audit),
            "esm2_baseline": sha256(args.esm2_baseline),
            "metal_anchored_results": sha256(args.metal_anchored_results),
            "fargene_results": sha256(args.fargene_results),
            "fargene_b1_results": sha256(args.fargene_b1_results),
            "plm_arg_results": sha256(args.plm_arg_results),
        },
        "control_seed": args.seed,
        "pair_distance_tolerance_angstrom": args.pair_distance_tolerance,
        "population_sizes": {name: len(ids) for name, ids in populations.items()},
        "evaluation": evaluation,
        "status_counts": dict(Counter(row["native"]["status"] for row in results.values())),
        "false_positive_ids": false_positives,
        "false_positive_ids_by_family": dict(false_positives_by_family),
        "metal_independent_recovery_over_metal_anchored_B1": recovered_vs_anchored,
        "fargene_negative_B1_recovered": [sid for sid in fargene_misses if full_calls[sid]],
        "unclassified_positive_architecture_calls": [
            sid for sid in unclassified_positives
            if calls["six_donor_pharmacophore"][sid]
        ],
        "per_example": {
            sid: {
                "label": labels[sid], "subclass": subclasses[sid],
                **row,
            }
            for sid, row in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    log.info("Wrote metal-independent B1 evaluation -> %s", args.out)


if __name__ == "__main__":
    main()
