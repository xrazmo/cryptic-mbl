"""Evaluate the sequence-blind B1 catalytic-architecture model.

The primary population is subclass B1, not all MBL subclasses.  Every
non-positive structure remains a negative control, with hard-negative family
rates reported separately.  Missing metal predictions are counted as misses
in end-to-end sensitivity rather than silently removed.

Controls distinguish the full structural model from a cysteine lookup:

* DCH-only: coordinating cysteine within 2.8 A;
* pharmacophore-only: site-resolved donor/metal fit, ligand-pose gates relaxed;
* full architecture: pharmacophore plus transferred product-pose sanity gates;
* donor-direction scramble: distances retained, angular geometry destroyed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from b1_structural_model import load_b1_template, score_b1_structure
from catalytic_feasibility import score_template
from evaluate_catalytic_feasibility import scramble_donor_directions
from utils import PocketSubgraph, get_logger

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(ids: list[str], truth: dict[str, bool], calls: dict[str, bool]) -> dict:
    tp = sum(truth[sid] and calls[sid] for sid in ids)
    fn = sum(truth[sid] and not calls[sid] for sid in ids)
    fp = sum(not truth[sid] and calls[sid] for sid in ids)
    tn = sum(not truth[sid] and not calls[sid] for sid in ids)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(ids), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
    }


def external_calls(path: Path | None, key: str) -> dict[str, bool] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return {sid: bool(row[key]) for sid, row in payload["per_example"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--similarity-audit", required=True, type=Path)
    parser.add_argument("--esm2-baseline", required=True, type=Path)
    parser.add_argument("--fargene-results", type=Path)
    parser.add_argument("--plm-arg-results", type=Path)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open()))
    catalog = {row["accession"]: row for row in csv.DictReader(args.catalog.open())}
    split = json.loads(args.splits.read_text())
    similarity = json.loads(args.similarity_audit.read_text())
    esm2 = json.loads(args.esm2_baseline.read_text())
    template = load_b1_template(args.template)
    rng = np.random.default_rng(args.seed)

    expected_ids = [row["structure_id"] for row in manifest]
    truth = {row["structure_id"]: row["label"] == "positive" and row["subclass"] == "B1"
             for row in manifest}
    labels = {row["structure_id"]: row["label"] for row in manifest}
    subclasses = {row["structure_id"]: row["subclass"] or None for row in manifest}
    results = {}
    calls_by_method = defaultdict(dict)
    for index, sid in enumerate(expected_ids, start=1):
        path = args.pockets_dir / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        pocket = PocketSubgraph.load(path)
        full = score_b1_structure(pocket, template)
        pharmacophore = score_template(
            pocket, template, max_hard_clash_fraction=1.0, min_pocket_contact_fraction=0.0
        )
        scrambled = score_b1_structure(scramble_donor_directions(pocket, rng), template)
        calls_by_method["dch_only"][sid] = full["dch_partial_evidence"]["status"] == "supported"
        calls_by_method["site_resolved_pharmacophore"][sid] = pharmacophore["status"] == "supported"
        calls_by_method["full_b1_catalytic_architecture"][sid] = full["positive_call"]
        calls_by_method["donor_direction_scrambled"][sid] = scrambled["positive_call"]
        results[sid] = {
            "label": labels[sid], "subclass": subclasses[sid],
            "status": full["status"],
            "positive_call": full["positive_call"],
            "dch_call": calls_by_method["dch_only"][sid],
            "pharmacophore_call": calls_by_method["site_resolved_pharmacophore"][sid],
            "scrambled_call": calls_by_method["donor_direction_scrambled"][sid],
            "full_architecture_reason": full["full_architecture"].get("reason"),
            "pharmacophore_rmsd": full["full_architecture"].get("pharmacophore_rmsd"),
        }
        if index % 200 == 0:
            log.info("Scored %d/%d", index, len(expected_ids))

    b1_b2_test = split["panels"]["B1_B2_transfer"]["test_ids"]
    b1_test = [sid for sid in b1_b2_test if truth[sid]]
    test_negatives = [sid for sid in b1_b2_test if labels[sid] != "positive"]
    panel_eval = b1_test + test_negatives
    zero_identity = [
        sid for sid in b1_test
        if similarity["panel:B1_B2_transfer"]["examples"][sid]["max_identity_at_80cov"] == 0.0
    ]
    esm2_misses = [
        sid for sid in b1_test
        if esm2["panel:B1_B2_transfer"]["per_example"][sid]["pred"] == "negative"
    ]
    all_b1 = [sid for sid in expected_ids if truth[sid]]
    all_negatives = [sid for sid in expected_ids if labels[sid] != "positive"]

    populations = {
        "B1_B2_transfer_B1_vs_panel_negatives": panel_eval,
        "B1_B2_transfer_B1_only": b1_test,
        "B1_B2_transfer_zero_80cov_identity_B1": zero_identity,
        "B1_B2_transfer_ESM2_missed_B1": esm2_misses,
        "all_B1": all_b1,
        "all_labeled_negatives": all_negatives,
    }
    evaluation = {
        method: {name: metrics(ids, truth, calls) for name, ids in populations.items()}
        for method, calls in calls_by_method.items()
    }

    comparator_sources = {}
    orthogonal_recovery = {}
    for name, path, key in (
        ("fargene_B1_B2_HMM", args.fargene_results, "predicted_positive"),
        ("PLM_ARG_beta_lactam", args.plm_arg_results, "predicted_beta_lactam"),
    ):
        calls = external_calls(path, key)
        if calls is None:
            continue
        missing = set(expected_ids) - set(calls)
        if missing:
            raise ValueError(f"{name} missing {len(missing)} IDs")
        evaluation[name] = {pop: metrics(ids, truth, calls) for pop, ids in populations.items()}
        comparator_sources[name] = {"path": str(path), "sha256": sha256(path)}
        missed_b1 = [sid for sid in all_b1 if not calls[sid]]
        recovered = [sid for sid in missed_b1
                     if calls_by_method["full_b1_catalytic_architecture"][sid]]
        orthogonal_recovery[name] = {
            "n_B1_missed_by_comparator": len(missed_b1),
            "n_recovered_by_full_structure": len(recovered),
            "missed_B1_ids": missed_b1,
            "recovered_B1_ids": recovered,
        }

    # ESM2 is already a frozen panel-specific comparator, not a whole-corpus model.
    esm2_calls = {
        sid: esm2["panel:B1_B2_transfer"]["per_example"][sid]["pred"] == "positive"
        for sid in b1_b2_test
    }
    evaluation["mean_ESM2_5NN"] = {
        name: metrics(ids, truth, esm2_calls)
        for name, ids in populations.items() if set(ids) <= set(esm2_calls)
    }

    family_rates = {}
    for method, calls in calls_by_method.items():
        by_family = defaultdict(list)
        for sid in all_negatives:
            by_family[catalog.get(sid, {}).get("neg_family") or "unknown"].append(sid)
        family_rates[method] = {
            family: {
                "n": len(ids), "false_positives": sum(calls[sid] for sid in ids),
                "fpr": sum(calls[sid] for sid in ids) / len(ids),
            }
            for family, ids in sorted(by_family.items())
        }

    output = {
        "schema_version": 1,
        "model": "b1_catalytic_architecture_v1",
        "primary_claim": (
            "canonical B1 catalytic-architecture support from structure alone; "
            "not a universal MBL detector and not proof of resistance"
        ),
        "uses_sequence": False,
        "uses_labeled_reference_panel": False,
        "thresholds_fitted_on_evaluation_labels": False,
        "template": {"path": str(args.template), "sha256": sha256(args.template),
                     "template_id": template.template_id, "pdb_id": template.pdb_id},
        "input_hashes": {
            "manifest": sha256(args.manifest), "splits": sha256(args.splits),
            "similarity_audit": sha256(args.similarity_audit),
            "esm2_baseline": sha256(args.esm2_baseline),
        },
        "population_sizes": {name: len(ids) for name, ids in populations.items()},
        "full_model_status_by_population": {
            name: dict(Counter(results[sid]["status"] for sid in ids))
            for name, ids in populations.items()
        },
        "evaluation": evaluation,
        "hard_negative_family_rates": family_rates,
        "status_counts": dict(Counter(row["status"] for row in results.values())),
        "comparator_sources": comparator_sources,
        "orthogonal_recovery": orthogonal_recovery,
        "per_example": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    log.info("Wrote B1 structural evaluation -> %s", args.out)


if __name__ == "__main__":
    main()
