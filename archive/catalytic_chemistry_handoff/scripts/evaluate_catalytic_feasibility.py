"""Evaluate Structural V3 on frozen challenge panels and mechanistic controls.

No template, threshold, or weight is learned from these panels.  The evaluator
scores every structure once, reports missingness explicitly, compares unique
recovery of mean-ESM2 misses, and repeats the evaluation after donor directions
are randomized while preserving donor-metal distances.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from catalytic_feasibility import candidate_donors, load_templates, score_catalytic_feasibility
from utils import PocketSubgraph, get_logger

log = get_logger(__name__)

FROZEN_GATE = {
    "minimum_combined_esm2_misses_recovered": 5,
    "minimum_all_negative_specificity": 0.95,
    "maximum_any_negative_family_fpr": 0.20,
    "minimum_overall_evaluable_fraction": 0.80,
    "maximum_scrambled_positive_support_relative_to_native": 0.50,
}


def label_is_positive(label: str) -> bool:
    return label == "positive"


def confusion(ids: list[str], scores: dict, labels: dict) -> dict:
    tp = fp = tn = fn = unavailable_positive = unavailable_negative = 0
    n_positive = n_negative = 0
    for sid in ids:
        positive = label_is_positive(labels[sid])
        if positive:
            n_positive += 1
        else:
            n_negative += 1
        status = scores[sid]["status"]
        if status == "unavailable":
            if positive:
                unavailable_positive += 1
            else:
                unavailable_negative += 1
            continue
        predicted = status == "supported"
        if positive and predicted:
            tp += 1
        elif positive:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    sensitivity_evaluable = tp / (tp + fn) if tp + fn else None
    specificity_evaluable = tn / (tn + fp) if tn + fp else None
    sensitivity_end_to_end = tp / n_positive if n_positive else None
    false_positive_fraction_all_negatives = fp / n_negative if n_negative else None
    unavailable = unavailable_positive + unavailable_negative
    return {
        "n": len(ids), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "unavailable": unavailable,
        "unavailable_positive": unavailable_positive,
        "unavailable_negative": unavailable_negative,
        "evaluable_fraction": (len(ids) - unavailable) / len(ids) if ids else None,
        "sensitivity_evaluable": sensitivity_evaluable,
        "specificity_evaluable": specificity_evaluable,
        "sensitivity_end_to_end": sensitivity_end_to_end,
        "false_positive_fraction_all_negatives": false_positive_fraction_all_negatives,
        "balanced_accuracy_evaluable": (
            (sensitivity_evaluable + specificity_evaluable) / 2
            if sensitivity_evaluable is not None and specificity_evaluable is not None else None
        ),
    }


def scramble_donor_directions(pocket: PocketSubgraph, rng: np.random.Generator) -> PocketSubgraph:
    """Destroy donor angles while preserving metal sites and bond lengths."""
    coords = pocket.coords.copy()
    for donor in candidate_donors(pocket):
        metal = pocket.metal_coords[donor.site_index]
        radius = float(np.linalg.norm(coords[donor.atom_index] - metal))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        coords[donor.atom_index] = metal + radius * direction
    return replace(pocket, coords=coords)


def family_false_positive_rates(
    ids: list[str], scores: dict, labels: dict, neg_families: dict,
) -> dict:
    by_family: dict[str, list[str]] = {}
    for sid in ids:
        if label_is_positive(labels[sid]):
            continue
        by_family.setdefault(neg_families.get(sid, "unknown") or "unknown", []).append(sid)
    output = {}
    for family, family_ids in sorted(by_family.items()):
        fp = sum(scores[sid]["status"] == "supported" for sid in family_ids)
        output[family] = {
            "n": len(family_ids), "false_positives": fp, "fpr": fp / len(family_ids),
            "unavailable": sum(scores[sid]["status"] == "unavailable" for sid in family_ids),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets-dir", required=True, type=Path)
    parser.add_argument("--templates-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--esm2-baseline", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    baseline = json.loads(args.esm2_baseline.read_text())
    catalog = {row["accession"]: row for row in csv.DictReader(args.catalog.open())}
    templates = load_templates(args.templates_dir)
    if not templates:
        raise RuntimeError(f"No reaction templates in {args.templates_dir}")

    panel_ids = {
        name: panel["test_ids"] for name, panel in splits["panels"].items()
    }
    all_ids = sorted(set().union(*map(set, panel_ids.values())))
    labels = {}
    scores = {}
    scrambled_scores = {}
    rng = np.random.default_rng(args.seed)
    for index, sid in enumerate(all_ids, start=1):
        pocket_path = args.pockets_dir / f"{sid}.npz"
        if not pocket_path.exists():
            raise FileNotFoundError(f"Frozen split ID missing pocket: {pocket_path}")
        pocket = PocketSubgraph.load(pocket_path)
        labels[sid] = pocket.metadata.label
        scores[sid] = score_catalytic_feasibility(pocket, templates)
        scrambled_scores[sid] = score_catalytic_feasibility(
            scramble_donor_directions(pocket, rng), templates
        )
        if index % 200 == 0:
            log.info("Scored %d/%d", index, len(all_ids))

    panels = {name: confusion(ids, scores, labels) for name, ids in panel_ids.items()}
    scrambled_panels = {
        name: confusion(ids, scrambled_scores, labels) for name, ids in panel_ids.items()
    }

    miss_details = []
    for panel_name in ("B1_B2_transfer", "B3_transfer"):
        examples = baseline[f"panel:{panel_name}"]["per_example"]
        for sid, prediction in examples.items():
            if prediction["true"] == "positive" and prediction["pred"] == "negative":
                miss_details.append({
                    "panel": panel_name,
                    "structure_id": sid,
                    "structural_status": scores[sid]["status"],
                    "best_supported_template": scores[sid]["best_supported_template"],
                })
    recovered = sum(row["structural_status"] == "supported" for row in miss_details)

    negative_ids = [sid for sid in all_ids if not label_is_positive(labels[sid])]
    positive_ids = [sid for sid in all_ids if label_is_positive(labels[sid])]
    overall = confusion(all_ids, scores, labels)
    all_negative = confusion(negative_ids, scores, labels)
    native_positive_support = sum(scores[sid]["status"] == "supported" for sid in positive_ids)
    scrambled_positive_support = sum(
        scrambled_scores[sid]["status"] == "supported" for sid in positive_ids
    )
    scrambled_relative = (
        scrambled_positive_support / native_positive_support if native_positive_support else None
    )
    family_fpr = family_false_positive_rates(
        all_ids, scores, labels,
        {sid: catalog.get(sid, {}).get("neg_family", "") for sid in all_ids},
    )
    maximum_family_fpr = max((entry["fpr"] for entry in family_fpr.values()), default=1.0)

    checks = {
        "esm2_miss_recovery": recovered >= FROZEN_GATE["minimum_combined_esm2_misses_recovered"],
        "negative_specificity": all_negative["specificity_evaluable"] >= FROZEN_GATE["minimum_all_negative_specificity"],
        "family_fpr": maximum_family_fpr <= FROZEN_GATE["maximum_any_negative_family_fpr"],
        "evaluable_fraction": overall["evaluable_fraction"] >= FROZEN_GATE["minimum_overall_evaluable_fraction"],
        "donor_scramble": (
            scrambled_relative is not None
            and scrambled_relative <= FROZEN_GATE["maximum_scrambled_positive_support_relative_to_native"]
        ),
    }
    gate_passed = all(checks.values())

    output = {
        "scoring_regime": "v3_catalytic_feasibility",
        "scientific_scope": {
            "uses_sequence": False,
            "uses_reference_protein_panel": False,
            "uses_labels_or_fitted_weights": False,
            "uses_experimental_reaction_state_templates": True,
            "claim": "reaction-state geometric compatibility screen, not proof of hydrolysis",
        },
        "frozen_gate": FROZEN_GATE,
        "gate_checks": checks,
        "gate_passed": gate_passed,
        "decision": "GO_integrate_as_separate_channel" if gate_passed else "NO_GO_do_not_integrate",
        "overall": overall,
        "panels": panels,
        "esm2_miss_recovery": {
            "n_esm2_misses": len(miss_details),
            "n_recovered": recovered,
            "recovery_fraction": recovered / len(miss_details) if miss_details else None,
            "per_example": miss_details,
        },
        "hard_negative_robustness": {
            "all_negative": all_negative,
            "by_family": family_fpr,
            "maximum_family_fpr": maximum_family_fpr,
        },
        "donor_direction_scramble_control": {
            "seed": args.seed,
            "native_positive_support": native_positive_support,
            "scrambled_positive_support": scrambled_positive_support,
            "scrambled_relative_to_native": scrambled_relative,
            "panels": scrambled_panels,
        },
        "status_counts": dict(Counter(score["status"] for score in scores.values())),
        "per_example": {
            sid: {
                "label": labels[sid],
                "status": scores[sid]["status"],
                "best_supported_template": scores[sid]["best_supported_template"],
                "n_evaluable": scores[sid]["n_evaluable"],
                "n_supported": scores[sid]["n_supported"],
                "scrambled_status": scrambled_scores[sid]["status"],
            }
            for sid in all_ids
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    log.info("Decision: %s", output["decision"])
    log.info("Wrote evaluation -> %s", args.out)


if __name__ == "__main__":
    main()
