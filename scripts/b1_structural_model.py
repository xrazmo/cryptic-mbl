"""Sequence-blind B1 catalytic-architecture detector.

This is deliberately not a family-neighbour model and not a learned GNN.
It tests whether a candidate structure can support the complete local B1
reaction architecture observed in the experimental NDM-1/hydrolysed-
meropenem complex (PDB 4EYL): a dinuclear site, the site-resolved 3N and
N/O/S donor arrangement (including the DCH cysteine), the donor/metal
geometry, and a transferred carbapenem-product pose without gross clashes.

The cysteine-only DCH result is retained as an explicit partial-evidence
ablation.  It never substitutes for the full positive call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalytic_feasibility import ReactionTemplate, score_template
from dch_score import score_dch
from utils import PocketSubgraph, get_logger

log = get_logger(__name__)

B1_TEMPLATE_ID = "B1_NDM1_hydrolyzed_meropenem_4EYL"


def load_b1_template(path: Path) -> ReactionTemplate:
    template = ReactionTemplate.load(path)
    if template.template_id != B1_TEMPLATE_ID or template.subclass != "B1":
        raise ValueError(f"Expected {B1_TEMPLATE_ID}, got {template.template_id}")
    if len(template.metal_coords) != 2:
        raise ValueError("B1 catalytic template must be dinuclear")
    return template


def score_b1_structure(pocket: PocketSubgraph, template: ReactionTemplate) -> dict:
    dch = score_dch(pocket)
    full = score_template(pocket, template)
    if full["status"] == "supported":
        status = "supported"
        reason = "complete_B1_catalytic_architecture_and_product_pose"
    elif dch["status"] == "supported":
        status = "partial_support"
        reason = "coordinating_cysteine_without_complete_B1_architecture"
    elif pocket.metal_coords is None or len(pocket.metal_coords) == 0:
        status = "unavailable"
        reason = "no_predicted_metal_site"
    elif len(pocket.metal_coords) != 2:
        status = "unavailable"
        reason = "dinuclear_geometry_not_available"
    else:
        status = "not_supported"
        reason = "complete_B1_architecture_not_observed"
    return {
        "status": status,
        "positive_call": status == "supported",
        "reason": reason,
        "dch_partial_evidence": dch,
        "full_architecture": full,
        "scientific_scope": (
            "structural support for a canonical dinuclear B1-like carbapenem reaction "
            "architecture; not proof of expression, turnover, or resistance"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pocket = PocketSubgraph.load(args.pocket)
    output = {
        "structure_id": pocket.metadata.source_structure_id,
        "model": "b1_catalytic_architecture_v1",
        "uses_sequence": False,
        "uses_labeled_reference_panel": False,
        "result": score_b1_structure(pocket, load_b1_template(args.template)),
    }
    text = json.dumps(output, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote B1 structural score -> %s", args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
