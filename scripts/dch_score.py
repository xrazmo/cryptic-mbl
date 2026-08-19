"""
dch_score.py

DCH (Asp-Cys-His) structural-support channel for production scoring:
does this candidate have a cysteine sulfur coordinating one of its
predicted metal sites, the way canonical B1/B2 MBLs do? This is the
mechanistic rule found in the coordination-fingerprint work
(reports/coordination_fingerprint_findings.md) -- 0.948 sensitivity /
0.983 specificity on the B1_B2_transfer held-out panel, operating on
the corrected data/pockets_v2 (post metal-site-corruption fix).

Three-state, not binary -- "no metal predicted" and "metal predicted,
no Cys nearby" are different things and must not be collapsed:
  - "unavailable": no metal site was predicted for this candidate at
    all (Metal3D found nothing above threshold, or cavity_fallback).
    This is a missing-input state, not evidence against the candidate.
  - "not_supported": at least one metal site exists, but no Cys SG atom
    is within DCH_SULFUR_DISTANCE (2.8A) of any accepted site.
  - "supported": a Cys SG atom is within range of at least one site.

CLI:
    python dch_score.py --pockets-dir data/pockets_v2 --out data/dch_scores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

DCH_SULFUR_DISTANCE = 2.8  # Angstrom, same threshold as coordination_fingerprint.py's DONOR_SHELL_RADIUS


def score_dch(pocket: PocketSubgraph) -> dict:
    if pocket.metal_coords is None or len(pocket.metal_coords) == 0:
        return {
            "status": "unavailable", "min_sulfur_metal_distance": None,
            "responsible_site_index": None, "site_probability": None,
        }

    best_dist, best_site = None, None
    for i in range(len(pocket.res_ids)):
        if pocket.res_names[i] != "CYS" or pocket.atom_names[i] != "SG":
            continue
        dists = np.linalg.norm(pocket.metal_coords - pocket.coords[i][None, :], axis=1)
        site_idx = int(np.argmin(dists))
        d = float(dists[site_idx])
        if best_dist is None or d < best_dist:
            best_dist, best_site = d, site_idx

    if best_dist is None:
        return {
            "status": "not_supported", "min_sulfur_metal_distance": None,
            "responsible_site_index": None, "site_probability": None,
        }

    status = "supported" if best_dist <= DCH_SULFUR_DISTANCE else "not_supported"
    site_prob = float(pocket.metal_probabilities[best_site]) if pocket.metal_probabilities is not None else None
    return {
        "status": status, "min_sulfur_metal_distance": round(best_dist, 3),
        "responsible_site_index": best_site, "site_probability": site_prob,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    results = {}
    status_counts = {"supported": 0, "not_supported": 0, "unavailable": 0}
    for f in sorted(args.pockets_dir.glob("*.npz")):
        pocket = PocketSubgraph.load(f)
        sid = pocket.metadata.source_structure_id
        r = score_dch(pocket)
        results[sid] = r
        status_counts[r["status"]] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"status_counts": status_counts, "scores": results}, indent=2))
    log.info(f"DCH status counts: {status_counts}")
    log.info(f"Wrote DCH scores -> {args.out}")


if __name__ == "__main__":
    main()
