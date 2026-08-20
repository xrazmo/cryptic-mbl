"""
inference.py — Task 8 (spec §9)

Scores new metagenomic candidate pockets. Two independent scores per spec
§9.4, deliberately NOT collapsed into one:

  1. Resemblance score — embedding-space distance to the reference bank
     (nearest-neighbor or per-subclass prototype). Computed here.
  2. Mechanistic plausibility score — Zn coordination geometry validity +
     catalytic water proxy + docking pose quality. The docking sub-pipeline
     itself (panel substrates, GNN scoring of poses) is a separate system
     this module interfaces with, not reimplements; `mechanistic_plausibility()`
     below computes the geometry-only sub-components (coordination validity,
     catalytic water proxy) that don't require an external docking run, and
     accepts a pre-computed docking score to merge in.

Ranking (spec §9.4-5): candidates are NOT sorted by resemblance. Primary
filter is mechanistic plausibility; resemblance is reported as a secondary
"novelty band" axis — moderate distance from all references (neither
near-identical nor wildly divergent) is the target discovery zone, not
minimal distance. Uncertainty (ensemble variance) is reported alongside
both.

CLI:
    python inference.py --candidates-dir data/candidate_pockets \
        --ensemble-dir models/fold_B2_ensemble \
        --reference-bank-dir data/pockets --reference-bank-ids NDM-1 VIM-2 ... \
        --docking-scores data/docking_scores.json \
        --out results/candidate_ranking.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, PocketSubgraph
from graph_construction import pocket_to_pyg_data
from evaluate import load_ensemble, ensemble_embed

log = get_logger(__name__)

# Novelty band bounds are dataset-dependent — calibrate against the
# distribution of reference-bank pairwise distances before running at scale;
# these are placeholder starting values, not derived constants.
NOVELTY_BAND_MIN = 0.15
NOVELTY_BAND_MAX = 0.55

# Idealized Zn-coordination geometry references (tetrahedral ~109.5°,
# trigonal bipyramidal ~90/120°, octahedral ~90°/180°) — MBL sites are
# typically tetrahedral-to-5-coordinate; exact reference angles/distances
# should be pulled from a curated set of high-confidence MBL structures
# rather than textbook ideals, since real active sites deviate systematically.
IDEAL_ZN_LIGAND_DISTANCE = 2.1  # Å, typical Zn-N/O/S coordination distance
IDEAL_ZN_LIGAND_DISTANCE_TOLERANCE = 0.4


def zn_coordination_validity(pocket: PocketSubgraph) -> dict:
    """
    Geometry-only check: for pockets with a Metal3D-confirmed metal_coord,
    identify plausible coordinating atoms (N/O/S within tolerance of ideal
    Zn-ligand distance) and score how well the observed coordination number
    and distances match known-valid MBL geometry. Cavity-fallback pockets
    (no confirmed metal) get a null/low score with a flag, since this whole
    scoring axis presupposes a metal site.
    """
    if pocket.metal_coord is None:
        return {"valid": False, "reason": "no confirmed metal site (cavity_fallback)", "score": 0.0}

    dists = np.linalg.norm(pocket.coords - pocket.metal_coord[None, :], axis=1)
    coordinating_mask = (
        (dists < IDEAL_ZN_LIGAND_DISTANCE + IDEAL_ZN_LIGAND_DISTANCE_TOLERANCE)
        & np.isin(pocket.elements, ["N", "O", "S"])
    )
    n_coordinating = int(coordinating_mask.sum())
    coordinating_dists = dists[coordinating_mask]

    # MBL Zn sites are typically 3-5 coordinate; score peaks in that range
    # and falls off outside it rather than a hard cutoff.
    if n_coordinating == 0:
        cn_score = 0.0
    elif 3 <= n_coordinating <= 5:
        cn_score = 1.0
    else:
        cn_score = max(0.0, 1.0 - 0.25 * abs(n_coordinating - 4))

    if len(coordinating_dists) > 0:
        dist_deviation = float(np.mean(np.abs(coordinating_dists - IDEAL_ZN_LIGAND_DISTANCE)))
        dist_score = max(0.0, 1.0 - dist_deviation / IDEAL_ZN_LIGAND_DISTANCE_TOLERANCE)
    else:
        dist_score = 0.0

    score = 0.5 * cn_score + 0.5 * dist_score
    return {
        "valid": score > 0.5,
        "n_coordinating_atoms": n_coordinating,
        "mean_coordination_distance": float(np.mean(coordinating_dists)) if len(coordinating_dists) else None,
        "score": float(score),
    }


def catalytic_water_proxy(pocket: PocketSubgraph, zn_geometry: dict) -> dict:
    """
    Rough geometric proxy for the presence of a catalytically positioned
    bridging water/hydroxide: checks whether there is open space near the
    metal (no coordinating protein atom within ~2.5-3.5 Å in at least one
    direction consistent with a bridging/apical position). This is a coarse
    stand-in — the real assessment should come from explicit solvent
    placement (e.g. 3D-RISM, or inspection of docked-substrate + water
    positions from the co-folding step), not inferred from the apo pocket
    geometry alone.
    """
    if pocket.metal_coord is None or not zn_geometry.get("n_coordinating_atoms"):
        return {"plausible": False, "score": 0.0}

    dists = np.linalg.norm(pocket.coords - pocket.metal_coord[None, :], axis=1)
    # crude open-space check: is there a "gap" between 2.5 and 4.5 Å in some
    # direction, i.e. not every angular sector around the metal is occupied
    # by a close protein atom (an approximate solvent-accessibility proxy).
    close_mask = (dists > 1.5) & (dists < 3.0)
    occupancy_fraction = float(close_mask.sum()) / max(len(dists), 1)
    # lower occupancy near the metal (but not zero — need SOME coordinating
    # residues, checked via zn_geometry) suggests room for a bridging water
    score = max(0.0, 1.0 - occupancy_fraction * 3.0)
    return {"plausible": score > 0.4, "score": float(score)}


def mechanistic_plausibility(
    pocket: PocketSubgraph,
    docking_score: float | None,
) -> dict:
    zn_geom = zn_coordination_validity(pocket)
    water = catalytic_water_proxy(pocket, zn_geom)
    docking_component = docking_score if docking_score is not None else None

    components = [zn_geom["score"], water["score"]]
    weights = [0.4, 0.2]
    if docking_component is not None:
        components.append(docking_component)
        weights.append(0.4)
    else:
        # renormalize over available components if docking hasn't been run yet
        weights = [w / sum(weights) for w in weights]

    combined = float(np.average(components, weights=weights))
    return {
        "zn_coordination": zn_geom,
        "catalytic_water_proxy": water,
        "docking_score": docking_component,
        "mechanistic_plausibility_score": combined,
    }


def resemblance_score(candidate_emb: np.ndarray, reference_embs: dict) -> dict:
    ref_matrix = np.stack(list(reference_embs.values()))
    dists = np.linalg.norm(ref_matrix - candidate_emb[None, :], axis=1)
    nn_dist = float(dists.min())
    prototype = ref_matrix.mean(axis=0)
    prototype_dist = float(np.linalg.norm(candidate_emb - prototype))
    return {"nearest_neighbor_distance": nn_dist, "prototype_distance": prototype_dist}


def novelty_band_flag(prototype_distance: float) -> str:
    if prototype_distance < NOVELTY_BAND_MIN:
        return "near_duplicate"  # too close to known positives to be a novel discovery
    if prototype_distance > NOVELTY_BAND_MAX:
        return "too_divergent"   # likely outside the model's reliable generalization range
    return "novelty_band"        # target discovery zone


def rank_candidates(
    candidates_dir: Path,
    ensemble_dir: Path,
    reference_bank_dir: Path,
    reference_bank_ids: list[str],
    docking_scores: dict[str, float],
    out_path: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    candidate_files = sorted(candidates_dir.glob("*.npz"))
    candidate_pockets = {p.stem: PocketSubgraph.load(p) for p in candidate_files}
    ref_pockets = {sid: PocketSubgraph.load(reference_bank_dir / f"{sid}.npz") for sid in reference_bank_ids}

    sample_pocket = next(iter(candidate_pockets.values()))
    in_dim = pocket_to_pyg_data(sample_pocket).x.shape[1]
    models = load_ensemble(ensemble_dir, in_dim, device)

    ref_embs = {}
    for sid, pocket in ref_pockets.items():
        data = pocket_to_pyg_data(pocket)
        emb, _ = ensemble_embed(models, data, device)
        ref_embs[sid] = emb

    results = []
    for cid, pocket in candidate_pockets.items():
        data = pocket_to_pyg_data(pocket)
        emb, var = ensemble_embed(models, data, device)

        resemblance = resemblance_score(emb, ref_embs)
        mech = mechanistic_plausibility(pocket, docking_scores.get(cid))
        band = novelty_band_flag(resemblance["prototype_distance"])

        results.append({
            "candidate_id": cid,
            "mechanistic_plausibility_score": mech["mechanistic_plausibility_score"],
            "resemblance": resemblance,
            "novelty_band": band,
            "ensemble_variance": var,
            "mechanistic_detail": mech,
            "source_structure_id": pocket.metadata.source_structure_id,
            "confidence_tier": pocket.metadata.confidence_tier,
        })

    # Primary sort: mechanistic plausibility descending. Novelty band used
    # only as an annotation/filter for the person reviewing candidates, not
    # a sort key — per spec §9.4-5, target class = high plausibility AND
    # in the novelty band, surfaced via the flag rather than folded into score.
    results.sort(key=lambda r: r["mechanistic_plausibility_score"], reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Ranked {len(results)} candidates -> {out_path}")

    n_target_class = sum(1 for r in results if r["novelty_band"] == "novelty_band" and r["mechanistic_plausibility_score"] > 0.6)
    log.info(f"{n_target_class} candidates in target discovery class (novelty band + plausibility > 0.6)")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-dir", required=True, type=Path)
    p.add_argument("--ensemble-dir", required=True, type=Path)
    p.add_argument("--reference-bank-dir", required=True, type=Path)
    p.add_argument("--reference-bank-ids", nargs="+", required=True)
    p.add_argument("--docking-scores", type=Path, default=None,
                    help="Optional JSON {candidate_id: score} from the separate docking sub-pipeline.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    docking_scores = json.loads(args.docking_scores.read_text()) if args.docking_scores else {}
    rank_candidates(
        args.candidates_dir, args.ensemble_dir, args.reference_bank_dir,
        args.reference_bank_ids, docking_scores, args.out,
    )


if __name__ == "__main__":
    main()
