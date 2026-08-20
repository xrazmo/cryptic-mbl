"""
pocket_extraction.py — Task 1 (spec §3)

For a single structure, produce a PocketSubgraph:
  1. Predict/confirm metal site with Metal3D (primary) + PinMyMetal (cross-check).
  2. If Metal3D confidence is low/absent, fall back to fpocket cavity detection
     and flag pocket_source="cavity_fallback".
  3. Extract all residues with any atom within [radius_min, radius_max] Å of
     the metal ion (or cavity centroid).
  4. QC filter on mean pocket pLDDT — downweight, don't hard-exclude.

Metal3D / PinMyMetal / fpocket are treated as external tools invoked via
their own CLIs or Python APIs (all three ship as separate installs with
their own model weights / dependencies, so they are wrapped rather than
reimplemented here). Swap the `run_metal3d` / `run_pinmymetal` / `run_fpocket`
bodies for the actual calls once those tools are installed in the target
environment — the function signatures and return contracts are the stable
interface the rest of the pipeline depends on.

CLI:
    python pocket_extraction.py \
        --structure path/to/structure.pdb \
        --structure-id NDM-1_ref \
        --label positive --tier 1 --subclass B1 \
        --out-dir data/pockets/ \
        --is-predicted false
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from utils import (
    PocketMetadata,
    PocketSubgraph,
    get_logger,
    load_structure,
    load_probe_structure,
    get_per_residue_confidence,
    residue_centroids,
    min_distance_to_point,
)

log = get_logger(__name__)

RADIUS_MIN_DEFAULT = 8.0
RADIUS_MAX_DEFAULT = 12.0
PLDDT_QC_THRESHOLD = 70.0
METAL3D_CONFIDENCE_THRESHOLD = 0.5  # below this, use cavity fallback
SITE_CLUSTER_RADIUS = 5.0  # Angstrom: a second probe within this of the top site is kept as a candidate dinuclear partner
MAX_METAL_SITES = 2  # mononuclear or dinuclear; never keep more


# --------------------------------------------------------------------------- #
# External tool wrappers — replace bodies with real Metal3D / PinMyMetal /
# fpocket calls. Kept as isolated functions so integration is a one-file edit.
# --------------------------------------------------------------------------- #

def run_metal3d(structure_path: Path) -> Optional[list[tuple[np.ndarray, float]]]:
    """Call Metal3D CLI to predict metal binding sites.

    Uses conda run to execute Metal3D in the metal3d environment
    while staying in the cryptic-mbl environment.

    Returns every candidate site Metal3D's clustering (find_unique_sites in
    the vendored Metal3D/utils/helpers.py) found, as (coord, probability) --
    NOT a single averaged point. --maxp was previously passed here on the
    (incorrect) assumption it would restrict --probefile to the top site;
    it does not -- it triggers a completely separate maxprobability() call
    that writes its own file, which this code never read. --probefile
    always contains every clustered site regardless of --maxp, so the old
    `metal_pdb.coord.mean(axis=0)` was averaging every candidate site
    (including low-probability noise clusters) into one point -- on a
    multichain crystal structure this can land tens of Angstroms from any
    real metal (verified directly: NDM-1/3SPU, 30 probe sites, two within
    <1A of the real dinuclear Zn pair, averaged centroid ~24A from both).
    Site selection now happens explicitly in select_metal_sites().
    """
    import subprocess
    import tempfile

    metal3d_wrapper = (Path(__file__).parent.parent / 'run_metal3d.sh').resolve()

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
        tmp_output = tmp.name

    try:
        result = subprocess.run(
            [
                metal3d_wrapper,
                '--pdb', str(structure_path.absolute()),
                '--metalbinding',
                '--writeprobes',
                '--probefile', tmp_output,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            log.warning(f"Metal3D failed: {result.stderr}")
            return None

        metal_pdb = load_probe_structure(tmp_output)
        if metal_pdb.array_length() == 0:
            return None

        # find_unique_sites() writes each site's max cluster probability into
        # the occupancy column (see assets/.../helpers.py's HETATM line format).
        probabilities = getattr(metal_pdb, "occupancy", None)
        if probabilities is None:
            log.warning(f"{structure_path.name}: probe PDB has no occupancy field, defaulting all site probabilities to 1.0")
            probabilities = np.ones(metal_pdb.array_length())
        sites = [(metal_pdb.coord[i].copy(), float(probabilities[i])) for i in range(metal_pdb.array_length())]
        return sites

    except subprocess.TimeoutExpired:
        log.warning("Metal3D timed out after 300s")
        return None
    except Exception as e:
        log.warning(f"Metal3D failed: {e}")
        return None
    finally:
        try:
            Path(tmp_output).unlink()
        except:
            pass


def select_metal_sites(
    sites: list[tuple[np.ndarray, float]], cluster_radius: float = SITE_CLUSTER_RADIUS,
) -> list[tuple[np.ndarray, float]]:
    """
    Deterministic primary-site selection, no label/subclass information used:
      1. Highest-probability probe = primary site.
      2. The next-highest-probability probe is also kept, IF it's within
         cluster_radius of the primary (candidate second metal of a
         dinuclear site) -- otherwise the primary is the only site kept.
      3. Never more than MAX_METAL_SITES (2) sites survive.
    A real dinuclear B1/B3 site's two Zn ions sit ~3.5-4A apart (verified
    against this project's own reference bank: NDM-1 3.8A, L1/FEZ-1
    similar); cluster_radius=5A gives headroom without accepting an
    unrelated, more distant probe as a second site.
    """
    if not sites:
        return []
    ranked = sorted(sites, key=lambda s: s[1], reverse=True)
    accepted = [ranked[0]]
    for coord, prob in ranked[1:]:
        if len(accepted) >= MAX_METAL_SITES:
            break
        if np.linalg.norm(coord - accepted[0][0]) <= cluster_radius:
            accepted.append((coord, prob))
    return accepted


def run_pinmymetal(structure_path: Path) -> Optional[dict]:
    """
    PinMyMetal Docker image available online is environment-only (code must be
    mounted separately). Since Metal3D provides robust metal-site prediction,
    we skip PinMyMetal for this pipeline.
    
    If needed later, PinMyMetal can be run via: https://PMM.biocloud.top (web server)
    """
    log.debug("PinMyMetal skipped — Metal3D provides sufficient metal-site prediction")
    return None


def run_fpocket(structure_path: Path) -> Optional[dict]:
    """
    Cavity-detection fallback when Metal3D confidence is low/absent.
    Returns {"centroid": np.ndarray(3,), "score": float} for the top-ranked
    pocket, or None if fpocket finds nothing druggable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local_copy = tmp / structure_path.name
        local_copy.write_bytes(structure_path.read_bytes())
        try:
            subprocess.run(
                ["fpocket", "-f", str(local_copy)],
                check=True, capture_output=True, timeout=300,
            )
        except FileNotFoundError:
            log.error("fpocket not found on PATH — install it or stub this call.")
            return None
        except subprocess.CalledProcessError as e:
            log.warning(f"fpocket failed on {structure_path.name}: {e.stderr}")
            return None

        out_dir = tmp / f"{local_copy.stem}_out"
        info_file = out_dir / f"{local_copy.stem}_info.txt"
        pockets_pdb = out_dir / "pockets" / "pocket1_atm.pdb"
        if not pockets_pdb.exists():
            return None

        pocket_arr = load_structure(pockets_pdb)
        centroid = pocket_arr.coord.mean(axis=0)
        score = _parse_fpocket_top_score(info_file)
        return {"centroid": centroid, "score": score}


def _parse_fpocket_top_score(info_file: Path) -> float:
    if not info_file.exists():
        return float("nan")
    for line in info_file.read_text().splitlines():
        if "Pocket Score" in line:
            return float(line.split(":")[-1].strip())
    return float("nan")


# --------------------------------------------------------------------------- #
# Core extraction logic
# --------------------------------------------------------------------------- #
def determine_pocket_sites(
    structure_path: Path,
) -> tuple[list[tuple[np.ndarray, float]], str]:
    """
    Returns (accepted_sites, pocket_source). accepted_sites is a list of
    1-2 (coord, probability) tuples for "metal3d", or a single
    (centroid, confidence=nan) for "cavity_fallback".
    Tries Metal3D first; falls back to fpocket cavity detection.
    PinMyMetal is skipped (Metal3D + fpocket are sufficient).
    """
    try:
        raw_sites = run_metal3d(structure_path)
    except NotImplementedError:
        log.warning("Metal3D wrapper not implemented — using fpocket fallback directly.")
        raw_sites = None

    if raw_sites:
        accepted = select_metal_sites(raw_sites)
        if accepted and accepted[0][1] >= METAL3D_CONFIDENCE_THRESHOLD:
            return accepted, "metal3d"

    log.info(f"{structure_path.name}: low/no Metal3D confidence, falling back to fpocket.")
    fpocket_result = run_fpocket(structure_path)
    if fpocket_result is None:
        raise RuntimeError(
            f"{structure_path.name}: neither Metal3D nor fpocket produced a "
            "usable pocket center. Exclude this structure or inspect manually."
        )
    return [(fpocket_result["centroid"], float("nan"))], "cavity_fallback"

def extract_pocket(
    structure_path: Path,
    structure_id: str,
    label: str,
    confidence_tier: int,
    subclass: Optional[str],
    is_predicted: bool,
    radius_min: float = RADIUS_MIN_DEFAULT,
    radius_max: float = RADIUS_MAX_DEFAULT,
) -> PocketSubgraph:
    arr = load_structure(structure_path)
    accepted_sites, pocket_source = determine_pocket_sites(structure_path)
    metal_confidence = accepted_sites[0][1] if pocket_source == "metal3d" else None

    res_ids, centroids = residue_centroids(arr)
    # A residue qualifies if it's within radius_max of ANY accepted site
    # (both sites of a dinuclear pair get their neighborhoods included, not
    # just the top-probability one).
    dists = np.min(
        np.stack([min_distance_to_point(centroids, coord) for coord, _ in accepted_sites]), axis=0,
    )
    # Use the wider radius as the inclusion boundary; radius_min is reserved
    # for future graded-weighting (e.g. core vs. periphery residues) rather
    # than a second hard cutoff.
    keep_res_ids = set(res_ids[dists <= radius_max].tolist())

    atom_mask = np.array([rid in keep_res_ids for rid in arr.res_id])
    pocket_arr = arr[atom_mask]

    if pocket_arr.array_length() == 0:
        raise RuntimeError(f"{structure_id}: zero atoms within {radius_max} Å of pocket center.")

    mean_plddt = None
    if is_predicted:
        conf = get_per_residue_confidence(pocket_arr)
        if not np.all(np.isnan(conf)):
            mean_plddt = float(np.nanmean(conf))
            if mean_plddt < PLDDT_QC_THRESHOLD:
                log.warning(
                    f"{structure_id}: mean pocket pLDDT {mean_plddt:.1f} < "
                    f"{PLDDT_QC_THRESHOLD} — downweight in loss, not excluding."
                )

    is_sidechain = np.array([
        name not in ("N", "CA", "C", "O") for name in pocket_arr.atom_name
    ])

    metadata = PocketMetadata(
        source_structure_id=structure_id,
        label=label,
        confidence_tier=confidence_tier,
        pocket_source=pocket_source,
        metal_confidence=metal_confidence,
        mean_pocket_plddt=mean_plddt,
        subclass=subclass,
    )

    if pocket_source == "metal3d":
        metal_coords = np.stack([coord for coord, _ in accepted_sites])
        metal_probabilities = np.array([prob for _, prob in accepted_sites])
    else:
        metal_coords, metal_probabilities = None, None

    return PocketSubgraph(
        res_ids=pocket_arr.res_id,
        res_names=pocket_arr.res_name,
        coords=pocket_arr.coord,
        atom_names=pocket_arr.atom_name,
        elements=pocket_arr.element,
        is_sidechain=is_sidechain,
        metal_coords=metal_coords,
        metal_probabilities=metal_probabilities,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def main():
    p = argparse.ArgumentParser(description="Extract a pocket subgraph from one structure.")
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--structure-id", required=True)
    p.add_argument("--label", required=True,
                    choices=["positive", "hard_negative", "easy_negative", "unlabeled"])
    p.add_argument("--tier", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--subclass", default=None, choices=["B1", "B2", "B3", "environmental", None])
    p.add_argument("--is-predicted", type=_bool, default=True,
                    help="true for AF/ESMFold models (enables pLDDT QC), false for crystal structures.")
    p.add_argument("--radius-min", type=float, default=RADIUS_MIN_DEFAULT)
    p.add_argument("--radius-max", type=float, default=RADIUS_MAX_DEFAULT)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pocket = extract_pocket(
        structure_path=args.structure,
        structure_id=args.structure_id,
        label=args.label,
        confidence_tier=args.tier,
        subclass=args.subclass,
        is_predicted=args.is_predicted,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
    )
    out_path = args.out_dir / f"{args.structure_id}.npz"
    pocket.save(out_path)
    log.info(f"Saved pocket subgraph -> {out_path}")


if __name__ == "__main__":
    main()
