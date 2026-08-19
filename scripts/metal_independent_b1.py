"""Metal3D-independent search for the canonical B1 catalytic architecture.

The existing B1 scorer aligns an experimental reaction template to predicted
metal coordinates.  That is highly specific, but it fails whenever Metal3D
misses one site (including known VIM-2) or places a site too far from a donor.

This module instead enumerates the two donor triads directly in a complete
single-chain structure:

* Zn1-like site: three distinct histidine N donors;
* Zn2-like site: Asp O, Cys S, and His N donors.

No sequence order, residue numbering, motif regular expression, ESM embedding,
label, or protein-reference panel is used. Candidate triads are pruned by the
experimental within-site distance fingerprints, aligned to the 4EYL B1
reaction-state template, and subjected to the same transferred-product clash
and pocket-contact gates as the metal-anchored scorer. The template metals are
transferred with the fitted donor frame, so an explicit metal prediction is
not required.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from b1_structural_model import load_b1_template
from catalytic_feasibility import (
    DEFAULT_MAX_HARD_CLASH_FRACTION,
    DEFAULT_MAX_PHARMACOPHORE_RMSD,
    DEFAULT_MIN_POCKET_CONTACT_FRACTION,
    HARD_CLASH_DISTANCE,
    POCKET_CONTACT_DISTANCE,
    ReactionTemplate,
    apply_transform,
    kabsch_transform,
)
from utils import get_logger, load_structure

log = get_logger(__name__)

PAIR_DISTANCE_TOLERANCE = 1.50
SITE_CENTROID_DISTANCE_TOLERANCE = 2.50
MAX_FULL_ALIGNMENTS = 250_000

# Oxidized cysteines are commonly present in crystallographic B1 structures.
# Their SG position still reports the conserved B1 donor architecture, but the
# modified side chain is not evidence for an intact catalytic thiolate.  Keep
# these residues for architecture matching and expose the distinction in the
# result instead of silently dropping or normalizing it.
MODIFIED_CYSTEINE_RESNAMES = frozenset({"CSD", "CSO", "OCS"})


@dataclass(frozen=True)
class DonorAtom:
    coord: np.ndarray
    resname: str
    res_id: int
    atom_name: str
    atom_index: int
    chain_id: str
    ins_code: str
    modified_from_cysteine: bool = False

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return self.chain_id, self.res_id, self.ins_code

    @property
    def label(self) -> str:
        insertion = self.ins_code.strip()
        residue_number = f"{self.res_id}{insertion}" if insertion else str(self.res_id)
        return f"{self.chain_id}:{self.resname}:{residue_number}:{self.atom_name}"


def extract_donors(structure_path: Path) -> tuple[object, dict[str, list[DonorAtom]]]:
    arr = load_structure(structure_path)
    # Preserve only protein atoms plus explicitly recognized modified
    # cysteines.  This excludes metals, waters, and crystallization ligands
    # from both donor enumeration and protein-ligand clash calculations.
    modified_cysteine = np.isin(arr.res_name, list(MODIFIED_CYSTEINE_RESNAMES))
    arr = arr[(~arr.hetero) | modified_cysteine]
    if len(set(arr.chain_id.tolist())) != 1:
        raise ValueError(f"{structure_path}: expected a single-chain structure")
    roles = {"HIS_N": [], "ASP_O": [], "CYS_S": []}
    for index, (resname, atom_name) in enumerate(zip(arr.res_name, arr.atom_name)):
        role = None
        if resname == "HIS" and atom_name in {"ND1", "NE2"}:
            role = "HIS_N"
        elif resname == "ASP" and atom_name in {"OD1", "OD2"}:
            role = "ASP_O"
        elif resname in ({"CYS"} | MODIFIED_CYSTEINE_RESNAMES) and atom_name == "SG":
            role = "CYS_S"
        if role:
            roles[role].append(DonorAtom(
                coord=np.asarray(arr.coord[index], dtype=float),
                resname=str(resname), res_id=int(arr.res_id[index]),
                atom_name=str(atom_name), atom_index=index,
                chain_id=str(arr.chain_id[index]), ins_code=str(arr.ins_code[index]),
                modified_from_cysteine=str(resname) in MODIFIED_CYSTEINE_RESNAMES,
            ))
    return arr, roles


def _pair_distances(coords: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(coords[i] - coords[j])
        for i in range(len(coords)) for j in range(i + 1, len(coords))
    ])


def enumerate_dch_triads(
    roles: dict[str, list[DonorAtom]], template_coords: np.ndarray,
    pair_distance_tolerance: float = PAIR_DISTANCE_TOLERANCE,
) -> list[tuple[DonorAtom, DonorAtom, DonorAtom]]:
    """Return triads ordered exactly as template O, S, N donors."""
    expected = _pair_distances(template_coords)
    output = []
    for oxygen in roles["ASP_O"]:
        for sulfur in roles["CYS_S"]:
            if np.linalg.norm(oxygen.coord - sulfur.coord) > expected[0] + pair_distance_tolerance:
                continue
            for nitrogen in roles["HIS_N"]:
                if len({oxygen.residue_key, sulfur.residue_key, nitrogen.residue_key}) != 3:
                    continue
                coords = np.array([oxygen.coord, sulfur.coord, nitrogen.coord])
                actual = _pair_distances(coords)
                if np.max(np.abs(actual - expected)) <= pair_distance_tolerance:
                    output.append((oxygen, sulfur, nitrogen))
    return output


def enumerate_three_his_triads(
    roles: dict[str, list[DonorAtom]], template_coords: np.ndarray,
    pair_distance_tolerance: float = PAIR_DISTANCE_TOLERANCE,
) -> list[tuple[DonorAtom, DonorAtom, DonorAtom]]:
    expected = _pair_distances(template_coords)
    output = []
    atoms = roles["HIS_N"]
    for triad in itertools.combinations(atoms, 3):
        if len({atom.residue_key for atom in triad}) != 3:
            continue
        coords = np.array([atom.coord for atom in triad])
        if np.max(np.abs(
            np.sort(_pair_distances(coords)) - np.sort(expected)
        )) <= pair_distance_tolerance:
            output.append(triad)
    return output


def _pose_metrics(
    ligand_coords: np.ndarray, ligand_elements: np.ndarray, protein_arr,
) -> dict:
    protein_mask = ~np.isin(np.char.upper(protein_arr.element.astype(str)), ["H", "D"])
    protein_coords = protein_arr.coord[protein_mask]
    ligand_mask = ~np.isin(np.char.upper(ligand_elements.astype(str)), ["H", "D"])
    ligand = ligand_coords[ligand_mask]
    distances = np.linalg.norm(ligand[:, None, :] - protein_coords[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    return {
        "hard_clash_fraction": float(np.mean(nearest < HARD_CLASH_DISTANCE)),
        "pocket_contact_fraction": float(np.mean(nearest <= POCKET_CONTACT_DISTANCE)),
        "minimum_protein_ligand_distance": float(nearest.min()),
    }


def score_donor_roles(
    protein,
    roles: dict[str, list[DonorAtom]],
    template: ReactionTemplate,
    max_pharmacophore_rmsd: float = DEFAULT_MAX_PHARMACOPHORE_RMSD,
    max_hard_clash_fraction: float = DEFAULT_MAX_HARD_CLASH_FRACTION,
    min_pocket_contact_fraction: float = DEFAULT_MIN_POCKET_CONTACT_FRACTION,
    pair_distance_tolerance: float = PAIR_DISTANCE_TOLERANCE,
    site_centroid_distance_tolerance: float = SITE_CENTROID_DISTANCE_TOLERANCE,
) -> dict:
    site0_indices = np.flatnonzero(template.donor_site_indices == 0)
    site1_indices = np.flatnonzero(template.donor_site_indices == 1)
    # The experimental template is stored O,S,N at site 0 and N,N,N at site 1.
    if template.donor_elements[site0_indices].tolist() != ["O", "S", "N"]:
        raise ValueError("B1 template site 0 must have ordered O,S,N donors")
    if template.donor_elements[site1_indices].tolist() != ["N", "N", "N"]:
        raise ValueError("B1 template site 1 must have three N donors")

    dch_triads = enumerate_dch_triads(
        roles, template.donor_coords[site0_indices], pair_distance_tolerance
    )
    histidine_triads = enumerate_three_his_triads(
        roles, template.donor_coords[site1_indices], pair_distance_tolerance
    )
    template_centroid_distance = float(np.linalg.norm(
        template.donor_coords[site0_indices].mean(axis=0)
        - template.donor_coords[site1_indices].mean(axis=0)
    ))

    source = np.vstack([
        template.donor_coords[site0_indices], template.donor_coords[site1_indices]
    ])
    best = None
    n_alignments = 0
    for dch in dch_triads:
        dch_keys = {atom.residue_key for atom in dch}
        dch_coords = np.array([atom.coord for atom in dch])
        for hhh in histidine_triads:
            if dch_keys & {atom.residue_key for atom in hhh}:
                continue
            hhh_coords = np.array([atom.coord for atom in hhh])
            centroid_distance = float(np.linalg.norm(dch_coords.mean(0) - hhh_coords.mean(0)))
            if abs(centroid_distance - template_centroid_distance) > site_centroid_distance_tolerance:
                continue
            for permutation in itertools.permutations(range(3)):
                n_alignments += 1
                if n_alignments > MAX_FULL_ALIGNMENTS:
                    return {
                        "status": "unavailable", "positive_call": False,
                        "architecture_call": False, "full_pose_call": False,
                        "reason": "alignment_enumeration_cap_exceeded",
                        "n_dch_triads": len(dch_triads),
                        "n_three_his_triads": len(histidine_triads),
                        "n_alignments": n_alignments,
                    }
                ordered_hhh = tuple(hhh[i] for i in permutation)
                target = np.vstack([dch_coords, [atom.coord for atom in ordered_hhh]])
                rotation, translation, rmsd = kabsch_transform(source, target)
                transferred_ligand = apply_transform(template.ligand_coords, rotation, translation)
                pose = _pose_metrics(transferred_ligand, template.ligand_elements, protein)
                gates = {
                    "pharmacophore_rmsd": rmsd <= max_pharmacophore_rmsd,
                    "hard_clash_fraction": pose["hard_clash_fraction"] <= max_hard_clash_fraction,
                    "pocket_contact_fraction": pose["pocket_contact_fraction"] >= min_pocket_contact_fraction,
                }
                candidate = {
                    "pharmacophore_rmsd": rmsd,
                    "pose_metrics": pose,
                    "gates": gates,
                    "donor_mapping": [atom.label for atom in dch + ordered_hhh],
                    "uses_modified_cysteine_donor": any(
                        atom.modified_from_cysteine for atom in dch
                    ),
                    "transferred_metal_coords": apply_transform(
                        template.metal_coords, rotation, translation
                    ).round(4).tolist(),
                }
                # A passing physical placement always outranks a failing one;
                # within the same pass/fail class prefer the tighter donor fit.
                rank = (not all(gates.values()), rmsd)
                if best is None or rank < best["rank"]:
                    candidate["rank"] = rank
                    best = candidate

    donor_counts = {key: len(value) for key, value in roles.items()}
    if best is None:
        return {
            "status": "not_supported", "positive_call": False,
            "architecture_call": False, "full_pose_call": False,
            "reason": "no_complete_six_donor_B1_pharmacophore",
            "donor_counts": donor_counts,
            "n_dch_triads": len(dch_triads),
            "n_three_his_triads": len(histidine_triads),
            "n_alignments": n_alignments,
        }
    pose = best["pose_metrics"]
    gates = best["gates"]
    return {
        "status": "supported" if all(gates.values()) else "not_supported",
        "architecture_call": bool(gates["pharmacophore_rmsd"]),
        "full_pose_call": bool(all(gates.values())),
        # Backward-compatible alias for the stricter full-pose result. New
        # discovery code should use architecture_call as the primary channel.
        "positive_call": all(gates.values()),
        "reason": "all_geometry_and_pose_gates_passed" if all(gates.values())
                  else "best_pharmacophore_failed_geometry_or_pose_gate",
        "donor_counts": donor_counts,
        "n_dch_triads": len(dch_triads),
        "n_three_his_triads": len(histidine_triads),
        "n_alignments": n_alignments,
        "pharmacophore_rmsd": round(float(best["pharmacophore_rmsd"]), 4),
        "pose_metrics": {key: round(value, 4) for key, value in pose.items()},
        "gates": gates,
        "donor_mapping": best["donor_mapping"],
        "uses_modified_cysteine_donor": best["uses_modified_cysteine_donor"],
        "native_thiolate_architecture_call": bool(
            gates["pharmacophore_rmsd"] and not best["uses_modified_cysteine_donor"]
        ),
        "native_thiolate_positive_call": bool(
            all(gates.values()) and not best["uses_modified_cysteine_donor"]
        ),
        "transferred_metal_coords": best["transferred_metal_coords"],
    }


def score_without_predicted_metals(
    structure_path: Path,
    template: ReactionTemplate,
    max_pharmacophore_rmsd: float = DEFAULT_MAX_PHARMACOPHORE_RMSD,
    max_hard_clash_fraction: float = DEFAULT_MAX_HARD_CLASH_FRACTION,
    min_pocket_contact_fraction: float = DEFAULT_MIN_POCKET_CONTACT_FRACTION,
    pair_distance_tolerance: float = PAIR_DISTANCE_TOLERANCE,
    site_centroid_distance_tolerance: float = SITE_CENTROID_DISTANCE_TOLERANCE,
) -> dict:
    protein, roles = extract_donors(structure_path)
    return score_donor_roles(
        protein, roles, template,
        max_pharmacophore_rmsd=max_pharmacophore_rmsd,
        max_hard_clash_fraction=max_hard_clash_fraction,
        min_pocket_contact_fraction=min_pocket_contact_fraction,
        pair_distance_tolerance=pair_distance_tolerance,
        site_centroid_distance_tolerance=site_centroid_distance_tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--pair-distance-tolerance", type=float, default=1.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "model": "metal_independent_b1_pharmacophore_v1",
        "primary_output": "result.architecture_call",
        "uses_sequence": False,
        "uses_predicted_metal_coordinates": False,
        "pair_distance_tolerance_angstrom": args.pair_distance_tolerance,
        "result": score_without_predicted_metals(
            args.structure, load_b1_template(args.template),
            pair_distance_tolerance=args.pair_distance_tolerance,
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote metal-independent B1 score -> %s", args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
