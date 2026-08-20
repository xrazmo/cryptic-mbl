"""
catalytic_feasibility.py

Structural V3: a family-reference-independent, substrate-conditioned
catalytic-feasibility channel.

The previous structural classifiers compared a candidate representation with
positive/negative protein centroids or a reference bank.  That is useful for
recognition, but it asks whether the candidate resembles known MBLs.  This
module asks a different question: can an experimentally observed hydrolyzed
beta-lactam reaction state be transferred onto the candidate's predicted
metal/donor geometry while retaining a plausible pocket placement?

The scorer deliberately does NOT use:
  * amino-acid sequence or ESM embeddings;
  * labels, class centroids, nearest neighbours, or trained weights;
  * an MBL reference-protein vote.

It does use a small, provenance-tracked collection of experimental
enzyme/product structures spanning B1, B2, and B3.  Those structures supply
reaction-state coordinates, not a family-similarity target.  A candidate with
a different sequence or global fold can pass if its local metal/donor frame
supports the reaction state.  This is still a geometric screen, not proof of
hydrolysis: dynamics, proton transfer, water placement, and turnover require
later docking/refinement and experiment.

Typical use:
    python build_catalytic_templates.py ...   # one-time template preparation
    python catalytic_feasibility.py \
        --pocket data/pockets_v2/CANDIDATE.npz \
        --templates-dir data/catalytic_templates \
        --out results/CANDIDATE.catalytic.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from structural_chemistry import LIGAND_ATOMS
from utils import PocketSubgraph, get_logger

log = get_logger(__name__)

DONOR_RADIUS_ANGSTROM = 2.9
MAX_ALIGNMENT_ENUMERATIONS = 100_000

# Physical-sanity gates, intentionally fixed before corpus evaluation.  They
# are not label-fitted operating thresholds and must be reported as such.
DEFAULT_MAX_PHARMACOPHORE_RMSD = 1.25
DEFAULT_MAX_HARD_CLASH_FRACTION = 0.10
DEFAULT_MIN_POCKET_CONTACT_FRACTION = 0.50
HARD_CLASH_DISTANCE = 1.50
POCKET_CONTACT_DISTANCE = 4.50


@dataclass
class ReactionTemplate:
    template_id: str
    pdb_id: str
    subclass: str
    protein_chain: str
    ligand_resname: str
    substrate_class: str
    reaction_state: str
    source_url: str
    citation_doi: str
    resolution_angstrom: float
    metal_coords: np.ndarray
    donor_coords: np.ndarray
    donor_elements: np.ndarray
    donor_site_indices: np.ndarray
    donor_labels: np.ndarray
    ligand_coords: np.ndarray
    ligand_elements: np.ndarray
    ligand_atom_names: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = asdict(self)
        array_keys = {
            "metal_coords", "donor_coords", "donor_elements",
            "donor_site_indices", "donor_labels", "ligand_coords",
            "ligand_elements", "ligand_atom_names",
        }
        arrays = {key: metadata.pop(key) for key in array_keys}
        np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)

    @staticmethod
    def load(path: Path) -> "ReactionTemplate":
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        return ReactionTemplate(
            **metadata,
            metal_coords=data["metal_coords"],
            donor_coords=data["donor_coords"],
            donor_elements=data["donor_elements"],
            donor_site_indices=data["donor_site_indices"],
            donor_labels=data["donor_labels"],
            ligand_coords=data["ligand_coords"],
            ligand_elements=data["ligand_elements"],
            ligand_atom_names=data["ligand_atom_names"],
        )


@dataclass(frozen=True)
class CandidateDonor:
    coord: np.ndarray
    element: str
    site_index: int
    atom_index: int
    label: str


def candidate_donors(pocket: PocketSubgraph) -> list[CandidateDonor]:
    """Canonical side-chain donor atoms assigned to their nearest metal."""
    if pocket.metal_coords is None or len(pocket.metal_coords) == 0:
        return []
    donors = []
    for i, (resname, atomname, element) in enumerate(
        zip(pocket.res_names, pocket.atom_names, pocket.elements)
    ):
        allowed = LIGAND_ATOMS.get(str(resname))
        if not allowed or str(atomname) not in allowed:
            continue
        distances = np.linalg.norm(pocket.metal_coords - pocket.coords[i], axis=1)
        site_index = int(np.argmin(distances))
        if float(distances[site_index]) > DONOR_RADIUS_ANGSTROM:
            continue
        donors.append(CandidateDonor(
            coord=pocket.coords[i].astype(float),
            element=str(element).upper(),
            site_index=site_index,
            atom_index=i,
            label=f"{resname}:{int(pocket.res_ids[i])}:{atomname}",
        ))
    return donors


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return row-vector rotation/translation mapping source onto target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source and target must both be (N,3), got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError("at least three pharmacophore points are required")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    x = source - source_centroid
    y = target - target_centroid
    u, _s, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_centroid - source_centroid @ rotation
    fitted = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation + translation


def _site_permutations(n_sites: int) -> Iterable[tuple[int, ...]]:
    return itertools.permutations(range(n_sites))


def _donor_assignment_groups(
    template: ReactionTemplate,
    donors: list[CandidateDonor],
    site_map: tuple[int, ...],
) -> list[tuple[np.ndarray, list[tuple[CandidateDonor, ...]]]] | None:
    """Create mapping choices grouped by (template site, donor element)."""
    groups = []
    for template_site in range(len(template.metal_coords)):
        for element in sorted(set(template.donor_elements[template.donor_site_indices == template_site].tolist())):
            template_idx = np.flatnonzero(
                (template.donor_site_indices == template_site)
                & (template.donor_elements == element)
            )
            candidates = [
                donor for donor in donors
                if donor.site_index == site_map[template_site] and donor.element == str(element)
            ]
            if len(candidates) < len(template_idx):
                return None
            choices = list(itertools.permutations(candidates, len(template_idx)))
            groups.append((template_idx, choices))
    return groups


def _alignment_candidates(
    template: ReactionTemplate,
    pocket: PocketSubgraph,
    donors: list[CandidateDonor],
) -> Iterable[tuple[np.ndarray, np.ndarray, dict]]:
    n_sites = len(template.metal_coords)
    if pocket.metal_coords is None or len(pocket.metal_coords) != n_sites:
        return

    n_enumerated = 0
    for site_map in _site_permutations(n_sites):
        groups = _donor_assignment_groups(template, donors, site_map)
        if groups is None:
            continue
        group_choices = [choices for _idx, choices in groups]
        for selected_groups in itertools.product(*group_choices):
            n_enumerated += 1
            if n_enumerated > MAX_ALIGNMENT_ENUMERATIONS:
                raise RuntimeError(
                    f"alignment enumeration exceeded {MAX_ALIGNMENT_ENUMERATIONS}; "
                    "inspect duplicate donor/site assignments"
                )

            source_points = [coord for coord in template.metal_coords]
            target_points = [pocket.metal_coords[site_map[i]] for i in range(n_sites)]
            mapping_labels = []
            for (template_indices, _choices), chosen in zip(groups, selected_groups):
                for template_index, candidate in zip(template_indices, chosen):
                    source_points.append(template.donor_coords[template_index])
                    target_points.append(candidate.coord)
                    mapping_labels.append({
                        "template_donor": str(template.donor_labels[template_index]),
                        "candidate_donor": candidate.label,
                    })
            yield np.asarray(source_points), np.asarray(target_points), {
                "site_map_template_to_candidate": list(site_map),
                "donor_mapping": mapping_labels,
            }


def _transferred_pose_metrics(
    ligand_coords: np.ndarray,
    ligand_elements: np.ndarray,
    pocket: PocketSubgraph,
) -> dict:
    protein_mask = ~np.isin(np.char.upper(pocket.elements.astype(str)), ["H", "D"])
    protein_coords = pocket.coords[protein_mask]
    ligand_mask = ~np.isin(np.char.upper(ligand_elements.astype(str)), ["H", "D"])
    ligand_heavy = ligand_coords[ligand_mask]
    if len(ligand_heavy) == 0 or len(protein_coords) == 0:
        raise ValueError("transferred pose and candidate pocket require heavy atoms")

    distances = np.linalg.norm(ligand_heavy[:, None, :] - protein_coords[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    hard_clash_fraction = float(np.mean(nearest < HARD_CLASH_DISTANCE))
    contact_fraction = float(np.mean(nearest <= POCKET_CONTACT_DISTANCE))

    metal_distances = np.linalg.norm(
        ligand_heavy[:, None, :] - pocket.metal_coords[None, :, :], axis=2
    )
    hetero = np.isin(np.char.upper(ligand_elements[ligand_mask].astype(str)), ["N", "O", "S"])
    hetero_min = float(np.min(metal_distances[hetero])) if hetero.any() else None
    return {
        "hard_clash_fraction": hard_clash_fraction,
        "pocket_contact_fraction": contact_fraction,
        "minimum_protein_ligand_distance": float(nearest.min()),
        "minimum_ligand_metal_distance": float(metal_distances.min()),
        "minimum_heteroatom_metal_distance": hetero_min,
        "n_ligand_heavy_atoms": int(len(ligand_heavy)),
    }


def score_template(
    pocket: PocketSubgraph,
    template: ReactionTemplate,
    max_pharmacophore_rmsd: float = DEFAULT_MAX_PHARMACOPHORE_RMSD,
    max_hard_clash_fraction: float = DEFAULT_MAX_HARD_CLASH_FRACTION,
    min_pocket_contact_fraction: float = DEFAULT_MIN_POCKET_CONTACT_FRACTION,
) -> dict:
    if pocket.metal_coords is None or len(pocket.metal_coords) == 0:
        return {"status": "unavailable", "reason": "candidate_has_no_predicted_metal_site"}
    if len(pocket.metal_coords) != len(template.metal_coords):
        return {
            "status": "not_applicable",
            "reason": "metal_site_count_mismatch",
            "candidate_n_sites": int(len(pocket.metal_coords)),
            "template_n_sites": int(len(template.metal_coords)),
        }

    donors = candidate_donors(pocket)
    best = None
    for source, target, mapping in _alignment_candidates(template, pocket, donors):
        rotation, translation, rmsd = kabsch_transform(source, target)
        if best is None or rmsd < best["pharmacophore_rmsd"]:
            transferred = apply_transform(template.ligand_coords, rotation, translation)
            pose_metrics = _transferred_pose_metrics(transferred, template.ligand_elements, pocket)
            best = {
                "pharmacophore_rmsd": rmsd,
                "mapping": mapping,
                "pose_metrics": pose_metrics,
            }

    if best is None:
        return {
            "status": "not_applicable",
            "reason": "candidate_lacks_required_site_resolved_donor_elements",
            "candidate_donors": [donor.label for donor in donors],
        }

    pose = best["pose_metrics"]
    gates = {
        "pharmacophore_rmsd": best["pharmacophore_rmsd"] <= max_pharmacophore_rmsd,
        "hard_clash_fraction": pose["hard_clash_fraction"] <= max_hard_clash_fraction,
        "pocket_contact_fraction": pose["pocket_contact_fraction"] >= min_pocket_contact_fraction,
    }
    return {
        "status": "supported" if all(gates.values()) else "not_supported",
        "template_id": template.template_id,
        "pdb_id": template.pdb_id,
        "subclass": template.subclass,
        "substrate_class": template.substrate_class,
        "reaction_state": template.reaction_state,
        "pharmacophore_rmsd": round(float(best["pharmacophore_rmsd"]), 4),
        "pose_metrics": {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in pose.items()
        },
        "gates": gates,
        "mapping": best["mapping"],
    }


def score_catalytic_feasibility(
    pocket: PocketSubgraph,
    templates: list[ReactionTemplate],
    max_pharmacophore_rmsd: float = DEFAULT_MAX_PHARMACOPHORE_RMSD,
    max_hard_clash_fraction: float = DEFAULT_MAX_HARD_CLASH_FRACTION,
    min_pocket_contact_fraction: float = DEFAULT_MIN_POCKET_CONTACT_FRACTION,
) -> dict:
    if not templates:
        return {
            "status": "unavailable",
            "reason": "no_reaction_templates_loaded",
            "template_results": [],
        }
    results = [
        score_template(
            pocket, template,
            max_pharmacophore_rmsd=max_pharmacophore_rmsd,
            max_hard_clash_fraction=max_hard_clash_fraction,
            min_pocket_contact_fraction=min_pocket_contact_fraction,
        )
        for template in templates
    ]
    evaluable = [result for result in results if result["status"] in {"supported", "not_supported"}]
    supported = [result for result in evaluable if result["status"] == "supported"]
    if supported:
        status = "supported"
        best = min(supported, key=lambda result: result["pharmacophore_rmsd"])
    elif evaluable:
        status = "not_supported"
        best = min(evaluable, key=lambda result: result["pharmacophore_rmsd"])
    else:
        status = "unavailable"
        best = None
    return {
        "status": status,
        "interpretation": (
            "local reaction-state geometry support; not proof of beta-lactam hydrolysis"
            if status == "supported" else
            "no passing reaction-state transfer among evaluable templates"
            if status == "not_supported" else
            "reaction-state transfer could not be evaluated"
        ),
        "threshold_provenance": "predeclared_physical_sanity_gates_not_label_fitted",
        "thresholds": {
            "max_pharmacophore_rmsd": max_pharmacophore_rmsd,
            "max_hard_clash_fraction": max_hard_clash_fraction,
            "min_pocket_contact_fraction": min_pocket_contact_fraction,
        },
        "best_supported_template": best["template_id"] if supported else None,
        "best_evaluable_template": best["template_id"] if best else None,
        "n_templates": len(results),
        "n_evaluable": len(evaluable),
        "n_supported": len(supported),
        "template_results": results,
    }


def load_templates(templates_dir: Path) -> list[ReactionTemplate]:
    return [ReactionTemplate.load(path) for path in sorted(templates_dir.glob("*.npz"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate-local beta-lactam catalytic feasibility.")
    parser.add_argument("--pocket", required=True, type=Path)
    parser.add_argument("--templates-dir", required=True, type=Path)
    parser.add_argument("--max-pharmacophore-rmsd", type=float, default=DEFAULT_MAX_PHARMACOPHORE_RMSD)
    parser.add_argument("--max-hard-clash-fraction", type=float, default=DEFAULT_MAX_HARD_CLASH_FRACTION)
    parser.add_argument("--min-pocket-contact-fraction", type=float, default=DEFAULT_MIN_POCKET_CONTACT_FRACTION)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pocket = PocketSubgraph.load(args.pocket)
    result = {
        "structure_id": pocket.metadata.source_structure_id,
        "scoring_regime": "v3_catalytic_feasibility",
        "uses_sequence": False,
        "uses_reference_protein_panel": False,
        "uses_experimental_reaction_state_templates": True,
        "catalytic_feasibility": score_catalytic_feasibility(
            pocket,
            load_templates(args.templates_dir),
            max_pharmacophore_rmsd=args.max_pharmacophore_rmsd,
            max_hard_clash_fraction=args.max_hard_clash_fraction,
            min_pocket_contact_fraction=args.min_pocket_contact_fraction,
        ),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        log.info("Wrote catalytic-feasibility score -> %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
