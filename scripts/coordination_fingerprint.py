"""
coordination_fingerprint.py

Structural V2, part 1: an explicit metal-centered geometric descriptor,
computed directly from the predicted metal site and its coordinating
atoms -- not learned, not pooled through a GNN. This exists because the
current GNN encoder (model.py's PocketEncoder/StructuralEncoder) never
had access to this information at all: it sees one scalar distance from
each residue's centroid to the metal (dist_to_metal) and one from the
nearest donor atom plus a boolean bond-length flag (ligand_geometry) --
no coordination number, no donor-donor angles, no geometry-template fit,
no bond-valence, no donor-element composition. If coordination geometry
is what actually separates a true Zn-MBL site from a superficially
similar metal-adjacent decoy (lactonase, glyoxalase-II), the current
encoder is structurally incapable of learning it regardless of training
time -- this fingerprint is the fast, interpretable way to find out
whether that signal exists before investing in a bigger encoder.

Known limitation, not silently worked around: metal-metal distance for
dinuclear sites is NOT computable from the current pipeline.
pocket_extraction.py's Metal3D call uses --maxp (single top site) and
then does `metal_pdb.coord.mean(axis=0)` -- collapsing whatever probe
atoms Metal3D returns into ONE averaged point, so a real dinuclear B1
site would already be corrupted into a single phantom midpoint before
this script ever sees it, not just omitted. Recovering real dinuclear
geometry would require re-running Metal3D without --maxp and changing
pocket_extraction.py to keep discrete site clusters -- out of scope
here; this fingerprint has no metal-metal-distance feature at all
rather than a fabricated placeholder for one.

Feature vector (22 dims, see FEATURE_NAMES), computed per structure from:
  - coordination shell: canonical donor atoms (His ND1/NE2, Asp OD1/OD2,
    Glu OE1/OE2, Cys SG -- graph_construction.LIGAND_ATOMS) within
    ZN_BOND_CUTOFF (2.8A) of the predicted metal center.
  - donor-metal-donor angles, all pairs, at the metal vertex.
  - deviation from ideal tetrahedral/trigonal-bipyramidal/octahedral
    angle sets: NaN when the actual donor count doesn't match that
    template's ideal coordination number (not a fabricated penalty --
    HistGradientBoostingClassifier natively handles NaN).
  - bond-valence sum, using generic biological Zn2+ parameters
    (R0: Zn-N=1.77, Zn-O=1.70, Zn-S=2.01 Angstrom, B=0.37 -- commonly
    cited approximations, not re-derived here; this is a first-pass
    diagnostic feature, not a calibrated valence analysis).
  - second-shell H-bond network density: polar (N/O) atoms within 3.5A
    of any first-shell donor atom, excluding the donors themselves.
  - full-chain SASA of the donor residues (data/domain_pdbs/*.pdb,
    NOT the truncated pocket -- graph_construction.compute_sasa's own
    docstring notes pocket-truncation overestimates exposure).

CLI:
    python coordination_fingerprint.py --pockets-dir data/pockets \
        --domain-pdbs-dir data/domain_pdbs --out data/coordination_fingerprint.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph
from graph_construction import LIGAND_ATOMS

log = get_logger(__name__)

# Donor-shell search radius: NOT graph_construction.ZN_BOND_CUTOFF (2.8A, the
# textbook Zn-N/O/S bond length). Checked empirically across all 145 labeled
# positives with a predicted metal: at 2.8A only 59/145 (41%) have ANY
# canonical donor atom that close -- these are ML-predicted apo structures
# with a Metal3D-predicted probable site, not refined experimental metal
# coordinates, so real positional slop is expected. 5.0A captures 113/145
# (78%). This is a practical widening for finding the candidate coordinating
# shell on imprecise predictions, not a claim that true Zn-donor bonds are
# this long -- the existing ZN_BOND_CUTOFF=2.8 elsewhere in the codebase
# (graph_construction.py's ligand_geometry bond-length flag) has the same
# miscalibration and was not fixed here; out of scope for this script.
DONOR_SHELL_RADIUS = 5.0

# Generic biological Zn2+ bond-valence parameters (R0 in Angstrom, universal B=0.37).
BVS_R0 = {"N": 1.77, "O": 1.70, "S": 2.01}
BVS_B = 0.37
HBOND_SHELL_CUTOFF = 3.5  # Angstrom, second-shell polar-atom search radius

IDEAL_ANGLES = {
    # coordination number -> sorted list of ideal pairwise donor-metal-donor angles (degrees)
    4: [109.5] * 6,                                              # tetrahedral: C(4,2)=6 pairs
    5: [180.0] * 1 + [120.0] * 3 + [90.0] * 6,                   # trigonal bipyramidal: C(5,2)=10 pairs
    6: [180.0] * 3 + [90.0] * 12,                                # octahedral: C(6,2)=15 pairs
}

FEATURE_NAMES = [
    "coordination_number", "donor_n_count", "donor_o_count", "donor_s_count",
    "metal_donor_dist_1", "metal_donor_dist_2", "metal_donor_dist_3",
    "metal_donor_dist_4", "metal_donor_dist_5", "metal_donor_dist_6",
    "donor_angle_mean", "donor_angle_std", "donor_angle_min", "donor_angle_max",
    "template_deviation_tetrahedral", "template_deviation_trig_bipyramidal", "template_deviation_octahedral",
    "bond_valence_sum", "bond_valence_deviation_from_2",
    "hbond_second_shell_count",
    "sasa_donor_residues_full_chain",
    "has_metal",
]
assert len(FEATURE_NAMES) == 22


def _donor_atoms(pocket: PocketSubgraph) -> list[tuple[np.ndarray, str, int]]:
    """Returns [(coord, element, res_id)] for every canonical donor atom present."""
    out = []
    for i in range(len(pocket.res_ids)):
        res_name = pocket.res_names[i]
        atom_name = pocket.atom_names[i]
        ligand_names = LIGAND_ATOMS.get(res_name)
        if ligand_names and atom_name in ligand_names:
            out.append((pocket.coords[i], pocket.elements[i], int(pocket.res_ids[i])))
    return out


def _pairwise_angle(metal: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - metal, b - metal
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def _template_deviation(actual_angles: list[float], cn: int) -> float:
    ideal = IDEAL_ANGLES.get(cn)
    if ideal is None or len(actual_angles) != len(ideal):
        return float("nan")
    a_sorted = sorted(actual_angles)
    i_sorted = sorted(ideal)
    return float(np.sqrt(np.mean((np.array(a_sorted) - np.array(i_sorted)) ** 2)))


def _bond_valence(donor_dists: list[tuple[float, str]]) -> float:
    total = 0.0
    for dist, element in donor_dists:
        r0 = BVS_R0.get(element)
        if r0 is None:
            continue
        total += np.exp((r0 - dist) / BVS_B)
    return float(total)


def _full_chain_sasa_for_residues(domain_pdb_path: Path, res_ids: set[int]) -> float:
    import biotite.structure as bstruc
    from utils import load_structure

    arr = load_structure(domain_pdb_path)
    try:
        atom_sasa = np.nan_to_num(bstruc.sasa(arr), nan=0.0)
    except Exception as exc:
        log.warning(f"{domain_pdb_path}: full-chain SASA failed ({exc}), defaulting to 0.")
        return 0.0
    mask = np.isin(arr.res_id, list(res_ids))
    if not mask.any():
        return 0.0
    return float(np.sum(atom_sasa[mask]) / 100.0)


def compute_fingerprint(pocket: PocketSubgraph, domain_pdb_path: Path) -> np.ndarray:
    x = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    if pocket.metal_coord is None:
        x[:] = 0.0
        x[FEATURE_NAMES.index("has_metal")] = 0.0
        return x

    metal = pocket.metal_coord
    donors = _donor_atoms(pocket)
    donor_dists = []
    for coord, element, res_id in donors:
        d = float(np.linalg.norm(coord - metal))
        if d <= DONOR_SHELL_RADIUS:
            donor_dists.append((d, element, coord, res_id))
    donor_dists.sort(key=lambda t: t[0])

    cn = len(donor_dists)
    x[FEATURE_NAMES.index("coordination_number")] = cn
    for elem in ("N", "O", "S"):
        x[FEATURE_NAMES.index(f"donor_{elem.lower()}_count")] = sum(1 for _, e, _, _ in donor_dists if e == elem)

    for i in range(6):
        x[FEATURE_NAMES.index(f"metal_donor_dist_{i+1}")] = donor_dists[i][0] if i < cn else np.nan

    angles = []
    for i in range(cn):
        for j in range(i + 1, cn):
            angles.append(_pairwise_angle(metal, donor_dists[i][2], donor_dists[j][2]))
    if angles:
        x[FEATURE_NAMES.index("donor_angle_mean")] = np.mean(angles)
        x[FEATURE_NAMES.index("donor_angle_std")] = np.std(angles)
        x[FEATURE_NAMES.index("donor_angle_min")] = np.min(angles)
        x[FEATURE_NAMES.index("donor_angle_max")] = np.max(angles)

    x[FEATURE_NAMES.index("template_deviation_tetrahedral")] = _template_deviation(angles, 4)
    x[FEATURE_NAMES.index("template_deviation_trig_bipyramidal")] = _template_deviation(angles, 5)
    x[FEATURE_NAMES.index("template_deviation_octahedral")] = _template_deviation(angles, 6)

    bvs = _bond_valence([(d, e) for d, e, _, _ in donor_dists])
    x[FEATURE_NAMES.index("bond_valence_sum")] = bvs
    x[FEATURE_NAMES.index("bond_valence_deviation_from_2")] = abs(bvs - 2.0)

    donor_res_ids = {rid for _, _, _, rid in donor_dists}
    donor_coords = np.array([c for _, _, c, _ in donor_dists]) if donor_dists else np.zeros((0, 3))
    if len(donor_coords):
        all_dists = np.linalg.norm(pocket.coords[:, None, :] - donor_coords[None, :, :], axis=2).min(axis=1)
        is_donor_atom = np.array([
            (pocket.res_names[i] in LIGAND_ATOMS and pocket.atom_names[i] in LIGAND_ATOMS[pocket.res_names[i]]
             and int(pocket.res_ids[i]) in donor_res_ids)
            for i in range(len(pocket.res_ids))
        ])
        polar_mask = np.isin(pocket.elements, ["N", "O"]) & ~is_donor_atom
        hbond_count = int(np.sum((all_dists <= HBOND_SHELL_CUTOFF) & polar_mask))
        x[FEATURE_NAMES.index("hbond_second_shell_count")] = hbond_count

    x[FEATURE_NAMES.index("sasa_donor_residues_full_chain")] = (
        _full_chain_sasa_for_residues(domain_pdb_path, donor_res_ids) if donor_res_ids and domain_pdb_path.exists() else 0.0
    )
    x[FEATURE_NAMES.index("has_metal")] = 1.0
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--domain-pdbs-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    ids, features, labels, subclasses, neg_families = [], [], [], [], []
    files = sorted(args.pockets_dir.glob("*.npz"))
    for i, f in enumerate(files):
        pocket = PocketSubgraph.load(f)
        sid = pocket.metadata.source_structure_id
        fp = compute_fingerprint(pocket, args.domain_pdbs_dir / f"{sid}.pdb")
        ids.append(sid)
        features.append(fp)
        labels.append(pocket.metadata.label)
        subclasses.append(pocket.metadata.subclass or "")
        if (i + 1) % 200 == 0:
            log.info(f"  {i+1}/{len(files)} structures fingerprinted")

    features = np.stack(features)
    n_no_metal = int((features[:, FEATURE_NAMES.index("has_metal")] == 0).sum())
    log.info(f"Fingerprinted {len(ids)} structures ({n_no_metal} with no predicted metal -> all-zero fingerprint)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, ids=np.array(ids), features=features, feature_names=np.array(FEATURE_NAMES),
        labels=np.array(labels), subclasses=np.array(subclasses),
    )
    log.info(f"Wrote coordination fingerprints -> {args.out} (shape {features.shape})")


if __name__ == "__main__":
    main()
