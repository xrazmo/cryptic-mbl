"""
graph_construction.py — Task 2 (spec §4)

Converts a PocketSubgraph (from pocket_extraction.py) into a PyTorch
Geometric `Data` object with residue-level nodes.

Node features (concatenated, in this order):
    [0:20]   one-hot amino acid identity
    [20]     backbone/sidechain flag (1.0 = has sidechain atoms present, i.e.
             not GLY-backbone-only) — residue-level proxy for the atom-level
             flag in the spec; atom-level sub-featurization of metal-
             coordinating residues is added separately (see
             `metal_coordinating_mask` and `--atom-level-metal-shell`).
    [21]     distance to nearest predicted metal ion (Å), 0 if no metal
             (cavity_fallback instances) — CA-centroid based; see [30:32]
             for a ligand-atom-specific version of this same idea.
    [22:30]  physicochemical descriptors (see AA_PROPERTIES): hydrophobicity,
             charge, polarity, side-chain volume, aromaticity, H-bond donor
             count, H-bond acceptor count, canonical-Zn-ligand-capable flag
             (His/Asp/Glu/Cys). Continuous and chemistry-based rather than
             categorical, so e.g. Asp<->Glu sit close in feature space
             instead of being orthogonal one-hot categories — added
             alongside (not replacing) the AA one-hot so the two encodings
             can be ablated against each other the way distance_to_metal was.
    [30]     min distance from this residue's actual ligand-capable atom(s)
             (His Nδ1/Nε2, Asp Oδ1/Oδ2, Glu Oε1/Oε2, Cys Sγ) to the
             predicted metal center, 0 if no metal or non-ligand-capable
             residue. Unlike [21] (CA centroid), this targets the specific
             side-chain atom that would actually coordinate a real Zn site.
    [31]     binary flag: is [30] within a plausible Zn-coordination bond
             length (<= 2.8 Å)
    [32:36]  backbone dihedral geometry: (sin phi, cos phi, sin psi, cos psi),
             computed only when both flanking residues (res_id-1, res_id+1)
             are present in this pocket with valid N/CA/C atoms — pockets are
             a spatial, not sequential, subset of the chain, so this is often
             unavailable; 0s (not a real angle) mark "undefined" rather than
             silently emitting a spurious 0 rad angle.
    [36]     per-residue SASA (Å², /100), computed on the isolated pocket
             substructure — solvent exposure of the truncated boundary is
             overestimated relative to the full chain, but the *relative*
             ordering of buried-vs-exposed residues within one pocket is
             still informative.
    [37:40]  radial shell one-hot: (coordination shell <=5A, pocket core
             5-9A, outer boundary 9A-edge of pocket) by centroid distance
             to the predicted metal (same distance as [21]); all-zero if
             no metal. This bins *within* the existing pocket extraction
             radius (12A max, see pocket_extraction.py) rather than a
             true wider outer-loop shell -- extending the actual
             extraction radius (e.g. to 16A) would require re-running
             pocket extraction, which this does not do.
    [40:40+D_ESM]  optional frozen ESM2 per-residue embedding (zeros if disabled)

Edges: the union of (a) all residue pairs within `edge_cutoff` Å (default
10 Å) of each other's centroid, and (b) explicit sequence-adjacency pairs
(res_id, res_id +/- 1) when both are present in the pocket -- pockets are
a spatial, not sequential, subset of the chain, so a loop's own backbone
neighbors are otherwise only connected when they happen to also be
spatially close, i.e. never signaled as "this is the same loop" per se.
edge_attr = (distance, is_sequence_adjacent flag); the flag is consumed
by the message-passing layer (see model.py) so backbone-adjacency and
spatial-contact edges can be weighted differently, not just recorded as
metadata. The SE(3)-equivariant layers this is a placeholder for consume
raw coordinates directly for geometry, so distance itself remains only a
coarse gate + auxiliary feature.

Coordinates are stored as `data.pos` (PyG convention) for consumption by
SE(3)-equivariant layers (e3nn / EZSpecificity backbone), which operate on
`pos` directly rather than through edge_attr distances.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from utils import PocketSubgraph, get_logger

log = get_logger(__name__)

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

EDGE_CUTOFF_DEFAULT = 10.0
ESM2_DIM = 1280  # esm2_t33_650M_UR50D per-residue embedding dim; adjust to chosen checkpoint

# Physicochemical descriptors per residue, in the order documented above:
#   [hydrophobicity (Kyte-Doolittle), charge, polarity (0/1), side-chain
#    volume (A^3 / 100, Zamyatnin), aromaticity (0/1), H-bond donor count,
#    H-bond acceptor count, canonical Zn-ligand-capable (0/1)]
# Continuous/chemistry-based, deliberately NOT a one-hot, so chemically
# similar residues (e.g. Asp/Glu, both carboxylate Zn-ligands) sit close in
# feature space instead of being orthogonal categories -- meant to reduce
# how much the model has to rely on exact sequence identity to recognize
# "this position does Zn-ligand chemistry", which matters for generalizing
# to sequence-divergent cryptic homologs (see project README).
AA_PROPERTIES = {
    "ALA": (1.8, 0, 0, 0.88, 0, 0, 0, 0),
    "ARG": (-4.5, 1, 1, 1.73, 0, 5, 0, 0),
    "ASN": (-3.5, 0, 1, 1.14, 0, 2, 2, 0),
    "ASP": (-3.5, -1, 1, 1.11, 0, 1, 3, 1),
    "CYS": (2.5, 0, 0, 1.08, 0, 1, 0, 1),
    "GLN": (-3.5, 0, 1, 1.44, 0, 2, 2, 0),
    "GLU": (-3.5, -1, 1, 1.38, 0, 1, 3, 1),
    "GLY": (-0.4, 0, 0, 0.60, 0, 0, 0, 0),
    "HIS": (-3.2, 0, 1, 1.53, 1, 2, 1, 1),
    "ILE": (4.5, 0, 0, 1.67, 0, 0, 0, 0),
    "LEU": (3.8, 0, 0, 1.67, 0, 0, 0, 0),
    "LYS": (-3.9, 1, 1, 1.69, 0, 3, 0, 0),
    "MET": (1.9, 0, 0, 1.62, 0, 0, 0, 0),
    "PHE": (2.8, 0, 0, 1.89, 1, 0, 0, 0),
    "PRO": (-1.6, 0, 0, 1.13, 0, 0, 0, 0),
    "SER": (-0.8, 0, 1, 0.89, 0, 1, 1, 0),
    "THR": (-0.7, 0, 1, 1.16, 0, 1, 1, 0),
    "TRP": (-0.9, 0, 1, 2.28, 1, 1, 0, 0),
    "TYR": (-1.3, 0, 1, 1.94, 1, 1, 1, 0),
    "VAL": (4.2, 0, 0, 1.40, 0, 0, 0, 0),
}
N_CHEM_PROPS = 8

# Named slices over the node-feature layout documented in the module
# docstring, for modality-ablation experiments (see --ablate-* flags below)
# and for model.py's branched encoder (which needs to split structural vs
# ESM2 columns): zero out / slice a whole block while preserving in_dim, so
# ablated and full models stay directly comparable (same pattern as
# --ablate-distance-to-metal).
N_AA_IDENTITY = 20  # aa_onehot
N_RADIAL_SHELL = 3  # coordination shell / pocket core / outer boundary, one-hot
N_STRUCTURAL = 1 + 1 + N_CHEM_PROPS + 2 + 4 + 1 + N_RADIAL_SHELL  # sidechain, dist_to_metal, chem_props, ligand_geometry, dihedrals, sasa, radial_shell = 20
AA_IDENTITY_SLICE = slice(0, N_AA_IDENTITY)
STRUCTURAL_SLICE = slice(N_AA_IDENTITY, N_AA_IDENTITY + N_STRUCTURAL)
ESM2_SLICE = slice(N_AA_IDENTITY + N_STRUCTURAL, N_AA_IDENTITY + N_STRUCTURAL + ESM2_DIM)

# Radial shell bin edges (Angstrom, centroid distance to predicted metal):
# [0, 5) -> coordination shell, [5, 9) -> pocket core, [9, inf) -> outer
# boundary (bounded in practice by the 12A pocket-extraction radius).
RADIAL_SHELL_BOUNDARIES = (5.0, 9.0)

# Side-chain atom(s) that actually coordinate a Zn ion in canonical MBL
# active sites (3H, DCH, and related B1/B2/B3 coordination schemes).
LIGAND_ATOMS = {
    "HIS": ("ND1", "NE2"),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "CYS": ("SG",),
}
ZN_BOND_CUTOFF = 2.8  # Angstrom; generous upper bound on Zn-N/O/S bond lengths


def _one_hot_aa(res_name: str) -> np.ndarray:
    v = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    idx = AA_TO_IDX.get(res_name)
    if idx is not None:
        v[idx] = 1.0
    # unrecognized residues (modified AAs, ligand remnants that slipped
    # through) get an all-zero one-hot rather than crashing the pipeline;
    # this is a data-quality signal to check upstream, not silently ignore.
    return v


def _chem_properties(res_name: str) -> np.ndarray:
    props = AA_PROPERTIES.get(res_name)
    if props is None:
        return np.zeros(N_CHEM_PROPS, dtype=np.float32)
    return np.array(props, dtype=np.float32)


def collapse_to_residue_level(pocket: PocketSubgraph) -> dict:
    """
    Reduce the atom-level PocketSubgraph to one node per residue:
    CA coordinate (or atom centroid if no CA), majority res_name,
    and a flag for whether any sidechain atoms are present.
    """
    uniq_res_ids = np.unique(pocket.res_ids)
    n = len(uniq_res_ids)
    centroids = np.zeros((n, 3), dtype=np.float32)
    res_names = []
    has_sidechain = np.zeros(n, dtype=np.float32)

    for i, rid in enumerate(uniq_res_ids):
        mask = pocket.res_ids == rid
        names = pocket.atom_names[mask]
        coords = pocket.coords[mask]
        ca_mask = names == "CA"
        centroids[i] = coords[ca_mask][0] if ca_mask.any() else coords.mean(axis=0)
        res_names.append(pocket.res_names[mask][0])
        has_sidechain[i] = float(pocket.is_sidechain[mask].any())

    return {
        "res_ids": uniq_res_ids,
        "centroids": centroids,
        "res_names": np.array(res_names),
        "has_sidechain": has_sidechain,
    }


def compute_ligand_geometry(
    pocket: PocketSubgraph, residue_level: dict, metal_coord: Optional[np.ndarray],
) -> np.ndarray:
    """
    Returns (n, 2): [min distance from this residue's actual ligand-capable
    atom(s) to the metal center, binary flag that the distance is within a
    plausible Zn bond length]. Both 0 if no metal, or the residue has no
    canonical ligand atoms (LIGAND_ATOMS) present.
    """
    n = len(residue_level["res_ids"])
    out = np.zeros((n, 2), dtype=np.float32)
    if metal_coord is None:
        return out
    for i, rid in enumerate(residue_level["res_ids"]):
        ligand_names = LIGAND_ATOMS.get(residue_level["res_names"][i])
        if not ligand_names:
            continue
        mask = (pocket.res_ids == rid) & np.isin(pocket.atom_names, ligand_names)
        if not mask.any():
            continue
        min_dist = float(np.linalg.norm(pocket.coords[mask] - metal_coord[None, :], axis=1).min())
        out[i, 0] = min_dist
        out[i, 1] = 1.0 if min_dist <= ZN_BOND_CUTOFF else 0.0
    return out


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Praxeolitic dihedral formula; returns radians."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def compute_backbone_dihedrals(pocket: PocketSubgraph, residue_level: dict) -> np.ndarray:
    """
    Returns (n, 4): [sin(phi), cos(phi), sin(psi), cos(psi)] (sin/cos, not
    raw radians, so the feature doesn't have a spurious discontinuity at
    +-pi). Only computed when both flanking residues (res_id-1, res_id+1)
    are present in this pocket with valid backbone atoms -- pockets are a
    spatial, not sequential, subset of the chain, so this is frequently
    unavailable; left as (0,0,0,0) ("undefined") rather than a fabricated angle.
    """
    n = len(residue_level["res_ids"])
    out = np.zeros((n, 4), dtype=np.float32)
    rid_set = set(residue_level["res_ids"].tolist())

    def backbone_atom(rid: int, name: str) -> Optional[np.ndarray]:
        mask = (pocket.res_ids == rid) & (pocket.atom_names == name)
        return pocket.coords[mask][0] if mask.any() else None

    for i, rid in enumerate(residue_level["res_ids"]):
        prev_rid, next_rid = int(rid) - 1, int(rid) + 1
        if prev_rid not in rid_set or next_rid not in rid_set:
            continue
        atoms = (
            backbone_atom(prev_rid, "C"), backbone_atom(rid, "N"), backbone_atom(rid, "CA"),
            backbone_atom(rid, "C"), backbone_atom(next_rid, "N"),
        )
        if any(a is None for a in atoms):
            continue
        c_prev, n_curr, ca_curr, c_curr, n_next = atoms
        phi = _dihedral_angle(c_prev, n_curr, ca_curr, c_curr)
        psi = _dihedral_angle(n_curr, ca_curr, c_curr, n_next)
        out[i] = [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)]
    return out


def compute_radial_shell(dist_to_metal: np.ndarray, has_metal: bool) -> np.ndarray:
    """
    One-hot (n, N_RADIAL_SHELL) bin of each residue's centroid distance to
    the metal (same values as the dist_to_metal feature) into coordination
    shell / pocket core / outer boundary, per RADIAL_SHELL_BOUNDARIES.
    All-zero when there's no metal (cavity_fallback instances), matching
    dist_to_metal's own convention for that case.
    """
    n = dist_to_metal.shape[0]
    out = np.zeros((n, N_RADIAL_SHELL), dtype=np.float32)
    if not has_metal:
        return out
    d = dist_to_metal.reshape(-1)
    inner, outer = RADIAL_SHELL_BOUNDARIES
    out[d < inner, 0] = 1.0
    out[(d >= inner) & (d < outer), 1] = 1.0
    out[d >= outer, 2] = 1.0
    return out


def compute_sasa(pocket: PocketSubgraph, residue_level: dict) -> np.ndarray:
    """
    Per-residue SASA (Angstrom^2 / 100), computed on the isolated pocket
    substructure. Solvent exposure at the pocket's truncated boundary is
    overestimated relative to the full chain (neighboring atoms that would
    normally block access are missing), but the *relative* buried-vs-exposed
    ordering within one pocket is still informative. Falls back to zeros if
    the SASA calculation errors (e.g. unrecognized elements) rather than
    failing graph construction for the whole pocket.
    """
    import biotite.structure as bstruc

    n = len(residue_level["res_ids"])
    try:
        arr = bstruc.AtomArray(len(pocket.res_ids))
        arr.coord = pocket.coords
        arr.res_id = pocket.res_ids
        arr.res_name = pocket.res_names
        arr.atom_name = pocket.atom_names
        arr.element = pocket.elements
        arr.chain_id = np.full(len(pocket.res_ids), "A")
        atom_sasa = np.nan_to_num(bstruc.sasa(arr), nan=0.0)
    except Exception as exc:
        log.warning(f"SASA computation failed ({exc}); defaulting SASA feature to 0.")
        return np.zeros((n, 1), dtype=np.float32)

    out = np.zeros((n, 1), dtype=np.float32)
    for i, rid in enumerate(residue_level["res_ids"]):
        out[i, 0] = float(np.sum(atom_sasa[pocket.res_ids == rid])) / 100.0
    return out


def build_node_features(
    pocket: PocketSubgraph,
    residue_level: dict,
    metal_coord: Optional[np.ndarray],
    esm2_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    esm2_embeddings, if provided, must be pre-aligned to residue_level["res_ids"]
    order (shape [n_residues, ESM2_DIM]) — computed upstream by a separate
    ESM2 inference pass over the full parent sequence, then sliced to the
    pocket residues. Not computed here to avoid re-running ESM2 per pocket
    when many pockets come from the same parent structure.
    """
    n = len(residue_level["res_ids"])
    aa_onehot = np.stack([_one_hot_aa(rn) for rn in residue_level["res_names"]])
    sidechain_flag = residue_level["has_sidechain"].reshape(-1, 1)
    chem_props = np.stack([_chem_properties(rn) for rn in residue_level["res_names"]])

    if metal_coord is not None:
        dist_to_metal = np.linalg.norm(
            residue_level["centroids"] - metal_coord[None, :], axis=1
        ).reshape(-1, 1).astype(np.float32)
    else:
        dist_to_metal = np.zeros((n, 1), dtype=np.float32)

    ligand_geometry = compute_ligand_geometry(pocket, residue_level, metal_coord)
    dihedrals = compute_backbone_dihedrals(pocket, residue_level)
    sasa = compute_sasa(pocket, residue_level)
    radial_shell = compute_radial_shell(dist_to_metal, has_metal=metal_coord is not None)

    if esm2_embeddings is not None:
        assert esm2_embeddings.shape == (n, ESM2_DIM), (
            f"ESM2 embedding shape {esm2_embeddings.shape} != expected ({n}, {ESM2_DIM})"
        )
        esm_block = esm2_embeddings.astype(np.float32)
    else:
        esm_block = np.zeros((n, ESM2_DIM), dtype=np.float32)

    return np.concatenate([
        aa_onehot, sidechain_flag, dist_to_metal, chem_props,
        ligand_geometry, dihedrals, sasa, radial_shell, esm_block,
    ], axis=1)


def build_edges(centroids: np.ndarray, res_ids: np.ndarray, cutoff: float = EDGE_CUTOFF_DEFAULT):
    """
    Returns (edge_index [2, E], edge_attr [E, 2]) for the union of (a) all
    residue pairs within `cutoff` of each other's centroid and (b) explicit
    sequence-adjacency pairs (res_id +/- 1), deduplicated so a pair that
    qualifies both ways isn't double-counted in the mean aggregation.
    edge_attr columns: [pairwise distance, is_sequence_adjacent flag].
    """
    n = centroids.shape[0]
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    spatial_mask = (dist <= cutoff) & (~np.eye(n, dtype=bool))

    rid_to_idx = {int(rid): i for i, rid in enumerate(res_ids)}
    seq_mask = np.zeros((n, n), dtype=bool)
    for i, rid in enumerate(res_ids):
        for neighbor_rid in (int(rid) - 1, int(rid) + 1):
            j = rid_to_idx.get(neighbor_rid)
            if j is not None:
                seq_mask[i, j] = True

    combined_mask = spatial_mask | seq_mask
    src, dst = np.where(combined_mask)
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = np.stack(
        [dist[src, dst], seq_mask[src, dst].astype(np.float32)], axis=1
    ).astype(np.float32)
    return edge_index, edge_attr


def pocket_to_pyg_data(
    pocket: PocketSubgraph,
    esm2_embeddings: Optional[np.ndarray] = None,
    edge_cutoff: float = EDGE_CUTOFF_DEFAULT,
    ablate_distance_to_metal: bool = False,
    ablate_aa_identity: bool = False,
    ablate_structural: bool = False,
    ablate_esm2: bool = False,
):
    """
    Returns a torch_geometric.data.Data object. Imports torch/PyG lazily so
    this module can be introspected/tested without those (heavy, GPU-linked)
    dependencies installed.

    The three ablate_* block flags (aa_identity, structural, esm2) zero out
    whole feature blocks -- see AA_IDENTITY_SLICE/STRUCTURAL_SLICE/ESM2_SLICE
    -- to build the matched modality-comparison models (structure-only,
    identity-reduced structure, ESM-only) without changing in_dim, so all
    variants stay directly comparable in parameter count.
    """
    import torch
    from torch_geometric.data import Data

    residue_level = collapse_to_residue_level(pocket)
    x = build_node_features(pocket, residue_level, pocket.metal_coord, esm2_embeddings)
    if ablate_distance_to_metal:
        # Feature layout: see module docstring. Preserve the input
        # dimensionality so ablated and baseline models have exactly the
        # same parameter count and differ only in this one feature.
        x[:, 21] = 0.0
    if ablate_aa_identity:
        x[:, AA_IDENTITY_SLICE] = 0.0
    if ablate_structural:
        x[:, STRUCTURAL_SLICE] = 0.0
    if ablate_esm2:
        x[:, ESM2_SLICE] = 0.0
    edge_index, edge_attr = build_edges(residue_level["centroids"], residue_level["res_ids"], cutoff=edge_cutoff)

    label_map = {"positive": 1, "hard_negative": 0, "easy_negative": 0, "unlabeled": -1}

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(residue_level["centroids"], dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([label_map[pocket.metadata.label]], dtype=torch.long),
    )
    # Metadata carried as plain attributes (PyG allows arbitrary extra fields).
    data.structure_id = pocket.metadata.source_structure_id
    data.confidence_tier = pocket.metadata.confidence_tier
    data.subclass = pocket.metadata.subclass
    data.pocket_source = pocket.metadata.pocket_source
    return data


def main():
    p = argparse.ArgumentParser(description="Convert a saved PocketSubgraph (.npz) into a PyG graph (.pt).")
    p.add_argument("--pocket", required=True, type=Path)
    p.add_argument("--esm2-embeddings", type=Path, default=None,
                    help="Optional .npy file, pre-aligned to residue order, shape (n_res, ESM2_DIM).")
    p.add_argument("--edge-cutoff", type=float, default=EDGE_CUTOFF_DEFAULT)
    p.add_argument("--ablate-distance-to-metal", action="store_true")
    p.add_argument("--ablate-aa-identity", action="store_true",
                    help="Zero the 20-dim amino-acid one-hot block.")
    p.add_argument("--ablate-structural", action="store_true",
                    help="Zero the 17-dim chemistry/geometry block (sidechain flag through SASA).")
    p.add_argument("--ablate-esm2", action="store_true",
                    help="Zero the ESM2 embedding block.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import torch

    pocket = PocketSubgraph.load(args.pocket)
    esm2 = np.load(args.esm2_embeddings) if args.esm2_embeddings else None
    data = pocket_to_pyg_data(
        pocket, esm2_embeddings=esm2, edge_cutoff=args.edge_cutoff,
        ablate_distance_to_metal=args.ablate_distance_to_metal,
        ablate_aa_identity=args.ablate_aa_identity,
        ablate_structural=args.ablate_structural,
        ablate_esm2=args.ablate_esm2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    log.info(f"Saved graph -> {args.out} ({data.num_nodes} nodes, {data.num_edges} edges)")


if __name__ == "__main__":
    main()
