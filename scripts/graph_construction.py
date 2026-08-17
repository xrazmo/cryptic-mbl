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
             (cavity_fallback instances)
    [22:22+D_ESM]  optional frozen ESM2 per-residue embedding (zeros if disabled)

Edges: all residue pairs within `edge_cutoff` Å (default 10 Å) of each
other's centroid, edge_attr = pairwise distance (single scalar; the
SE(3)-equivariant layers consume raw coordinates directly for geometry,
so this is only a coarse gate + auxiliary feature, not a hand-engineered
interaction feature).

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


def _one_hot_aa(res_name: str) -> np.ndarray:
    v = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    idx = AA_TO_IDX.get(res_name)
    if idx is not None:
        v[idx] = 1.0
    # unrecognized residues (modified AAs, ligand remnants that slipped
    # through) get an all-zero one-hot rather than crashing the pipeline;
    # this is a data-quality signal to check upstream, not silently ignore.
    return v


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


def build_node_features(
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

    if metal_coord is not None:
        dist_to_metal = np.linalg.norm(
            residue_level["centroids"] - metal_coord[None, :], axis=1
        ).reshape(-1, 1).astype(np.float32)
    else:
        dist_to_metal = np.zeros((n, 1), dtype=np.float32)

    if esm2_embeddings is not None:
        assert esm2_embeddings.shape == (n, ESM2_DIM), (
            f"ESM2 embedding shape {esm2_embeddings.shape} != expected ({n}, {ESM2_DIM})"
        )
        esm_block = esm2_embeddings.astype(np.float32)
    else:
        esm_block = np.zeros((n, ESM2_DIM), dtype=np.float32)

    return np.concatenate([aa_onehot, sidechain_flag, dist_to_metal, esm_block], axis=1)


def build_edges(centroids: np.ndarray, cutoff: float = EDGE_CUTOFF_DEFAULT):
    """Returns (edge_index [2, E], edge_attr [E, 1]) for all pairs within cutoff."""
    n = centroids.shape[0]
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    src, dst = np.where((dist <= cutoff) & (~np.eye(n, dtype=bool)))
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = dist[src, dst].reshape(-1, 1).astype(np.float32)
    return edge_index, edge_attr


def pocket_to_pyg_data(
    pocket: PocketSubgraph,
    esm2_embeddings: Optional[np.ndarray] = None,
    edge_cutoff: float = EDGE_CUTOFF_DEFAULT,
    ablate_distance_to_metal: bool = False,
):
    """
    Returns a torch_geometric.data.Data object. Imports torch/PyG lazily so
    this module can be introspected/tested without those (heavy, GPU-linked)
    dependencies installed.
    """
    import torch
    from torch_geometric.data import Data

    residue_level = collapse_to_residue_level(pocket)
    x = build_node_features(residue_level, pocket.metal_coord, esm2_embeddings)
    if ablate_distance_to_metal:
        # Feature layout is AA[0:20], sidechain[20], metal distance[21], ESM2[22:].
        # Preserve the input dimensionality so ablated and baseline models have
        # exactly the same parameter count and differ only in this information.
        x[:, 21] = 0.0
    edge_index, edge_attr = build_edges(residue_level["centroids"], cutoff=edge_cutoff)

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
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import torch

    pocket = PocketSubgraph.load(args.pocket)
    esm2 = np.load(args.esm2_embeddings) if args.esm2_embeddings else None
    data = pocket_to_pyg_data(
        pocket, esm2_embeddings=esm2, edge_cutoff=args.edge_cutoff,
        ablate_distance_to_metal=args.ablate_distance_to_metal,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    log.info(f"Saved graph -> {args.out} ({data.num_nodes} nodes, {data.num_edges} edges)")


if __name__ == "__main__":
    main()
