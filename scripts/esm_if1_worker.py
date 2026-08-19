"""
esm_if1_worker.py

Runs INSIDE the `esm-if1` conda environment (Python 3.10, torch 2.9.1+cu128,
torch_scatter/torch_cluster/torch_geometric matching prebuilt wheels,
biotite<1.0 for esm.inverse_folding.util's filter_backbone import) --
never imported directly by the main cryptic-mbl environment, which has
an incompatible torch/CUDA combination for torch_scatter (see
run_esm_if1.sh and the environment-setup notes in this project's history).

Loads the frozen pretrained ESM-IF1 structure encoder
(esm_if1_gvp4_t16_142M_UR50, 141.6M params, GVP-GNN + Transformer,
trained on ~12M AlphaFoldDB structures for inverse folding) and extracts
per-residue encoder output for one chain of one PDB file. The encoder
takes ONLY backbone (N, CA, C) coordinates -- no amino-acid identity --
so this is a geometry-only embedding by construction, not a design
choice made here.

Output: .npz with res_ids (int, PDB numbering) and embeddings (L, 512).

CLI:
    python esm_if1_worker.py --pdb path/to/structure.pdb --chain A --out out.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import esm
import esm.inverse_folding.util as ifutil


def load_coords_and_res_ids(pdb_path: Path, chain: str):
    structure = ifutil.load_structure(str(pdb_path), chain)
    coords = ifutil.get_atom_coords_residuewise(["N", "CA", "C"], structure)
    res_ids, res_names = ifutil.get_residues(structure)
    return coords, res_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True, type=Path)
    p.add_argument("--chain", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    coords, res_ids = load_coords_and_res_ids(args.pdb, args.chain)
    with torch.no_grad():
        rep = ifutil.get_encoder_output(model, alphabet, coords)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, res_ids=res_ids, embeddings=rep.cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
