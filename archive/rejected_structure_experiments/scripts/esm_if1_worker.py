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

Output: one .npz per structure, with res_ids (int, PDB numbering) and
embeddings (L, 512).

Two CLIs:
  single structure:
    python esm_if1_worker.py --pdb path/to/structure.pdb --chain A --out out.npz
  batch (loads the 141.6M-param model ONCE, not once per structure --
  reloading it repeatedly was ~5s/structure of pure model-load overhead,
  versus ~0.1-0.3s of actual inference; for 1077 structures this is the
  difference between ~95 minutes and a few minutes):
    python esm_if1_worker.py --batch-dir data/domain_pdbs --ids-file ids.txt \
        --chain A --out-dir data/outer_pocket_embeddings
"""

from __future__ import annotations

import argparse
import sys
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


def embed_one(model, alphabet, pdb_path: Path, chain: str) -> tuple[np.ndarray, np.ndarray]:
    coords, res_ids = load_coords_and_res_ids(pdb_path, chain)
    with torch.no_grad():
        rep = ifutil.get_encoder_output(model, alphabet, coords)
    return res_ids, rep.cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", type=Path, default=None)
    p.add_argument("--chain", default="A")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch-dir", type=Path, default=None)
    p.add_argument("--ids-file", type=Path, default=None, help="one structure_id per line; reads {batch-dir}/{id}.pdb")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    if args.batch_dir is not None:
        ids = [l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        n_ok, n_failed = 0, 0
        for i, sid in enumerate(ids):
            out_path = args.out_dir / f"{sid}.npz"
            if out_path.exists():
                continue
            pdb_path = args.batch_dir / f"{sid}.pdb"
            try:
                res_ids, emb = embed_one(model, alphabet, pdb_path, args.chain)
                np.savez(out_path, res_ids=res_ids, embeddings=emb)
                n_ok += 1
            except Exception as e:
                print(f"[{i}] {sid}: FAILED: {e}", file=sys.stderr)
                n_failed += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(ids)} ({n_ok} ok, {n_failed} failed)", flush=True)
        print(f"Done. {n_ok} ok, {n_failed} failed -> {args.out_dir}")
    else:
        assert args.pdb is not None and args.out is not None, "single-structure mode needs --pdb and --out"
        res_ids, emb = embed_one(model, alphabet, args.pdb, args.chain)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, res_ids=res_ids, embeddings=emb)


if __name__ == "__main__":
    main()
