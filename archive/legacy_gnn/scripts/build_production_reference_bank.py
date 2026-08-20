"""
build_production_reference_bank.py

After train_production_model.py finishes (8 seeds, trained on the full
1077-structure pool, final.pt only): embeds that same full pool once per
seed with that seed's own model, and saves each seed's embeddings to disk
separately -- this is the frozen "reference bank" future candidates are
scored against. Per-seed files, never merged/averaged, because seed
embeddings live in unrelated latent spaces (see evaluate_per_seed.py's
docstring) -- score_candidate.py does per-seed k-NN, never a cross-seed
average of coordinates.

Also computes and saves the training-free mean-pooled ESM2 embedding for
the same pool, as the frozen reference for the auxiliary raw-ESM2 score
(no model, no ensembling -- a single deterministic embedding per
structure).

CLI:
    python build_production_reference_bank.py --pockets-dir data/pockets \
        --esm2-dir data/esm2_embeddings --models-dir models/production \
        --out-dir data/production/reference_bank
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, PocketSubgraph
from graph_construction import pocket_to_pyg_data
from model import PocketEncoder, SiameseTripletModel

log = get_logger(__name__)


@torch.no_grad()
def embed_pool_one_seed(model: SiameseTripletModel, graphs: dict, ids: list[str], device: str) -> np.ndarray:
    model.eval()
    out = []
    for sid in ids:
        data = graphs[sid].to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        out.append(model.embed(data).squeeze(0).cpu().numpy())
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--esm2-dir", required=True, type=Path)
    p.add_argument("--models-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-seeds", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(f.stem for f in args.pockets_dir.glob("*.npz"))
    labels, subclasses = {}, {}
    for sid in ids:
        pocket = PocketSubgraph.load(args.pockets_dir / f"{sid}.npz")
        labels[sid] = "positive" if pocket.metadata.label == "positive" else "negative"
        subclasses[sid] = pocket.metadata.subclass

    log.info(f"Building graphs for the full {len(ids)}-structure reference pool...")
    graphs = {}
    mean_esm2 = {}
    for sid in ids:
        pocket = PocketSubgraph.load(args.pockets_dir / f"{sid}.npz")
        esm2_path = args.esm2_dir / f"{sid}.npy"
        esm2_emb = np.load(esm2_path) if esm2_path.exists() else None
        graphs[sid] = pocket_to_pyg_data(pocket, esm2_embeddings=esm2_emb)
        if esm2_emb is not None:
            mean_esm2[sid] = esm2_emb.mean(axis=0)
    in_dim = graphs[ids[0]].x.shape[1]

    for seed in range(args.n_seeds):
        ckpt = args.models_dir / f"seed_{seed}" / "final.pt"
        assert ckpt.exists(), f"missing {ckpt} -- run train_production_model.py first"
        encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        embeddings = embed_pool_one_seed(model, graphs, ids, device)
        out_path = args.out_dir / f"seed_{seed}.npz"
        np.savez(
            out_path, ids=np.array(ids), embeddings=embeddings,
            labels=np.array([labels[sid] for sid in ids]),
        )
        log.info(f"seed {seed}: saved {embeddings.shape} reference embeddings -> {out_path}")

    mean_esm2_ids = [sid for sid in ids if sid in mean_esm2]
    mean_esm2_matrix = np.stack([mean_esm2[sid] for sid in mean_esm2_ids])
    mean_esm2_path = args.out_dir / "mean_esm2.npz"
    np.savez(
        mean_esm2_path, ids=np.array(mean_esm2_ids), embeddings=mean_esm2_matrix,
        labels=np.array([labels[sid] for sid in mean_esm2_ids]),
    )
    log.info(f"mean-ESM2 auxiliary reference: saved {mean_esm2_matrix.shape} -> {mean_esm2_path}")
    if len(mean_esm2_ids) != len(ids):
        log.warning(f"{len(ids) - len(mean_esm2_ids)} structures had no ESM2 embedding file -- excluded from the auxiliary reference.")


if __name__ == "__main__":
    main()
