"""
train.py — Task 6 (spec §8)

Trains one Siamese/triplet model on one fold. Run this script once per
(fold x seed) to build the 5-10 model deep ensemble the spec calls for;
`run_ensemble()` at the bottom drives that loop.

Batch construction batches TRIPLETS (not independent examples), re-mines
semi-hard negatives every `remine_every_n_epochs` epochs using the current
encoder's embeddings over the full train pool (spec §6, §8).

CLI (single model):
    python train.py --fold-json data/splits.json --fold-id 0 \
        --pockets-dir data/pockets --graphs-dir data/graphs \
        --seed 0 --out-dir models/fold_0_seed0

CLI (full ensemble for one fold):
    python train.py --fold-json data/splits.json --fold-id 0 \
        --pockets-dir data/pockets --graphs-dir data/graphs \
        --ensemble --n-seeds 8 --out-dir models/fold_0_ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils import get_logger, set_seed, PocketSubgraph, TIER_LOSS_WEIGHTS, load_esm2_embedding
from graph_construction import pocket_to_pyg_data, ESM2_SLICE
from model import PocketEncoder, BranchedPocketEncoder, SiameseTripletModel, triplet_loss
from triplet_sampling import random_triplets, semi_hard_triplets

log = get_logger(__name__)


def load_fold_graphs(
    pockets_dir: Path, ids: list[str], ablate_distance_to_metal: bool = False,
    esm2_dir: Optional[Path] = None,
    ablate_aa_identity: bool = False, ablate_structural: bool = False, ablate_esm2: bool = False,
) -> dict[str, object]:
    graphs = {}
    for sid in ids:
        npz_path = pockets_dir / f"{sid}.npz"
        pocket = PocketSubgraph.load(npz_path)
        n_residues = len(np.unique(pocket.res_ids))
        esm2_emb = load_esm2_embedding(esm2_dir, sid, n_residues)
        graph = pocket_to_pyg_data(
            pocket, esm2_embeddings=esm2_emb, ablate_distance_to_metal=ablate_distance_to_metal,
            ablate_aa_identity=ablate_aa_identity, ablate_structural=ablate_structural,
            ablate_esm2=ablate_esm2,
        )
        # Preserve the original class, rather than losing hard/easy negative
        # identity when it is converted to the binary tensor target.
        graph.label_name = pocket.metadata.label
        graphs[sid] = graph
    return graphs


def partition_by_label(ids: list[str], graphs: dict) -> dict[str, list[str]]:
    buckets = {"positive": [], "hard": [], "easy": []}
    for sid in ids:
        label = graphs[sid].label_name
        if label == "positive":
            buckets["positive"].append(sid)
        elif label == "hard_negative":
            buckets["hard"].append(sid)
        elif label == "easy_negative":
            buckets["easy"].append(sid)
    return buckets


@torch.no_grad()
def compute_embeddings(model: SiameseTripletModel, graphs: dict, ids: list[str], device: str) -> dict:
    model.eval()
    embeddings = {}
    for sid in ids:
        data = graphs[sid].to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        embeddings[sid] = model.embed(data).squeeze(0).cpu().numpy()
    return embeddings


def collate_triplet_batch(triplets, graphs, device):
    from torch_geometric.data import Batch
    anchors = Batch.from_data_list([graphs[t.anchor_id] for t in triplets]).to(device)
    positives = Batch.from_data_list([graphs[t.positive_id] for t in triplets]).to(device)
    negatives = Batch.from_data_list([graphs[t.negative_id] for t in triplets]).to(device)
    weights = torch.tensor([
        min(
            TIER_LOSS_WEIGHTS.get(graphs[t.anchor_id].confidence_tier, 1.0),
            TIER_LOSS_WEIGHTS.get(graphs[t.positive_id].confidence_tier, 1.0),
        )
        for t in triplets
    ], dtype=torch.float32, device=device)
    return anchors, positives, negatives, weights


def train_one_model(
    train_ids: list[str],
    val_ids: list[str],
    pockets_dir: Path,
    out_dir: Path,
    seed: int,
    n_epochs: int = 60,
    triplets_per_epoch: int = 2000,
    batch_size: int = 64,
    margin: float = 0.3,
    lr: float = 1e-4,
    remine_every_n_epochs: int = 3,
    ablate_distance_to_metal: bool = False,
    esm2_dir: Optional[Path] = None,
    ablate_aa_identity: bool = False, ablate_structural: bool = False, ablate_esm2: bool = False,
    architecture: str = "flat",
    esm_dropout_prob: float = 0.4,
    structural_aux_loss_weight: float = 0.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = train_ids + val_ids
    graphs = load_fold_graphs(
        pockets_dir, all_ids,
        ablate_distance_to_metal=ablate_distance_to_metal, esm2_dir=esm2_dir,
        ablate_aa_identity=ablate_aa_identity, ablate_structural=ablate_structural,
        ablate_esm2=ablate_esm2,
    )
    buckets = partition_by_label(train_ids, graphs)
    val_buckets = partition_by_label(val_ids, graphs)

    in_dim = graphs[all_ids[0]].x.shape[1]
    if architecture == "branched":
        encoder = BranchedPocketEncoder(
            structural_dim=ESM2_SLICE.start, esm_dropout_prob=esm_dropout_prob,
        ).to(device)
    elif architecture == "flat":
        encoder = PocketEncoder(in_dim=in_dim).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r} (expected 'flat' or 'branched')")
    model = SiameseTripletModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    negative_kind_lookup = {sid: "hard" for sid in buckets["hard"]}
    negative_kind_lookup.update({sid: "easy" for sid in buckets["easy"]})

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        if epoch == 0 or epoch % remine_every_n_epochs != 0:
            triplets = random_triplets(
                buckets["positive"], buckets["hard"], buckets["easy"], triplets_per_epoch, seed=seed * 1000 + epoch
            )
        else:
            embeddings = compute_embeddings(model, graphs, train_ids, device)
            triplets = semi_hard_triplets(
                buckets["positive"], buckets["hard"] + buckets["easy"], negative_kind_lookup,
                embeddings, margin, triplets_per_epoch, seed=seed * 1000 + epoch,
            )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i + batch_size]
            anchors, positives, negatives, weights = collate_triplet_batch(batch_triplets, graphs, device)
            if architecture == "branched":
                (a_emb, p_emb, n_emb), (a_struct, p_struct, n_struct) = model(
                    anchors, positives, negatives, return_components=True,
                )
                loss = triplet_loss(a_emb, p_emb, n_emb, margin=margin, sample_weight=weights)
                loss = loss + structural_aux_loss_weight * triplet_loss(
                    a_struct, p_struct, n_struct, margin=margin, sample_weight=weights,
                )
            else:
                a_emb, p_emb, n_emb = model(anchors, positives, negatives)
                loss = triplet_loss(a_emb, p_emb, n_emb, margin=margin, sample_weight=weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_loss = evaluate_val_loss(model, graphs, val_buckets, margin, device, seed=seed)
        log.info(
            f"[seed {seed}] epoch {epoch+1}/{n_epochs} "
            f"train_loss={epoch_loss / max(n_batches,1):.4f} val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "final.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "seed": seed, "margin": margin, "n_epochs": n_epochs, "in_dim": in_dim,
        "best_val_loss": best_val_loss,
        "ablate_distance_to_metal": ablate_distance_to_metal,
        "esm2_dir": str(esm2_dir) if esm2_dir is not None else None,
        "ablate_aa_identity": ablate_aa_identity, "ablate_structural": ablate_structural,
        "ablate_esm2": ablate_esm2,
        "architecture": architecture, "esm_dropout_prob": esm_dropout_prob,
        "structural_aux_loss_weight": structural_aux_loss_weight,
    }, indent=2))
    return out_dir / "best.pt"


@torch.no_grad()
def evaluate_val_loss(model, graphs, val_buckets, margin, device, seed, n_triplets=500):
    if len(val_buckets["positive"]) < 2 or (not val_buckets["hard"] and not val_buckets["easy"]):
        return float("nan")
    triplets = random_triplets(
        val_buckets["positive"], val_buckets["hard"], val_buckets["easy"], n_triplets, seed=seed
    )
    model.eval()
    losses = []
    for i in range(0, len(triplets), 64):
        anchors, positives, negatives, weights = collate_triplet_batch(triplets[i:i+64], graphs, device)
        a_emb, p_emb, n_emb = model(anchors, positives, negatives)
        losses.append(triplet_loss(a_emb, p_emb, n_emb, margin=margin).item())
    return float(np.mean(losses))


def run_ensemble(fold_json: Path, fold_id: int, pockets_dir: Path, out_dir: Path, n_seeds: int = 8, **kwargs):
    folds = json.loads(fold_json.read_text())["folds"]
    fold = next(f for f in folds if f["fold_id"] == fold_id)
    for seed in range(n_seeds):
        seed_dir = out_dir / f"seed_{seed}"
        log.info(f"Training ensemble member seed={seed} -> {seed_dir}")
        train_one_model(fold["train"], fold["val"], pockets_dir, seed_dir, seed=seed, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-json", required=True, type=Path)
    p.add_argument("--fold-id", required=True, type=int)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--ablate-distance-to-metal", action="store_true",
                   help="Zero the metal-distance node feature while preserving model shape.")
    p.add_argument("--esm2-dir", type=Path, default=None,
                   help="Directory of esm2_embed.py .npy outputs; omit to use zeros (default).")
    p.add_argument("--ablate-aa-identity", action="store_true",
                   help="Zero the 20-dim amino-acid one-hot block.")
    p.add_argument("--ablate-structural", action="store_true",
                   help="Zero the 17-dim chemistry/geometry block.")
    p.add_argument("--ablate-esm2", action="store_true",
                   help="Zero the ESM2 embedding block.")
    p.add_argument("--architecture", choices=["flat", "branched"], default="flat",
                   help="'flat': single GNN over concatenated features (default). "
                        "'branched': separate structural/ESM2 branches + fusion (see model.py).")
    p.add_argument("--esm-dropout-prob", type=float, default=0.4,
                   help="branched only: probability of zeroing a graph's projected ESM2 "
                        "embedding before fusion, during training.")
    p.add_argument("--structural-aux-loss-weight", type=float, default=0.3,
                   help="branched only: weight of the auxiliary structure-only triplet loss.")
    args = p.parse_args()

    ablation_kwargs = dict(
        ablate_distance_to_metal=args.ablate_distance_to_metal, esm2_dir=args.esm2_dir,
        ablate_aa_identity=args.ablate_aa_identity, ablate_structural=args.ablate_structural,
        ablate_esm2=args.ablate_esm2,
        architecture=args.architecture, esm_dropout_prob=args.esm_dropout_prob,
        structural_aux_loss_weight=args.structural_aux_loss_weight,
    )
    if args.ensemble:
        run_ensemble(args.fold_json, args.fold_id, args.pockets_dir, args.out_dir,
                     n_seeds=args.n_seeds, n_epochs=args.n_epochs, **ablation_kwargs)
    else:
        folds = json.loads(args.fold_json.read_text())["folds"]
        fold = next(f for f in folds if f["fold_id"] == args.fold_id)
        train_one_model(fold["train"], fold["val"], args.pockets_dir, args.out_dir,
                         seed=args.seed, n_epochs=args.n_epochs, **ablation_kwargs)


if __name__ == "__main__":
    main()
