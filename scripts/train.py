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

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils import get_logger, set_seed, PocketSubgraph, TIER_LOSS_WEIGHTS
from graph_construction import pocket_to_pyg_data
from model import PocketEncoder, SiameseTripletModel, triplet_loss
from triplet_sampling import random_triplets, semi_hard_triplets

log = get_logger(__name__)


def load_fold_graphs(pockets_dir: Path, ids: list[str]) -> dict[str, object]:
    graphs = {}
    for sid in ids:
        npz_path = pockets_dir / f"{sid}.npz"
        pocket = PocketSubgraph.load(npz_path)
        graphs[sid] = pocket_to_pyg_data(pocket)
    return graphs


def partition_by_label(ids: list[str], graphs: dict) -> dict[str, list[str]]:
    buckets = {"positive": [], "hard": [], "easy": []}
    for sid in ids:
        y = graphs[sid].y.item()
        # label metadata isn't preserved on the Data object beyond y (1/0/-1);
        # negative "hard vs easy" distinction must come from the manifest.
        # Re-derive here from pocket_source / a side lookup passed in by caller
        # in the real pipeline; placeholder groups all y==0 as "hard".
        if y == 1:
            buckets["positive"].append(sid)
        elif y == 0:
            buckets["hard"].append(sid)
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = train_ids + val_ids
    graphs = load_fold_graphs(pockets_dir, all_ids)
    buckets = partition_by_label(train_ids, graphs)
    val_buckets = partition_by_label(val_ids, graphs)

    in_dim = graphs[all_ids[0]].x.shape[1]
    encoder = PocketEncoder(in_dim=in_dim).to(device)
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
    args = p.parse_args()

    if args.ensemble:
        run_ensemble(args.fold_json, args.fold_id, args.pockets_dir, args.out_dir,
                     n_seeds=args.n_seeds, n_epochs=args.n_epochs)
    else:
        folds = json.loads(args.fold_json.read_text())["folds"]
        fold = next(f for f in folds if f["fold_id"] == args.fold_id)
        train_one_model(fold["train"], fold["val"], args.pockets_dir, args.out_dir,
                         seed=args.seed, n_epochs=args.n_epochs)


if __name__ == "__main__":
    main()
