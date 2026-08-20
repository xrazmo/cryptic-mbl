"""
dump_predictions.py

Diagnostic: dumps per-structure kNN predictions (against the fold's own
train set, same methodology as evaluate.py's knn_classification_metrics)
for one ensemble on one fold, instead of just the aggregate confusion
matrix. Used to diff which specific structures flip between two model
variants (e.g. base vs richfeat) rather than only seeing that the FP count
changed.

--legacy-feature-dim lets this run against checkpoints trained before a
graph_construction.py feature-set change: it reconstructs the OLDER
feature layout by slicing the current (larger) feature vector back down --
valid only because new feature blocks were appended/inserted without
altering the columns that existed before them (AA one-hot[0:20],
sidechain[20], distance_to_metal[21] stay put; ESM2 block moves but its
content is unchanged) -- see graph_construction.py's module docstring for
the exact column layout at each point in time. This is a one-off
reproducibility aid for comparing across a feature-set transition, not a
general mechanism -- expect to update the slicing indices (or retire this
flag) if the layout changes again.

CLI:
    python dump_predictions.py --fold-json data/splits.json --fold-id 1 \
        --pockets-dir data/pockets --ensemble-dir models/fold_1_ensemble \
        --legacy-feature-dim 1302 --out /tmp/fold1_base_preds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, PocketSubgraph
from graph_construction import pocket_to_pyg_data
from model import PocketEncoder, SiameseTripletModel

log = get_logger(__name__)


def slice_to_legacy_1302(x: np.ndarray) -> np.ndarray:
    """AA[0:20] + sidechain[20] + dist_to_metal[21] unchanged; ESM2 block
    moved from [22:1302] to [37:1317] but content is identical."""
    return np.concatenate([x[:, :22], x[:, 37:]], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-json", required=True, type=Path)
    p.add_argument("--fold-id", required=True, type=int)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--ensemble-dir", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--legacy-feature-dim", type=int, default=None,
                    help="If set, slice features down to this dim (see module docstring).")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = json.loads(args.fold_json.read_text())["folds"]
    fold = next(f for f in folds if f["fold_id"] == args.fold_id)

    all_ids = sorted(set(fold["train"] + fold["test"]))
    graphs = {}
    for sid in all_ids:
        pocket = PocketSubgraph.load(args.pockets_dir / f"{sid}.npz")
        data = pocket_to_pyg_data(pocket)
        if args.legacy_feature_dim is not None:
            data.x = torch.tensor(slice_to_legacy_1302(data.x.numpy()), dtype=torch.float32)
        graphs[sid] = (data, pocket.metadata.label)

    in_dim = graphs[all_ids[0]][0].x.shape[1]
    models = []
    for seed_dir in sorted(args.ensemble_dir.glob("seed_*")):
        ckpt = seed_dir / "best.pt"
        if not ckpt.exists():
            continue
        encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models.append(model)
    log.info(f"Loaded {len(models)} ensemble members, in_dim={in_dim}")

    embeddings = {}
    with torch.no_grad():
        for sid, (data, _) in graphs.items():
            d = data.to(device)
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
            embs = np.stack([m.embed(d).squeeze(0).cpu().numpy() for m in models])
            embeddings[sid] = embs.mean(axis=0)

    train_ids = fold["train"]
    test_ids = fold["test"]
    train_embs = np.stack([embeddings[sid] for sid in train_ids])
    train_labels = ["positive" if graphs[sid][1] == "positive" else "negative" for sid in train_ids]

    rows = []
    for sid in test_ids:
        true_label = "positive" if graphs[sid][1] == "positive" else "negative"
        dists = np.linalg.norm(train_embs - embeddings[sid][None, :], axis=1)
        neighbors = np.argsort(dists)[:args.k]
        votes = Counter(train_labels[i] for i in neighbors)
        pred_label = votes.most_common(1)[0][0]
        rows.append({
            "structure_id": sid,
            "true_class": graphs[sid][1],
            "true_binary": true_label,
            "pred_binary": pred_label,
            "correct": true_label == pred_label,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "true_class", "true_binary", "pred_binary", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    n_fp = sum(1 for r in rows if r["true_binary"] == "negative" and r["pred_binary"] == "positive")
    n_fn = sum(1 for r in rows if r["true_binary"] == "positive" and r["pred_binary"] == "negative")
    log.info(f"Wrote {len(rows)} predictions -> {args.out} (FP={n_fp}, FN={n_fn})")


if __name__ == "__main__":
    main()
