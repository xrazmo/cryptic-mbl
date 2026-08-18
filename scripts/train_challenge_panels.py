"""
train_challenge_panels.py

Trains and evaluates the current combined pipeline (chemistry + geometry +
ESM2, i.e. train.py's defaults with --esm2-dir) on component_challenge_split.py's
three panels, instead of the old stratified k-fold splits.

Validation is carved out of each panel's TRAIN set at the sequence-component
level (using data/split_graph.json's sequence_components -- the same
components the panel itself respects), never from test, and never splitting
a component across train/val. This is monitoring/checkpoint-selection only,
not a second independent test set -- with only 6 positive components total
and up to 2 already consumed by test+train's largest chunks, val's positive
coverage per panel is small and its signal should be read as weak (reported
alongside results, not hidden).

Evaluation reuses the trained ensemble's embeddings for k-NN against the
panel's own train set (same methodology as evaluate.py/mean_esm2_baseline.py),
so results are directly comparable to the mean-ESM2 baseline already run.

CLI:
    python train_challenge_panels.py --challenge-splits data/challenge_splits.json \
        --split-graph data/split_graph.json --pockets-dir data/pockets \
        --esm2-dir data/esm2_embeddings --models-out-dir models/challenge \
        --results-out data/challenge_training_results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, PocketSubgraph
from graph_construction import pocket_to_pyg_data
from model import PocketEncoder, SiameseTripletModel
from train import train_one_model

log = get_logger(__name__)


def carve_validation(
    train_ids: list[str], sequence_components: dict[str, list[str]], val_frac: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Splits train_ids into (train, val) at the component level -- val_frac
    of the TRAIN-assigned components (by member count), chosen deterministically
    by seed, never breaking a component across the two.

    Smallest components first (ties broken by a seeded shuffle): components
    here are extremely lumpy (one can be 280+ members), so picking in random
    order can wildly overshoot the target -- e.g. grabbing one large
    component first turned a 15% target into 47% for one panel. Smallest-first
    bounds the overshoot to the size of whichever component tips it over the
    target, instead of to an arbitrarily large one.
    """
    train_set = set(train_ids)
    comps_in_train = [m for m in sequence_components.values() if set(m) <= train_set]
    assert sum(len(m) for m in comps_in_train) == len(train_ids), \
        "some train ids are not fully covered by whole sequence components -- panel construction bug"

    rng = np.random.default_rng(seed)
    order = list(range(len(comps_in_train)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(comps_in_train[i]))  # smallest first; seeded shuffle only breaks ties

    target_val_size = int(val_frac * len(train_ids))
    val_ids, running = [], 0
    for i in order:
        if running >= target_val_size:
            break
        val_ids.extend(comps_in_train[i])
        running += len(comps_in_train[i])
    val_set = set(val_ids)
    new_train_ids = [t for t in train_ids if t not in val_set]
    return new_train_ids, val_ids


@torch.no_grad()
def ensemble_embed_all(models, graphs: dict, ids: list[str], device: str) -> dict[str, np.ndarray]:
    out = {}
    for sid in ids:
        data = graphs[sid].to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        embs = np.stack([m.embed(data).squeeze(0).cpu().numpy() for m in models])
        out[sid] = embs.mean(axis=0)
    return out


def knn_classify(test_ids, train_ids, embeddings, labels, k=5):
    train_ids = [t for t in train_ids if t in embeddings]
    train_matrix = np.stack([embeddings[t] for t in train_ids])
    train_labels = [labels[t] for t in train_ids]
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for tid in test_ids:
        if tid not in embeddings:
            continue
        dists = np.linalg.norm(train_matrix - embeddings[tid][None, :], axis=1)
        neighbors = np.argsort(dists)[:min(k, len(train_ids))]
        votes = Counter(train_labels[i] for i in neighbors)
        pred = votes.most_common(1)[0][0]
        truth = labels[tid]
        if truth == "positive" and pred == "positive":
            counts["tp"] += 1
        elif truth == "negative" and pred == "negative":
            counts["tn"] += 1
        elif truth == "negative":
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    n = sum(counts.values())
    sens = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    spec = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    return {
        "n_test": n, "confusion": counts,
        "accuracy": (counts["tp"] + counts["tn"]) / max(n, 1),
        "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--esm2-dir", required=True, type=Path)
    p.add_argument("--models-out-dir", required=True, type=Path)
    p.add_argument("--results-out", required=True, type=Path)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    challenge = json.loads(args.challenge_splits.read_text())
    split_graph = json.loads(args.split_graph.read_text())
    sequence_components = split_graph["sequence_components"]

    labels: dict[str, str] = {}
    for f in args.pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        labels[pocket.metadata.source_structure_id] = "positive" if pocket.metadata.label == "positive" else "negative"

    results = {}
    for panel_name, panel in challenge["panels"].items():
        log.info(f"=== Panel {panel_name} ===")
        train_ids, val_ids = carve_validation(panel["train_ids"], sequence_components, args.val_frac, seed=0)
        n_train_pos = sum(1 for t in train_ids if labels.get(t) == "positive")
        n_val_pos = sum(1 for t in val_ids if labels.get(t) == "positive")
        log.info(f"train={len(train_ids)} ({n_train_pos} positive), val={len(val_ids)} ({n_val_pos} positive)")

        ensemble_dir = args.models_out_dir / panel_name
        for seed in range(args.n_seeds):
            seed_dir = ensemble_dir / f"seed_{seed}"
            log.info(f"[{panel_name}] training seed={seed} -> {seed_dir}")
            train_one_model(
                train_ids, val_ids, args.pockets_dir, seed_dir, seed=seed,
                n_epochs=args.n_epochs, esm2_dir=args.esm2_dir, device=device,
            )

        # load ensemble, embed train+test, evaluate via k-NN against train
        all_needed = sorted(set(train_ids + panel["test_ids"]))
        graphs = {}
        for sid in all_needed:
            pocket = PocketSubgraph.load(args.pockets_dir / f"{sid}.npz")
            esm2_path = args.esm2_dir / f"{sid}.npy"
            esm2_emb = np.load(esm2_path) if esm2_path.exists() else None
            graphs[sid] = pocket_to_pyg_data(pocket, esm2_embeddings=esm2_emb)
        in_dim = next(iter(graphs.values())).x.shape[1]

        models = []
        n_used_final_fallback = 0
        for seed_dir in sorted(ensemble_dir.glob("seed_*")):
            ckpt = seed_dir / "best.pt"
            if not ckpt.exists():
                # val loss is NaN every epoch whenever val has <2 positives
                # (evaluate_val_loss's own guard), so best_val_loss (starts at
                # inf) never improves and best.pt never gets written -- fall
                # back to the unconditionally-saved final.pt rather than
                # silently dropping this seed from the ensemble.
                ckpt = seed_dir / "final.pt"
                n_used_final_fallback += 1
            if not ckpt.exists():
                continue
            encoder = PocketEncoder(in_dim=in_dim).to(device)
            m = SiameseTripletModel(encoder).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            models.append(m)
        log.info(f"[{panel_name}] loaded {len(models)} ensemble members for evaluation "
                 f"({n_used_final_fallback} used final.pt fallback, no val-loss improvement recorded)")

        embeddings = ensemble_embed_all(models, graphs, all_needed, device)
        eval_result = knn_classify(panel["test_ids"], train_ids, embeddings, labels)
        results[panel_name] = {
            "n_train": len(train_ids), "n_train_positive": n_train_pos,
            "n_val": len(val_ids), "n_val_positive": n_val_pos,
            "n_ensemble_members": len(models),
            **eval_result,
        }
        log.info(f"[{panel_name}] RESULT: {eval_result}")

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(results, indent=2))
    log.info(f"Wrote challenge training results -> {args.results_out}")


if __name__ == "__main__":
    main()
