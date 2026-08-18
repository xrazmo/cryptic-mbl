"""
mean_esm2_baseline.py

Cheapest possible diagnostic, run before any GNN training: mean-pool each
structure's pocket-residue ESM2 embeddings (esm2_embed.py output) into one
vector, classify each test example by k-NN majority vote against the
panel's own train set. No training, no graph, no geometry, no chemistry
features -- just "does raw sequence embedding similarity already solve this
panel." If it does, that's the signal to inspect before spending GPU time
on the full retraining matrix (per the reviewer's ordering: mean-ESM2 and
max-train-identity results for all three panels come before the training
matrix, not after).

CLI:
    python mean_esm2_baseline.py --challenge-splits data/challenge_splits.json \
        --esm2-dir data/esm2_embeddings --pockets-dir data/pockets --k 5 \
        --out data/mean_esm2_baseline.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)


def mean_embedding(structure_id: str, esm2_dir: Path) -> np.ndarray | None:
    path = esm2_dir / f"{structure_id}.npy"
    if not path.exists():
        return None
    emb = np.load(path)
    return emb.mean(axis=0)


def knn_classify(
    test_ids: list[str], train_ids: list[str], embeddings: dict[str, np.ndarray],
    labels: dict[str, str], k: int,
) -> dict:
    train_ids = [t for t in train_ids if t in embeddings]
    train_matrix = np.stack([embeddings[t] for t in train_ids])
    train_labels = [labels[t] for t in train_ids]

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_example = {}
    for tid in test_ids:
        if tid not in embeddings:
            continue
        dists = np.linalg.norm(train_matrix - embeddings[tid][None, :], axis=1)
        neighbors = np.argsort(dists)[:min(k, len(train_ids))]
        votes = Counter(train_labels[i] for i in neighbors)
        pred = votes.most_common(1)[0][0]
        truth = labels[tid]
        per_example[tid] = {"true": truth, "pred": pred}
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
        "n_test": n,
        "confusion": counts,
        "accuracy": (counts["tp"] + counts["tn"]) / max(n, 1),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "per_example": per_example,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--esm2-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    challenge = json.loads(args.challenge_splits.read_text())

    labels: dict[str, str] = {}
    for f in args.pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        raw = pocket.metadata.label
        labels[pocket.metadata.source_structure_id] = "positive" if raw == "positive" else "negative"

    all_ids = set()
    configs: dict[str, dict] = {}
    for name, panel in challenge["panels"].items():
        configs[f"panel:{name}"] = panel
        all_ids.update(panel["train_ids"] + panel["test_ids"])
    for name, cfg in challenge["leave_one_negative_family_out"].items():
        configs[f"lono:{name}"] = cfg
        all_ids.update(cfg["train_ids"] + cfg["test_ids"])

    log.info(f"Loading mean ESM2 embeddings for {len(all_ids)} structures...")
    embeddings = {}
    missing = 0
    for sid in all_ids:
        emb = mean_embedding(sid, args.esm2_dir)
        if emb is None:
            missing += 1
            continue
        embeddings[sid] = emb
    if missing:
        log.warning(f"{missing}/{len(all_ids)} structures had no ESM2 embedding file.")

    report = {}
    for config_name, cfg in configs.items():
        result = knn_classify(cfg["test_ids"], cfg["train_ids"], embeddings, labels, args.k)
        report[config_name] = result
        log.info(
            f"{config_name}: n_test={result['n_test']} acc={result['accuracy']:.3f} "
            f"bal_acc={result['balanced_accuracy']:.3f} sens={result['sensitivity']:.3f} "
            f"spec={result['specificity']:.3f} confusion={result['confusion']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote mean-ESM2 baseline results -> {args.out}")


if __name__ == "__main__":
    main()
