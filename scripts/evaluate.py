"""
evaluate.py — Task 7 (spec §10)

Given a trained ensemble (directory of seed_*/best.pt checkpoints) and a
fold's test split + the external holdout, computes:
  - k-NN classification accuracy in embedding space (k=3-5) against the
    reference bank, for the fold's test set.
  - Recall@K of known positives ranked against the full hard-negative pool.
  - External validation: rank of curated environmental candidates relative
    to hard negatives (the key feasibility checkpoint before real metagenomes).
  - Embedding visualization (UMAP if installed, else PCA fallback),
    colored by subclass/label (subclass shown for inspection only — it is
    not the split axis, see clustering_split.py), saved as a static PNG.
  - All metrics additionally stratified by confidence_tier.

CLI:
    python evaluate.py --fold-json data/splits.json --fold-id 0 \
        --pockets-dir data/pockets --ensemble-dir models/fold_0_ensemble \
        --reference-bank-ids NDM-1 VIM-2 IMP-1 CphA Sfh-I L1 FEZ-1 \
        --external-ids AMM-1 SZM-1 CAM-2 \
        --out-dir results/fold_B2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, PocketSubgraph, load_esm2_embedding
from graph_construction import pocket_to_pyg_data
from model import PocketEncoder, SiameseTripletModel

log = get_logger(__name__)


def load_ensemble(
    ensemble_dir: Path, in_dim: int, device: str,
    ablate_distance_to_metal: bool = False,
) -> list[SiameseTripletModel]:
    models = []
    for seed_dir in sorted(ensemble_dir.glob("seed_*")):
        ckpt = seed_dir / "best.pt"
        if not ckpt.exists():
            continue
        config_path = seed_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            trained_ablation = bool(config.get("ablate_distance_to_metal", False))
            if trained_ablation != ablate_distance_to_metal:
                raise ValueError(
                    f"{seed_dir} ablation={trained_ablation}, but evaluation "
                    f"ablation={ablate_distance_to_metal}"
                )
        encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError(f"No checkpoints found under {ensemble_dir}")
    log.info(f"Loaded {len(models)} ensemble members from {ensemble_dir}")
    return models


@torch.no_grad()
def ensemble_embed(models: list[SiameseTripletModel], data, device: str):
    """Returns (mean_embedding, variance) across the ensemble for one instance."""
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    embs = np.stack([m.embed(data).squeeze(0).cpu().numpy() for m in models])
    return embs.mean(axis=0), float(embs.var(axis=0).mean())


def load_graphs(
    pockets_dir: Path, ids: list[str], ablate_distance_to_metal: bool = False,
    esm2_dir: Path = None,
) -> dict:
    graphs = {}
    for sid in ids:
        p = pockets_dir / f"{sid}.npz"
        pocket = PocketSubgraph.load(p)
        n_residues = len(np.unique(pocket.res_ids))
        esm2_emb = load_esm2_embedding(esm2_dir, sid, n_residues)
        graphs[sid] = (
            pocket_to_pyg_data(
                pocket, esm2_embeddings=esm2_emb, ablate_distance_to_metal=ablate_distance_to_metal,
            ),
            pocket.metadata,
        )
    return graphs


def knn_accuracy(
    query_embs: dict, query_labels: dict, reference_embs: dict,
    reference_labels: dict, k: int = 5,
) -> float:
    ref_ids = list(reference_embs.keys())
    ref_matrix = np.stack([reference_embs[i] for i in ref_ids])
    correct = 0
    for qid, qemb in query_embs.items():
        dists = np.linalg.norm(ref_matrix - qemb[None, :], axis=1)
        nn_idx = np.argsort(dists)[:k]
        votes = Counter(reference_labels[ref_ids[i]] for i in nn_idx)
        pred = votes.most_common(1)[0][0]
        if pred == query_labels[qid]:
            correct += 1
    return correct / max(len(query_embs), 1)


def knn_classification_metrics(
    query_embs: dict, query_labels: dict, reference_embs: dict,
    reference_labels: dict, k: int = 5,
) -> dict:
    """Binary held-out metrics, including class-balanced performance."""
    ref_ids = list(reference_embs)
    ref_matrix = np.stack([reference_embs[i] for i in ref_ids])
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for qid, qemb in query_embs.items():
        dists = np.linalg.norm(ref_matrix - qemb[None, :], axis=1)
        neighbors = np.argsort(dists)[:min(k, len(ref_ids))]
        votes = Counter(reference_labels[ref_ids[i]] for i in neighbors)
        pred = votes.most_common(1)[0][0]
        truth = query_labels[qid]
        if truth == "positive" and pred == "positive":
            counts["tp"] += 1
        elif truth == "negative" and pred == "negative":
            counts["tn"] += 1
        elif truth == "negative":
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    sensitivity = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    specificity = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    accuracy = (counts["tp"] + counts["tn"]) / max(len(query_embs), 1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion": counts,
    }


def recall_at_k(positive_embs: dict, negative_embs: dict, k_values=(1, 5, 10, 20)) -> dict:
    """
    For each positive, rank ALL other positives + all negatives by distance;
    recall@K = fraction of positives whose nearest-K neighbors include at
    least one other true positive.
    """
    pos_ids = list(positive_embs.keys())
    neg_ids = list(negative_embs.keys())
    all_ids = pos_ids + neg_ids
    all_matrix = np.stack([positive_embs[i] if i in positive_embs else negative_embs[i] for i in all_ids])
    is_positive = np.array([i in positive_embs for i in all_ids])

    results = {k: 0 for k in k_values}
    for pid in pos_ids:
        anchor_idx = all_ids.index(pid)
        dists = np.linalg.norm(all_matrix - all_matrix[anchor_idx][None, :], axis=1)
        dists[anchor_idx] = np.inf
        order = np.argsort(dists)
        for k in k_values:
            top_k = order[:k]
            if is_positive[top_k].any():
                results[k] += 1
    return {k: v / max(len(pos_ids), 1) for k, v in results.items()}


def external_validation_ranks(external_embs: dict, negative_embs: dict, positive_prototype: np.ndarray) -> dict:
    """
    For each external structure (AMM-1, SZM-1, CAM-2), rank its distance to
    the positive prototype against the distribution of hard-negative
    distances to the same prototype. Lower percentile = more MBL-like.
    """
    neg_dists = np.array([np.linalg.norm(e - positive_prototype) for e in negative_embs.values()])
    out = {}
    for sid, emb in external_embs.items():
        d = np.linalg.norm(emb - positive_prototype)
        percentile = float((neg_dists < d).mean() * 100)
        out[sid] = {"distance_to_prototype": float(d), "percentile_vs_hard_negatives": percentile}
    return out


def reference_bank_leave_one_out_ranks(
    reference_embs: dict, negative_embs: dict,
) -> dict:
    """Rank each reference against a prototype that does not contain itself."""
    if len(reference_embs) < 2:
        raise ValueError("Leave-one-out reference validation needs at least two references")
    out = {}
    for sid, emb in reference_embs.items():
        prototype = np.mean(
            [other for oid, other in reference_embs.items() if oid != sid], axis=0,
        )
        out[sid] = external_validation_ranks(
            {sid: emb}, negative_embs, prototype,
        )[sid]
    return out


def plot_embedding_space(embeddings: dict, labels: dict, subclasses: dict, out_path: Path):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=0)
        method = "UMAP"
    except Exception as exc:
        log.warning(f"UMAP unavailable ({exc}); falling back to PCA.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        method = "PCA"

    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in ids])
    coords_2d = reducer.fit_transform(matrix)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = {"B1": "tab:blue", "B2": "tab:orange", "B3": "tab:green",
                 "environmental": "tab:red", None: "tab:gray"}
    marker_map = {"positive": "o", "hard_negative": "x", "easy_negative": "+", "unlabeled": "."}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sid in enumerate(ids):
        ax.scatter(
            coords_2d[i, 0], coords_2d[i, 1],
            c=color_map.get(subclasses.get(sid)), marker=marker_map.get(labels.get(sid), "."),
            s=40, alpha=0.8,
        )
    ax.set_title(f"Pocket embedding space ({method})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info(f"Embedding plot saved -> {out_path}")


def stratify_by_tier(metrics_fn, graphs_by_tier: dict) -> dict:
    return {tier: metrics_fn(ids) for tier, ids in graphs_by_tier.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-json", required=True, type=Path)
    p.add_argument("--fold-id", required=True, type=int)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--ensemble-dir", required=True, type=Path)
    p.add_argument("--reference-bank-ids", nargs="+", required=True)
    p.add_argument("--external-ids", nargs="*", default=[])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ablate-distance-to-metal", action="store_true")
    p.add_argument("--esm2-dir", type=Path, default=None,
                    help="Directory of esm2_embed.py .npy outputs; omit to use zeros (default).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = json.loads(args.fold_json.read_text())["folds"]
    fold = next(f for f in folds if f["fold_id"] == args.fold_id)

    overlap = set(args.reference_bank_ids) & set(args.external_ids)
    if overlap:
        raise ValueError(
            "--external-ids must be independent of --reference-bank-ids; overlap: "
            + ", ".join(sorted(overlap))
        )

    all_needed_ids = fold["train"] + fold["test"] + args.reference_bank_ids + args.external_ids
    graphs = load_graphs(
        args.pockets_dir, sorted(set(all_needed_ids)),
        ablate_distance_to_metal=args.ablate_distance_to_metal, esm2_dir=args.esm2_dir,
    )

    in_dim = next(iter(graphs.values()))[0].x.shape[1]
    models = load_ensemble(
        args.ensemble_dir, in_dim, device,
        ablate_distance_to_metal=args.ablate_distance_to_metal,
    )

    embeddings, variances, labels, subclasses, tiers = {}, {}, {}, {}, {}
    for sid, (data, meta) in graphs.items():
        emb, var = ensemble_embed(models, data, device)
        embeddings[sid] = emb
        variances[sid] = var
        labels[sid] = meta.label
        subclasses[sid] = meta.subclass
        tiers[sid] = meta.confidence_tier

    ref_embs = {sid: embeddings[sid] for sid in args.reference_bank_ids}
    train_embs = {sid: embeddings[sid] for sid in fold["train"]}
    train_labels = {
        sid: "positive" if labels[sid] == "positive" else "negative"
        for sid in fold["train"]
    }
    test_pos_ids = [sid for sid in fold["test"] if labels[sid] == "positive"]
    test_neg_ids = [sid for sid in fold["test"] if labels[sid] in ("hard_negative", "easy_negative")]

    test_ids = test_pos_ids + test_neg_ids
    test_labels = {sid: "positive" if sid in test_pos_ids else "negative" for sid in test_ids}
    knn_metrics = knn_classification_metrics(
        {sid: embeddings[sid] for sid in test_ids}, test_labels,
        train_embs, train_labels, k=args.k,
    )
    recall = recall_at_k(
        {sid: embeddings[sid] for sid in test_pos_ids},
        {sid: embeddings[sid] for sid in test_neg_ids},
    )
    prototype = np.mean([embeddings[sid] for sid in args.reference_bank_ids], axis=0)
    external = external_validation_ranks(
        {sid: embeddings[sid] for sid in args.external_ids},
        {sid: embeddings[sid] for sid in test_neg_ids},
        prototype,
    )
    reference_loo = reference_bank_leave_one_out_ranks(
        ref_embs, {sid: embeddings[sid] for sid in test_neg_ids},
    )

    results = {
        "fold_id": args.fold_id,
        "knn_accuracy": knn_metrics["accuracy"],
        "knn": knn_metrics,
        "recall_at_k": recall,
        "external_validation": external,
        "reference_bank_leave_one_out": reference_loo,
        "ablate_distance_to_metal": args.ablate_distance_to_metal,
        "mean_ensemble_variance": {
            "test_positives": float(np.mean([variances[s] for s in test_pos_ids])) if test_pos_ids else None,
            "test_negatives": float(np.mean([variances[s] for s in test_neg_ids])) if test_neg_ids else None,
            "external": {s: variances[s] for s in args.external_ids},
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info(json.dumps(results, indent=2))

    plot_embedding_space(embeddings, labels, subclasses, args.out_dir / "embedding_space.png")


if __name__ == "__main__":
    main()
