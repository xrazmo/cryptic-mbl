"""
score_candidate.py

Production inference: scores one new candidate pocket against the frozen
reference bank built by build_production_reference_bank.py.

Per the frozen scoring rule -- never average latent embedding coordinates
across seeds:
  1. For each of the 8 new_graph_flat seeds: embed the candidate in that
     seed's own space, k-NN (k=5, Euclidean, matching every prior
     evaluation in this project) against that SAME seed's reference-bank
     embeddings, get one predicted label + one positive-neighbor fraction.
  2. Aggregate across seeds by majority vote (label) and by averaging the
     positive-neighbor fraction (a scalar score, not a coordinate -- valid
     to average).
  3. Separately, report the training-free raw-ESM2 5-NN score (mean-pooled
     ESM2 embedding, k-NN against the frozen mean_esm2 reference) as an
     auxiliary signal -- not fused into the GNN ensemble result, reported
     alongside it.

CLI:
    python score_candidate.py --pocket data/pockets/CANDIDATE.npz \
        --esm2-embedding data/esm2_embeddings/CANDIDATE.npy \
        --models-dir models/production --reference-bank data/production/reference_bank \
        --k 5
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

log = get_logger(__name__)


def knn_score(candidate_emb: np.ndarray, ref_embeddings: np.ndarray, ref_labels: np.ndarray, k: int) -> dict:
    dists = np.linalg.norm(ref_embeddings - candidate_emb[None, :], axis=1)
    neighbor_idx = np.argsort(dists)[:min(k, len(ref_labels))]
    neighbor_labels = ref_labels[neighbor_idx]
    positive_fraction = float(np.mean(neighbor_labels == "positive"))
    predicted = "positive" if positive_fraction > 0.5 else "negative"
    return {
        "predicted_label": predicted,
        "positive_neighbor_fraction": positive_fraction,
        "nearest_distance": float(dists[neighbor_idx[0]]),
    }


@torch.no_grad()
def embed_candidate_one_seed(model: SiameseTripletModel, data, device: str) -> np.ndarray:
    model.eval()
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    return model.embed(data).squeeze(0).cpu().numpy()


def score_gnn_ensemble(
    pocket: PocketSubgraph, esm2_emb: np.ndarray | None, models_dir: Path,
    reference_bank_dir: Path, n_seeds: int, k: int, device: str,
) -> dict:
    graph = pocket_to_pyg_data(pocket, esm2_embeddings=esm2_emb)
    in_dim = graph.x.shape[1]

    per_seed = []
    for seed in range(n_seeds):
        ckpt = models_dir / f"seed_{seed}" / "final.pt"
        ref_path = reference_bank_dir / f"seed_{seed}.npz"
        if not ckpt.exists() or not ref_path.exists():
            log.warning(f"seed {seed}: missing checkpoint or reference file, skipping")
            continue
        encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        candidate_emb = embed_candidate_one_seed(model, graph, device)

        ref = np.load(ref_path)
        result = knn_score(candidate_emb, ref["embeddings"], ref["labels"], k)
        result["seed"] = seed
        per_seed.append(result)

    votes = Counter(r["predicted_label"] for r in per_seed)
    ensemble_vote = votes.most_common(1)[0][0]
    ensemble_positive_fraction = float(np.mean([r["positive_neighbor_fraction"] for r in per_seed]))
    return {
        "n_seeds_used": len(per_seed),
        "ensemble_vote": ensemble_vote,
        "ensemble_positive_fraction": ensemble_positive_fraction,
        "per_seed": per_seed,
    }


def score_esm2_auxiliary(esm2_emb: np.ndarray | None, reference_bank_dir: Path, k: int) -> dict | None:
    if esm2_emb is None:
        return None
    mean_esm2_path = reference_bank_dir / "mean_esm2.npz"
    if not mean_esm2_path.exists():
        log.warning(f"{mean_esm2_path} missing, skipping auxiliary ESM2 score")
        return None
    ref = np.load(mean_esm2_path)
    candidate_mean = esm2_emb.mean(axis=0)
    return knn_score(candidate_mean, ref["embeddings"], ref["labels"], k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pocket", required=True, type=Path)
    p.add_argument("--esm2-embedding", type=Path, default=None)
    p.add_argument("--models-dir", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pocket = PocketSubgraph.load(args.pocket)
    esm2_emb = np.load(args.esm2_embedding) if args.esm2_embedding and args.esm2_embedding.exists() else None

    gnn_result = score_gnn_ensemble(
        pocket, esm2_emb, args.models_dir, args.reference_bank, args.n_seeds, args.k, device,
    )
    esm2_result = score_esm2_auxiliary(esm2_emb, args.reference_bank, args.k)

    output = {
        "structure_id": pocket.metadata.source_structure_id,
        "gnn_ensemble": gnn_result,
        "esm2_auxiliary": esm2_result,
    }
    log.info(
        f"{output['structure_id']}: GNN ensemble vote={gnn_result['ensemble_vote']} "
        f"(positive_fraction={gnn_result['ensemble_positive_fraction']:.3f}, "
        f"{gnn_result['n_seeds_used']} seeds) | "
        f"ESM2 auxiliary={esm2_result['predicted_label'] if esm2_result else 'n/a'}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2))
        log.info(f"Wrote score -> {args.out}")


if __name__ == "__main__":
    main()
