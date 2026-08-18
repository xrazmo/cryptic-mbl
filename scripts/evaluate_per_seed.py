"""
evaluate_per_seed.py

Fixes the invalid ensemble aggregation in train_challenge_panels.py's
ensemble_embed_all(), which averaged raw embedding *coordinates* across
8 independently-trained metric-learning models
(`embs.mean(axis=0)` over models). Independently-trained triplet-loss
embeddings have no shared basis -- they can be arbitrarily rotated/
permuted relative to each other -- so averaging their coordinates is not
a valid ensembling operation. The k-NN results reported so far for
combined/flat_v2/branched are therefore not trustworthy "8-seed
consensus" numbers; this script re-evaluates the same already-trained
checkpoints (no retraining) the right way:

  1. Per seed: build a k-NN classifier against that seed's OWN train
     embeddings, classify each test example. This never mixes
     coordinates across seeds.
  2. Aggregate across seeds by MAJORITY VOTE over the 8 (or fewer, see
     final.pt fallback) per-seed predictions -- not by averaging
     anything continuous.
  3. Report per-seed sensitivity/specificity mean+std alongside the vote
     result, so single-seed variance is visible rather than hidden
     inside an averaged point.

For the branched architecture, also evaluates z_structure (the
structural branch's own embedding, pre-fusion) independently, using the
same per-seed + vote procedure against a same-modality train reference
set -- this is the only way to test whether the structural branch does
anything on its own, since the auxiliary loss only proves it was
optimized, not that it predicts.

The "original_flat" config re-derives the pre-Stage-2 graph
(1317-dim, single-column edge_attr, no radial-shell feature) locally,
via build_legacy_graph(), since models/challenge/*'s checkpoints were
trained before graph_construction.py grew the radial-shell block and
sequence-adjacency edges -- their input_proj layer shape is tied to
that older, narrower feature layout.

CLI:
    python evaluate_per_seed.py --challenge-splits data/challenge_splits.json \
        --pockets-dir data/pockets --esm2-dir data/esm2_embeddings \
        --out data/per_seed_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from utils import get_logger, PocketSubgraph
from graph_construction import (
    collapse_to_residue_level, _one_hot_aa, _chem_properties,
    compute_ligand_geometry, compute_backbone_dihedrals, compute_sasa,
    ESM2_DIM, ESM2_SLICE, EDGE_CUTOFF_DEFAULT,
)
from model import PocketEncoder, BranchedPocketEncoder, SiameseTripletModel

log = get_logger(__name__)

LEGACY_STRUCTURAL_DIM = 37  # pre-Stage-2: 20 aa + 1 sidechain + 1 dist_to_metal + 8 chem + 2 ligand_geom + 4 dihedral + 1 sasa


class _LegacyDistanceAwareConv(MessagePassing):
    """Exact pre-Stage-2 model.py::_DistanceAwareConv (message = f(x_i, x_j,
    dist) only, no edge_attr/sequence-adjacency term) -- reconstructed here,
    not in model.py, purely so models/challenge/*'s checkpoints (trained
    before the sequence-adjacency-flag change) remain loadable with matching
    weight shapes."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.SiLU()
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)
        return self.msg_mlp(torch.cat([x_i, x_j, dist], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class LegacyPocketEncoder(nn.Module):
    """Exact pre-Stage-2 model.py::PocketEncoder."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, embed_dim: int = 128, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([_LegacyDistanceAwareConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, data) -> torch.Tensor:
        x = self.input_proj(data.x)
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, data.pos, data.edge_index)
            x = norm(x + residual)
            x = self.dropout(x)
        pooled = global_mean_pool(x, data.batch if hasattr(data, "batch") else
                                   torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return F.normalize(self.readout(pooled), p=2, dim=-1)


def build_legacy_edges(centroids: np.ndarray, cutoff: float = EDGE_CUTOFF_DEFAULT):
    """Pre-Stage-2 build_edges: spatial-cutoff pairs only, edge_attr = (E, 1) distance."""
    n = centroids.shape[0]
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    src, dst = np.where((dist <= cutoff) & (~np.eye(n, dtype=bool)))
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = dist[src, dst].reshape(-1, 1).astype(np.float32)
    return edge_index, edge_attr


def build_legacy_graph(pocket: PocketSubgraph, esm2_embeddings: np.ndarray | None):
    """Reconstructs the exact pre-Stage-2 graph (in_dim=1317, edge_attr 1-col)
    that models/challenge/* was trained on, using graph_construction's
    unchanged per-feature helper functions directly."""
    from torch_geometric.data import Data

    residue_level = collapse_to_residue_level(pocket)
    n = len(residue_level["res_ids"])
    aa_onehot = np.stack([_one_hot_aa(rn) for rn in residue_level["res_names"]])
    sidechain_flag = residue_level["has_sidechain"].reshape(-1, 1)
    chem_props = np.stack([_chem_properties(rn) for rn in residue_level["res_names"]])

    metal_coord = pocket.metal_coord
    if metal_coord is not None:
        dist_to_metal = np.linalg.norm(
            residue_level["centroids"] - metal_coord[None, :], axis=1
        ).reshape(-1, 1).astype(np.float32)
    else:
        dist_to_metal = np.zeros((n, 1), dtype=np.float32)

    ligand_geometry = compute_ligand_geometry(pocket, residue_level, metal_coord)
    dihedrals = compute_backbone_dihedrals(pocket, residue_level)
    sasa = compute_sasa(pocket, residue_level)

    if esm2_embeddings is not None:
        esm_block = esm2_embeddings.astype(np.float32)
    else:
        esm_block = np.zeros((n, ESM2_DIM), dtype=np.float32)

    x = np.concatenate([
        aa_onehot, sidechain_flag, dist_to_metal, chem_props, ligand_geometry, dihedrals, sasa, esm_block,
    ], axis=1)
    assert x.shape[1] == LEGACY_STRUCTURAL_DIM + ESM2_DIM

    edge_index, edge_attr = build_legacy_edges(residue_level["centroids"])
    label_map = {"positive": 1, "hard_negative": 0, "easy_negative": 0, "unlabeled": -1}
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(residue_level["centroids"], dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([label_map[pocket.metadata.label]], dtype=torch.long),
    )
    return data


def load_graphs(ids: list[str], pockets_dir: Path, esm2_dir: Path, legacy: bool) -> dict:
    from graph_construction import pocket_to_pyg_data
    graphs = {}
    for sid in ids:
        pocket = PocketSubgraph.load(pockets_dir / f"{sid}.npz")
        esm2_path = esm2_dir / f"{sid}.npy"
        esm2_emb = np.load(esm2_path) if esm2_path.exists() else None
        graphs[sid] = build_legacy_graph(pocket, esm2_emb) if legacy else pocket_to_pyg_data(pocket, esm2_embeddings=esm2_emb)
    return graphs


def seed_checkpoints(ensemble_dir: Path) -> list[tuple[int, Path]]:
    """Returns [(seed, ckpt_path)], best.pt if present else final.pt, same
    fallback rule as train_challenge_panels.py (see its docstring)."""
    out = []
    for seed_dir in sorted(ensemble_dir.glob("seed_*")):
        ckpt = seed_dir / "best.pt"
        if not ckpt.exists():
            ckpt = seed_dir / "final.pt"
        if ckpt.exists():
            out.append((int(seed_dir.name.split("_")[1]), ckpt))
    return out


@torch.no_grad()
def embed_all(model: SiameseTripletModel, graphs: dict, ids: list[str], device: str, structure_only: bool) -> dict:
    model.eval()
    out = {}
    for sid in ids:
        data = graphs[sid].to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        if structure_only:
            fused, z_structure = model.encoder(data, return_components=True)
            out[sid] = z_structure.squeeze(0).cpu().numpy()
        else:
            out[sid] = model.embed(data).squeeze(0).cpu().numpy()
    return out


def knn_predict_one_seed(train_ids, test_ids, embeddings, labels, k=5) -> dict[str, str]:
    train_ids = [t for t in train_ids if t in embeddings]
    train_matrix = np.stack([embeddings[t] for t in train_ids])
    train_labels = [labels[t] for t in train_ids]
    preds = {}
    for tid in test_ids:
        if tid not in embeddings:
            continue
        dists = np.linalg.norm(train_matrix - embeddings[tid][None, :], axis=1)
        neighbors = np.argsort(dists)[:min(k, len(train_ids))]
        votes = Counter(train_labels[i] for i in neighbors)
        preds[tid] = votes.most_common(1)[0][0]
    return preds


def score(preds: dict[str, str], labels: dict[str, str]) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for tid, pred in preds.items():
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
        "n_test": n, "confusion": counts, "accuracy": (counts["tp"] + counts["tn"]) / max(n, 1),
        "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
    }


def vote_aggregate(per_seed_preds: list[dict[str, str]], test_ids: list[str]) -> dict[str, str]:
    out = {}
    for tid in test_ids:
        votes = [p[tid] for p in per_seed_preds if tid in p]
        if not votes:
            continue
        out[tid] = Counter(votes).most_common(1)[0][0]
    return out


def evaluate_config(
    name: str, ensemble_dir: Path, panel: dict, labels: dict, pockets_dir: Path, esm2_dir: Path,
    device: str, legacy: bool, architecture: str, structure_only: bool,
) -> dict:
    all_ids = sorted(set(panel["train_ids"] + panel["test_ids"]))
    graphs = load_graphs(all_ids, pockets_dir, esm2_dir, legacy=legacy)
    in_dim = graphs[all_ids[0]].x.shape[1]

    ckpts = seed_checkpoints(ensemble_dir)
    per_seed_scores = []
    per_seed_preds = []
    for seed, ckpt in ckpts:
        if architecture == "branched":
            encoder = BranchedPocketEncoder(structural_dim=LEGACY_STRUCTURAL_DIM if legacy else ESM2_SLICE.start).to(device)
        elif legacy:
            encoder = LegacyPocketEncoder(in_dim=in_dim).to(device)
        else:
            encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        embeddings = embed_all(model, graphs, all_ids, device, structure_only=structure_only)
        preds = knn_predict_one_seed(panel["train_ids"], panel["test_ids"], embeddings, labels)
        per_seed_preds.append(preds)
        per_seed_scores.append(score(preds, labels))

    vote_preds = vote_aggregate(per_seed_preds, panel["test_ids"])
    vote_result = score(vote_preds, labels)

    sens_list = [s["sensitivity"] for s in per_seed_scores]
    spec_list = [s["specificity"] for s in per_seed_scores]
    bal_list = [s["balanced_accuracy"] for s in per_seed_scores]
    return {
        "config": name, "n_seeds_used": len(ckpts),
        "vote_ensemble": vote_result,
        "per_seed_sensitivity": {"mean": float(np.mean(sens_list)), "std": float(np.std(sens_list)), "values": sens_list},
        "per_seed_specificity": {"mean": float(np.mean(spec_list)), "std": float(np.std(spec_list)), "values": spec_list},
        "per_seed_balanced_accuracy": {"mean": float(np.mean(bal_list)), "std": float(np.std(bal_list)), "values": bal_list},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--esm2-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    challenge = json.loads(args.challenge_splits.read_text())

    labels: dict[str, str] = {}
    for f in args.pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        labels[pocket.metadata.source_structure_id] = "positive" if pocket.metadata.label == "positive" else "negative"

    configs = [
        ("original_flat", Path("models/challenge"), True, "flat", False),
        ("new_graph_flat", Path("models/challenge_flat_v2"), False, "flat", False),
        ("branched_fused", Path("models/challenge_branched"), False, "branched", False),
        ("branched_structure_only", Path("models/challenge_branched"), False, "branched", True),
    ]

    report = {}
    for panel_name, panel in challenge["panels"].items():
        log.info(f"=== Panel {panel_name} ===")
        report[panel_name] = {}
        for name, ensemble_root, legacy, architecture, structure_only in configs:
            ensemble_dir = ensemble_root / panel_name
            if not ensemble_dir.exists():
                log.warning(f"  {name}: {ensemble_dir} missing, skipping")
                continue
            log.info(f"  evaluating {name} ...")
            result = evaluate_config(
                name, ensemble_dir, panel, labels, args.pockets_dir, args.esm2_dir,
                device, legacy, architecture, structure_only,
            )
            report[panel_name][name] = result
            v = result["vote_ensemble"]
            log.info(
                f"  {name}: n_seeds={result['n_seeds_used']} VOTE sens={v['sensitivity']:.3f} "
                f"spec={v['specificity']:.3f} bal_acc={v['balanced_accuracy']:.3f} | "
                f"per-seed sens {result['per_seed_sensitivity']['mean']:.3f}+/-{result['per_seed_sensitivity']['std']:.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote per-seed evaluation -> {args.out}")


if __name__ == "__main__":
    main()
