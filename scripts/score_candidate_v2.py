"""
score_candidate_v2.py

Production scoring regime v2: asymmetric, evidence-channel scoring, not
a forced universal binary label. Per the agreed architecture --

    B1/B2 channel:  corrected Metal3D complex -> DCH cysteine rule
    B3 channel:     ESM2 alone (outer-pocket encoder tested and
                     REJECTED by its predeclared go/no-go gate -- see
                     reports/outer_pocket_encoder_rejection.md; never
                     run during production scoring, not wired in here)
    Sequence:       mean ESM2 retrieval, with subclass/family neighbor
                     context and an OOD distance percentile
    Final output:   two evidence channels kept separate + one triage
                     label; no opaque weighted fusion.

Final triage (see combine_triage docstring for the exact matrix):
    dual_support | dch_only | esm2_only | unresolved

"unresolved" is the honest default whenever a channel's required input
is missing -- never silently coerced to "negative" (per the explicit
instruction that missingness must never read as a negative call).

legacy_gnn_v1 (the flat/branched GNN trained on the PRE-fix, corrupted
V1 pockets) is available via --include-legacy-gnn-v1 for reference/
comparison ONLY, disabled by default, and is NEVER part of the triage
decision -- it predates the metal-site corruption fix this whole branch
exists to correct and should not drive production calls.

This is a NEW, separately versioned scoring regime -- see
freeze_production_v2_manifest.py. It does not overwrite the frozen V1
manifest (reports/production_model_manifest.json) or V1 artifacts
(models/production/, data/production/reference_bank/).

CLI:
    python score_candidate_v2.py --pocket data/pockets_v2/CANDIDATE.npz \
        --esm2-embedding data/esm2_embeddings/CANDIDATE.npy \
        --reference-bank data/production/reference_bank_v2/mean_esm2_v2.npz \
        --k 5 [--include-legacy-gnn-v1 --legacy-models-dir models/production \
        --legacy-reference-bank data/production/reference_bank]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph
from dch_score import score_dch

log = get_logger(__name__)


def score_esm2_retrieval(esm2_emb: np.ndarray | None, reference_bank_path: Path, k: int) -> dict:
    if esm2_emb is None:
        return {"status": "unavailable", "neighbors": [], "positive_neighbor_fraction": None, "ood_percentile": None}

    ref = np.load(reference_bank_path, allow_pickle=False)
    ids, embeddings, labels = ref["ids"], ref["embeddings"], ref["labels"]
    subclasses = ref["subclasses"] if "subclasses" in ref else np.array([""] * len(ids))
    neg_families = ref["neg_families"] if "neg_families" in ref else np.array([""] * len(ids))
    loo_dists = ref["ood_calibration_sorted_loo_distances"] if "ood_calibration_sorted_loo_distances" in ref else None

    candidate_mean = esm2_emb.mean(axis=0)
    dists = np.linalg.norm(embeddings - candidate_mean[None, :], axis=1)
    order = np.argsort(dists)[:k]

    neighbors = [
        {
            "id": str(ids[i]), "distance": round(float(dists[i]), 4), "label": str(labels[i]),
            "subclass": str(subclasses[i]), "neg_family": str(neg_families[i]),
        }
        for i in order
    ]
    positive_fraction = float(np.mean(labels[order] == "positive"))
    status = "positive" if positive_fraction > 0.5 else "negative"

    ood_percentile = None
    if loo_dists is not None:
        nearest = float(dists[order[0]])
        ood_percentile = round(float(np.searchsorted(loo_dists, nearest) / len(loo_dists)), 4)

    return {
        "status": status, "positive_neighbor_fraction": positive_fraction,
        "neighbors": neighbors, "ood_percentile": ood_percentile,
    }


def combine_triage(dch_status: str, esm2_status: str) -> str:
    """
    dch_status in {supported, not_supported, unavailable}
    esm2_status in {positive, negative, unavailable}

    dual_support: dch supported AND esm2 positive.
    dch_only: dch supported AND esm2 not positive (negative or
        unavailable) -- highest-interest structurally-supported
        candidate, potentially sequence-remote.
    esm2_only: dch not supported (not_supported or unavailable) AND
        esm2 positive -- sequence-supported MBL-like candidate; could
        be B3, a noncanonical B1/B2, or a DCH extraction failure. Not
        automatically B3.
    unresolved: neither channel supports it, OR a required channel's
        input is unavailable with the other channel also not
        supporting -- never coerced to a negative call.
    """
    dch_support = dch_status == "supported"
    esm2_support = esm2_status == "positive"
    if dch_support and esm2_support:
        return "dual_support"
    if dch_support:
        return "dch_only"
    if esm2_support:
        return "esm2_only"
    return "unresolved"


def score_legacy_gnn_v1(pocket, esm2_dir, models_dir: Path, reference_bank_dir: Path, n_seeds: int, k: int, device: str) -> dict:
    from graph_construction import pocket_to_pyg_data
    from model import PocketEncoder, SiameseTripletModel
    from utils import load_esm2_embedding
    import torch

    # load_esm2_embedding, not a raw esm2_emb array: it falls back to None
    # (-> zeros) with a warning when the cached embedding's residue count
    # doesn't match this pocket, rather than crashing -- which it never
    # will for a v2 pocket, since pocket residue sets changed under the
    # metal-site-corruption fix (e.g. NDM-1: 293 residues pre-fix, 52
    # post-fix) and cached embeddings were never realigned. Acceptable
    # here specifically because this path is reference-only, off by
    # default, and excluded from the actual triage decision.
    n_residues = len(set(pocket.res_ids.tolist()))
    esm2_emb = load_esm2_embedding(esm2_dir, pocket.metadata.source_structure_id, n_residues)
    graph = pocket_to_pyg_data(pocket, esm2_embeddings=esm2_emb)
    in_dim = graph.x.shape[1]
    per_seed = []
    for seed in range(n_seeds):
        ckpt = models_dir / f"seed_{seed}" / "final.pt"
        ref_path = reference_bank_dir / f"seed_{seed}.npz"
        if not ckpt.exists() or not ref_path.exists():
            continue
        encoder = PocketEncoder(in_dim=in_dim).to(device)
        model = SiameseTripletModel(encoder).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        data = graph.to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            emb = model.embed(data).squeeze(0).cpu().numpy()
        ref = np.load(ref_path)
        dists = np.linalg.norm(ref["embeddings"] - emb[None, :], axis=1)
        neighbors = np.argsort(dists)[:k]
        votes = Counter(ref["labels"][neighbors])
        per_seed.append(votes.most_common(1)[0][0])
    if not per_seed:
        return {"status": "unavailable", "warning": "trained on pre-fix corrupted V1 pockets; reference only, not used in triage"}
    vote = Counter(per_seed).most_common(1)[0][0]
    return {
        "status": vote, "n_seeds_used": len(per_seed),
        "warning": "trained on pre-fix corrupted V1 pockets; reference only, not used in triage",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pocket", required=True, type=Path)
    p.add_argument("--esm2-embedding", type=Path, default=None)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--include-legacy-gnn-v1", action="store_true")
    p.add_argument("--legacy-models-dir", type=Path, default=Path("models/production"))
    p.add_argument("--legacy-reference-bank", type=Path, default=Path("data/production/reference_bank"))
    p.add_argument("--legacy-n-seeds", type=int, default=8)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pocket = PocketSubgraph.load(args.pocket)
    esm2_emb = np.load(args.esm2_embedding) if args.esm2_embedding and args.esm2_embedding.exists() else None

    dch = score_dch(pocket)
    esm2 = score_esm2_retrieval(esm2_emb, args.reference_bank, args.k)
    triage = combine_triage(dch["status"], esm2["status"])

    output = {
        "structure_id": pocket.metadata.source_structure_id,
        "scoring_regime": "v2_asymmetric",
        "dch_structural_support": dch,
        "esm2_retrieval": esm2,
        "final_triage": triage,
    }

    if args.include_legacy_gnn_v1:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output["legacy_gnn_v1"] = score_legacy_gnn_v1(
            pocket, args.esm2_embedding.parent if args.esm2_embedding else None,
            args.legacy_models_dir, args.legacy_reference_bank, args.legacy_n_seeds, args.k, device,
        )

    log.info(f"{output['structure_id']}: DCH={dch['status']} ESM2={esm2['status']} -> triage={triage}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2))
        log.info(f"Wrote score -> {args.out}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
