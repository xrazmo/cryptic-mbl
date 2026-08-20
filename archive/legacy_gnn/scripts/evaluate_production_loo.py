"""
evaluate_production_loo.py

In-distribution (leave-one-out) detection performance of the production
new_graph_flat ensemble, for structures it WAS trained on -- as opposed
to the challenge panels, which deliberately held out whole sequence
clusters to measure transfer to unseen families. This answers "how good
is detection of a family it has learned from," the same question
Berglund et al. 2017 (HMM built from 20 known B1 genes, leave-one-out
cross-validated) asked of their B1-only model, so results are reported
in the same shape: true-positive rate on the target cluster, plus false-
positive rate broken out by negative-family similarity (glyoxalase_ii/
rnase_z/phosphodiesterase/lactonase = other metallo-hydrolase-superfamily
folds, analogous to Berglund's "MBL superfamily" FPR figure;
tim_barrel/rossmann_sdr/alpha_beta_hydrolase/thioredoxin_fold/
globin_fold/lysozyme_like = unrelated folds, analogous to their
"random sequence" FPR figure).

Uses the SAME per-seed, never-average-coordinates methodology as
evaluate_per_seed.py and score_candidate.py: for each of the 8 seeds,
k=5 nearest neighbors against that seed's own reference-bank embeddings
with the query itself excluded (true leave-one-out, not just "not in
the training triplets" -- the point is to never let a query see its own
embedding in its own neighbor pool), majority vote per seed, then
majority vote across seeds.

CLI:
    python evaluate_production_loo.py --pockets-dir data/pockets \
        --catalog full_structure_catalog.csv \
        --reference-bank data/production/reference_bank \
        --split-graph data/split_graph.json \
        --out data/production_loo_evaluation.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

MBL_SUPERFAMILY_LIKE = {"glyoxalase_ii", "rnase_z", "phosphodiesterase", "lactonase"}
UNRELATED_FOLD = {"tim_barrel", "rossmann_sdr", "alpha_beta_hydrolase", "thioredoxin_fold", "globin_fold", "lysozyme_like"}


def load_metadata(pockets_dir: Path, catalog_path: Path) -> dict[str, dict]:
    meta = {}
    for f in pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        meta[pocket.metadata.source_structure_id] = {"label": pocket.metadata.label, "subclass": pocket.metadata.subclass}
    catalog = {r["accession"]: r for r in csv.DictReader(open(catalog_path))}
    for sid, row in catalog.items():
        if sid in meta:
            meta[sid]["neg_family"] = row.get("neg_family", "")
    return meta


def loo_predict_one_seed(query_idx: int, embeddings: np.ndarray, labels: np.ndarray, k: int) -> str:
    dists = np.linalg.norm(embeddings - embeddings[query_idx][None, :], axis=1)
    dists[query_idx] = np.inf  # exclude self
    neighbor_idx = np.argsort(dists)[:k]
    votes = Counter(labels[neighbor_idx])
    return votes.most_common(1)[0][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    meta = load_metadata(args.pockets_dir, args.catalog)
    split_graph = json.loads(args.split_graph.read_text())
    b1_b2_ids = set(split_graph["sequence_components"]["Sfh-I"])  # 116-member B1/B2 cluster (challenge_splits.json B1_B2_transfer's test_positive_component_ids)
    b3_ids = set(split_graph["sequence_components"]["AJP77057.1"])  # 26-member B3 cluster (challenge_splits.json B3_transfer's test_positive_component_ids)

    seed_files = sorted(args.reference_bank.glob("seed_*.npz"))
    assert seed_files, f"no seed_*.npz under {args.reference_bank}"

    per_seed_preds: list[dict[str, str]] = []
    for seed_file in seed_files:
        ref = np.load(seed_file)
        ids = ref["ids"]
        embeddings = ref["embeddings"]
        labels = ref["labels"]
        preds = {}
        for i, sid in enumerate(ids):
            preds[str(sid)] = loo_predict_one_seed(i, embeddings, labels, args.k)
        per_seed_preds.append(preds)
        log.info(f"{seed_file.name}: leave-one-out predictions computed for {len(ids)} structures")

    all_ids = list(per_seed_preds[0].keys())
    vote_preds = {}
    for sid in all_ids:
        votes = [p[sid] for p in per_seed_preds]
        vote_preds[sid] = Counter(votes).most_common(1)[0][0]

    def rate(ids: set[str], predicted_positive_is_hit: bool) -> dict:
        ids = [i for i in ids if i in vote_preds]
        n_hit = sum(1 for i in ids if (vote_preds[i] == "positive") == predicted_positive_is_hit)
        return {"n": len(ids), "rate": n_hit / max(len(ids), 1)}

    all_positive_ids = {sid for sid, m in meta.items() if m["label"] == "positive"}
    hard_neg_ids = {sid for sid, m in meta.items() if m.get("neg_family") in MBL_SUPERFAMILY_LIKE}
    easy_neg_ids = {sid for sid, m in meta.items() if m.get("neg_family") in UNRELATED_FOLD}

    report = {
        "methodology": (
            "Leave-one-out k-NN (k=5, Euclidean) per seed against that seed's own "
            "production reference-bank embedding, query excluded from its own neighbor "
            "pool; final call by majority vote across the 8 seeds. Same shape as "
            "Berglund et al. 2017's LOOCV of their B1-only HMM."
        ),
        "b1_b2_cluster_sensitivity_TPR": rate(b1_b2_ids, True),
        "b3_cluster_sensitivity_TPR": rate(b3_ids, True),
        "all_positives_sensitivity_TPR": rate(all_positive_ids, True),
        "false_positive_rate_vs_MBL_superfamily_like_negatives": rate(hard_neg_ids, True),
        "false_positive_rate_vs_unrelated_fold_negatives": rate(easy_neg_ids, True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
