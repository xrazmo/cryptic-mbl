"""
evaluate_coordination_fingerprint.py

Structural V2 part 1, continued: does the coordination fingerprint
(coordination_fingerprint.py) carry real signal, before spending any
engineering effort on a bigger encoder? Per the agreed plan, feed it to
a regularized GBM and a subclass-specific prototype-distance classifier
first -- not a GNN -- since 146 positives / 6 independent lineages is
too little for a flexible network to learn metal-site physics reliably.

Evaluated on the SAME leave-cluster-out challenge panels as everything
else in this project (component_challenge_split.py), and reported in
the SAME shape as the earlier structure-only-branch check: sensitivity/
specificity/balanced accuracy, plus the overlap-with-ESM2-baseline
table (does this recover any of the positives the raw-ESM2 5-NN
baseline missed, and at what false-positive cost) -- aggregate
sensitivity alone was shown to be misleading once we looked at overlap
in the branched-architecture analysis.

Two classifiers:
  - GBM: sklearn HistGradientBoostingClassifier (handles NaN features
    natively -- coordination_fingerprint.py leaves a feature NaN rather
    than fabricating a value when e.g. no donor is found within range).
  - Prototype: nearest-centroid in per-feature-standardized space
    (mean/std from TRAIN only), one centroid per positive subclass
    (B1/B2/B3/UNCLASSIFIED) and one per negative family
    (glyoxalase_ii/rnase_z/phosphodiesterase/lactonase/etc.) -- the
    "subclass-specific prototype distance" option from the agreed plan.
    Distance is computed only over the dimensions where both the query
    and the centroid are non-NaN (nan-aware Euclidean), normalized by
    the number of valid dimensions so partial fingerprints (e.g. no
    metal found) aren't penalized just for having fewer valid features.

CLI:
    python evaluate_coordination_fingerprint.py \
        --fingerprint data/coordination_fingerprint.npz \
        --challenge-splits data/challenge_splits.json \
        --mean-esm2-baseline data/mean_esm2_baseline.json \
        --catalog full_structure_catalog.csv \
        --out data/coordination_fingerprint_evaluation.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import get_logger

log = get_logger(__name__)


def load_fingerprints(path: Path):
    d = np.load(path, allow_pickle=False)
    ids = list(d["ids"])
    features = d["features"]
    labels = {sid: lab for sid, lab in zip(ids, d["labels"])}
    subclasses = {sid: sub for sid, sub in zip(ids, d["subclasses"])}
    return {sid: features[i] for i, sid in enumerate(ids)}, labels, subclasses


def score(preds: dict[str, str], labels: dict[str, str]) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for tid, pred in preds.items():
        truth = "positive" if labels[tid] == "positive" else "negative"
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


def train_gbm(train_ids, features, labels):
    X = np.stack([features[t] for t in train_ids])
    y = np.array([1 if labels[t] == "positive" else 0 for t in train_ids])
    # class_weight="balanced": train pools here are 1:8 to 1:18 positive:negative
    # (e.g. 26 positive / 440 negative on B1_B2_transfer) -- an unweighted fit's
    # easiest loss-minimizing solution is close to "always predict negative",
    # which is what an earlier unweighted run in fact did (near-zero
    # sensitivity across every panel).
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, random_state=0, class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def predict_gbm(clf, test_ids, features) -> dict[str, str]:
    X = np.stack([features[t] for t in test_ids])
    preds = clf.predict(X)
    return {t: ("positive" if p == 1 else "negative") for t, p in zip(test_ids, preds)}


def nan_aware_dist(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 3:  # too few comparable dims to trust
        return float("inf")
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def build_prototypes(train_ids, features, labels, subclasses, neg_family_of):
    X = np.stack([features[t] for t in train_ids])
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0

    def standardize(v):
        return (v - mean) / std

    groups: dict[str, list[np.ndarray]] = {}
    for t in train_ids:
        if labels[t] == "positive":
            key = f"positive:{subclasses.get(t) or 'UNCLASSIFIED'}"
        else:
            fam = neg_family_of.get(t, "") or "unknown"
            key = f"negative:{fam}"
        groups.setdefault(key, []).append(standardize(features[t]))

    import warnings
    prototypes = {}
    for key, vecs in groups.items():
        stacked = np.stack(vecs)
        with warnings.catch_warnings():
            # benign: a feature dimension where every member of this group is
            # NaN (e.g. no metal found for any train example of this class)
            # correctly nanmean()s to NaN -- nan_aware_dist skips it later.
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            prototypes[key] = np.nanmean(stacked, axis=0)
    return prototypes, mean, std


def predict_prototype(test_ids, features, prototypes, mean, std) -> dict[str, str]:
    preds = {}
    for t in test_ids:
        v = (features[t] - mean) / std
        best_key, best_dist = None, float("inf")
        for key, proto in prototypes.items():
            d = nan_aware_dist(v, proto)
            if d < best_dist:
                best_dist, best_key = d, key
        preds[t] = "positive" if best_key is not None and best_key.startswith("positive:") else "negative"
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fingerprint", required=True, type=Path)
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--mean-esm2-baseline", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    features, labels, subclasses = load_fingerprints(args.fingerprint)
    challenge = json.loads(args.challenge_splits.read_text())
    baseline = json.loads(args.mean_esm2_baseline.read_text())
    catalog = {r["accession"]: r for r in csv.DictReader(open(args.catalog))}
    neg_family_of = {sid: row.get("neg_family", "") for sid, row in catalog.items()}

    report = {}
    for panel_name in ["B1_B2_transfer", "B3_transfer", "remote_outlier"]:
        panel = challenge["panels"][panel_name]
        train_ids = [t for t in panel["train_ids"] if t in features]
        test_ids = [t for t in panel["test_ids"] if t in features]

        gbm = train_gbm(train_ids, features, labels)
        gbm_preds = predict_gbm(gbm, test_ids, features)
        gbm_result = score(gbm_preds, labels)

        prototypes, mean, std = build_prototypes(train_ids, features, labels, subclasses, neg_family_of)
        proto_preds = predict_prototype(test_ids, features, prototypes, mean, std)
        proto_result = score(proto_preds, labels)

        esm2_preds = {t: baseline[f"panel:{panel_name}"]["per_example"].get(t, {}).get("pred") for t in test_ids}
        test_positive_ids = [t for t in test_ids if labels[t] == "positive"]
        esm2_miss = [t for t in test_positive_ids if esm2_preds.get(t) != "positive"]
        gbm_recovers = [t for t in esm2_miss if gbm_preds.get(t) == "positive"]
        proto_recovers = [t for t in esm2_miss if proto_preds.get(t) == "positive"]

        report[panel_name] = {
            "n_train": len(train_ids), "n_test": len(test_ids), "n_esm2_misses": len(esm2_miss),
            "gbm": {**gbm_result, "recovers_esm2_misses": len(gbm_recovers), "recovered_ids": gbm_recovers},
            "prototype": {**proto_result, "recovers_esm2_misses": len(proto_recovers), "recovered_ids": proto_recovers},
        }
        log.info(
            f"{panel_name}: n_test_pos={len(test_positive_ids)} esm2_misses={len(esm2_miss)} | "
            f"GBM sens={gbm_result['sensitivity']:.3f} spec={gbm_result['specificity']:.3f} "
            f"bal_acc={gbm_result['balanced_accuracy']:.3f} recovers={len(gbm_recovers)}/{len(esm2_miss)} | "
            f"Prototype sens={proto_result['sensitivity']:.3f} spec={proto_result['specificity']:.3f} "
            f"bal_acc={proto_result['balanced_accuracy']:.3f} recovers={len(proto_recovers)}/{len(esm2_miss)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote coordination-fingerprint evaluation -> {args.out}")


if __name__ == "__main__":
    main()
