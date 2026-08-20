"""
build_reference_bank_v2.py

Enriches the existing mean-ESM2 reference bank
(data/production/reference_bank/mean_esm2.npz -- ids/embeddings/labels
only) with subclass and negative-family metadata (joined from
configs/manifest.csv and full_structure_catalog.csv), and adds an
out-of-distribution (OOD) distance calibration: for every reference
member, its distance to its own nearest OTHER reference member
(leave-one-out, matching this project's established LOO methodology --
see evaluate_production_loo.py), giving a real empirical distribution
to place a new candidate's nearest-neighbor distance against as a
percentile, rather than an arbitrary fixed cutoff.

The embeddings themselves are NOT recomputed -- ESM2 embeddings come
from esm2_embed.py's per-sequence inference, which was never affected
by the Metal3D/pocket-extraction corruption bug fixed in this branch
(sequences are independent of predicted metal geometry).

CLI:
    python build_reference_bank_v2.py \
        --reference-bank data/production/reference_bank/mean_esm2.npz \
        --manifest configs/manifest.csv --catalog full_structure_catalog.csv \
        --out data/production/reference_bank_v2/mean_esm2_v2.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils import get_logger

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    ref = np.load(args.reference_bank)
    ids, embeddings, labels = list(ref["ids"]), ref["embeddings"], list(ref["labels"])

    manifest_rows = {r["structure_id"]: r for r in csv.DictReader(open(args.manifest))}
    catalog_rows = {r["accession"]: r for r in csv.DictReader(open(args.catalog))}

    subclasses, neg_families = [], []
    n_missing_meta = 0
    for sid in ids:
        sub = manifest_rows.get(sid, {}).get("subclass", "") or ""
        fam = catalog_rows.get(sid, {}).get("neg_family", "") or ""
        if sid not in manifest_rows and sid not in catalog_rows:
            n_missing_meta += 1
        subclasses.append(sub)
        neg_families.append(fam)
    if n_missing_meta:
        log.warning(f"{n_missing_meta}/{len(ids)} reference ids had no manifest or catalog row -- subclass/family left empty")

    # Leave-one-out OOD calibration: each member's distance to its own
    # nearest OTHER member, for placing a new candidate's distance as a
    # percentile against the known distribution.
    log.info("Computing leave-one-out nearest-neighbor distances for OOD calibration...")
    n = len(ids)
    loo_nearest_dists = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(embeddings - embeddings[i][None, :], axis=1)
        dists[i] = np.inf
        loo_nearest_dists[i] = dists.min()
    loo_nearest_dists.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, ids=np.array(ids), embeddings=embeddings, labels=np.array(labels),
        subclasses=np.array(subclasses), neg_families=np.array(neg_families),
        ood_calibration_sorted_loo_distances=loo_nearest_dists,
    )
    log.info(f"Wrote enriched reference bank v2 -> {args.out} ({n} members, k={args.k})")
    log.info(
        f"LOO nearest-neighbor distance distribution: min={loo_nearest_dists[0]:.4f} "
        f"p50={loo_nearest_dists[n//2]:.4f} p90={loo_nearest_dists[int(0.9*n)]:.4f} "
        f"p99={loo_nearest_dists[int(0.99*n)]:.4f} max={loo_nearest_dists[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
