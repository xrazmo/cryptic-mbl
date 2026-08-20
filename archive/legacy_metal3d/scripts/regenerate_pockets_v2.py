"""
regenerate_pockets_v2.py

Full-pool pocket regeneration using the fixed pocket_extraction.py (see
"Fix upstream metal-site corruption and cross-chain pocket contamination"),
now that the 7-reference validation gate has passed
(reports/metal_site_fix_validation.json). Reads configs/manifest.csv for
label/tier/subclass/is_predicted (the same source data_assembly.py's
original run used), but extracts from data/domain_pdbs/{id}.pdb (single-
chain, already verified) instead of the raw catalog structure path.

Writes to data/pockets_v2/, NOT overwriting data/pockets/ -- keeps the
old (corrupted) pockets available for direct before/after comparison,
matching this project's established practice of versioned directories
(models/challenge_flat_v2 etc.) rather than in-place overwrites.

This regenerates pocket EXTRACTION only (fast, deterministic, no
training involved) -- explicitly NOT retraining any GNN checkpoint,
which stays a separate, later decision.

CLI:
    python regenerate_pockets_v2.py --manifest configs/manifest.csv \
        --domain-pdbs-dir data/domain_pdbs --out-dir data/pockets_v2 \
        --start-index 0
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from utils import get_logger
from pocket_extraction import extract_pocket

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--domain-pdbs-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--start-index", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest)))
    log.info(f"{len(rows)} manifest rows, starting at index {args.start_index}")

    n_ok, n_failed, n_skipped = 0, 0, 0
    t0 = time.time()
    for i, row in enumerate(rows):
        if i < args.start_index:
            continue
        sid = row["structure_id"]
        out_path = args.out_dir / f"{sid}.npz"
        if out_path.exists():
            n_skipped += 1
            continue

        domain_pdb = args.domain_pdbs_dir / f"{sid}.pdb"
        if not domain_pdb.exists():
            log.error(f"[{i}] {sid}: no domain_pdb at {domain_pdb}, skipping")
            n_failed += 1
            continue

        try:
            pocket = extract_pocket(
                structure_path=domain_pdb, structure_id=sid, label=row["label"],
                confidence_tier=int(row["tier"]), subclass=(row["subclass"] or None),
                is_predicted=row["is_predicted"].lower() in ("1", "true", "yes"),
            )
            pocket.save(out_path)
            n_ok += 1
        except Exception as e:
            log.error(f"[{i}] {sid}: extraction failed: {e}")
            n_failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info(f"  {i+1}/{len(rows)} processed ({n_ok} ok, {n_failed} failed, {n_skipped} skipped) -- {elapsed:.0f}s elapsed")

    log.info(f"Done. {n_ok} ok, {n_failed} failed, {n_skipped} already existed -> {args.out_dir}")


if __name__ == "__main__":
    main()
