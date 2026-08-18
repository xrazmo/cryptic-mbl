"""
train_production_model.py

Final production training run, per the frozen decision: new_graph_flat
(the flat architecture over the current, Stage-2 graph -- sequence-adjacency
edges + radial-shell feature, no branched fusion) trained once on the
*entire* labeled pool (all 1077 structures: 146 positive, 931 negative),
not a held-out panel. No validation carve, no checkpoint selection --
best.pt is never written when val is empty (evaluate_val_loss's own NaN
guard means best_val_loss, which starts at inf, never improves), which is
exactly the desired behavior: every seed's final.pt is the one and only
artifact. Same architecture, epoch count (60), and hyperparameters as
new_graph_flat's evaluation run -- no further tuning here.

This is architecturally identical to train_challenge_panels.py's per-seed
training loop, just over the full pool with an empty val set instead of a
panel's train/val split -- kept as its own script rather than a flag on
that one, since "train on everything, no split, no panel bookkeeping" is
a meaningfully different and simpler operation, not a variant of it.

CLI:
    python train_production_model.py --pockets-dir data/pockets \
        --esm2-dir data/esm2_embeddings --out-dir models/production \
        --n-seeds 8 --n-epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import get_logger, PocketSubgraph
from train import train_one_model

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--esm2-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--start-seed", type=int, default=0,
                    help="Resume from this seed (e.g. after an interrupted run); "
                         "seeds before it are left untouched.")
    args = p.parse_args()

    all_ids = sorted(f.stem for f in args.pockets_dir.glob("*.npz"))
    n_pos = sum(
        1 for sid in all_ids
        if PocketSubgraph.load(args.pockets_dir / f"{sid}.npz").metadata.label == "positive"
    )
    log.info(f"Training on the full pool: {len(all_ids)} structures ({n_pos} positive, {len(all_ids) - n_pos} negative)")
    log.info("No validation carve (val_ids=[]) -- best.pt will not be written (NaN val loss guard), "
              "final.pt is the single artifact per seed, as intended.")

    for seed in range(args.start_seed, args.n_seeds):
        seed_dir = args.out_dir / f"seed_{seed}"
        if (seed_dir / "final.pt").exists():
            log.info(f"seed={seed}: final.pt already exists at {seed_dir}, skipping (resume-safe).")
            continue
        log.info(f"Training production seed={seed} -> {seed_dir}")
        train_one_model(
            train_ids=all_ids, val_ids=[], pockets_dir=args.pockets_dir, out_dir=seed_dir,
            seed=seed, n_epochs=args.n_epochs, esm2_dir=args.esm2_dir, architecture="flat",
        )
        assert (seed_dir / "final.pt").exists(), f"seed {seed}: final.pt missing"
        assert not (seed_dir / "best.pt").exists(), f"seed {seed}: best.pt unexpectedly written with no val set"

    log.info(f"Done. {args.n_seeds} seeds trained -> {args.out_dir}")


if __name__ == "__main__":
    main()
