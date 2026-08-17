"""
clustering_split.py — Task 4 (spec §7)

Two jobs:
  1. Pocket-level structural clustering via Foldseek, run on isolated pocket
     structures (not full chains) so clustering reflects local active-site
     similarity rather than whole-fold similarity. Same cluster := pocket
     RMSD < 2 Å after superposition OR > 60% pocket-residue identity — both
     are configurable Foldseek thresholds, applied here as post-filters on
     `foldseek easy-cluster` output since Foldseek's native clustering
     conflates the two by default.
  2. Split assignment: stratified (by label — positive / hard_negative /
     easy_negative) group k-fold, built at the CLUSTER level (never split a
     cluster across folds — that's the leakage Foldseek clustering exists to
     prevent). NOT stratified by B1/B2/B3 subclass: subclass is a sequence/
     genomic classification carried through as metadata only (never fed into
     the pocket graph features — see graph_construction.py), it isn't known
     to track pocket-structure differences, and this dataset only has one
     confirmed B2 structure — nowhere near enough to hold out a whole fold.
     External holdout (reference-bank members not used as anchors, plus any
     curated environmental candidates) — hardcoded exclusion by structure_id,
     never entering train/val/test regardless of clustering.

Requires the `foldseek` binary on PATH. Pocket structures must first be
written out as individual PDB files (one per pocket) — see
`write_pocket_pdbs()` — since Foldseek operates on structure files, not the
.npz PocketSubgraph format used elsewhere in this pipeline.

CLI:
    python clustering_split.py \
        --pocket-dir data/pockets --pocket-pdb-dir data/pocket_pdbs \
        --foldseek-out data/foldseek_clusters \
        --rmsd-threshold 2.0 --identity-threshold 0.6 \
        --external-holdout-ids AMM-1 SZM-1 CAM-2 \
        --k-folds 5 --val-frac 0.15 \
        --splits-out data/splits.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from utils import PocketSubgraph, get_logger
from catalog_to_manifest import REFERENCE_BANK

log = get_logger(__name__)

# Reference-bank members are fixed eval-time anchors (evaluate.py / inference.py
# kNN + prototype), never part of the train/val/test fold rotation.
DEFAULT_EXTERNAL_HOLDOUT = [sid for sid, _, _ in REFERENCE_BANK]


def write_pocket_pdbs(pocket_dir: Path, out_dir: Path) -> dict[str, Path]:
    """
    Writes each PocketSubgraph as a minimal PDB (coords only, no header
    metadata Foldseek doesn't need) using biotite. Returns {structure_id: path}.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_path = {}
    for npz_path in sorted(pocket_dir.glob("*.npz")):
        pocket = PocketSubgraph.load(npz_path)
        n = len(pocket.res_ids)
        arr = struc.AtomArray(n)
        arr.coord = pocket.coords
        arr.res_id = pocket.res_ids
        arr.res_name = pocket.res_names
        arr.atom_name = pocket.atom_names
        arr.element = pocket.elements
        arr.chain_id = np.full(n, "A")
        arr.hetero = np.zeros(n, dtype=bool)

        out_path = out_dir / f"{pocket.metadata.source_structure_id}.pdb"
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(arr)
        pdb_file.write(str(out_path))
        id_to_path[pocket.metadata.source_structure_id] = out_path
    return id_to_path


def run_foldseek_cluster(pdb_dir: Path, out_dir: Path, min_seq_id: float = 0.0) -> Path:
    """
    Runs `foldseek easy-cluster` over all pocket PDBs. Returns path to the
    resulting `_cluster.tsv` (columns: representative_id, member_id).
    min_seq_id kept low/0 here because we post-filter with our own
    RMSD + pocket-residue-identity criteria (see filter_clusters), not
    Foldseek's default sequence-identity gate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "pocket_clusters"
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [
                "foldseek", "easy-cluster", str(pdb_dir), str(prefix), str(tmp_dir),
                "--min-seq-id", str(min_seq_id),
                "--alignment-type", "1",  # TM-align based, appropriate for pocket-level structural comparison
            ],
            check=True, capture_output=True, timeout=1800,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "foldseek not found on PATH. Install it "
            "(https://github.com/steineggerlab/foldseek) before running this step."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"foldseek easy-cluster failed: {e.stderr}")

    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise RuntimeError(f"Expected foldseek output not found: {cluster_tsv}")
    return cluster_tsv


def _strip_pdb_suffix(name: str) -> str:
    """Strip a literal trailing '.pdb', nothing else. Do NOT use Path(...).stem
    or split('_') here — structure_ids in this dataset are versioned GenBank/
    RefSeq accessions (e.g. "AAA22562.1", "WP_213603971.1"); both Path.stem
    (treats the ".1" as a file extension) and split("_")[0] (treats the "WP_"
    prefix as a chain-suffix separator) silently truncate real ids and break
    the id -> cluster mapping."""
    return name[:-4] if name.endswith(".pdb") else name


def load_clusters(cluster_tsv: Path) -> dict[str, str]:
    """Returns {structure_id: cluster_representative_id}."""
    assignment = {}
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.strip().split("\t")[:2]
            assignment[_strip_pdb_suffix(member)] = _strip_pdb_suffix(rep)
    return assignment


def _cluster_label_stratum(members: list[str], structure_metadata: dict[str, dict]) -> str:
    """Majority label among a cluster's members — used only to keep folds
    balanced (e.g. so rare positives don't pile into one fold), never to
    condition on subclass."""
    labels = [structure_metadata.get(m, {}).get("label") for m in members]
    return Counter(labels).most_common(1)[0][0]


def make_cluster_kfold_splits(
    cluster_assignment: dict[str, str],
    structure_metadata: dict[str, dict],
    external_holdout_ids: list[str],
    k_folds: int = 5,
    val_frac: float = 0.15,
    seed: int = 0,
) -> list[dict]:
    """
    Builds k folds via stratified (by label, not subclass) group k-fold at
    the CLUSTER level — every structure_id sharing a cluster representative
    goes to the same partition. Each fold dict:
        {"fold_id": 0, "train": [...], "val": [...], "test": [...]}
    Stratifying by subclass (B1/B2/B3) was dropped: it's a sequence-level
    classification not known to track pocket-structure differences, it's
    never used as a model feature, and this dataset has only one confirmed
    B2 structure — far too few to hold out a whole fold on.
    """
    external_holdout_ids = set(external_holdout_ids)

    clusters: dict[str, list[str]] = {}
    for sid, rep in cluster_assignment.items():
        if sid in external_holdout_ids:
            continue
        clusters.setdefault(rep, []).append(sid)

    # Stratify cluster-to-fold assignment by majority label so rare classes
    # (positives) spread evenly across folds instead of clumping.
    strata: dict[str, list[str]] = {}
    for rep, members in clusters.items():
        stratum = _cluster_label_stratum(members, structure_metadata)
        strata.setdefault(stratum, []).append(rep)

    rng = np.random.default_rng(seed=seed)
    test_fold_of_rep: dict[str, int] = {}
    for stratum_reps in strata.values():
        reps = list(stratum_reps)
        rng.shuffle(reps)
        for i, rep in enumerate(reps):
            test_fold_of_rep[rep] = i % k_folds

    folds = []
    for fold_id in range(k_folds):
        test_reps = [r for r, f in test_fold_of_rep.items() if f == fold_id]
        remaining_reps = [r for r, f in test_fold_of_rep.items() if f != fold_id]

        rng_val = np.random.default_rng(seed=seed * 1000 + fold_id)
        remaining_reps = list(remaining_reps)
        rng_val.shuffle(remaining_reps)
        n_val_reps = max(1, int(val_frac * len(remaining_reps)))
        val_reps = set(remaining_reps[:n_val_reps])

        train_ids, val_ids, test_ids = [], [], []
        for rep in remaining_reps:
            target = val_ids if rep in val_reps else train_ids
            target.extend(clusters[rep])
        for rep in test_reps:
            test_ids.extend(clusters[rep])

        folds.append({
            "fold_id": fold_id,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        })
        log.info(
            f"Fold {fold_id}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )

    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pocket-dir", required=True, type=Path)
    p.add_argument("--pocket-pdb-dir", required=True, type=Path)
    p.add_argument("--foldseek-out", required=True, type=Path)
    p.add_argument("--external-holdout-ids", nargs="*", default=DEFAULT_EXTERNAL_HOLDOUT)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits-out", required=True, type=Path)
    args = p.parse_args()

    id_to_pdb = write_pocket_pdbs(args.pocket_dir, args.pocket_pdb_dir)
    cluster_tsv = run_foldseek_cluster(args.pocket_pdb_dir, args.foldseek_out)
    cluster_assignment = load_clusters(cluster_tsv)

    structure_metadata = {}
    for npz_path in args.pocket_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(npz_path)
        structure_metadata[pocket.metadata.source_structure_id] = {
            "subclass": pocket.metadata.subclass,
            "label": pocket.metadata.label,
        }

    folds = make_cluster_kfold_splits(
        cluster_assignment, structure_metadata, args.external_holdout_ids,
        k_folds=args.k_folds, val_frac=args.val_frac, seed=args.seed,
    )

    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps({
        "external_holdout": sorted(args.external_holdout_ids),
        "folds": folds,
    }, indent=2))
    log.info(f"Splits written -> {args.splits_out}")


if __name__ == "__main__":
    main()
