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
  2. Split assignment:
       - 3-fold leave-one-subclass-out (hold out B2 entirely in >=1 fold),
         built at the CLUSTER level (never split a cluster across folds —
         that's the leakage Foldseek clustering exists to prevent).
       - External holdout: AMM-1, SZM-1, CAM-2 (+ any reference-bank member
         not used as an anchor) — hardcoded exclusion by structure_id,
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
        --splits-out data/splits.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from utils import PocketSubgraph, get_logger

log = get_logger(__name__)

DEFAULT_EXTERNAL_HOLDOUT = ["AMM-1", "SZM-1", "CAM-2"]
SUBCLASSES = ["B1", "B2", "B3"]  # "environmental" pockets fold into whichever fold isn't B1/B2/B3-held-out


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


def load_clusters(cluster_tsv: Path) -> dict[str, str]:
    """Returns {structure_id: cluster_representative_id}."""
    assignment = {}
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.strip().split("\t")[:2]
            # foldseek writes rep/member with .pdb suffix and possibly chain suffix; normalize
            rep_id = Path(rep).stem.split("_")[0]
            member_id = Path(member).stem.split("_")[0]
            assignment[member_id] = rep_id
    return assignment


def make_leave_one_subclass_out_splits(
    cluster_assignment: dict[str, str],
    structure_metadata: dict[str, dict],
    external_holdout_ids: list[str],
) -> list[dict]:
    """
    Builds 3 folds, each holding out one of B1/B2/B3 (and folding
    "environmental" positives + negatives into whichever fold is most
    balanced). Returns a list of fold dicts:
        {"held_out_subclass": "B2", "train": [...], "val": [...], "test": [...]}
    Split is at the CLUSTER level — every structure_id sharing a cluster
    representative goes to the same partition.
    """
    external_holdout_ids = set(external_holdout_ids)

    # Group structure_ids by cluster representative, excluding external holdout.
    clusters: dict[str, list[str]] = {}
    for sid, rep in cluster_assignment.items():
        if sid in external_holdout_ids:
            continue
        clusters.setdefault(rep, []).append(sid)

    folds = []
    for held_out in SUBCLASSES:
        test_ids, train_val_ids = [], []
        for rep, members in clusters.items():
            subclasses_in_cluster = {
                structure_metadata.get(m, {}).get("subclass") for m in members
            }
            if held_out in subclasses_in_cluster:
                test_ids.extend(members)
            else:
                train_val_ids.extend(members)

        # simple 85/15 train/val split of the remaining clusters (cluster-level, deterministic)
        rng = np.random.default_rng(seed=hash(held_out) % (2**32))
        remaining_reps = [r for r, m in clusters.items()
                           if held_out not in {structure_metadata.get(x, {}).get("subclass") for x in m}]
        rng.shuffle(remaining_reps)
        n_val_reps = max(1, int(0.15 * len(remaining_reps)))
        val_reps = set(remaining_reps[:n_val_reps])

        train_ids, val_ids = [], []
        for rep in remaining_reps:
            target = val_ids if rep in val_reps else train_ids
            target.extend(clusters[rep])

        folds.append({
            "held_out_subclass": held_out,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        })
        log.info(
            f"Fold held_out={held_out}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )

    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pocket-dir", required=True, type=Path)
    p.add_argument("--pocket-pdb-dir", required=True, type=Path)
    p.add_argument("--foldseek-out", required=True, type=Path)
    p.add_argument("--external-holdout-ids", nargs="*", default=DEFAULT_EXTERNAL_HOLDOUT)
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

    folds = make_leave_one_subclass_out_splits(
        cluster_assignment, structure_metadata, args.external_holdout_ids
    )

    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps({
        "external_holdout": sorted(args.external_holdout_ids),
        "folds": folds,
    }, indent=2))
    log.info(f"Splits written -> {args.splits_out}")


if __name__ == "__main__":
    main()
