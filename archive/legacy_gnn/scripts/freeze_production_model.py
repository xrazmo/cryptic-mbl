"""
freeze_production_model.py

Writes the frozen model card for the production new_graph_flat ensemble:
everything needed to know exactly what produced a given score and to
detect drift later -- git commit, input manifest hash, ESM2 checkpoint
id, feature configuration, epoch count/seeds, reference-bank embedding
file hashes, and the scoring rule. Run once after
train_production_model.py and build_production_reference_bank.py both
finish. Output is TRACKED (reports/, unlike gitignored data/models),
matching this project's existing convention for audit artifacts
(reports/split_graph_audit.json, reports/challenge_split_audit.json).

CLI:
    python freeze_production_model.py --models-dir models/production \
        --reference-bank data/production/reference_bank \
        --catalog full_structure_catalog.csv --out reports/production_model_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from utils import get_logger

log = get_logger(__name__)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_is_dirty() -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(status.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    seed_dirs = sorted(args.models_dir.glob("seed_*"))
    assert seed_dirs, f"no seed_* dirs found under {args.models_dir}"
    configs = [json.loads((d / "config.json").read_text()) for d in seed_dirs]
    seeds = [c["seed"] for c in configs]
    assert seeds == sorted(seeds) and len(set(seeds)) == len(seed_dirs), "seed dirs/config seeds mismatch"

    # All seeds must agree on everything except `seed` and `best_val_loss`
    # (the latter is expected to be inf/never-improved -- no validation
    # carve, see train_production_model.py).
    shared_keys = [k for k in configs[0] if k not in ("seed", "best_val_loss")]
    for c in configs[1:]:
        for k in shared_keys:
            assert c[k] == configs[0][k], f"seed configs disagree on {k}: {c[k]!r} vs {configs[0][k]!r}"
        assert (args.models_dir / f"seed_{c['seed']}" / "best.pt").exists() is False, \
            f"seed {c['seed']}: best.pt exists -- violates the no-checkpoint-selection production rule"

    ref_files = sorted(args.reference_bank.glob("*.npz"))
    assert ref_files, f"no reference-bank files found under {args.reference_bank}"

    if git_is_dirty():
        log.warning("Working tree has uncommitted changes -- commit before freezing, or this manifest's "
                     "git_commit will not fully describe the code that produced these weights.")

    manifest = {
        "model_name": "new_graph_flat_production",
        "decision": (
            "Flat architecture (PocketEncoder) over the Stage-2 graph (sequence-adjacency "
            "edges + radial-shell feature), no ESM2/structural branch separation. Selected "
            "over the branched fusion model: most balanced trained architecture, improves the "
            "B1/B2 floor without the branched model's remote-outlier regression, and the "
            "branched model's advantage was not consistent enough across seeds to justify "
            "production complexity (see evaluate_per_seed.py's per-seed voted results). The "
            "structural branch (BranchedPocketEncoder) is kept as supporting research evidence "
            "of real structure-only signal on B3, not a second production model."
        ),
        "git_commit": git_commit_hash(),
        "git_dirty_at_freeze_time": git_is_dirty(),
        "input_manifest": {
            "catalog_path": str(args.catalog),
            "catalog_sha256": file_hash(args.catalog),
        },
        "esm2_checkpoint": "esm2_t33_650M_UR50D",
        "feature_config": {
            "architecture": configs[0]["architecture"],
            "in_dim": configs[0]["in_dim"],
            "ablate_distance_to_metal": configs[0]["ablate_distance_to_metal"],
            "ablate_aa_identity": configs[0]["ablate_aa_identity"],
            "ablate_structural": configs[0]["ablate_structural"],
            "ablate_esm2": configs[0]["ablate_esm2"],
            "esm2_dir": configs[0]["esm2_dir"],
        },
        "training": {
            "n_epochs": configs[0]["n_epochs"],
            "margin": configs[0]["margin"],
            "seeds": seeds,
            "n_seeds": len(seeds),
            "validation_carve": None,
            "checkpoint_selection": "final.pt only, unconditionally saved (no best.pt -- no validation to select on)",
            "trained_on": "full labeled pool (146 positive / 931 negative, all 1077 structures), no held-out test set",
        },
        "reference_bank": {
            "path": str(args.reference_bank),
            "files": {f.name: file_hash(f) for f in ref_files},
        },
        "scoring_rule": (
            "Per candidate: for each of the n_seeds GNN models, embed the candidate in that "
            "seed's own space and take k=5 nearest neighbors (Euclidean) against that SAME "
            "seed's reference-bank embeddings -- one predicted label + one positive-neighbor "
            "fraction per seed. Aggregate across seeds by majority vote (label) and by "
            "averaging the positive-neighbor fraction (a scalar score, not a coordinate). "
            "Latent embedding coordinates are never averaged across seeds (see "
            "evaluate_per_seed.py's docstring for why that was invalid). The training-free "
            "mean-pooled-ESM2 k-NN score is reported as a separate auxiliary signal, not fused "
            "into the GNN ensemble result. See score_candidate.py."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote production model manifest -> {args.out}")
    log.info(f"git_commit={manifest['git_commit']} git_dirty={manifest['git_dirty_at_freeze_time']}")


if __name__ == "__main__":
    main()
