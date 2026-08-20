"""
freeze_production_v2_manifest.py

Freezes the v2 asymmetric production scoring regime -- a NEW,
separately versioned manifest (reports/production_model_v2_manifest.json).
Does NOT overwrite the frozen V1 manifest (reports/production_model_manifest.json)
or V1 artifacts (models/production/, data/production/reference_bank/) --
V1 remains available only as legacy_gnn_v1, disabled by default, for
reference.

CLI:
    python freeze_production_v2_manifest.py \
        --dch-scores data/dch_scores.json \
        --reference-bank data/production/reference_bank_v2/mean_esm2_v2.npz \
        --coverage-report reports/production_v2_coverage.json \
        --catalog full_structure_catalog.csv \
        --out reports/production_model_v2_manifest.json
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
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dch-scores", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--coverage-report", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    dch = json.loads(args.dch_scores.read_text())
    coverage = json.loads(args.coverage_report.read_text())

    if git_is_dirty():
        log.warning("Working tree has uncommitted changes -- commit before freezing.")

    manifest = {
        "scoring_regime": "v2_asymmetric",
        "supersedes": None,
        "coexists_with": "v1 (models/production/, reports/production_model_manifest.json), "
                          "accessible only as legacy_gnn_v1 via score_candidate_v2.py --include-legacy-gnn-v1, "
                          "disabled by default, excluded from the triage decision -- trained on pre-fix corrupted pockets.",
        "git_commit": git_commit_hash(),
        "git_dirty_at_freeze_time": git_is_dirty(),
        "input_manifest": {
            "catalog_path": str(args.catalog),
            "catalog_sha256": file_hash(args.catalog),
        },
        "pockets_source": "data/pockets_v2 (post metal-site-corruption fix -- see git log "
                           "'Fix upstream metal-site corruption and cross-chain pocket contamination')",
        "channels": {
            "dch_structural_support": {
                "rule": "Cys SG atom within 2.8A of either retained metal site (scripts/dch_score.py)",
                "states": ["supported", "not_supported", "unavailable"],
                "unavailable_meaning": "no metal site predicted for this candidate -- a missing-input state, "
                                        "never coerced to not_supported or negative",
                "validated_on": "B1_B2_transfer held-out panel: 0.948 sensitivity / 0.983 specificity "
                                 "(see reports/coordination_fingerprint_findings.md); qualifications there "
                                 "(selected on the complete labeled corpus, misses both nominal B1 remote "
                                 "outliers, specificity 0.880 on the phosphodiesterase-heavy remote panel) "
                                 "still apply -- prospective Atlas validation required.",
                "full_pool_status_counts": dch["status_counts"],
                "coverage_by_panel": coverage["by_panel"],
            },
            "esm2_retrieval": {
                "reference_bank_path": str(args.reference_bank),
                "reference_bank_sha256": file_hash(args.reference_bank),
                "method": "k=5 nearest neighbor (Euclidean, mean-pooled ESM2 embedding) against the frozen "
                           "reference bank; majority vote for status, full neighbor list (id/distance/label/"
                           "subclass/neg_family) returned; OOD percentile from leave-one-out nearest-neighbor "
                           "distance calibration over the reference bank itself.",
                "b3_role": "sole B3 channel -- the outer-pocket structural encoder was tested and rejected "
                           "(see reports/outer_pocket_encoder_rejection.md); never run in production.",
            },
        },
        "final_triage": {
            "states": ["dual_support", "dch_only", "esm2_only", "unresolved"],
            "logic": "scripts/score_candidate_v2.py::combine_triage -- dual_support requires DCH=supported "
                     "AND ESM2=positive; dch_only requires DCH=supported with ESM2 not positive; esm2_only "
                     "requires ESM2=positive with DCH not supported (covers B3, noncanonical B1/B2, and DCH "
                     "extraction failures -- NOT automatically labeled B3); unresolved is the default whenever "
                     "neither channel supports the candidate, or a required channel's input is unavailable -- "
                     "never coerced to a negative call.",
            "no_opaque_fusion": "Both channel scores are always returned alongside the triage label; the "
                                 "triage label is a lookup table over the two channel states, not a learned "
                                 "or weighted combination.",
        },
        "rejected_components": {
            "outer_pocket_encoder": "see reports/outer_pocket_encoder_rejection.md -- 0/8 unique ESM2-miss "
                                     "recovery, worse FPR than ESM2 on 5/9 hard-negative families. Never run "
                                     "during production scoring.",
        },
        "excluded_from_triage": {
            "legacy_gnn_v1": "trained on pre-fix corrupted V1 pockets (see git log); available only via an "
                              "explicit opt-in flag for reference, never influences dual_support/dch_only/"
                              "esm2_only/unresolved.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote v2 production manifest -> {args.out}")
    log.info(f"git_commit={manifest['git_commit']} git_dirty={manifest['git_dirty_at_freeze_time']}")


if __name__ == "__main__":
    main()
