"""
analyze_identity_stratified_sensitivity.py

The question that actually matters for this project's premise (finding
MBLs a sequence-based method would miss): does detection specifically
hold up as sequence identity to anything in the training pool drops
toward zero? Aggregate sensitivity on a held-out panel doesn't answer
this -- it could be driven entirely by the easier, higher-identity
subset of that panel. Every test positive here already sits below the
30%-identity/80%-coverage grouping threshold by construction (the split
is leakage-free), so this stratifies *within* that already-remote
regime using the per-example identities similarity_audit.json already
computed, joined against:
  - the training-free raw-ESM2 5-NN baseline's per-example predictions
    (data/mean_esm2_baseline.json)
  - the per-seed-voted GNN predictions for new_graph_flat, branched_fused,
    and branched_structure_only (data/per_seed_evaluation.json, after
    evaluate_per_seed.py was extended to include vote_preds)

If structure-only detection stays roughly flat while the ESM2 baseline's
hit rate collapses in the near-zero-identity bin, that is the actual
evidence this endeavor needs. If everything collapses together, that is
an equally important negative result -- it means the current structural
representation is not yet ready to replace sequence-based screening for
truly cryptic (sequence-divergent) candidates.

CLI:
    python analyze_identity_stratified_sensitivity.py \
        --similarity-audit data/similarity_audit.json \
        --mean-esm2-baseline data/mean_esm2_baseline.json \
        --per-seed-evaluation data/per_seed_evaluation.json \
        --challenge-splits data/challenge_splits.json \
        --pockets-dir data/pockets \
        --out data/identity_stratified_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

BINS = [(0.0, 0.0), (0.0001, 0.10), (0.10, 0.20), (0.20, 0.30)]
BIN_LABELS = ["0% (no detectable hit)", "0-10%", "10-20%", "20-30%"]


def bin_of(identity: float) -> int:
    if identity == 0.0:
        return 0
    for i, (lo, hi) in enumerate(BINS[1:], start=1):
        if lo <= identity <= hi:
            return i
    return len(BINS) - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--similarity-audit", required=True, type=Path)
    p.add_argument("--mean-esm2-baseline", required=True, type=Path)
    p.add_argument("--per-seed-evaluation", required=True, type=Path)
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    audit = json.loads(args.similarity_audit.read_text())
    baseline = json.loads(args.mean_esm2_baseline.read_text())
    per_seed = json.loads(args.per_seed_evaluation.read_text())
    challenge = json.loads(args.challenge_splits.read_text())

    labels = {}
    for f in args.pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        labels[pocket.metadata.source_structure_id] = "positive" if pocket.metadata.label == "positive" else "negative"

    report = {}
    for panel_name in ["B1_B2_transfer", "B3_transfer"]:
        test_ids = challenge["panels"][panel_name]["test_ids"]
        positive_ids = [t for t in test_ids if labels.get(t) == "positive"]
        identities = audit[f"panel:{panel_name}"]["examples"]
        esm2_preds = baseline[f"panel:{panel_name}"]["per_example"]

        configs = ["new_graph_flat", "branched_fused", "branched_structure_only"]
        config_votes = {c: per_seed[panel_name][c]["vote_preds"] for c in configs}

        bins = [{"label": lbl, "n": 0, "hits": {"esm2_baseline": 0, **{c: 0 for c in configs}}} for lbl in BIN_LABELS]
        per_example = []
        for tid in positive_ids:
            identity = identities[tid]["max_identity_at_80cov"]
            b = bin_of(identity)
            bins[b]["n"] += 1
            esm2_hit = esm2_preds.get(tid, {}).get("pred") == "positive"
            bins[b]["hits"]["esm2_baseline"] += int(esm2_hit)
            row = {"id": tid, "max_identity_at_80cov": identity, "esm2_baseline_hit": esm2_hit}
            for c in configs:
                hit = config_votes[c].get(tid) == "positive"
                bins[b]["hits"][c] += int(hit)
                row[f"{c}_hit"] = hit
            per_example.append(row)

        for b in bins:
            b["sensitivity"] = {k: (v / b["n"] if b["n"] else None) for k, v in b["hits"].items()}

        report[panel_name] = {"n_test_positives": len(positive_ids), "bins": bins, "per_example": per_example}
        log.info(f"=== {panel_name} ({len(positive_ids)} test positives) ===")
        for b in bins:
            log.info(f"  {b['label']:26s} n={b['n']:3d}  sens: " + " ".join(
                f"{k}={v:.2f}" if v is not None else f"{k}=n/a" for k, v in b["sensitivity"].items()
            ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote identity-stratified sensitivity -> {args.out}")


if __name__ == "__main__":
    main()
