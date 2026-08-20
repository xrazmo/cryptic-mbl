"""
production_v2_coverage_report.py

Explicit missingness accounting for the v2 scoring regime, per the
audit requirement: the DCH channel's "unavailable" rate (no metal
predicted) must be exposed, not buried. Reports overall and by
challenge panel -- missingness is not uniform (remote_outlier is
worst, 27.8%, vs B1_B2_transfer's 8.0%), which matters for reading any
aggregate triage-distribution statistic downstream.

CLI:
    python production_v2_coverage_report.py --dch-scores data/dch_scores.json \
        --challenge-splits data/challenge_splits.json \
        --out reports/production_v2_coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import get_logger

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dch-scores", required=True, type=Path)
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    dch = json.loads(args.dch_scores.read_text())
    challenge = json.loads(args.challenge_splits.read_text())

    report = {"overall": dch["status_counts"], "by_panel": {}}
    scores = dch["scores"]
    for panel_name, panel in challenge["panels"].items():
        test_ids = panel["test_ids"]
        n = len(test_ids)
        unavailable = sum(1 for t in test_ids if scores.get(t, {}).get("status") == "unavailable")
        report["by_panel"][panel_name] = {
            "n_test": n, "dch_unavailable": unavailable,
            "dch_unavailable_fraction": round(unavailable / n, 4) if n else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))
    log.info(f"Wrote coverage report -> {args.out}")


if __name__ == "__main__":
    main()
