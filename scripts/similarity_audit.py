"""
similarity_audit.py

For every test example across every component_challenge_split.py panel and
LONO config, records:
  - nearest train id by sequence identity (grouping metric: >=80% bidirectional coverage)
  - max sequence identity at >=80% coverage, and which train id achieves it
  - best local sequence identity regardless of coverage (informational --
    a short high-identity local match that wouldn't have qualified as a
    grouping edge, but is still worth seeing)
  - max full-chain min(qTM, tTM), and which train id achieves it
  - max pocket-fragment min(qTM, tTM) -- DIAGNOSTIC ONLY, this was never
    the grouping criterion (see build_split_graph.py's docstring on why)
  - whether the nearest (grouping-metric) train neighbor shares this
    example's label, and (for positives) subclass, and (for negatives)
    neg_family

Also re-verifies the split's core guarantee directly from the raw pair
files, independent of trusting the component construction: asserts no
pair (test_id, train_id) within any panel/LONO config has sequence
identity >= build_split_graph.py's grouping threshold (0.3) at >=80%
bidirectional coverage. A failure here would mean the split is leaking
despite the component-level construction -- e.g. a bug in how
sequence_components was built, not just in how panels were assembled from it.

CLI:
    python similarity_audit.py --split-graph data/split_graph.json \
        --challenge-splits data/challenge_splits.json --pockets-dir data/pockets \
        --catalog full_structure_catalog.csv --out data/similarity_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

GROUPING_IDENTITY_THRESHOLD = 0.3
GROUPING_COVERAGE_THRESHOLD = 0.8


def _strip_pdb_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".pdb") else name


def load_seq_pairs(tsv_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Symmetric: d[a][b] == d[b][a] == (pident, min_cov). Either direction's
    report is treated as sufficient, matching build_split_graph.py's edges."""
    d: dict[str, dict[str, tuple[float, float]]] = {}
    with open(tsv_path) as f:
        for line in f:
            q, t, pident, _alnlen, qcov, tcov, _evalue = line.strip().split("\t")
            if q == t:
                continue
            val = (float(pident) / 100.0, min(float(qcov), float(tcov)))
            d.setdefault(q, {})[t] = val
            d.setdefault(t, {})[q] = val
    return d


def load_tm_pairs(tsv_path: Path, has_rmsd_col: bool) -> dict[str, dict[str, float]]:
    d: dict[str, dict[str, float]] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if has_rmsd_col:
                q, t, _pident, _alnlen, _rmsd, _qcov, _tcov, qtm, ttm, _evalue = parts
            else:
                q, t, _pident, _alnlen, _qcov, _tcov, qtm, ttm, _evalue = parts
            q, t = _strip_pdb_suffix(q), _strip_pdb_suffix(t)
            if q == t:
                continue
            val = min(float(qtm), float(ttm))
            d.setdefault(q, {})[t] = val
            d.setdefault(t, {})[q] = val
    return d


def load_metadata(pockets_dir: Path, catalog_path: Path) -> dict[str, dict]:
    meta = {}
    for f in pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        meta[pocket.metadata.source_structure_id] = {
            "label": pocket.metadata.label,
            "subclass": pocket.metadata.subclass,
        }
    catalog = {r["accession"]: r for r in csv.DictReader(open(catalog_path))}
    for sid, row in catalog.items():
        if sid in meta:
            meta[sid]["neg_family"] = row.get("neg_family", "")
    return meta


def audit_example(
    test_id: str, train_ids: list[str], seq_pairs: dict, tm_pairs: dict,
    pocket_pairs: dict, meta: dict[str, dict],
) -> dict:
    seq_neighbors = seq_pairs.get(test_id, {})
    best_covered_id, best_covered_pident = None, 0.0
    best_local_id, best_local_pident = None, 0.0
    for tid in train_ids:
        hit = seq_neighbors.get(tid)
        if hit is None:
            continue
        pident, cov = hit
        if pident > best_local_pident:
            best_local_pident, best_local_id = pident, tid
        if cov >= GROUPING_COVERAGE_THRESHOLD and pident > best_covered_pident:
            best_covered_pident, best_covered_id = pident, tid

    tm_neighbors = tm_pairs.get(test_id, {})
    best_tm_id, best_tm = None, 0.0
    for tid in train_ids:
        val = tm_neighbors.get(tid)
        if val is not None and val > best_tm:
            best_tm, best_tm_id = val, tid

    pocket_neighbors = pocket_pairs.get(test_id, {})
    best_pocket_tm = max((pocket_neighbors.get(tid, 0.0) for tid in train_ids), default=0.0)

    nearest_id = best_covered_id or best_local_id
    same_label = same_subclass = same_family = None
    if nearest_id is not None:
        same_label = meta.get(nearest_id, {}).get("label") == meta.get(test_id, {}).get("label")
        if meta.get(test_id, {}).get("label") == "positive":
            same_subclass = meta.get(nearest_id, {}).get("subclass") == meta.get(test_id, {}).get("subclass")
        else:
            same_family = meta.get(nearest_id, {}).get("neg_family") == meta.get(test_id, {}).get("neg_family")

    return {
        "nearest_train_id": nearest_id,
        "max_identity_at_80cov": round(best_covered_pident, 4),
        "max_identity_at_80cov_train_id": best_covered_id,
        "best_local_identity_any_coverage": round(best_local_pident, 4),
        "best_local_identity_train_id": best_local_id,
        "max_fullchain_tm": round(best_tm, 4),
        "max_fullchain_tm_train_id": best_tm_id,
        "max_pocket_tm_diagnostic_only": round(best_pocket_tm, 4),
        "nearest_neighbor_same_label": same_label,
        "nearest_neighbor_same_subclass": same_subclass,
        "nearest_neighbor_same_neg_family": same_family,
    }


def assert_no_leaking_edge(
    train_ids: list[str], test_ids: list[str], seq_pairs: dict, config_name: str,
) -> None:
    train_set = set(train_ids)
    violations = []
    for tid in test_ids:
        for other, (pident, cov) in seq_pairs.get(tid, {}).items():
            if other in train_set and pident >= GROUPING_IDENTITY_THRESHOLD and cov >= GROUPING_COVERAGE_THRESHOLD:
                violations.append((tid, other, round(pident, 3), round(cov, 3)))
    assert not violations, (
        f"{config_name}: {len(violations)} qualifying sequence edge(s) cross train/test "
        f"(id>=30%, cov>=80%) -- split is leaking. Examples: {violations[:5]}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    split_graph = json.loads(args.split_graph.read_text())
    challenge = json.loads(args.challenge_splits.read_text())
    meta = load_metadata(args.pockets_dir, args.catalog)

    log.info("Loading raw pairwise files...")
    seq_pairs = load_seq_pairs(Path(split_graph["raw_pair_files"]["sequence_pairs_tsv"]))
    tm_pairs = load_tm_pairs(Path(split_graph["raw_pair_files"]["domain_pairs_tsv"]), has_rmsd_col=False)
    pocket_pairs = load_tm_pairs(Path(split_graph["raw_pair_files"]["pocket_pairs_tsv (diagnostic only)"]), has_rmsd_col=True)

    configs: dict[str, dict] = {}
    for name, panel in challenge["panels"].items():
        configs[f"panel:{name}"] = panel
    for name, cfg in challenge["leave_one_negative_family_out"].items():
        configs[f"lono:{name}"] = cfg
    configs["operational_reference_anchored_retrieval"] = challenge["operational_reference_anchored_retrieval"]

    report = {}
    for config_name, cfg in configs.items():
        log.info(f"Auditing {config_name} ({len(cfg['test_ids'])} test examples)...")
        assert_no_leaking_edge(cfg["train_ids"], cfg["test_ids"], seq_pairs, config_name)
        examples = {
            tid: audit_example(tid, cfg["train_ids"], seq_pairs, tm_pairs, pocket_pairs, meta)
            for tid in cfg["test_ids"]
        }
        max_identities = [e["max_identity_at_80cov"] for e in examples.values()]
        report[config_name] = {
            "n_test": len(cfg["test_ids"]),
            "no_leaking_edge_assertion": "PASSED",
            "max_identity_at_80cov_distribution": {
                "max": round(max(max_identities), 4) if max_identities else None,
                "mean": round(sum(max_identities) / len(max_identities), 4) if max_identities else None,
                "n_below_grouping_threshold": sum(1 for v in max_identities if v < GROUPING_IDENTITY_THRESHOLD),
            },
            "examples": examples,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote similarity audit -> {args.out}")
    for name, r in report.items():
        log.info(f"  {name}: n_test={r['n_test']} max_id@80cov_dist={r['max_identity_at_80cov_distribution']}")


if __name__ == "__main__":
    main()
