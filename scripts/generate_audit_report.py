"""
generate_audit_report.py

Writes a compact, TRACKED audit report (input hashes, tool versions,
thresholds, and search-completeness counts) alongside the full split graph
data/split_graph.json produces -- which lives under the gitignored data/
directory and so isn't itself part of the repo. This is the piece meant to
answer "was this actually the exhaustive, verified run it claims to be" a
year from now, without needing to regenerate multi-hundred-thousand-row
pair files.

Completeness check: for each search, reports the per-query hit-count
distribution and how many queries sit exactly at --max-seqs (a query at
the cap is evidence the search may still be truncated for that query,
even with --exhaustive-search 1, if the database is large enough -- worth
watching even though this dataset is small enough that it shouldn't happen).

CLI:
    python generate_audit_report.py --split-graph data/split_graph.json \
        --manifest configs/manifest.csv --sequences-fasta data/pocket_sequences.fasta \
        --domain-pdb-dir data/domain_pdbs --pocket-pdb-dir data/pocket_pdbs \
        --out reports/split_graph_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from utils import get_logger

log = get_logger(__name__)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(dir_path: Path, pattern: str = "*") -> dict:
    """Combined hash over every matching file's (name, content-hash), so a
    changed/added/removed/reordered file changes the result. Also returns
    per-file count and total size for a quick human sanity check."""
    files = sorted(dir_path.glob(pattern))
    combined = hashlib.sha256()
    total_size = 0
    for f in files:
        combined.update(f.name.encode())
        combined.update(file_hash(f).encode())
        total_size += f.stat().st_size
    return {"n_files": len(files), "total_size_bytes": total_size, "combined_sha256": combined.hexdigest()}


def query_hit_completeness(tsv_path: Path, max_seqs: int) -> dict:
    counts = Counter()
    with open(tsv_path) as f:
        for line in f:
            counts[line.split("\t", 1)[0]] += 1
    values = list(counts.values())
    n_at_cap = sum(1 for v in values if v >= max_seqs)
    return {
        "n_queries_with_any_hit": len(values),
        "min_hits": min(values) if values else 0,
        "max_hits": max(values) if values else 0,
        "mean_hits": round(sum(values) / len(values), 1) if values else 0,
        "n_queries_at_or_above_max_seqs": n_at_cap,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--sequences-fasta", required=True, type=Path)
    p.add_argument("--domain-pdb-dir", required=True, type=Path)
    p.add_argument("--pocket-pdb-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = json.loads(args.split_graph.read_text())
    max_seqs = d["thresholds"]["max_seqs"]

    report = {
        "generated_from": str(args.split_graph),
        "thresholds": d["thresholds"],
        "tool_versions": d["tool_versions"],
        "input_hashes": {
            "manifest_csv": file_hash(args.manifest),
            "sequences_fasta": file_hash(args.sequences_fasta),
            "domain_pdb_dir": directory_hash(args.domain_pdb_dir, "*.pdb"),
            "pocket_pdb_dir": directory_hash(args.pocket_pdb_dir, "*.pdb"),
        },
        "search_completeness": {
            "sequence_search": query_hit_completeness(Path(d["raw_pair_files"]["sequence_pairs_tsv"]), max_seqs),
            "domain_full_chain_search": query_hit_completeness(Path(d["raw_pair_files"]["domain_pairs_tsv"]), max_seqs),
            "pocket_search_diagnostic_only": query_hit_completeness(
                Path(d["raw_pair_files"]["pocket_pairs_tsv (diagnostic only)"]), max_seqs),
        },
        "component_summary": {
            regime: {
                "n_components": len(d[regime]),
                "largest_component": max(len(m) for m in d[regime].values()),
            }
            for regime in ["sequence_components", "structure_components_foldremote", "structure_components_redundancy"]
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote audit report -> {args.out}")
    for name, stats in report["search_completeness"].items():
        flag = " <-- QUERIES AT CAP, search may be incomplete" if stats["n_queries_at_or_above_max_seqs"] > 0 else ""
        log.info(f"  {name}: {stats}{flag}")


if __name__ == "__main__":
    main()
