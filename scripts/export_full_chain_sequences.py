"""Export the complete dominant-chain protein sequence for external tools.

Unlike ``export_pocket_sequences.py`` this export is not truncated to ESM2's
1022-residue limit.  It is intended for protein-level comparators such as
fARGene.  Chain selection is nevertheless identical to the pocket pipeline:
the chain resolving the largest number of pocket residues is used.

The command fails rather than writing a partial FASTA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import data_assembly
from esm2_embed import resolve_dominant_chain
from utils import PocketSubgraph, get_logger

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--pockets-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    records: dict[str, str] = {}
    failures: list[str] = []
    for row in data_assembly.read_manifest(args.manifest):
        pocket_path = args.pockets_dir / f"{row.structure_id}.npz"
        if not pocket_path.exists():
            failures.append(f"{row.structure_id}: missing pocket")
            continue
        try:
            structure_path = data_assembly.resolve_structure(
                row.source_uri, row.structure_id, args.raw_dir
            )
            pocket = PocketSubgraph.load(pocket_path)
            _chain, _resids, sequence = resolve_dominant_chain(
                structure_path, pocket, max_length=sys.maxsize
            )
            if not sequence:
                raise ValueError("empty dominant-chain sequence")
            if row.structure_id in records:
                raise ValueError("duplicate structure_id")
            records[row.structure_id] = sequence
        except Exception as exc:
            failures.append(f"{row.structure_id}: {exc}")

    expected = len(data_assembly.read_manifest(args.manifest))
    if failures or len(records) != expected:
        raise RuntimeError(
            f"Refusing partial FASTA: {len(records)}/{expected} records; failures={failures}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f">{sid}\n{seq}\n" for sid, seq in records.items()))
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps({
            "schema_version": 1,
            "purpose": "untruncated dominant-chain sequences for external comparators",
            "n_records": len(records),
            "minimum_length": min(map(len, records.values())),
            "maximum_length": max(map(len, records.values())),
            "manifest_sha256": sha256(args.manifest),
            "fasta_sha256": sha256(args.out),
        }, indent=2) + "\n")
    log.info("Exported %d full-chain sequences -> %s", len(records), args.out)


if __name__ == "__main__":
    main()
