"""
export_pocket_sequences.py

Exports one FASTA record per structure_id: the full-chain sequence of
whichever chain esm2_embed.py actually resolved the most pocket residues
to (its "dominant chain") -- i.e. the exact sequence ESM2 embeddings were
drawn from, truncated identically (esm2_embed.resolve_dominant_chain).
Downstream sequence clustering (see build_split_graph.py) must group
structures by this exact homology relationship, not some other notion of
"the sequence" for that structure, or the resulting split wouldn't
actually be independent with respect to what the model sees.

Fails loudly (non-zero exit) rather than silently producing a partial
FASTA: a missing/duplicate/empty record here means the sequence-clustering
graph is quietly missing a structure, which can only make the split's
"no cross-partition homology" guarantee weaker, not stronger.

CLI:
    python export_pocket_sequences.py --manifest configs/manifest.csv \
        --raw-dir data/raw --pockets-dir data/pockets \
        --out data/pocket_sequences.fasta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import get_logger, PocketSubgraph
from esm2_embed import resolve_dominant_chain, MAX_LENGTH_DEFAULT
import data_assembly

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT,
                    help="Must match esm2_embed.py's --max-length for the export to reflect "
                         "exactly what ESM2 embedded.")
    args = p.parse_args()

    rows = data_assembly.read_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, str] = {}
    failures: list[str] = []
    for row in rows:
        pocket_path = args.pockets_dir / f"{row.structure_id}.npz"
        if not pocket_path.exists():
            continue
        try:
            structure_path = data_assembly.resolve_structure(row.source_uri, row.structure_id, args.raw_dir)
            pocket = PocketSubgraph.load(pocket_path)
            _chain_id, _res_ids, sequence = resolve_dominant_chain(structure_path, pocket, args.max_length)
            if not sequence:
                raise ValueError("empty sequence after chain resolution")
            if row.structure_id in records:
                raise ValueError(f"duplicate structure_id (already exported)")
            records[row.structure_id] = sequence
        except Exception as e:
            log.error(f"FAILED: {row.structure_id} — {e}")
            failures.append(row.structure_id)

    if failures:
        log.error(f"{len(failures)} structure(s) failed sequence export: {failures}")
        log.error("Refusing to write a partial FASTA -- fix the failures or explicitly "
                   "exclude these structures from the manifest before re-running.")
        sys.exit(1)

    with open(args.out, "w") as f:
        for sid, sequence in records.items():
            f.write(f">{sid}\n{sequence}\n")
    log.info(f"Exported {len(records)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
