"""
export_pocket_sequences.py

Exports one FASTA record per structure_id: the full-chain sequence of
whichever chain esm2_embed.py actually resolved the most pocket residues
to (its "dominant chain") -- i.e. the exact sequence ESM2 embeddings were
drawn from. Downstream sequence clustering (see build_split_graph.py) must
group structures by this exact homology relationship, not some other
notion of "the sequence" for that structure, or the resulting split
wouldn't actually be independent with respect to what the model sees.

CLI:
    python export_pocket_sequences.py --manifest configs/manifest.csv \
        --raw-dir data/raw --pockets-dir data/pockets \
        --out data/pocket_sequences.fasta
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import get_logger, PocketSubgraph
from graph_construction import collapse_to_residue_level
from esm2_embed import chain_sequences, resolve_pocket_residues
import data_assembly

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rows = data_assembly.read_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0
    with open(args.out, "w") as f:
        for row in rows:
            pocket_path = args.pockets_dir / f"{row.structure_id}.npz"
            if not pocket_path.exists():
                continue
            try:
                structure_path = data_assembly.resolve_structure(row.source_uri, row.structure_id, args.raw_dir)
                chains = chain_sequences(structure_path)
                pocket = PocketSubgraph.load(pocket_path)
                residue_level = collapse_to_residue_level(pocket)
                resolution = resolve_pocket_residues(residue_level["res_ids"], residue_level["centroids"], chains)
                if not resolution:
                    raise ValueError("no chain resolved for any pocket residue")
                dominant_chain = max(resolution, key=lambda c: len(resolution[c]))
                _res_ids, _ca, sequence = chains[dominant_chain]
                f.write(f">{row.structure_id}\n{sequence}\n")
                n_ok += 1
            except Exception as e:
                log.error(f"FAILED: {row.structure_id} — {e}")
                n_fail += 1
    log.info(f"Exported {n_ok} sequences ({n_fail} failed) -> {args.out}")


if __name__ == "__main__":
    main()
