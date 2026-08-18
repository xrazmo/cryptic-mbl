"""
export_dominant_chain_structures.py

Writes one standalone, single-chain, non-hetero PDB per structure_id: the
COMPLETE dominant chain (the chain esm2_embed.py resolves the most pocket
residues to) -- untruncated. Used for whole-CHAIN structural comparison
(Foldseek TM-align, see build_split_graph.py) -- NOT the isolated
~40-residue pocket fragment, whose TM-align/RMSD comparisons turned out to
be dominated by small-fragment alignment noise rather than real fold
relationships (see git history: the pocket-fragment RMSD<2A OR pident>60%
rule chained 926/1077 structures -- including labeled-unrelated folds like
globin/TIM-barrel/Rossmann-SDR -- into one component).

Deliberately NOT truncated to ESM2's 1022-residue limit, unlike
export_pocket_sequences.py: a biological whole-chain structural comparison
should cover the real chain, not an ESM2-visibility-limited prefix. The two
exports only differ for structures over that length (currently Q8MM62,
A0A0V1L202) -- esm2_embed.resolve_dominant_chain() is still used here, but
only to determine WHICH chain is dominant (the chain identity is the same
regardless of truncation); the residue set exported is the chain's full,
untruncated residue set from the source structure.

NOTE ON NAMING: this is the full chain, not a segmented catalytic domain --
no domain-boundary detection is performed. Multi-domain/fusion proteins
are compared as their whole chain, which can understate similarity for a
shared catalytic domain buried in a longer chain with a different
N/C-terminal fusion. Referred to as "structure-remote" (full-chain), not
"domain-remote", throughout this codebase for that reason.

Fails loudly (non-zero exit) on any export failure, for the same reason
export_pocket_sequences.py does: a silently-dropped structure here can
only make the split's grouping guarantees weaker.

CLI:
    python export_dominant_chain_structures.py --manifest configs/manifest.csv \
        --raw-dir data/raw --pockets-dir data/pockets --out-dir data/domain_pdbs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import get_logger, PocketSubgraph, load_structure
from esm2_embed import resolve_dominant_chain
import data_assembly

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    import biotite.structure.io.pdb as pdb

    rows = data_assembly.read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    failures: list[str] = []
    for row in rows:
        pocket_path = args.pockets_dir / f"{row.structure_id}.npz"
        if not pocket_path.exists():
            continue
        try:
            structure_path = data_assembly.resolve_structure(row.source_uri, row.structure_id, args.raw_dir)
            pocket = PocketSubgraph.load(pocket_path)
            # Only the chain identity is used here -- deliberately ignore the
            # truncated res_ids/sequence resolve_dominant_chain also returns,
            # since this export must cover the FULL chain, not the ESM2-visible
            # prefix (see module docstring).
            chain_id, _truncated_res_ids, _truncated_sequence = resolve_dominant_chain(structure_path, pocket)

            arr = load_structure(structure_path)
            arr = arr[~arr.hetero]
            arr = arr[arr.chain_id == chain_id]
            if arr.array_length() == 0:
                raise ValueError(f"no atoms left for chain {chain_id}")

            out_path = args.out_dir / f"{row.structure_id}.pdb"
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(arr)
            pdb_file.write(str(out_path))
            n_ok += 1
        except Exception as e:
            log.error(f"FAILED: {row.structure_id} — {e}")
            failures.append(row.structure_id)

    if failures:
        log.error(f"{len(failures)} structure(s) failed full-chain structure export: {failures}")
        log.error("Refusing to leave a partial export -- fix the failures or explicitly "
                   "exclude these structures from the manifest before re-running.")
        sys.exit(1)

    log.info(f"Exported {n_ok} full-chain structures (untruncated) -> {args.out_dir}")


if __name__ == "__main__":
    main()
