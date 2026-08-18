"""
esm2_embed.py

Computes ESM2 (esm2_t33_650M_UR50D, 1280-dim) per-residue embeddings over
each structure's FULL parent chain (not just the pocket), then slices down
to the residues that actually ended up in that structure's extracted
pocket, saving one aligned .npy per structure_id -- matching the contract
`graph_construction.build_node_features`'s `esm2_embeddings` parameter
expects (shape (n_pocket_residues, 1280), same residue order as
`collapse_to_residue_level`, i.e. ascending res_id).

Run once over the whole manifest before training with --esm2-dir; this is
upstream of pocket_extraction's pocket-level granularity precisely so a
sequence isn't reprocessed once per pocket, and so the embedding captures
the residue's context in the full chain, not just the truncated pocket
subset.

Chain/model selection mirrors utils.load_structure (used by
pocket_extraction.py) exactly -- no chain filtering, first model only --
so residue numbering lines up with what the pocket's res_ids were computed
against. Hetero residues (waters, ions, ligands) ARE excluded here (unlike
the pocket pipeline, which has no hetero flag by the time it reaches
PocketSubgraph) since they aren't part of the amino acid sequence.

CLI:
    python esm2_embed.py --manifest configs/manifest.csv --raw-dir data/raw \
        --pockets-dir data/pockets --out-dir data/esm2_embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from utils import get_logger, load_structure, PocketSubgraph
from graph_construction import collapse_to_residue_level
import data_assembly

log = get_logger(__name__)

MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
ESM2_DIM = 1280
MAX_LENGTH_DEFAULT = 1022  # ESM2 positional embedding limit is 1024 incl. BOS/EOS

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def load_model(device: str):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().to(device)
    return model, alphabet.get_batch_converter(), alphabet


def chain_sequences(structure_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """
    Returns {chain_id: (res_ids ascending, CA coords aligned to res_ids, sequence)}
    for every chain in the full (non-hetero) structure.

    pocket_extraction / collapse_to_residue_level group purely by res_id with
    no chain awareness, so on a multi-chain structure (common for real PDB
    crystal depositions -- multiple copies per asymmetric unit) different
    chains can share the same res_id numbers. Building sequences per-chain
    and resolving each pocket residue by coordinate match (see
    resolve_pocket_residues) avoids silently pulling a residue's identity
    from the wrong chain.
    """
    arr = load_structure(structure_path)
    arr = arr[~arr.hetero]
    out = {}
    for chain_id in sorted(set(arr.chain_id.tolist())):
        chain_arr = arr[arr.chain_id == chain_id]
        res_ids = np.unique(chain_arr.res_id)
        seq_chars, kept_res_ids, kept_ca = [], [], []
        for rid in res_ids:
            mask = chain_arr.res_id == rid
            res_name = chain_arr.res_name[mask][0]
            aa = THREE_TO_ONE.get(res_name)
            if aa is None:
                continue  # skip non-standard residues rather than emit a spurious 'X'
            ca_mask = mask & (chain_arr.atom_name == "CA")
            ca = chain_arr.coord[ca_mask][0] if ca_mask.any() else chain_arr.coord[mask].mean(axis=0)
            seq_chars.append(aa)
            kept_res_ids.append(rid)
            kept_ca.append(ca)
        if seq_chars:
            out[chain_id] = (np.array(kept_res_ids), np.array(kept_ca, dtype=np.float32), "".join(seq_chars))
    return out


def resolve_pocket_residues(
    pocket_res_ids: np.ndarray, pocket_ca_coords: np.ndarray,
    chains: dict[str, tuple[np.ndarray, np.ndarray, str]],
    coord_tol: float = 0.05,
) -> dict[str, list[int]]:
    """
    For each pocket residue, finds which chain's CA coordinate at that
    res_id matches the pocket's own stored coordinate (should be an exact
    copy, since pocket_extraction only subsets/filters the original
    AtomArray -- distances near 0 confirm the right chain, not a same-
    numbered residue from a different copy in the asymmetric unit).

    Returns {chain_id: [pocket-array indices resolved to this chain]}, so
    ESM2 only needs to run once per chain actually used, not once per chain
    that merely exists in the file.
    """
    resolution: dict[str, list[int]] = {}
    for i, (rid, ca) in enumerate(zip(pocket_res_ids, pocket_ca_coords)):
        best_chain, best_dist = None, float("inf")
        for chain_id, (chain_res_ids, chain_ca, _seq) in chains.items():
            match = np.where(chain_res_ids == rid)[0]
            if match.size == 0:
                continue
            d = float(np.linalg.norm(chain_ca[match[0]] - ca))
            if d < best_dist:
                best_dist, best_chain = d, chain_id
        if best_chain is not None and best_dist <= coord_tol:
            resolution.setdefault(best_chain, []).append(i)
    return resolution


def resolve_dominant_chain(
    structure_path: Path, pocket: PocketSubgraph, max_length: int = MAX_LENGTH_DEFAULT,
) -> tuple[str, np.ndarray, str]:
    """
    Returns (chain_id, res_ids, sequence) for the chain that resolves the
    most of this pocket's residues, truncated to max_length exactly the way
    main()'s embedding loop truncates before calling ESM2 -- so every
    consumer of "the sequence for this structure" (sequence export for
    clustering, domain-structure export, and the embedding itself) sees
    identical ESM-visible content. Before this was centralized, the
    exporters wrote the FULL untruncated chain while embed_sequence() only
    ever saw the first max_length residues -- for the 2 structures over the
    limit (Q8MM62, A0A0V1L202), sequence-based clustering was grouping by
    residues ESM2 never actually embedded (and the corresponding pocket
    residues silently got all-zero embeddings past the cutoff).

    Only meaningful for the common case where the pocket resolves cleanly
    to one chain; the true per-residue-multi-chain resolution stays in
    resolve_pocket_residues (used directly by the real embedding loop in
    main()), since a pocket occasionally legitimately spans residues
    resolved to different chains.
    """
    chains = chain_sequences(structure_path)
    residue_level = collapse_to_residue_level(pocket)
    resolution = resolve_pocket_residues(residue_level["res_ids"], residue_level["centroids"], chains)
    if not resolution:
        raise ValueError("no chain resolved for any pocket residue")
    dominant_chain = max(resolution, key=lambda c: len(resolution[c]))
    res_ids, _ca, sequence = chains[dominant_chain]
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
        res_ids = res_ids[:max_length]
    return dominant_chain, res_ids, sequence


@torch.no_grad()
def embed_sequence(model, batch_converter, device: str, sequence: str) -> np.ndarray:
    """Returns (len(sequence), ESM2_DIM) per-residue embedding."""
    _, _, tokens = batch_converter([("query", sequence)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    reps = out["representations"][REPR_LAYER][0]  # (len(seq)+2, D) incl. BOS/EOS
    return reps[1:1 + len(sequence)].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT,
                    help="ESM2 positional embedding limit is 1024 incl. BOS/EOS.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {MODEL_NAME} on {device} (first run downloads ~2.6GB)...")
    model, batch_converter, alphabet = load_model(device)

    rows = data_assembly.read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    for row in rows:
        pocket_path = args.pockets_dir / f"{row.structure_id}.npz"
        out_path = args.out_dir / f"{row.structure_id}.npy"
        if not pocket_path.exists():
            continue  # pocket extraction failed/skipped for this row; nothing to align to
        if out_path.exists():
            n_skip += 1
            continue

        try:
            structure_path = data_assembly.resolve_structure(row.source_uri, row.structure_id, args.raw_dir)
            chains = chain_sequences(structure_path)
            if not chains:
                raise ValueError("no chains with a valid sequence found")

            pocket = PocketSubgraph.load(pocket_path)
            residue_level = collapse_to_residue_level(pocket)
            pocket_res_ids = residue_level["res_ids"]
            pocket_ca = residue_level["centroids"]

            resolution = resolve_pocket_residues(pocket_res_ids, pocket_ca, chains)

            aligned = np.zeros((len(pocket_res_ids), ESM2_DIM), dtype=np.float32)
            resolved = 0
            for chain_id, pocket_indices in resolution.items():
                chain_res_ids, _ca, sequence = chains[chain_id]
                if len(sequence) > args.max_length:
                    log.warning(f"{row.structure_id} chain {chain_id}: sequence length "
                                f"{len(sequence)} > {args.max_length}, truncating.")
                    sequence = sequence[:args.max_length]
                    chain_res_ids = chain_res_ids[:args.max_length]
                per_residue = embed_sequence(model, batch_converter, device, sequence)
                row_idx_of = {int(rid): i for i, rid in enumerate(chain_res_ids)}
                for pi in pocket_indices:
                    row_idx = row_idx_of.get(int(pocket_res_ids[pi]))
                    if row_idx is None:
                        continue  # truncated away
                    aligned[pi] = per_residue[row_idx]
                    resolved += 1

            missing = len(pocket_res_ids) - resolved
            if missing:
                unresolved_names = sorted(set(
                    residue_level["res_names"][i] for i in range(len(pocket_res_ids))
                    if i not in {pi for idxs in resolution.values() for pi in idxs}
                ))
                log.warning(f"{row.structure_id}: {missing}/{len(pocket_res_ids)} pocket "
                            f"residues left as zero-embedding ({', '.join(unresolved_names)}) -- "
                            f"expected for non-amino-acid pseudo-residues (waters, ions) that "
                            f"pocket_extraction.py includes without hetero filtering; a genuine "
                            f"chain-resolution failure would show a standard 3-letter AA code here.")

            np.save(out_path, aligned)
            n_ok += 1
            if n_ok % 50 == 0:
                log.info(f"...{n_ok} done")
        except Exception as e:
            log.error(f"FAILED: {row.structure_id} — {e}")
            n_fail += 1

    log.info(f"ESM2 embedding complete: {n_ok} ok, {n_skip} already done, {n_fail} failed.")


if __name__ == "__main__":
    main()
