"""
build_split_graph.py

Builds TWO independent grouping regimes for clustering_split.py, instead of
one conflated graph. An earlier version unioned pocket-fragment structural
edges (Foldseek RMSD<2A OR pident>60% on the isolated ~40-residue pocket)
with sequence edges into a single graph -- that pocket-fragment criterion
turned out to chain 926/1077 structures into one component, including
structurally-unrelated decoy folds (globin, TIM-barrel, Rossmann-SDR)
alongside real MBLs. Small-fragment TM-align/RMSD comparisons are not a
reliable same-fold signal; the criterion was measuring alignment noise, not
homology. See git history for that version and the audit that found this.

The two regimes answer different scientific questions and must NOT be
merged into one split:

  - PRIMARY / "sequence-remote": groups by sequence homology of the exact
    full chains ESM2 embeds (mmseqs2 easy-search, edge if pident >=
    --seq-identity-threshold AND both qcov/tcov >= --seq-coverage-threshold).
    Structurally similar proteins ARE allowed across train/test here --
    that's the point: it tests whether structural/chemical features help
    when close sequence recognition is unavailable.

  - SECONDARY / "structure-remote": groups by WHOLE-CHAIN structural
    similarity (Foldseek TM-align on export_dominant_chain_structures.py's
    single-chain PDBs -- not the pocket fragment), edge if
    min(qtmscore, ttmscore) >= --tm-threshold AND both qcov/tcov >=
    --struct-coverage-threshold. Tests generalization to a genuinely novel
    fold topology. A stricter --tm-redundancy-threshold is also computed
    and reported (not used for grouping) to distinguish "fold-remote" from
    "structurally near-redundant". NOTE: this is the whole chain, NOT a
    segmented catalytic domain -- no domain-boundary detection is done, so
    a multi-domain/fusion protein is compared as its entire chain, which
    can understate similarity for a shared catalytic domain buried in a
    longer chain with a different fusion partner. Called "structure-remote
    (full-chain)" throughout, deliberately not "domain-remote".

Pocket-fragment structural similarity is STILL computed here, but purely
as a diagnostic/reporting signal (max cross-partition pocket similarity in
clustering_split.py's audit block) -- never as a grouping edge again.

Both searches use --max-seqs comfortably above the dataset size (default
5000, well over n=1077) rather than trusting either tool's default
prefilter cap (mmseqs: 300, foldseek: 1000) -- an earlier run silently hit
those caps for 532/1077 and 960/1077 queries respectively, meaning the
"all-vs-all" search was truncated and the resulting components could only
be an UNDER-estimate of true connectivity (missing edges can merge
components once found, never split them further).

main() also asserts that the manifest's pocket-having structure_ids, the
FASTA record ids, and both PDB directories' ids are all EXACTLY the same
set before running any search -- a silent mismatch here (a failed export,
a stale directory) would make the graph wrong in a way that's easy to miss.

CLI:
    python build_split_graph.py \
        --pocket-pdb-dir data/pocket_pdbs --domain-pdb-dir data/domain_pdbs \
        --sequences-fasta data/pocket_sequences.fasta \
        --work-dir data/split_graph --out data/split_graph.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from utils import get_logger

log = get_logger(__name__)

DEFAULT_SEQ_IDENTITY_THRESHOLD = 0.3
DEFAULT_SEQ_COVERAGE_THRESHOLD = 0.8
DEFAULT_TM_THRESHOLD = 0.5          # fold-remote grouping edge (Xu & Zhang 2010 same-fold rule of thumb)
DEFAULT_TM_REDUNDANCY_THRESHOLD = 0.65  # stricter, reported only
DEFAULT_STRUCT_COVERAGE_THRESHOLD = 0.8
DEFAULT_MAX_SEQS = 5000  # comfortably above n=1077; mmseqs/foldseek default to 300/1000


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _strip_pdb_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".pdb") else name


def components_from_edges(all_ids: list[str], edge_pairs) -> dict[str, list[str]]:
    uf = UnionFind(all_ids)
    id_set = set(all_ids)
    for u, v in edge_pairs:
        if u in id_set and v in id_set:
            uf.union(u, v)
    components: dict[str, list[str]] = {}
    for sid in all_ids:
        components.setdefault(uf.find(sid), []).append(sid)
    return components


# --------------------------------------------------------------------------- #
# Sequence similarity (mmseqs2) -- PRIMARY grouping regime
# --------------------------------------------------------------------------- #

def run_mmseqs_search(fasta_path: Path, work_dir: Path, max_seqs: int = DEFAULT_MAX_SEQS) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = work_dir / "sequence_pairs.tsv"
    tmp = work_dir / "mmseqs_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        [
            "mmseqs", "easy-search", str(fasta_path), str(fasta_path),
            str(out_tsv), str(tmp),
            "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue",
            "-e", "1000", "--min-seq-id", "0.0", "-c", "0", "--cov-mode", "0",
            "--max-seqs", str(max_seqs),
        ],
        check=True, capture_output=True, timeout=1800,
    )
    return out_tsv


def parse_sequence_pairs(tsv_path: Path):
    """Yields (id1, id2, pident, min_cov) for every pair, unfiltered."""
    with open(tsv_path) as f:
        for line in f:
            query, target, pident, _alnlen, qcov, tcov, _evalue = line.strip().split("\t")
            if query == target:
                continue
            yield query, target, float(pident) / 100.0, min(float(qcov), float(tcov))


def sequence_edges(tsv_path: Path, identity_threshold: float, coverage_threshold: float):
    for u, v, pident, cov in parse_sequence_pairs(tsv_path):
        if pident >= identity_threshold and cov >= coverage_threshold:
            yield u, v


# --------------------------------------------------------------------------- #
# Whole-domain structural similarity (Foldseek TM-align) -- SECONDARY regime
# --------------------------------------------------------------------------- #

def run_foldseek_domain_search(domain_pdb_dir: Path, work_dir: Path, max_seqs: int = DEFAULT_MAX_SEQS) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = work_dir / "domain_pairs.tsv"
    tmp = work_dir / "foldseek_domain_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        [
            "foldseek", "easy-search", str(domain_pdb_dir), str(domain_pdb_dir),
            str(out_tsv), str(tmp),
            "--alignment-type", "1",  # TM-align
            "-e", "1000",
            "--format-output", "query,target,pident,alnlen,qcov,tcov,qtmscore,ttmscore,evalue",
            "--max-seqs", str(max_seqs),
        ],
        check=True, capture_output=True, timeout=1800,
    )
    return out_tsv


def parse_domain_pairs(tsv_path: Path):
    """Yields (id1, id2, min_tmscore, min_cov) for every pair, unfiltered."""
    with open(tsv_path) as f:
        for line in f:
            query, target, _pident, _alnlen, qcov, tcov, qtm, ttm, _evalue = line.strip().split("\t")
            query, target = _strip_pdb_suffix(query), _strip_pdb_suffix(target)
            if query == target:
                continue
            yield query, target, min(float(qtm), float(ttm)), min(float(qcov), float(tcov))


def domain_structure_edges(tsv_path: Path, tm_threshold: float, coverage_threshold: float):
    for u, v, min_tm, min_cov in parse_domain_pairs(tsv_path):
        if min_tm >= tm_threshold and min_cov >= coverage_threshold:
            yield u, v


# --------------------------------------------------------------------------- #
# Pocket-fragment structural similarity -- DIAGNOSTIC ONLY, never grouping
# --------------------------------------------------------------------------- #

def run_foldseek_pocket_search(pocket_pdb_dir: Path, work_dir: Path, max_seqs: int = DEFAULT_MAX_SEQS) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = work_dir / "pocket_pairs.tsv"
    tmp = work_dir / "foldseek_pocket_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        [
            "foldseek", "easy-search", str(pocket_pdb_dir), str(pocket_pdb_dir),
            str(out_tsv), str(tmp),
            "--alignment-type", "1",
            "-e", "1000",
            "--format-output", "query,target,pident,alnlen,rmsd,qcov,tcov,qtmscore,ttmscore,evalue",
            "--max-seqs", str(max_seqs),
        ],
        check=True, capture_output=True, timeout=1800,
    )
    return out_tsv


def tool_version(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def fasta_ids(fasta_path: Path) -> list[str]:
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    return ids


def assert_consistent_ids(
    manifest_path: Path, pockets_dir: Path, fasta_path: Path,
    domain_pdb_dir: Path, pocket_pdb_dir: Path,
) -> list[str]:
    """
    All four id sources must be EXACTLY the same set -- a silent mismatch
    here (a failed export that didn't propagate, a stale directory left
    over from a previous run) would make the similarity graph wrong for
    whichever structures are missing from one side, in a way that's easy
    to miss (a structure simply never gets an edge, so it looks like its
    own component -- indistinguishable from "genuinely has no homologs").
    """
    import data_assembly
    manifest_rows = data_assembly.read_manifest(manifest_path)
    manifest_ids = {r.structure_id for r in manifest_rows if (pockets_dir / f"{r.structure_id}.npz").exists()}

    fasta_id_list = fasta_ids(fasta_path)
    if len(fasta_id_list) != len(set(fasta_id_list)):
        dupes = sorted({x for x in fasta_id_list if fasta_id_list.count(x) > 1})
        raise AssertionError(f"Duplicate FASTA record ids: {dupes}")

    id_sets = {
        "manifest (pocket-having rows)": manifest_ids,
        "sequences FASTA": set(fasta_id_list),
        "domain/full-chain PDB dir": {f.stem for f in domain_pdb_dir.glob("*.pdb")},
        "pocket PDB dir": {f.stem for f in pocket_pdb_dir.glob("*.pdb")},
    }
    reference = id_sets["manifest (pocket-having rows)"]
    problems = []
    for name, ids in id_sets.items():
        missing = reference - ids
        extra = ids - reference
        if missing:
            problems.append(f"{name}: missing {len(missing)} ids present in manifest, e.g. {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{name}: has {len(extra)} ids NOT in manifest, e.g. {sorted(extra)[:5]}")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--pocket-pdb-dir", required=True, type=Path)
    p.add_argument("--domain-pdb-dir", required=True, type=Path)
    p.add_argument("--sequences-fasta", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--seq-identity-threshold", type=float, default=DEFAULT_SEQ_IDENTITY_THRESHOLD)
    p.add_argument("--seq-coverage-threshold", type=float, default=DEFAULT_SEQ_COVERAGE_THRESHOLD)
    p.add_argument("--tm-threshold", type=float, default=DEFAULT_TM_THRESHOLD)
    p.add_argument("--tm-redundancy-threshold", type=float, default=DEFAULT_TM_REDUNDANCY_THRESHOLD)
    p.add_argument("--struct-coverage-threshold", type=float, default=DEFAULT_STRUCT_COVERAGE_THRESHOLD)
    p.add_argument("--max-seqs", type=int, default=DEFAULT_MAX_SEQS)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    log.info("Checking manifest/FASTA/PDB-dir id consistency before running any search...")
    problems = assert_consistent_ids(
        args.manifest, args.pockets_dir, args.sequences_fasta, args.domain_pdb_dir, args.pocket_pdb_dir,
    )
    if problems:
        for p_ in problems:
            log.error(f"  - {p_}")
        raise AssertionError(
            f"{len(problems)} id-consistency problem(s) found -- fix the upstream export(s) "
            f"before building the split graph (see log above)."
        )
    log.info("IDs consistent across manifest, FASTA, and both PDB directories.")

    all_ids = sorted(f.stem for f in args.domain_pdb_dir.glob("*.pdb"))
    log.info(f"{len(all_ids)} structures.")

    # --- primary: sequence ---
    log.info("Running mmseqs all-vs-all sequence search (exact ESM2 chains)...")
    seq_tsv = run_mmseqs_search(args.sequences_fasta, args.work_dir, args.max_seqs)
    seq_edge_list = list(sequence_edges(seq_tsv, args.seq_identity_threshold, args.seq_coverage_threshold))
    sequence_components = components_from_edges(all_ids, seq_edge_list)
    log.info(f"PRIMARY (sequence-remote): {len(seq_edge_list)} edges, "
             f"{len(sequence_components)} components, largest="
             f"{max(len(m) for m in sequence_components.values())}.")

    # --- secondary: whole-domain structure ---
    log.info("Running foldseek all-vs-all whole-domain structural search...")
    domain_tsv = run_foldseek_domain_search(args.domain_pdb_dir, args.work_dir, args.max_seqs)
    domain_edge_list_foldremote = list(domain_structure_edges(
        domain_tsv, args.tm_threshold, args.struct_coverage_threshold))
    structure_components_foldremote = components_from_edges(all_ids, domain_edge_list_foldremote)
    log.info(f"SECONDARY (structure-remote, TM>={args.tm_threshold}): "
             f"{len(domain_edge_list_foldremote)} edges, "
             f"{len(structure_components_foldremote)} components, largest="
             f"{max(len(m) for m in structure_components_foldremote.values())}.")

    domain_edge_list_redundancy = list(domain_structure_edges(
        domain_tsv, args.tm_redundancy_threshold, args.struct_coverage_threshold))
    structure_components_redundancy = components_from_edges(all_ids, domain_edge_list_redundancy)
    log.info(f"REPORTED (structure near-redundancy, TM>={args.tm_redundancy_threshold}): "
             f"{len(domain_edge_list_redundancy)} edges, "
             f"{len(structure_components_redundancy)} components, largest="
             f"{max(len(m) for m in structure_components_redundancy.values())}.")

    # --- diagnostic only: pocket-fragment structure, never grouping ---
    log.info("Running foldseek all-vs-all POCKET-fragment search (diagnostic only, not used for grouping)...")
    pocket_tsv = run_foldseek_pocket_search(args.pocket_pdb_dir, args.work_dir, args.max_seqs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "note": "structure_components_* are built from WHOLE-CHAIN comparison "
                "(export_dominant_chain_structures.py), not a segmented catalytic "
                "domain -- no domain-boundary detection is performed.",
        "thresholds": {
            "sequence_identity_threshold": args.seq_identity_threshold,
            "sequence_coverage_threshold": args.seq_coverage_threshold,
            "tm_threshold_fold_remote": args.tm_threshold,
            "tm_threshold_redundancy": args.tm_redundancy_threshold,
            "struct_coverage_threshold": args.struct_coverage_threshold,
            "max_seqs": args.max_seqs,
        },
        "tool_versions": {
            "foldseek": tool_version(["foldseek", "version"]),
            "mmseqs": tool_version(["mmseqs", "version"]),
        },
        "n_structures": len(all_ids),
        "raw_pair_files": {
            "sequence_pairs_tsv": str(seq_tsv),
            "domain_pairs_tsv": str(domain_tsv),
            "pocket_pairs_tsv (diagnostic only)": str(pocket_tsv),
        },
        "sequence_components": sequence_components,
        "structure_components_foldremote": structure_components_foldremote,
        "structure_components_redundancy": structure_components_redundancy,
    }, indent=2))
    log.info(f"Wrote split graph -> {args.out}")


if __name__ == "__main__":
    main()
