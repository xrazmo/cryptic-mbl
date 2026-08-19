"""Run a published fARGene full-protein HMM on a protein FASTA.

This adapter calls HMMER directly rather than the legacy Python-2 fARGene
front end.  It preserves its full-protein classifier: ``hmmsearch -E 1000
--domE 1000`` followed by a positive call when any domain score (domtblout
column 14) exceeds the supplied model-specific threshold. ``--sensitive`` adds
HMMER ``--max``, matching fARGene's optional sensitive mode; it is off by
default in the upstream CLI.

The fARGene model is supplied by path and is not copied into this repository.
The output records the upstream revision and content hash when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_THRESHOLD = 127.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fasta_ids(path: Path) -> list[str]:
    ids = [line[1:].split()[0] for line in path.read_text().splitlines() if line.startswith(">")]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate FASTA identifiers")
    return ids


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_domtblout(path: Path, ids: list[str]) -> dict[str, dict]:
    hits: dict[str, list[dict]] = {sid: [] for sid in ids}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 23:
            raise ValueError(f"Malformed HMMER domtblout row: {line}")
        sid = fields[0]
        if sid not in hits:
            raise ValueError(f"HMMER returned unknown FASTA ID: {sid}")
        hits[sid].append({
            "domain_score": float(fields[13]),
            "domain_i_evalue": float(fields[12]),
            "hmm_from": int(fields[15]),
            "hmm_to": int(fields[16]),
            "ali_from": int(fields[17]),
            "ali_to": int(fields[18]),
        })
    output = {}
    for sid, rows in hits.items():
        best = max(rows, key=lambda row: row["domain_score"]) if rows else None
        output[sid] = {"best_domain_score": best["domain_score"] if best else None,
                       "best_domain": best, "n_domains": len(rows)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--hmm", required=True, type=Path)
    parser.add_argument("--hmmsearch", default="hmmsearch",
                        help="HMMER hmmsearch executable or absolute path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--model-name",
        help="Report label for the supplied HMM (default: HMM filename stem)",
    )
    parser.add_argument("--sensitive", action="store_true",
                        help="Use fARGene's optional sensitive mode (HMMER --max)")
    parser.add_argument("--upstream-checkout", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domtblout", type=Path)
    args = parser.parse_args()
    model_name = args.model_name or args.hmm.stem

    ids = fasta_ids(args.fasta)
    if not ids:
        raise ValueError("input FASTA is empty")
    with tempfile.TemporaryDirectory(prefix="fargene-comparator-") as tmp:
        domtblout = args.domtblout or Path(tmp) / "hits.domtblout"
        domtblout.parent.mkdir(parents=True, exist_ok=True)
        command = [args.hmmsearch]
        if args.sensitive:
            command.append("--max")
        command += [
            "-E", "1000", "--domE", "1000",
            "--domtblout", str(domtblout), str(args.hmm), str(args.fasta),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        scores = parse_domtblout(domtblout, ids)

    for row in scores.values():
        score = row["best_domain_score"]
        row["predicted_positive"] = score is not None and score > args.threshold
    hmm_version = next(
        (line.lstrip("# ") for line in completed.stdout.splitlines() if "HMMER " in line),
        None,
    )
    output = {
        "schema_version": 1,
        "method": f"fARGene {model_name} full-protein HMM",
        "upstream_url": "https://github.com/fannyhb/fargene",
        "upstream_revision": git_revision(args.upstream_checkout) if args.upstream_checkout else None,
        "hmm_sha256": sha256(args.hmm),
        "fasta_sha256": sha256(args.fasta),
        "threshold": {"operator": ">", "domain_score": args.threshold,
                      "source": f"fargene_analysis.py predefined {model_name} model"},
        "sensitive_mode": args.sensitive,
        "command": command,
        "hmmsearch_version": hmm_version,
        "n_sequences": len(ids),
        "n_positive": sum(row["predicted_positive"] for row in scores.values()),
        "per_example": scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
