"""
data_assembly.py — Task 3 (spec §2, §12.3)

Reads a manifest describing where each structure lives (local path or a
BLDB/PDB/UniProt accession to fetch) and what it is (positive MBL, Pfam
CL0381 hard negative, reference-bank anchor, etc.), fetches/validates the
structures, tags confidence tiers, and calls pocket_extraction.py for each
entry to produce the pocket dataset consumed by graph_construction.py.

This script does NOT hardcode a hit list of PDB/UniProt IDs — BLDB, Pfam,
and NCBI access all require network calls the sandbox this was authored in
cannot make, and the actual composition of the positive/negative sets is a
curation decision for Mo to make and version-control. Instead this provides:
  - a manifest schema (CSV) that encodes every field the rest of the
    pipeline depends on,
  - a fetcher that resolves "pdb:XXXX" / "local:path" URIs,
  - validation that flags manifest rows likely to break downstream steps
    (missing tier, missing label, reference-bank entries also marked
    "positive" and therefore at leakage risk — see spec §7).

Manifest CSV columns:
    structure_id, source_uri, label, tier, subclass, is_predicted, is_reference_bank

    source_uri examples:
        pdb:5YD4                    -> fetch from RCSB
        local:/data/raw/ndm1.pdb    -> already on disk
        af:/data/af_preds/xxx.pdb   -> AlphaFold/ESMFold output (treated as predicted)

CLI:
    python data_assembly.py --manifest manifest.csv --raw-dir data/raw \
        --pocket-out-dir data/pockets --report-out data/assembly_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils import get_logger, CONFIDENCE_TIERS
import pocket_extraction

log = get_logger(__name__)

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass
class ManifestRow:
    structure_id: str
    source_uri: str
    label: str
    tier: int
    subclass: Optional[str]
    is_predicted: bool
    is_reference_bank: bool


def read_manifest(path: Path) -> list[ManifestRow]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(ManifestRow(
                structure_id=r["structure_id"],
                source_uri=r["source_uri"],
                label=r["label"],
                tier=int(r["tier"]),
                subclass=r["subclass"] or None,
                is_predicted=r["is_predicted"].lower() in ("1", "true", "yes"),
                is_reference_bank=r["is_reference_bank"].lower() in ("1", "true", "yes"),
            ))
    return rows


def validate_manifest(rows: list[ManifestRow]) -> list[str]:
    """Returns a list of human-readable warnings/errors; empty = clean."""
    problems = []
    ids_seen = set()
    for r in rows:
        if r.structure_id in ids_seen:
            problems.append(f"Duplicate structure_id: {r.structure_id}")
        ids_seen.add(r.structure_id)

        if r.tier not in CONFIDENCE_TIERS:
            problems.append(f"{r.structure_id}: invalid tier {r.tier}")

        if r.label not in ("positive", "hard_negative", "easy_negative", "unlabeled"):
            problems.append(f"{r.structure_id}: invalid label '{r.label}'")

        if r.is_reference_bank and r.label != "positive":
            problems.append(
                f"{r.structure_id}: marked as reference_bank but label='{r.label}' "
                "— reference bank / external holdout should be positives excluded "
                "from training (see spec §2, §7)."
            )

        if r.label == "positive" and r.subclass is None:
            problems.append(
                f"{r.structure_id}: positive with no subclass — needed for "
                "leave-one-subclass-out splitting (spec §7)."
            )
    return problems


def resolve_structure(uri: str, structure_id: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if uri.startswith("local:") or uri.startswith("af:"):
        src = Path(uri.split(":", 1)[1])
        if not src.exists():
            raise FileNotFoundError(f"{structure_id}: local file not found at {src}")
        dst = raw_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        return dst
    if uri.startswith("pdb:"):
        pdb_id = uri.split(":", 1)[1].upper()
        dst = raw_dir / f"{pdb_id}.pdb"
        if dst.exists():
            return dst
        url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
        try:
            subprocess.run(["curl", "-sfL", "-o", str(dst), url], check=True, timeout=60)
        except Exception as e:
            raise RuntimeError(
                f"{structure_id}: failed to fetch {url} ({e}). "
                "If running in a network-restricted environment, pre-download "
                "structures and point the manifest at local: paths instead."
            )
        return dst
    raise ValueError(f"{structure_id}: unrecognized source_uri scheme in '{uri}'")


def assemble(manifest_path: Path, raw_dir: Path, pocket_out_dir: Path, report_out: Path):
    rows = read_manifest(manifest_path)
    problems = validate_manifest(rows)
    if problems:
        log.warning(f"Manifest validation found {len(problems)} issue(s):")
        for p in problems:
            log.warning(f"  - {p}")

    report = {"n_rows": len(rows), "problems": problems, "processed": [], "failed": []}

    for row in rows:
        try:
            structure_path = resolve_structure(row.source_uri, row.structure_id, raw_dir)
            pocket = pocket_extraction.extract_pocket(
                structure_path=structure_path,
                structure_id=row.structure_id,
                label=row.label,
                confidence_tier=row.tier,
                subclass=row.subclass,
                is_predicted=row.is_predicted,
            )
            pocket_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = pocket_out_dir / f"{row.structure_id}.npz"
            pocket.save(out_path)
            report["processed"].append({
                "structure_id": row.structure_id,
                "pocket_path": str(out_path),
                "pocket_source": pocket.metadata.pocket_source,
                "is_reference_bank": row.is_reference_bank,
            })
            log.info(f"OK: {row.structure_id}")
        except Exception as e:
            log.error(f"FAILED: {row.structure_id} — {e}")
            report["failed"].append({"structure_id": row.structure_id, "error": str(e)})

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))
    log.info(
        f"Assembly complete: {len(report['processed'])} ok, "
        f"{len(report['failed'])} failed. Report -> {report_out}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--pocket-out-dir", required=True, type=Path)
    p.add_argument("--report-out", required=True, type=Path)
    args = p.parse_args()
    assemble(args.manifest, args.raw_dir, args.pocket_out_dir, args.report_out)


if __name__ == "__main__":
    main()
