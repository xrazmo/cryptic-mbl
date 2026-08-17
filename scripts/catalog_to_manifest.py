"""
catalog_to_manifest.py

Converts full_structure_catalog.csv (1070 rows: positive / hard_negative /
true_negative, sourced from CARD / Berglund2017 / Gudeta2016b / UniProt) into
the manifest schema data_assembly.py consumes (see configs/manifest_example.csv):
    structure_id, source_uri, label, tier, subclass, is_predicted, is_reference_bank

Mappings applied:
  - label: true_negative -> easy_negative (unrelated-fold decoys; positive and
    hard_negative pass through unchanged). data_assembly/pocket_extraction/
    graph_construction only ever recognize the four canonical labels, and
    true_negative was never one of them.
  - tier: every catalog structure is predicted (afdb or manual AF3), so tiers
    1-2 (crystal) never apply here.
        source == CARD                -> tier 3 (curated resistance DB, gene
                                          entries carry a documented phenotype)
        source in {Berglund2017,
                   Gudeta2016b}        -> tier 3 (published, curated positive
                                          sets, not bulk-mined)
        source == UniProt              -> tier 4 (bulk-annotated negatives,
                                          no phenotype evidence)
    NOTE: this is a source-based heuristic, not derived from `confidence_score`
    (which is an AF3 structure-prediction confidence, unrelated to the
    evidence-tier definition in utils.CONFIDENCE_TIERS). Flag for review if
    Berglund2017/Gudeta2016b positives should instead be tier 4.
  - source_uri: local:{structures_dir}/{source_filename} (source_filename
    matches files in final_structures/ for all 1070 rows; original_structure_path
    in the catalog is stale and does not).
  - is_predicted: always true (no crystal structures in this catalog).
  - is_reference_bank: always false for catalog rows — the reference bank is
    a small hand-picked set of real crystal structures (NDM-1, VIM-2, IMP-1,
    CphA, Sfh-I, L1, FEZ-1), prepended separately, fetched live from RCSB.
    None of those accessions exist in this catalog.

CLI:
    python catalog_to_manifest.py \
        --catalog full_structure_catalog.csv --structures-dir final_structures \
        --out configs/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABEL_MAP = {
    "positive": "positive",
    "hard_negative": "hard_negative",
    "true_negative": "easy_negative",
}

SOURCE_TIER = {
    "CARD": 3,
    "Berglund2017": 3,
    "Gudeta2016b": 3,
    "UniProt": 4,
}

# Real crystal structures, fetched live from RCSB — not derived from the
# catalog. Tier 1 (crystal + published kinetics). Covers B1/B2/B3.
REFERENCE_BANK = [
    # structure_id, pdb_id, subclass
    ("NDM-1", "3SPU", "B1"),
    ("VIM-2", "1KO3", "B1"),
    ("IMP-1", "1DD6", "B1"),
    ("CphA", "1X8G", "B2"),
    ("Sfh-I", "2QDS", "B2"),
    ("L1", "1SML", "B3"),
    ("FEZ-1", "1K07", "B3"),
]


def convert_catalog_rows(catalog_path: Path, structures_dir: Path) -> list[dict]:
    rows = []
    with open(catalog_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = LABEL_MAP.get(r["label"])
            if label is None:
                raise ValueError(f"{r['accession']}: unrecognized catalog label '{r['label']}'")

            source_filename = r["source_filename"]
            structure_path = structures_dir / source_filename
            if not structure_path.exists():
                raise FileNotFoundError(
                    f"{r['accession']}: {structure_path} not found (source_filename mismatch)"
                )

            tier = SOURCE_TIER.get(r["source"])
            if tier is None:
                raise ValueError(f"{r['accession']}: unrecognized catalog source '{r['source']}'")

            subclass = r["subclass"] or ""

            rows.append({
                "structure_id": r["accession"],
                "source_uri": f"local:{structure_path}",
                "label": label,
                "tier": tier,
                "subclass": subclass,
                "is_predicted": "true",
                "is_reference_bank": "false",
            })
    return rows


def reference_bank_rows() -> list[dict]:
    return [
        {
            "structure_id": sid,
            "source_uri": f"pdb:{pdb_id}",
            "label": "positive",
            "tier": 1,
            "subclass": subclass,
            "is_predicted": "false",
            "is_reference_bank": "true",
        }
        for sid, pdb_id, subclass in REFERENCE_BANK
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--structures-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--skip-reference-bank", action="store_true",
                    help="Omit the hand-picked crystal-structure reference bank (e.g. for offline dry runs).")
    args = p.parse_args()

    rows = [] if args.skip_reference_bank else reference_bank_rows()
    rows += convert_catalog_rows(args.catalog, args.structures_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "structure_id", "source_uri", "label", "tier", "subclass",
            "is_predicted", "is_reference_bank",
        ])
        writer.writeheader()
        writer.writerows(rows)

    n_ref = 0 if args.skip_reference_bank else len(REFERENCE_BANK)
    print(f"Wrote {len(rows)} rows ({n_ref} reference-bank + {len(rows) - n_ref} catalog) -> {args.out}")


if __name__ == "__main__":
    main()