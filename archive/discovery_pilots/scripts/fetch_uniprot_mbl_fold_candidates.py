"""
fetch_uniprot_mbl_fold_candidates.py

Pilot prospective-discovery candidate sourcing (see the project plan for
the B1-only structural detector pilot): queries UniProt for reviewed
(Swiss-Prot) bacterial proteins annotated with the metallo-beta-lactamase
superfamily fold (Pfam PF00753), excludes every accession already in the
labeled corpus (configs/manifest.csv, full_structure_catalog.csv), and
fetches each remaining candidate's AlphaFold DB predicted structure.

Restricted to reviewed+bacteria to keep the pilot a tractable size (389
total before exclusion, well within the planned few-hundred-to-2000
pilot scope) and to favor higher-quality sequence/annotation data for
interpreting the triage shortlist later -- this is a genuine "outside
the training corpus" candidate pool, not a resample of already-labeled
data.

Exclusion is exact-string match against both configs/manifest.csv's
structure_id column and full_structure_catalog.csv's accession column
(union) -- deliberately not truncated/parsed (a prior bug in this
project came from cleverly truncating versioned accessions; exact
match only).

CLI:
    python fetch_uniprot_mbl_fold_candidates.py \
        --manifest configs/manifest.csv --catalog full_structure_catalog.csv \
        --out-dir data/b1_pilot --pfam PF00753 --taxonomy-id 2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import requests

from utils import get_logger

log = get_logger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.pdb"


def load_exclusion_set(manifest_path: Path, catalog_path: Path) -> set[str]:
    excluded = set()
    for row in csv.DictReader(open(manifest_path)):
        excluded.add(row["structure_id"])
    for row in csv.DictReader(open(catalog_path)):
        excluded.add(row["accession"])
    return excluded


def query_uniprot(pfam: str, taxonomy_id: str) -> list[dict]:
    query = f"xref:pfam-{pfam} AND reviewed:true AND taxonomy_id:{taxonomy_id}"
    fields = "accession,sequence,organism_name,protein_name,ec,cc_function"
    results, cursor = [], None
    while True:
        params = {"query": query, "format": "json", "size": 500, "fields": fields}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data["results"])
        link = resp.headers.get("Link", "")
        cursor = None
        if 'rel="next"' in link:
            next_url = link.split(";")[0].strip("<> ")
            cursor = dict(p.split("=") for p in next_url.split("?", 1)[1].split("&")).get("cursor")
        if not cursor:
            break
    return results


def parse_uniprot_entry(entry: dict) -> dict:
    protein_desc = entry.get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
    function_texts = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for t in comment.get("texts", []):
                function_texts.append(t.get("value", ""))
    return {
        "accession": entry["primaryAccession"],
        "sequence": entry.get("sequence", {}).get("value", ""),
        "organism": entry.get("organism", {}).get("scientificName", ""),
        "protein_name": rec_name,
        "function_annotation": " ".join(function_texts),
    }


def fetch_alphafold_structure(accession: str, out_path: Path) -> dict | None:
    if out_path.exists():
        return {"status": "cached"}
    try:
        resp = requests.get(ALPHAFOLD_API_URL.format(accession=accession), timeout=30)
        if resp.status_code != 200 or not resp.json():
            return None
        meta = resp.json()[0]
        version = meta["latestVersion"]
        pdb_url = ALPHAFOLD_PDB_URL.format(accession=accession, version=version)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["curl", "-sfL", "-o", str(out_path), pdb_url], timeout=60,
        )
        if result.returncode != 0 or not out_path.exists():
            return None
        return {
            "status": "fetched", "af_version": version,
            "global_plddt": meta.get("globalMetricValue"),
            "fraction_plddt_very_low": meta.get("fractionPlddtVeryLow"),
        }
    except Exception as e:
        log.warning(f"{accession}: AlphaFold fetch failed: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--pfam", default="PF00753")
    p.add_argument("--taxonomy-id", default="2", help="NCBI taxonomy id, default 2 = Bacteria")
    p.add_argument("--limit", type=int, default=None, help="cap candidates fetched, for a dry run")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = args.out_dir / "structures"

    excluded = load_exclusion_set(args.manifest, args.catalog)
    log.info(f"Loaded {len(excluded)} corpus accessions to exclude")

    entries = query_uniprot(args.pfam, args.taxonomy_id)
    log.info(f"UniProt query returned {len(entries)} reviewed bacterial {args.pfam} entries")

    candidates = [parse_uniprot_entry(e) for e in entries]
    candidates = [c for c in candidates if c["accession"] not in excluded]
    log.info(f"{len(candidates)} candidates remain after excluding corpus accessions")

    if args.limit:
        candidates = candidates[: args.limit]
        log.info(f"Limited to {len(candidates)} candidates for this run")

    n_fetched, n_failed = 0, 0
    for i, c in enumerate(candidates):
        out_path = structures_dir / f"{c['accession']}.pdb"
        af_result = fetch_alphafold_structure(c["accession"], out_path)
        if af_result is None:
            c["structure_status"] = "unavailable"
            n_failed += 1
        else:
            c.update(af_result)
            c["structure_status"] = af_result["status"]
            c["structure_path"] = str(out_path)
            n_fetched += 1
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(candidates)} processed ({n_fetched} fetched, {n_failed} unavailable)")
        time.sleep(0.1)  # be polite to the AlphaFold DB API

    manifest_out = args.out_dir / "candidate_manifest.json"
    manifest_out.write_text(json.dumps(candidates, indent=2))
    log.info(f"Done. {n_fetched} structures fetched, {n_failed} unavailable -> {manifest_out}")


if __name__ == "__main__":
    main()
