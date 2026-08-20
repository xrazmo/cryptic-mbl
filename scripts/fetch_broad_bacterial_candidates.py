"""
fetch_broad_bacterial_candidates.py

Broad, unbiased-by-MBL-relevance candidate sourcing for the second B1
discovery pilot. Deliberately different from
fetch_uniprot_mbl_fold_candidates.py's PF00753-filtered pool: that
filter is itself a sequence-profile selection, so restricting to it
biases toward exactly what a sequence method would already flag --
undermining a genuine beyond-HMM test (see project discussion).

Selection here uses only domain-agnostic constraints:
  - bacterial (taxonomy_id:2), unreviewed (TrEMBL, not Swiss-Prot --
    reviewed entries are by definition already well-characterized, the
    opposite of "cryptic")
  - protein length in a broad plausible-MBL-fold range (default 180-400
    residues)
  - a loose, non-MBL-specific composition floor: at least
    MIN_HIS/MIN_ASP/MIN_CYS of the three donor-relevant residue types,
    since six_donor pharmacophore matching is mechanically impossible
    without them regardless of fold -- this is a compute-saving floor,
    not a fold/family filter
  - excludes every accession already in the labeled corpus (same
    exact-match convention as fetch_uniprot_mbl_fold_candidates.py)

No Pfam/InterPro/domain annotation is used to select candidates. PF00753
presence, if any, is looked up afterward as a novelty ANNOTATION, not a
filter -- see run_b1_broad_discovery.py.

Diversity caveat, stated explicitly rather than assumed: UniProt's REST
API has no native random-sample mode. This scans a bounded prefix of
matching results (paginated, up to --max-scan) and pseudo-randomly
subsamples via an accession hash (~ACCEPT_RATE), rather than drawing a
statistically uniform sample from the full ~34M-entry pool. This is
adequate for a bounded pilot testing whether unbiased sourcing surfaces
anything -- it is not a claim of unbiased population sampling.

CLI:
    python fetch_broad_bacterial_candidates.py \
        --manifest configs/manifest.csv --catalog full_structure_catalog.csv \
        --out-dir data/b1_broad_pilot --target-count 10000 --max-scan 300000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from utils import get_logger

log = get_logger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.pdb"

MIN_HIS, MIN_ASP, MIN_CYS = 3, 1, 1


def load_exclusion_set(manifest_path: Path, catalog_path: Path) -> set[str]:
    excluded = set()
    for row in csv.DictReader(open(manifest_path)):
        excluded.add(row["structure_id"])
    for row in csv.DictReader(open(catalog_path)):
        excluded.add(row["accession"])
    return excluded


def passes_composition_floor(sequence: str) -> bool:
    return (
        sequence.count("H") >= MIN_HIS
        and sequence.count("D") >= MIN_ASP
        and sequence.count("C") >= MIN_CYS
    )


def accept_by_hash(accession: str, accept_rate: float) -> bool:
    digest = hashlib.sha256(accession.encode()).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < accept_rate


def scan_uniprot(
    length_min: int, length_max: int, target_count: int, max_scan: int, accept_rate: float, excluded: set[str],
) -> list[dict]:
    query = f"reviewed:false AND taxonomy_id:2 AND length:[{length_min} TO {length_max}]"
    fields = "accession,sequence,organism_name,protein_name"
    candidates, cursor, n_scanned = [], None, 0
    while n_scanned < max_scan and len(candidates) < target_count:
        params = {"query": query, "format": "json", "size": 500, "fields": fields}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for entry in data["results"]:
            n_scanned += 1
            acc = entry["primaryAccession"]
            if acc in excluded:
                continue
            if not accept_by_hash(acc, accept_rate):
                continue
            seq = entry.get("sequence", {}).get("value", "")
            if not passes_composition_floor(seq):
                continue
            candidates.append({
                "accession": acc, "sequence": seq,
                "organism": entry.get("organism", {}).get("scientificName", ""),
                "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            })
            if len(candidates) >= target_count:
                break
        link = resp.headers.get("Link", "")
        cursor = None
        if 'rel="next"' in link:
            next_url = link.split(";")[0].strip("<> ")
            cursor = dict(p.split("=") for p in next_url.split("?", 1)[1].split("&")).get("cursor")
        if not cursor:
            break
        if n_scanned % 25000 < 500:
            log.info(f"  scanned ~{n_scanned}, accepted {len(candidates)} so far")
    log.info(f"Scanned {n_scanned} raw UniProt hits, accepted {len(candidates)} candidates")
    return candidates


def fetch_one_structure(accession: str, out_path: Path) -> dict | None:
    if out_path.exists():
        return {"structure_status": "cached", "structure_path": str(out_path)}
    try:
        resp = requests.get(ALPHAFOLD_API_URL.format(accession=accession), timeout=20)
        if resp.status_code != 200 or not resp.json():
            return None
        meta = resp.json()[0]
        version = meta["latestVersion"]
        pdb_url = ALPHAFOLD_PDB_URL.format(accession=accession, version=version)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(pdb_url, timeout=30)
        if r.status_code != 200:
            return None
        out_path.write_bytes(r.content)
        return {
            "structure_status": "fetched", "structure_path": str(out_path),
            "af_version": version, "global_plddt": meta.get("globalMetricValue"),
        }
    except Exception as e:
        log.debug(f"{accession}: AlphaFold fetch failed: {e}")
        return None


def fetch_structures_parallel(candidates: list[dict], structures_dir: Path, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one_structure, c["accession"], structures_dir / f"{c['accession']}.pdb"): c
            for c in candidates
        }
        n_done = 0
        for future in as_completed(futures):
            c = futures[future]
            result = future.result()
            if result is None:
                c["structure_status"] = "unavailable"
            else:
                c.update(result)
            n_done += 1
            if n_done % 500 == 0:
                log.info(f"  structures: {n_done}/{len(candidates)} processed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--length-min", type=int, default=180)
    p.add_argument("--length-max", type=int, default=400)
    p.add_argument("--target-count", type=int, default=10000)
    p.add_argument("--max-scan", type=int, default=300000)
    p.add_argument("--accept-rate", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusion_set(args.manifest, args.catalog)
    log.info(f"Loaded {len(excluded)} corpus accessions to exclude")

    candidates = scan_uniprot(
        args.length_min, args.length_max, args.target_count, args.max_scan, args.accept_rate, excluded,
    )

    log.info(f"Fetching AlphaFold structures for {len(candidates)} candidates ({args.workers} workers)...")
    structures_dir = args.out_dir / "structures"
    fetch_structures_parallel(candidates, structures_dir, args.workers)

    n_fetched = sum(1 for c in candidates if c.get("structure_status") in ("fetched", "cached"))
    manifest_out = args.out_dir / "candidate_manifest.json"
    manifest_out.write_text(json.dumps(candidates, indent=2))
    log.info(f"Done. {n_fetched}/{len(candidates)} structures available -> {manifest_out}")


if __name__ == "__main__":
    main()
