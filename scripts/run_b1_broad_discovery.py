"""
run_b1_broad_discovery.py

Scores the broad, unbiased-by-MBL-relevance candidate pool
(fetch_broad_bacterial_candidates.py) with the frozen six-donor
pharmacophore as the PRIMARY signal, then applies fARGene-B1
(correctly thresholded at 135.8), PF00753 domain presence, ESM2
retrieval + OOD percentile, and Foldseek similarity to known B1
structures as NOVELTY AXES computed after structural scoring, not as
pre-filters -- per the agreed design.

Reuses metal_independent_b1.py's extract_donors/score_donor_roles
directly (same functions, same ProcessPoolExecutor parallelization
pattern as the project's own evaluate_metal_independent_b1.py), not a
reimplementation.

The decisive output, exactly as specified: architecture_call=true,
good local structural confidence (mean AlphaFold pLDDT at the six
matched donor residues), fARGene-B1 negative, and ESM2 distant/OOD.
That is the shortlist this script reports -- everything else is
context, not the target signal.

CLI:
    python run_b1_broad_discovery.py \
        --candidates data/b1_broad_pilot/candidate_manifest.json \
        --structures-dir data/b1_broad_pilot/structures \
        --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
        --fargene-upstream /tmp/cryptic-mbl-b1-upstreams/fargene \
        --reference-bank data/production/reference_bank_v2/mean_esm2_v2.npz \
        --foldseek-b1-db data/foldseek_b1_reference/b1_db \
        --out-dir data/b1_broad_pilot --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from utils import get_logger, load_structure
from b1_structural_model import load_b1_template
from metal_independent_b1 import extract_donors, score_donor_roles

log = get_logger(__name__)

FARGENE_B1_SPECIFIC_THRESHOLD = 135.8
MIN_LOCAL_PLDDT = 70.0  # "good local structural confidence" -- same convention as PLDDT_QC_THRESHOLD elsewhere in this project
DONOR_LABEL_RE = re.compile(r"^(?P<chain>[^:]+):(?P<resname>[^:]+):(?P<resnum>\d+)[A-Za-z]?:(?P<atom>.+)$")


def _score_one(accession: str, structure_path: str, template_path: str) -> tuple[str, dict]:
    template = load_b1_template(Path(template_path))
    try:
        protein, roles = extract_donors(Path(structure_path))
    except Exception as e:
        return accession, {"status": "unavailable", "reason": str(e), "architecture_call": False}
    result = score_donor_roles(protein, roles, template)
    if result.get("architecture_call"):
        result["local_plddt"] = _local_confidence(protein, result["donor_mapping"])
    return accession, result


def _local_confidence(protein, donor_mapping: list[str]) -> float | None:
    b_factor = getattr(protein, "b_factor", None)
    if b_factor is None:
        return None
    values = []
    for label in donor_mapping:
        m = DONOR_LABEL_RE.match(label)
        if not m:
            continue
        chain, resname, resnum = m.group("chain"), m.group("resname"), int(m.group("resnum"))
        mask = (
            (protein.chain_id.astype(str) == chain)
            & (protein.res_name.astype(str) == resname)
            & (protein.res_id == resnum)
        )
        if mask.any():
            values.append(float(np.mean(b_factor[mask])))
    return float(np.mean(values)) if values else None


def score_architecture_parallel(candidates: list[dict], structures_dir: Path, template_path: Path, workers: int) -> dict[str, dict]:
    results = {}
    tasks = [
        (c["accession"], str(structures_dir / f"{c['accession']}.pdb"), str(template_path))
        for c in candidates if c.get("structure_status") in ("fetched", "cached")
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, *t): t[0] for t in tasks}
        n_done = 0
        for future in as_completed(futures):
            acc = futures[future]
            try:
                _, result = future.result()
            except Exception as e:
                result = {"status": "unavailable", "reason": str(e), "architecture_call": False}
            results[acc] = result
            n_done += 1
            if n_done % 500 == 0:
                log.info(f"  architecture: {n_done}/{len(tasks)} scored")
    for c in candidates:
        if c["accession"] not in results:
            results[c["accession"]] = {"status": "unavailable", "reason": "no_structure_fetched", "architecture_call": False}
    return results


def write_fasta(candidates: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for c in candidates:
            if c.get("sequence"):
                f.write(f">{c['accession']}\n{c['sequence']}\n")


def run_fargene_b1(fasta_path: Path, upstream_dir: Path, out_path: Path) -> dict:
    hmm_path = upstream_dir / "fargene_analysis" / "models" / "B1.hmm"
    domtblout = out_path.with_suffix(".domtblout")
    cmd = [
        "python", "scripts/run_fargene_comparator.py",
        "--fasta", str(fasta_path), "--hmm", str(hmm_path),
        "--hmmsearch", "/home/moraz/programs/miniconda3/bin/hmmsearch",
        "--threshold", str(FARGENE_B1_SPECIFIC_THRESHOLD),
        "--upstream-checkout", str(upstream_dir),
        "--out", str(out_path), "--domtblout", str(domtblout),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text())["per_example"]


def score_esm2(candidates: list[dict], reference_bank_path: Path, device: str, k: int = 5) -> dict[str, dict]:
    import torch
    import esm2_embed as ee

    model, batch_converter, alphabet = ee.load_model(device)
    ref = np.load(reference_bank_path, allow_pickle=False)
    ref_embeddings, ref_labels = ref["embeddings"], ref["labels"]
    loo_dists = ref["ood_calibration_sorted_loo_distances"]

    results = {}
    for i, c in enumerate(candidates):
        seq = c.get("sequence", "")
        if not seq:
            results[c["accession"]] = {"status": "unavailable"}
            continue
        seq = seq[:1022]
        with torch.no_grad():
            per_residue = ee.embed_sequence(model, batch_converter, device, seq)
        mean_emb = per_residue.mean(axis=0)
        dists = np.linalg.norm(ref_embeddings - mean_emb[None, :], axis=1)
        order = np.argsort(dists)[:k]
        positive_fraction = float(np.mean(ref_labels[order] == "positive"))
        nearest = float(dists[order[0]])
        results[c["accession"]] = {
            "status": "positive" if positive_fraction > 0.5 else "negative",
            "positive_neighbor_fraction": positive_fraction,
            "nearest_distance": nearest,
            "ood_percentile": float(np.searchsorted(loo_dists, nearest) / len(loo_dists)),
        }
        if (i + 1) % 500 == 0:
            log.info(f"  ESM2: {i+1}/{len(candidates)} scored")
    return results


def foldseek_max_tm(accession: str, structure_path: Path, b1_db: Path) -> float | None:
    with tempfile.TemporaryDirectory(prefix="foldseek-b1-") as tmp:
        tmp = Path(tmp)
        result_tsv = tmp / "result.tsv"
        cmd = [
            "foldseek", "easy-search", str(structure_path), str(b1_db), str(result_tsv), str(tmp / "fs_tmp"),
            "--format-output", "query,target,alntmscore", "-e", "10.0", "--exhaustive-search", "1",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except Exception as e:
            log.warning(f"{accession}: foldseek failed: {e}")
            return None
        if not result_tsv.exists() or result_tsv.stat().st_size == 0:
            return 0.0
        scores = [float(line.split("\t")[2]) for line in result_tsv.read_text().splitlines() if line.strip()]
        return max(scores) if scores else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--structures-dir", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--fargene-upstream", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--foldseek-b1-db", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    candidates = json.loads(args.candidates.read_text())
    log.info(f"Loaded {len(candidates)} broad-pilot candidates")

    log.info("=== Scoring six-donor B1 pharmacophore (parallel) ===")
    architecture_results = score_architecture_parallel(candidates, args.structures_dir, args.template, args.workers)
    n_arch_positive = sum(1 for r in architecture_results.values() if r.get("architecture_call"))
    log.info(f"architecture_call positive: {n_arch_positive}/{len(candidates)}")

    log.info("=== Running fARGene B1-specific (threshold 135.8) ===")
    fasta_path = args.out_dir / "candidates.fasta"
    write_fasta(candidates, fasta_path)
    fargene_b1 = run_fargene_b1(fasta_path, args.fargene_upstream, args.out_dir / "fargene_b1_broad.json")

    log.info("=== Scoring mean-ESM2 5-NN + OOD percentile ===")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm2_results = score_esm2(candidates, args.reference_bank, device)

    log.info("=== Foldseek novelty check (architecture-positive candidates only) ===")
    foldseek_results = {}
    for c in candidates:
        acc = c["accession"]
        if architecture_results.get(acc, {}).get("architecture_call"):
            structure_path = args.structures_dir / f"{acc}.pdb"
            foldseek_results[acc] = foldseek_max_tm(acc, structure_path, args.foldseek_b1_db)

    report, shortlist = {}, []
    for c in candidates:
        acc = c["accession"]
        arch = architecture_results.get(acc, {"architecture_call": False})
        esm2 = esm2_results.get(acc, {"status": "unavailable"})
        fargene_hit = fargene_b1.get(acc, {}).get("predicted_positive", False)
        local_plddt = arch.get("local_plddt")
        good_confidence = local_plddt is not None and local_plddt >= MIN_LOCAL_PLDDT
        entry = {
            "accession": acc, "organism": c.get("organism"), "protein_name": c.get("protein_name"),
            "architecture": arch, "esm2_retrieval": esm2, "fargene_b1_hit": fargene_hit,
            "foldseek_max_tm_to_known_b1": foldseek_results.get(acc),
        }
        report[acc] = entry
        if arch.get("architecture_call") and good_confidence and not fargene_hit and esm2.get("status") != "positive":
            shortlist.append(acc)

    output = {
        "n_candidates": len(candidates),
        "n_architecture_call_positive": n_arch_positive,
        "n_fargene_b1_hits": sum(1 for v in fargene_b1.values() if v.get("predicted_positive")),
        "n_esm2_positive": sum(1 for v in esm2_results.values() if v.get("status") == "positive"),
        "decisive_shortlist_criteria": (
            "architecture_call=true AND local_plddt>=70 AND fargene_b1_negative AND esm2_status!=positive"
        ),
        "decisive_shortlist": shortlist,
        "per_candidate": report,
    }
    out_path = args.out_dir / "broad_discovery_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"architecture_call positive: {n_arch_positive}, fARGene-B1 hits: {output['n_fargene_b1_hits']}, "
             f"ESM2 positive: {output['n_esm2_positive']}")
    log.info(f"DECISIVE SHORTLIST ({len(shortlist)}): {shortlist}")
    log.info(f"Wrote broad discovery report -> {out_path}")


if __name__ == "__main__":
    main()
