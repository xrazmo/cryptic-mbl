"""
run_b1_pilot_discovery.py

Bounded pilot discovery run (see the approved plan): scores every
candidate from fetch_uniprot_mbl_fold_candidates.py's output with the
three existing, unmodified tools --

  - scripts/metal_independent_b1.py's six-donor pharmacophore
    (architecture_call is the primary channel)
  - fARGene (both the combined class_B_1_2.hmm and the B1-specific
    B1.hmm, against the real upstream clone in
    /tmp/cryptic-mbl-b1-upstreams/fargene)
  - mean-ESM2 5-NN against the frozen production reference bank

-- and flags the exact target the project docs define as missing
evidence: architecture_call=supported AND fARGene negative on BOTH
models. A hit in that set is a hypothesis for manual review, not a
discovery claim.

CLI:
    python run_b1_pilot_discovery.py \
        --candidates data/b1_pilot/candidate_manifest.json \
        --structures-dir data/b1_pilot/structures \
        --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
        --fargene-upstream /tmp/cryptic-mbl-b1-upstreams/fargene \
        --reference-bank data/production/reference_bank_v2/mean_esm2_v2.npz \
        --out-dir data/b1_pilot
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from utils import get_logger
from b1_structural_model import load_b1_template
from metal_independent_b1 import score_without_predicted_metals

log = get_logger(__name__)

# Per-model thresholds -- NOT interchangeable. Verified against the already-
# frozen reports/fargene_b1_b2_comparator.json ("class_b_1_2 model", 127.0)
# and reports/fargene_b1_specific_comparator.json ("B1 model", 135.8). An
# earlier version of this script applied 127.0 to both HMMs, which silently
# used the wrong (looser) cutoff for the B1-specific model.
FARGENE_COMBINED_THRESHOLD = 127.0
FARGENE_B1_SPECIFIC_THRESHOLD = 135.8


def score_architecture(candidates: list[dict], structures_dir: Path, template_path: Path) -> dict[str, dict]:
    template = load_b1_template(template_path)
    results = {}
    for i, c in enumerate(candidates):
        if c.get("structure_status") not in ("fetched", "cached"):
            results[c["accession"]] = {"status": "unavailable", "reason": "no_structure_fetched"}
            continue
        structure_path = structures_dir / f"{c['accession']}.pdb"
        try:
            results[c["accession"]] = score_without_predicted_metals(structure_path, template)
        except Exception as e:
            log.warning(f"{c['accession']}: architecture scoring failed: {e}")
            results[c["accession"]] = {"status": "unavailable", "reason": str(e)}
        if (i + 1) % 20 == 0:
            log.info(f"  architecture: {i+1}/{len(candidates)} scored")
    return results


def write_fasta(candidates: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for c in candidates:
            if c.get("sequence"):
                f.write(f">{c['accession']}\n{c['sequence']}\n")


def run_fargene(fasta_path: Path, hmm_path: Path, threshold: float, upstream_dir: Path, out_path: Path) -> dict:
    domtblout = out_path.with_suffix(".domtblout")
    cmd = [
        "python", "scripts/run_fargene_comparator.py",
        "--fasta", str(fasta_path), "--hmm", str(hmm_path),
        "--hmmsearch", "/home/moraz/programs/miniconda3/bin/hmmsearch",
        "--threshold", str(threshold),
        "--upstream-checkout", str(upstream_dir),
        "--out", str(out_path), "--domtblout", str(domtblout),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text())


def score_esm2(candidates: list[dict], reference_bank_path: Path, device: str, k: int = 5) -> dict[str, dict]:
    import torch
    import esm2_embed as ee

    model, batch_converter, alphabet = ee.load_model(device)
    ref = np.load(reference_bank_path, allow_pickle=False)
    ref_embeddings, ref_labels = ref["embeddings"], ref["labels"]

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
        results[c["accession"]] = {
            "status": "positive" if positive_fraction > 0.5 else "negative",
            "positive_neighbor_fraction": positive_fraction,
            "nearest_distance": float(dists[order[0]]),
        }
        if (i + 1) % 20 == 0:
            log.info(f"  ESM2: {i+1}/{len(candidates)} scored")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--structures-dir", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--fargene-upstream", required=True, type=Path)
    p.add_argument("--reference-bank", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    candidates = json.loads(args.candidates.read_text())
    log.info(f"Loaded {len(candidates)} pilot candidates")

    log.info("=== Scoring six-donor B1 pharmacophore ===")
    architecture_results = score_architecture(candidates, args.structures_dir, args.template)

    log.info("=== Running fARGene (combined B1/B2 and B1-specific) ===")
    fasta_path = args.out_dir / "candidates.fasta"
    write_fasta(candidates, fasta_path)
    combined_hmm = args.fargene_upstream / "fargene_analysis" / "models" / "class_B_1_2.hmm"
    b1_hmm = args.fargene_upstream / "fargene_analysis" / "models" / "B1.hmm"
    fargene_combined = run_fargene(fasta_path, combined_hmm, FARGENE_COMBINED_THRESHOLD, args.fargene_upstream, args.out_dir / "fargene_combined.json")
    fargene_b1 = run_fargene(fasta_path, b1_hmm, FARGENE_B1_SPECIFIC_THRESHOLD, args.fargene_upstream, args.out_dir / "fargene_b1_specific.json")

    log.info("=== Scoring mean-ESM2 5-NN ===")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm2_results = score_esm2(candidates, args.reference_bank, device)

    fargene_combined_hits = {
        sid for sid, row in fargene_combined["per_example"].items() if row["predicted_positive"]
    }
    fargene_b1_hits = {
        sid for sid, row in fargene_b1["per_example"].items() if row["predicted_positive"]
    }

    report = {}
    shortlist = []
    for c in candidates:
        acc = c["accession"]
        arch = architecture_results.get(acc, {"status": "unavailable"})
        esm2 = esm2_results.get(acc, {"status": "unavailable"})
        fargene_combined_hit = acc in fargene_combined_hits
        fargene_b1_hit = acc in fargene_b1_hits
        entry = {
            "accession": acc, "organism": c.get("organism"), "protein_name": c.get("protein_name"),
            "function_annotation": c.get("function_annotation"),
            "architecture": arch, "esm2_retrieval": esm2,
            "fargene_combined_hit": fargene_combined_hit, "fargene_b1_specific_hit": fargene_b1_hit,
        }
        report[acc] = entry
        if arch.get("architecture_call", False) and not fargene_combined_hit and not fargene_b1_hit:
            shortlist.append(acc)

    output = {
        "n_candidates": len(candidates),
        "n_architecture_call_positive": sum(1 for r in architecture_results.values() if r.get("architecture_call", False)),
        "n_full_pose_supported": sum(1 for r in architecture_results.values() if r.get("status") == "supported"),
        "n_fargene_combined_hits": len(fargene_combined_hits),
        "n_fargene_b1_specific_hits": len(fargene_b1_hits),
        "priority_shortlist_architecture_positive_fargene_negative": shortlist,
        "per_candidate": report,
    }
    out_path = args.out_dir / "pilot_discovery_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"Pilot: {output['n_architecture_call_positive']}/{len(candidates)} architecture_call-positive "
             f"({output['n_full_pose_supported']} also pass the secondary product-pose gate), "
             f"{len(fargene_combined_hits)} fARGene-combined hits, {len(fargene_b1_hits)} fARGene-B1 hits")
    log.info(f"Priority shortlist (architecture-positive, fARGene-negative both models): {len(shortlist)} -> {shortlist}")
    log.info(f"Wrote pilot discovery report -> {out_path}")


if __name__ == "__main__":
    main()
