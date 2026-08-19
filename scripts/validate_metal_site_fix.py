"""
validate_metal_site_fix.py

Bounded validation gate, per the agreed plan: before regenerating any of
the other 1070 pockets, re-extract the 7 reference-bank proteins with the
fixed pocket_extraction.py (data/domain_pdbs/ input, real multi-site
Metal3D parsing, no cross-chain contamination) and check them against
their real crystallographic Zn positions (extracted directly from the
raw PDB HETATM ZN records, dominant chain only -- see REFERENCE_ZN_SITES,
values pulled straight from data/raw/{pdb_id}.pdb's own HETATM lines).

Reports exactly the requested gate metrics per reference:
  - number of raw Metal3D probe sites (before site selection)
  - nearest predicted-to-observed metal error, per accepted site (best pairing)
  - whether the correct mono-/dinuclear site count was recovered
  - canonical donor coverage at 2.8, 3.2, 3.5A (of the primary site)
  - pocket residue count, and chain purity of the INPUT array (should be
    exactly 1 chain now that extraction runs on data/domain_pdbs/, versus
    the old NDM-1 pocket's 8724 atoms across 293 "residues" -- ~30
    atoms/residue, a direct symptom of cross-chain merging; a real
    single-chain pocket should read ~5-15 atoms/residue)

Acceptance criterion (from the agreed plan): median localization error
< ~1A, no cross-chain contamination, >=90% of references with plausible
donor coverage by 3.5A.

CLI:
    python validate_metal_site_fix.py --domain-pdbs-dir data/domain_pdbs \
        --raw-dir data/raw --out reports/metal_site_fix_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import get_logger, load_structure
from pocket_extraction import extract_pocket, run_metal3d
from graph_construction import LIGAND_ATOMS

log = get_logger(__name__)

# (structure_id, pdb_id, subclass) -- from catalog_to_manifest.py's REFERENCE_BANK
REFERENCE_STRUCTURES = [
    ("NDM-1", "3SPU", "B1"),
    ("VIM-2", "1KO3", "B1"),
    ("IMP-1", "1DD6", "B1"),
    ("CphA", "1X8G", "B2"),
    ("Sfh-I", "2QDS", "B2"),
    ("L1", "1SML", "B3"),
    ("FEZ-1", "1K07", "B3"),
]

# Expected number of catalytic Zn sites per subclass (B1/B3 canonically
# dinuclear, B2 canonically mononuclear) -- used only to report whether the
# recovered site count matches expectation, NOT fed into site selection
# itself (select_metal_sites never sees subclass/label).
EXPECTED_N_SITES = {"B1": 2, "B2": 1, "B3": 2}

DONOR_COVERAGE_RADII = [2.8, 3.2, 3.5]


def parse_real_zn_sites(raw_pdb_path: Path, chain: str) -> list[np.ndarray]:
    sites = []
    for line in raw_pdb_path.read_text().splitlines():
        if line.startswith("HETATM") and line[12:14].strip() == "ZN" and line[21] == chain:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            sites.append(np.array([x, y, z]))
    return sites


def donor_coverage(pocket, primary_site: np.ndarray, radii: list[float]) -> dict[float, int]:
    coverage = {}
    for r in radii:
        n = 0
        for i in range(len(pocket.res_ids)):
            res_name, atom_name = pocket.res_names[i], pocket.atom_names[i]
            ligand_names = LIGAND_ATOMS.get(res_name)
            if ligand_names and atom_name in ligand_names:
                if float(np.linalg.norm(pocket.coords[i] - primary_site)) <= r:
                    n += 1
        coverage[r] = n
    return coverage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-pdbs-dir", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    report = {}
    localization_errors = []
    n_donor_coverage_ok_35 = 0

    for ref_id, pdb_id, subclass in REFERENCE_STRUCTURES:
        domain_pdb = args.domain_pdbs_dir / f"{ref_id}.pdb"
        raw_pdb = args.raw_dir / f"{pdb_id}.pdb"

        input_arr = load_structure(domain_pdb)
        n_chains = len(set(input_arr.chain_id.tolist()))

        raw_probes = run_metal3d(domain_pdb) or []

        pocket = extract_pocket(
            structure_path=domain_pdb, structure_id=ref_id, label="positive",
            confidence_tier=1, subclass=subclass, is_predicted=False,
        )

        real_sites = parse_real_zn_sites(raw_pdb, chain=str(input_arr.chain_id[0]))
        pred_sites = pocket.metal_coords if pocket.metal_coords is not None else np.zeros((0, 3))

        # best pairing: for each predicted site, nearest real site
        errors = []
        for ps in pred_sites:
            if len(real_sites) == 0:
                continue
            d = min(float(np.linalg.norm(ps - rs)) for rs in real_sites)
            errors.append(d)
        localization_errors.extend(errors)

        n_predicted = len(pred_sites)
        n_expected = EXPECTED_N_SITES.get(subclass)
        site_count_correct = (n_predicted == n_expected) if n_expected is not None else None

        coverage = donor_coverage(pocket, pred_sites[0], DONOR_COVERAGE_RADII) if len(pred_sites) else {}
        if coverage.get(3.5, 0) >= 3:  # a real coordination shell has >=3 donors typically
            n_donor_coverage_ok_35 += 1

        n_res = len(set(pocket.res_ids.tolist()))
        n_atoms = len(pocket.res_ids)

        report[ref_id] = {
            "pdb_id": pdb_id, "subclass": subclass,
            "n_input_chains": n_chains,
            "n_raw_metal3d_probes": len(raw_probes),
            "n_predicted_sites": n_predicted, "n_expected_sites": n_expected,
            "site_count_correct": site_count_correct,
            "predicted_site_probabilities": [round(float(p), 3) for p in pocket.metal_probabilities] if pocket.metal_probabilities is not None else [],
            "localization_errors_A": [round(e, 3) for e in errors],
            "donor_coverage_by_radius": {str(r): n for r, n in coverage.items()},
            "pocket_n_residues": n_res, "pocket_n_atoms": n_atoms,
            "pocket_atoms_per_residue": round(n_atoms / max(n_res, 1), 2),
        }
        log.info(
            f"{ref_id} ({subclass}): input_chains={n_chains} raw_probes={len(raw_probes)} "
            f"sites={n_predicted}/{n_expected} errors={[round(e,2) for e in errors]} "
            f"donor_cov@3.5={coverage.get(3.5)} pocket={n_res}res/{n_atoms}atoms "
            f"({n_atoms/max(n_res,1):.1f} atoms/res)"
        )

    errs = sorted(localization_errors)
    n = len(errs)
    summary = {
        "median_localization_error_A": errs[n // 2] if n else None,
        "max_localization_error_A": errs[-1] if n else None,
        "n_localization_measurements": n,
        "n_references_with_single_input_chain": sum(1 for r in report.values() if r["n_input_chains"] == 1),
        "n_references_correct_site_count": sum(1 for r in report.values() if r["site_count_correct"]),
        "n_references_donor_coverage_ok_at_3.5A": n_donor_coverage_ok_35,
        "n_references_total": len(REFERENCE_STRUCTURES),
        "acceptance_gate": {
            "median_error_below_1A": (errs[n // 2] < 1.0) if n else False,
            "no_cross_chain_contamination": all(r["n_input_chains"] == 1 for r in report.values()),
            "donor_coverage_ge_90pct_at_3.5A": (n_donor_coverage_ok_35 / len(REFERENCE_STRUCTURES)) >= 0.9,
        },
    }
    output = {"summary": summary, "per_reference": report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info(f"Wrote validation gate report -> {args.out}")
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
