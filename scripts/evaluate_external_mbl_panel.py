"""Evaluate the metal-independent B1 detector on a declared PDB panel.

The panel is literature-selected before scoring. For each PDB, the largest
standard-amino-acid chain is chosen deterministically (ties: chain ID), written
as a single-chain structure, and scored without predicted metal coordinates.
The same chain sequence is exported for fARGene/PLM-ARG comparators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from b1_structural_model import load_b1_template
from sequence_constants import THREE_TO_ONE
from metal_independent_b1 import MODIFIED_CYSTEINE_RESNAMES, score_without_predicted_metals
from utils import get_logger, load_structure

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if not payload.startswith(b"HEADER"):
        raise RuntimeError(f"{url}: response is not a legacy PDB file")
    path.write_bytes(payload)


def select_chain(raw_path: Path):
    arr = load_structure(raw_path)
    candidates = []
    for chain_id in sorted(set(arr.chain_id.tolist())):
        chain = arr[arr.chain_id == chain_id]
        protein_residue = np.isin(
            chain.res_name,
            list(THREE_TO_ONE) + list(MODIFIED_CYSTEINE_RESNAMES),
        )
        chain = chain[protein_residue]
        n_ca = int(np.sum(chain.atom_name == "CA"))
        if n_ca:
            candidates.append((n_ca, str(chain_id), chain))
    if not candidates:
        raise ValueError(f"{raw_path}: no standard-amino-acid chain")
    _n_ca, chain_id, chain = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return chain_id, chain


def chain_sequence(chain) -> str:
    sequence = []
    seen = set()
    for res_id, ins_code, resname in zip(chain.res_id, chain.ins_code, chain.res_name):
        key = (int(res_id), str(ins_code))
        if key in seen:
            continue
        seen.add(key)
        sequence.append(
            "C" if str(resname) in MODIFIED_CYSTEINE_RESNAMES
            else THREE_TO_ONE[str(resname)]
        )
    return "".join(sequence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--chains-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--fasta-out", required=True, type=Path)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--fargene-results", type=Path)
    parser.add_argument("--fargene-b1-results", type=Path)
    parser.add_argument("--plm-arg-results", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import biotite.structure.io.pdb as pdb

    config = json.loads(args.config.read_text())
    template = load_b1_template(args.template)
    args.chains_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    fasta = []
    for spec in config["entries"]:
        pdb_id = spec["pdb_id"].upper()
        raw_path = args.raw_dir / f"{pdb_id}.pdb"
        if not raw_path.exists():
            if not args.download_missing:
                raise FileNotFoundError(raw_path)
            download(config["download_url"].format(pdb_id=pdb_id), raw_path)
        chain_id, chain = select_chain(raw_path)
        chain_path = args.chains_dir / f"{pdb_id}.pdb"
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(chain)
        pdb_file.write(str(chain_path))
        sequence = chain_sequence(chain)
        fasta.append(f">{pdb_id}\n{sequence}\n")
        records[pdb_id] = {
            **spec,
            "selected_chain": chain_id,
            "sequence_length": len(sequence),
            "raw_pdb_sha256": sha256(raw_path),
            "chain_pdb_sha256": sha256(chain_path),
            "structural_result": score_without_predicted_metals(chain_path, template),
        }
        log.info("%s %s: %s", pdb_id, spec["name"], records[pdb_id]["structural_result"]["status"])

    args.fasta_out.parent.mkdir(parents=True, exist_ok=True)
    args.fasta_out.write_text("".join(fasta))
    full_calls = {sid: row["structural_result"]["positive_call"] for sid, row in records.items()}
    architecture_calls = {
        sid: bool(row["structural_result"]["architecture_call"])
        for sid, row in records.items()
    }
    native_thiolate_calls = {
        sid: bool(row["structural_result"].get("native_thiolate_positive_call", False))
        for sid, row in records.items()
    }
    native_thiolate_architecture_calls = {
        sid: bool(row["structural_result"].get("native_thiolate_architecture_call", False))
        for sid, row in records.items()
    }
    canonical = [sid for sid, row in records.items() if row["group"] == "canonical_B1"]
    controls = [sid for sid, row in records.items() if row["group"] in {"B2_control", "B3_control"}]

    def summary(ids: list[str], expected_positive: bool, method_calls: dict[str, bool]) -> dict:
        correct = sum(method_calls[sid] == expected_positive for sid in ids)
        return {"n": len(ids), "correct": correct, "rate": correct / len(ids),
                "positive_calls": sum(method_calls[sid] for sid in ids)}

    comparator_paths = {}
    comparator_calls = {}
    for method, path, key in (
        ("fargene_B1_B2_HMM", args.fargene_results, "predicted_positive"),
        ("fargene_B1_specific_HMM", args.fargene_b1_results, "predicted_positive"),
        ("PLM_ARG_beta_lactam", args.plm_arg_results, "predicted_beta_lactam"),
    ):
        if path is None:
            continue
        payload = json.loads(path.read_text())
        if payload.get("fasta_sha256") != sha256(args.fasta_out):
            raise ValueError(
                f"{path}: comparator FASTA hash does not match the newly exported panel"
            )
        comparator_paths[method] = {"path": str(path), "sha256": sha256(path)}
        comparator_calls[method] = {}
        for sid in records:
            records[sid][method] = bool(payload["per_example"][sid][key])
            comparator_calls[method][sid] = records[sid][method]

    all_method_calls = {
        "six_donor_architecture": architecture_calls,
        "six_donor_architecture_native_thiolate_only": native_thiolate_architecture_calls,
        "full_architecture_and_product_pose": full_calls,
        "full_architecture_and_product_pose_native_thiolate_only": native_thiolate_calls,
        **comparator_calls,
    }
    method_summaries = {
        method: {
            "canonical_B1_sensitivity": summary(canonical, True, method_calls),
            "B2_B3_control_specificity": summary(controls, False, method_calls),
        }
        for method, method_calls in all_method_calls.items()
    }

    output = {
        "schema_version": 1,
        "panel_selection_policy": config["selection_policy"],
        "primary_source": config["primary_source"],
        "secondary_source": config["secondary_source"],
        "config_sha256": sha256(args.config),
        "template_sha256": sha256(args.template),
        "fasta_sha256": sha256(args.fasta_out),
        "method_summaries": method_summaries,
        "comparator_sources": comparator_paths,
        "per_example": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    log.info("Wrote external experimental panel -> %s", args.out)


if __name__ == "__main__":
    main()
