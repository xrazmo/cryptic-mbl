"""Build provenance-tracked reaction-state templates from experimental PDBs.

This script is intentionally separate from candidate scoring.  It selects the
one or two crystallographic Zn ions nearest the named hydrolyzed beta-lactam,
extracts canonical protein donor atoms around those metals, and stores the
ligand pose and local pharmacophore in a compact NPZ file.

It never reads the labeled MBL corpus and therefore cannot tune a template to
the evaluation panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from catalytic_feasibility import DONOR_RADIUS_ANGSTROM, ReactionTemplate
from graph_construction import LIGAND_ATOMS
from utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PDBAtom:
    record: str
    atom_name: str
    resname: str
    chain: str
    resseq: int
    insertion_code: str
    coord: np.ndarray
    element: str
    occupancy: float


def parse_pdb(path: Path) -> list[PDBAtom]:
    atoms = []
    model_seen = False
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            if model_seen:
                break
            model_seen = True
            continue
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper()
        if not element:
            element = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper()
        occupancy_text = line[54:60].strip()
        atoms.append(PDBAtom(
            record=line[:6].strip(),
            atom_name=atom_name,
            resname=line[17:20].strip(),
            chain=line[21].strip(),
            resseq=int(line[22:26]),
            insertion_code=line[26].strip(),
            coord=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            element=element,
            occupancy=float(occupancy_text) if occupancy_text else 1.0,
        ))
    if not atoms:
        raise ValueError(f"{path}: no PDB atoms parsed")
    return atoms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_content_sha256(template: ReactionTemplate) -> str:
    """Stable hash of scientific content, independent of NPZ zip timestamps."""
    digest = hashlib.sha256()
    array_keys = {
        "metal_coords", "donor_coords", "donor_elements", "donor_site_indices",
        "donor_labels", "ligand_coords", "ligand_elements", "ligand_atom_names",
    }
    scalar_metadata = {
        key: value for key, value in template.__dict__.items() if key not in array_keys
    }
    digest.update(json.dumps(scalar_metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(array_keys):
        array = np.ascontiguousarray(getattr(template, key))
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _expected_n_sites(subclass: str) -> int:
    return 1 if subclass == "B2" else 2


def extract_template(pdb_path: Path, spec: dict) -> ReactionTemplate:
    atoms = parse_pdb(pdb_path)
    chain = spec["protein_chain"]
    ligand_resname = spec["ligand_resname"]

    ligand_groups: dict[tuple, list[PDBAtom]] = {}
    for atom in atoms:
        if atom.record == "HETATM" and atom.chain == chain and atom.resname == ligand_resname:
            ligand_groups.setdefault((atom.chain, atom.resseq, atom.insertion_code), []).append(atom)
    if not ligand_groups:
        raise ValueError(f"{pdb_path}: ligand {ligand_resname} chain {chain} not found")

    zinc = [
        atom for atom in atoms
        if atom.record == "HETATM" and atom.chain == chain and atom.element == "ZN"
    ]
    if not zinc:
        raise ValueError(f"{pdb_path}: no Zn atoms on chain {chain}")

    # Pick the ligand instance closest to any zinc.  This handles structures
    # with multiple crystallographic copies without mixing their coordinates.
    def ligand_distance(group: list[PDBAtom]) -> float:
        ligand_coords = np.array([atom.coord for atom in group])
        zinc_coords = np.array([atom.coord for atom in zinc])
        return float(np.linalg.norm(ligand_coords[:, None] - zinc_coords[None, :], axis=2).min())

    ligand = min(ligand_groups.values(), key=ligand_distance)
    ligand_coords = np.array([atom.coord for atom in ligand])
    n_sites = _expected_n_sites(spec["subclass"])
    zinc = sorted(
        zinc,
        key=lambda atom: float(np.linalg.norm(ligand_coords - atom.coord, axis=1).min()),
    )[:n_sites]
    metal_coords = np.array([atom.coord for atom in zinc])
    if len(metal_coords) != n_sites:
        raise ValueError(f"{pdb_path}: expected {n_sites} catalytic Zn, found {len(metal_coords)}")
    if float(np.linalg.norm(ligand_coords[:, None] - metal_coords[None, :], axis=2).min()) > 4.0:
        raise ValueError(f"{pdb_path}: named ligand is not positioned at selected metal site")

    donors = []
    for atom in atoms:
        if atom.record != "ATOM" or atom.chain != chain:
            continue
        allowed = LIGAND_ATOMS.get(atom.resname)
        if not allowed or atom.atom_name not in allowed:
            continue
        distances = np.linalg.norm(metal_coords - atom.coord, axis=1)
        site_index = int(np.argmin(distances))
        if float(distances[site_index]) <= DONOR_RADIUS_ANGSTROM:
            donors.append((atom, site_index))
    if len(donors) < 3:
        raise ValueError(f"{pdb_path}: only {len(donors)} canonical donors around catalytic zinc")
    if set(site for _atom, site in donors) != set(range(n_sites)):
        raise ValueError(f"{pdb_path}: at least one selected Zn has no canonical protein donor")

    heavy_ligand = [atom for atom in ligand if atom.element not in {"H", "D"}]
    if len(heavy_ligand) < 8:
        raise ValueError(f"{pdb_path}: named ligand has only {len(heavy_ligand)} heavy atoms")

    return ReactionTemplate(
        template_id=spec["template_id"],
        pdb_id=spec["pdb_id"],
        subclass=spec["subclass"],
        protein_chain=chain,
        ligand_resname=ligand_resname,
        substrate_class=spec["substrate_class"],
        reaction_state=spec["reaction_state"],
        source_url=spec["source_url"],
        citation_doi=spec["citation_doi"],
        resolution_angstrom=float(spec["resolution_angstrom"]),
        metal_coords=metal_coords,
        donor_coords=np.array([atom.coord for atom, _site in donors]),
        donor_elements=np.array([atom.element for atom, _site in donors]),
        donor_site_indices=np.array([site for _atom, site in donors], dtype=int),
        donor_labels=np.array([
            f"{atom.resname}:{atom.resseq}{atom.insertion_code}:{atom.atom_name}"
            for atom, _site in donors
        ]),
        ligand_coords=np.array([atom.coord for atom in heavy_ligand]),
        ligand_elements=np.array([atom.element for atom in heavy_ligand]),
        ligand_atom_names=np.array([atom.atom_name for atom in heavy_ligand]),
    )


def download_pdb(pdb_id: str, url_template: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = url_template.format(pdb_id=pdb_id.upper())
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if not payload.startswith(b"HEADER"):
        raise RuntimeError(f"{url}: response is not a legacy PDB file")
    destination.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/catalytic_reaction_templates.json"))
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in config["templates"]:
        pdb_path = args.pdb_dir / f"{spec['pdb_id'].upper()}.pdb"
        if not pdb_path.exists():
            if not args.download_missing:
                raise FileNotFoundError(
                    f"Missing {pdb_path}; supply it or rerun with --download-missing"
                )
            download_pdb(spec["pdb_id"], config["download_base_url"], pdb_path)
        template = extract_template(pdb_path, spec)
        out_path = args.out_dir / f"{template.template_id}.npz"
        template.save(out_path)
        records.append({
            "template_id": template.template_id,
            "pdb_id": template.pdb_id,
            "subclass": template.subclass,
            "pdb_sha256": sha256(pdb_path),
            "template_sha256": sha256(out_path),
            "template_content_sha256": template_content_sha256(template),
            "n_metal_sites": int(len(template.metal_coords)),
            "n_donors": int(len(template.donor_coords)),
            "donor_elements": template.donor_elements.tolist(),
            "n_ligand_heavy_atoms": int(len(template.ligand_coords)),
            "source_url": template.source_url,
            "citation_doi": template.citation_doi,
        })
        log.info("Built %s", out_path)

    report = {
        "schema_version": 1,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "templates": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Wrote template audit -> %s", args.report)


if __name__ == "__main__":
    main()
