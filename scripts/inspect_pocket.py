#!/usr/bin/env python
"""
inspect_pocket.py — Inspect and visualize a PocketSubgraph (.npz) file

Usage:
    python inspect_pocket.py data/pockets/NDM-1.npz
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add cryptic-mbl to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from utils import PocketSubgraph

def inspect_pocket(npz_path: Path):
    """Load and inspect a pocket NPZ file."""
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        print(f"✗ File not found: {npz_path}")
        return
    
    # Load the pocket
    pocket = PocketSubgraph.load(npz_path)
    
    print("=" * 70)
    print(f"POCKET: {pocket.metadata.source_structure_id}")
    print("=" * 70)
    
    # Metadata
    print("\n[METADATA]")
    print(f"  Source structure: {pocket.metadata.source_structure_id}")
    print(f"  Label: {pocket.metadata.label}")
    print(f"  Confidence tier: {pocket.metadata.confidence_tier}")
    print(f"  Pocket source: {pocket.metadata.pocket_source}")
    print(f"  Subclass: {pocket.metadata.subclass}")
    print(f"  Metal confidence: {pocket.metadata.metal_confidence}")
    print(f"  Mean pocket pLDDT: {pocket.metadata.mean_pocket_plddt}")
    
    # Geometry
    print("\n[GEOMETRY]")
    print(f"  N atoms: {len(pocket.coords)}")
    print(f"  N residues: {len(np.unique(pocket.res_ids))}")
    print(f"  Coordinate range (Å):")
    coords_min = pocket.coords.min(axis=0)
    coords_max = pocket.coords.max(axis=0)
    print(f"    X: [{coords_min[0]:.2f}, {coords_max[0]:.2f}]")
    print(f"    Y: [{coords_min[1]:.2f}, {coords_max[1]:.2f}]")
    print(f"    Z: [{coords_min[2]:.2f}, {coords_max[2]:.2f}]")
    
    # Metal site
    print("\n[METAL SITE]")
    if pocket.metal_coord is not None:
        print(f"  ✓ Metal site predicted:")
        print(f"    Coordinates: [{pocket.metal_coord[0]:.2f}, {pocket.metal_coord[1]:.2f}, {pocket.metal_coord[2]:.2f}]")
        
        # Distance from metal to all atoms
        dists = np.linalg.norm(pocket.coords - pocket.metal_coord[None, :], axis=1)
        coordinating_mask = (dists < 2.5) & np.isin(pocket.elements, ["N", "O", "S"])
        n_coord = coordinating_mask.sum()
        print(f"    Potential coordinating atoms (N/O/S, <2.5Å): {n_coord}")
        if n_coord > 0:
            coord_dists = dists[coordinating_mask]
            print(f"    Coordination distances: {coord_dists.min():.2f} - {coord_dists.max():.2f} Å")
    else:
        print(f"  ✗ No metal site (cavity fallback used)")
    
    # Residue composition
    print("\n[RESIDUE COMPOSITION]")
    unique_res, counts = np.unique(pocket.res_names, return_counts=True)
    for res, count in sorted(zip(unique_res, counts), key=lambda x: -x[1]):
        print(f"  {res}: {count}")
    
    # Element composition
    print("\n[ELEMENT COMPOSITION]")
    unique_elem, counts = np.unique(pocket.elements, return_counts=True)
    for elem, count in sorted(zip(unique_elem, counts), key=lambda x: -x[1]):
        print(f"  {elem}: {count}")
    
    print("\n" + "=" * 70)
    return pocket


def validate_pocket(pocket: PocketSubgraph) -> dict:
    """Validate pocket quality and return QC metrics."""
    qc = {
        "has_metal_site": pocket.metal_coord is not None,
        "n_atoms": len(pocket.coords),
        "n_residues": len(np.unique(pocket.res_ids)),
        "has_hydrophobic": bool(np.isin(pocket.res_names, ["ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET"]).any()),
        "has_polar": bool(np.isin(pocket.res_names, ["SER", "THR", "TYR", "ASN", "GLN"]).any()),
        "has_charged": bool(np.isin(pocket.res_names, ["ASP", "GLU", "LYS", "ARG", "HIS"]).any()),
    }
    
    if pocket.metal_coord is not None:
        dists = np.linalg.norm(pocket.coords - pocket.metal_coord[None, :], axis=1)
        coordinating = (dists < 2.5) & np.isin(pocket.elements, ["N", "O", "S"])
        qc["n_coordinating_atoms"] = int(coordinating.sum())
        qc["coordination_valid"] = 3 <= qc["n_coordinating_atoms"] <= 6  # MBL sites are typically 3-5 coord
    
    if pocket.metadata.mean_pocket_plddt is not None:
        qc["plddt_ok"] = pocket.metadata.mean_pocket_plddt > 70
    
    return qc


def print_validation(pocket: PocketSubgraph):
    """Print QC validation results."""
    qc = validate_pocket(pocket)
    
    print("\n[QUALITY CONTROL]")
    checks = [
        ("Metal site detected", qc["has_metal_site"]),
        ("Enough atoms (>50)", qc["n_atoms"] > 50),
        ("Enough residues (>10)", qc["n_residues"] > 10),
        ("Has hydrophobic residues", qc["has_hydrophobic"]),
        ("Has polar residues", qc["has_polar"]),
        ("Has charged residues", qc["has_charged"]),
    ]
    
    if qc["has_metal_site"]:
        checks.extend([
            (f"Valid coordination number ({qc['n_coordinating_atoms']} atoms)", qc.get("coordination_valid", False)),
        ])
    
    if qc.get("plddt_ok") is not None:
        checks.append(("pLDDT > 70", qc["plddt_ok"]))
    
    all_pass = True
    for check_name, passed in checks:
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {check_name}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("  ✓ All QC checks passed!")
    else:
        print("  ⚠ Some QC checks failed — review above")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pocket.npz>")
        sys.exit(1)
    
    pocket = inspect_pocket(Path(sys.argv[1]))
    if pocket:
        print_validation(pocket)
