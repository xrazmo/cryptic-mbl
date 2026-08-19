"""
utils.py — shared helpers used across the MBL pocket-discovery pipeline.

Covers: reproducible seeding, logging setup, structure loading (PDB/mmCIF/AF
output) via biotite, and small geometry helpers reused by pocket extraction,
graph construction, and inference.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import biotite.structure as struc
    import biotite.structure.io as strucio
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "biotite is required (`pip install biotite`) for structure I/O."
    ) from e


# --------------------------------------------------------------------------- #
# Logging / reproducibility
# --------------------------------------------------------------------------- #

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Confidence tiers (carried through the whole pipeline; see spec §2)
# --------------------------------------------------------------------------- #

CONFIDENCE_TIERS = {
    1: "crystal + published kinetics",
    2: "crystal, qualitative activity only",
    3: "AF-predicted + in vivo resistance phenotype",
    4: "augmented / synthetic pose",
}

# Loss-weight multiplier applied per tier during training (spec §8).
TIER_LOSS_WEIGHTS = {1: 1.0, 2: 0.85, 3: 0.6, 4: 0.35}


# --------------------------------------------------------------------------- #
# Structure loading
# --------------------------------------------------------------------------- #

def load_structure(path: str | Path) -> struc.AtomArray:
    """
    Load a PDB / mmCIF / AlphaFold-output structure into a biotite AtomArray.
    Returns the first model only (AtomArray, not AtomArrayStack).

    Requests the b_factor extra field explicitly — biotite does not load it
    by default, and AF-predicted structures store per-residue pLDDT there
    (see get_per_residue_confidence). Without it, arr.b_factor doesn't exist
    at all and pLDDT-based QC silently never fires.
    """
    path = Path(path)
    try:
        arr = strucio.load_structure(str(path), extra_fields=["b_factor"])
    except TypeError:
        # some formats and biotite versions do not accept extra_fields
        arr = strucio.load_structure(str(path))
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


def load_probe_structure(path: str | Path) -> struc.AtomArray:
    """
    Like load_structure, but also requests the occupancy extra field --
    biotite does not load it by default either (same reason as b_factor
    above). Metal3D's find_unique_sites() writes each candidate site's max
    cluster probability into the occupancy column of its probe PDB (see
    assets/metal-site-prediction/Metal3D/utils/helpers.py); without this,
    arr.occupancy doesn't exist and probe probabilities are unreadable.
    Only pocket_extraction.py's probe-parsing needs this -- kept separate
    from load_structure rather than adding occupancy there, to not change
    behavior for the 1077 real structure loads that don't need it.
    """
    path = Path(path)
    try:
        arr = strucio.load_structure(str(path), extra_fields=["occupancy"])
    except TypeError:
        arr = strucio.load_structure(str(path))
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


def get_per_residue_confidence(arr: struc.AtomArray) -> np.ndarray:
    """
    Extract per-atom confidence (pLDDT for AF/ESMFold outputs, stored in the
    B-factor column by convention; crystallographic B-factors are NOT
    confidence and should not be passed through this path — check structure
    provenance upstream and set is_predicted=False for experimental
    structures to skip pLDDT-based filtering).
    """
    if hasattr(arr, "b_factor"):
        return arr.b_factor
    return np.full(arr.array_length(), np.nan)


def load_esm2_embedding(esm2_dir: Optional[Path], structure_id: str, n_residues: int) -> Optional[np.ndarray]:
    """
    Loads a precomputed esm2_embed.py .npy for this structure, falling back
    to None (-> zeros in graph_construction.build_node_features) if missing
    or if its residue count doesn't match this pocket (e.g. pocket_extraction
    was re-run with different radii after embeddings were computed).
    Shared by train.py and evaluate.py so both stay consistent.
    """
    if esm2_dir is None:
        return None
    path = Path(esm2_dir) / f"{structure_id}.npy"
    if not path.exists():
        return None
    emb = np.load(path)
    if emb.shape[0] != n_residues:
        get_logger(__name__).warning(
            f"{structure_id}: esm2 embedding has {emb.shape[0]} residues, pocket has "
            f"{n_residues} -- ignoring (falling back to zeros)."
        )
        return None
    return emb


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def residue_centroids(arr: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (res_ids, centroids) — one 3D centroid per residue (CA-based if
    present, else mean of all atoms in the residue).

    Groups by bare res_id, NOT (chain_id, res_id, ins_code) -- deliberately
    relies on the caller (pocket_extraction.extract_pocket) passing an
    already single-chain AtomArray, which eliminates the catastrophic case
    (residue 120 in chain A silently merged with residue 120 in chain B --
    this collapsed NDM-1's true ~230-residue single-chain pocket into an
    8724-atom, 5-chain contaminated one before this fix). Insertion codes
    remain a theoretical residual risk within one chain; checked empirically
    across 200 single-chain exports and found zero real collisions, so this
    asserts loudly instead of silently merging rather than building the
    full composite-key schema change that would require touching
    PocketSubgraph's storage format and every downstream consumer.
    """
    if arr.ins_code is not None and np.any(arr.ins_code != ""):
        counts: dict[int, set] = {}
        for rid, ic in zip(arr.res_id, arr.ins_code):
            counts.setdefault(int(rid), set()).add(ic)
        colliding = {rid: codes for rid, codes in counts.items() if len(codes) > 1}
        assert not colliding, (
            f"residue_centroids: res_id(s) with multiple insertion codes would be silently "
            f"merged: {colliding} -- residue keying here is bare res_id only, by design; "
            f"see this function's docstring."
        )
    res_ids = np.unique(arr.res_id)
    centroids = np.zeros((len(res_ids), 3))
    for i, rid in enumerate(res_ids):
        res_mask = arr.res_id == rid
        ca_mask = res_mask & (arr.atom_name == "CA")
        if ca_mask.sum() == 1:
            centroids[i] = arr.coord[ca_mask][0]
        else:
            centroids[i] = arr.coord[res_mask].mean(axis=0)
    return res_ids, centroids


def min_distance_to_point(coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords - point[None, :], axis=1)


# --------------------------------------------------------------------------- #
# Serialization dataclasses shared by pocket_extraction / graph_construction
# --------------------------------------------------------------------------- #

@dataclass
class PocketMetadata:
    source_structure_id: str
    label: str                    # "positive" | "hard_negative" | "easy_negative" | "unlabeled"
    confidence_tier: int          # 1-4, see CONFIDENCE_TIERS
    pocket_source: str            # "metal3d" | "cavity_fallback"
    metal_confidence: Optional[float] = None
    mean_pocket_plddt: Optional[float] = None
    subclass: Optional[str] = None  # "B1" | "B2" | "B3" | "environmental" | None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class PocketSubgraph:
    """Container produced by pocket_extraction.py, consumed by graph_construction.py."""
    res_ids: np.ndarray
    res_names: np.ndarray          # 3-letter residue names, one per node
    coords: np.ndarray             # (N_atoms, 3) or (N_res, 3) depending on granularity
    atom_names: np.ndarray
    elements: np.ndarray
    is_sidechain: np.ndarray       # bool
    metal_coords: Optional[np.ndarray]        # (n_sites, 3), n_sites in {0,1,2}; None/(0,3) for cavity_fallback
    metal_probabilities: Optional[np.ndarray]  # (n_sites,), Metal3D cluster max-probability per site
    metadata: PocketMetadata

    @property
    def metal_coord(self) -> Optional[np.ndarray]:
        """Primary (highest-probability) site, (3,) or None -- for the many
        existing consumers (dist_to_metal, radial shell, etc.) that only
        need one point. Prefer metal_coords directly for anything that
        should consider a possible second (dinuclear) site."""
        if self.metal_coords is None or len(self.metal_coords) == 0:
            return None
        return self.metal_coords[0]

    def save(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        empty = np.zeros((0, 3))
        np.savez_compressed(
            out_path,
            res_ids=self.res_ids,
            res_names=self.res_names,
            coords=self.coords,
            atom_names=self.atom_names,
            elements=self.elements,
            is_sidechain=self.is_sidechain,
            metal_coords=self.metal_coords if self.metal_coords is not None else empty,
            metal_probabilities=self.metal_probabilities if self.metal_probabilities is not None else np.zeros((0,)),
            metadata_json=self.metadata.to_json(),
        )

    @staticmethod
    def load(path: str | Path) -> "PocketSubgraph":
        d = np.load(path, allow_pickle=False)
        meta_dict = json.loads(str(d["metadata_json"]))
        metadata = PocketMetadata(**meta_dict)
        if "metal_coords" in d:
            metal_coords = d["metal_coords"]
            metal_coords = None if metal_coords.size == 0 else metal_coords
            metal_probabilities = d["metal_probabilities"]
            metal_probabilities = None if metal_probabilities.size == 0 else metal_probabilities
        else:
            # Backward compat: pockets saved before this fix (see
            # pocket_extraction.py's coordination-fingerprint fix commit)
            # stored a single averaged, potentially-corrupted metal_coord.
            # Still loadable during the staged rollout -- only regenerated
            # pockets get real metal_coords/metal_probabilities.
            legacy = d["metal_coord"]
            metal_coords = None if legacy.size == 0 else legacy.reshape(1, 3)
            metal_probabilities = None if legacy.size == 0 else np.array([metadata.metal_confidence or 0.7])
        return PocketSubgraph(
            res_ids=d["res_ids"],
            res_names=d["res_names"],
            coords=d["coords"],
            atom_names=d["atom_names"],
            elements=d["elements"],
            is_sidechain=d["is_sidechain"],
            metal_coords=metal_coords,
            metal_probabilities=metal_probabilities,
            metadata=metadata,
        )
