"""
evaluate_outer_pocket_encoder.py

Structural V2, Part 2: the bounded, single go/no-go experiment for a B3
structural channel, per the agreed architecture --

    B3 structural channel:
        corrected 3D pocket -> frozen pretrained geometric encoder
                             -> small calibrated classifier/prototype score

Pools ESM-IF1's frozen per-residue structure embeddings
(data/outer_pocket_embeddings/*.npz, esm_if1_worker.py, geometry-only by
construction -- the encoder takes only backbone N/CA/C coordinates, no
amino-acid identity) over the residues within `radius` (default 16A, the
outer bound of "roughly 12-16A") of EITHER accepted metal site (from the
corrected data/pockets_v2 pockets), for two variants:
  - geometry-only: mean-pooled ESM-IF1 embedding alone.
  - geometry+chemistry: geometry embedding concatenated with mean-pooled
    AA physicochemical properties (graph_construction.AA_PROPERTIES) over
    the same residues -- lets the identity-ablation control ask "is
    chemistry/identity doing the work?" without silently smuggling
    sequence identity into the "geometry-only" arm.

A nearest-centroid prototype classifier is used (not a trained
classifier) -- the same choice made for the coordination fingerprint,
for the same reason: too few independent B3 positives (26-27) for a
learned classifier's threshold to be trustworthy. Evaluated on
B3_transfer's existing leave-cluster-out train/test split, matching
every other evaluation in this project.

Controls run here (see main() for what's NOT yet run -- coordinate-
scrambled needs a second ESM-IF1 batch pass and is deferred pending this
experiment's primary result, per "cheap checks first"):
  - outer pocket (radius) vs first-shell-only (DONOR_SHELL_RADIUS=2.8A)
    pooling, same embeddings -- does the wider context help at all?
  - geometry-only vs geometry+chemistry -- does adding identity change
    the result?
  - B3 vs EACH hard-negative family separately, not pooled.
  - unique recovery of ESM2-baseline misses (same framing as every
    other structural-signal check in this project).
  - lactonase/phosphodiesterase leave-one-negative-family-out (reuses
    challenge_splits.json's leave_one_negative_family_out configs).

Go/no-go criterion (from the agreed plan): keep this channel only if it
recovers B3 positives missed by ESM2 or suppresses hard-negative false
positives at matched sensitivity/specificity; otherwise use ESM2 alone
for B3 and stop.

CLI:
    python evaluate_outer_pocket_encoder.py \
        --embeddings-dir data/outer_pocket_embeddings --pockets-dir data/pockets_v2 \
        --domain-pdbs-dir data/domain_pdbs --challenge-splits data/challenge_splits.json \
        --mean-esm2-baseline data/mean_esm2_baseline.json --catalog full_structure_catalog.csv \
        --out data/outer_pocket_encoder_evaluation.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils import get_logger, PocketSubgraph, load_structure, residue_centroids
from graph_construction import AA_PROPERTIES, N_CHEM_PROPS

log = get_logger(__name__)

OUTER_POCKET_RADIUS = 16.0
FIRST_SHELL_RADIUS = 2.8


def load_embeddings(path: Path):
    d = np.load(path)
    return d["res_ids"], d["embeddings"]


def residues_within(centroids: np.ndarray, res_ids: np.ndarray, metal_coords: np.ndarray, radius: float) -> set[int]:
    dists = np.min(np.stack([np.linalg.norm(centroids - c[None, :], axis=1) for c in metal_coords]), axis=0)
    return set(res_ids[dists <= radius].tolist())


def chem_vector(res_names: list[str]) -> np.ndarray:
    vecs = [np.array(AA_PROPERTIES[rn], dtype=np.float32) if rn in AA_PROPERTIES else np.zeros(N_CHEM_PROPS, dtype=np.float32) for rn in res_names]
    return np.mean(vecs, axis=0) if vecs else np.zeros(N_CHEM_PROPS, dtype=np.float32)


def pool_structure(sid: str, embeddings_dir: Path, pockets_dir: Path, domain_pdbs_dir: Path) -> dict | None:
    emb_path = embeddings_dir / f"{sid}.npz"
    if not emb_path.exists():
        return None
    res_ids_emb, embeddings = load_embeddings(emb_path)
    row_of = {int(rid): i for i, rid in enumerate(res_ids_emb)}

    pocket = PocketSubgraph.load(pockets_dir / f"{sid}.npz")
    if pocket.metal_coords is None or len(pocket.metal_coords) == 0:
        return None

    domain_pdb = domain_pdbs_dir / f"{sid}.pdb"
    arr = load_structure(domain_pdb)
    res_ids_struct, centroids = residue_centroids(arr)

    outer_ids = residues_within(centroids, res_ids_struct, pocket.metal_coords, OUTER_POCKET_RADIUS)
    shell_ids = residues_within(centroids, res_ids_struct, pocket.metal_coords, FIRST_SHELL_RADIUS)

    def pooled(ids: set[int]) -> tuple[np.ndarray, np.ndarray] | None:
        rows = [row_of[rid] for rid in ids if rid in row_of]
        if not rows:
            return None
        geom = embeddings[rows].mean(axis=0)
        res_names = [arr.res_name[arr.res_id == rid][0] for rid in ids if rid in row_of]
        chem = chem_vector(res_names)
        return geom, chem

    outer = pooled(outer_ids)
    shell = pooled(shell_ids)
    if outer is None:
        return None
    out = {"outer_geom": outer[0], "outer_chem": outer[1]}
    if shell is not None:
        out["shell_geom"] = shell[0]
        out["shell_chem"] = shell[1]
    return out


def build_prototype(train_ids, pooled, labels, variant: str) -> dict[str, np.ndarray]:
    vecs = {"positive": [], "negative": []}
    for t in train_ids:
        if t not in pooled or variant not in pooled[t]:
            continue
        key = "positive" if labels[t] == "positive" else "negative"
        vecs[key].append(pooled[t][variant])
    return {k: np.mean(np.stack(v), axis=0) for k, v in vecs.items() if v}


def predict(test_ids, pooled, prototypes, variant: str) -> dict[str, str]:
    preds = {}
    for t in test_ids:
        if t not in pooled or variant not in pooled[t]:
            continue
        v = pooled[t][variant]
        dists = {k: float(np.linalg.norm(v - proto)) for k, proto in prototypes.items()}
        preds[t] = min(dists, key=dists.get)
    return preds


def score(preds, labels) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for tid, pred in preds.items():
        truth = "positive" if labels[tid] == "positive" else "negative"
        if truth == "positive" and pred == "positive":
            counts["tp"] += 1
        elif truth == "negative" and pred == "negative":
            counts["tn"] += 1
        elif truth == "negative":
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    n = sum(counts.values())
    sens = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    spec = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    return {"n_test": n, "confusion": counts, "sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--domain-pdbs-dir", required=True, type=Path)
    p.add_argument("--challenge-splits", required=True, type=Path)
    p.add_argument("--mean-esm2-baseline", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    challenge = json.loads(args.challenge_splits.read_text())
    baseline = json.loads(args.mean_esm2_baseline.read_text())
    catalog = {r["accession"]: r for r in csv.DictReader(open(args.catalog))}
    neg_family_of = {sid: row.get("neg_family", "") for sid, row in catalog.items()}

    labels = {}
    for f in args.pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        labels[pocket.metadata.source_structure_id] = "positive" if pocket.metadata.label == "positive" else "negative"

    configs = {"panel:B3_transfer": challenge["panels"]["B3_transfer"]}
    for fam in ["lactonase", "phosphodiesterase"]:
        key = f"lono:{fam}"
        if fam in challenge.get("leave_one_negative_family_out", {}):
            configs[key] = challenge["leave_one_negative_family_out"][fam]

    all_ids = sorted({t for cfg in configs.values() for t in cfg["train_ids"] + cfg["test_ids"]})
    log.info(f"Pooling outer-pocket embeddings for {len(all_ids)} structures...")
    pooled = {}
    for i, sid in enumerate(all_ids):
        r = pool_structure(sid, args.embeddings_dir, args.pockets_dir, args.domain_pdbs_dir)
        if r is not None:
            pooled[sid] = r
        if (i + 1) % 200 == 0:
            log.info(f"  {i+1}/{len(all_ids)} pooled")
    log.info(f"Pooled {len(pooled)}/{len(all_ids)} (rest missing embeddings or no predicted metal)")

    variants = {
        "outer_geometry_only": "outer_geom",
        "outer_geometry_plus_chemistry": "outer_chem",  # combined below
        "first_shell_geometry_only": "shell_geom",
    }

    report = {}
    for config_name, cfg in configs.items():
        report[config_name] = {}
        train_ids, test_ids = cfg["train_ids"], cfg["test_ids"]
        test_positive_ids = [t for t in test_ids if labels.get(t) == "positive"]

        for variant_name, key in variants.items():
            if key == "outer_chem":
                pooled_v = {sid: {"v": np.concatenate([r["outer_geom"], r["outer_chem"]])} for sid, r in pooled.items() if "outer_geom" in r}
                variant_key = "v"
            else:
                pooled_v = {sid: {"v": r[key]} for sid, r in pooled.items() if key in r}
                variant_key = "v"

            proto = build_prototype(train_ids, pooled_v, labels, variant_key)
            if len(proto) < 2:
                report[config_name][variant_name] = {"error": "insufficient train examples for prototype"}
                continue
            preds = predict(test_ids, pooled_v, proto, variant_key)
            result = score(preds, labels)

            if config_name == "panel:B3_transfer":
                esm2_preds = baseline[config_name]["per_example"]
                esm2_miss = [t for t in test_positive_ids if esm2_preds.get(t, {}).get("pred") != "positive"]
                recovers = [t for t in esm2_miss if preds.get(t) == "positive"]
                result["esm2_misses"] = len(esm2_miss)
                result["recovers_esm2_misses"] = len(recovers)
                result["recovered_ids"] = recovers

                by_family = {}
                for t in test_ids:
                    if labels.get(t) == "positive" or t not in preds:
                        continue
                    fam = neg_family_of.get(t, "") or "unknown"
                    by_family.setdefault(fam, {"n": 0, "fp": 0})
                    by_family[fam]["n"] += 1
                    if preds[t] == "positive":
                        by_family[fam]["fp"] += 1
                result["false_positive_rate_by_hard_negative_family"] = {
                    fam: v["fp"] / max(v["n"], 1) for fam, v in by_family.items()
                }

            report[config_name][variant_name] = result
            log.info(
                f"{config_name} / {variant_name}: n_test={result['n_test']} "
                f"sens={result['sensitivity']:.3f} spec={result['specificity']:.3f} "
                f"bal_acc={result['balanced_accuracy']:.3f}"
                + (f" recovers_esm2_misses={result.get('recovers_esm2_misses')}/{result.get('esm2_misses')}" if "esm2_misses" in result else "")
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    log.info(f"Wrote outer-pocket encoder evaluation -> {args.out}")


if __name__ == "__main__":
    main()
