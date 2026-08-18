"""
component_challenge_split.py

Replaces clustering_split.py's stratified-k-fold design. That design
doesn't fit this dataset: honest sequence-remote grouping (see
build_split_graph.py / feasibility_report.py) leaves only 6 positive
components (116, 26, 1, 1, 1, 1 members), 5 of which overlap the
reference-bank's two largest components. A conventional 5-fold CV over
that would either put a tiny, uninterpretable slice of positives in each
test fold, or silently leak reference-adjacent homologs across folds.
Component-level challenge panels are the honest alternative: each panel is
defined by WHICH positive component it holds out, not by a fold index.

Uses the identity>=0.3, coverage>=0.8 sequence_components regime as the
primary remote-family split (matches build_split_graph.py's default
--seq-identity-threshold; NOT chosen because it produces a convenient
number of folds -- see feasibility_report.py's sensitivity sweep, which
shows 30%/40% behave similarly and 50% qualitatively differs by no longer
chaining positives together).

Three panels, all negative-only sequence components distributed across
them (stratified by neg_family, then by size, greedily balanced) so every
negative component is tested in exactly one panel:

  - B1_B2_transfer:  test positives = the 116-member component (B1-major, all B2)
  - B3_transfer:     test positives = the 26-member component (B3)
  - remote_outlier:  test positives = the 4 singleton positive components,
                      reported both individually and jointly (see the
                      "singletons" list in the output -- report per-id,
                      not just as one pooled number)

Plus, NOT part of the 3-panel CV:

  - leave_one_negative_family_out: for each of {rnase_z, glyoxalase_ii,
    lactonase, phosphodiesterase}, test = every negative-only component
    whose dominant neg_family matches; train = everything else.

  - operational_reference_anchored_retrieval: same test set as
    remote_outlier (the 4 singleton positives -- the only positives NOT in
    a reference-bank-containing component), but train explicitly keeps the
    reference bank + both large positive components as retrieval anchors.
    This is the SAME split as remote_outlier's train/test by construction
    (nothing else made sense as the analogous train set); the only
    difference is interpretive, and the output labels it as such: report
    these 4 as individual discovery case studies, NOT as an aggregate
    sensitivity estimate or population-level generalization claim. The
    reference bank's role here is "known anchor for retrieval", not
    "independent validation set" -- see feasibility_report.py, which
    confirmed the two largest positive components already contain 5/7 and
    2/7 of the reference bank, so a reference-vs-hard-negative ranking
    metric on this data is measuring near-identical-homolog retrieval, not
    external generalization. evaluate.py's reference_bank_leave_one_out_ranks
    should be read with that caveat (see its docstring) rather than removed
    outright, since it's still a legitimate "does retrieval work at all"
    sanity check.

CLI:
    python component_challenge_split.py \
        --split-graph data/split_graph.json --pockets-dir data/pockets \
        --catalog full_structure_catalog.csv --out data/challenge_splits.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from utils import get_logger, PocketSubgraph

log = get_logger(__name__)

REFERENCE_BANK_IDS = {"NDM-1", "VIM-2", "IMP-1", "CphA", "Sfh-I", "L1", "FEZ-1"}
NEGATIVE_FAMILIES_FOR_LONO = ["rnase_z", "glyoxalase_ii", "lactonase", "phosphodiesterase"]


def load_metadata(pockets_dir: Path, catalog_path: Path) -> dict[str, dict]:
    meta = {}
    for f in pockets_dir.glob("*.npz"):
        pocket = PocketSubgraph.load(f)
        meta[pocket.metadata.source_structure_id] = {
            "label": pocket.metadata.label,
            "subclass": pocket.metadata.subclass,
        }
    catalog = {r["accession"]: r for r in csv.DictReader(open(catalog_path))}
    for sid, row in catalog.items():
        if sid in meta:
            meta[sid]["neg_family"] = row.get("neg_family", "")
    return meta


def dominant_neg_family(members: list[str], meta: dict[str, dict]) -> str:
    fams = Counter(meta.get(s, {}).get("neg_family", "") for s in members)
    fams.pop("", None)
    return fams.most_common(1)[0][0] if fams else "unknown"


def distribute_negative_components(
    neg_components: dict[str, list[str]], meta: dict[str, dict], panel_names: list[str],
) -> dict[str, list[str]]:
    """
    Greedy family- and size-balanced round-robin: within each neg_family,
    largest components first, always to whichever panel currently has the
    fewest members from that family, breaking ties by OVERALL panel size
    so far. Every negative component ends up assigned to exactly one panel.

    The overall-size tiebreak matters because several neg_families are
    themselves ONE single connected component (e.g. glyoxalase_ii, 286/286
    members, cannot be split across panels without breaking component
    integrity) -- with only a per-family tiebreak, every "brand new" family
    ties at zero and the tie always resolved to the same first panel,
    stacking multiple monolithic families onto it. Processing families
    largest-first (not dict order) plus the overall-size tiebreak spreads
    that unavoidable lumpiness across panels instead of concentrating it.
    """
    by_family: dict[str, list[tuple[str, list[str]]]] = {}
    for comp_id, members in neg_components.items():
        fam = dominant_neg_family(members, meta)
        by_family.setdefault(fam, []).append((comp_id, members))
    families_by_total_size = sorted(by_family, key=lambda f: -sum(len(m) for _c, m in by_family[f]))

    assignment: dict[str, str] = {}  # comp_id -> panel_name
    panel_family_load = {p: Counter() for p in panel_names}
    panel_total_load = Counter()
    for fam in families_by_total_size:
        comps = sorted(by_family[fam], key=lambda cm: -len(cm[1]))
        for comp_id, members in comps:
            target = min(panel_names, key=lambda p: (panel_family_load[p][fam], panel_total_load[p]))
            assignment[comp_id] = target
            panel_family_load[target][fam] += len(members)
            panel_total_load[target] += len(members)
    return assignment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-graph", required=True, type=Path)
    p.add_argument("--pockets-dir", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = json.loads(args.split_graph.read_text())
    comps: dict[str, list[str]] = d["sequence_components"]
    meta = load_metadata(args.pockets_dir, args.catalog)

    pos_components = {cid: m for cid, m in comps.items() if any(meta.get(s, {}).get("label") == "positive" for s in m)}
    neg_components = {cid: m for cid, m in comps.items() if cid not in pos_components}

    sizes = sorted(pos_components.items(), key=lambda cm: -len(cm[1]))
    assert len(sizes) == 6, f"expected 6 positive components, found {len(sizes)} -- data has changed, review before trusting panel assignment"
    b1b2_id, b1b2_members = sizes[0]
    b3_id, b3_members = sizes[1]
    singleton_ids = [cid for cid, m in sizes[2:]]
    singleton_members = [m[0] for _cid, m in sizes[2:]]  # each is a 1-member list

    panel_names = ["B1_B2_transfer", "B3_transfer", "remote_outlier"]
    neg_assignment = distribute_negative_components(neg_components, meta, panel_names)

    def negs_for(panel: str) -> list[str]:
        out = []
        for cid, target in neg_assignment.items():
            if target == panel:
                out.extend(neg_components[cid])
        return out

    def build_panel(panel_name: str, test_positive_ids: list[str], test_positive_component_ids: list[str]) -> dict:
        test_neg_ids = negs_for(panel_name)
        test_ids = sorted(test_positive_ids + test_neg_ids)
        held_out_component_ids = set(test_positive_component_ids) | {
            cid for cid, target in neg_assignment.items() if target == panel_name
        }
        train_ids = sorted(
            sid for cid, members in comps.items() if cid not in held_out_component_ids for sid in members
        )
        return {
            "test_positive_component_ids": test_positive_component_ids,
            "n_test_positive": len(test_positive_ids),
            "n_test_negative": len(test_neg_ids),
            "test_negative_family_counts": dict(Counter(
                meta.get(s, {}).get("neg_family", "") for s in test_neg_ids
            )),
            "test_ids": test_ids,
            "train_ids": train_ids,
        }

    panels = {
        "B1_B2_transfer": build_panel("B1_B2_transfer", b1b2_members, [b1b2_id]),
        "B3_transfer": build_panel("B3_transfer", b3_members, [b3_id]),
        "remote_outlier": build_panel("remote_outlier", singleton_members, singleton_ids),
    }
    panels["remote_outlier"]["singletons"] = [
        {"structure_id": m, "component_id": cid} for cid, m in zip(singleton_ids, singleton_members)
    ]

    # leave-one-negative-family-out (independent of the 3 panels above).
    # "Negative-family OOD false-positive challenges": test sets contain NO
    # positives, so these measure specificity/FPR only, never balanced
    # accuracy. A component is held out if it contains ANY member of the
    # target family (not just a majority) -- selecting by dominant family
    # alone left e.g. one RNase-Z protein in train during the RNase-Z LONO
    # (its component's majority was phosphodiesterase), and vice versa. This
    # guarantees zero target-family examples survive in train; the price is
    # that a held-out component's other, non-target-family members ("collateral")
    # also leave train, reported explicitly per family below.
    lono = {}
    for fam in NEGATIVE_FAMILIES_FOR_LONO:
        held_out_comp_ids = {
            cid for cid, m in neg_components.items()
            if fam in {meta.get(s, {}).get("neg_family", "") for s in m}
        }
        test_ids = sorted(sid for cid in held_out_comp_ids for sid in neg_components[cid])
        train_ids = sorted(
            sid for cid, members in comps.items() if cid not in held_out_comp_ids for sid in members
        )
        test_family_counts = Counter(meta.get(s, {}).get("neg_family", "") for s in test_ids)
        lono[fam] = {
            "n_test": len(test_ids),
            "n_target_family_in_test": test_family_counts.get(fam, 0),
            "collateral_family_counts_in_test": {k: v for k, v in test_family_counts.items() if k != fam},
            "n_target_family_in_train": sum(
                1 for sid in train_ids if meta.get(sid, {}).get("neg_family", "") == fam
            ),
            "test_ids": test_ids,
            "train_ids": train_ids,
        }
        assert lono[fam]["n_target_family_in_train"] == 0, f"{fam} LONO leaked target-family examples into train"

    # operational reference-anchored retrieval: same split as remote_outlier,
    # reported separately -- see module docstring.
    operational = {
        "note": "NOT a population sensitivity estimate. Report these as individual "
                "discovery case studies. Train set intentionally includes the "
                "reference bank as retrieval anchors -- these 4 are the only positives "
                "not already in a reference-bank-containing sequence component.",
        "test_ids": panels["remote_outlier"]["test_ids"][:0] + sorted(singleton_members),
        "train_ids": panels["remote_outlier"]["train_ids"],
    }

    # audit: every negative component tested exactly once; no component has
    # ANY member split between a panel's train and test (a subset-based check
    # -- "is this component entirely inside test" / "entirely inside train" --
    # would silently pass a component with, say, 3 members in test and 2 in
    # train, since neither subset condition holds for it either way; this
    # checks intersection with both sides directly instead).
    neg_test_counts = Counter(neg_assignment.values())
    for panel_name, panel in panels.items():
        train_set, test_set = set(panel["train_ids"]), set(panel["test_ids"])
        assert not (train_set & test_set), f"{panel_name}: train/test id overlap: {train_set & test_set}"
        for cid, members in comps.items():
            in_train = any(m in train_set for m in members)
            in_test = any(m in test_set for m in members)
            assert not (in_train and in_test), f"{panel_name}: component {cid} has members split across train/test"
    for fam, cfg in lono.items():
        train_set, test_set = set(cfg["train_ids"]), set(cfg["test_ids"])
        assert not (train_set & test_set), f"LONO {fam}: train/test id overlap: {train_set & test_set}"
        for cid, members in comps.items():
            in_train = any(m in train_set for m in members)
            in_test = any(m in test_set for m in members)
            assert not (in_train and in_test), f"LONO {fam}: component {cid} has members split across train/test"

    output = {
        "regime": "sequence_components (identity>=0.3, coverage>=0.8) -- see build_split_graph.py thresholds",
        "reference_bank_ids": sorted(REFERENCE_BANK_IDS),
        "panels": panels,
        "panels_note": "Each panel confounds two axes: (1) can the model recognize the held-out POSITIVE "
                        "component, and (2) can it reject whichever monolithic NEGATIVE family/component "
                        "happened to land in that panel via distribute_negative_components (e.g. B1_B2_transfer "
                        "carries the entire glyoxalase_ii component, B3_transfer carries almost all rnase_z). "
                        "Do NOT compare panel-level balanced accuracy across panels as though negative difficulty "
                        "were matched -- report positive sensitivity per held-out positive component, and "
                        "negative specificity/FPR pooled across panels AND stratified by negative family/component, "
                        "as separate axes.",
        "leave_one_negative_family_out": lono,
        "leave_one_negative_family_out_note": "Negative-family OOD false-positive challenges: test sets contain "
                                               "NO positives, so these measure specificity/FPR only, not balanced "
                                               "accuracy. Every component containing ANY member of the target "
                                               "family is held out entirely (not just majority-family components), "
                                               "so n_target_family_in_train is always 0; collateral_family_counts_in_test "
                                               "reports which other families got swept out of train as a side effect.",
        "operational_reference_anchored_retrieval": operational,
        "audit": {
            "n_positive_components": len(pos_components),
            "n_negative_components": len(neg_components),
            "n_negative_components_per_panel": dict(neg_test_counts),
            "every_negative_component_tested_exactly_once": len(neg_assignment) == len(neg_components),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info(f"Wrote component-challenge splits -> {args.out}")
    for name, panel in panels.items():
        log.info(f"  {name}: n_test_positive={panel['n_test_positive']} "
                 f"n_test_negative={panel['n_test_negative']} n_train={len(panel['train_ids'])}")
    log.info(f"  negative components per panel: {dict(neg_test_counts)}")


if __name__ == "__main__":
    main()
