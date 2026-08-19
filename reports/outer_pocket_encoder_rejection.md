# Outer-pocket pretrained encoder: predeclared rejection

## Decision

**Rejected.** Not part of the production scoring path. B3 uses ESM2
retrieval alone.

## The bounded experiment

Per the agreed architecture, a single bounded go/no-go test was run for
a B3 structural channel: frozen pretrained ESM-IF1 GVP-GNN structure
encoder (`esm_if1_gvp4_t16_142M_UR50`, geometry-only by construction --
takes only backbone N/CA/C coordinates, no amino-acid identity), pooled
over residues within 16A of either corrected metal site (outer pocket),
nearest-centroid prototype classifier, evaluated on B3_transfer plus
lactonase/phosphodiesterase leave-one-negative-family-out.

## Go/no-go criterion (predeclared, not adjusted after seeing results)

> Keep Part 2 only if it recovers B3 positives missed by ESM2 or
> suppresses hard-negative false positives at the same sensitivity and
> specificity. Otherwise stop and use ESM2 as the B3 channel.

## Result: fails both conditions

| check | result |
|---|---|
| Unique recovery of ESM2-baseline misses (8 missed positives) | **0/8** |
| FPR vs. ESM2 baseline, per hard-negative family | **worse on 5/9 families** (alpha_beta_hydrolase 10% vs 0%, globin_fold 14.3% vs 0%, rossmann_sdr 33.3% vs 0%, thioredoxin_fold 25% vs 18.8%, lysozyme_like 100% vs 0%, n=3) |
| Outer-pocket vs. first-shell-only pooling (same embeddings) | outer pocket clearly better (0.615 vs 0.077 sensitivity) -- confirms wider context carries real signal, just not signal sharper than ESM2's |
| Geometry-only vs. geometry+chemistry | identical results -- chemistry's 8 dims made no measurable difference against a 512-dim geometry vector; this control was dimensionally too weak to be decisive, but does not affect the decision since geometry alone already failed both gates |
| LONO (lactonase, phosphodiesterase) | zero positives in either split by design (pure negative-family FPR test); lactonase LONO produced 18/101 false positives, phosphodiesterase LONO 23/277 -- both consistent with the primary result, not contradicting it |

## Why, briefly

ESM-IF1 was pretrained for inverse folding -- predicting a plausible
sequence from a backbone -- which optimizes for generic fold/geometry
consistency, not enzyme-family functional discrimination. It clearly
carries real structural signal (outer-pocket sensitivity 0.615, well
above the first-shell-only 0.077), just not signal sharper than ESM2's
sequence embedding for separating true B3 MBLs from structurally
similar decoys.

## Deployment status

Never run during production scoring (`score_candidate_v2.py` does not
call it). The `esm-if1` conda environment and `run_esm_if1.sh` /
`scripts/esm_if1_worker.py` remain in the repository as a record of
what was tried and how, not as a deployment dependency. No further
encoder experiments (coordinate-scrambled control, GearNet, or
otherwise) are planned per this decision.
