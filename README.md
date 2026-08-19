# cryptic-mbl

Structure-guided discovery of cryptic metallo-β-lactamases from environmental
metagenomes. The repository combines structure prediction, Foldseek, corrected
multi-site Metal3D pocket extraction, ESM2 retrieval, and explicitly separated
structural evidence channels.

## Current scientific status

The production path is `scripts/score_candidate_v2.py`. It keeps canonical
B1/B2 DCH coordination support and ESM2 retrieval separate; the legacy GNN is
not part of the decision. See `reports/production_model_v2_manifest.json`.

A family-reference-independent, substrate-conditioned Structural V3 experiment
is implemented in `scripts/catalytic_feasibility.py`. It transfers experimental
hydrolyzed beta-lactam reaction states into a candidate's metal/donor frame,
without sequence, labels, nearest neighbours, or class centroids. It was tested
under a frozen gate and **rejected for production**: despite recovering 8/20
ESM2 misses and passing a donor-geometry destruction control, specificity among
evaluable negatives was only 0.338 because related metallohydrolases share the
same catalytic-looking metal scaffold. See
`reports/catalytic_feasibility_no_go.md` and
`reports/catalytic_feasibility_evaluation.json`.

No ESM Metagenomic Atlas processing should begin from this branch. Structural
V3 is retained as a reproducible negative experiment, not a deployed discovery
score.

## B1 structural detector branch

`feature/b1-structural-detector` narrows the structural question to canonical
subclass B1 rather than requiring one model to solve B1, B2, and B3. The
preferred scorer is now `scripts/metal_independent_b1.py`. It searches the
full chain for the complete three-His plus Asp-Cys-His six-donor architecture,
without sequence, a labeled reference panel, ESM embeddings, or predicted
metal coordinates. The hydrolyzed-meropenem pose is retained as secondary
evidence rather than the primary detector.

The frozen evaluation and external sequence comparators are described in
`docs/B1_STRUCTURAL_DETECTOR.md`. On the sequence-remote B1 panel the
six-donor model detected 109/110 B1 structures with 0/410 false positives,
including all ten B1 examples missed by mean-ESM2 5-NN. It also detected 14/15
literature-selected canonical B1 PDB structures and rejected all ten B2/B3
controls at the frozen setting. fARGene's B1 HMM still detected every known B1
in those evaluated panels, so prospective HMM-negative discovery remains the
next biological validation rather than an established claim. Atlas screening
remains deliberately out of scope on this branch.
