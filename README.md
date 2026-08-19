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
