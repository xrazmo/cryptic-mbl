# Script status

## Final supported path

- `metal_independent_b1.py` — preferred full-chain canonical-B1 scorer.
- `evaluate_metal_independent_b1.py` — principal internal evaluation.
- `evaluate_external_mbl_panel.py` — external experimental-structure panel.
- `evaluate_b1_threshold_sensitivity.py` — frozen threshold sensitivity audit.
- `run_fargene_comparator.py` and `run_plm_arg_comparator.py` — external
  sequence comparators.

`b1_structural_model.py`, `catalytic_feasibility.py`, `dch_score.py`,
`structural_chemistry.py`, and `utils.py` provide retained template, geometry,
chemistry, legacy-comparison, or data-loading functions used by the supported
path. `graph_construction.py` is retained for the historical graph pipeline but
is no longer imported by the final scorer.

## Retained audit and split utilities

The remaining graph construction, sequence export, split, and audit scripts
form the provenance of the final evaluation. They are not alternative
production models.

Learned GNN training/inference, rejected catalytic-chemistry and outer-pocket
experiments, Metal3D pocket generation, and UniProt discovery-pilot entry points
were moved under `archive/`; see `archive/README.md`.
