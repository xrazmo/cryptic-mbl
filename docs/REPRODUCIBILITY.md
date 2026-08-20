# Reproducibility guide

## Active environment

The final B1 scorer is a CPU-compatible NumPy/BioPython workflow. Install the
repository requirements in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

`requirements.txt` and the CUDA environment are retained for historical graph,
sequence, and model comparators; they are not required for a single B1
structural score.

## Score one structure

Supply one cleaned single-chain PDB structure:

```bash
python scripts/metal_independent_b1.py \
  --structure candidate.pdb \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --pair-distance-tolerance 1.5 \
  --out candidate.b1_structure.json
```

The frozen primary decision is `result.architecture_call`. Do not substitute
`positive_call`, which is a backward-compatible alias for the stricter and
less portable transferred-product pose gate.

## Reproduce the principal evaluations

```bash
python scripts/evaluate_metal_independent_b1.py \
  --structures-dir data/domain_pdbs \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --manifest configs/manifest.csv \
  --catalog full_structure_catalog.csv \
  --splits data/challenge_splits.json \
  --similarity-audit data/similarity_audit.json \
  --esm2-baseline data/mean_esm2_baseline.json \
  --metal-anchored-results reports/b1_structural_model_evaluation.json \
  --fargene-results reports/fargene_b1_b2_comparator.json \
  --fargene-b1-results reports/fargene_b1_specific_comparator.json \
  --plm-arg-results reports/plm_arg_comparator.json \
  --pair-distance-tolerance 1.5 \
  --workers 8 \
  --out /tmp/metal_independent_b1_evaluation.json

python scripts/evaluate_external_mbl_panel.py \
  --config configs/external_experimental_mbl_panel.json \
  --raw-dir data/external_experimental_mbl/raw \
  --chains-dir data/external_experimental_mbl/chains \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --fasta-out /tmp/external_mbl_sequences.fasta \
  --fargene-results reports/external_fargene_b1_b2_comparator.json \
  --fargene-b1-results reports/external_fargene_b1_specific_comparator.json \
  --plm-arg-results reports/external_plm_arg_comparator.json \
  --out /tmp/external_experimental_mbl_panel.json

python scripts/evaluate_b1_threshold_sensitivity.py \
  --structures-dir data/domain_pdbs \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --manifest configs/manifest.csv \
  --splits data/challenge_splits.json \
  --external-config configs/external_experimental_mbl_panel.json \
  --external-chains-dir data/external_experimental_mbl/chains \
  --out /tmp/metal_independent_b1_threshold_sensitivity.json
```

Generated files should match the tracked reports in their scientific metrics.
Some metadata hashes or timestamps may differ if source files are regenerated.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## External comparators

Comparator outputs are tracked because their upstream packages and model files
have independent installation requirements. The JSON reports record source
revisions, thresholds, hashes, and runtime qualifications:

- `reports/fargene_b1_specific_comparator.json`
- `reports/fargene_b1_b2_comparator.json`
- `reports/plm_arg_comparator.json`
- corresponding `external_*` reports

The correctly applied fARGene B1-specific threshold is 135.8. Do not infer a
threshold from another fARGene model's output.

## Data and archive policy

The main derived inputs remain under `data/`. Large historical checkpoints,
embeddings, discovery-pilot downloads, and plots were moved to
`archive/artifacts/`, which is intentionally ignored by Git. See
`archive/README.md` for the inventory and original locations.
