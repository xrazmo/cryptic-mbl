# B1 catalytic-architecture detector

## Scientific question

This branch asks a narrower and defensible question: can a candidate structure
support the canonical dinuclear catalytic architecture of subclass B1
metallo-beta-lactamases without using its sequence or comparing it with a panel
of labeled proteins? It does not attempt to recognize B3, and it does not claim
that a compatible structure confers antibiotic resistance.

The distinction matters. The earlier mixed-subclass structural experiment was
dominated by B3-like metallohydrolases, whose first-shell chemistry overlaps
glyoxalase-II, RNase Z, phosphodiesterases, and lactonases. Canonical B1 has a
more distinctive local architecture that can be tested separately.

## Model

`scripts/b1_structural_model.py` uses one provenance-tracked experimental
reaction-state structure, NDM-1 with hydrolyzed meropenem (PDB 4EYL). It uses
the structure only as a physical coordinate template, not as a nearest-neighbor
reference. A candidate can have an unrelated sequence or global fold.

A full positive call requires all of the following:

1. two predicted metal sites;
2. the site-resolved 3-N and N/O/S protein-donor architecture, including the
   canonical DCH cysteine ligand;
3. a donor/metal pharmacophore RMSD no greater than the predeclared 1.25 A
   physical-sanity gate;
4. transfer of the hydrolyzed-carbapenem pose with no more than 10% hard
   clashes and at least 50% pocket contact.

The output has four states. `supported` is the full structural call;
`partial_support` means a coordinating cysteine is present but the complete
architecture failed; `not_supported` means an evaluable dinuclear site lacks
the architecture; and `unavailable` means the required metal-site input was
not available. Missing input is never silently converted into biological
absence.

The implementation reads no sequence, ESM embedding, label, negative-family
identity, class centroid, or trained weight.

## Frozen results

On the sequence-remote B1/B2 challenge panel, restricted to the 110 B1
positives and the same 410 negatives:

| method | sensitivity | specificity | balanced accuracy |
|---|---:|---:|---:|
| DCH cysteine only | 0.945 (104/110) | 0.983 (403/410) | 0.964 |
| Site-resolved pharmacophore | 0.864 (95/110) | 1.000 (410/410) | 0.932 |
| Full B1 catalytic architecture | 0.855 (94/110) | 1.000 (410/410) | 0.927 |
| Mean-ESM2 5-NN | 0.909 (100/110) | 0.998 (409/410) | 0.953 |
| fARGene B1/B2 HMM | 1.000 (110/110) | 1.000 (410/410) | 1.000 |
| PLM-ARG beta-lactam call | 1.000 (110/110) | 0.990 (406/410) | 0.995 |

The full structural call also produced zero positive calls among all 931
labeled negative structures. This is an observed result on the present curated
corpus, not a universal specificity guarantee. In particular, 241/931
negatives had an unavailable full-model input state (usually no acceptable
dinuclear Metal3D prediction). The reported 1.000 is therefore operational
positive-call specificity with abstentions treated as non-hits, not proof that
all 931 negatives were geometrically evaluated and rejected.

Among 105 B1 panel positives with no MMseqs hit at the split's 80%-coverage
criterion, the full structural model detected 90 (0.857). It recovered 8 of
the 10 B1 examples missed by the panel-specific mean-ESM2 baseline. However,
fARGene detected all of them. “No MMseqs hit at 80% coverage” is therefore not
equivalent to “HMM-negative.”

Destroying donor directions while preserving donor-metal distances reduced
the full model from 94/110 to 1/110 B1 detections and retained zero negative
calls. The useful signal therefore depends on angular 3D organization, not
only the presence of cysteine or the list of donor elements.

## Honest conclusion

This is now a real structural model and a strong orthogonal B1 confirmation
channel. It is more specific than the cysteine-only rule and uses geometry in
a falsifiable way. It has not yet demonstrated discovery beyond fARGene:
fARGene recognized 111/112 B1 positives in the full labeled corpus, and the
sole HMM-negative B1 (`AAF94716.1`) also failed the structural detector.
The PLM-ARG compatibility run recognized all 112/112 B1 positives, with 9/931
negative beta-lactam calls. Its released category checkpoint was loaded with
the exact upstream XGBoost 1.6.1 but a newer scikit-learn runtime; the report
records this version mismatch, so these PLM-ARG numbers are useful comparator
evidence rather than a bit-for-bit legacy-environment certification.

The missing validation resource is not another architecture. It is a
prospective set of structures selected specifically because they fall below
the frozen fARGene threshold, followed by biochemical testing. Until that set
exists, the supported claim is “sequence-independent structural confirmation
of canonical B1 architecture,” not “superior HMM-negative discovery.”

No Atlas work is performed by this branch.

## External comparators

`scripts/run_fargene_comparator.py` invokes the MIT-licensed fARGene B1/B2 HMM
through HMMER and applies its published full-protein domain-score threshold of
127. The run is pinned by upstream Git revision and model SHA-256 in
`reports/fargene_b1_b2_comparator.json`.

`scripts/run_plm_arg_comparator.py` adapts the released PLM-ARG ESM-1b and
XGBoost checkpoints. PLM-ARG's upstream repository had no explicit license at
the pinned revision, so its source and checkpoints are not redistributed;
they must be supplied by path and are content-hashed in the result.

## Reproduction

Run these commands from the repository root in the `cryptic-mbl` Conda
environment. Derived FASTA and HMMER tables are ignored under `data/`.

```bash
python scripts/export_full_chain_sequences.py \
  --manifest configs/manifest.csv --raw-dir data/raw \
  --pockets-dir data/pockets_v2 \
  --out data/b1_benchmark/full_chain_sequences.fasta \
  --audit reports/b1_full_chain_sequence_audit.json

python scripts/run_fargene_comparator.py \
  --fasta data/b1_benchmark/full_chain_sequences.fasta \
  --hmm /path/to/fargene/fargene_analysis/models/class_B_1_2.hmm \
  --threshold 127 --upstream-checkout /path/to/fargene \
  --out reports/fargene_b1_b2_comparator.json

python scripts/run_plm_arg_comparator.py \
  --fasta data/b1_benchmark/full_chain_sequences.fasta \
  --esm1b-model /path/to/PLM-ARG/models/esm1b_t33_650M_UR50S.pt \
  --arg-model /path/to/PLM-ARG/models/arg_model.pkl \
  --category-model /path/to/PLM-ARG/models/cat_model.pkl \
  --category-index /path/to/PLM-ARG/models/Category_Index.csv \
  --upstream-checkout /path/to/PLM-ARG \
  --out reports/plm_arg_comparator.json

python scripts/evaluate_b1_structural_model.py \
  --pockets-dir data/pockets_v2 \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --manifest configs/manifest.csv --catalog full_structure_catalog.csv \
  --splits data/challenge_splits.json \
  --similarity-audit data/similarity_audit.json \
  --esm2-baseline data/mean_esm2_baseline.json \
  --fargene-results reports/fargene_b1_b2_comparator.json \
  --plm-arg-results reports/plm_arg_comparator.json \
  --out reports/b1_structural_model_evaluation.json
```
