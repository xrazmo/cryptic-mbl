# Reference-independent B1 structural detector

> **Final status (2026-08-20):** frozen as a canonical-B1 structural
> confirmation method. The project did not demonstrate recovery of a known B1
> missed by the correctly thresholded fARGene B1 HMM, and Atlas-scale screening
> is not recommended from this repository.

## Scientific question

This branch tests a deliberately narrow hypothesis: can a protein structure
support the canonical B1 metallo-beta-lactamase catalytic architecture without
using its sequence or comparing it with a labeled reference panel?

This is not a B2/B3 detector and it is not a claim of catalytic activity. B3
MBLs share first-shell His/Asp/Glu zinc chemistry with several non-MBL
metallohydrolases, whereas canonical B1 enzymes contain a more distinctive
two-site architecture: a three-histidine site and an Asp-Cys-His site. Treating
those biological problems separately avoids forcing one model to solve
incompatible subclasses with very unequal training data.

## Model and information boundary

The primary implementation is `scripts/metal_independent_b1.py`. It enumerates
donor atoms directly from a complete single-chain structure:

- a Zn1-like triad of three distinct histidine N donors;
- a Zn2-like triad of Asp O, Cys S, and His N donors;
- the relative three-dimensional arrangement of all six donor atoms.

Candidate donor sets are fitted to one provenance-tracked experimental
reaction-state template: NDM-1 with hydrolyzed meropenem (PDB 4EYL). The final
six-donor pharmacophore gate is RMSD <= 1.25 A. The frozen donor-pair
enumeration tolerance is 1.50 A; it is a search prefilter, not the final match
criterion. A transferred hydrolyzed-meropenem pose supplies separate clash and
pocket-contact evidence.

The scorer does **not** read:

- primary sequence order, residue numbering, or motif regular expressions;
- ESM embeddings, HMM scores, or sequence alignments;
- labels, negative-family identities, centroids, or a reference bank;
- predicted metal coordinates.

Residue chemistry is necessarily used to define possible donor atoms. This is
not the one-feature cysteine rule: the donor-inventory control has almost no
specificity, and coordinate permutation destroys the useful signal.

`scripts/b1_structural_model.py` is the earlier Metal3D-anchored version. It is
retained for comparison but is no longer the preferred detector because a
missed or displaced predicted metal site can make a known B1 unevaluable. The
full-chain implementation removes that bottleneck.

## Output interpretation

The recommended primary channel is `six_donor_pharmacophore`, derived from the
`pharmacophore_rmsd` gate. The transferred product-pose result should be used
as a secondary rank or review flag. It is more specific to the NDM-derived
template pocket and rejected a valid BlaB structure despite a tight six-donor
fit.

Some crystallographic structures encode the catalytic cysteine as an oxidized
modified residue (`CSD`, `CSO`, or `OCS`). The scorer preserves its SG position
for architecture recognition but emits `uses_modified_cysteine_donor` and
separate native-thiolate calls. A modified-cysteine match supports conserved
spatial architecture; it does not establish an intact catalytic thiolate in
that experimental structure.

A positive call means “compatible with the canonical B1 donor architecture.”
It does not prove beta-lactam hydrolysis, antibiotic resistance, expression,
or clinical relevance.

## Frozen internal results

The primary evaluation used the audited sequence-remote B1/B2 transfer panel,
restricted to 110 B1 positives and the same 410 labeled negatives.

| method | sensitivity | specificity | balanced accuracy |
|---|---:|---:|---:|
| Donor inventory only | 1.000 (110/110) | 0.059 (24/410) | 0.529 |
| Within-site geometry | 0.991 (109/110) | 0.990 (406/410) | 0.991 |
| **Six-donor B1 pharmacophore** | **0.991 (109/110)** | **1.000 (410/410)** | **0.995** |
| Six-donor plus transferred-product pose | 0.982 (108/110) | 1.000 (410/410) | 0.991 |
| Mean-ESM2 5-NN | 0.909 (100/110) | 0.998 (409/410) | 0.953 |
| fARGene B1-specific HMM | 1.000 (110/110) | 1.000 (410/410) | 1.000 |
| PLM-ARG beta-lactam category | 1.000 (110/110) | 0.990 (406/410) | 0.995 |

The six-donor model made zero positive calls among all 931 labeled negative
structures and recovered all 10 B1 panel positives missed by mean-ESM2 5-NN.
Among the 105 B1 panel positives with no MMseqs hit at the audited 80%-coverage
criterion, it detected 104. Unlike the older metal-anchored model, every input
was geometrically evaluated; absence of a Metal3D prediction is no longer
treated as a negative call.

The result is not reducible to detecting cysteine. The donor-inventory rule
called 386/410 panel negatives positive. Randomly permuting donor coordinates
while preserving the number and chemical type of donor atoms reduced the
six-donor result to 2/110 B1 detections and produced 15/410 false positives.
Native role-specific 3D organization is therefore essential.

The internal B1 misses are informative:

- `MBS5055441.1` lacks a complete DCH plus three-His geometry;
- `G09` contains DCH chemistry but lacks an acceptable three-His triad;
- `AAF94716.1`, the sole fARGene-negative labeled B1, contains no cysteine
  residue/SG atom in its supplied full-chain structure and cannot support the
  canonical architecture as represented.

The model recovered 15 B1 examples that the Metal3D-anchored implementation
missed, including VIM-2 and IMP-1. This demonstrates that predicted metal-site
placement was a real technical bottleneck. It rejected all 30 known B2/B3
positives in the internal corpus and called three of four unclassified
positives (`UPP01678.1`, `AFV91534.1`, and `AVX51087.1`); those three are
testable subclass-assignment hypotheses, not newly discovered enzymes.

## External experimental-structure panel

`configs/external_experimental_mbl_panel.json` declares a literature-derived
PDB panel before structural scoring: 15 canonical B1 enzymes, one
noncanonical-B1 boundary case (SPS-1), two B2 controls, and eight B3 controls.
The list is based on representative structures tabulated in
[a modern MBL review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8792953/) and
[an earlier structure-function review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3970115/).

At the frozen settings:

| method | canonical B1 sensitivity | B2/B3 control specificity |
|---|---:|---:|
| **Six-donor B1 pharmacophore** | **0.933 (14/15)** | **1.000 (10/10)** |
| Native-thiolate six-donor call | 0.800 (12/15) | 1.000 (10/10) |
| Six-donor plus product pose | 0.867 (13/15) | 1.000 (10/10) |
| fARGene B1-specific HMM | 1.000 (15/15) | 1.000 (10/10) |
| fARGene combined B1/B2 HMM | 1.000 (15/15) | 0.800 (8/10) |
| PLM-ARG beta-lactam category | 1.000 (15/15) | 0.000 (0/10) |

This is an external experimental-structure portability panel, not an
independent novel-family discovery set. Several entries are canonical families
also represented in the project corpus, and the physical template itself is
NDM-derived. It tests whether the coordinate rule survives across deposited
experimental structures and excludes other MBL subclasses; it cannot establish
prospective novelty.

The combined-HMM and PLM-ARG specificity columns are not error rates for their
intended tasks: B2/B3 MBLs are expected positives for those broader sequence
models. They show that the structural output is specifically a canonical-B1
architecture call rather than a generic beta-lactamase call.

The primary external miss is NDM-1 PDB 3S0Z, whose three-His geometry is too
distorted for the 1.50 A enumeration prefilter. BlaB-1 passes the six-donor
architecture but fails only the transferred NDM-product clash gate. SPM-1 and
FIM-1 pass using oxidized crystallographic cysteine residues and are explicitly
flagged as modified-ligand architecture matches. SPS-1 and all B2/B3 controls
are rejected, consistent with the declared canonical-B1 scope.

## Threshold sensitivity

The donor-pair prefilter was swept from 1.00 to 2.00 A without changing the
fixed 1.25 A six-donor RMSD gate. The internal result remained exactly 109/110
and 0/410 at every setting, with 0/931 calls among all labeled negatives. The
external panel changed from 13/15 at 1.00-1.25 A, to 14/15 at the frozen 1.50 A,
to 15/15 at 1.75-2.00 A; B2/B3 calls remained 0/10 throughout. Because the
external miss motivated inspection of this margin, 1.75 A is reported only as
a post-hoc sensitivity result and is not adopted as a new validated primary
threshold.

## Comparison with sequence methods and honest claim

The structural model outperforms the panel-specific mean-ESM2 nearest-neighbor
baseline on this B1 task and recovers all of that baseline's misses. It does
not outperform fARGene's B1 HMM on the available known positives. The present
dataset therefore establishes a strong sequence-homology-independent and
reference-independent structural confirmation method. It does not establish
prospective discovery of a biochemically verified fARGene-negative B1 enzyme,
and substantial unique recall beyond a modern B1 HMM appears unlikely.

That distinction is crucial. This method is not another HMM wrapper: its score
cannot change when non-donor sequence residues are replaced while coordinates
and donor chemistry remain fixed. It can in principle recognize a sequence
outside an HMM boundary. The scientifically defensible final scope is
canonical-B1 architecture confirmation and structural/sequence disagreement
review, not primary rare-ARG discovery.

No Atlas processing was performed, and none is recommended from this branch.

## Reproduction

Run from the repository root in the `cryptic-mbl` Conda environment. External
raw PDB files and generated chain files are derived data under `data/`.

```bash
python scripts/evaluate_metal_independent_b1.py \
  --structures-dir data/domain_pdbs \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --manifest configs/manifest.csv --catalog full_structure_catalog.csv \
  --splits data/challenge_splits.json \
  --similarity-audit data/similarity_audit.json \
  --esm2-baseline data/mean_esm2_baseline.json \
  --metal-anchored-results reports/b1_structural_model_evaluation.json \
  --fargene-results reports/fargene_b1_b2_comparator.json \
  --fargene-b1-results reports/fargene_b1_specific_comparator.json \
  --plm-arg-results reports/plm_arg_comparator.json \
  --pair-distance-tolerance 1.5 --workers 8 \
  --out reports/metal_independent_b1_evaluation.json

python scripts/evaluate_external_mbl_panel.py \
  --config configs/external_experimental_mbl_panel.json \
  --raw-dir data/external_experimental_mbl/raw \
  --chains-dir data/external_experimental_mbl/chains \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --fasta-out data/external_experimental_mbl/sequences.fasta \
  --download-missing \
  --fargene-results reports/external_fargene_b1_b2_comparator.json \
  --fargene-b1-results reports/external_fargene_b1_specific_comparator.json \
  --plm-arg-results reports/external_plm_arg_comparator.json \
  --out reports/external_experimental_mbl_panel.json

python scripts/evaluate_b1_threshold_sensitivity.py \
  --structures-dir data/domain_pdbs \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --manifest configs/manifest.csv --splits data/challenge_splits.json \
  --external-config configs/external_experimental_mbl_panel.json \
  --external-chains-dir data/external_experimental_mbl/chains \
  --out reports/metal_independent_b1_threshold_sensitivity.json
```

Official comparator implementations are [fARGene](https://github.com/fannyhb/fargene)
and [PLM-ARG](https://github.com/Junwu302/PLM-ARG). Comparator reports record
upstream revisions, model hashes, thresholds, and runtime-version caveats.
