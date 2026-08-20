# Project context: concluded canonical-B1 structural detector

Last updated: 2026-08-20

Status: concluded; no Atlas screening authorized from this repository

## Biological objective

The original objective was to discover rare, sequence-divergent antibiotic
resistance metallo-beta-lactamases using three-dimensional structure. The
central concern was that HMMs, BLAST-like searches, and ESM-derived models may
mainly recover evolutionary relatives of known MBLs. A useful structural
method would need to contribute information not reducible to sequence-family
recognition.

The labeled corpus contains 1,077 structures: 146 positives and 931 negatives.
Positive coverage is highly uneven (112 B1, 27 B3, 3 B2, and 4 unclassified),
and most positives occupy very few sequence or structural components. This
prevents a credible claim of uniform generalization across MBL subclasses.

## What was established

1. The first large ESM2 improvement was primarily homology recognition. A
   training-free mean-ESM2 nearest-neighbor baseline reproduced the trained
   model, and the original split contained extensive sequence overlap.
2. The documented pocket RMSD/identity split rule had not been implemented.
   When implemented literally, pocket-fragment alignments formed a biologically
   meaningless giant component. The replacement split uses exhaustive MMseqs2
   sequence edges and exact full-chain Foldseek TM-scores; pocket similarity is
   diagnostic only.
3. Structure-only GNN, coordination-fingerprint, and outer-pocket encoder
   variants contained real signal but did not reliably outperform sequence
   methods. B3 first-shell chemistry was indistinguishable from several related
   metallohydrolase negative families.
4. Canonical B1 has a distinctive structural opportunity: the complete
   three-His plus Asp-Cys-His six-donor architecture. The full-chain scorer
   evaluates this geometry without a reference bank or predicted metal site.
5. The frozen B1 scorer achieved 109/110 sensitivity and 410/410 specificity on
   the audited sequence-remote panel, 0/931 calls among all labeled negatives,
   and 14/15 sensitivity with 0/10 B2/B3 calls on external experimental
   structures.
6. The method did not demonstrate complementary recovery beyond fARGene's
   B1-specific HMM. On a 92-protein reviewed external pilot, its 17 primary
   calls exactly matched the B1 HMM. A broader 10,000-sequence pilot contained
   no structural or HMM B1 calls; because B1 ARGs are extremely rare, this is a
   background-specificity observation, not a prevalence or discovery-sensitivity
   estimate.

## Final interpretation

The repository delivers a highly specific, mechanistically interpretable
canonical-B1 structural confirmation assay. It does not deliver a validated
standalone system for discovering HMM-negative ARGs. The prospect of unique
recall is nonzero—spatial convergence or unusual residue ordering could defeat
a sequence model—but likely too small to justify indiscriminate structure
prediction or Atlas-scale screening.

The term "sequence independent" means independent of sequence similarity,
alignment, HMMs, and protein-language-model embeddings. The detector still
uses residue chemical identity to enumerate His, Asp, and Cys donor atoms.

A positive structural call means that a supplied structure is compatible with
the canonical B1 donor architecture. It does not establish beta-lactam
hydrolysis, resistance phenotype, expression, localization, mobility, or
clinical risk.

## Frozen active method

- Scorer: `scripts/metal_independent_b1.py`
- Template:
  `data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz`
- Primary output: `result.architecture_call`
- Frozen pharmacophore RMSD gate: 1.25 A
- Frozen donor-pair enumeration tolerance: 1.50 A
- Main evaluation: `reports/metal_independent_b1_evaluation.json`
- External evaluation: `reports/external_experimental_mbl_panel.json`
- Threshold audit: `reports/metal_independent_b1_threshold_sensitivity.json`

## Explicitly rejected or demoted approaches

- Legacy fused GNN: sequence-dominant, seed-sensitive, and not the final method.
- Mean-ESM2 retrieval: strong family-recognition comparator, not structural
  evidence.
- Metal3D-anchored B1 scoring: technically useful predecessor, but missed known
  positives when predicted sites were absent or displaced.
- Cysteine/DCH-only rule: biologically informative for B1/B2 but insufficient
  as a complete structural model.
- First-shell B3 fingerprint: cannot separate B3 from related metal hydrolases.
- ESM-IF1 outer-pocket prototype: real structural signal but failed its
  predeclared incremental-value gate.
- Generic substrate-conditioned catalytic-feasibility prototype: failed
  specificity and belongs to a separate catalytic-chemistry project.
- Random-protein discovery sampling: useful for background call burden, not for
  estimating rare-MBL discovery sensitivity.

## Boundary for future work

Do not enlarge the GNN, retune the B1 pharmacophore on discovery candidates, or
run the ESM Atlas from this repository. If catalytic-chemistry discovery is
pursued, create a separate repository centered on reaction geometry, water and
metal chemistry, pocket electrostatics, substrate accommodation, and catalytic
dynamics. Historical starting material is in
`archive/catalytic_chemistry_handoff/`.

For the full scientific history and audit trail, see
`archive/project_history/PROJECT_CONTEXT_2026-08-19.md`.
