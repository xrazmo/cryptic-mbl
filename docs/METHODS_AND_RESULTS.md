# Canonical B1 structural pharmacophore: methods and final results

## Rationale

The project asked whether explicit three-dimensional active-site information
could identify metallo-beta-lactamases beyond sequence-family recognition.
Initial ESM2-enhanced graph neural networks performed strongly, but audit showed
that a training-free nearest-neighbor classifier in mean ESM2 space reproduced
their predictions under a sequence-leaky split. Evaluation was therefore
rebuilt around exhaustive sequence similarity, exact full-chain structural
similarity, component-level challenge panels, and training-free sequence
comparators.

The final method focuses on subclass B1. Canonical B1 enzymes possess two
spatially coupled metal-coordination sites: a three-histidine site and an
Asp-Cys-His site. B3 enzymes and related metallohydrolases do not provide an
equally distinctive first-shell signature, and were not forced into the same
model.

## Structural representation

The detector enumerates canonical donor atoms from a complete single-chain
protein structure. Candidate triads comprise three nitrogen donors from three
distinct histidines and oxygen, sulfur, and nitrogen donors from distinct Asp,
Cys, and His residues. Triads are pruned by within-site distance fingerprints
and paired according to their inter-site centroid separation.

The six candidate donor coordinates are rigidly aligned by the Kabsch algorithm
to the donor frame of an experimental NDM-1/hydrolyzed-meropenem structure (PDB
4EYL). A candidate passes the primary pharmacophore when the fitted six-donor
RMSD is at most 1.25 A. The enumeration tolerance was frozen at 1.50 A. The
method does not use residue sequence order, residue numbering, alignments,
ESM embeddings, HMM scores, training labels, learned parameters, nearest
neighbors, or predicted metal coordinates.

Residue identity remains part of the chemical definition of a donor. The method
is therefore sequence-homology independent, not chemistry blind.

A hydrolyzed-product pose can be transferred with the fitted donor frame and
checked for clashes and pocket contacts. This is retained as secondary evidence
because the NDM-derived pose gate rejected valid B1 structures with different
pocket organization.

## Evaluation design

The principal panel contained 110 B1 structures and 410 negatives under the
audited sequence-remote component split. All test examples had less than 30%
identity at 80% bidirectional coverage to the corresponding training
partition; 105 B1 examples had no qualifying MMseqs hit. Additional evaluation
used all 931 labeled negatives and a literature-declared external panel of 15
canonical B1 experimental structures plus ten B2/B3 controls.

Controls included donor inventory alone, within-site geometry, coordinate
permutation preserving donor counts and chemistry, mean-ESM2 nearest-neighbor
retrieval, fARGene B1 and combined B1/B2 HMMs, and PLM-ARG.

## Results

Donor inventory alone detected all 110 panel B1 structures but called 386 of
410 negatives, demonstrating that residue composition was insufficient.
Within-site geometry reduced false positives to four. The complete six-donor
pharmacophore detected 109 of 110 B1 structures and called none of the 410
negatives, giving sensitivity 0.991, specificity 1.000, and balanced accuracy
0.995. It made zero calls among all 931 labeled negatives. Coordinate
permutation largely destroyed performance, supporting a dependence on native
three-dimensional organization rather than donor counts alone.

The detector recovered all ten panel B1 proteins missed by mean-ESM2
nearest-neighbor retrieval. Nevertheless, the fARGene B1-specific HMM detected
all 110 panel B1 examples and all 15 canonical external B1 structures. The
structural detector detected 14 of the 15 external B1 structures and rejected
all ten B2/B3 controls.

In a separate reviewed external pilot, the structural detector and fARGene B1
HMM produced identical sets of 17 calls. No structure-positive,
fARGene-negative discovery was obtained. A prevalence-blind pilot of 10,000
unreviewed bacterial proteins yielded no structural calls among 8,669 available
AlphaFold structures and no fARGene B1 calls. This latter result measures low
background activation, not the prevalence or absence of rare environmental
MBLs.

## Interpretation

The six-donor pharmacophore is a robust, interpretable detector of canonical B1
architecture and a useful orthogonal confirmation channel. It is more specific
to canonical B1 than broad beta-lactamase sequence classifiers and can recover
examples missed by simple ESM2 retrieval. It has not demonstrated incremental
recall over a properly thresholded B1 profile HMM on available known enzymes.

The likely discovery niche is therefore narrow: unusual sequence ordering,
spacing, or convergent topology that preserves the canonical donor geometry
while escaping a profile HMM. The current evidence does not justify using the
method as the primary engine for Atlas-scale novel-ARG discovery.

## Limitations

The positive corpus is dominated by B1 and contains few independent positive
components. Source provenance is confounded with class label. The external
panel tests structural portability rather than an independent novel-family
discovery. AlphaFold side-chain conformations may distort donor geometry.
Finally, geometric compatibility does not establish catalytic turnover,
resistance phenotype, expression, localization, mobility, or clinical risk.
