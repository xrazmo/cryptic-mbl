# Project closure statement

## Decision

This project is concluded as a canonical-B1 structural pharmacophore study.
The final method is scientifically useful as an orthogonal structural
confirmation channel, but the accumulated evidence does not justify presenting
it as a primary engine for novel ARG discovery or scaling it to the ESM
Metagenomic Atlas.

No result was discarded because it was negative. The archived experiments are
part of the scientific output: they establish which apparent improvements were
caused by sequence homology, which structural representations added signal,
and where that signal failed to provide unique practical recall.

## Final supported claim

> A frozen, full-chain search for the canonical B1 three-His plus Asp-Cys-His
> six-donor geometry recognizes canonical B1 architecture with high sensitivity
> and specificity across the available sequence-remote and experimental-
> structure panels, without using sequence similarity, learned weights,
> predicted metal coordinates, or a labeled reference bank.

The preferred detector called 109 of 110 B1 positives and none of 410 negatives
in the primary sequence-remote panel. Across all 931 labeled negatives it made
zero calls. On the external experimental-structure panel it called 14 of 15
canonical B1 structures and rejected all ten B2/B3 controls.

These values are properties of the evaluated datasets, not estimates of
population prevalence or positive predictive value in environmental data.

## Claim that is not supported

The project did not identify an experimentally supported B1 enzyme detected by
structure and missed by the correctly thresholded fARGene B1 HMM. The reviewed
external pilot produced identical primary call sets for the structural detector
and B1 HMM. Therefore the repository does not establish prospective discovery
of HMM-negative B1 ARGs.

This limitation has a biological reason. The structural detector depends on
the canonical B1 ligand inventory and geometry. Those His, Asp, and Cys ligands
are encoded in primary sequence and constrained by the B1 fold, giving profile
HMMs and modern protein language models access to much of the same evolutionary
signal. Three-dimensional arrangement can still rescue unusual ordering,
spacing, or convergent topology, but the available evidence suggests that this
will be a narrow exception rather than a large undiscovered class.

## Appropriate uses

- Confirm canonical B1 donor architecture in a supplied structure.
- Distinguish canonical B1 architecture from B2/B3 controls.
- Flag disagreement between sequence and structural channels for expert review.
- Provide interpretable donor mappings and geometry for a candidate report.
- Serve as a reproducible negative/positive control in later research.

## Inappropriate uses

- Calling an ARG from structure alone.
- Treating `architecture_call` as a probability of resistance.
- Detecting B2, B3, or every beta-lactamase class.
- Detecting an alternative fold that hydrolyses beta-lactams by different
  chemistry.
- Retuning thresholds after examining a discovery pool.
- Atlas-scale brute-force screening as the primary discovery strategy.

## Why the project remains valuable

The repository contains more than a final classifier. It documents several
generally important methodological findings:

1. Pretrained sequence embeddings can appear to improve a structural model
   while merely reproducing family nearest-neighbor recognition.
2. Small pocket-fragment RMSD is unsafe as a transitive redundancy edge;
   short-alignment artifacts can join unrelated folds into giant components.
3. Full-chain structure and exact search settings are essential for honest
   structural leakage audits.
4. B1 and B3 should not be forced into one structural discrimination problem:
   their useful chemistry and confusing negative families differ.
5. First-shell coordination can be biologically genuine yet functionally
   non-discriminating across a metallohydrolase superfamily.
6. A simple mechanistic pharmacophore can be more stable and interpretable than
   a small-data GNN, while still having a narrower discovery domain than an HMM.

## New-project boundary

Fold-independent carbapenem catalytic chemistry is deliberately not continued
here. A future project would need to model reaction compatibility—metal and
water placement, substrate orientation, intermediate stabilization,
second-shell electrostatics, loop accommodation, proton transfer, and product
release—rather than recognize the canonical B1 motif. That is a distinct
scientific hypothesis, dataset requirement, and validation program.

The rejected reaction-state and outer-pocket prototypes are retained in the
archive as starting material, with their failure modes documented. They should
not be silently promoted into the active B1 method.
