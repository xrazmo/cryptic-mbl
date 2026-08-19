# Structural V3 catalytic-feasibility experiment: NO-GO

## Decision

Do **not** integrate Structural V3 into production and do **not** begin an
Atlas screen with it. The existing V2 scoring regime remains unchanged.

This was a bounded test of the hypothesis that the reference-protein panel
was preventing the model from learning a generic structural signal. V3
removed that bottleneck completely: it used no sequence, ESM embedding,
training label, class centroid, nearest neighbour, or reference-protein vote.
Instead, it transferred experimentally observed hydrolyzed beta-lactam
reaction states into each candidate's corrected metal/donor frame and tested
local pharmacophore fit, severe clashes, and pocket contact.

The result shows that reference-panel comparison was **not the main
bottleneck**. Candidate-local 3D geometry recovered some sequence misses, but
the same geometry was common in related metallohydrolases and produced far too
many false positives.

## Frozen gate and result

The acceptance gate was written into
`scripts/evaluate_catalytic_feasibility.py` before the corpus run.

| condition | required | observed | pass |
|---|---:|---:|---:|
| combined B1/B2+B3 ESM2 misses recovered | at least 5/20 | **8/20** | yes |
| specificity among evaluable negatives | at least 0.95 | **0.338** | no |
| maximum FPR in any negative family | at most 0.20 | **0.608** | no |
| fraction of structures evaluable | at least 0.80 | **0.509** | no |
| scrambled-donor positive support / native | at most 0.50 | **0/115 = 0** | yes |

Because all conditions were required, the result is a decisive NO-GO.

## Panel results

Unavailable cases are counted as misses for sensitivity and are reported
separately. They are not silently presented as structural negatives.

| panel | end-to-end sensitivity | specificity among evaluable negatives | unavailable |
|---|---:|---:|---:|
| B1/B2 transfer | 0.836 | 0.283 | 170/526 |
| B3 transfer | 0.615 | 0.240 | 173/314 |
| remote outlier | 0.500 | 0.878 | 186/237 |
| all structures | 0.788 | 0.338 | 529/1,077 |

Across all 931 negatives, 286 were called supported, a 0.307 false-positive
fraction even before resolving the 499 structurally unavailable negatives.

The tool recovered eight positives missed by mean-ESM2, all from the B1/B2
panel. It recovered none of the eight B3 ESM2 misses because those candidates
were structurally unavailable under the required two-site/donor-role match.

## Hard-negative failure

| negative family | n | false positives | FPR |
|---|---:|---:|---:|
| glyoxalase-II | 286 | 174 | **0.608** |
| RNase Z | 162 | 73 | **0.451** |
| phosphodiesterase | 123 | 32 | **0.260** |
| lactonase | 101 | 7 | 0.069 |

The dominant errors occur exactly where the biology predicts difficulty:
these enzymes share a metallo-hydrolase scaffold and reaction-compatible Zn
coordination geometry. A hydrolyzed-product pose can therefore be transferred
through the first-shell donor frame even when beta-lactam hydrolysis is not the
annotated biological function. The current steric/contact test does not encode
the dynamics, water/proton network, transition-state energetics, or substrate
access/release needed to separate them.

## Mechanistic control

Randomizing donor directions while preserving each donor-metal bond length
reduced positive support from 115/146 to 0/146. Specificity increased to
0.983-0.996 across panels. The native signal is therefore genuinely dependent
on three-dimensional donor geometry; the failure is not caused by accidentally
reading sequence or residue IDs. The problem is that this real 3D signal is
not function-specific enough.

## Experimental templates

The audited template set contains:

- 4EYL: B1 NDM-1 with hydrolyzed meropenem;
- 1X8I: B2 CphA with a trapped biapenem reaction state;
- 2AIO: B3 L1 with hydrolyzed moxalactam;
- 6U0Z: B3 L1 with hydrolyzed penicillin G.

Source PDB and generated-template hashes are frozen in
`reports/catalytic_template_audit.json`. The templates are mechanistic
coordinate sources, not a positive-protein reference bank.

## Scientific conclusion

Removing the reference bank does allow a structural method to retrieve some
ESM2 misses, but it does not provide the specificity needed for discovery.
This falsifies the simple version of the hypothesis that panel comparison was
the primary obstacle. The remaining obstacle is functional degeneracy:
first-shell metal geometry and a rigid transferred product pose are shared by
multiple metallohydrolase activities.

Further progress would require qualitatively new evidence—reaction-energy or
transition-state modelling, substrate-conditioned flexible docking with
matched decoy calibration, or experimental activity labels—not another
classifier or threshold fitted to the same 1,077 structures. Those additions
are outside this bounded experiment. No Atlas processing was performed.

## Reproduction

```bash
python scripts/build_catalytic_templates.py \
  --pdb-dir /path/to/experimental_pdbs \
  --out-dir data/catalytic_templates \
  --report reports/catalytic_template_audit.json

python scripts/evaluate_catalytic_feasibility.py \
  --pockets-dir data/pockets_v2 \
  --templates-dir data/catalytic_templates \
  --splits data/challenge_splits.json \
  --esm2-baseline data/mean_esm2_baseline.json \
  --catalog full_structure_catalog.csv \
  --out reports/catalytic_feasibility_evaluation.json
```

The first command can use `--download-missing` to fetch only the four public
RCSB entries. It does not access the ESM Metagenomic Atlas.
