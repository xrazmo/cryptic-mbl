# Canonical B1 structural pharmacophore

This repository is a concluded research project on structure-guided detection
of metallo-beta-lactamases (MBLs). Its final supported deliverable is a frozen,
reference-independent structural detector for the canonical subclass B1
six-donor architecture. It is not an Atlas-scale discovery system and it is
not a general detector of beta-lactam hydrolysis.

Project status: **concluded and frozen on 2026-08-20**.

## Final scientific conclusion

The full-chain detector in `scripts/metal_independent_b1.py` searches directly
for the spatial arrangement of a three-histidine site and an Asp-Cys-His site.
It uses donor chemistry and coordinates, but no sequence alignment, HMM score,
ESM embedding, labeled reference bank, trained GNN, or predicted metal
coordinate.

The detector is a strong and interpretable **canonical-B1 structural
confirmation channel**:

| evaluation | structural result | sequence comparator |
|---|---:|---:|
| Sequence-remote internal B1 panel | 109/110 sensitivity; 410/410 specificity | fARGene B1: 110/110; 410/410 |
| All labeled negatives | 0/931 structural calls | - |
| External experimental structures | 14/15 canonical B1; 0/10 B2/B3 calls | fARGene B1: 15/15; 0/10 calls |
| Reviewed external discovery pilot | 17 structural calls | exactly the same 17 called by fARGene B1 |
| Broad 10,000-sequence pilot | 0 structural calls among 8,669 available structures | 0 fARGene B1 calls |

The structural rule recovered all ten internal B1 examples missed by the
project's mean-ESM2 nearest-neighbor baseline. However, it recovered **no
experimentally supported B1 missed by the correctly thresholded fARGene B1
HMM**. Because the canonical donor residues also create a conserved sequence
signature, the probability of substantial unique recall beyond modern HMM and
protein-language-model methods is judged low.

Accordingly:

- do use this method to confirm and interpret canonical B1 architecture;
- do use structural/sequence disagreement as a review flag;
- do not use it as the primary engine for rare-ARG discovery;
- do not claim that a positive call proves hydrolysis or resistance;
- do not begin ESM Atlas-scale screening from this repository.

The separate question of fold-independent carbapenem catalytic chemistry has
been deliberately moved outside this project's scope. Historical prototypes
are preserved under `archive/catalytic_chemistry_handoff/` for transfer into a
new repository.

## Primary entry point

```bash
python scripts/metal_independent_b1.py \
  --structure path/to/single_chain_structure.pdb \
  --template data/catalytic_templates/B1_NDM1_hydrolyzed_meropenem_4EYL.npz \
  --out candidate.b1_structure.json
```

Interpret `result.architecture_call` as compatibility with the frozen
six-donor B1 pharmacophore. `result.full_pose_call` is a secondary,
NDM-template-dependent pose check and must not replace the primary call.

## Documentation

- `docs/PROJECT_CLOSURE.md` — final decision, evidence, limitations, and
  scientific boundary.
- `docs/B1_STRUCTURAL_DETECTOR.md` — implementation and complete evaluation.
- `docs/METHODS_AND_RESULTS.md` — manuscript-ready description of the final
  method and findings.
- `docs/REPRODUCIBILITY.md` — environments, required data, commands, and tests.
- `reports/project_closure_manifest.json` — hashes and verification state for
  the sealed active method and evidence.
- `PROJECT_CONTEXT.md` — concise handoff context for future maintainers.
- `archive/README.md` — inventory and status of historical work.

## Repository layout

```text
configs/            Frozen manifests and external-panel declarations
data/               Active derived inputs required by documented evaluations
docs/               Final scientific and reproducibility documentation
reports/            Audited final B1 results and sequence-comparator outputs
scripts/            Active detector plus retained historical pipeline code
tests/              Lightweight structural-geometry unit tests
archive/            Superseded experiments, legacy documents, and artifacts
```

Large generated artifacts are not versioned. Their locations and scientific
status are recorded in `archive/README.md`.

## License

See `LICENSE`.
