# Archive inventory

This directory preserves superseded work and large generated artifacts. Nothing
was deleted during project closure. Archived material is retained for audit,
historical reproduction, or transfer into a separate research project; it is
not part of the recommended B1 scoring path.

## Tracked historical material

- `project_history/` — the detailed project context before final closure.
- `publication/legacy_gnn/` — the earlier production-GNN DOCX and HTML. These
  documents predate the final B1 pharmacophore conclusion and must not be read
  as the current model description.
- `legacy_gnn/reports/` — frozen manifests and validation reports for the
  sequence-dominant production pipeline.
- `legacy_metal3d/` — the historical Metal3D launcher and environment.
- `rejected_structure_experiments/` — ESM-IF1 outer-pocket code and rejection
  report plus the coordination-fingerprint findings.
- `catalytic_chemistry_handoff/` — the rejected generic reaction-state
  feasibility experiment. This is starting material for a new, explicitly
  separate catalytic-chemistry repository.
- `discovery_pilots/` — scripts for the reviewed and prevalence-blind UniProt
  pilots. They are preserved as evidence, not recommended sourcing workflows.

The active `scripts/catalytic_feasibility.py` remains in place because the B1
scorers reuse its provenance-aware template container and rigid-geometry
primitives. Its generic multi-subclass catalytic-feasibility interpretation was
rejected and is documented here; its presence does not make it a production
channel.

## Large untracked artifacts

`archive/artifacts/` is intentionally ignored by Git. The closure operation
moved the following recoverable directories there:

| archived path | original path | status |
|---|---|---|
| `artifacts/legacy_gnn/models/` | `models/` | trained legacy checkpoints |
| `artifacts/legacy_gnn/results/` | `results/` | legacy fold plots and metrics |
| `artifacts/legacy_gnn/data/` | selected `data/` entries | GNN outputs, pockets, ESM2 embeddings, and production banks |
| `artifacts/discovery_pilots/b1_pilot/` | `data/b1_pilot/` | 92-protein reviewed pilot |
| `artifacts/discovery_pilots/b1_broad_pilot/` | `data/b1_broad_pilot/` | 10,000-sequence broad pilot |
| `artifacts/discovery_pilots/foldseek_b1_reference/` | `data/foldseek_b1_reference/` | generated Foldseek database |
| `artifacts/rejected_structure/outer_pocket_embeddings/` | `data/outer_pocket_embeddings/` | ESM-IF1 embeddings |
| `artifacts/rejected_structure/data/` | selected `data/` entries | coordination fingerprints and DCH experiment outputs |
| `artifacts/legacy_metal3d/assets/` | `assets/` | vendored historical tool assets |
| `artifacts/misc/files.zip` | `files.zip` | unrelated legacy bundle |

These files remain on disk and can be moved back if an old command must be
replayed. Git history at the original experiment commit is the authoritative
source for historical paths.
