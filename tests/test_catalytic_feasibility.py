from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalytic_feasibility import (  # noqa: E402
    ReactionTemplate,
    apply_transform,
    candidate_donors,
    kabsch_transform,
    score_catalytic_feasibility,
)
from utils import PocketMetadata, PocketSubgraph  # noqa: E402


def synthetic_template() -> ReactionTemplate:
    return ReactionTemplate(
        template_id="synthetic",
        pdb_id="TEST",
        subclass="B2",
        protein_chain="A",
        ligand_resname="LIG",
        substrate_class="carbapenem",
        reaction_state="hydrolyzed_product",
        source_url="https://example.invalid",
        citation_doi="test",
        resolution_angstrom=1.0,
        metal_coords=np.array([[0.0, 0.0, 0.0]]),
        donor_coords=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        donor_elements=np.array(["N", "O", "S"]),
        donor_site_indices=np.array([0, 0, 0]),
        donor_labels=np.array(["HIS:1:NE2", "ASP:2:OD1", "CYS:3:SG"]),
        ligand_coords=np.array([
            [-1.8, -1.8, -1.8], [-2.4, -1.8, -1.8], [-1.8, -2.4, -1.8],
            [-1.8, -1.8, -2.4], [-2.4, -2.4, -1.8], [-2.4, -1.8, -2.4],
            [-1.8, -2.4, -2.4], [-2.4, -2.4, -2.4],
        ]),
        ligand_elements=np.array(["C", "C", "O", "N", "C", "O", "C", "S"]),
        ligand_atom_names=np.array([f"X{i}" for i in range(8)]),
    )


def synthetic_pocket() -> PocketSubgraph:
    donor_coords = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    contact_coords = np.array([
        [-4.5, -1.8, -1.8], [-1.8, -4.5, -1.8], [-1.8, -1.8, -4.5],
        [-4.5, -4.5, -1.8], [-4.5, -1.8, -4.5], [-1.8, -4.5, -4.5],
    ])
    coords = np.vstack([donor_coords, contact_coords])
    return PocketSubgraph(
        res_ids=np.arange(1, len(coords) + 1),
        res_names=np.array(["HIS", "ASP", "CYS"] + ["ALA"] * len(contact_coords)),
        coords=coords,
        atom_names=np.array(["NE2", "OD1", "SG"] + ["CA"] * len(contact_coords)),
        elements=np.array(["N", "O", "S"] + ["C"] * len(contact_coords)),
        is_sidechain=np.array([True, True, True] + [False] * len(contact_coords)),
        metal_coords=np.array([[0.0, 0.0, 0.0]]),
        metal_probabilities=np.array([1.0]),
        metadata=PocketMetadata(
            source_structure_id="synthetic", label="unlabeled", confidence_tier=1,
            pocket_source="metal3d",
        ),
    )


class CatalyticFeasibilityTests(unittest.TestCase):
    def test_kabsch_recovers_proper_rigid_transform(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rotation_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation_true = np.array([3.0, -2.0, 5.0])
        target = source @ rotation_true + translation_true
        rotation, translation, rmsd = kabsch_transform(source, target)
        self.assertLess(rmsd, 1e-8)
        np.testing.assert_allclose(apply_transform(source, rotation, translation), target, atol=1e-8)
        self.assertGreater(np.linalg.det(rotation), 0.999)

    def test_template_round_trip(self):
        template = synthetic_template()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.npz"
            template.save(path)
            restored = ReactionTemplate.load(path)
        self.assertEqual(restored.template_id, template.template_id)
        np.testing.assert_array_equal(restored.donor_elements, template.donor_elements)
        np.testing.assert_allclose(restored.ligand_coords, template.ligand_coords)

    def test_exact_local_pharmacophore_is_supported(self):
        result = score_catalytic_feasibility(synthetic_pocket(), [synthetic_template()])
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["n_evaluable"], 1)
        self.assertEqual(result["n_supported"], 1)
        self.assertLess(result["template_results"][0]["pharmacophore_rmsd"], 1e-6)

    def test_candidate_donor_selection_excludes_noncanonical_atoms(self):
        pocket = synthetic_pocket()
        donors = candidate_donors(pocket)
        self.assertEqual([donor.element for donor in donors], ["N", "O", "S"])
        self.assertEqual([donor.atom_index for donor in donors], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
