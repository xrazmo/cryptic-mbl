from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from b1_structural_model import B1_TEMPLATE_ID, score_b1_structure  # noqa: E402
from catalytic_feasibility import ReactionTemplate  # noqa: E402
from utils import PocketMetadata, PocketSubgraph  # noqa: E402


def template() -> ReactionTemplate:
    metal = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    donors = np.array([
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0],
        [4.0, 2.0, 0.0], [4.0, -2.0, 0.0], [4.0, 0.0, 2.0],
    ])
    return ReactionTemplate(
        template_id=B1_TEMPLATE_ID, pdb_id="TEST", subclass="B1", protein_chain="A",
        ligand_resname="LIG", substrate_class="carbapenem", reaction_state="product",
        source_url="https://example.invalid", citation_doi="test", resolution_angstrom=1.0,
        metal_coords=metal, donor_coords=donors,
        donor_elements=np.array(["N", "N", "N", "O", "S", "N"]),
        donor_site_indices=np.array([0, 0, 0, 1, 1, 1]),
        donor_labels=np.array(["H1", "H2", "H3", "D", "C", "H4"]),
        ligand_coords=np.array([
            [2.0, 3.0, 0.0], [2.5, 3.0, 0.0], [1.5, 3.0, 0.0], [2.0, 3.5, 0.0],
            [2.0, 3.0, 0.5], [2.5, 3.5, 0.0], [1.5, 3.5, 0.0], [2.0, 3.5, 0.5],
        ]),
        ligand_elements=np.array(["C", "C", "O", "N", "C", "O", "C", "S"]),
        ligand_atom_names=np.array([f"X{i}" for i in range(8)]),
    )


def pocket(include_sulfur: bool = True) -> PocketSubgraph:
    donor_coords = template().donor_coords.copy()
    names = np.array(["HIS", "HIS", "HIS", "ASP", "CYS" if include_sulfur else "ALA", "HIS"])
    atom_names = np.array(["NE2", "ND1", "NE2", "OD1", "SG" if include_sulfur else "CB", "NE2"])
    elements = np.array(["N", "N", "N", "O", "S" if include_sulfur else "C", "N"])
    contacts = np.array([[2.0, 5.5, 0.0], [4.5, 3.0, 0.0], [-0.5, 3.0, 0.0]])
    return PocketSubgraph(
        res_ids=np.arange(1, 10), res_names=np.concatenate([names, ["ALA"] * 3]),
        coords=np.vstack([donor_coords, contacts]),
        atom_names=np.concatenate([atom_names, ["CA"] * 3]),
        elements=np.concatenate([elements, ["C"] * 3]),
        is_sidechain=np.array([True] * 6 + [False] * 3),
        metal_coords=template().metal_coords.copy(), metal_probabilities=np.ones(2),
        metadata=PocketMetadata(source_structure_id="synthetic", label="unlabeled",
                                confidence_tier=1, pocket_source="metal3d"),
    )


class B1StructuralModelTests(unittest.TestCase):
    def test_complete_architecture_is_supported(self):
        result = score_b1_structure(pocket(), template())
        self.assertEqual(result["status"], "supported")
        self.assertTrue(result["positive_call"])

    def test_cysteine_is_necessary_but_not_sufficient_output_channel(self):
        result = score_b1_structure(pocket(include_sulfur=False), template())
        self.assertFalse(result["positive_call"])
        self.assertNotEqual(result["status"], "partial_support")


if __name__ == "__main__":
    unittest.main()
