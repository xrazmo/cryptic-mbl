from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalytic_feasibility import ReactionTemplate  # noqa: E402
from metal_independent_b1 import DonorAtom, score_donor_roles  # noqa: E402


def template() -> ReactionTemplate:
    donor_coords = np.array([
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0],
        [4.0, 2.0, 0.0], [4.0, -2.0, 0.0], [4.0, 0.0, 2.0],
    ])
    return ReactionTemplate(
        template_id="synthetic_B1", pdb_id="TEST", subclass="B1", protein_chain="A",
        ligand_resname="LIG", substrate_class="carbapenem", reaction_state="product",
        source_url="https://example.invalid", citation_doi="test", resolution_angstrom=1.0,
        metal_coords=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        donor_coords=donor_coords,
        donor_elements=np.array(["O", "S", "N", "N", "N", "N"]),
        donor_site_indices=np.array([0, 0, 0, 1, 1, 1]),
        donor_labels=np.array(["D", "C", "H4", "H1", "H2", "H3"]),
        ligand_coords=np.array([[2.0, 6.0, 0.0]]),
        ligand_elements=np.array(["C"]), ligand_atom_names=np.array(["X1"]),
    )


def donor(coord, resname, res_id, atom_name, modified=False) -> DonorAtom:
    return DonorAtom(
        coord=np.asarray(coord, dtype=float), resname=resname, res_id=res_id,
        atom_name=atom_name, atom_index=res_id, chain_id="A", ins_code="",
        modified_from_cysteine=modified,
    )


def roles(modified_cysteine=False):
    coords = template().donor_coords
    return {
        "ASP_O": [donor(coords[0], "ASP", 1, "OD1")],
        "CYS_S": [donor(
            coords[1], "CSO" if modified_cysteine else "CYS", 2, "SG",
            modified=modified_cysteine,
        )],
        "HIS_N": [
            donor(coords[2], "HIS", 3, "NE2"),
            donor(coords[3], "HIS", 4, "NE2"),
            donor(coords[4], "HIS", 5, "ND1"),
            donor(coords[5], "HIS", 6, "NE2"),
        ],
    }


def protein():
    coords = template().donor_coords
    return SimpleNamespace(
        coord=coords.copy(), element=np.array(["O", "S", "N", "N", "N", "N"]),
    )


class MetalIndependentB1Tests(unittest.TestCase):
    def score(self, donor_roles):
        return score_donor_roles(
            protein(), donor_roles, template(),
            max_hard_clash_fraction=1.0, min_pocket_contact_fraction=0.0,
        )

    def test_exact_six_donor_architecture_is_supported(self):
        result = self.score(roles())
        self.assertTrue(result["positive_call"])
        self.assertTrue(result["architecture_call"])
        self.assertTrue(result["native_thiolate_positive_call"])
        self.assertTrue(result["native_thiolate_architecture_call"])
        self.assertFalse(result["uses_modified_cysteine_donor"])

    def test_complete_dch_site_is_required(self):
        donor_roles = roles()
        donor_roles["CYS_S"] = []
        result = self.score(donor_roles)
        self.assertFalse(result["positive_call"])
        self.assertFalse(result["architecture_call"])
        self.assertEqual(result["n_dch_triads"], 0)

    def test_modified_cysteine_support_is_reported_separately(self):
        result = self.score(roles(modified_cysteine=True))
        self.assertTrue(result["positive_call"])
        self.assertTrue(result["uses_modified_cysteine_donor"])
        self.assertFalse(result["native_thiolate_positive_call"])
        self.assertFalse(result["native_thiolate_architecture_call"])


if __name__ == "__main__":
    unittest.main()
