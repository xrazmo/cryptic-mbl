"""Small chemistry constants shared by structural scoring code."""

# Side-chain atoms that can coordinate zinc in the MBL-related chemistries
# evaluated in this repository.
LIGAND_ATOMS = {
    "HIS": ("ND1", "NE2"),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "CYS": ("SG",),
}

ZN_BOND_CUTOFF = 2.8
