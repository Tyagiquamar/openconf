"""Regression tests for stereochemistry inferred from 3D inputs."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf.api import _filter_stereochemistry_consistent_conformers
from openconf.perceive import prepare_molecule, specified_stereochemistry


@pytest.mark.parametrize(
    ("smiles", "flipped_smiles"),
    [
        ("C/C=C/C", r"C/C=C\C"),
        ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
    ],
    ids=["2-butene", "lactic-acid"],
)
def test_3d_input_filters_stereochemical_flips(smiles: str, flipped_smiles: str) -> None:
    """Unannotated 3D inputs reject E/Z and tetrahedral flips, even at lower energy."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    flipped = Chem.AddHs(Chem.MolFromSmiles(flipped_smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    assert AllChem.EmbedMolecule(flipped, randomSeed=2) == 0
    Chem.RemoveStereochemistry(mol)

    mol = prepare_molecule(mol, add_hs=False)
    reference = specified_stereochemistry(mol)
    mol.AddConformer(Chem.Conformer(flipped.GetConformer()), assignId=True)
    conf_ids, energies = _filter_stereochemistry_consistent_conformers(mol, [0, 1], [0.0, -1.0], reference)

    assert conf_ids == [0]
    assert energies == [0.0]
    assert [conf.GetId() for conf in mol.GetConformers()] == [0]


def test_explicit_stereochemistry_takes_precedence_over_3d() -> None:
    """Explicit trans-2-butene labels survive conflicting cis coordinates."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C/C=C/C"))
    cis = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C\C"))
    assert AllChem.EmbedMolecule(cis, randomSeed=1) == 0
    mol.AddConformer(Chem.Conformer(cis.GetConformer()), assignId=True)

    prepared = prepare_molecule(mol, add_hs=False)

    assert Chem.MolToSmiles(Chem.RemoveHs(prepared)) == "C/C=C/C"


def test_2d_input_keeps_stereochemistry_unspecified() -> None:
    """2D butene coordinates do not specify E/Z stereochemistry."""
    mol = Chem.MolFromSmiles("CC=CC")
    AllChem.Compute2DCoords(mol)

    reference = specified_stereochemistry(prepare_molecule(mol))

    assert reference.tetrahedral == {}
    assert reference.bonds == {}
