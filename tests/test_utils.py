"""
test_utils.py
tests for utils.py

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np
from bayes_cn_hfs import utils


def test_get_molecule_data():
    with pytest.raises(ValueError):
        utils.get_molecule_data(molecule="BAD MOLECULE NAME")
    data = utils.get_molecule_data(
        "CN, v = 0, 1",
        vibrational_state=0,
        rot_state_lower=0,
    )
    assert "freq" in data.keys()
    assert "Aul" in data.keys()
    assert "degu" in data.keys()
    assert "El" in data.keys()
    assert "Eu" in data.keys()
    assert "relative_int" in data.keys()
    assert "log10_Q_terms" in data.keys()
    assert "Qu" in data.keys()
    assert "Ql" in data.keys()


def test_supplement_mol_data():
    with pytest.raises(ValueError):
        utils.supplement_mol_data(molecule="BAD MOLECULE NAME")
    mol_data, mol_weight = utils.supplement_mol_data("CN", mol_data=None)
    assert "Gu" in mol_data.keys()
    assert "Gl" in mol_data.keys()
    assert "state_u_idx" in mol_data.keys()
    assert "state_l_idx" in mol_data.keys()
    assert "states" in mol_data.keys()
    assert not np.isnan(mol_weight)
    mol_data, mol_weight = utils.supplement_mol_data("CN", mol_data=mol_data)
    mol_data, mol_weight = utils.supplement_mol_data("13CN", mol_data=None)
    mol_data, mol_weight = utils.supplement_mol_data("13CN", mol_data=mol_data)
