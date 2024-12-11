"""
test_hfs_anomaly_model.py - tests for HFSAnomalyModel

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_spec import SpecData
from bayes_cn_hfs import HFSAnomalyModel
from bayes_cn_hfs.utils import get_molecule_data

_MOL_DATA = get_molecule_data(
    "CN, v = 0, 1",
    vibrational_state=0,
    rot_state_lower=0,
)


def test_hfs_anomaly_model():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = HFSAnomalyModel(
        data, n_clouds=2, mol_weight=26.0, mol_data=_MOL_DATA, baseline_degree=1
    )
    assert isinstance(model.mol_data, dict)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()


def test_hfs_anomaly_model_ordered():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = HFSAnomalyModel(
        data, n_clouds=2, mol_weight=26.0, mol_data=_MOL_DATA, baseline_degree=1
    )
    assert isinstance(model.mol_data, dict)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()
