"""
test_cn_model.py
tests for CNModel

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_spec import SpecData
from bayes_cn_hfs import CNModel
from bayes_cn_hfs.utils import get_molecule_data

_MOL_DATA = get_molecule_data(
    "CN, v = 0, 1",
    vibrational_state=0,
    rot_state_lower=0,
)


def test_cn_model():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = CNModel(data, n_clouds=2, molecule="CN", mol_data=_MOL_DATA, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(prior_baseline_coeffs={"observation": [1.0, 1.0]}, prior_rms=1.0, prior_fwhm_nonthermal=1.0)
    model.add_likelihood()
    assert model._validate()


def test_cn_model_ordered():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = CNModel(data, n_clouds=2, molecule="CN", mol_data=_MOL_DATA, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()


def test_cn_model_assume_nonLTE():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = CNModel(data, n_clouds=2, molecule="CN", mol_data=_MOL_DATA, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(assume_LTE=False)
    model.add_likelihood()
    assert model._validate()


def test_cn_model_fix_Tkin():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = CNModel(data, n_clouds=2, molecule="CN", mol_data=_MOL_DATA, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(fix_log10_Tkin=1.5)
    model.add_likelihood()
    assert model._validate()


def test_cn_model_anomalies():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}
    model = CNModel(data, n_clouds=2, molecule="CN", mol_data=_MOL_DATA, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(prior_log_boltz_factor_sigma=1.0)
    model.add_likelihood()
    assert model._validate()
