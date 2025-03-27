"""
test_cn_ratio_model.py
tests for CNRatioModel

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
import pytest

from bayes_spec import SpecData
from bayes_cn_hfs import CNRatioModel


def test_cn_ratio_model():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "12CN": SpecData(freq_axis, brightness, 1.0),
        "13CN": SpecData(freq_axis, brightness, 1.0),
    }
    model = CNRatioModel(data, n_clouds=1, baseline_degree=1)
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    with pytest.raises(ValueError):
        model.add_priors(
            assume_LTE=True, assume_CTEX_12CN=False, assume_CTEX_13CN=False
        )
    model.add_priors(
        prior_baseline_coeffs={"12CN": [1.0, 1.0], "13CN": [1.0, 1.0]},
        prior_rms={"12CN": 1.0, "13CN": 1.0},
        prior_fwhm_nonthermal=1.0,
        prior_fwhm_L=1.0,
    )
    model.add_likelihood()
    assert model._validate()


def test_cn_ratio_model_ordered():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "12CN": SpecData(freq_axis, brightness, 1.0),
        "13CN": SpecData(freq_axis, brightness, 1.0),
    }
    model = CNRatioModel(data, n_clouds=1, baseline_degree=1)
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()


def test_cn_ratio_model_nonLTE():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "12CN": SpecData(freq_axis, brightness, 1.0),
        "13CN": SpecData(freq_axis, brightness, 1.0),
    }
    model = CNRatioModel(data, n_clouds=1, baseline_degree=1)
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    model.add_priors(assume_LTE=False, assume_CTEX_12CN=False, assume_CTEX_13CN=False)
    model.add_likelihood()
    assert model._validate()


def test_cn_ratio_model_fix_Tkin():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "12CN": SpecData(freq_axis, brightness, 1.0),
        "13CN": SpecData(freq_axis, brightness, 1.0),
    }
    model = CNRatioModel(data, n_clouds=1, baseline_degree=1)
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    model.add_priors(fix_log10_Tkin=1.5)
    model.add_likelihood()
    assert model._validate()
