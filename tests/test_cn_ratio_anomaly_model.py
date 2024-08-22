"""
test_cn_ratio_anomaly_model.py - tests for CNRatioAnomalyModel

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from bayes_spec import SpecData
from bayes_cn_hfs import CNRatioAnomalyModel
from bayes_cn_hfs.utils import get_molecule_data

_MOL_DATA_12CN = get_molecule_data(
    "CN, v = 0, 1",
    vibrational_state=0,
    rot_state_lower=0,
)
_MOL_DATA_13CN = get_molecule_data(
    "C-13-N",
    rot_state_lower=0,
)


def test_cn_ratio_anomaly_model():
    freq_axis_12CN = np.linspace(113470, 113530, 825)
    freq_axis_13CN = np.linspace(108750, 108820, 825)
    brightness = np.random.randn(825)
    data = {
        "12CN": SpecData(freq_axis_12CN, brightness, 1.0),
        "13CN": SpecData(freq_axis_13CN, brightness, 1.0),
    }
    model = CNRatioAnomalyModel(data, n_clouds=2, baseline_degree=1)
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    model = CNRatioAnomalyModel(
        data, n_clouds=2, baseline_degree=1, mol_data_12CN=_MOL_DATA_12CN, mol_data_13CN=_MOL_DATA_13CN
    )
    assert isinstance(model.mol_data_12CN, dict)
    assert isinstance(model.mol_data_13CN, dict)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()


def test_cn_ratio_anomaly_model_ordered():
    freq_axis_12CN = np.linspace(113470, 113530, 825)
    freq_axis_13CN = np.linspace(108750, 108820, 825)
    brightness = np.random.randn(825)
    data = {
        "12CN": SpecData(freq_axis_12CN, brightness, 1.0),
        "13CN": SpecData(freq_axis_13CN, brightness, 1.0),
    }
    model = CNRatioAnomalyModel(data, n_clouds=2, baseline_degree=1)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()
