"""
test_physics.py - tests for physics.py

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
from numpy.testing import assert_allclose

from bayes_cn_hfs import physics
from bayes_cn_hfs.utils import get_molecule_data

_MOL_DATA = get_molecule_data(
    "CN, v = 0, 1",
    vibrational_state=0,
    rot_state_lower=0,
)


def test_calc_frequency():
    velocity = np.linspace(-10.0, 10.0, 101)
    frequency = physics.calc_frequency(_MOL_DATA, velocity)
    assert not np.any(np.isnan(frequency))


def test_calc_fwhm_freq():
    fwhm = np.array([10.0, 20.0, 30.0])
    fwhm_freq = physics.calc_fwhm_freq(_MOL_DATA, fwhm)
    assert not np.any(np.isnan(fwhm_freq))


def test_calc_line_profile():
    freq_axis = np.linspace(1000.0, 1100.0, 1001)
    frequency = np.array([[1050.0, 1051.0], [1050.0, 1051.0]])
    fwhm = np.array([[1.0, 2.0], [1.0, 2.0]])
    line_profile = physics.calc_line_profile(freq_axis, frequency, fwhm).eval()
    assert line_profile.shape == (1001, 2, 2)
    assert_allclose(line_profile.sum(axis=0) * (freq_axis[1] - freq_axis[0]), np.ones((2, 2)))
    exp_line_profile = np.array(
        [
            np.sqrt(4.0 * np.log(2.0) / np.pi),
            0.5 * np.sqrt(np.log(2.0) / np.pi),
        ]
    )
    assert_allclose(line_profile[500, 0, :], exp_line_profile)


def test_detailed_balance():
    tex = np.array([5.0, 10.0, 20.0])
    abundance = physics.detailed_balance(_MOL_DATA, tex).eval()
    assert not np.any(np.isnan(abundance))


def test_calc_optical_depth():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    frequency = np.stack([_MOL_DATA["freq"] for i in range(2)]).T
    fwhm = np.stack([[1.0] * len(_MOL_DATA["freq"]) for i in range(2)]).T
    line_profile = physics.calc_line_profile(freq_axis, frequency, fwhm).eval()
    N = np.array([1.0e14, 1.5e14])
    cloud_tex = np.array([8.0, 12.0])
    component_tex = cloud_tex[None, :]
    optical_depth = physics.calc_optical_depth(_MOL_DATA, N, cloud_tex, component_tex, line_profile).eval()
    assert optical_depth.shape == (500, len(_MOL_DATA["freq"]), 2)
    assert np.all(optical_depth >= 0)


def test_predict_tau():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    log10_N = np.array([14.0, 13.5])
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    cloud_tex = np.array([8.0, 12.0])
    component_tex = cloud_tex[None, :]
    tau = physics.predict_tau(_MOL_DATA, freq_axis, log10_N, velocity, fwhm, cloud_tex, component_tex).eval()
    assert tau.shape == (500, len(_MOL_DATA["freq"]), 2)
    assert np.all(tau >= 0)


def test_rj_temperature():
    assert physics.rj_temperature(1800.0, -10.0).eval() < physics.rj_temperature(1800.0, 10.0).eval()


def test_radiative_transfer():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    log10_N = np.array([14.0, 13.5])
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    cloud_tex = np.array([8.0, -12.0])
    component_tex = cloud_tex[None, :]
    tau = physics.predict_tau(_MOL_DATA, freq_axis, log10_N, velocity, fwhm, cloud_tex, component_tex).eval()
    bg_temp = 2.7
    tb = physics.radiative_transfer(freq_axis, tau, component_tex, bg_temp).eval()
    assert tb.shape == (500,)
