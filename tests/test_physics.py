"""
test_physics.py
tests for physics.py

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_cn_hfs import physics
from bayes_cn_hfs.utils import supplement_mol_data

_MOL_DATA, _MOL_WEIGHT = supplement_mol_data("CN")


def test_gaussian():
    x = np.linspace(-10.0, 10.0, 101)
    y = physics.gaussian(x, 0.0, 1.0).eval()
    assert not np.any(np.isnan(y))


def test_lorentzian():
    x = np.linspace(-10.0, 10.0, 101)
    y = physics.lorentzian(x, 0.0, 1.0)
    assert not np.any(np.isnan(y))


def test_balance():
    transitions_free, transition_l, transition_u = physics.balance(
        _MOL_DATA, log10_N0=None, log_boltz_factor=None, verbose=False
    )
    assert len(transitions_free) > 0
    assert len(transition_l) > 0
    assert len(transition_u) > 0

    log_boltz_factor = np.random.normal(size=len(_MOL_DATA["freq"]))
    transitions_free, log10_Nl, log10_Nu = physics.balance(
        _MOL_DATA, log10_N0=12.0, log_boltz_factor=log_boltz_factor, verbose=False
    )
    assert not np.any(np.isnan(log10_Nl.eval()))
    assert not np.any(np.isnan(log10_Nu.eval()))


def test_calc_frequency():
    velocity = np.linspace(-10.0, 10.0, 101)
    frequency = physics.calc_frequency(_MOL_DATA["freq"][:, None], velocity)
    assert not np.any(np.isnan(frequency))


def test_calc_fwhm_freq():
    fwhm = np.array([10.0, 20.0, 30.0])
    fwhm_freq = physics.calc_fwhm_freq(_MOL_DATA["freq"][:, None], fwhm)
    assert not np.any(np.isnan(fwhm_freq))


def test_calc_thermal_fwhm():
    kinetic_temp = 100.0
    weight = 26.0
    fwhm = physics.calc_thermal_fwhm(kinetic_temp, weight).eval()
    assert not np.isnan(fwhm)


def test_calc_log_boltz_factor():
    freq = 112000.0
    Tex = 50.0
    log_boltz_factor = physics.calc_log_boltz_factor(freq, Tex)
    assert not np.isnan(log_boltz_factor)


def test_calc_Tex():
    freq = 112000.0
    log_boltz_factor = 5.0
    Tex = physics.calc_Tex(freq, log_boltz_factor)
    assert not np.isnan(Tex)


def test_calc_psuedo_voight():
    freq_axis = np.linspace(1000.0, 1100.0, 101)
    frequency = np.array([[1050.0, 1051.0], [1050.0, 1051.0]])
    fwhm = np.array([[10.0, 20.0], [10.0, 20.0]])
    fwhm_L = np.array([1.0, 1.0])
    line_profile = physics.calc_pseudo_voigt(freq_axis, frequency, fwhm, fwhm_L).eval()
    assert line_profile.shape == (101, 2, 2)


def test_calc_optical_depth():
    gu = 6.0
    gl = 2.0
    Nl = 1.0e12
    log_boltz_factor = 1.5
    line_profile = 1.0
    freq = 112000.0
    Aul = 1.0e-12
    optical_depth = physics.calc_optical_depth(gu, gl, Nl, log_boltz_factor, line_profile, freq, Aul).eval()
    assert not np.isnan(optical_depth)


def test_predict_tau():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    Nl = np.ones((len(_MOL_DATA["freq"]), 2)) * 1.0e12
    log_boltz_factor = np.random.normal(size=(len(_MOL_DATA["freq"]), 2))
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    fwhm_L = 1.0
    tau = physics.predict_tau(_MOL_DATA, freq_axis, Nl, log_boltz_factor, velocity, fwhm, fwhm_L).eval()
    assert tau.shape == (500, len(_MOL_DATA["freq"]), 2)
    assert not np.any(np.isnan(tau))


def test_rj_temperature():
    assert physics.rj_temperature(1800.0, -10.0).eval() < physics.rj_temperature(1800.0, 10.0).eval()


def test_radiative_transfer():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    Nl = np.ones((len(_MOL_DATA["freq"]), 2)) * 1.0e12
    log_boltz_factor = np.random.normal(size=(len(_MOL_DATA["freq"]), 2))
    Tex = physics.calc_Tex(115000.0, log_boltz_factor)
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    fwhm_L = 1.0
    tau = physics.predict_tau(_MOL_DATA, freq_axis, Nl, log_boltz_factor, velocity, fwhm, fwhm_L).eval()
    bg_temp = 2.7
    tb = physics.radiative_transfer(freq_axis, tau, Tex, bg_temp).eval()
    assert tb.shape == (500,)
