"""
physics.py
Hyperfine physics

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt

import astropy.constants as c

_K_B = c.k_B.to("erg K-1").value
_H = c.h.to("erg MHz-1").value
_C = c.c.to("km/s").value
_C_CM_MHZ = c.c.to("cm MHz").value


def gaussian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Gaussian function

    :param x: Position at which to evaluate
    :type x: float
    :param center: Gaussian centroid
    :type center: float
    :param fwhm: Gaussian full-width at half-maximum
    :type fwhm: float
    :return: Gaussian evaluated at :param:x
    :rtype: float
    """
    return pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0) * pt.sqrt(
        4.0 * pt.log(2.0) / (np.pi * fwhm**2.0)
    )


def lorentzian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Lorentzian function

    :param x: Position at which to evaluate
    :type x: float
    :param center: Centroid
    :type center: float
    :param fwhm: Full-width at half-maximum
    :type fwhm: float
    :return: Lorentzian evaluated at :param:x
    :rtype: float
    """
    return fwhm / (2.0 * np.pi) / ((x - center) ** 2.0 + (fwhm / 2.0) ** 2.0)


def calc_frequency(mol_data: dict, velocity: Iterable[float]) -> Iterable[float]:
    """Apply the Doppler equation to calculate the frequency in the same frame as the velocity.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    velocity : Iterable[float]
        Velocity (km/s) (length N)

    Returns
    -------
    Iterable[float]
        Radio-defined Doppler frequency (shape C x N)
    """
    return mol_data["freq"][:, None] * (1.0 - velocity / _C)


def calc_fwhm_freq(
    mol_data: dict,
    fwhm: Iterable[float],
) -> Iterable[float]:
    """Calculate the FWHM line width in frequency units

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    fwhm : Iterable[float]
        FWHM line width (km/s) in velocity units (length C)

    Returns
    -------
    Iterable[float]
        FWHM line width (MHz) in frequency units (shape C x N)
    """
    return mol_data["freq"][:, None] * fwhm / _C


def calc_thermal_fwhm(kinetic_temp: float, weight: float) -> float:
    """Calculate the thermal line broadening assuming a Maxwellian velocity distribution
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)
    weight : float
        Molecular weight (number of protons)

    Returns
    -------
    float
        Thermal FWHM line width (km/s)
    """
    # constant = sqrt(8*ln(2)*k_B/m_p)
    const = 0.21394418  # km/s K-1/2
    return const * pt.sqrt(kinetic_temp / weight)


def calc_nonthermal_fwhm(depth: float, nth_fwhm_1pc: float, depth_nth_fwhm_power: float) -> float:
    """Calculate the non-thermal line broadening assuming a power-law size-linewidth relationship.

    Parameters
    ----------
    depth : float
        Line-of-sight depth (pc)
    nth_fwhm_1pc : float
        Non-thermal broadening at 1 pc (km s-1)
    depth_nth_fwhm_power : float
        Power law index

    Returns
    -------
    float
        Non-thermal FWHM line width (km/s)
    """
    return nth_fwhm_1pc * depth**depth_nth_fwhm_power


def calc_log_boltz_factor(freq: float, Tex: float) -> float:
    """Evaluate the Boltzmann factor from a given excitation temperature.

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    Tex : float
        Excitation temperature (K)

    Returns
    -------
    float
        log Boltzmann factor = -h*freq/(k*Tex)
    """
    return -_H * freq / (_K_B * Tex)


def calc_Tex(freq: float, log_boltz_factor: float) -> float:
    """Evaluate the excitation temperature from a given Boltzmann factor.

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    log_boltz_factor : float
        log Boltzmann factor = -h*freq/(k*Tex)

    Returns
    -------
    float
        Excitation temperature
    """
    return -_H * freq / (_K_B * log_boltz_factor)


def calc_pseudo_voigt(
    freq_axis: Iterable[float], frequency: Iterable[float], fwhm: Iterable[float], fwhm_L: float
) -> Iterable[float]:
    """Evaluate a pseudo Voight profile in order to aid in posterior exploration
    of the parameter space. This parameterization includes a latent variable fwhm_L, which
    can be conditioned on zero to analyze the posterior. We also consider the spectral
    channelization. We do not perform a full boxcar convolution, rather
    we approximate the convolution by assuming an equivalent FWHM for the
    boxcar kernel of 4 ln(2) / pi * channel_width ~= 0.88 * channel_width

    Parameters
    ----------
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    frequency : Iterable[float]
        Cloud center frequency (MHz length C x N)
    fwhm : Iterable[float]
        Cloud FWHM line widths (MHz length C x N)
    fwhm_L : float
        Latent pseudo-Voigt profile Lorentzian FWHM (km/s)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x C x N)
    """
    channel_size = pt.abs(freq_axis[1] - freq_axis[0])
    channel_fwhm = 4.0 * pt.log(2.0) * channel_size / np.pi
    fwhm_conv = pt.sqrt(fwhm**2.0 + channel_fwhm**2.0 + fwhm_L**2.0)
    fwhm_L_frac = fwhm_L / fwhm_conv
    eta = 1.36603 * fwhm_L_frac - 0.47719 * fwhm_L_frac**2.0 + 0.11116 * fwhm_L_frac**3.0

    # gaussian component
    gauss_part = gaussian(freq_axis[:, None, None], frequency, fwhm_conv)

    # lorentzian component
    lorentz_part = lorentzian(freq_axis[:, None, None], frequency, fwhm_conv)

    # linear combination
    return eta * lorentz_part + (1.0 - eta) * gauss_part


def calc_optical_depth(
    tau_total: Iterable[float],
    tau_weight: Iterable[float],
) -> Iterable[float]:
    """Evaluate the relative optical depth contribution of each transition.

    Parameters
    ----------
    tau_total : Iterable[float]
        Total optical depth (length N)
    tau_weight : Iterable[float]
        Optical depth weight per component (length C x N)

    Returns
    -------
    Iterable[float]
        Optical depth of each component (shape C x N)
    """
    return tau_total * tau_weight


def predict_tau(
    mol_data: dict,
    freq_axis: Iterable[float],
    tau_total: Iterable[float],
    velocity: Iterable[float],
    fwhm: Iterable[float],
    tau_weight: Iterable[float],
    fwhm_L: float,
) -> Iterable[float]:
    """Predict the optical depth spectra from model parameters.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    tau_total : Iterable[float]
        Total optical depth (length N)
    velocity : Iterable[float]
        Velocity (km s-1) (length N)
    fwhm : Iterable[float]
        FWHM line width (km s-1) (length N)
    tau_weight : Iterable[float]
        Optical depth weight per component (length C x N)
    fwhm_L : float
        Latent pseudo-Voigt profile Lorentzian FWHM (km/s)

    Returns
    -------
    Iterable[float]
        Predicted optical depth spectra (shape S x C x N)
    """
    # Frequency (MHz; shape: components, clouds)
    frequency = calc_frequency(mol_data, velocity)

    # Total FWHM line width in frequency units (MHz; shape: components, clouds)
    fwhm_freq = calc_fwhm_freq(mol_data, fwhm)

    # Line profile (MHz-1; shape: spectral, components, clouds)
    line_profile = calc_pseudo_voigt(freq_axis, frequency, fwhm_freq, fwhm_L)

    # Optical depth contributions (shape: components, clouds)
    tau = calc_optical_depth(tau_total, tau_weight)

    # Integrate over channels
    channel_size = pt.abs(freq_axis[1] - freq_axis[0])
    return tau * line_profile * channel_size


def rj_temperature(freq: float, temp: float):
    """Calculate the Rayleigh-Jeans equivalent temperature (AKA the brightness temperature)

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    temp : float
        Temperature (K)

    Returns
    -------
    float
        R-J equivalent temperature (K)
    """
    const = _H * freq / _K_B
    return const / (pt.exp(const / temp) - 1.0)


def radiative_transfer(
    freq_axis: Iterable[float],
    tau: Iterable[float],
    Tex: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest].

    Parameters
    ----------
    freq_axis : float
        Frequency axis (MHz) (length S)
    tau : Iterable[float]
        Optical depth spectra (shape S x C x N)
    tex : Iterable[float]
        Component/cloud excitation temperatures (K) (shape C x N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K) (length S)
    """
    front_tau = pt.zeros_like(tau[:, 0, 0:1])
    # sum over components and cumsum over clouds
    sum_tau = pt.concatenate([front_tau, pt.cumsum(tau.sum(axis=1), axis=1)], axis=1)

    # radiative transfer, assuming filling factor = 1.0
    emission_bg = rj_temperature(freq_axis, bg_temp)
    emission_bg_attenuated = emission_bg * pt.exp(-sum_tau[:, -1])
    emission_components_clouds = rj_temperature(freq_axis[:, None, None], Tex) * (1.0 - pt.exp(-tau))
    # sum over components
    emission_clouds = emission_components_clouds.sum(axis=1)
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - emission_bg
