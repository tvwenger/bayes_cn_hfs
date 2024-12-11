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

def gaussian(x: float, amp: float, center: float, fwhm2: float) -> float:
    """Evaluate a Gaussian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    amp : float
        Gaussian amplitude
    center : float
        Gaussian centroid
    fwhm2 : float
        Gaussian FWHM^2

    Returns
    -------
    float
        Gaussian evaluated at x
    """
    return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm2)


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


def calc_fwhm2_freq(
    mol_data: dict,
    fwhm2: Iterable[float],
) -> Iterable[float]:
    """Calculate the FWHM^2 line width in frequency units

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    fwhm2 : Iterable[float]
        FWHM^2 line width (km2/s2) in velocity units (length C)

    Returns
    -------
    Iterable[float]
        FWHM^2 line width (MHz2) in frequency units (shape C x N)
    """
    return (mol_data["freq"][:, None] / _C)**2.0 * fwhm2


def calc_thermal_fwhm2(kinetic_temp: float, weight: float) -> float:
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
        Thermal FWHM^2 line width (km2/s2)
    """
    # constant = sqrt(8*ln(2)*k_B/m_p)
    const = 0.21394418  # km/s K-1/2
    return const**2 * kinetic_temp / weight


def calc_nonthermal_fwhm2(depth: float, nth_fwhm_1pc: float, depth_nth_fwhm_power: float) -> float:
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
        Non-thermal FWHM^2 line width (km2/s2)
    """
    return (nth_fwhm_1pc * depth**depth_nth_fwhm_power)**2


def calc_line_profile(freq_axis: Iterable[float], frequency: Iterable[float], fwhm2: Iterable[float]) -> Iterable[float]:
    """Evaluate the Gaussian line profile, ensuring normalization.

    Parameters
    ----------
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    frequency : Iterable[float]
        Cloud center frequency (MHz length C x N)
    fwhm2 : Iterable[float]
        Cloud FWHM^2 line widths (MHz2 length C x N)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x C x N)
    """
    amp = pt.sqrt(4.0 * pt.log(2.0) / (np.pi * fwhm2))
    profile = gaussian(freq_axis[:, None, None], amp, frequency, fwhm2)

    # normalize
    channel_size = pt.abs(freq_axis[1] - freq_axis[0])
    profile_int = pt.sum(profile, axis=0)
    norm = pt.switch(pt.lt(profile_int, 1.0e-6), 1.0, profile_int * channel_size)
    return profile / norm


def detailed_balance(mol_data: dict, tex: Iterable[float]) -> Iterable[float]:
    """Evaluate the state abundance by assuming detailed balance at a constant
    excitation temperature, from Mangum & Shirley eq. 31.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    tex : Iterable[float]
        Excitation temperature (K) (length N)

    Returns
    -------
    Iterable[float]
        Fractional abundance (N_u/N_tot) per component (shape C x N)
    """
    # evaluate partition function from linear fit
    part_func = 10.0 ** mol_data["log10_Q_terms"][0] * tex ** mol_data["log10_Q_terms"][1]

    # detailed balance
    abundance = mol_data["degu"][:, None] / part_func / pt.exp(mol_data["Eu"][:, None] / (_K_B * tex))
    return abundance


def calc_optical_depth(
    mol_data: dict,
    N: Iterable[float],
    cloud_tex: Iterable[float],
    component_tex: Iterable[float],
    line_profile: Iterable[float],
) -> Iterable[float]:
    """Evaluate the optical depth, from Mangum & Shirley eq. 29, assuming a constant
    excitation temperature for the detailed balance calculation, but allowing for
    component-dependent excitation temperatures for the optical depth calculation.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    N : Iterable[float]
        Total column density (cm-2) (length N)
    cloud_tex : Iterable[float]
        Mean cloud excitation tempearture (K) (length N)
    component_tex : Iterable[float]
        Componenent excitation temperatures (K) (shape C x N)
    line_profile : Iterable[float]
        Line profile (MHz-1) (shape S x C x N)

    Returns
    -------
    Iterable[float]
        Optical depth spectra (shape S x C x N)
    """
    # detailed balance to get relative upper state abundances
    abundance = detailed_balance(mol_data, cloud_tex)

    const = _H * mol_data["freq"][None, :, None] / (_K_B * component_tex)
    return (
        _C_CM_MHZ**2.0  # cm2 MHz2
        / (8.0 * np.pi * mol_data["freq"][None, :, None] ** 2.0)  # MHz-2
        * (pt.exp(const) - 1.0)
        * mol_data["Aul"][None, :, None]  # s-1
        * (line_profile / 1e6)  # Hz-1
        * N  # cm-2
        * abundance  # upper state abundance
    )


def predict_tau(
    mol_data: dict,
    freq_axis: Iterable[float],
    log10_N: Iterable[float],
    velocity: Iterable[float],
    fwhm2: Iterable[float],
    cloud_tex: Iterable[float],
    component_tex: Iterable[float],
) -> Iterable[float]:
    """Predict the optical depth spectra from model parameters.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of component frequencies
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    log10_N : Iterable[float]
        log10 total column density (cm-2) (length N)
    velocity : Iterable[float]
        Velocity (km s-1) (length N)
    fwhm2 : Iterable[float]
        FWHM^2 line width (km2/s2) (length N)
    cloud_tex : Iterable[float]
        Mean cloud excitation tempearture (K) (length N)
    component_tex : Iterable[float]
        Componenent excitation temperatures (K) (shape C x N)

    Returns
    -------
    Iterable[float]
        Predicted optical depth spectra (shape S x C x N)
    """
    # Frequency (MHz; shape: components, clouds)
    frequency = calc_frequency(mol_data, velocity)

    # Total FWHM^2 line width in frequency units (MHz2; shape: components, clouds)
    fwhm2_freq = calc_fwhm2_freq(mol_data, fwhm2)

    # Line profile (MHz-1; shape: spectral, components, clouds)
    line_profile = calc_line_profile(freq_axis, frequency, fwhm2_freq)

    # Optical depth spectra (shape: spectral, components, clouds)
    tau = calc_optical_depth(
        mol_data,
        10.0**log10_N,
        cloud_tex,
        component_tex,
        line_profile,
    )
    return tau


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
    tex: Iterable[float],
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
        Componenent excitation temperatures (K) (shape C x N)
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
    emission_components_clouds = rj_temperature(freq_axis[:, None, None], tex) * (1.0 - pt.exp(-tau))
    # sum over components
    emission_clouds = emission_components_clouds.sum(axis=1)
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - emission_bg
