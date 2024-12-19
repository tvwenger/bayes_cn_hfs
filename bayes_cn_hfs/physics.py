"""
physics.py
Hyperfine physics

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

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


def detailed_balance(
    mol_data: dict,
    log10_N0: Optional[float] = None,
    log_boltz_factor: Optional[Iterable[float]] = None,
    verbose: bool = False,
):
    """Determine which transitions constrain which states, and optionally apply detailed balance
    to derive state column densities.

    Parameters
    ----------
    mol_data : dict
        Molecular data dictionary returned by supplement_mol_data().
    log10_N0 : Optional[float], optional
        Ground state column density, by default None. If not None, then the return values
        for constraint_l and constraint_u are populated with the column densities of each state.
    log_boltz_factor : Optional[Iterable[float]], optional
        Log Boltzmann factor for each transition, by default None. Only used if log10_N0 is not
        None.
    verbose : bool, optional
        If True, print information

    Returns
    -------
    list
        Free transitions
    list
        The transition that constrains each lower state. Value is -1 for the ground state.
        If log10_N0 is not None, then instead this list contains the column density for
        each state.
    list
        The transition that constrains each upper state. Value is -1 for the ground state.
        If log10_N0 is not None, then instead this list contains the column density for
        each state.
    """
    states_l = list(set(mol_data["Ql"]))
    states_u = list(set(mol_data["Qu"]))

    constraint_l = [None] * len(states_l)
    constraint_u = [None] * len(states_u)

    # Ground state is given
    for state_l, El in zip(mol_data["state_l"], mol_data["El"]):
        if El == 0.0:
            constraint_l[state_l] = -1 if log10_N0 is None else log10_N0

    # Loop over remaining transitions until all states are constrained
    transition_used = [False] * len(mol_data["freq"])
    while None in constraint_l + constraint_u:
        for i, (state_l, state_u) in enumerate(zip(mol_data["state_l"], mol_data["state_u"])):
            # check if either lower or upper state is populated
            if constraint_l[state_l] is not None and constraint_u[state_u] is None:
                constraint_u[state_u] = (
                    i
                    if log10_N0 is None
                    else (
                        log_boltz_factor[i] / pt.log(10.0)
                        + constraint_l[state_l]
                        + pt.log10(mol_data["Gu"][i] / mol_data["Gl"][i])
                    )
                )
                transition_used[i] = True
                if verbose:  # pragma: no cover
                    print(
                        f"Transition {mol_data['freq'][i]} is constraining upper state "
                        + f"{state_u} from lower state {state_l}"
                    )
            elif constraint_l[state_l] is None and constraint_u[state_u] is not None:
                constraint_l[state_l] = (
                    i
                    if log10_N0 is None
                    else (
                        -log_boltz_factor[i] / pt.log(10.0)
                        + constraint_u[state_u]
                        + pt.log10(mol_data["Gl"][i] / mol_data["Gu"][i])
                    )
                )
                transition_used[i] = True
                if verbose:  # pragma: no cover
                    print(
                        f"Transition {mol_data['freq'][i]} is constraining lower state "
                        + f"{state_l} from upper state {state_u}"
                    )
                transition_used[i] = True
    if verbose:  # pragma: no cover
        print(f"{np.sum(transition_used)}/{len(transition_used)} transitions used to constrain state densities")

    transition_free = mol_data["freq"][np.array(transition_used)]
    transition_derived = mol_data["freq"][~np.array(transition_used)]
    if verbose:  # pragma: no cover
        print(f"Free Tex transitions: {transition_free}")
        print(f"Derived Tex transitions: {transition_derived}")

    return transition_free, constraint_l, constraint_u


def calc_frequency(freq: float, velocity: float) -> float:
    """Apply the Doppler equation to calculate the frequency in the same frame as the velocity.

    Parameters
    ----------
    freq : float
        Frequencies. Output has same units.
    velocity : float
        Velocity (km/s)

    Returns
    -------
    float
        Radio-defined Doppler frequency
    """
    return freq * (1.0 - velocity / _C)


def calc_fwhm_freq(freq: float, fwhm: float) -> float:
    """Calculate the FWHM line width in frequency units

    Parameters
    ----------
    freq : float
        Frequencies. Output has same units.
    fwhm : float
        FWHM line width (km/s) in velocity units (length C)

    Returns
    -------
    float
        FWHM line width (MHz) in frequency units (shape C x N)
    """
    return freq * fwhm / _C


def calc_thermal_fwhm(kinetic_temp: float, weight: float) -> float:
    """Calculate the thermal line broadening assuming a Maxwellian velocity distribution
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)
    weight : float
        Molecular weight (number of nucleii)

    Returns
    -------
    float
        Thermal FWHM line width (km/s)
    """
    # constant = sqrt(8*ln(2)*k_B/m_p)
    const = 0.21394418  # km/s K-1/2
    return const * pt.sqrt(kinetic_temp / weight)


def calc_log_boltz_factor(freq: float, Tex: float) -> float:
    """Evaluate the log Boltzmann factor from a given excitation temperature.

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    Tex : float
        Excitation temperature (K)

    Returns
    -------
    float
        log Boltzmann factor = log(gl*Nu) - log(gu*Nl) = -h*freq/(k*Tex)
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
    freq_axis: Iterable[float],
    frequency: Iterable[float],
    fwhm: Iterable[float],
    fwhm_L: Iterable[float],
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
    fwhm_L : Iterable[float]
        Latent pseudo-Voigt profile Lorentzian FWHM (MHz length C)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x C x N)
    """
    channel_size = pt.abs(freq_axis[1] - freq_axis[0])
    channel_fwhm = 4.0 * pt.log(2.0) * channel_size / np.pi
    fwhm_conv = pt.sqrt(fwhm**2.0 + channel_fwhm**2.0 + fwhm_L[:, None] ** 2.0)
    fwhm_L_frac = fwhm_L[:, None] / fwhm_conv
    eta = 1.36603 * fwhm_L_frac - 0.47719 * fwhm_L_frac**2.0 + 0.11116 * fwhm_L_frac**3.0

    # gaussian component
    gauss_part = gaussian(freq_axis[:, None, None], frequency[None, :, :], fwhm_conv[None, :, :])

    # lorentzian component
    lorentz_part = lorentzian(freq_axis[:, None, None], frequency[None, :, :], fwhm_conv[None, :, :])

    # linear combination
    return eta * lorentz_part + (1.0 - eta) * gauss_part


def calc_optical_depth(
    gu: float,
    gl: float,
    Nl: float,
    log_boltz_factor: float,
    line_profile: float,
    freq: float,
    Aul: float,
) -> float:
    """Evaluate the optical depth spectra, from Mangum & Shirley eq. 29
    Pass line_profile = 1.0 to evaluate the total (integrated) optical depth.

    Parameters
    ----------
    gu : float
        Upper state degeneracy (shape C)
    gl : float
        Lower state degeneracy (shape C)
    Nl : float
        Lower state column densities (shape C x N)
    log_boltz_factor : float
        log Boltzmann factor (shape C x N)
    line_profile : float
        Line profile (MHz-1; shape S x C x N)
    freq : float
        Transition frequency (MHz) (shape C)
    Aul : float
        Transition Einstein A coefficient (s-1) (shape C)

    Returns
    -------
    float
        Optical depth spectral (shape S x C x N)
    """
    return (
        _C_CM_MHZ**2.0  # cm2 MHz2
        / (8.0 * np.pi * freq**2.0)  # MHz-2
        * (gu / gl)
        * Aul  # s-1
        * (line_profile / 1e6)  # Hz-1
        * Nl  # cm-2
        * (1.0 - pt.exp(log_boltz_factor))
    )


def predict_tau(
    mol_data: dict,
    freq_axis: Iterable[float],
    Nl: Iterable[float],
    log_boltz_factor: Iterable[float],
    velocity: Iterable[float],
    fwhm: Iterable[float],
    fwhm_L: float,
) -> Iterable[float]:
    """Predict the optical depth spectra from model parameters.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of transition frequencies
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    Nl : Iterable[float]
        Lower state column densities (shape C x N)
    log_boltz_factor : Iterable[float]
        log Boltzmann factor (shape C x N)
    velocity : Iterable[float]
        Velocity (km s-1) (length N)
    fwhm : Iterable[float]
        FWHM line width (km s-1) (length N)
    fwhm_L : float
        Latent pseudo-Voigt profile Lorentzian FWHM (km s-1)

    Returns
    -------
    Iterable[float]
        Predicted optical depth spectra (shape S x C x N)
    """
    # Frequency (MHz; shape: transitions, clouds)
    frequency = calc_frequency(mol_data["freq"][:, None], velocity)

    # Total FWHM line width in frequency units (MHz; shape: transitions, clouds)
    fwhm_freq = calc_fwhm_freq(mol_data["freq"][:, None], fwhm)

    # Latent Lorentzian FWHM line width in frequency units (MHz; shape: transitions)
    fwhm_L_freq = calc_fwhm_freq(mol_data["freq"], fwhm_L)

    # Line profile (MHz-1; shape: spectral, transitions, clouds)
    line_profile = calc_pseudo_voigt(freq_axis, frequency, fwhm_freq, fwhm_L_freq)

    # Optical depth  (shape: spectral, transitions, clouds)
    tau = calc_optical_depth(
        mol_data["Gu"][None, :, None],
        mol_data["Gl"][None, :, None],
        Nl[None, :, :],
        log_boltz_factor[None, :, :],
        line_profile,
        mol_data["freq"][None, :, None],
        mol_data["Aul"][None, :, None],
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
    Tex : Iterable[float]
        Excitation temperatures (K) (shape C x N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K) (length S)
    """
    front_tau = pt.zeros_like(tau[:, 0, 0:1])
    # sum over transitions and cumsum over clouds
    sum_tau = pt.concatenate([front_tau, pt.cumsum(tau.sum(axis=1), axis=1)], axis=1)

    # radiative transfer, assuming filling factor = 1.0
    emission_bg = rj_temperature(freq_axis, bg_temp)
    emission_bg_attenuated = emission_bg * pt.exp(-sum_tau[:, -1])
    emission_transitions_clouds = rj_temperature(freq_axis[:, None, None], Tex) * (1.0 - pt.exp(-tau))
    # sum over transitions
    emission_clouds = emission_transitions_clouds.sum(axis=1)
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - emission_bg
