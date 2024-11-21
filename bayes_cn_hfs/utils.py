"""
utils.py
Hyperfine utilities

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Optional

import numpy as np
from numpy.polynomial import Polynomial

import astropy.units as u
from astroquery.jplspec import JPLSpec


def get_molecule_data(
    molecule: str,
    fmin: float = 0.0,
    fmax: float = 10000.0,
    vibrational_state: Optional[int] = None,
    rot_state_lower: Optional[int] = None,
) -> dict:
    """Get molecular transition data from the JPL database. Lifted heavily from
    pyspeckit's get_molecular_parameters.

    Parameters
    ----------
    molecule : str
        Molecule name
    fmin : float, optional
        Minimum frequency (GHz), by default 0.0
    fmax : float, optional
        Maximum frequency (GHz), by default 10000.0
    vibrational_state : Optional[int], optional
        For CN, limit to this vibrational state, which is assumed to be the second quantum
        number returned by JPLSpec, by default None
    rot_state_lower : Optional[int], optional
        For CN or 13CN, limit to this lower rotational state, which is assumed
        to be the first quantum number returned by JPLSpec, by default None

    Returns
    -------
    dict
        Molecular transition data, with keys:
        "freq" (Iterable[float]) : Rest frequencies (MHz)
        "Aul" (Iterable[float]) : Einstein A coefficients (s-1)
        "degu" (Iterable[float]) : Upper state degeneracies
        "Eu" (Iterable[float]) : Upper state energies (erg)
        "relative_int" (Iterable[float]) : Relative intensities
        "log10_Q_terms" (Iterable[float]) : Polynomial coefficients for logQ vs. logT (K)

    Raises
    ------
    ValueError
        Molecule not found in JPLSpec database
    """
    # get meta data
    species = JPLSpec.get_species_table()
    if molecule not in species["NAME"]:
        raise ValueError(f"{molecule} not found in database")
    data = species[species["NAME"] == molecule][0]

    # partition function linear fit parameters
    log10_temps = np.log10(data.meta["Temperature (K)"])
    log10_Q = np.array([data[f"QLOG{i}"] for i in range(1, 8)])
    log10_Q_fit = Polynomial.fit(log10_temps, log10_Q, 1).convert()
    log10_Q_terms = log10_Q_fit.coef

    # get transition data
    output = JPLSpec.query_lines(
        min_frequency=fmin * u.GHz,
        max_frequency=fmax * u.GHz,
        molecule=f"{data['TAG']} {molecule}",
    )

    # limit states
    if vibrational_state is not None:
        output_vib_state = np.array([int(output["QN'"][i][2:4]) for i in range(len(output))])
        good = output_vib_state == vibrational_state
        output = output[good]
    if rot_state_lower is not None:
        output_rot_state = np.array([int(output['QN"'][i][0:2]) for i in range(len(output))])
        good = output_rot_state == rot_state_lower
        output = output[good]

    # rest frequencies, degeneracies, and upper energy levels
    freqs = output["FREQ"].quantity
    freq_MHz = freqs.to(u.MHz).value
    deg = np.array(output["GUP"])
    EL = output["ELO"].quantity.to(u.erg, u.spectral())
    dE = freqs.to(u.erg, u.spectral())
    EU = EL + dE

    # need elower, eupper in inverse centimeter units
    elower_icm = output["ELO"].quantity.to(u.cm**-1).value
    eupper_icm = elower_icm + (freqs.to(u.cm**-1, u.spectral()).value)

    # from Brett McGuire
    # https://github.com/bmcguir2/simulate_lte/blob/1f3f7c666946bc88c8d83c53389556a4c75c2bbd/simulate_lte.py#L2580-L2587

    # LGINT: Base 10 logarithm of the integrated intensity in units of nm2 MHz at 300 K.
    # (See Section 3 for conversions to other units.)
    # see also https://cdms.astro.uni-koeln.de/classic/predictions/description.html#description
    CT = 300.0
    logint = np.array(output["LGINT"])  # this should just be a number
    # from CDMS website
    sijmu = (
        (np.exp(np.float64(-(elower_icm / 0.695) / CT)) - np.exp(np.float64(-(eupper_icm / 0.695) / CT))) ** (-1)
        * ((10**logint) / freq_MHz)
        * (24025.120666)
        * 10.0 ** log10_Q_terms[0]
        * CT ** log10_Q_terms[1]
    )

    # aij formula from CDMS.  Verfied it matched spalatalogue's values
    aij = 1.16395 * 10 ** (-20) * freq_MHz**3 * sijmu / deg

    # relative intensity
    relative_int = 10.0**logint
    relative_int = relative_int / relative_int.sum()

    return {
        "freq": freqs.to(u.MHz).value,  # rest frequency (MHz)
        "Aul": aij,  # Einstein A (s-1)
        "degu": deg,  # upper state degeneracy
        "Eu": EU.to(u.erg).value,  # upper level energy (erg)
        "relative_int": relative_int,  # relative intensity
        "log10_Q_terms": log10_Q_terms,  # partition function linear fit coefficients
    }
