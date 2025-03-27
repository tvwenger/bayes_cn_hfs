"""
cn_model.py
CNModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from bayes_cn_hfs.utils import supplement_mol_data
from bayes_cn_hfs import physics


class CNModel(BaseModel):
    """Definition of the CNModel."""

    def __init__(
        self,
        *args,
        molecule: str = "CN",
        mol_data: Optional[dict] = None,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        molecule : str, optional
            Either "CN" or "13CN"
        mol_data : Optional[dict], optional
            Molecular data dictionary returned by get_molecule_data(). If None, it will
            be downloaded. Default is None
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        Beff : float, optional
            Beam efficiency, by default 1.0
        Feff : float, optional
            Forward efficiency, by default 1.0
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.molecule = molecule
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff

        # Get molecular data
        self.mol_data, self.mol_weight = supplement_mol_data(
            molecule, mol_data=mol_data
        )

        # Add transitions and states to model
        coords = {
            "transition": self.mol_data["freq"],
            "state": self.mol_data["states"]["state"],
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "log10_N": r"$\log_{10} N_{\rm tot}$ (cm$^{-2}$)",
                "log10_Tex_ul": r"$\log_{10} T_{{\rm ex}, ul}$ (K)",
                "Tex": r"$T_{\rm ex}$ (K)",
                "LTE_precision": r"$1/a_{\rm LTE}$",
                "tau": r"$\tau$",
                "tau_total": r"$\tau_{\rm tot}$",
                "TR": r"$T_R$ (K)",
            }
        )

    def add_priors(
        self,
        prior_log10_N: Iterable[float] = [13.5, 1.0],
        prior_log10_Tkin: Iterable[float] = [1.0, 0.5],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = 0.0,
        prior_fwhm_L: Optional[float] = None,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        assume_LTE: bool = True,
        prior_log10_Tex: Iterable[float] = [1.0, 0.5],
        assume_CTEX: bool = True,
        prior_LTE_precision: float = 100.0,
        fix_log10_Tkin: Optional[float] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N : Iterable[float], optional
            Prior distribution on total column density over all lower and upper states, by default [13.5, 1.0], where
            log10_N ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 cloud kinetic temperature (K), by default [1.0, 0.5], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm_nonthermal : float, optional
            Prior distribution on non-thermal FWHM (km s-1), by default 0.0, where
            fwhm_nonthermal ~ HalfNormal(sigma=prior_fwhm_nonthermal)
            If 0.0, assume no non-thermal broadening.
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the pseudo-Voight Lorentzian profile line width (km/s),
            by default None, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
            If None, the line profile is assumed Gaussian.
        prior_rms : Optional[dict[str, float]], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            Keys are dataset names and values are priors. If None, then the spectral rms is taken
            from dataset.noise and not inferred.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Prior distribution on the normalized baseline polynomial coefficients, by default None.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset.
        assume_LTE : bool, optional
            Assume local thermodynamic equilibrium by fixing the average excitation temperature to the
            cloud kinetic temperature, by default True.
        prior_log10_Tex : Iterable[float], optional
            Prior distribution on log10 average excitation temperature (K), by
            default [1.5, 0.5], where
            log10_Tex ~ Normal(mu=prior[0], sigma=prior[1])
            This parameter has no effect when assume_LTE = True. Otherwise, it characterizes either
            the CTEX excitation temperature (if assume_CTEX = True) or the "average" excitation temperature
            over all states (if assume_CTEX = False).
        assume_CTEX : bool, optional
            Assume that every transition has the same excitation temperature, by default True.
        prior_LTE_precision : float, optional
            Prior distribution on the state column density departures from LTE, by default 100.0, where
            LTE_precision ~ Gamma(alpha=1.0, beta=prior)
            stat_weights ~ Dirichlet(a=LTE_stat_weights/LTE_precision)
        fix_log10_Tkin : Optional[float], optional
            Fix the log10 cloud kinetic temperature at this value (K), by default None.
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            velocity(cloud = n) ~
                prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        if assume_LTE and not assume_CTEX:
            raise ValueError("Can't assume LTE and not CTEX")

        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # Velocity (km/s; shape: clouds)
            if ordered:
                velocity_offset_norm = pm.Gamma(
                    "velocity_norm", alpha=2.0, beta=1.0, dims="cloud"
                )
                velocity_offset = velocity_offset_norm * prior_velocity[1]
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + pm.math.cumsum(velocity_offset),
                    dims="cloud",
                )
            else:
                velocity_norm = pm.Normal(
                    "velocity_norm",
                    mu=0.0,
                    sigma=1.0,
                    dims="cloud",
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # kinetic temperature (K; shape: clouds)
            if fix_log10_Tkin:
                # log10_Tkin is fixed
                log10_Tkin = pm.Data(
                    "log10_Tkin", np.ones(self.n_clouds) * fix_log10_Tkin, dims="cloud"
                )
            else:
                log10_Tkin_norm = pm.Normal(
                    "log10_Tkin_norm", mu=0.0, sigma=1.0, dims="cloud"
                )
                log10_Tkin = pm.Deterministic(
                    "log10_Tkin",
                    prior_log10_Tkin[0] + prior_log10_Tkin[1] * log10_Tkin_norm,
                    dims="cloud",
                )

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic(
                "fwhm_thermal",
                physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight),
                dims="cloud",
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = 0.0
            if prior_fwhm_nonthermal > 0:
                fwhm_nonthermal_norm = pm.HalfNormal(
                    "fwhm_nonthermal_norm", sigma=1.0, dims="cloud"
                )
                fwhm_nonthermal = pm.Deterministic(
                    "fwhm_nonthermal",
                    prior_fwhm_nonthermal * fwhm_nonthermal_norm,
                    dims="cloud",
                )

            # Total (physical) FWHM (km/s; shape: clouds)
            _ = pm.Deterministic(
                "fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud"
            )

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

            # Spectral rms (K)
            if prior_rms is not None:
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms[label])

            # Total column density (cm-2; shape: clouds)
            log10_N_norm = pm.Normal("log10_N_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_N = pm.Deterministic(
                "log10_N",
                prior_log10_N[0] + prior_log10_N[1] * log10_N_norm,
                dims="cloud",
            )
            N_tot = 10.0**log10_N

            if assume_LTE:
                # Upper-lower excitation temperature is fixed at kinetic temperature (K; shape: clouds)
                log10_Tex_ul = pm.Deterministic(
                    "log10_Tex_ul", log10_Tkin, dims="cloud"
                )
            else:
                # Upper->lower excitation temperature (K; shape: clouds)
                log10_Tex_ul_norm = pm.Normal(
                    "log10_Tex_ul_norm", mu=0.0, sigma=1.0, dims="cloud"
                )
                log10_Tex_ul = pm.Deterministic(
                    "log10_Tex_ul",
                    prior_log10_Tex[0] + prior_log10_Tex[1] * log10_Tex_ul_norm,
                    dims="cloud",
                )
            Tex_ul = 10.0**log10_Tex_ul

            # LTE statistical weights (shape: clouds, states)
            LTE_weights = physics.calc_stat_weight(
                self.mol_data["states"]["deg"][None, :],
                self.mol_data["states"]["E"][None, :],
                Tex_ul[:, None],
            )
            LTE_weights = LTE_weights / pt.sum(LTE_weights, axis=1)[:, None]

            if assume_CTEX:
                # constant across transitions (K; shape: transitions, clouds)
                Tex = pm.Deterministic(
                    "Tex",
                    pt.repeat(Tex_ul[None, :], len(self.mol_data["freq"]), 0),
                    dims=["transition", "cloud"],
                )

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor = physics.calc_boltz_factor(
                    self.mol_data["freq"][:, None], Tex
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state = N_tot[:, None] * LTE_weights

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu = pt.stack([N_state[:, idx] for idx in self.mol_data["state_u_idx"]])
                Nl = pt.stack([N_state[:, idx] for idx in self.mol_data["state_l_idx"]])
            else:
                # LTE precision (inverse Dirichlet concentration) (shape: clouds)
                LTE_precision = pm.Gamma(
                    "LTE_precision", alpha=1.0, beta=prior_LTE_precision, dims="cloud"
                )

                # Dirichlet state fraction (shape: cloud, state)
                weights = pm.Dirichlet(
                    "weights",
                    a=LTE_weights / LTE_precision[:, None],
                    dims=["cloud", "state"],
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state = N_tot[:, None] * weights

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu = pt.stack([N_state[:, idx] for idx in self.mol_data["state_u_idx"]])
                Nl = pt.stack([N_state[:, idx] for idx in self.mol_data["state_l_idx"]])

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor = (
                    Nu
                    * self.mol_data["Gl"][:, None]
                    / (Nl * self.mol_data["Gu"][:, None])
                )

                # Excitation temperature (shape: transition, cloud)
                Tex = pm.Deterministic(
                    "Tex",
                    physics.calc_Tex(self.mol_data["freq"][:, None], boltz_factor),
                    dims=["transition", "cloud"],
                )

            # Optical depth (shape: transitions, clouds)
            tau = pm.Deterministic(
                "tau",
                physics.calc_optical_depth(
                    self.mol_data["freq"][:, None],
                    self.mol_data["Gl"][:, None],
                    self.mol_data["Gu"][:, None],
                    Nl,
                    Nu,
                    self.mol_data["Aul"][:, None],
                    1.0,  # integrated line profile
                ),
                dims=["transition", "cloud"],
            )

            # Total optical depth (shape: clouds)
            _ = pm.Deterministic("tau_total", pt.sum(tau, axis=0), dims="cloud")

            # Radiation temperature (K; shape: transitions, clouds)
            TR = pm.Deterministic(
                "TR",
                physics.calc_TR(self.mol_data["freq"][:, None], boltz_factor),
                dims=["transition", "cloud"],
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            # Optical depth spectra (shape: spectral, transitions, clouds)
            tau_spectra = physics.predict_tau_spectra(
                self.mol_data,
                dataset.spectral,
                self.model["tau"],
                self.model["velocity"],
                self.model["fwhm"],
                self.model["fwhm_L"] if "fwhm_L" in self.model else 0.0,
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau_spectra,
                    self.model["TR"],
                    self.bg_temp,
                )
            )

            # Add baseline model
            predicted = predicted_line + baseline_models[label]

            with self.model:
                sigma = dataset.noise
                if f"rms_{label}" in self.model:
                    sigma = self.model[f"rms_{label}"]

                # Evaluate likelihood
                obs = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=sigma,
                    observed=dataset.brightness,
                )
