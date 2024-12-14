"""
hfs_model.py
HFSModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from bayes_cn_hfs import physics


class HFSModel(BaseModel):
    """Definition of the HFS model. SpecData key must be "observation"."""

    def __init__(self, *args, mol_weight: float = 0.0, mol_data: dict = None, bg_temp: float = 2.7, **kwargs):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        mol_weight : float
            Molecular weight (number of protons)
        mol_data : dict, optional
            Dictionary of molecular line data output from utils.get_molecule_data, by default None
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.mol_weight = mol_weight
        self.mol_data = mol_data.copy()
        self.bg_temp = bg_temp

        """
        # Drop un-observed components
        drop = np.ones(len(self.mol_data["freq"]), dtype=bool)
        for dataset in self.data.values():
            for i, freq in enumerate(self.mol_data["freq"]):
                if dataset.spectral.min() <= freq <= dataset.spectral.max():
                    drop[i] = False
        if self.verbose:
            print("Dropping the un-observed transitions at the following frequencies (MHz):")
            print(self.mol_data["freq"][drop])
        for key in ["freq", "Aul", "degu", "Eu", "relative_int"]:
            self.mol_data[key] = self.mol_data[key][~drop]
        """

        # Add components to model
        coords = {
            "component": self.mol_data["freq"],
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "tau_total",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "tau_total": r"$\tau_0$",
                "log10_anomaly": r"$\log_{10} a_i$",
                "tau_weight": r"$\tau_i/\tau_0$",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "Tex": r"$T_{\rm ex}$ (K)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "rms_observation": r"rms (K)",
                "fwhm_L": r"$\Delta V_{L}$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_tau_total: float = 1.0,
        prior_log10_Tkin: Iterable[float] = [1.0, 0.2],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = 0.1,
        prior_fwhm_L: float = 1.0,
        prior_rms: Optional[float] = None,
        prior_baseline_coeffs: Optional[Iterable[float]] = None,
        fix_tau_total: bool = False,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_tau_total : float, optional
            Prior distribution on total optical depth, by default 1.0, where
            tau_total ~ HalfNormal(sigma=prior)
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 kinetic temperature (K), by default [2.0, 1.0], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm_nonthermal : float, optional
            Prior distribution on non-thermal FWHM (km s-1), by default None, where
            fwhm_nonthermal ~ HalfNormal(sigma=prior_fwhm_nonthermal)
            If None, assume no non-thermal broadening.
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the latent pseudo-Voight Lorentzian profile line width (km/s),
            by default 1.0, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
        prior_rms : Optional[float], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            If None, then the spectral rms is taken from the data and not inferred.
        prior_baseline_coeffs : Optional[Iterable[float]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Must be a list of length `baseline_degree+1`. If None, use `[1.0]*(baseline_degree+1)`,
            by default None
        fix_tau_total : bool, optional
            If True, fix the total optical depth at prior_tau_total. Otherwise, tau_total is a free
            parameter. Set fix_tau_total = True when the excitation temperature cannot be reliably
            estimated from the line width (i.e., when non-thermal broadening is important, or when
            the channels are wide compared to the line width, or in non-LTE when Tkin != Tex.)
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            velocity(cloud = n) ~
                prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        if prior_baseline_coeffs is not None:
            prior_baseline_coeffs = {"observation": prior_baseline_coeffs}

        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # total optical depth (shape: clouds)
            if fix_tau_total:
                _ = pm.Data("tau_total", np.ones(self.n_clouds) * prior_tau_total, dims="cloud")
            else:
                tau_total_norm = pm.HalfNormal("tau_total_norm", sigma=1.0, dims="cloud")
                _ = pm.Deterministic("tau_total", prior_tau_total * tau_total_norm, dims="cloud")

            # kinetic temperature (K; shape: clouds)
            log10_Tkin_norm = pm.Normal("log10_Tkin_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_Tkin = pm.Deterministic(
                "log10_Tkin",
                prior_log10_Tkin[0] + prior_log10_Tkin[1] * log10_Tkin_norm,
                dims="cloud",
            )

            # Velocity (km/s; shape: clouds)
            if ordered:
                velocity_offset_norm = pm.Gamma("velocity_norm", alpha=2.0, beta=1.0, dims="cloud")
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

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic(
                "fwhm_thermal", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight), dims="cloud"
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = 0.0
            if prior_fwhm_nonthermal is not None:
                fwhm_nonthermal_norm = pm.HalfNormal("fwhm_nonthermal_norm", sigma=1.0, dims="cloud")
                fwhm_nonthermal = pm.Deterministic(
                    "fwhm_nonthermal", prior_fwhm_nonthermal * fwhm_nonthermal_norm, dims="cloud"
                )

            # Spectral rms (K)
            if prior_rms is not None:
                rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
                _ = pm.Deterministic("rms_observation", rms_observation_norm * prior_rms)

            # Total (physical) FWHM (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
            _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict optical depth spectra (shape: spectral, components, clouds)
        tau = physics.predict_tau(
            self.mol_data,
            self.data["observation"].spectral,
            self.model["tau_total"],
            self.model["velocity"],
            self.model["fwhm"],
            self.model["tau_weight"],
            self.model["fwhm_L"],
        )

        # Radiative transfer (shape: spectral)
        predicted_line = physics.radiative_transfer(
            self.data["observation"].spectral,
            tau,
            self.model["Tex"],
            self.bg_temp,
        )

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["observation"]

        with self.model:
            # Evaluate likelihood
            sigma = self.model["rms_observation"] if "rms_observation" in self.model else self.data["observation"].noise
            _ = pm.Normal(
                "observation",
                mu=predicted,
                sigma=sigma,
                observed=self.data["observation"].brightness,
            )
