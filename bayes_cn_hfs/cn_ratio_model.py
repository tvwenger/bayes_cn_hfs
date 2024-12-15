"""
cn_ratio_model.py
CNRatioModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bayes_spec import BaseModel

from bayes_cn_hfs import physics
from bayes_cn_hfs.utils import get_molecule_data


class CNRatioModel(BaseModel):
    """Definition of the CNRatioModel. SpecData keys must include the strings "12CN" and "13CN"."""

    def __init__(
        self,
        *args,
        mol_data_12CN: Optional[dict] = None,
        mol_data_13CN: Optional[dict] = None,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize CNRatioModel isntance.

        Parameters
        ----------
        mol_data_12CN : Optional[dict], optional
            12CN molecular data dictionary returned by get_molecule_data(). If None, it will
            be downloaded. Default is None
        mol_data_13CN : Optional[dict], optional
            13CN molecular data dictionary returned by get_molecule_data(). If None, it will
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

        # check datasets
        found_12CN = False
        found_13CN = False
        for label in self.data.keys():
            found_12CN = found_12CN or "12CN" in label
            found_13CN = found_13CN or "13CN" in label
        if not found_12CN and not found_13CN:
            raise ValueError("dataset labels must contain '12CN' and '13CN'")

        # Save inputs
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff
        if mol_data_12CN is None:
            self.mol_data_12CN = get_molecule_data(
                "CN, v = 0, 1",  # molecule name in JPLSpec
                vibrational_state=0,  # vibrational state number
                rot_state_lower=0,  # lower rotational state
            )
        else:
            self.mol_data_12CN = mol_data_12CN.copy()
        self.mol_weight_12CN = 12.0 + 14.0
        if mol_data_13CN is None:
            self.mol_data_13CN = get_molecule_data(
                "C-13-N",  # molecule name in JPLSpec
                rot_state_lower=0,  # lower rotational state
            )
        else:
            self.mol_data_13CN = mol_data_13CN.copy()
        self.mol_weight_13CN = 13.0 + 14.0

        # Add components to model
        coords = {
            "component_12CN": self.mol_data_12CN["freq"],
            "component_13CN": self.mol_data_13CN["freq"],
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "fwhm_12CN",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "tau_total_12CN": r"$\tau_{0, \rm CN}$",
                "tau_total_13CN": r"$\tau_{0, ^{13}\rm CN}$",
                "ratio_13C_12C": r"$^{13}{\rm C}/^{12}{\rm C}$",
                "log10_anomaly_12CN": r"$\log_{10} a_{i, \rm CN}$",
                "tau_weight_12CN": r"$\tau_{i, \rm CN}/\tau_{0, \rm CN}$",
                "tau_weight_13CN": r"$\tau_{i, ^{13}\rm CN}/\tau_{0, \rm CN}$",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "Tex_12CN": r"$T_{\rm ex, CN}$ (K)",
                "Tex_13CN": r"$T_{\rm ex, $^{13}$CN}$ (K)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "fwhm_thermal_12CN": r"$\Delta V_{\rm th, CN}$ (km s$^{-1}$)",
                "fwhm_thermal_13CN": r"$\Delta V_{\rm th, $^{13}$CN}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm_12CN": r"$\Delta V_{\rm CN}$ (km s$^{-1}$)",
                "fwhm_13CN": r"$\Delta V_{^{13} CN}$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_{L}$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_tau_total_12CN: float = 1.0,
        prior_ratio_13C_12C: float = 0.01,
        prior_log10_Tkin: Iterable[float] = [1.0, 0.2],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = 0.1,
        prior_fwhm_L: float = 1.0,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        fix_tau_total_12CN: Optional[float] = None,
        fix_log10_Tkin: Optional[float] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_tau_total_12CN : float, optional
            Prior distribution on total CN optical depth, by default 1.0, where
            tau_total_12CN ~ HalfNormal(sigma=prior)
        prior_ratio_13C_12C : float, optional
            Prior distribution on 13C/12C ratio, by default 0.01, where
            ratio_13C_12C ~ Gamma(alpha=2, beta=1/prior)
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
        prior_rms : Optional[dict[str, float]], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            Keys are dataset names and values are priors. If None, then the spectral rms is taken
            from dataset.noise and not inferred.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        fix_tau_total_12CN : Optional[float], optional
            If not None, fix the total CN optical depth at this value. Otherwise, tau_total_12CN is a free
            parameter. Set fix_tau_total_12CN or fix_log10_Tkin to true when the excitation temperature cannot be reliably
            estimated from the line width (i.e., when non-thermal broadening is important, or when
            the channels are wide compared to the line width, or in non-LTE when Tkin != Tex.)
        fix_log10_Tkin : Optional[float], optional
            If not None, fix the log10_Tkin at this value. Otherwise, log10_Tkin is a free
            parameter. Set fix_tau_total or fix_log10_Tkin to true when the excitation temperature cannot be reliably
            estimated from the line width (i.e., when non-thermal broadening is important, or when
            the channels are wide compared to the line width, or in non-LTE when Tkin != Tex.)
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            velocity(cloud = n) ~
                prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # total CN optical depth (shape: clouds)
            if fix_tau_total_12CN is not None:
                # tau_total is fixed
                tau_total_12CN = pm.Data("tau_total_12CN", np.ones(self.n_clouds) * fix_tau_total_12CN, dims="cloud")
            else:
                # Also cluster in tau_total
                self._cluster_features += ["tau_total_12CN"]

                tau_total_norm = pm.HalfNormal("tau_total_12CN_norm", sigma=1.0, dims="cloud")
                tau_total_12CN = pm.Deterministic("tau_total_12CN", prior_tau_total_12CN * tau_total_norm, dims="cloud")

            # kinetic temperature (K; shape: clouds)
            if fix_log10_Tkin:
                # log10_Tkin is fixed
                log10_Tkin = pm.Data("log10_Tkin", np.ones(self.n_clouds) * fix_log10_Tkin, dims="cloud")
            else:
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
            fwhm_thermal_12CN = pm.Deterministic(
                "fwhm_thermal_12CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_12CN), dims="cloud"
            )
            fwhm_thermal_13CN = pm.Deterministic(
                "fwhm_thermal_13CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_13CN), dims="cloud"
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
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms)

            # Total (physical) FWHM (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm_12CN", pt.sqrt(fwhm_thermal_12CN**2.0 + fwhm_nonthermal**2.0), dims="cloud")
            _ = pm.Deterministic("fwhm_13CN", pt.sqrt(fwhm_thermal_13CN**2.0 + fwhm_nonthermal**2.0), dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
            _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

            # 13C/12C ratio
            ratio_13C_12C_norm = pm.Gamma("ratio_13C_12C_norm", alpha=2.0, beta=1.0, dims="cloud")
            ratio_13C_12C = pm.Deterministic("ratio_13C_12C", prior_ratio_13C_12C * ratio_13C_12C_norm, dims="cloud")

            # Total 13CN optical depth (shape: clouds)
            _ = pm.Deterministic("tau_total_13CN", tau_total_12CN * ratio_13C_12C, dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "12CN" and "13CN"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            if "12CN" in label:
                tau_total = self.model["tau_total_12CN"]
                fwhm = self.model["fwhm_12CN"]
                tau_weight = self.model["tau_weight_12CN"]
                Tex = self.model["Tex_12CN"]
                mol_data = self.mol_data_12CN
            else:
                tau_total = self.model["tau_total_13CN"]
                fwhm = self.model["fwhm_13CN"]
                tau_weight = self.model["tau_weight_13CN"]
                Tex = self.model["Tex_13CN"]
                mol_data = self.mol_data_13CN

            tau = physics.predict_tau(
                mol_data,
                dataset.spectral,
                tau_total,
                self.model["velocity"],
                fwhm,
                tau_weight,
                self.model["fwhm_L"],
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau,
                    Tex,
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
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=sigma,
                    observed=dataset.brightness,
                )
