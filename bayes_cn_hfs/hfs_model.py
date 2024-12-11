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

from bayes_spec import BaseModel

from bayes_cn_hfs import physics


class HFSModel(BaseModel):
    """Definition of the HFS model. SpecData key must be "observation"."""

    def __init__(
        self,
        *args,
        mol_weight: float = 0.0,
        mol_data: dict = None,
        bg_temp: float = 2.7,
        **kwargs,
    ):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        *args : Additional arguments passed to BaseModel
        mol_weight : float
            Molecular weight (number of protons)
        mol_data : dict
            Dictionary of molecular line data output from utils.get_molecule_data, by default None
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        **kwargs : Additional arguments passed to BaseModel
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.mol_weight = mol_weight
        self.mol_data = mol_data
        self.bg_temp = bg_temp

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N": r"log$_{10}$ $N$ (cm$^{-2}$)",
                "log10_tex": r"log$_{10}$ $T_{\rm ex}$ (K)",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "log10_nth_fwhm_1pc": r"log$_{10}$ $\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
                "depth_nth_fwhm_power": r"$\alpha$",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_log10_N: Iterable[float] = [14.0, 1.0],
        prior_log10_tex: Iterable[float] = [1.0, 0.1],
        prior_log10_depth: Iterable[float] = [0.0, 0.25],
        prior_log10_Tkin: Iterable[float] = [2.0, 1.0],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_log10_nth_fwhm_1pc: Iterable[float] = [0.2, 0.1],
        prior_depth_nth_fwhm_power: Iterable[float] = [0.4, 0.1],
        prior_baseline_coeffs: Optional[Iterable[float]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N : Iterable[float], optional
            Prior distribution on log10 total column density (cm-2), by default [14.0, 1.0], where
            log10_N ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_tex : Iterable[float], optional
            Prior distribution on log10 excitation temperature (K), by default [1.0, 0.1], where
            log10_tex ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_depth : Iterable[float], optional
            Prior distribution on log10 depth (pc), by default [0.0, 0.25], where
            log10_depth ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 kinetic temperature (K), by default [2.0, 1.0], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_nth_fwhm_1pc : Iterable[float], optional
            Prior distribution on non-thermal line width at 1 pc, by default [0.2, 0.1], where
            log10_nth_fwhm_1pc ~ Normal(mu=prior[0], sigma=prior[1])
        prior_depth_nth_fwhm_power : Iterable[float], optional
            Prior distribution on depth vs. non-thermal line width power law index, by default [0.4, 0.1], where
            depth_nth_fwhm_power ~ Normal(mu=prior[0], sigma=prior[1])
        prior_baseline_coeffs : Optional[Iterable[float]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Must be a list of length `baseline_degree+1`. If None, use `[1.0]*(baseline_degree+1)`,
            by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        if prior_baseline_coeffs is not None:
            prior_baseline_coeffs = {"observation": prior_baseline_coeffs}

        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # column density (cm-2; shape: clouds)
            log10_N_norm = pm.Normal("log10_N_norm", mu=0.0, sigma=1.0, dims="cloud")
            _ = pm.Deterministic(
                "log10_N",
                prior_log10_N[0] + prior_log10_N[1] * log10_N_norm,
                dims="cloud",
            )

            # gas excitation temperature (K; shape: clouds)
            log10_tex_norm = pm.Normal(
                "log10_tex_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_tex",
                prior_log10_tex[0] + prior_log10_tex[1] * log10_tex_norm,
                dims="cloud",
            )

            # depth (pc; shape: clouds)
            log10_depth_norm = pm.Normal(
                "log10_depth_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_depth = pm.Deterministic(
                "log10_depth",
                prior_log10_depth[0] + prior_log10_depth[1] * log10_depth_norm,
                dims="cloud",
            )

            # kinetic temperature (K; shape: clouds)
            log10_Tkin_norm = pm.Normal(
                "log10_Tkin_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_Tkin = pm.Deterministic(
                "log10_Tkin",
                prior_log10_Tkin[0] + prior_log10_Tkin[1] * log10_Tkin_norm,
                dims="cloud",
            )

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

            # Non-thermal FWHM at 1 pc (km s-1; shape: clouds)
            log10_nth_fwhm_1pc_norm = pm.Normal(
                "log10_nth_fwhm_1pc_norm", mu=0.0, sigma=1.0
            )
            log10_nth_fwhm_1pc = pm.Deterministic(
                "log10_nth_fwhm_1pc",
                prior_log10_nth_fwhm_1pc[0]
                + prior_log10_nth_fwhm_1pc[1] * log10_nth_fwhm_1pc_norm,
            )

            # Non-thermal FWHM vs. depth power law index (shape: clouds)
            depth_nth_fwhm_power_norm = pm.Normal(
                "depth_nth_fwhm_power_norm", mu=0.0, sigma=1.0
            )
            depth_nth_fwhm_power = pm.Deterministic(
                "depth_nth_fwhm_power",
                prior_depth_nth_fwhm_power[0]
                + prior_depth_nth_fwhm_power[1] * depth_nth_fwhm_power_norm,
            )

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic(
                "fwhm_thermal",
                physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight),
                dims="cloud",
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = pm.Deterministic(
                "fwhm_nonthermal",
                physics.calc_nonthermal_fwhm(
                    10.0**log10_depth, 10.0**log10_nth_fwhm_1pc, depth_nth_fwhm_power
                ),
                dims="cloud",
            )

            # FWHM (km/s; shape: clouds)
            _ = pm.Deterministic(
                "fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud"
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict optical depth spectrum (shape: spectral, components, clouds)
        cloud_tex = 10.0 ** self.model["log10_tex"]
        component_tex = cloud_tex[None, :]
        tau = physics.predict_tau(
            self.mol_data,
            self.data["observation"].spectral,
            self.model["log10_N"],
            self.model["velocity"],
            self.model["fwhm"],
            cloud_tex,
            component_tex,
        )

        # Radiative transfer (shape: spectral)
        predicted_line = physics.radiative_transfer(
            self.data["observation"].spectral,
            tau,
            component_tex,
            self.bg_temp,
        )

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["observation"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "observation",
                mu=predicted,
                sigma=self.data["observation"].noise,
                observed=self.data["observation"].brightness,
            )
