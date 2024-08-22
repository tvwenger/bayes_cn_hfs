"""
hfs_model.py
HFSModel definition

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

from typing import Iterable, Optional

import numpy as np
import pymc as pm

from bayes_spec import BaseModel

from bayes_cn_hfs import physics


class HFSModel(BaseModel):
    """Definition of the HFS model. SpecData key must be "observation"."""

    def __init__(self, *args, mol_data: dict = None, bg_temp: float = 2.7, **kwargs):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        mol_data : dict, optional
            Dictionary of molecular line data output from utils.get_molecule_data, by default None
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.mol_data = mol_data
        self.bg_temp = bg_temp

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N",
            "log10_tex",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N": r"log$_{10}$ $N$ (cm$^{-2}$)",
                "log10_tex": r"log$_{10}$ $T_{\rm ex}$ (K)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "rms_observation": r"rms (K)",
            }
        )

    def add_priors(
        self,
        prior_log10_N: Iterable[float] = [14.0, 1.0],
        prior_log10_tex: Iterable[float] = [1.0, 0.1],
        prior_fwhm: float = 1.0,
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_rms: float = 0.01,
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
        prior_fwhm : float, optional
            Prior distribution on FWHM line width (km s-1), by default 1.0, where
            fwhm ~ Gamma(alpha=2.0, beta=1.0/prior)
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_rms : float, optional
            Prior distribution on spectral rms (K), by default 0.01, where
            rms ~ HalfNormal(sigma=prior)
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
            log10_tex_norm = pm.Normal("log10_tex_norm", mu=0.0, sigma=1.0, dims="cloud")
            _ = pm.Deterministic(
                "log10_tex",
                prior_log10_tex[0] + prior_log10_tex[1] * log10_tex_norm,
                dims="cloud",
            )

            # line width (km/s; shape: clouds)
            fwhm_norm = pm.Gamma("fwhm_norm", alpha=2.0, beta=1.0, dims="cloud")
            _ = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

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
                    initval=np.linspace(-1.0, 1.0, self.n_clouds),
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # Spectral rms (K)
            rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
            _ = pm.Deterministic("rms_observation", rms_observation_norm * prior_rms)

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
                sigma=self.model["rms_observation"],
                observed=self.data["observation"].brightness,
            )
