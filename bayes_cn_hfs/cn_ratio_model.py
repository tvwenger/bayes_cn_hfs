"""
cn_ratio_model.py
CNRatioModel definition

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
import pytensor.tensor as pt

from bayes_spec import BaseModel

from bayes_cn_hfs import physics
from bayes_cn_hfs.utils import get_molecule_data


class CNRatioModel(BaseModel):
    """Definition of the CNRatioModel. SpecData keys must be "12CN" and "13CN"."""

    def __init__(
        self,
        *args,
        mol_data_12CN: Optional[dict] = None,
        mol_data_13CN: Optional[dict] = None,
        bg_temp: float = 2.7,
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
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.bg_temp = bg_temp
        self.mol_data_12CN = mol_data_12CN
        if mol_data_12CN is None:
            self.mol_data_12CN = get_molecule_data(
                "CN, v = 0, 1",  # molecule name in JPLSpec
                vibrational_state=0,  # vibrational state number
                rot_state_lower=0,  # lower rotational state
            )
        self.mol_data_13CN = mol_data_13CN
        if mol_data_13CN is None:
            self.mol_data_13CN = get_molecule_data(
                "C-13-N",  # molecule name in JPLSpec
                rot_state_lower=0,  # lower rotational state
            )

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N_12CN",
            "log10_tex",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N_12CN": r"log$_{10}$ $N_{\rm CN}$ (cm$^{-2}$)",
                "log10_tex": r"log$_{10}$ $T_{\rm ex}$ (K)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "ratio_13C_12C": r"$^{13}{\rm C}/^{12}{\rm C}$",
                "rms_12CN": r"rms$_{\rm CN}$ (K)",
                "rms_13CN": r"rms$_{^{13}{\rm CN}}$ (K)",
            }
        )

    def add_priors(
        self,
        prior_log10_N_12CN: Iterable[float] = [14.0, 1.0],
        prior_log10_tex: Iterable[float] = [1.0, 0.1],
        prior_fwhm: float = 1.0,
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_ratio_13C_12C: float = 0.01,
        prior_rms_12CN: float = 0.01,
        prior_rms_13CN: float = 0.01,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N_12CN : Iterable[float], optional
            Prior distribution on log10 total 12CN column density (cm-2), by default [14.0, 1.0], where
            log10_N_12CN ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_tex : Iterable[float], optional
            Prior distribution on log10 excitation temperature (K), by default [1.0, 0.1], where
            log10_tex ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm : float, optional
            Prior distribution on FWHM line width (km s-1), by default 1.0, where
            fwhm ~ Gamma(alpha=2.0, beta=1.0/prior)
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_ratio_13C_12C : float, optional
            Prior distribution on 13C/12C ratio, by default 0.01, where
            ratio_13C_12C ~ HalfNormal(sigma=prior)
        prior_rms_12CN : float, optional
            Prior distribution on 12CN spectral rms (K), by default 0.01, where
            rms ~ HalfNormal(sigma=prior)
        prior_rms_13CN : float, optional
            Prior distribution on 13CN spectral rms (K), by default 0.01, where
            rms ~ HalfNormal(sigma=prior)
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # 12CN upper level column density (cm-2; shape: clouds)
            log10_N_12CN_norm = pm.Normal("log10_N_12CN_norm", mu=0.0, sigma=1.0, dims="cloud")
            _ = pm.Deterministic(
                "log10_N_12CN",
                prior_log10_N_12CN[0] + prior_log10_N_12CN[1] * log10_N_12CN_norm,
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
            fwhm_norm = pm.HalfNormal("fwhm_norm", sigma=1.0, dims="cloud")
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

            # 13C/12C ratio
            ratio_13C_12C_norm = pm.HalfNormal("ratio_13C_12C_norm", sigma=1.0, dims="cloud")
            _ = pm.Deterministic("ratio_13C_12C", prior_ratio_13C_12C * ratio_13C_12C_norm, dims="cloud")

            # Spectral rms (K)
            labels = ["12CN", "13CN"]
            prior_rmss = [prior_rms_12CN, prior_rms_13CN]
            for label, prior_rms in zip(labels, prior_rmss):
                # Spectral rms (K)
                rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms)

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "12CN" and "13CN"."""
        # 13CN column density (cm-2; shape: clouds)
        log10_N_13CN = self.model["log10_N_12CN"] + pt.log10(self.model["ratio_13C_12C"])

        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict both spectra
        labels = ["12CN", "13CN"]
        mol_datas = [self.mol_data_12CN, self.mol_data_13CN]
        log10_Ns = [self.model["log10_N_12CN"], log10_N_13CN]
        for label, mol_data, log10_N in zip(labels, mol_datas, log10_Ns):
            # Predict optical depth spectrum, sum over components (shape: spectral, components, clouds)
            cloud_tex = 10.0 ** self.model["log10_tex"]
            component_tex = cloud_tex[None, :]
            tau = physics.predict_tau(
                mol_data,
                self.data[label].spectral,
                log10_N,
                self.model["velocity"],
                self.model["fwhm"],
                cloud_tex,
                component_tex,
            )

            # Radiative transfer (shape: spectral)
            predicted_line = physics.radiative_transfer(
                self.data[label].spectral,
                tau,
                component_tex,
                self.bg_temp,
            )

            # Add baseline model
            predicted = predicted_line + baseline_models[label]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=self.model[f"rms_{label}"],
                    observed=self.data[label].brightness,
                )
