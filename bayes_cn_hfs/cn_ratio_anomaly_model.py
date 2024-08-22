"""
cn_ratio_anomaly_model.py
CNRatioAnomalyModel definition

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

import pymc as pm
import pytensor.tensor as pt

from bayes_cn_hfs import CNRatioModel
from bayes_cn_hfs import physics


class CNRatioAnomalyModel(CNRatioModel):
    """Definition of the CNRatioAnomalyModel. SpecData keys must be "12CN" and "13CN"."""

    def __init__(self, *args, **kwargs):
        """Initialize a new CNRatioAnomalyModel instance"""
        # Initialize CNRatioModel
        super().__init__(*args, **kwargs)

        # Add 12CN components to model
        coords = {
            "component_12CN": self.mol_data_12CN["freq"],
        }
        self.model.add_coords(coords=coords)

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "tex_12CN": r"$T_{\rm ex, CN}$ (K)",
            }
        )

    def add_priors(
        self,
        *args,
        prior_tex_12CN_anomaly: float = 1.0,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_tex_12CN_anomaly : float, optional
            Prior distribution on the 12CN excitation temperature anomaly (K), by default 1.0, where
            tex_12CN ~ Normal(mu=10**log10_tex, sigma=prior)
        """
        # add CNRatioModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # Excitation temperature anomaly (shape: components, clouds)
            tex_12CN_anomaly_norm = pm.Normal(
                "tex_12CN_anomaly_norm", mu=0.0, sigma=1.0, dims=["component_12CN", "cloud"]
            )
            tex_12CN_anomaly = pm.Deterministic(
                "tex_12CN_anomaly", prior_tex_12CN_anomaly * tex_12CN_anomaly_norm, dims=["component_12CN", "cloud"]
            )

            # Excitation temperature (shape: components, clouds)
            _ = pm.Deterministic(
                "tex_12CN", 10.0 ** self.model["log10_tex"] + tex_12CN_anomaly, dims=["component_12CN", "cloud"]
            )

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
            # Predict optical depth spectrum (shape: spectral, components, clouds)
            cloud_tex = 10.0 ** self.model["log10_tex"]
            component_tex = cloud_tex[None, :]
            if label == "12CN":
                component_tex = self.model["tex_12CN"]
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
