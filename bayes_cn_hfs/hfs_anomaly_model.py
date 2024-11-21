"""
hfs_anomaly_model.py
HFSAnomalyModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm

from bayes_cn_hfs import HFSModel
from bayes_cn_hfs import physics


class HFSAnomalyModel(HFSModel):
    """Definition of the HFSAnomalyModel. SpecData key must be "observation"."""

    def __init__(self, *args, **kwargs):
        """Initialize a new HFSAnomalyModel instance"""
        # Initialize HFSModel
        super().__init__(*args, **kwargs)

        # Add components to model
        coords = {
            "component": self.mol_data["freq"],
        }
        self.model.add_coords(coords=coords)

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "tex": r"$T_{\rm ex}$ (K)",
            }
        )

    def add_priors(
        self,
        *args,
        prior_tex_anomaly: float = 1.0,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_tex_anomaly : float, optional
            Prior distribution on the excitation temperature anomaly (K), by default 1.0, where
            tex_anomaly ~ Normal(mu=0.0, sigma=prior)
        """
        # add HFSModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # Excitation temperature anomaly (shape: components, clouds)
            tex_anomaly_norm = pm.Normal("tex_anomaly_norm", mu=0.0, sigma=1.0, dims=["component", "cloud"])
            tex_anomaly = pm.Deterministic(
                "tex_anomaly", prior_tex_anomaly * tex_anomaly_norm, dims=["component", "cloud"]
            )

            # Excitation temperature (shape: components, clouds)
            _ = pm.Deterministic("tex", 10.0 ** self.model["log10_tex"] + tex_anomaly, dims=["component", "cloud"])

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict optical depth spectrum (shape: spectral, components, clouds)
        cloud_tex = 10.0 ** self.model["log10_tex"]
        component_tex = self.model["tex"]
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
