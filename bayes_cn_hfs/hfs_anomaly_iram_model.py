"""
hfs_anomaly_iram_model.py
HFSAnomalyIRAMModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm

from bayes_cn_hfs import HFSAnomalyModel
from bayes_cn_hfs import physics


class HFSAnomalyIRAMModel(HFSAnomalyModel):
    """Definition of the HFSAnomalyIRAMModel. SpecData keys must be
    "12CN-1/2" and "12CN-3/2". Predicted spectra
    are Ta* instead of brightness temperature (TB)."""

    def __init__(self, *args, Beff: float = 0.78, Feff: float = 0.94, **kwargs):
        """Initialize a new CNRatioAnomalyIRAMModel instance.

        Parameters
        ----------
        Beff : float, optional
            Beam efficiency, by default 0.78 (IRAM 30m at 115 GHz)
        Feff : float, optional
            Forward efficiency, by default 0.94 (IRAM 30m at 115 GHz)
        """
        # Initialize HFSAnomalyModel
        super().__init__(*args, **kwargs)

        # save inputs
        self.Beff = Beff
        self.Feff = Feff

    def add_priors(self, *args, **kwargs):
        """Add priors and deterministics to the model."""
        # add CNRatioAnomalyModel priors
        super().add_priors(*args, **kwargs)

    def add_likelihood(self):
        """Add likelihood to the model. SpecData keys must be
        "12CN-1/2" and  "12CN-3/2"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        labels = ["12CN-1/2", "12CN-3/2"]
        for label in labels:
            cloud_tex = 10.0 ** self.model["log10_tex"]
            component_tex = 10.0 ** self.model["log10_tex_comp"]
            tau = physics.predict_tau(
                self.mol_data,
                self.data[label].spectral,
                self.model["log10_N"],
                self.model["velocity"],
                self.model["fwhm"],
                cloud_tex,
                component_tex,
            )
            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    self.data[label].spectral,
                    tau,
                    component_tex,
                    self.bg_temp,
                )
            )

            # Add baseline model
            baseline_models = self.predict_baseline()
            predicted = predicted_line + baseline_models[label]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=self.data[label].noise,
                    observed=self.data[label].brightness,
                )
