"""
cn_ratio_anomaly_iram_model.py
CNRatioAnomalyIRAMModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm
import pytensor.tensor as pt

from bayes_cn_hfs import CNRatioAnomalyModel
from bayes_cn_hfs import physics


class CNRatioAnomalyIRAMModel(CNRatioAnomalyModel):
    """Definition of the CNRatioAnomalyIRAMModel. SpecData keys must be
    "12CN-1/2", "12CN-3/2", "13CN-1/2", and "13CN-3/2". Predicted spectra
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
        # Initialize CNRatioAnomalyModel
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
        "12CN-1/2", "12CN-3/2", "13CN-1/2", and "13CN-3/2"."""
        # 13CN column density (cm-2; shape: clouds)
        log10_N_13CN = self.model["log10_N_12CN"] + pt.log10(self.model["ratio_13C_12C"])

        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        labels = ["12CN-1/2", "12CN-3/2", "13CN-1/2", "13CN-3/2"]
        mol_datas = [
            self.mol_data_12CN,
            self.mol_data_12CN,
            self.mol_data_13CN,
            self.mol_data_13CN,
        ]
        log10_Ns = [
            self.model["log10_N_12CN"],
            self.model["log10_N_12CN"],
            log10_N_13CN,
            log10_N_13CN,
        ]
        component_texs = [
            10.0 ** self.model["log10_tex_12CN_comp"],
            10.0 ** self.model["log10_tex_12CN_comp"],
            10.0 ** self.model["log10_tex"][None, :],
            10.0 ** self.model["log10_tex"][None, :],
        ]
        for label, mol_data, log10_N, component_tex in zip(labels, mol_datas, log10_Ns, component_texs):
            # Predict optical depth spectrum (shape: spectral, components, clouds)
            cloud_tex = 10.0 ** self.model["log10_tex"]
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
            predicted = predicted_line + baseline_models[label]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=self.data[label].noise,
                    observed=self.data[label].brightness,
                )
