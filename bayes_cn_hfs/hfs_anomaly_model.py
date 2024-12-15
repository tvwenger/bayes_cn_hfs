"""
hfs_anomaly_model.py
HFSAnomalyModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
import pymc as pm

from bayes_cn_hfs.hfs_model import HFSModel


class HFSAnomalyModel(HFSModel):
    """Definition of the HFSAnomalyModel. SpecData key must be "observation"."""

    def add_priors(self, *args, prior_log10_anomaly: float = 0.1, **kwargs):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_anomaly : float, optional
            Prior distribution on the CN excitation temperature anomaly (dex), by default 0.1, where
            log10_Tex ~ Normal(mu=log10_Tex, sigma=prior)
        """
        # add HFSModel priors, and break degeneracy between optical depth and excitation temperature
        # by assuming tau_total = prior_tau_total.
        super().add_priors(*args, **kwargs)

        with self.model:
            # Hyperfine anomaly
            log10_anomaly_norm = pm.Normal("log10_anomaly_norm", mu=0.0, sigma=1.0, dims=["component", "cloud"])
            log10_anomaly = pm.Deterministic(
                "log10_anomaly", prior_log10_anomaly * log10_anomaly_norm, dims=["component", "cloud"]
            )
            anomaly = 10.0**log10_anomaly

            # Anomalous excitation temperature (K; shape: components, clouds)
            _ = pm.Deterministic(
                "Tex", anomaly * 10.0 ** self.model["log10_Tkin"][None, :], dims=["component", "cloud"]
            )

            # Component optical depths are fixed and given by LTE ratios
            weights = self.mol_data["degu"] * self.mol_data["Aul"]
            weights = weights / weights.sum()
            _ = pm.Data("tau_weight", np.repeat(weights[:, None], self.n_clouds, axis=1), dims=["component", "cloud"])
