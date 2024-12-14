"""
hfs_lte_model.py
HFSLTEModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bayes_cn_hfs.hfs_model import HFSModel


class HFSLTEModel(HFSModel):
    """Definition of the HFSLTEModel. SpecData key must be "observation"."""

    def add_priors(self, *args, **kwargs):
        """Add priors and deterministics to the model"""
        # add HFSModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # Excitation temperature (K; shape: components, clouds)
            # LTE assumption: Tkin = Tex for all clouds and transitions
            _ = pm.Deterministic(
                "Tex",
                pt.repeat(10.0 ** self.model["log10_Tkin"][None, :], len(self.mol_data["freq"]), 0),
                dims=["component", "cloud"],
            )

            # Component optical depths are fixed and given by LTE ratios
            weights = self.mol_data["degu"] * self.mol_data["Aul"]
            weights = weights / weights.sum()
            _ = pm.Data("tau_weight", np.repeat(weights[:, None], self.n_clouds, axis=1), dims=["component", "cloud"])
