"""
cn_ratio_lte_model.py
CNRatioLTEModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bayes_cn_hfs.cn_ratio_model import CNRatioModel


class CNRatioLTEModel(CNRatioModel):
    """Definition of the CNRatioLTEModel. SpecData keys must include the strings "12CN" and "13CN"."""

    def add_priors(
        self,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model"""
        # add CNRatioModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # Excitation temperature (K; shape: components, clouds)
            # LTE assumption: Tkin = Tex for all clouds and transitions
            _ = pm.Deterministic(
                "Tex_12CN",
                pt.repeat(10.0 ** self.model["log10_Tkin"][None, :], len(self.mol_data_12CN["freq"]), 0),
                dims=["component_12CN", "cloud"],
            )
            _ = pm.Deterministic(
                "Tex_13CN",
                pt.repeat(10.0 ** self.model["log10_Tkin"][None, :], len(self.mol_data_13CN["freq"]), 0),
                dims=["component_13CN", "cloud"],
            )

            # Component optical depths are fixed and given by LTE ratios
            weights_12CN = self.mol_data_12CN["degu"] * self.mol_data_12CN["Aul"]
            weights_12CN = weights_12CN / weights_12CN.sum()
            _ = pm.Data(
                "tau_weight_12CN",
                np.repeat(weights_12CN[:, None], self.n_clouds, axis=1),
                dims=["component_12CN", "cloud"],
            )
            weights_13CN = self.mol_data_13CN["degu"] * self.mol_data_13CN["Aul"]
            weights_13CN = weights_13CN / weights_13CN.sum()
            _ = pm.Data(
                "tau_weight_13CN",
                np.repeat(weights_13CN[:, None], self.n_clouds, axis=1),
                dims=["component_13CN", "cloud"],
            )
