"""
cn_model.py
CNModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from bayes_cn_hfs.utils import supplement_mol_data
from bayes_cn_hfs import physics


class CNModel(BaseModel):
    """Definition of the CNModel."""

    def __init__(
        self,
        *args,
        molecule: str = "CN",
        mol_data: Optional[dict] = None,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        molecule : str, optional
            Either "CN" or "13CN"
        mol_data : Optional[dict], optional
            Molecular data dictionary returned by get_molecule_data(). If None, it will
            be downloaded. Default is None
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        Beff : float, optional
            Beam efficiency, by default 1.0
        Feff : float, optional
            Forward efficiency, by default 1.0
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.molecule = molecule
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff

        # Get molecular data
        self.mol_data, self.mol_weight = supplement_mol_data(molecule, mol_data=mol_data)

        # Determine which transitions are used to populate state column densities
        transition_free, _, _ = physics.balance(self.mol_data, log10_N0=None, verbose=self.verbose)

        # Add transitions and states to model
        coords = {
            "transition": self.mol_data["freq"],
            "transition_free": transition_free,
            "state_u": list(set(self.mol_data["Qu"])),
            "state_l": list(set(self.mol_data["Ql"])),
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N0",
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "log10_N0": r"$\log_{10} N_0$ (cm$^{-2}$)",
                "log10_Tex": r"$\log_{10} T_{\rm ex}$ (K)",
                "Tex": r"$T_{\rm ex}$ (K)",
                "log10_inv_boltz_factor": r"$\log_{10} B^{-1}$",
                "log10_Nl": r"$\log_{10} N_l$ (cm$^{-2}$)",
                "log10_Nu": r"$\log_{10} N_u$ (cm$^{-2}$)",
                "log10_N": r"$\log_{10} N_{\rm tot}$ (cm$^{-2}$)",
                "tau": r"$\tau$",
                "tau_total": r"$\tau_{\rm tot}$",
                "TR": r"$T_R$ (K)",
            }
        )

    def add_priors(
        self,
        prior_log10_N0: Iterable[float] = [13.0, 0.25],
        prior_log10_Tkin: Iterable[float] = [1.5, 0.5],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = 0.0,
        prior_fwhm_L: float = 1.0,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        assume_LTE: bool = True,
        prior_log10_Tex: Iterable[float] = [1.5, 0.5],
        assume_CTEX: bool = True,
        prior_log10_inv_boltz_factor: float = 5.0,
        allow_masers: bool = False,
        fix_log10_Tkin: Optional[float] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N0 : Iterable[float], optional
            Prior distribution on ground state column density, by default [13.0, 0.25], where
            log10_N0 ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 cloud kinetic temperature (K), by default
            [1.5, 0.5], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm_nonthermal : float, optional
            Prior distribution on non-thermal FWHM (km s-1), by default 0.0, where
            fwhm_nonthermal ~ HalfNormal(sigma=prior_fwhm_nonthermal)
            If 0.0, assume no non-thermal broadening.
        prior_fwhm_L : float, optional
            Prior distribution on the pseudo-Voight Lorentzian profile line width (km/s),
            by default 1.0, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
            Setting this parameter to the typical separation between spectral lines aids in
            sampling efficiency.
        prior_rms : Optional[dict[str, float]], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            Keys are dataset names and values are priors. If None, then the spectral rms is taken
            from dataset.noise and not inferred.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Prior distribution on the normalized baseline polynomial coefficients, by default None.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset.
        assume_LTE : bool, optional
            Assume local thermodynamic equilibrium by fixing the excitation temperature of every
            transition to the cloud kinetic temperature, by default True. assume_LTE = True implies
            assume_CTEX = True.
        prior_log10_Tex : Iterable[float], optional
            Prior distribution on CTEX log10 excitation temperature (K), by
            default [1.5, 0.5], where
            log10_Tex ~ Normal(mu=prior[0], sigma=prior[1])
            This parameter has no effect when assume_LTE = True or when assume_CTEX = False.
        assume_CTEX : bool, optional
            Assume that every transition has the same excitation temperature, by default True.
            If False, use derive the excitation temperature of every transition.
            assume_LTE = True implies assume_CTEX = True.
        prior_log10_inv_boltz_factor : float, optional
            Prior distribution on free transition log10 inverse Boltzmann factors, by default 5.0, where
            log10_inv_boltz_factor_free ~ Gamma(alpha=2.0, beta=prior)
            This parameter has no effect when assume_LTE = True or when assume_CTEX = False.
            This is a convenient parameterization because it prevents non-physical solutions.
        allow_masers : bool, optional
            If True, allow derived transition excitation temperatures to be negative, by default False.
            This parameter has no effect when assume_LTE = True or when assume_CTEX = False.
        fix_log10_Tkin : Optional[float], optional
            Fix the log10 cloud kinetic temperature at this value (K), by default None.
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            velocity(cloud = n) ~
                prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        if assume_LTE and not assume_CTEX:
            raise ValueError("Can't assume LTE and not CTEX")

        with self.model:
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
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # kinetic temperature (K; shape: clouds)
            if fix_log10_Tkin:
                # log10_Tkin is fixed
                log10_Tkin = pm.Data("log10_Tkin", np.ones(self.n_clouds) * fix_log10_Tkin, dims="cloud")
            else:
                log10_Tkin_norm = pm.Normal("log10_Tkin_norm", mu=0.0, sigma=1.0, dims="cloud")
                log10_Tkin = pm.Deterministic(
                    "log10_Tkin",
                    prior_log10_Tkin[0] + prior_log10_Tkin[1] * log10_Tkin_norm,
                    dims="cloud",
                )

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic(
                "fwhm_thermal", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight), dims="cloud"
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = 0.0
            if prior_fwhm_nonthermal > 0:
                fwhm_nonthermal_norm = pm.HalfNormal("fwhm_nonthermal_norm", sigma=1.0, dims="cloud")
                fwhm_nonthermal = pm.Deterministic(
                    "fwhm_nonthermal", prior_fwhm_nonthermal * fwhm_nonthermal_norm, dims="cloud"
                )

            # Total (physical) FWHM (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
            _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

            # Spectral rms (K)
            if prior_rms is not None:
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms[label])

            # Ground state column density (shape: clouds)
            log10_N0_norm = pm.Normal("log10_N0_norm", mu=0.0, sigma=1.0, dims=["cloud"])
            log10_N0 = pm.Deterministic(
                "log10_N0", prior_log10_N0[0] + prior_log10_N0[1] * log10_N0_norm, dims=["cloud"]
            )

            if assume_LTE:
                # Excitation temperature is fixed at kinetic temperature
                Tex = pm.Deterministic(
                    "Tex",
                    pt.repeat(10.0 ** log10_Tkin[None, :], len(self.mol_data["freq"]), 0),
                    dims=["transition", "cloud"],
                )

                # log10 inverse Boltzmann factors (shape: transitions, clouds)
                log10_inv_boltz_factor = pm.Deterministic(
                    "log10_inv_boltz_factor",
                    physics.calc_log10_inv_boltz_factor(self.mol_data["freq"][:, None], Tex),
                    dims=["transition", "cloud"],
                )

                # State column densities (cm2; shape: states, clouds)
                _, log10_Nl, log10_Nu = physics.balance(
                    self.mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
                )
            elif assume_CTEX:
                # Excitation temperature (K) (shape: transitions, clouds)
                log10_Tex_norm = pm.Normal("log10_Tex_norm", mu=0.0, sigma=1.0, dims="cloud")
                log10_Tex = pm.Deterministic(
                    "log10_Tex", prior_log10_Tex[0] + prior_log10_Tex[1] * log10_Tex_norm, dims="cloud"
                )

                # assumed constant across transitions
                Tex = pm.Deterministic(
                    "Tex",
                    pt.repeat(10.0 ** log10_Tex[None, :], len(self.mol_data["freq"]), 0),
                    dims=["transition", "cloud"],
                )

                # log10 inverse Boltzmann factors (shape: transitions, clouds)
                log10_inv_boltz_factor = pm.Deterministic(
                    "log10_inv_boltz_factor",
                    physics.calc_log10_inv_boltz_factor(self.mol_data["freq"][:, None], Tex),
                    dims=["transition", "cloud"],
                )

                # State column densities (cm2; shape: states, clouds)
                _, log10_Nl, log10_Nu = physics.balance(
                    self.mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
                )
            else:
                # Non-LTE case
                # Boltzmann factor = gL*nU/(gU*nL) < 1 for Tex > 0
                #                                  > 1 for Tex < 0
                #                                  = 0 for Tex = inf
                # Assuming no masers, we can place a Beta distribution prior on Boltzmann factor to
                # prevent Tex = 0 and Tex = inf solutions. We approximate log10(Beta-1) by a Gamma distribution.
                # log10 inverse Boltzmann factor (shape: transition_free, clouds)
                log10_inv_boltz_factor_free = pm.Gamma(
                    "log10_inv_boltz_factor_free",
                    alpha=2.0,
                    beta=prior_log10_inv_boltz_factor,
                    dims=["transition_free", "cloud"],
                )

                # Derive other states
                log10_inv_boltz_factor = [
                    (
                        log10_inv_boltz_factor_free[self.model.coords["transition_free"].index(freq)]
                        if freq in self.model.coords["transition_free"]
                        else None
                    )
                    for freq in self.mol_data["freq"]
                ]
                _, log10_Nl, log10_Nu = physics.balance(
                    self.mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
                )

                # Derive remaining inverse Boltzmann factors (shape: transitions, clouds)
                log10_inv_boltz_factor = pm.Deterministic(
                    "log10_inv_boltz_factor",
                    pt.stack(
                        [
                            (
                                log10_inv_boltz_factor[idx]
                                if log10_inv_boltz_factor[idx] is not None
                                else log10_Nl[state_l] - log10_Nu[state_u] + np.log10(Gu / Gl)
                            )
                            for idx, (Gl, Gu, state_l, state_u) in enumerate(
                                zip(
                                    self.mol_data["Gl"],
                                    self.mol_data["Gu"],
                                    self.mol_data["state_l"],
                                    self.mol_data["state_u"],
                                )
                            )
                        ]
                    ),
                    dims=["transition", "cloud"],
                )

                # Prevent negative excitation temperatures
                if not allow_masers:
                    _ = pm.Potential("restrict_masers", pt.switch(pt.gt(log10_inv_boltz_factor, 0.0), 0.0, -np.inf))

                # Derive excitation temperatures (shape: transitions, clouds)
                Tex = pm.Deterministic(
                    "Tex",
                    physics.calc_Tex(self.mol_data["freq"][:, None], log10_inv_boltz_factor),
                    dims=["transition", "cloud"],
                )

            # State column densities (cm-2; shape: states, clouds)
            log10_Nl = pm.Deterministic("log10_Nl", log10_Nl, dims=["state_l", "cloud"])
            log10_Nu = pm.Deterministic("log10_Nu", log10_Nu, dims=["state_u", "cloud"])

            # Total column density across all states (cm-2; shape: clouds)
            _ = pm.Deterministic(
                "log10_N",
                pt.log10(pt.sum(10.0**log10_Nl, axis=0) + pt.sum(10.0**log10_Nu, axis=0)),
                dims="cloud",
            )

            # State column densities for each transition (cm-2; shape: transitions, clouds)
            Nl = pt.stack([10.0 ** log10_Nl[state_l] for state_l in self.mol_data["state_l"]])
            Nu = pt.stack([10.0 ** log10_Nu[state_u] for state_u in self.mol_data["state_u"]])

            # Optical depth (shape: transitions, clouds)
            tau = pm.Deterministic(
                "tau",
                physics.calc_optical_depth(
                    self.mol_data["freq"][:, None],
                    self.mol_data["Gl"][:, None],
                    self.mol_data["Gu"][:, None],
                    Nl,
                    Nu,
                    self.mol_data["Aul"][:, None],
                    1.0,  # integrated line profile
                ),
                dims=["transition", "cloud"],
            )

            # Total optical depth (shape: clouds)
            _ = pm.Deterministic("tau_total", pt.sum(tau, axis=0), dims="cloud")

            # Radiation temperature (K; shape: transitions, clouds)
            _ = pm.Deterministic(
                "TR",
                physics.calc_TR(self.mol_data["freq"][:, None], 10.0**log10_inv_boltz_factor),
                dims=["transition", "cloud"],
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            # Optical depth spectra (shape: spectral, transitions, clouds)
            tau_spectra = physics.predict_tau_spectra(
                self.mol_data,
                dataset.spectral,
                self.model["tau"],
                self.model["velocity"],
                self.model["fwhm"],
                self.model["fwhm_L"],
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau_spectra,
                    self.model["TR"],
                    self.bg_temp,
                )
            )

            # Add baseline model
            predicted = predicted_line + baseline_models[label]

            with self.model:
                sigma = dataset.noise
                if f"rms_{label}" in self.model:
                    sigma = self.model[f"rms_{label}"]

                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=sigma,
                    observed=dataset.brightness,
                )
