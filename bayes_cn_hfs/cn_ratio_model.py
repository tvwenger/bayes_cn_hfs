"""
cn_ratio_model.py
CNRatioModel definition

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


class CNRatioModel(BaseModel):
    """Definition of the CNRatioModel."""

    def __init__(
        self,
        *args,
        mol_data_12CN: Optional[dict] = None,
        mol_data_13CN: Optional[dict] = None,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        mol_data_12CN : Optional[dict], optional
            Molecular data dictionary returned by get_molecule_data() for CN. If None, it will
            be downloaded. Default is None
        mol_data_13CN : Optional[dict], optional
            Molecular data dictionary returned by get_molecule_data() for 13CN. If None, it will
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
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff

        # Get molecular data
        self.mol_data_12CN, self.mol_weight_12CN = supplement_mol_data("CN", mol_data=mol_data_12CN)
        self.mol_data_13CN, self.mol_weight_13CN = supplement_mol_data("13CN", mol_data=mol_data_13CN)

        # Determine which CN transitions are used to populate state column densities
        transition_free_12CN, _, _ = physics.balance(self.mol_data_12CN, log10_N0=None, verbose=self.verbose)
        transition_free_13CN, _, _ = physics.balance(self.mol_data_13CN, log10_N0=None, verbose=self.verbose)

        # Add transitions and states to model
        coords = {
            "transition_12CN": self.mol_data_12CN["freq"],
            "transition_free_12CN": transition_free_12CN,
            "state_u_12CN": list(set(self.mol_data_12CN["Qu"])),
            "state_l_12CN": list(set(self.mol_data_12CN["Ql"])),
            "transition_13CN": self.mol_data_13CN["freq"],
            "transition_free_13CN": transition_free_13CN,
            "state_u_13CN": list(set(self.mol_data_13CN["Qu"])),
            "state_l_13CN": list(set(self.mol_data_13CN["Ql"])),
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N0_12CN",
            "fwhm_12CN",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "fwhm_thermal_12CN": r"$\Delta V_{\rm th, CN}$ (km s$^{-1}$)",
                "fwhm_thermal_13CN": r"$\Delta V_{{\rm th}, ^{13}\rm CN}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm_12CN": r"$\Delta V_{\rm CN}$ (km s$^{-1}$)",
                "fwhm_13CN": r"$\Delta V_{^{13}\rm CN}$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "log10_N0_12CN": r"$\log_{10} N_{0, \rm CN}$ (cm$^{-2}$)",
                "log10_N0_13CN": r"$\log_{10} N_{0, ^{13}\rm CN}$ (cm$^{-2}$)",
                "log10_Tex": r"$\log_{10} T_{\rm ex}$ (K)",
                "Tex_12CN": r"$T_{\rm ex, CN}$ (K)",
                "Tex_13CN": r"$T_{{\rm ex}, ^{13}\rm CN}$ (K)",
                "log10_inv_boltz_factor_12CN": r"$\log_{10} B^{-1}_{\rm CN}$",
                "log10_inv_boltz_factor_13CN": r"$\log_{10} B^{-1}_{^{13}\rm CN}$",
                "log10_Nl_12CN": r"$\log_{10} N_{l, \rm CN}$ (cm$^{-2}$)",
                "log10_Nu_12CN": r"$\log_{10} N_{u, \rm CN}$ (cm$^{-2}$)",
                "log10_N_12CN": r"$\log_{10} N_{\rm tot, CN}$ (cm$^{-2}$)",
                "log10_Nl_13CN": r"$\log_{10} N_{l, ^{13}\rm CN}$ (cm$^{-2}$)",
                "log10_Nu_13CN": r"$\log_{10} N_{u, ^{13}\rm CN}$ (cm$^{-2}$)",
                "log10_N_13CN": r"$\log_{10} N_{{\rm tot}, ^{13}\rm CN}$ (cm$^{-2}$)",
                "tau_12CN": r"$\tau_{\rm CN}$",
                "tau_13CN": r"$\tau_{^{13}\rm CN}$",
                "tau_total_12CN": r"$\tau_{\rm tot, CN}$",
                "tau_total_13CN": r"$\tau_{{\rm tot}, ^{13}\rm CN}$",
                "TR_12CN": r"$T_{R, \rm CN}$ (K)",
                "TR_13CN": r"$T_{R, ^{13}\rm CN}$ (K)",
                "12C_13C_ratio": r"$^{12}{\rm C}/^{13}{\rm C}$",
            }
        )

    def add_priors(
        self,
        prior_log10_N0_12CN: Iterable[float] = [13.0, 0.25],
        prior_log10_N0_13CN: Iterable[float] = [11.0, 0.25],
        prior_log10_Tkin: Iterable[float] = [1.5, 0.5],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = 0.0,
        prior_fwhm_L: float = 1.0,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        assume_LTE: bool = True,
        prior_log10_Tex: Iterable[float] = [1.5, 0.5],
        assume_CTEX_12CN: bool = True,
        assume_CTEX_13CN: bool = True,
        prior_log10_inv_boltz_factor: float = 5.0,
        fix_log10_Tkin: Optional[float] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N0_12CN : Iterable[float], optional
            Prior distribution on CN ground state column density, by default [13.0, 0.25], where
            log10_N0_12CN ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_N0_13CN : Iterable[float], optional
            Prior distribution on 13CN ground state column density, by default [11.0, 0.25], where
            log10_N0_13CN ~ Normal(mu=prior[0], sigma=prior[1])
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
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the latent pseudo-Voight Lorentzian profile line width (km/s),
            by default 1.0, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
        prior_rms : Optional[dict[str, float]], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            Keys are dataset names and values are priors. If None, then the spectral rms is taken
            from dataset.noise and not inferred.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        assume_LTE : bool, optional
            Assume local thermodynamic equilibrium by fixing the excitation temperature of every
            transition to the cloud kinetic temperature, by default True. assume_LTE = True implies
            assume_CTEX_12CN = assume_CTEX_13CN = True.
        prior_log10_Tex : Iterable[float], optional
            Prior distribution on CTEX log10 excitation temperature (K), by
            default [1.5, 0.5], where
            log10_Tex ~ Normal(mu=prior[0], sigma=prior[1])
            This parameter has no effect when assume_LTE = True, or when assume_CTEX_12CN = assume_CTEX_13CN = False
        assume_CTEX_12CN : bool, optional
            Assume that every 12CN transition has the same excitation temperature, by default True.
            If False, derive the excitation temperature of every 12CN transition.
            assume_LTE = True implies assume_CTEX_12CN = True.
        assume_CTEX_13CN : bool, optional
            Assume that every 13CN transition has the same excitation temperature, by default True.
            If False, derive the excitation temperature of every 13CN transition.
            assume_LTE = True implies assume_CTEX_13CN = True.
        prior_log10_inv_boltz_factor : float, optional
            Prior distribution on free transition log10 inverse Boltzmann factors, by default 5.0, where
            log10_inv_boltz_factor_12CN_free ~ Gamma(alpha=2.0, beta=prior)
            This parameter has no effect when assume_LTE = True. The same prior is used for both 12CN
            (if assume_CTEX_12CN is False) and 13CN (if assume_CTEX_13CN is False).
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

        if assume_LTE and not (assume_CTEX_12CN and assume_CTEX_13CN):
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
            fwhm_thermal_12CN = pm.Deterministic(
                "fwhm_thermal_12CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_12CN), dims="cloud"
            )
            fwhm_thermal_13CN = pm.Deterministic(
                "fwhm_thermal_13CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_13CN), dims="cloud"
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = 0.0
            if prior_fwhm_nonthermal > 0:
                fwhm_nonthermal_norm = pm.HalfNormal("fwhm_nonthermal_norm", sigma=1.0, dims="cloud")
                fwhm_nonthermal = pm.Deterministic(
                    "fwhm_nonthermal", prior_fwhm_nonthermal * fwhm_nonthermal_norm, dims="cloud"
                )

            # Total (physical) FWHM (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm_12CN", pt.sqrt(fwhm_thermal_12CN**2.0 + fwhm_nonthermal**2.0), dims="cloud")
            _ = pm.Deterministic("fwhm_13CN", pt.sqrt(fwhm_thermal_13CN**2.0 + fwhm_nonthermal**2.0), dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
            _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

            # Spectral rms (K)
            if prior_rms is not None:
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms)

            # CN ground state column density (shape: clouds)
            log10_N0_12CN_norm = pm.Normal("log10_N0_12CN_norm", mu=0.0, sigma=1.0, dims=["cloud"])
            log10_N0_12CN = pm.Deterministic(
                "log10_N0_12CN", prior_log10_N0_12CN[0] + prior_log10_N0_12CN[1] * log10_N0_12CN_norm, dims=["cloud"]
            )

            # 13CN ground state column density (shape: clouds)
            log10_N0_13CN_norm = pm.Normal("log10_N0_13CN_norm", mu=0.0, sigma=1.0, dims=["cloud"])
            log10_N0_13CN = pm.Deterministic(
                "log10_N0_13CN", prior_log10_N0_13CN[0] + prior_log10_N0_13CN[1] * log10_N0_13CN_norm, dims=["cloud"]
            )

            molecules = ["12CN", "13CN"]
            mol_datas = [self.mol_data_12CN, self.mol_data_13CN]
            assume_CTEXs = [assume_CTEX_12CN, assume_CTEX_13CN]
            log10_N0s = [log10_N0_12CN, log10_N0_13CN]
            log10_Tex = None  # in case where 12CN and 13CN share CTEX
            Ntots = []  # storage for total column densities

            for molecule, mol_data, assume_CTEX, log10_N0 in zip(molecules, mol_datas, assume_CTEXs, log10_N0s):
                if assume_LTE:
                    # Excitation temperature is fixed at kinetic temperature
                    Tex = pm.Deterministic(
                        f"Tex_{molecule}",
                        pt.repeat(10.0 ** log10_Tkin[None, :], len(mol_data["freq"]), 0),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                    # log10 inverse Boltzmann factors (shape: transitions, clouds)
                    log10_inv_boltz_factor = pm.Deterministic(
                        f"log10_inv_boltz_factor_{molecule}",
                        physics.calc_log10_inv_boltz_factor(mol_data["freq"][:, None], Tex),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                    # State column densities (cm2; shape: states, clouds)
                    _, log10_Nl, log10_Nu = physics.balance(
                        mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
                    )
                elif assume_CTEX:
                    if log10_Tex is None:
                        # Excitation temperature (K) (shape: transitions, clouds)
                        log10_Tex_norm = pm.Normal(f"log10_Tex_{molecule}_norm", mu=0.0, sigma=1.0, dims="cloud")
                        log10_Tex = pm.Deterministic(
                            "log10_Tex", prior_log10_Tex[0] + prior_log10_Tex[1] * log10_Tex_norm, dims="cloud"
                        )

                    # assumed constant across transitions
                    Tex = pm.Deterministic(
                        f"Tex_{molecule}",
                        pt.repeat(10.0 ** log10_Tex[None, :], len(mol_data["freq"]), 0),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                    # log10 inverse Boltzmann factors (shape: transitions, clouds)
                    log10_inv_boltz_factor = pm.Deterministic(
                        f"log10_inv_boltz_factor_{molecule}",
                        physics.calc_log10_inv_boltz_factor(mol_data["freq"][:, None], Tex),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                    # State column densities (cm2; shape: states, clouds)
                    _, log10_Nl, log10_Nu = physics.balance(
                        mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
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
                        f"log10_inv_boltz_factor_free_{molecule}",
                        alpha=2.0,
                        beta=prior_log10_inv_boltz_factor,
                        dims=[f"transition_free_{molecule}", "cloud"],
                    )

                    # Detailed balance
                    log10_inv_boltz_factor = [
                        (
                            log10_inv_boltz_factor_free[self.model.coords[f"transition_free_{molecule}"].index(freq)]
                            if freq in self.model.coords[f"transition_free_{molecule}"]
                            else None
                        )
                        for freq in mol_data["freq"]
                    ]
                    _, log10_Nl, log10_Nu = physics.balance(
                        mol_data, log10_N0=log10_N0, log10_inv_boltz_factor=log10_inv_boltz_factor, verbose=False
                    )

                    # Derive remaining inverse Boltzmann factors (shape: transitions, clouds)
                    log10_inv_boltz_factor = pm.Deterministic(
                        f"log10_inv_boltz_factor_{molecule}",
                        pt.stack(
                            [
                                (
                                    log10_inv_boltz_factor[idx]
                                    if log10_inv_boltz_factor[idx] is not None
                                    else log10_Nl[state_l] - log10_Nu[state_u] + np.log10(Gu / Gl)
                                )
                                for idx, (Gl, Gu, state_l, state_u) in enumerate(
                                    zip(
                                        mol_data["Gl"],
                                        mol_data["Gu"],
                                        mol_data["state_l"],
                                        mol_data["state_u"],
                                    )
                                )
                            ]
                        ),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                    # Derive excitation temperatures (shape: transitions, clouds)
                    Tex = pm.Deterministic(
                        f"Tex_{molecule}",
                        physics.calc_Tex(mol_data["freq"][:, None], log10_inv_boltz_factor),
                        dims=[f"transition_{molecule}", "cloud"],
                    )

                # State column densities (cm-2; shape: states, clouds)
                log10_Nl = pm.Deterministic(f"log10_Nl_{molecule}", log10_Nl, dims=[f"state_l_{molecule}", "cloud"])
                log10_Nu = pm.Deterministic(f"log10_Nu_{molecule}", log10_Nu, dims=[f"state_u_{molecule}", "cloud"])

                # Total column density across all states (cm-2; shape: clouds)
                Ntot = pt.sum(10.0**log10_Nl, axis=0) + pt.sum(10.0**log10_Nu, axis=0)
                Ntots.append(Ntot)
                _ = pm.Deterministic(f"log10_N_{molecule}", pt.log10(Ntot), dims="cloud")

                # State column densities for each transition (cm-2; shape: transitions, clouds)
                Nl = pt.stack([10.0 ** log10_Nl[state_l] for state_l in mol_data["state_l"]])
                Nu = pt.stack([10.0 ** log10_Nu[state_u] for state_u in mol_data["state_u"]])

                # Optical depth (shape: transitions, clouds)
                tau = pm.Deterministic(
                    f"tau_{molecule}",
                    physics.calc_optical_depth(
                        mol_data["freq"][:, None],
                        mol_data["Gl"][:, None],
                        mol_data["Gu"][:, None],
                        Nl,
                        Nu,
                        mol_data["Aul"][:, None],
                        1.0,  # integrated line profile
                    ),
                    dims=[f"transition_{molecule}", "cloud"],
                )

                # Total optical depth (shape: clouds)
                _ = pm.Deterministic(f"tau_total_{molecule}", pt.sum(tau, axis=0), dims="cloud")

                # Radiation temperature (K; shape: transitions, clouds)
                _ = pm.Deterministic(
                    f"TR_{molecule}",
                    physics.calc_TR(mol_data["freq"][:, None], 10.0**log10_inv_boltz_factor),
                    dims=[f"transition_{molecule}", "cloud"],
                )

            # Total column density ratio over all states
            _ = pm.Deterministic("12C_13C_ratio", Ntots[0] / Ntots[1], dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            if "12CN" in label:
                mol_data = self.mol_data_12CN
                molecule = "12CN"
            elif "13CN" in label:
                mol_data = self.mol_data_13CN
                molecule = "13CN"
            else:  # pragma: no cover
                raise ValueError(f"Invalid dataset label: {label}")

            # Optical depth spectra (shape: spectral, transitions, clouds)
            tau_spectra = physics.predict_tau_spectra(
                mol_data,
                dataset.spectral,
                self.model[f"tau_{molecule}"],
                self.model["velocity"],
                self.model[f"fwhm_{molecule}"],
                self.model["fwhm_L"],
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau_spectra,
                    self.model[f"TR_{molecule}"],
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
