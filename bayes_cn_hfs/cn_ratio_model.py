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
        transition_free_12CN, _, _ = physics.detailed_balance(self.mol_data_12CN, log10_N0=None, verbose=self.verbose)
        transition_free_13CN, _, _ = physics.detailed_balance(self.mol_data_13CN, log10_N0=None, verbose=self.verbose)

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
            "fwhm",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N0_12CN": r"$\log_{10} N_0$ (cm$^{-2}$)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "log10_Tex_mean": r"$\langle\log_{10} T_{\rm ex}\rangle$ (K)",
                "Tex_12CN": r"$T_{\rm ex, CN}$ (K)",
                "Tex_13CN": r"$T_{\rm ex, $^{13}$CN}$ (K)",
                "log_boltz_factor_12CN": r"$\ln B_{\rm CN}$",
                "log_boltz_factor_13CN": r"$\ln B_{^{13}\rm CN}$",
                "log_boltz_factor_12CN_mean": r"$\langle\ln B_{\rm CN}\rangle$",
                "log_boltz_factor_12CN_sigma": r"$\sigma_{\ln B, {\rm CN}}$",
                "log10_Nl_12CN": r"$\log_{10} N_{l, \rm CN}$ (cm$^{-2}$)",
                "log10_N_12CN": r"$\log_{10} N_{\rm tot, CN}$ (cm$^{-2}$)",
                "log10_Nl_13CN": r"$\log_{10} N_{l, ^{13}\rm CN}$ (cm$^{-2}$)",
                "log10_N_13CN": r"$\log_{10} N_{\rm tot, $^{13}$CN}$ (cm$^{-2}$)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "fwhm_thermal_12CN": r"$\Delta V_{\rm th, CN}$ (km s$^{-1}$)",
                "fwhm_thermal_13CN": r"$\Delta V_{\rm th, $^{13}$CN}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nt}$ (km s$^{-1}$)",
                "fwhm_12CN": r"$\Delta V_{\rm CN}$ (km s$^{-1}$)",
                "fwhm_13CN": r"$\Delta V_{^{13}\rm CN}$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "tau_peak_12CN": r"$\tau_{0, \rm CN}$",
                "tau_peak_13CN": r"$\tau_{0, ^{13}\rm CN}$",
                "tau_total_12CN": r"$\tau_{\rm tot, CN}$",
                "tau_total_13CN": r"$\tau_{\rm tot, $^{13}$CN}$",
                "log10_13C_12C_ratio": r"$\log_{10} ^{13}{\rm C}/^{12}{\rm C}$",
            }
        )

    def add_priors(
        self,
        prior_log10_N0_12CN: Iterable[float] = [12.0, 1.0],
        prior_log10_N0_13CN: Iterable[float] = [10.0, 1.0],
        prior_log10_Tex: Iterable[float] = [1.75, 0.25],
        prior_log10_Tkin: Iterable[float] = [1.75, 0.25],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_fwhm_nonthermal: float = None,
        prior_fwhm_L: float = 1.0,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        assume_LTE: bool = True,
        prior_log_boltz_factor_12CN_sigma: float = 0.0,
        fix_log10_Tkin: Optional[float] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_N0_12CN : Iterable[float], optional
            Prior distribution on CN ground state column density, by default [12.0, 1.0], where
            log10_N0_12CN ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_N0_13CN : Iterable[float], optional
            Prior distribution on 13CN ground state column density, by default [10.0, 1.0], where
            log10_N0_13CN ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tex : Iterable[float], optional
            Prior distribution on log10 cloud mean excitation temperature (K), by
            default [1.75, 0.25], where
            log10_Tex_mean ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 cloud kinetic temperature (K), by default
            [1.75, 0.25], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm_nonthermal : float, optional
            Prior distribution on non-thermal FWHM (km s-1), by default None, where
            fwhm_nonthermal ~ HalfNormal(sigma=prior_fwhm_nonthermal)
            If None, assume no non-thermal broadening.
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
            Assume local thermodynamic equilibrium by fixing the cloud mean excitation
            temperature to the cloud kinetic temperature, by default True. If True,
            the value passed to prior_log10_Tex has no effect.
        prior_log_boltz_factor_12CN_sigma : float, optional
            Prior distribution on log Boltzmann factor = -h*freq/(k*Tex) anomaly for CN, by default
            0.0, where
            log_boltz_factor_12CN_sigma ~ HalfNormal(sigma=prior)
            log_boltz_factor_12CN ~ normal(mu=log_boltz_factor_12CN_mean, sigma=log_boltz_factor_12CN_sigma)
            If 0.0, the Boltzmann factor for each CN transition is fixed based on the mean cloud
            excitation temperature. Note that setting this value greater than zero will allow
            for population inversions (i.e., negative excitation temperatures). We assume 13CN does
            not suffer from hyperfine anomalies.
        fix_log10_Tkin : Optional[float], optional
            Fix the log10 cloud kinetic temperature at this value (K), by default None. The posterior
            distribution is conditioned on this value of the kinetic temperature, which could bias the
            inference (especially if assume_LTE = True) but improve numerical stability and
            sampling efficiency.
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            velocity(cloud = n) ~
                prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # CN round state column density (shape: clouds)
            log10_N0_12CN_norm = pm.Normal("log10_N0_12CN_norm", mu=0.0, sigma=1.0, dims=["cloud"])
            log10_N0_12CN = pm.Deterministic(
                "log10_N0_12CN", prior_log10_N0_12CN[0] + prior_log10_N0_12CN[1] * log10_N0_12CN_norm, dims=["cloud"]
            )

            # 13CN round state column density (shape: clouds)
            log10_N0_13CN_norm = pm.Normal("log10_N0_13CN_norm", mu=0.0, sigma=1.0, dims=["cloud"])
            log10_N0_13CN = pm.Deterministic(
                "log10_N0_13CN", prior_log10_N0_13CN[0] + prior_log10_N0_13CN[1] * log10_N0_13CN_norm, dims=["cloud"]
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

            # Cloud average excitation temperature (K) (shape: clouds)
            if assume_LTE:
                # log10_Tex_mean is fixed
                log10_Tex_mean = pm.Deterministic("log10_Tex_mean", log10_Tkin, dims="cloud")
            else:
                log10_Tex_mean_norm = pm.Normal("log10_Tex_mean_norm", mu=0.0, sigma=1.0, dims="cloud")
                log10_Tex_mean = pm.Deterministic(
                    "log10_Tex_mean", prior_log10_Tex[0] + prior_log10_Tex[1] * log10_Tex_mean_norm, dims="cloud"
                )

            # 13CN excitation temperature (K) (shape: transitions_13CN, clouds)
            # No hyperfine anomalies for 13CN
            Tex_13CN = pm.Deterministic(
                "Tex_13CN",
                pt.repeat(10.0 ** log10_Tex_mean[None, :], len(self.mol_data_13CN["freq"]), 0),
                dims=["transition_13CN", "cloud"],
            )

            # 13CN log Boltzmann factors = log(gl*Nu) - log(gu*Nl) = -h*freq/(k*Tex) (shape: transitions_13CN, clouds)
            log_boltz_factor_13CN = pm.Deterministic(
                "log_boltz_factor_13CN",
                physics.calc_log_boltz_factor(self.mol_data_13CN["freq"][:, None], Tex_13CN),
                dims=["transition_13CN", "cloud"],
            )

            # 13CN state column densities (cm2; shape: states_13CN, clouds)
            _, log10_Nl_13CN, log10_Nu_13CN = physics.detailed_balance(
                self.mol_data_13CN, log10_N0=log10_N0_13CN, log_boltz_factor=log_boltz_factor_13CN, verbose=False
            )
            log10_Nl_13CN = pm.Deterministic("log10_Nl_13CN", pt.stack(log10_Nl_13CN), dims=["state_l_13CN", "cloud"])
            log10_Nu_13CN = pt.stack(log10_Nu_13CN)

            # Total 13CN column density across all states (cm-2; shape: clousd)
            log10_N_13CN = pm.Deterministic(
                "log10_N_13CN",
                pt.log10(pt.sum(10.0**log10_Nl_13CN, axis=0) + pt.sum(10.0**log10_Nu_13CN, axis=0)),
                dims="cloud",
            )

            if prior_log_boltz_factor_12CN_sigma == 0.0:
                # 12CN excitation temperature (K) (shape: transitions_12CN, clouds)
                # assumed constant across transitions
                Tex_12CN = pm.Deterministic(
                    "Tex_12CN",
                    pt.repeat(10.0 ** log10_Tex_mean[None, :], len(self.mol_data_12CN["freq"]), 0),
                    dims=["transition_12CN", "cloud"],
                )

                # 12CN log Boltzmann factors = log(gl*Nu) - log(gu*Nl) = -h*freq/(k*Tex) (shape: transitions_12CN, clouds)
                log_boltz_factor_12CN = pm.Deterministic(
                    "log_boltz_factor_12CN",
                    physics.calc_log_boltz_factor(self.mol_data_12CN["freq"][:, None], Tex_12CN),
                    dims=["transition_12CN", "cloud"],
                )

                # 12CN state column densities (cm2; shape: states_12CN, clouds)
                _, log10_Nl_12CN, log10_Nu_12CN = physics.detailed_balance(
                    self.mol_data_12CN, log10_N0=log10_N0_12CN, log_boltz_factor=log_boltz_factor_12CN, verbose=False
                )
            else:
                # CN Boltzmann factor anomaly (shape: clouds)
                log_boltz_factor_12CN_sigma_norm = pm.HalfNormal(
                    "log_boltz_factor_12CN_sigma_norm", sigma=1.0, dims="cloud"
                )
                log_boltz_factor_12CN_sigma = pm.Deterministic(
                    "log_boltz_factor_12CN_sigma",
                    prior_log_boltz_factor_12CN_sigma * log_boltz_factor_12CN_sigma_norm,
                    dims="cloud",
                )

                # CN Boltzmann factors derived from cloud mean excitation temperature (shape: transitions_free_12CN, clouds)
                log_boltz_factor_12CN_mean = pm.Deterministic(
                    "log_boltz_factor_12CN_mean",
                    physics.calc_log_boltz_factor(
                        np.array(self.model.coords["transition_free_12CN"])[:, None], 10.0**log10_Tex_mean
                    ),
                    dims=["transition_free_12CN", "cloud"],
                )

                # CN free log Boltzmann factors = log(gl*Nu) - log(gu*Nl) = -h*freq/(k*Tex) (shape: transitions_free_12CN, clouds)
                log_boltz_factor_12CN_free_norm = pm.Normal(
                    "log_boltz_factor_12CN_free_norm", mu=0.0, sigma=1.0, dims=["transition_free_12CN", "cloud"]
                )
                log_boltz_factor_12CN_free = pm.Deterministic(
                    "log_boltz_factor_12CN_free",
                    log_boltz_factor_12CN_mean + log_boltz_factor_12CN_sigma * log_boltz_factor_12CN_free_norm,
                    dims=["transition_free_12CN", "cloud"],
                )

                # CN State column densities
                log_boltz_factor_12CN = [
                    (
                        log_boltz_factor_12CN_free[self.model.coords["transition_free"].index(freq)]
                        if freq in self.model.coords["transition_free_12CN"]
                        else None
                    )
                    for freq in self.mol_data_12CN["freq"]
                ]
                _, log10_Nl_12CN, log10_Nu_12CN = physics.detailed_balance(
                    self.mol_data_12CN, log10_N0=log10_N0_12CN, log_boltz_factor=log_boltz_factor_12CN, verbose=False
                )

                # log Boltzman factors (shape: transitions, clouds)
                log_boltz_factor_12CN = pm.Deterministic(
                    "log_boltz_factor_12CN",
                    pt.stack(
                        [
                            (
                                log_boltz_factor_12CN_free[self.model.coords["transition_free"].index(freq)]
                                if freq in self.model.coords["transition_free"]
                                else pt.log(10.0)
                                * (pt.log10(Gl / Gu) + log10_Nu_12CN[state_u] - log10_Nl_12CN[state_l])
                            )
                            for freq, Gl, Gu, state_l, state_u in zip(
                                self.mol_data_12CN["freq"],
                                self.mol_data_12CN["Gl"],
                                self.mol_data_12CN["Gu"],
                                self.mol_data_12CN["state_l"],
                                self.mol_data_12CN["state_u"],
                            )
                        ]
                    ),
                    dims=["transition_12CN", "cloud"],
                )

                # Excitation temperarature (K; shape: transitions, clouds)
                Tex_12CN = pm.Deterministic(
                    "Tex_12CN",
                    physics.calc_Tex(self.mol_data_12CN["freq"][:, None], log_boltz_factor_12CN),
                    dims=["transition_12CN", "cloud"],
                )

            # CN state column densities (cm2; shape: states_12CN, clouds)
            log10_Nl_12CN = pm.Deterministic("log10_Nl_12CN", pt.stack(log10_Nl_12CN), dims=["state_l_12CN", "cloud"])
            log10_Nu_12CN = pt.stack(log10_Nu_12CN)

            # Total CN column density across all states (cm-2; shape: clousd)
            log10_N_12CN = pm.Deterministic(
                "log10_N_12CN",
                pt.log10(pt.sum(10.0**log10_Nl_12CN, axis=0) + pt.sum(10.0**log10_Nu_12CN, axis=0)),
                dims="cloud",
            )

            # 13C/12C ratio (shape: clouds)
            _ = pm.Deterministic("log10_13C_12C_ratio", log10_N_13CN - log10_N_12CN, dims="cloud")

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

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal_12CN = pm.Deterministic(
                "fwhm_thermal_12CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_12CN), dims="cloud"
            )
            fwhm_thermal_13CN = pm.Deterministic(
                "fwhm_thermal_13CN", physics.calc_thermal_fwhm(10.0**log10_Tkin, self.mol_weight_13CN), dims="cloud"
            )

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = 0.0
            if prior_fwhm_nonthermal is not None:
                fwhm_nonthermal_norm = pm.HalfNormal("fwhm_nonthermal_norm", sigma=1.0, dims="cloud")
                fwhm_nonthermal = pm.Deterministic(
                    "fwhm_nonthermal", prior_fwhm_nonthermal * fwhm_nonthermal_norm, dims="cloud"
                )

            # Spectral rms (K)
            if prior_rms is not None:
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms)

            # Total (physical) FWHM (km/s; shape: clouds)
            fwhm_12CN = pm.Deterministic(
                "fwhm_12CN", pt.sqrt(fwhm_thermal_12CN**2.0 + fwhm_nonthermal**2.0), dims="cloud"
            )
            fwhm_13CN = pm.Deterministic(
                "fwhm_13CN", pt.sqrt(fwhm_thermal_13CN**2.0 + fwhm_nonthermal**2.0), dims="cloud"
            )

            # Pseudo-Voigt profile latent variable (km/s)
            fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
            fwhm_L = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)

            # line profile amplitude (km-1 s; shape: transitions, clouds)
            line_profile_amplitude_12CN = physics.calc_line_profile_amplitude(fwhm_12CN, fwhm_L)
            line_profile_amplitude_13CN = physics.calc_line_profile_amplitude(fwhm_13CN, fwhm_L)

            # Get lower state column density for each transition (shape: transitions, clouds)
            Nl_12CN = pt.stack([10.0 ** log10_Nl_12CN[state_l] for state_l in self.mol_data_12CN["state_l"]])
            Nl_13CN = pt.stack([10.0 ** log10_Nl_13CN[state_l] for state_l in self.mol_data_13CN["state_l"]])

            # peak optical depths (shape: transitions, clouds)
            tau_peak_12CN = pm.Deterministic(
                "tau_peak_12CN",
                physics.calc_optical_depth(
                    self.mol_data_12CN["Gu"][:, None],
                    self.mol_data_12CN["Gl"][:, None],
                    Nl_12CN,
                    log_boltz_factor_12CN,
                    line_profile_amplitude_12CN,
                    self.mol_data_12CN["freq"][:, None],
                    self.mol_data_12CN["Aul"][:, None],
                ),
                dims=["transition_12CN", "cloud"],
            )
            tau_peak_13CN = pm.Deterministic(
                "tau_peak_13CN",
                physics.calc_optical_depth(
                    self.mol_data_13CN["Gu"][:, None],
                    self.mol_data_13CN["Gl"][:, None],
                    Nl_13CN,
                    log_boltz_factor_13CN,
                    line_profile_amplitude_13CN,
                    self.mol_data_13CN["freq"][:, None],
                    self.mol_data_13CN["Aul"][:, None],
                ),
                dims=["transition_13CN", "cloud"],
            )

            # total optical depth (shape: clouds)
            _ = pm.Deterministic("tau_total_12CN", pt.sum(tau_peak_12CN, axis=0), dims="cloud")
            _ = pm.Deterministic("tau_total_13CN", pt.sum(tau_peak_13CN, axis=0), dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            if "12CN" in label:
                mol_data = self.mol_data_12CN
                Nl = pt.stack([10.0 ** self.model["log10_Nl_12CN"][state_l] for state_l in mol_data["state_l"]])
                log_boltz_factor = self.model["log_boltz_factor_12CN"]
                fwhm = self.model["fwhm_12CN"]
                Tex = self.model["Tex_12CN"]
            elif "13CN" in label:
                mol_data = self.mol_data_13CN
                Nl = pt.stack([10.0 ** self.model["log10_Nl_13CN"][state_l] for state_l in mol_data["state_l"]])
                log_boltz_factor = self.model["log_boltz_factor_13CN"]
                fwhm = self.model["fwhm_13CN"]
                Tex = self.model["Tex_13CN"]
            else:
                raise ValueError(f"Invalid dataset label: {label}")

            # Derive optical depth spectra (shape: spectral, clouds)
            tau = physics.predict_tau(
                mol_data,
                dataset.spectral,
                Nl,
                log_boltz_factor,
                self.model["velocity"],
                fwhm,
                self.model["fwhm_L"],
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau,
                    Tex,
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
