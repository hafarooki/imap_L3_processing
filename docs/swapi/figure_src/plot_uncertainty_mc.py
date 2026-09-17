#!/usr/bin/env python3
"""
MC validation of the proton-fit Huber–White sandwich uncertainty estimator.

Holds one typical solar-wind ground truth fixed and varies only the Poisson
seed across many synthetic 5-sweep chunks. The width of the resulting
distribution of fitted parameters is the *true* sampling sigma; the
distribution of per-fit estimated sigmas is what the sandwich estimator
reports. A correctly-calibrated estimator places the histogram of estimated
sigmas (row 2) at the spread of the histogram of fitted parameters (row 1).

Output: docs/swapi/figures/uncertainty_mc.png
Usage:  conda run -n imapL3 python docs/swapi/figure_src/plot_uncertainty_mc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import types

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import SWAPI_LIVETIME_S
from figure_utils import (
    NOMINAL_EPOCH_TT2000,
    COARSE_BIN_INDICES_IN_SWEEP,
    COARSE_SWEEP_VOLTAGES_MEAN_V,
    FIGURES_DIR,
    compute_per_bin_rotation_matrices,
    load_swapi_response,
    run_parallel_map,
    save_figure,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.calculate_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    _derive_uncertainties,
    escape_local_minimum,
    optimize_solar_wind_proton_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    VELOCITY_SLICE,
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.species import Species

_N_SWEEPS = 5
_N_BINS = len(COARSE_SWEEP_VOLTAGES_MEAN_V)

# Typical moderate-speed solar-wind ground truth: density 5 cm^-3, T 1e5 K,
# bulk +450 km/s radial with small T/N components.
_TRUTH_DENSITY_CM3 = 5.0
_TRUTH_TEMPERATURE_K = 1.0e5
_TRUTH_BULK_RTN_KM_S = np.array([450.0, 5.0, -3.0])
_N_MC_SAMPLES = 1000
_LOGNORMAL_REL_SIGMA = 0.01  # 1% relative error in count rates
# Energy-independent Poisson noise floor (Hz), applied alongside the log-normal
# multiplicative case. Models a constant detector dark/leakage rate that the
# proton model does not account for — every bin gets Poisson(floor·τ) counts
# added on top of the signal.
_NOISE_FLOOR_HZ = 10.0

_worker_state: types.SimpleNamespace | None = None


def main():
    _initialize_worker_state()
    poisson_lognormal_floor_data = _run_fits(_N_MC_SAMPLES)
    _plot_results(
        [
            (
                f"Poisson + {_LOGNORMAL_REL_SIGMA:.0%} log-normal "
                f"+ {_NOISE_FLOOR_HZ:g} Hz floor",
                poisson_lognormal_floor_data,
            ),
        ]
    )


def _initialize_worker_state() -> None:
    global _worker_state

    swapi_response = load_swapi_response()
    all_esa_voltages = np.tile(COARSE_SWEEP_VOLTAGES_MEAN_V, _N_SWEEPS)
    swapi_response.warm_cache(all_esa_voltages)
    per_bin_rotation_matrices = compute_per_bin_rotation_matrices(
        _N_SWEEPS, COARSE_BIN_INDICES_IN_SWEEP
    )
    base_ctx = build_solar_wind_fit_context(
        count_rate=np.ones_like(all_esa_voltages),
        esa_voltage=all_esa_voltages,
        swapi_response=swapi_response,
        time_as_tt2000=NOMINAL_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=per_bin_rotation_matrices,
    )

    truth_params = SolarWindParams(
        density=_TRUTH_DENSITY_CM3,
        velocity_rtn=_TRUTH_BULK_RTN_KM_S.copy(),
        temperature=_TRUTH_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(truth_params, base_ctx)
    truth_count_rates = ideal_rates * deadtime_factor(ideal_rates)

    _worker_state = types.SimpleNamespace(
        swapi_response=swapi_response,
        all_esa_voltages=all_esa_voltages,
        per_bin_rotation_matrices=per_bin_rotation_matrices,
        truth_count_rates=truth_count_rates,
    )


def _run_fits(n_samples: int) -> pd.DataFrame:
    rows = run_parallel_map(_process_one, n_samples, desc="mc", chunksize=10)
    data = pd.DataFrame(rows)
    print(f"Bad-fit flags: {data['bad_flag'].sum()}/{n_samples}")
    return data


def _process_one(i):
    ws = _worker_state
    rng = np.random.default_rng(i)
    counts = rng.poisson(np.maximum(ws.truth_count_rates * SWAPI_LIVETIME_S, 0.0))
    count_rates = counts.astype(float) / SWAPI_LIVETIME_S
    # Multiplicative log-normal noise on top of Poisson sampling, with
    # std-dev ≈ _LOGNORMAL_REL_SIGMA in linear space. Mean-1 correction:
    # shift by -σ²/2 in log space.
    sigma_log = np.sqrt(np.log1p(_LOGNORMAL_REL_SIGMA**2))
    log_factor = rng.normal(-0.5 * sigma_log**2, sigma_log, count_rates.shape)
    count_rates = np.maximum(count_rates * np.exp(log_factor), 0.0)
    # Energy-independent Poisson noise floor: every bin gets Poisson(floor·τ)
    # counts added — a stand-in for unmodelled detector dark/leakage rate.
    # The mean adds 10 Hz to obs at every bin; the variance is √(floor·τ)/τ
    # ≈ 8 Hz per bin. The fit's model does not account for this.
    floor_counts = rng.poisson(
        _NOISE_FLOOR_HZ * SWAPI_LIVETIME_S, size=count_rates.shape
    )
    count_rates = count_rates + floor_counts.astype(float) / SWAPI_LIVETIME_S

    fit_ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=ws.all_esa_voltages,
        swapi_response=ws.swapi_response,
        time_as_tt2000=NOMINAL_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=ws.per_bin_rotation_matrices,
    )
    sigma_keys = [
        "fit_density_sigma",
        "fit_temperature_sigma",
        "fit_radial_speed_sigma",
        "fit_tangential_speed_sigma",
        "fit_normal_speed_sigma",
    ]
    nan_row = {
        k: float("nan")
        for k in [
            "fit_density",
            "fit_temperature",
            "fit_radial_speed",
            "fit_tangential_speed",
            "fit_normal_speed",
            *sigma_keys,
            *(f"ols_{k}" for k in sigma_keys),
        ]
    } | {"bad_flag": True}
    try:
        initial_guess = calculate_initial_guess(fit_ctx)
        first_result = optimize_solar_wind_proton_params(initial_guess, fit_ctx)
        final_result = escape_local_minimum(first_result, fit_ctx)
    except Exception as e:
        print(f"  case {i}: fit failed ({type(e).__name__}: {e})")
        return nan_row

    sw = final_result.sw_params
    sandwich_n_sigma, sandwich_T_sigma, sandwich_v_cov = _derive_uncertainties(
        final_result, fit_ctx
    )
    ols_n_sigma, ols_T_sigma, ols_v_sigma = _uncorrected_sigmas(final_result)
    sandwich_v_sigma = np.sqrt(np.maximum(np.diag(sandwich_v_cov), 0.0))

    return {
        "fit_density": sw.density,
        "fit_temperature": sw.temperature,
        "fit_radial_speed": float(sw.velocity_rtn[0]),
        "fit_tangential_speed": float(sw.velocity_rtn[1]),
        "fit_normal_speed": float(sw.velocity_rtn[2]),
        "fit_density_sigma": sandwich_n_sigma,
        "fit_temperature_sigma": sandwich_T_sigma,
        "fit_radial_speed_sigma": float(sandwich_v_sigma[0]),
        "fit_tangential_speed_sigma": float(sandwich_v_sigma[1]),
        "fit_normal_speed_sigma": float(sandwich_v_sigma[2]),
        "ols_fit_density_sigma": ols_n_sigma,
        "ols_fit_temperature_sigma": ols_T_sigma,
        "ols_fit_radial_speed_sigma": float(ols_v_sigma[0]),
        "ols_fit_tangential_speed_sigma": float(ols_v_sigma[1]),
        "ols_fit_normal_speed_sigma": float(ols_v_sigma[2]),
        "bad_flag": bool(not final_result.success),
    }


def _uncorrected_sigmas(opt_result) -> tuple[float, float, np.ndarray]:
    """Previous uncorrected formula: σ² = (Σr²/(N − K)) · pinv(JᵀJ). Assumes
    homoscedastic residuals; replaced by the Huber–White sandwich. Reproduced
    here for MC comparison only."""
    J = opt_result.jacobian
    n_params = J.shape[1]
    n_obs = len(opt_result.residuals)
    residual_variance = float(np.sum(opt_result.residuals**2)) / max(
        n_obs - n_params, 1
    )
    try:
        cov = residual_variance * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), np.full(3, float("nan"))
    log_n_var = cov[LOG_DENSITY_IDX, LOG_DENSITY_IDX]
    log_T_var = cov[LOG_TEMPERATURE_IDX, LOG_TEMPERATURE_IDX]
    velocity_cov = cov[VELOCITY_SLICE, VELOCITY_SLICE]
    sw = opt_result.sw_params
    return (
        float(sw.density * np.sqrt(max(log_n_var, 0.0))),
        float(sw.temperature * np.sqrt(max(log_T_var, 0.0))),
        np.sqrt(np.maximum(np.diag(velocity_cov), 0.0)),
    )


def _plot_results(experiments: list[tuple[str, pd.DataFrame]]) -> None:
    columns = [
        (
            "Density (cm⁻³)",
            "fit_density",
            "fit_density_sigma",
            "ols_fit_density_sigma",
            _TRUTH_DENSITY_CM3,
        ),
        (
            "Temperature (K)",
            "fit_temperature",
            "fit_temperature_sigma",
            "ols_fit_temperature_sigma",
            _TRUTH_TEMPERATURE_K,
        ),
        (
            "$v_R$ (km/s)",
            "fit_radial_speed",
            "fit_radial_speed_sigma",
            "ols_fit_radial_speed_sigma",
            _TRUTH_BULK_RTN_KM_S[0],
        ),
        (
            "$v_T$ (km/s)",
            "fit_tangential_speed",
            "fit_tangential_speed_sigma",
            "ols_fit_tangential_speed_sigma",
            _TRUTH_BULK_RTN_KM_S[1],
        ),
        (
            "$v_N$ (km/s)",
            "fit_normal_speed",
            "fit_normal_speed_sigma",
            "ols_fit_normal_speed_sigma",
            _TRUTH_BULK_RTN_KM_S[2],
        ),
    ]

    n_experiments = len(experiments)
    fig, axes = plt.subplots(2 * n_experiments, 5, figsize=(17, 6.5 * n_experiments))
    if n_experiments == 1:
        axes = np.array([axes[0], axes[1]]).reshape(2, 5)
    fig.suptitle(
        f"Sandwich-estimator MC validation: one truth\n"
        f"(n={_TRUTH_DENSITY_CM3:g} cm⁻³, T={_TRUTH_TEMPERATURE_K:.0e} K, "
        f"v_RTN=({_TRUTH_BULK_RTN_KM_S[0]:g}, {_TRUTH_BULK_RTN_KM_S[1]:g}, "
        f"{_TRUTH_BULK_RTN_KM_S[2]:g}) km/s, "
        f"{_N_SWEEPS} sweeps × {_N_BINS} coarse bins)",
        fontsize=11,
    )

    color_fits = "tab:blue"
    color_sandwich = "tab:orange"
    color_ols = "tab:green"

    for exp_idx, (exp_label, data) in enumerate(experiments):
        good = ~data["bad_flag"]
        n_good = int(good.sum())
        row_top = 2 * exp_idx
        row_bot = 2 * exp_idx + 1
        # Group label on the left margin
        fig.text(
            0.005,
            1.0 - (exp_idx + 0.5) / n_experiments,
            f"{exp_label}\n(n={n_good} MC samples)",
            ha="left",
            va="center",
            fontsize=10,
            rotation=90,
            weight="bold",
        )

        for col_idx, (
            label,
            fit_key,
            sigma_key,
            ols_sigma_key,
            truth,
        ) in enumerate(columns):
            fits = data.loc[good, fit_key].to_numpy()
            sigmas = data.loc[good, sigma_key].to_numpy()
            ols_sigmas = data.loc[good, ols_sigma_key].to_numpy()
            empirical_sigma = float(np.nanstd(fits, ddof=1))
            bias = float(np.nanmean(fits)) - truth

            ax_top = axes[row_top, col_idx]
            ax_top.hist(
                fits,
                bins=40,
                color=color_fits,
                alpha=0.75,
                edgecolor="white",
                rasterized=True,
            )
            ax_top.axvline(truth, color="k", lw=1.0, ls="--", alpha=0.7, label="truth")
            ax_top.set_title(label, fontsize=10)
            ax_top.set_xlabel(f"Fitted {label}", fontsize=9)
            ax_top.set_ylabel("MC count", fontsize=9)
            ax_top.tick_params(labelsize=8)
            ax_top.grid(True, alpha=0.25)
            ax_top.annotate(
                f"empirical σ = {_format_sigma(empirical_sigma, label)}\n"
                f"bias (mean − truth) = {_format_signed_sigma(bias, label)} "
                f"({bias / empirical_sigma:+.2f} σ)",
                xy=(0.04, 0.97),
                xycoords="axes fraction",
                va="top",
                ha="left",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            if col_idx == 0:
                ax_top.legend(fontsize=7, loc="upper right", framealpha=0.8)

            # Bottom row: histograms of normalized residuals (fit − truth) / σ̂.
            # A well-calibrated σ̂ estimator gives a unit Gaussian; a σ̂ that
            # understates the true scatter by factor k gives a Gaussian of
            # width k. Tells the reader at a glance whether the bars are right.
            ax_bot = axes[row_bot, col_idx]
            normalized_resid_sandwich = (fits - truth) / sigmas
            normalized_resid_ols = (fits - truth) / ols_sigmas
            finite_sw = normalized_resid_sandwich[
                np.isfinite(normalized_resid_sandwich)
            ]
            finite_ols_n = normalized_resid_ols[np.isfinite(normalized_resid_ols)]
            x_max = float(np.nanmax(np.abs(np.concatenate([finite_sw, finite_ols_n]))))
            x_max = max(x_max, 4.0)  # always show at least ±4σ
            bins_norm = np.linspace(-x_max, x_max, 51)
            ax_bot.hist(
                normalized_resid_ols,
                bins=bins_norm,
                color=color_ols,
                alpha=0.6,
                edgecolor="white",
                label="Uncorrected",
                rasterized=True,
            )
            ax_bot.hist(
                normalized_resid_sandwich,
                bins=bins_norm,
                color=color_sandwich,
                alpha=0.75,
                edgecolor="white",
                label="HC3 sandwich",
                rasterized=True,
            )
            # Reference: the unit-σ Gaussian a calibrated estimator should produce.
            n_good_finite = max(len(finite_sw), 1)
            bin_width = bins_norm[1] - bins_norm[0]
            x_ref = np.linspace(-x_max, x_max, 200)
            unit_gaussian_pdf = (
                (np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi))
                * n_good_finite
                * bin_width
            )
            ax_bot.plot(
                x_ref,
                unit_gaussian_pdf,
                color="k",
                lw=1.2,
                ls="--",
                alpha=0.7,
                label="N(0, 1) (calibrated)",
            )
            sandwich_calibration = float(np.nanstd(finite_sw, ddof=1))
            ols_calibration = float(np.nanstd(finite_ols_n, ddof=1))
            ax_bot.set_xlabel(f"(fit − truth) / σ̂ for {label}", fontsize=9)
            ax_bot.set_ylabel("MC count", fontsize=9)
            ax_bot.tick_params(labelsize=8)
            ax_bot.grid(True, alpha=0.25)
            ax_bot.annotate(
                f"std of (fit − truth)/σ̂_HC3 = {sandwich_calibration:.2f}\n"
                f"std of (fit − truth)/σ̂_uncorrected = {ols_calibration:.2f}",
                xy=(0.04, 0.97),
                xycoords="axes fraction",
                va="top",
                ha="left",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            if col_idx == 0:
                ax_bot.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "uncertainty_mc.png"
    save_figure(fig, out_path, dpi=200)
    print(f"Saved {out_path}")


def _format_sigma(value: float, label: str) -> str:
    if "Density" in label:
        return f"{value:.3f} cm⁻³"
    if "Temperature" in label:
        return f"{value:.2e} K"
    return f"{value:.2f} km/s"


def _format_signed_sigma(value: float, label: str) -> str:
    if "Density" in label:
        return f"{value:+.3f} cm⁻³"
    if "Temperature" in label:
        return f"{value:+.2e} K"
    return f"{value:+.2f} km/s"


if __name__ == "__main__":
    main()
