#!/usr/bin/env python3
"""
Plot alpha peak-finding walkthrough on real L2 SWAPI spectra.

Three-panel figure (one per real-data fixture: strong alpha, hot plasma,
cold plasma) illustrating the `get_alpha_peak_indices` peak-finder and the
subsequent Gaussian fit on the count-rate residual:

  Panel top: 5-sweep-averaged observed count rate vs ESA voltage, overlaid
      with the frozen proton model and the detected alpha peak region shaded.
  Panel bottom: residual (observed − proton model) at the peak bins, with
      the fitted Gaussian.

Spectra are extracted from imap_swapi_l2_sci_20260101_v005.cdf; Stage 1
proton fits and rotation matrices are pre-computed in the test fixture at
tests/test_data/swapi/alpha_fit_test_spectra.npz.

Output: docs/swapi/figures/alpha_peak_finding.png
Usage:  python docs/swapi/figure_src/plot_alpha_peak_finding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.optimize

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_moments import (
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    _model_count_rates,
    apply_deadtime_correction_array,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_K_FACTOR,
    esa_voltage_to_alpha_speed,
    get_alpha_peak_indices,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_FIXTURE_PATH = (
    _REPO_ROOT / "tests" / "test_data" / "swapi" / "alpha_fit_test_spectra.npz"
)
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_N_SWEEPS = 5
_N_BINS = 62

_CASES = [
    ("strong_alpha", "Strong alpha (chunk 384)"),
    ("hot_plasma", "Hot plasma (chunk 250)"),
    ("cold_plasma", "Cold plasma (chunk 550)"),
]


def _load_fixture(data, name):
    prefix = f"{name}__"
    return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}


def _plot_case(axes_top, axes_bot, sr, f, title):
    count_rates = f["count_rates"]  # (5, 62)
    voltage_per_sweep = f["voltage_per_sweep"]  # (62,)
    esa_flat = f["esa_flat"]  # (310,)
    rotation_matrices = f["rotation_matrices"]  # (310, 3, 3)
    proton_density = float(f["proton_density"])
    proton_temperature = float(f["proton_temperature"])
    proton_velocity_rtn = f["proton_velocity_rtn"]
    proton_eff_scale = float(f["proton_eff_scale"])
    alpha_eff_scale = float(f["alpha_eff_scale"])
    b_hat_rtn = f["b_hat_rtn"]
    cr_flat = f["cr_flat"]  # (310,)

    # Build passband grids and model arrays
    grids = numba.typed.List([sr.create_passband_grid(v) for v in esa_flat])
    p_cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in esa_flat])
    p_cea = cea * proton_eff_scale
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    # Frozen proton model
    proton_true = _model_count_rates(
        proton_density,
        proton_temperature,
        proton_velocity_rtn,
        grids,
        p_cs,
        p_cea,
        at,
        ats,
        rotation_matrices,
        PROTON_MASS_KG,
    )
    proton_obs_per_sweep = apply_deadtime_correction_array(
        proton_true.reshape(_N_SWEEPS, _N_BINS)
    )
    proton_bg_avg = proton_obs_per_sweep.mean(axis=0)
    count_avg = count_rates.mean(axis=0)

    # Stage-2 alpha moments fit
    proton_moments_obj = ProtonSolarWindMoments(
        density=proton_density,
        temperature=proton_temperature,
        bulk_velocity_rtn=proton_velocity_rtn,
        bad_fit_flag=int(f["proton_bad_fit_flag"]),
        velocity_covariance=f["proton_velocity_covariance"],
    )
    alpha_moments = fit_solar_wind_alpha_moments(
        count_rate=cr_flat,
        esa_voltage=esa_flat,
        measurement_time=np.zeros(len(esa_flat)),  # unused: rotation_matrices provided
        swapi_response=sr,
        proton_moments=proton_moments_obj,
        b_hat_rtn=b_hat_rtn,
        alpha_effective_area_scale=alpha_eff_scale,
        proton_effective_area_scale=proton_eff_scale,
        rotation_matrices=rotation_matrices,
    )
    combined_fit_avg = None
    alpha_contribution_avg = None
    if np.isfinite(alpha_moments.density):
        alpha_cs = np.array(
            [sr.central_speed(v, ALPHA_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
        )
        alpha_cea = cea * alpha_eff_scale
        alpha_true_fit = _model_count_rates(
            alpha_moments.density,
            alpha_moments.temperature,
            alpha_moments.bulk_velocity_rtn,
            grids,
            alpha_cs,
            alpha_cea,
            at,
            ats,
            rotation_matrices,
            ALPHA_PARTICLE_MASS_KG,
        )
        combined_fit = apply_deadtime_correction_array(proton_true + alpha_true_fit)
        combined_fit_avg = combined_fit.reshape(_N_SWEEPS, _N_BINS).mean(axis=0)
        alpha_contribution_avg = np.maximum(combined_fit_avg - proton_bg_avg, 0.0)

    # Run peak finder
    energies = SWAPI_K_FACTOR * np.abs(voltage_per_sweep)
    peak = get_alpha_peak_indices(count_avg - proton_bg_avg, energies, count_avg.argmax())
    peak_idx = np.arange(peak.start, peak.stop)
    residual_peak = np.maximum(count_avg[peak_idx] - proton_bg_avg[peak_idx], 0.0)

    # Gaussian fit on residual
    speed_peak = esa_voltage_to_alpha_speed(voltage_per_sweep[peak_idx])
    p0 = [residual_peak.max(), speed_peak[int(np.argmax(residual_peak))], 50.0]
    try:
        (A_fit, mu_fit, sigma_fit), _ = scipy.optimize.curve_fit(
            lambda v, A, mu, sigma: A * np.exp(-((v - mu) ** 2) / (2 * sigma**2)),
            speed_peak,
            residual_peak,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError:
        A_fit = residual_peak.max()
        mu_fit = float(speed_peak[int(np.argmax(residual_peak))])
        sigma_fit = 50.0

    abs_v = np.abs(voltage_per_sweep)
    sort_idx = np.argsort(abs_v)
    abs_v_s = abs_v[sort_idx]

    # --- Top panel: count rates and proton model ---
    axes_top.plot(
        abs_v_s,
        count_avg[sort_idx],
        ".",
        color="tab:blue",
        markersize=4,
        label="Observed (5-sweep avg)",
        zorder=3,
    )
    axes_top.plot(
        abs_v_s,
        proton_bg_avg[sort_idx],
        color="tab:orange",
        lw=1.5,
        label="Proton model",
        zorder=2,
    )
    if combined_fit_avg is not None:
        axes_top.plot(
            abs_v_s,
            combined_fit_avg[sort_idx],
            color="tab:purple",
            lw=1.5,
            linestyle="--",
            label="Combined fit (p + α)",
            zorder=2,
        )

    # Shade the alpha peak region
    peak_voltages = abs_v[peak_idx]
    v_lo, v_hi = peak_voltages.min(), peak_voltages.max()
    axes_top.axvspan(
        v_lo, v_hi, alpha=0.15, color="tab:green", label="Alpha peak", zorder=1
    )

    # Mark peak bins on observed data
    axes_top.plot(
        abs_v[peak_idx],
        count_avg[peak_idx],
        "o",
        color="tab:green",
        markersize=5,
        markerfacecolor="none",
        lw=1.2,
        zorder=4,
    )

    axes_top.set_xscale("log")
    axes_top.set_yscale("log")
    axes_top.set_ylim(bottom=0.5)
    axes_top.set_ylabel("Count rate [Hz]")
    axes_top.set_title(title, fontsize=10)
    axes_top.legend(fontsize=7, loc="upper left")
    axes_top.grid(True, which="both", alpha=0.2)

    # --- Bottom panel: residual and Gaussian fit ---
    all_speeds = esa_voltage_to_alpha_speed(voltage_per_sweep)
    all_residual = np.maximum(count_avg - proton_bg_avg, 0.0)

    # Full residual as thin grey bars
    axes_bot.vlines(
        all_speeds[sort_idx],
        0,
        all_residual[sort_idx],
        colors="lightgrey",
        linewidth=1.5,
        zorder=1,
    )
    axes_bot.plot(
        all_speeds[sort_idx],
        all_residual[sort_idx],
        ".",
        color="grey",
        markersize=3,
        zorder=1,
        label="Residual (all bins)",
    )

    # Highlight peak bins
    axes_bot.vlines(
        speed_peak, 0, residual_peak, colors="tab:green", linewidth=2, zorder=2
    )
    axes_bot.plot(
        speed_peak,
        residual_peak,
        "o",
        color="tab:green",
        markersize=5,
        zorder=2,
        label="Peak bins",
    )

    # Gaussian fit curve (initial-guess estimator)
    v_fine = np.linspace(speed_peak.min() - 50, speed_peak.max() + 50, 200)
    gauss_fine = A_fit * np.exp(-((v_fine - mu_fit) ** 2) / (2 * sigma_fit**2))
    axes_bot.plot(
        v_fine,
        gauss_fine,
        color="tab:red",
        lw=1.5,
        linestyle=":",
        label=rf"Gaussian (init. guess): $v_\alpha$={mu_fit:.0f} km/s",
        zorder=3,
    )

    # Full moments fit
    if alpha_contribution_avg is not None:
        alpha_speed = np.linalg.norm(alpha_moments.bulk_velocity_rtn)
        axes_bot.plot(
            all_speeds[sort_idx],
            alpha_contribution_avg[sort_idx],
            color="tab:purple",
            lw=1.5,
            linestyle="--",
            label=rf"Moments fit: $v_\alpha$={alpha_speed:.0f} km/s",
            zorder=4,
        )

    axes_bot.set_xlabel(r"$\alpha$ speed [km/s]")
    axes_bot.set_ylabel("Residual [Hz]")
    axes_bot.legend(fontsize=7, loc="upper right")
    axes_bot.grid(True, alpha=0.2)
    y_top = max(residual_peak.max(), A_fit) * 1.3
    axes_bot.set_ylim(0, max(y_top, 10))


def main():
    print("Loading calibration data...")
    sr = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    data = np.load(_FIXTURE_PATH)
    n_cases = len(_CASES)
    fig, axes = plt.subplots(
        2,
        n_cases,
        figsize=(4.5 * n_cases, 7),
        gridspec_kw={"height_ratios": [2, 1.2]},
    )
    fig.suptitle(
        r"Alpha peak-finding on real L2 spectra (imap\_swapi\_l2\_sci\_20260101)",
        fontsize=11,
    )

    for col, (case_name, case_title) in enumerate(_CASES):
        print(f"Plotting {case_name}...")
        f = _load_fixture(data, case_name)
        _plot_case(axes[0, col], axes[1, col], sr, f, case_title)

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "alpha_peak_finding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
