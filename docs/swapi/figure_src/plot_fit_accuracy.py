#!/usr/bin/env python3
"""
Scatter plots comparing the SWAPI proton-moments fit against ground truth on
**real solar wind conditions** sampled from WIND/SWE 2-min ASCII data.

Uses the 62 coarse-sweep bins (indices 1..62 of a 72-bin sweep) to show that
the fit recovers (n, T, v_R, v_T, v_N) without fine-sweep coverage.

Per-bin RTN→SWAPI rotation matrices are generated synthetically by spinning a
single anchor matrix (a real SPICE attitude near 2026-01-01) about its own
spin axis at the nominal SWAPI spin period.

Synthetic count rates are produced from the SWAPI forward model (5 sweeps per
fit, Poisson noise); the ground truth is read from a CSV produced by
scripts/swapi/sample_wind_solar_wind.py.

Generate the CSV first:
  conda run -n imapL3 python scripts/swapi/sample_wind_solar_wind.py \
      --year 2025 --n 10000 --seed 7

Output: docs/swapi/figures/fit_accuracy.png
Usage:  conda run -n imapL3 python docs/swapi/figure_src/plot_fit_accuracy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import time
import types
import multiprocessing

import numpy as np
import pandas as pd
import numba
from tqdm.contrib.concurrent import process_map
from uncertainties import UFloat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from figure_utils import load_swapi_response
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    fit_solar_wind_proton_moments,
)
from imap_l3_processing.swapi.l3a.science.proton_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    SolarWindParams,
    apply_deadtime_correction_array,
    model_solar_wind_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    build_solar_wind_fit_context,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_N_SWEEPS = 5
_SWEEP_DURATION_S = 12.0  # full 72-bin SWAPI sweep cadence
_BINS_PER_SWEEP = 72
_DT_S = _SWEEP_DURATION_S / _BINS_PER_SWEEP
_SPIN_PERIOD_S = 15.13  # typical IMAP spin period

# Coarse-sweep bin indices within a 72-bin sweep (1-indexed): 1..62.
_BIN_INDICES_IN_SWEEP = np.arange(1, 63)

# Mean SWAPI L2 coarse-sweep voltages (V), descending — bins 1..62 of the
# 72-bin sweep, averaged over many real sweeps.
_VOLTAGES = np.array([
     9895.52,  9088.69,  8348.80,  7667.55,  7042.16,  6469.31,  5941.77,  5457.31,
     5013.22,  4603.65,  4230.77,  3886.92,  3569.16,  3278.72,  3011.13,  2766.25,
     2539.54,  2333.83,  2144.24,  1969.31,  1808.74,  1660.86,  1525.75,  1401.82,
     1287.58,  1182.24,  1085.15,   995.55,   914.31,   839.94,   771.70,   709.46,
      651.59,   598.47,   549.91,   505.12,   463.89,   425.92,   391.18,   359.35,
      329.94,   303.02,   278.25,   255.55,   234.77,   215.61,   197.95,   181.82,
      167.04,   153.46,   140.91,   129.50,   118.91,   109.20,   100.30,    92.11,
       84.61,    77.73,    71.40,    65.59,    60.23,    55.34,
])
_N_BINS = len(_VOLTAGES)
assert _VOLTAGES.shape == _BIN_INDICES_IN_SWEEP.shape == (62,)


# Single RTN→SWAPI anchor matrix from a real SPICE attitude near 2026-01-01,
# at the first sweep midpoint (t = _ANCHOR_TIME_S). Reflects the ~4° spin-axis
# tilt off -R̂_RTN. Per-bin matrices are built by spinning this anchor about
# its own +Y row (the spin axis in RTN) at the nominal SWAPI spin period.
_ANCHOR_ROTATION_MATRIX = np.array([
    [+0.0705, +0.9157, +0.3955],
    [-0.9968, +0.0792, -0.0057],
    [-0.0365, -0.3939, +0.9184],
])
_ANCHOR_TIME_S = 0.5 * _SWEEP_DURATION_S
# Sign chosen so R(t) = anchor @ Rot(δφ, spin_axis_RTN) reproduces independent
# SPICE-derived sweep midpoints over a 5-sweep cycle.
_SPIN_OMEGA_RAD_S = -2.0 * np.pi / _SPIN_PERIOD_S


# Set by main() before forking; children inherit via fork.
_worker_state: types.SimpleNamespace | None = None


def main():
    csv_path = _REPO_ROOT / "docs/swapi/figure_src/wind_solar_wind_samples_2025.csv"
    ground_truth_params = _load_wind_samples(csv_path)
    _initialize_worker_state(ground_truth_params)
    data = _run_fits(n_samples=len(ground_truth_params[0]))
    _plot_results(data)


def _load_wind_samples(csv_path: Path) -> tuple[np.ndarray, ...]:
    """Load WIND-derived ground-truth proton parameters from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing WIND samples CSV: {csv_path}. "
            f"Run scripts/swapi/sample_wind_solar_wind.py first."
        )
    cols = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return (
        cols["v_R_km_s"].astype(float),
        cols["proton_temperature_K"].astype(float),
        cols["proton_density_cm3"].astype(float),
        cols["v_T_km_s"].astype(float),
        cols["v_N_km_s"].astype(float),
    )


def _initialize_worker_state(ground_truth_params: tuple[np.ndarray, ...]) -> None:
    global _worker_state

    print(f"Using {_N_BINS} coarse-sweep bins, "
          f"{_VOLTAGES.min():.1f}–{_VOLTAGES.max():.1f} V")

    swapi_response = load_swapi_response()
    all_esa_voltages = np.tile(_VOLTAGES, _N_SWEEPS)
    swapi_response.warm_cache(all_esa_voltages)
    per_bin_rotation_matrices = _compute_per_bin_rotation_matrices(
        _N_SWEEPS, _BIN_INDICES_IN_SWEEP,
    )
    # Base context: bundles per-bin response grids and rotation matrices. Reused
    # for both forward modeling (synthetic count rates) and per-fit context
    # construction. count_rate is a placeholder of ones to bypass the >0 filter.
    base_ctx = build_solar_wind_fit_context(
        count_rate=np.ones_like(all_esa_voltages),
        esa_voltage=all_esa_voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=per_bin_rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    _worker_state = types.SimpleNamespace(
        ground_truth_params=ground_truth_params,
        swapi_response=swapi_response,
        all_esa_voltages=all_esa_voltages,
        per_bin_rotation_matrices=per_bin_rotation_matrices,
        base_ctx=base_ctx,
    )


def _compute_per_bin_rotation_matrices(
    n_sweeps: int, bin_indices_in_sweep: np.ndarray,
) -> np.ndarray:
    """Synthetic per-bin RTN→SWAPI matrices: anchor spun about its spin axis.

    Bin sample times are t = sw·_SWEEP_DURATION_S + bin_idx·_DT_S. Returns
    shape (n_sweeps · n_bins, 3, 3) in (sweep-major, bin-minor) order.
    """
    sweep_index = np.repeat(np.arange(n_sweeps), len(bin_indices_in_sweep))
    bin_index = np.tile(bin_indices_in_sweep, n_sweeps)
    sample_times_s = sweep_index * _SWEEP_DURATION_S + bin_index * _DT_S

    spin_axis = _ANCHOR_ROTATION_MATRIX[1] / np.linalg.norm(_ANCHOR_ROTATION_MATRIX[1])
    delta_phi = _SPIN_OMEGA_RAD_S * (sample_times_s - _ANCHOR_TIME_S)

    ax, ay, az = spin_axis
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return _ANCHOR_ROTATION_MATRIX @ rot


def _run_fits(n_samples: int) -> pd.DataFrame:
    n_workers = os.cpu_count() or 1
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)

    print(f"Running {n_samples} fits across {n_workers} processes...")
    t0 = time.perf_counter()
    rows = process_map(
        _process_one, range(n_samples),
        max_workers=n_workers, chunksize=10, desc="fits",
    )
    print(f"  Fits done in {time.perf_counter() - t0:.1f}s.")

    data = pd.DataFrame(rows)
    print(f"Bad-fit flags: {data['bad_flag'].sum()}/{n_samples}")
    out_csv = Path("/tmp/fit_accuracy_results.csv")
    data.to_csv(out_csv, index=False)
    print(f"Saved fit results to {out_csv}")
    return data


def _process_one(i):
    ws = _worker_state
    radial_speeds, temperatures, densities, tangential_speeds, normal_speeds = ws.ground_truth_params
    radial_speed = float(radial_speeds[i])
    temperature = float(temperatures[i])
    density = float(densities[i])
    tangential_speed = float(tangential_speeds[i])
    normal_speed = float(normal_speeds[i])

    truth_params = SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.array([radial_speed, tangential_speed, normal_speed]),
        temperature=temperature,
        mass_kg=PROTON_MASS_KG,
    )
    count_rates = model_solar_wind_coincidence_rates(truth_params, ws.base_ctx)
    count_rates = apply_deadtime_correction_array(count_rates)
    count_rates = (
        np.random.default_rng(i).poisson(np.maximum(count_rates * 0.145, 0.0)).astype(float)
        / 0.145
    )

    fit_ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=ws.all_esa_voltages,
        swapi_response=ws.swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=ws.per_bin_rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    try:
        initial_guess = calculate_initial_guess(fit_ctx)
    except Exception:
        initial_guess = SolarWindParams(
            density=float("nan"),
            bulk_velocity_rtn=np.array([float("nan")] * 3),
            temperature=float("nan"),
            mass_kg=PROTON_MASS_KG,
        )
    try:
        result = fit_solar_wind_proton_moments(fit_ctx)
    except Exception as e:
        print(f"  case {i}: fit failed ({type(e).__name__}: {e}); flagging bad")
        from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
            ProtonSolarWindFitResult,
        )
        from uncertainties import ufloat
        nan_uf = ufloat(float("nan"), float("nan"))
        result = ProtonSolarWindFitResult(
            density=nan_uf, temperature=nan_uf,
            bulk_velocity_rtn=(nan_uf, nan_uf, nan_uf),
            bad_fit_flag=1,
        )

    return {
        "true_density": density,
        "true_temperature": temperature,
        "true_radial_speed": radial_speed,
        "true_tangential_speed": tangential_speed,
        "true_normal_speed": normal_speed,
        "init_density": initial_guess.density,
        "init_temperature": initial_guess.temperature,
        "init_radial_speed": float(initial_guess.bulk_velocity_rtn[0]),
        "init_tangential_speed": float(initial_guess.bulk_velocity_rtn[1]),
        "init_normal_speed": float(initial_guess.bulk_velocity_rtn[2]),
        "fit_density": _nominal(result.density),
        "fit_temperature": _nominal(result.temperature),
        "fit_radial_speed": _nominal(result.bulk_velocity_rtn[0]),
        "fit_tangential_speed": _nominal(result.bulk_velocity_rtn[1]),
        "fit_normal_speed": _nominal(result.bulk_velocity_rtn[2]),
        "fit_density_sigma": _sigma(result.density),
        "fit_temperature_sigma": _sigma(result.temperature),
        "fit_radial_speed_sigma": _sigma(result.bulk_velocity_rtn[0]),
        "fit_tangential_speed_sigma": _sigma(result.bulk_velocity_rtn[1]),
        "fit_normal_speed_sigma": _sigma(result.bulk_velocity_rtn[2]),
        "bad_flag": bool(result.bad_fit_flag),
    }


def _nominal(x):
    return x.nominal_value if isinstance(x, UFloat) else x


def _sigma(x):
    return x.std_dev if isinstance(x, UFloat) else float("nan")


def _plot_results(data: pd.DataFrame) -> None:
    n_samples = len(data)
    good = ~data["bad_flag"]
    n_bad = data["bad_flag"].sum()

    plot_columns = [
        ("Density (cm⁻³)", "true_density", "init_density", "fit_density", "fit_density_sigma", "log"),
        ("Temperature (K)", "true_temperature", "init_temperature", "fit_temperature", "fit_temperature_sigma", "log"),
        ("$v_R$ (km/s)", "true_radial_speed", "init_radial_speed", "fit_radial_speed", "fit_radial_speed_sigma", "linear"),
        ("$v_T$ (km/s)", "true_tangential_speed", "init_tangential_speed", "fit_tangential_speed", "fit_tangential_speed_sigma", "linear"),
        ("$v_N$ (km/s)", "true_normal_speed", "init_normal_speed", "fit_normal_speed", "fit_normal_speed_sigma", "linear"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    fig.suptitle(
        f"Initial guess vs. final optimizer vs. WIND ground truth\n"
        f"({n_samples} real solar wind cases from WIND/SWE 2-min 2025, "
        f"{_N_SWEEPS} sweeps × {_N_BINS} coarse-sweep bins, Poisson noise)",
        fontsize=11,
    )

    color_initial_guess = "tab:orange"
    color_final_fit = "tab:blue"

    for ax, (label, true_key, init_key, fit_key, fit_sigma_key, scale) in zip(axes, plot_columns):
        truth = data[true_key]
        init = data[init_key]
        fit = data[fit_key]
        fit_sigma = data[fit_sigma_key]

        lo = np.nanmin(np.concatenate([truth, init, fit]))
        hi = np.nanmax(np.concatenate([truth, init, fit]))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, "k--", lw=0.8, alpha=0.4, zorder=0)

        ax.scatter(
            truth[good], init[good],
            s=6, alpha=0.35, color=color_initial_guess, marker="o", zorder=2, label="Initial guess",
        )
        ax.errorbar(
            truth[good], fit[good], yerr=fit_sigma[good],
            fmt="^", markersize=2.5, color=color_final_fit,
            ecolor=color_final_fit, elinewidth=0.4, capsize=0,
            alpha=0.45, zorder=3, label="Final fit",
        )

        if n_bad:
            ax.scatter(
                truth[~good], init[~good],
                s=30, alpha=0.9, color=color_initial_guess, marker="X", edgecolors="k", linewidths=0.4, zorder=4,
            )
            ax.scatter(
                truth[~good], fit[~good],
                s=30, alpha=0.9, color=color_final_fit, marker="X", edgecolors="k", linewidths=0.4, zorder=4,
            )

        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel("Estimated", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.annotate(
            f"Init  {_rmse_label(truth[good], init[good], scale)}\n"
            f"Fit    {_rmse_label(truth[good], fit[good], scale)}",
            xy=(0.04, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.8)
    fig.tight_layout()

    out_dir = _REPO_ROOT / "docs" / "swapi" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def _rmse_label(truth, estimated, scale):
    mask = np.isfinite(estimated)
    if scale == "log":
        rmse = np.sqrt(np.mean((np.log10(estimated[mask]) - np.log10(truth[mask])) ** 2))
        return f"RMSE log₁₀ = {rmse:.3f}"
    else:
        rmse = np.sqrt(np.mean((estimated[mask] - truth[mask]) ** 2))
        return f"RMSE = {rmse:.1f} km/s"


if __name__ == "__main__":
    main()
