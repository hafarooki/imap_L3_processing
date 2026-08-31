#!/usr/bin/env python3
"""
Scatter plots comparing the SWAPI proton-moments fit against ground truth on
**real solar wind conditions** sampled from WIND/SWE 2-min ASCII data.

Uses the 62 coarse-sweep bins (indices 1..62 of a 72-bin sweep) to show that
the fit recovers (n, T, v_R, v_T, v_N) without fine-sweep coverage.

Per-bin SWAPI→RTN rotation matrices are generated synthetically by spinning a
single anchor matrix (a real SPICE attitude near 2026-01-01) about its own
spin axis at the nominal SWAPI spin period.

Synthetic count rates are produced from the SWAPI forward model (5 sweeps per
fit, Poisson noise); the ground truth is read from a CSV produced by
scripts/swapi/sample_wind_solar_wind.py.

Generate the CSV first:
  conda run -n imapL3 python scripts/swapi/sample_wind_solar_wind.py \
      --year 2025 --n 10000 --seed 7

Output: docs/swapi/figures/fit_accuracy.svg
Usage:  conda run -n imapL3 python docs/swapi/figure_src/plot_fit_accuracy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import types

import numpy as np
import pandas as pd
from uncertainties import UFloat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.constants import SWAPI_LIVETIME_S
from figure_utils import (
    COARSE_BIN_INDICES_IN_SWEEP,
    COARSE_SWEEP_VOLTAGES_MEAN_V,
    REPO_ROOT,
    compute_per_bin_rotation_matrices,
    load_swapi_response,
    run_parallel_map,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.calculate_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams

_N_SWEEPS = 5
_N_BINS = len(COARSE_SWEEP_VOLTAGES_MEAN_V)

_worker_state: types.SimpleNamespace | None = None


def main():
    csv_path = REPO_ROOT / "docs/swapi/figure_src/wind_solar_wind_samples_2025.csv"
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
    cols = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    return (
        cols["v_R_km_s"].astype(float),
        cols["proton_temperature_K"].astype(float),
        cols["proton_density_cm3"].astype(float),
        cols["v_T_km_s"].astype(float),
        cols["v_N_km_s"].astype(float),
    )


def _initialize_worker_state(ground_truth_params: tuple[np.ndarray, ...]) -> None:
    global _worker_state

    print(
        f"Using {_N_BINS} coarse-sweep bins, "
        f"{COARSE_SWEEP_VOLTAGES_MEAN_V.min():.1f}–{COARSE_SWEEP_VOLTAGES_MEAN_V.max():.1f} V"
    )

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


def _run_fits(n_samples: int) -> pd.DataFrame:
    rows = run_parallel_map(_process_one, n_samples, desc="fits", chunksize=10)
    data = pd.DataFrame(rows)
    print(f"Bad-fit flags: {data['bad_flag'].sum()}/{n_samples}")
    out_csv = Path("/tmp/fit_accuracy_results.csv")
    data.to_csv(out_csv, index=False)
    print(f"Saved fit results to {out_csv}")
    return data


def _process_one(i):
    ws = _worker_state
    radial_speeds, temperatures, densities, tangential_speeds, normal_speeds = (
        ws.ground_truth_params
    )
    radial_speed = float(radial_speeds[i])
    temperature = float(temperatures[i])
    density = float(densities[i])
    tangential_speed = float(tangential_speeds[i])
    normal_speed = float(normal_speeds[i])

    truth_params = SolarWindParams(
        density=density,
        velocity_rtn=np.array([radial_speed, tangential_speed, normal_speed]),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )
    count_rates, _ = model_solar_wind_ideal_coincidence_rates(truth_params, ws.base_ctx)
    count_rates = count_rates * deadtime_factor(count_rates)
    count_rates = (
        np.random.default_rng(i)
        .poisson(np.maximum(count_rates * SWAPI_LIVETIME_S, 0.0))
        .astype(float)
        / SWAPI_LIVETIME_S
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
            velocity_rtn=np.array([float("nan")] * 3),
            temperature=float("nan"),
            mass=PROTON_MASS_KG,
        )
    try:
        result = fit_solar_wind_proton_model(fit_ctx)
    except Exception as e:
        print(f"  case {i}: fit failed ({type(e).__name__}: {e}); flagging bad")
        from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
            ProtonSolarWindFitResult,
        )
        from uncertainties import ufloat

        nan_uf = ufloat(float("nan"), float("nan"))
        result = ProtonSolarWindFitResult(
            density=nan_uf,
            temperature=nan_uf,
            velocity_rtn=(nan_uf, nan_uf, nan_uf),
            quality_flag=1,
        )

    return {
        "true_density": density,
        "true_temperature": temperature,
        "true_radial_speed": radial_speed,
        "true_tangential_speed": tangential_speed,
        "true_normal_speed": normal_speed,
        "init_density": initial_guess.density,
        "init_temperature": initial_guess.temperature,
        "init_radial_speed": float(initial_guess.velocity_rtn[0]),
        "init_tangential_speed": float(initial_guess.velocity_rtn[1]),
        "init_normal_speed": float(initial_guess.velocity_rtn[2]),
        "fit_density": _nominal(result.density),
        "fit_temperature": _nominal(result.temperature),
        "fit_radial_speed": _nominal(result.velocity_rtn[0]),
        "fit_tangential_speed": _nominal(result.velocity_rtn[1]),
        "fit_normal_speed": _nominal(result.velocity_rtn[2]),
        "fit_density_sigma": _sigma(result.density),
        "fit_temperature_sigma": _sigma(result.temperature),
        "fit_radial_speed_sigma": _sigma(result.velocity_rtn[0]),
        "fit_tangential_speed_sigma": _sigma(result.velocity_rtn[1]),
        "fit_normal_speed_sigma": _sigma(result.velocity_rtn[2]),
        "bad_flag": bool(result.quality_flag),
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
        (
            "Density (cm⁻³)",
            "true_density",
            "init_density",
            "fit_density",
            "fit_density_sigma",
            "log",
        ),
        (
            "Temperature (K)",
            "true_temperature",
            "init_temperature",
            "fit_temperature",
            "fit_temperature_sigma",
            "log",
        ),
        (
            "$v_R$ (km/s)",
            "true_radial_speed",
            "init_radial_speed",
            "fit_radial_speed",
            "fit_radial_speed_sigma",
            "linear",
        ),
        (
            "$v_T$ (km/s)",
            "true_tangential_speed",
            "init_tangential_speed",
            "fit_tangential_speed",
            "fit_tangential_speed_sigma",
            "linear",
        ),
        (
            "$v_N$ (km/s)",
            "true_normal_speed",
            "init_normal_speed",
            "fit_normal_speed",
            "fit_normal_speed_sigma",
            "linear",
        ),
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

    for ax, (label, true_key, init_key, fit_key, fit_sigma_key, scale) in zip(
        axes, plot_columns
    ):
        truth = data[true_key]
        init = data[init_key]
        fit = data[fit_key]
        fit_sigma = data[fit_sigma_key]

        lo = np.nanmin(np.concatenate([truth, init, fit]))
        hi = np.nanmax(np.concatenate([truth, init, fit]))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, "k--", lw=0.8, alpha=0.4, zorder=0)

        ax.scatter(
            truth[good],
            init[good],
            s=6,
            alpha=0.35,
            color=color_initial_guess,
            marker="o",
            zorder=2,
            label="Initial guess",
            rasterized=True
        )
        ax.errorbar(
            truth[good],
            fit[good],
            yerr=fit_sigma[good],
            fmt="^",
            markersize=2.5,
            color=color_final_fit,
            ecolor=color_final_fit,
            elinewidth=0.4,
            capsize=0,
            alpha=0.45,
            zorder=3,
            label="Final fit",
            rasterized=True,
        )

        if n_bad:
            ax.scatter(
                truth[~good],
                init[~good],
                s=30,
                alpha=0.9,
                color=color_initial_guess,
                marker="X",
                edgecolors="k",
                linewidths=0.4,
                zorder=4,
                rasterized=True
            )
            ax.scatter(
                truth[~good],
                fit[~good],
                s=30,
                alpha=0.9,
                color=color_final_fit,
                marker="X",
                edgecolors="k",
                linewidths=0.4,
                zorder=4,
                rasterized=True
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
            xy=(0.04, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.8)
    fig.tight_layout()

    out_dir = REPO_ROOT / "docs" / "swapi" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_accuracy.svg"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved {out_path}")


def _rmse_label(truth, estimated, scale):
    mask = np.isfinite(estimated)
    if scale == "log":
        rmse = np.sqrt(
            np.mean((np.log10(estimated[mask]) - np.log10(truth[mask])) ** 2)
        )
        return f"RMSE log₁₀ = {rmse:.3f}"
    else:
        rmse = np.sqrt(np.mean((estimated[mask] - truth[mask]) ** 2))
        return f"RMSE = {rmse:.1f} km/s"


if __name__ == "__main__":
    main()
