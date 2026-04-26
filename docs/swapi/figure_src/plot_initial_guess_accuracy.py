#!/usr/bin/env python3
"""
Scatter plots comparing the initial guess and final optimizer output against
ground truth for 100 random solar wind parameter sets.

Synthetic count rates are generated from the forward model with realistic SWAPI
geometry (5 sweeps × 72 voltage steps over 60 s, 15 s spin period, spin axis =
boresight = +Y_SWAPI) and Poisson noise, then passed through _get_initial_guess
(Gaussian fit + density scaling, no sine-fit since no SPICE) and _optimize
(Levenberg-Marquardt).

Solar wind parameter ranges (seed=7):
  bulk_speed:   300–800 km/s   (uniform)
  temperature:    2–50 eV      (log-uniform)
  density:        2–20 cm⁻³   (uniform)
  vT, vN:       −50–50 km/s   (uniform)

Output: docs/swapi/figures/initial_guess_accuracy.png
Usage:  python docs/swapi/figure_src/plot_initial_guess_accuracy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import math
import numpy as np
import numba
import scipy.optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, METERS_PER_KILOMETER
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR, esa_voltage_to_proton_speed
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _optimize,
    _model_count_rates,
    ProtonSolarWindMoments,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_N_SAMPLES = 100
_RNG_SEED = 7
_N_SWEEPS = 5
_N_BINS = 72                # 72 ESA voltage steps per sweep
_SWEEP_S = 12.0             # 12 s per sweep
_SPIN_S  = 15.0             # spacecraft spin period
_DT_S    = _SWEEP_S / _N_BINS

_R_BASE_RTN_TO_SWAPI = np.array([[ 0.0, 1.0, 0.0],
                                  [-1.0, 0.0, 0.0],
                                  [ 0.0, 0.0, 1.0]])


def _peak_voltage(v_km_s: float) -> float:
    return (PROTON_MASS_KG * (v_km_s * METERS_PER_KILOMETER) ** 2
            / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS))


def _spin_rotation_matrices(n: int) -> np.ndarray:
    """Realistic SWAPI geometry: spin axis = boresight (+Y_SWAPI = -R_RTN).

    R(t) = R_spin_around_Y(2*pi*t/T_spin) @ R_base, where R_base maps the
    nominal anti-sunward bulk to -Y_SWAPI so phi=theta=0 at zero spin phase.
    Spin around the boresight leaves the dominant Y component unchanged and
    only introduces small phi/theta wobble of order arcsin(v_T,N / v_R).
    """
    times = np.arange(n) * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[ c, 0.0,  s],
                           [0.0, 1.0, 0.0],
                           [-s, 0.0,  c]])
        R[i] = R_spin @ _R_BASE_RTN_TO_SWAPI
    return R


def _run_cases(sr: SWAPIResponse) -> dict:
    rng = np.random.default_rng(_RNG_SEED)
    bulk_speeds  = rng.uniform(300, 800, _N_SAMPLES)
    temperatures = np.exp(rng.uniform(np.log(2), np.log(50), _N_SAMPLES))
    densities    = rng.uniform(2, 20, _N_SAMPLES)
    vTs          = rng.uniform(-50, 50, _N_SAMPLES)
    vNs          = rng.uniform(-50, 50, _N_SAMPLES)

    n_meas = _N_SWEEPS * _N_BINS
    sc_vel = np.zeros(3)

    out = {k: np.empty(_N_SAMPLES) for k in
           ("true_n", "true_T", "true_vR", "true_vT", "true_vN",
            "init_n", "init_T", "init_vR", "init_vT", "init_vN",
            "fit_n",  "fit_T",  "fit_vR",  "fit_vT",  "fit_vN")}
    out["bad_flag"] = np.zeros(_N_SAMPLES, dtype=bool)

    for i in range(_N_SAMPLES):
        v_b = float(bulk_speeds[i])
        T   = float(temperatures[i])
        n   = float(densities[i])
        vT_i = float(vTs[i])
        vN_i = float(vNs[i])
        true_vel = np.array([v_b, vT_i, vN_i])

        voltages = np.geomspace(_peak_voltage(v_b) * 0.3, _peak_voltage(v_b) * 3.0, _N_BINS)
        grids = numba.typed.List([sr.create_passband_grid(v)
                                  for _ in range(_N_SWEEPS) for v in voltages])
        rot = _spin_rotation_matrices(n_meas)
        esa_full = np.tile(voltages, _N_SWEEPS)

        count_rate = _model_count_rates(n, T, true_vel, grids, rot, sc_vel)
        count_rate = rng.poisson(np.maximum(count_rate, 0.0)).astype(float)

        ig     = _get_initial_guess(count_rate, esa_full, grids, rot, sc_vel)
        result = _optimize(count_rate, grids, rot, sc_vel, ig)

        out["true_n"][i] = n;   out["true_T"][i] = T
        out["true_vR"][i] = v_b; out["true_vT"][i] = vT_i; out["true_vN"][i] = vN_i

        out["init_n"][i] = ig.density;       out["init_T"][i] = ig.temperature
        out["init_vR"][i] = ig.bulk_velocity_rtn[0]
        out["init_vT"][i] = ig.bulk_velocity_rtn[1]
        out["init_vN"][i] = ig.bulk_velocity_rtn[2]

        out["fit_n"][i] = result.density;    out["fit_T"][i] = result.temperature
        out["fit_vR"][i] = result.bulk_velocity_rtn[0]
        out["fit_vT"][i] = result.bulk_velocity_rtn[1]
        out["fit_vN"][i] = result.bulk_velocity_rtn[2]
        out["bad_flag"][i] = bool(result.bad_fit_flag)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{_N_SAMPLES}  (bad flags so far: {out['bad_flag'][:i+1].sum()})")

    return out


def _rmse_label(truth, est, scale):
    mask = np.isfinite(est)
    if scale == "log":
        rmse = np.sqrt(np.mean((np.log10(est[mask]) - np.log10(truth[mask])) ** 2))
        return f"RMSE log₁₀ = {rmse:.3f}"
    else:
        rmse = np.sqrt(np.mean((est[mask] - truth[mask]) ** 2))
        return f"RMSE = {rmse:.1f} km/s"


def main():
    print("Loading calibration data...")
    sr = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    # JIT warm-up
    print("Warming up JIT...")
    _v0 = 450.0
    _vols0 = np.geomspace(_peak_voltage(_v0) * 0.3, _peak_voltage(_v0) * 3.0, _N_BINS)
    _grids0 = numba.typed.List([sr.create_passband_grid(v) for _ in range(_N_SWEEPS) for v in _vols0])
    _rot0 = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    _cr0 = _model_count_rates(5.0, 10.0, np.array([_v0, 0.0, 0.0]), _grids0, _rot0, np.zeros(3))
    _ig0 = _get_initial_guess(_cr0, np.tile(_vols0, _N_SWEEPS), _grids0, _rot0, np.zeros(3))
    _optimize(_cr0, _grids0, _rot0, np.zeros(3), _ig0)
    print("JIT ready. Running cases...")

    data = _run_cases(sr)
    n_bad = data["bad_flag"].sum()
    print(f"Done. Bad-fit flags: {n_bad}/{_N_SAMPLES}")

    # --- Figure ---
    good = ~data["bad_flag"]

    # Column spec: (axis label, true key, init key, fit key, scale)
    cols = [
        ("Density (cm⁻³)",    "true_n",  "init_n",  "fit_n",  "log"),
        ("Temperature (eV)",   "true_T",  "init_T",  "fit_T",  "log"),
        ("$v_R$ (km/s)",       "true_vR", "init_vR", "fit_vR", "linear"),
        ("$v_T$ (km/s)",       "true_vT", "init_vT", "fit_vT", "linear"),
        ("$v_N$ (km/s)",       "true_vN", "init_vN", "fit_vN", "linear"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    fig.suptitle(
        f"Initial guess vs. final optimizer vs. ground truth\n"
        f"({_N_SAMPLES} random cases, {_N_SWEEPS} sweeps × {_N_BINS} steps, "
        f"spin axis = boresight, Poisson noise)",
        fontsize=11,
    )

    C_INIT = "tab:orange"
    C_FIT  = "tab:blue"

    for ax, (label, tk, ik, fk, scale) in zip(axes, cols):
        truth = data[tk]
        init  = data[ik]
        fit   = data[fk]

        # 1:1 reference
        lo = np.nanmin(np.concatenate([truth, init, fit]))
        hi = np.nanmax(np.concatenate([truth, init, fit]))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, "k--", lw=0.8, alpha=0.4, zorder=0)

        # Good cases
        ax.scatter(truth[good], init[good],  s=14, alpha=0.55, color=C_INIT,
                   marker="o", zorder=2, label="Initial guess")
        ax.scatter(truth[good], fit[good],   s=14, alpha=0.65, color=C_FIT,
                   marker="^", zorder=3, label="Final fit")

        # Bad-flag cases (X markers, same colours)
        if n_bad:
            ax.scatter(truth[~good], init[~good], s=30, alpha=0.9, color=C_INIT,
                       marker="X", edgecolors="k", linewidths=0.4, zorder=4)
            ax.scatter(truth[~good], fit[~good],  s=30, alpha=0.9, color=C_FIT,
                       marker="X", edgecolors="k", linewidths=0.4, zorder=4)

        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel("Estimated", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        if scale == "log":
            ax.set_xscale("log"); ax.set_yscale("log")

        # RMSE annotations
        lbl_i = _rmse_label(truth[good], init[good],  scale)
        lbl_f = _rmse_label(truth[good], fit[good],   scale)
        ax.annotate(
            f"Init  {lbl_i}\nFit    {lbl_f}",
            xy=(0.04, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.8)
    fig.tight_layout()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "initial_guess_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
