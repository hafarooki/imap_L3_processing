#!/usr/bin/env python3
"""
Scatter plots comparing the initial guess and final optimizer output against
ground truth for 10000 random solar wind parameter sets.

Uses realistic SWAPI 71-bin science voltage sweep (from the test L2 CDF) with 5
sweeps per fit — matching the production processor exactly. Synthetic count rates
are generated from the forward model with realistic SWAPI geometry (spin axis =
boresight = +Y_SWAPI, 15 s spin period) and Poisson noise.

Because all cases share the same voltage sweep, passband grids are built once and
reused. The optimizer is parallelised across cases via ThreadPoolExecutor (numba
functions release the GIL).

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

import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba
import spacepy.pycdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_SCIENCE_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _optimize,
    _model_count_rates,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_TEST_L2_CDF = (
    _REPO_ROOT / "tests/test_data/swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf"
)
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_N_SAMPLES = 10000
_RNG_SEED = 7
_N_SWEEPS = 5
_SWEEP_S = 12.0
_SPIN_S = 15.0
_N_BINS = 71  # SWAPI_SCIENCE_BINS = slice(1, 72)
_DT_S = _SWEEP_S / 72  # bin spacing within a 72-bin sweep

_R_BASE_RTN_TO_SWAPI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _load_science_voltages() -> np.ndarray:
    """Return the 71 science bin voltages from a realistic L2 CDF."""
    with spacepy.pycdf.CDF(str(_TEST_L2_CDF)) as cdf:
        esa_energy = cdf["esa_energy"][...]  # shape (n_sweeps, 72), in eV
    return esa_energy.mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR


def _spin_rotation_matrices(n: int) -> np.ndarray:
    """Rotation matrices for n consecutive bin measurements across N_SWEEPS sweeps.

    Each sweep is _SWEEP_S = 12 s with 72 bins of _DT_S = 12/72 s spacing; bin 0
    is discarded so we use bins 1-71. Times within sweep s are
        t = s * 12 + bin_idx * 12/72,   bin_idx in {1, ..., 71}
    so there is a 2*_DT_S gap from bin 71 of one sweep to bin 1 of the next.
    """
    sweep_idx = np.arange(n) // _N_BINS
    bin_in_sweep = (np.arange(n) % _N_BINS) + 1
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        R[i] = R_spin @ _R_BASE_RTN_TO_SWAPI
    return R


def _run_cases(sr: SWAPIResponse, voltages: np.ndarray) -> dict:
    rng = np.random.default_rng(_RNG_SEED)
    bulk_speeds = rng.uniform(300, 800, _N_SAMPLES)
    temperatures = np.exp(rng.uniform(np.log(2), np.log(50), _N_SAMPLES))
    densities = rng.uniform(2, 20, _N_SAMPLES)
    vTs = rng.uniform(-50, 50, _N_SAMPLES)
    vNs = rng.uniform(-50, 50, _N_SAMPLES)

    sc_vel = np.zeros(3)

    # All cases share the same voltage sweep → build grids once.
    print("  Building passband grids (once for all cases)...")
    t0 = time.perf_counter()
    base_grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    # Tile for N_SWEEPS: [sweep0_bin0..bin70, sweep1_bin0..bin70, ...]
    tiled_grids = numba.typed.List()
    for _ in range(_N_SWEEPS):
        for g in base_grids:
            tiled_grids.append(g)
    print(f"  Grids done in {time.perf_counter() - t0:.2f}s.")

    # Rotation matrices: same structure for every case (synthetic geometry).
    rot = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    esa_full = np.tile(voltages, _N_SWEEPS)

    def run_one(i):
        v_b = float(bulk_speeds[i])
        T = float(temperatures[i])
        n = float(densities[i])
        vT = float(vTs[i])
        vN = float(vNs[i])
        true_vel = np.array([v_b, vT, vN])

        cr = _model_count_rates(n, T, true_vel, tiled_grids, rot, sc_vel)
        cr = (
            np.random.default_rng(i).poisson(np.maximum(cr * 0.145, 0.0)).astype(float)
            / 0.145
        )

        ig = _get_initial_guess(cr, esa_full, tiled_grids, rot, sc_vel)
        result = _optimize(cr, tiled_grids, rot, sc_vel, ig)

        return (
            n,
            T,
            v_b,
            vT,
            vN,
            ig.density,
            ig.temperature,
            ig.bulk_velocity_rtn[0],
            ig.bulk_velocity_rtn[1],
            ig.bulk_velocity_rtn[2],
            result.density,
            result.temperature,
            result.bulk_velocity_rtn[0],
            result.bulk_velocity_rtn[1],
            result.bulk_velocity_rtn[2],
            bool(result.bad_fit_flag),
        )

    print(f"  Running {_N_SAMPLES} fits in parallel...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        rows = list(pool.map(run_one, range(_N_SAMPLES)))
    print(f"  Fits done in {time.perf_counter() - t0:.1f}s.")

    keys = (
        "true_n",
        "true_T",
        "true_vR",
        "true_vT",
        "true_vN",
        "init_n",
        "init_T",
        "init_vR",
        "init_vT",
        "init_vN",
        "fit_n",
        "fit_T",
        "fit_vR",
        "fit_vT",
        "fit_vN",
        "bad_flag",
    )
    out = {k: np.array([r[j] for r in rows]) for j, k in enumerate(keys)}
    out["bad_flag"] = out["bad_flag"].astype(bool)
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

    print("Loading realistic SWAPI science voltages...")
    voltages = _load_science_voltages()
    print(f"  {len(voltages)} bins, {voltages.min():.1f}–{voltages.max():.1f} V")

    print("Warming up JIT...")
    _grids0 = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    _tiled0 = numba.typed.List()
    for _ in range(_N_SWEEPS):
        for g in _grids0:
            _tiled0.append(g)
    _rot0 = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    _esa0 = np.tile(voltages, _N_SWEEPS)
    _sc0 = np.zeros(3)
    _cr0 = _model_count_rates(
        8.0, 10.0, np.array([500.0, 0.0, 0.0]), _tiled0, _rot0, _sc0
    )
    _ig0 = _get_initial_guess(_cr0, _esa0, _tiled0, _rot0, _sc0)
    _optimize(_cr0, _tiled0, _rot0, _sc0, _ig0)
    print("JIT ready.")

    t_total = time.perf_counter()
    data = _run_cases(sr, voltages)
    print(f"Total wall time: {time.perf_counter() - t_total:.1f}s")

    n_bad = data["bad_flag"].sum()
    print(f"Bad-fit flags: {n_bad}/{_N_SAMPLES}")

    good = ~data["bad_flag"]

    cols = [
        ("Density (cm⁻³)", "true_n", "init_n", "fit_n", "log"),
        ("Temperature (eV)", "true_T", "init_T", "fit_T", "log"),
        ("$v_R$ (km/s)", "true_vR", "init_vR", "fit_vR", "linear"),
        ("$v_T$ (km/s)", "true_vT", "init_vT", "fit_vT", "linear"),
        ("$v_N$ (km/s)", "true_vN", "init_vN", "fit_vN", "linear"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    fig.suptitle(
        f"Initial guess vs. final optimizer vs. ground truth\n"
        f"({_N_SAMPLES} random cases, {_N_SWEEPS} sweeps × {_N_BINS} bins, "
        f"realistic SWAPI voltage sweep, Poisson noise)",
        fontsize=11,
    )

    C_INIT = "tab:orange"
    C_FIT = "tab:blue"

    for ax, (label, tk, ik, fk, scale) in zip(axes, cols):
        truth = data[tk]
        init = data[ik]
        fit = data[fk]

        lo = np.nanmin(np.concatenate([truth, init, fit]))
        hi = np.nanmax(np.concatenate([truth, init, fit]))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, "k--", lw=0.8, alpha=0.4, zorder=0)

        ax.scatter(
            truth[good],
            init[good],
            s=6,
            alpha=0.35,
            color=C_INIT,
            marker="o",
            zorder=2,
            label="Initial guess",
        )
        ax.scatter(
            truth[good],
            fit[good],
            s=6,
            alpha=0.45,
            color=C_FIT,
            marker="^",
            zorder=3,
            label="Final fit",
        )

        if n_bad:
            ax.scatter(
                truth[~good],
                init[~good],
                s=30,
                alpha=0.9,
                color=C_INIT,
                marker="X",
                edgecolors="k",
                linewidths=0.4,
                zorder=4,
            )
            ax.scatter(
                truth[~good],
                fit[~good],
                s=30,
                alpha=0.9,
                color=C_FIT,
                marker="X",
                edgecolors="k",
                linewidths=0.4,
                zorder=4,
            )

        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel("Estimated", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        lbl_i = _rmse_label(truth[good], init[good], scale)
        lbl_f = _rmse_label(truth[good], fit[good], scale)
        ax.annotate(
            f"Init  {lbl_i}\nFit    {lbl_f}",
            xy=(0.04, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=6.5,
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
