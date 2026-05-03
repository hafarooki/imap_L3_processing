#!/usr/bin/env python3
"""
Scatter plots comparing the initial guess and final optimizer output against
ground truth for 10000 random solar wind parameter sets.

Uses realistic SWAPI 71-bin science voltage sweep (from the test L2 CDF) with 5
sweeps per fit — matching the production processor exactly. Synthetic count rates
are generated from the forward model with realistic SWAPI geometry (spin axis =
boresight = +Y_SWAPI, 15 s spin period) and Poisson noise.

All cases share the same voltage sweep so passband grids are built once per worker.
Cases are split into chunks across a ProcessPoolExecutor — threads don't help here
because scipy.optimize.least_squares does most of its bookkeeping in pure Python
and holds the GIL.

Solar wind parameter ranges (seed=7):
  bulk_speed:   200–2000 km/s         (uniform)
  temperature:  23,000–580,000 K      (log-uniform)
  density:        2–20 cm⁻³          (uniform)
  vT, vN:       −50–50 km/s          (uniform)

Output: docs/swapi/figures/initial_guess_accuracy.png
Usage:  python docs/swapi/figure_src/plot_initial_guess_accuracy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: F401
import numpy as np
import numba
import spacepy.pycdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
    METERS_PER_KILOMETER,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    EV_TO_KELVIN,
)
from figure_utils import load_swapi_response
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_SCIENCE_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _optimize,
    _model_count_rates,
    apply_deadtime_correction_array,
    fit_solar_wind_proton_moments,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


# Worker-process state. Initialized once per worker via _init_worker, then reused
# across all chunks routed to that worker. Avoids per-task pickle of SWAPIResponse
# and per-task rebuild of the passband-grid typed.List.
_W_SR = None
_W_TILED = None
_W_CS = None
_W_CEA = None
_W_AT = None
_W_ATS = None
_W_ROT = None
_W_ESA = None
_W_PARAMS = None  # tuple of (bulk_speeds, temperatures, densities, vTs, vNs)


def _init_worker(voltages, params):
    global _W_SR, _W_TILED, _W_CS, _W_CEA, _W_AT, _W_ATS, _W_ROT, _W_ESA, _W_PARAMS
    _W_SR = load_swapi_response()
    all_voltages = np.tile(voltages, _N_SWEEPS)
    _W_SR.warm_cache(all_voltages)
    _W_TILED = numba.typed.List([_W_SR.create_passband_grid(v) for v in all_voltages])
    _W_CS = np.array(
        [_W_SR.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in all_voltages]
    )
    _W_CEA = np.array([_W_SR.get_central_effective_area(v) for v in all_voltages])
    _W_AT = np.asarray(_W_SR.azimuthal_transmission, dtype=float)
    _W_ATS = float(_W_SR.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    _W_ROT = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    _W_ESA = all_voltages
    _W_PARAMS = params

    # Force JIT compile in this worker so the chunk loop runs at full speed from i=0.
    _model_count_rates(
        8.0,
        10.0 * EV_TO_KELVIN,
        np.array([500.0, 0.0, 0.0]),
        _W_TILED,
        _W_CS,
        _W_CEA,
        _W_AT,
        _W_ATS,
        _W_ROT,
        PROTON_MASS_KG,
    )


def _process_chunk(idx_range):
    bulk_speeds, temperatures, densities, vTs, vNs = _W_PARAMS
    rows = []
    for i in idx_range:
        v_b = float(bulk_speeds[i])
        T = float(temperatures[i])
        n = float(densities[i])
        vT = float(vTs[i])
        vN = float(vNs[i])

        cr = _model_count_rates(
            n,
            T,
            np.array([v_b, vT, vN]),
            _W_TILED,
            _W_CS,
            _W_CEA,
            _W_AT,
            _W_ATS,
            _W_ROT,
            PROTON_MASS_KG,
        )
        cr = apply_deadtime_correction_array(cr)
        cr = (
            np.random.default_rng(i).poisson(np.maximum(cr * 0.145, 0.0)).astype(float)
            / 0.145
        )

        # Initial guess is reported on the unmasked input.
        ig = _get_initial_guess(
            cr, _W_ESA, _W_TILED, _W_CS, _W_CEA, _W_AT, _W_ATS, _W_ROT
        )
        # Final fit goes through the production entry point so the half-mean mask
        # is applied at the keep boundary before the JIT integrator runs.
        result = fit_solar_wind_proton_moments(
            cr,
            _W_ESA,
            swapi_response=_W_SR,
            central_effective_area_scale=1,
            rotation_matrices=_W_ROT,
        )

        rows.append(
            (
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
        )
    return rows


def _run_cases(voltages: np.ndarray) -> dict:
    rng = np.random.default_rng(_RNG_SEED)
    bulk_speeds = rng.uniform(200, 2000, _N_SAMPLES)
    temperatures = np.exp(
        rng.uniform(np.log(2 * EV_TO_KELVIN), np.log(50 * EV_TO_KELVIN), _N_SAMPLES)
    )
    densities = rng.uniform(2, 20, _N_SAMPLES)
    vTs = rng.uniform(-50, 50, _N_SAMPLES)
    vNs = rng.uniform(-50, 50, _N_SAMPLES)
    params = (bulk_speeds, temperatures, densities, vTs, vNs)

    # The fit is GIL-bound (scipy.optimize.least_squares does most of its bookkeeping
    # in pure Python), so threads give no speedup. Use processes; each worker initialises
    # its own SWAPIResponse + passband-grid typed.List once via initializer, then runs
    # a contiguous chunk of cases.
    n_workers = max(1, (os.cpu_count() or 1))
    chunks = [
        range(start, min(start + (_N_SAMPLES + n_workers - 1) // n_workers, _N_SAMPLES))
        for start in range(0, _N_SAMPLES, (_N_SAMPLES + n_workers - 1) // n_workers)
    ]
    print(
        f"  Running {_N_SAMPLES} fits across {n_workers} processes "
        f"({len(chunks)} chunks)..."
    )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(voltages, params),
    ) as pool:
        rows = []
        for chunk_rows in pool.map(_process_chunk, chunks):
            rows.extend(chunk_rows)
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
    sr = load_swapi_response()

    print("Loading realistic SWAPI science voltages...")
    voltages = _load_science_voltages()
    print(f"  {len(voltages)} bins, {voltages.min():.1f}–{voltages.max():.1f} V")

    print("Warming up JIT (driver process)...")
    _esa0 = np.tile(voltages, _N_SWEEPS)
    sr.warm_cache(_esa0)
    _tiled0 = numba.typed.List([sr.create_passband_grid(v) for v in _esa0])
    _cs0 = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in _esa0]
    )
    _cea0 = np.array([sr.get_central_effective_area(v) for v in _esa0])
    _at0 = np.asarray(sr.azimuthal_transmission, dtype=float)
    _ats0 = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    _rot0 = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    _cr0 = _model_count_rates(
        8.0,
        10.0 * EV_TO_KELVIN,
        np.array([500.0, 0.0, 0.0]),
        _tiled0,
        _cs0,
        _cea0,
        _at0,
        _ats0,
        _rot0,
        PROTON_MASS_KG,
    )
    _ig0 = _get_initial_guess(_cr0, _esa0, _tiled0, _cs0, _cea0, _at0, _ats0, _rot0)
    _optimize(_cr0, _tiled0, _cs0, _cea0, _at0, _ats0, _rot0, _ig0)
    fit_solar_wind_proton_moments(
        _cr0,
        _esa0,
        swapi_response=sr,
        central_effective_area_scale=1,
        rotation_matrices=_rot0,
    )
    print("JIT ready.")

    t_total = time.perf_counter()
    data = _run_cases(voltages)
    print(f"Total wall time: {time.perf_counter() - t_total:.1f}s")

    n_bad = data["bad_flag"].sum()
    print(f"Bad-fit flags: {n_bad}/{_N_SAMPLES}")

    good = ~data["bad_flag"]

    cols = [
        ("Density (cm⁻³)", "true_n", "init_n", "fit_n", "log"),
        ("Temperature (K)", "true_T", "init_T", "fit_T", "log"),
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
