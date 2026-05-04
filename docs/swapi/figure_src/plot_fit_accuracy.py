#!/usr/bin/env python3
"""
Scatter plots comparing the SWAPI proton-moments fit against ground truth on
**real solar wind conditions** sampled from WIND/SWE 2-min ASCII data.

Synthetic count rates are produced from the SWAPI forward model on a
realistic 71-bin science voltage sweep (5 sweeps per fit, Poisson noise);
the (n, T, v_R, v_T, v_N) ground truth is read from a CSV produced by
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

import time
import os
import multiprocessing as _mp
import numpy as np
import numba
from uncertainties import UFloat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    EV_TO_KELVIN,
)
from figure_utils import load_swapi_response
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _optimize,
    _model_count_rates,
    apply_deadtime_correction_array,
    fit_solar_wind_proton_moments,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_N_SWEEPS = 5
_N_BINS = 71

# Mean SWAPI L2 science-bin voltages (V), 71 bins. Indices 0..61 are the
# descending coarse sweep; 62..70 are the 9 fine-sweep bins near the proton peak.
_SCIENCE_VOLTAGES = np.array([
     9895.52,  9088.69,  8348.80,  7667.55,  7042.16,  6469.31,  5941.77,  5457.31,
     5013.22,  4603.65,  4230.77,  3886.92,  3569.16,  3278.72,  3011.13,  2766.25,
     2539.54,  2333.83,  2144.24,  1969.31,  1808.74,  1660.86,  1525.75,  1401.82,
     1287.58,  1182.24,  1085.15,   995.55,   914.31,   839.94,   771.70,   709.46,
      651.59,   598.47,   549.91,   505.12,   463.89,   425.92,   391.18,   359.35,
      329.94,   303.02,   278.25,   255.55,   234.77,   215.61,   197.95,   181.82,
      167.04,   153.46,   140.91,   129.50,   118.91,   109.20,   100.30,    92.11,
       84.61,    77.73,    71.40,    65.59,    60.23,    55.34,   769.27,   753.47,
      737.99,   722.84,   707.99,   693.45,   679.20,   665.25,   651.59,
])
assert _SCIENCE_VOLTAGES.shape == (_N_BINS,)

# RTN -> SWAPI rotation matrices, one per sweep at the sweep midpoint. SPICE-derived
# from kernels around 2026-01-01 — captures the real ~4° spin-axis tilt off -R_RTN
# and the real ~15.13 s spin period (sweep-to-sweep phase shift). Per-bin spin
# variation within a sweep is dropped; the 5 sweeps still span one full spin cycle,
# so v_T and v_N remain observable.
_ROTATION_MATRICES = np.array([
    [[+0.0705, +0.9157, +0.3955],
     [-0.9968, +0.0792, -0.0057],
     [-0.0365, -0.3939, +0.9184]],
    [[-0.0141, -0.1350, +0.9907],
     [-0.9972, +0.0743, -0.0041],
     [-0.0731, -0.9881, -0.1357]],
    [[-0.0721, -0.9884, +0.1340],
     [-0.9974, +0.0716, -0.0084],
     [-0.0013, -0.1342, -0.9909]],
    [[-0.0183, -0.3937, -0.9191],
     [-0.9971, +0.0750, -0.0122],
     [+0.0737, +0.9162, -0.3939]],
    [[+0.0683, +0.7775, -0.6251],
     [-0.9968, +0.0795, -0.0100],
     [+0.0420, +0.6238, +0.7805]],
])
assert _ROTATION_MATRICES.shape == (_N_SWEEPS, 3, 3)


def _nom(x):
    return x.nominal_value if isinstance(x, UFloat) else float(x)


def _load_wind_samples(csv_path: Path) -> tuple[np.ndarray, ...]:
    """Load WIND-derived ground-truth proton parameters from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing WIND samples CSV: {csv_path}. "
            f"Run scripts/swapi/sample_wind_solar_wind.py first."
        )
    cols = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    bulk_speeds = cols["v_R_km_s"].astype(float)
    temperatures = cols["proton_temperature_K"].astype(float)
    densities = cols["proton_density_cm3"].astype(float)
    vTs = cols["v_T_km_s"].astype(float)
    vNs = cols["v_N_km_s"].astype(float)
    return bulk_speeds, temperatures, densities, vTs, vNs


def _per_bin_rotation_matrices() -> np.ndarray:
    """Replicate each sweep's rotation matrix across its 71 bins → (5*71, 3, 3)."""
    return np.repeat(_ROTATION_MATRICES, _N_BINS, axis=0)


_W_SR = None
_W_TILED = None
_W_CS = None
_W_CEA = None
_W_AT = None
_W_ATS = None
_W_ROT = None
_W_ESA = None
_W_PARAMS = None


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
    _W_ROT = _per_bin_rotation_matrices()
    _W_ESA = all_voltages
    _W_PARAMS = params

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

        ig = _get_initial_guess(
            cr, _W_ESA, _W_TILED, _W_CS, _W_CEA, _W_AT, _W_ATS, _W_ROT
        )
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
                _nom(ig.density),
                _nom(ig.temperature),
                _nom(ig.bulk_velocity_rtn[0]),
                _nom(ig.bulk_velocity_rtn[1]),
                _nom(ig.bulk_velocity_rtn[2]),
                _nom(result.density),
                _nom(result.temperature),
                _nom(result.bulk_velocity_rtn[0]),
                _nom(result.bulk_velocity_rtn[1]),
                _nom(result.bulk_velocity_rtn[2]),
                bool(result.bad_fit_flag),
            )
        )
    return rows


def _run_cases(voltages: np.ndarray, params: tuple) -> dict:
    n_samples = len(params[0])
    n_workers = max(1, (os.cpu_count() or 1))
    chunk_size = (n_samples + n_workers - 1) // n_workers
    chunks = [
        range(start, min(start + chunk_size, n_samples))
        for start in range(0, n_samples, chunk_size)
    ]
    print(
        f"  Running {n_samples} fits across {n_workers} processes "
        f"({len(chunks)} chunks)..."
    )
    t0 = time.perf_counter()
    # Use fork context to match SwapiProcessor (and to avoid the spawn-side
    # import regression that breaks worker startup in this environment).
    ctx = _mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
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
    csv_path = _REPO_ROOT / "docs/swapi/figure_src/wind_solar_wind_samples_2025.csv"
    print(f"Loading WIND ground-truth samples from {csv_path}")
    params = _load_wind_samples(csv_path)
    n_samples = len(params[0])
    print(f"  {n_samples} samples")

    print("Loading calibration data...")
    sr = load_swapi_response()

    voltages = _SCIENCE_VOLTAGES
    print(f"Using {len(voltages)} science bins, "
          f"{voltages.min():.1f}–{voltages.max():.1f} V")

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
    _rot0 = _per_bin_rotation_matrices()
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
    data = _run_cases(voltages, params)
    print(f"Total wall time: {time.perf_counter() - t_total:.1f}s")

    n_bad = data["bad_flag"].sum()
    print(f"Bad-fit flags: {n_bad}/{n_samples}")

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
        f"Initial guess vs. final optimizer vs. WIND ground truth\n"
        f"({n_samples} real solar wind cases from WIND/SWE 2-min 2025, "
        f"{_N_SWEEPS} sweeps × {_N_BINS} bins, Poisson noise)",
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

    out_dir = _REPO_ROOT / "docs" / "swapi" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
