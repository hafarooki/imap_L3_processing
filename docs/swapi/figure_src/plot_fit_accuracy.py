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
import spacepy.pycdf
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
_DEFAULT_WIND_CSV = (
    _REPO_ROOT / "docs" / "swapi" / "figure_src" / "wind_solar_wind_samples_2025.csv"
)

_RNG_SEED = 7
_N_SWEEPS = 5
_SWEEP_S = 12.0
_N_BINS = 71  # SWAPI_SCIENCE_BINS = slice(1, 72)
_DT_S = _SWEEP_S / 72

# SPICE-derived rotation matrices (RTN → SWAPI) for one 5-sweep chunk near
# 2026-01-01, taken from the alpha-fit fixture
# tests/test_data/swapi/alpha_fit_test_spectra.npz (`strong_alpha__rotation_matrices`).
# That file stores 5 sweeps × 62 coarse-sweep bins (slice(1, 63)). The proton
# plot uses 5 sweeps × 71 science bins (slice(1, 72)) — the first 62 bins per
# sweep coincide with the fixture; the last 9 fine-sweep bins are extended by
# spinning the fixture's last bin forward at the fixture's measured spin rate
# around its measured spin axis. The result reflects the real ~4° offset of
# the spin axis from -R_RTN and the real ~15.13 s spin period.
_FIXTURE_PATH = (
    _REPO_ROOT / "tests" / "test_data" / "swapi" / "alpha_fit_test_spectra.npz"
)
_FIXTURE_KEY = "strong_alpha__rotation_matrices"
_FIXTURE_BINS_PER_SWEEP = 62  # slice(1, 63)


def _nom(x):
    return x.nominal_value if isinstance(x, UFloat) else float(x)


def _load_science_voltages() -> np.ndarray:
    with spacepy.pycdf.CDF(str(_TEST_L2_CDF)) as cdf:
        esa_energy = cdf["esa_energy"][...]
    return esa_energy.mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR


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


def _real_rotation_matrices(n_sweeps: int) -> np.ndarray:
    """Real SPICE-derived rotation matrices for `n_sweeps` × 71 science bins.

    Bins 1..62 per sweep are loaded from the alpha-fit fixture as-is (real
    SPICE matrices for IMAP_RTN → IMAP_SWAPI). Bins 63..71 are extended from
    bin 62 of the same sweep by rotating around the fixture's mean spin axis
    at the fixture's measured spin rate. Both spin axis and spin rate are
    derived from the fixture itself, so the result inherits the real ~4°
    spin-axis tilt and the real ~15.13 s spin period.
    """
    fixture = np.load(_FIXTURE_PATH)
    R_fix = fixture[_FIXTURE_KEY]  # (n_fix_sweeps × 62, 3, 3)
    if R_fix.shape[0] % _FIXTURE_BINS_PER_SWEEP != 0:
        raise ValueError(
            f"{_FIXTURE_KEY} length {R_fix.shape[0]} not a multiple of "
            f"{_FIXTURE_BINS_PER_SWEEP}"
        )
    n_fix_sweeps = R_fix.shape[0] // _FIXTURE_BINS_PER_SWEEP
    if n_sweeps > n_fix_sweeps:
        raise ValueError(
            f"requested {n_sweeps} sweeps; fixture only has {n_fix_sweeps}"
        )
    R_fix = R_fix.reshape(n_fix_sweeps, _FIXTURE_BINS_PER_SWEEP, 3, 3)[:n_sweeps]

    # Mean spin axis: row 1 of R(RTN→SWAPI) is +Y_SWAPI in RTN, which is the
    # SWAPI boresight (and the spacecraft spin axis).
    spin_axis = R_fix[:, :, 1, :].reshape(-1, 3).mean(axis=0)
    spin_axis /= np.linalg.norm(spin_axis)

    # Measured spin rate: project +X_SWAPI(t) onto the plane perpendicular to
    # the spin axis and fit a line to the unwrapped phase vs measurement time.
    bin_offset = 1  # SWAPI_COARSE_SWEEP_BINS.start
    times_fix = (np.arange(n_sweeps * _FIXTURE_BINS_PER_SWEEP) // _FIXTURE_BINS_PER_SWEEP) * _SWEEP_S \
        + ((np.arange(n_sweeps * _FIXTURE_BINS_PER_SWEEP) % _FIXTURE_BINS_PER_SWEEP) + bin_offset) * _DT_S
    x_axis = R_fix.reshape(-1, 3, 3)[:, 0, :]
    x_perp = x_axis - (x_axis @ spin_axis)[:, None] * spin_axis
    e1 = x_perp[0] / np.linalg.norm(x_perp[0])
    e2 = np.cross(spin_axis, e1)
    phases = np.unwrap(np.arctan2(x_perp @ e2, x_perp @ e1))
    omega, _ = np.polyfit(times_fix, phases, 1)

    # Build the 71-bin output: copy fixture matrices for bins 1..62, then
    # spin bin 62 forward by Δt to fill bins 63..71.
    n_total = n_sweeps * _N_BINS
    R = np.empty((n_total, 3, 3))
    for sw in range(n_sweeps):
        out_lo = sw * _N_BINS
        # Copy first 62 bins per sweep verbatim.
        R[out_lo : out_lo + _FIXTURE_BINS_PER_SWEEP] = R_fix[sw]
        # Extend by 9 fine-sweep bins (indices 63..71). Reference bin = 62
        # (last fixture bin), at time t_ref = sw*12 + 62*dt.
        R_ref = R_fix[sw, -1]
        t_ref = sw * _SWEEP_S + (_FIXTURE_BINS_PER_SWEEP + bin_offset - 1) * _DT_S
        for j, bin_idx in enumerate(range(63, 72)):
            t = sw * _SWEEP_S + bin_idx * _DT_S
            dphi = omega * (t - t_ref)
            # Active rotation about the spin axis by dphi (Rodrigues' formula).
            ax = spin_axis
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R_extra = np.eye(3) + np.sin(dphi) * K + (1 - np.cos(dphi)) * (K @ K)
            # Apply rotation in the inertial RTN frame: bring SWAPI → RTN with
            # R_ref.T, rotate by R_extra (about spin axis in RTN), then map back
            # to SWAPI. Since spin advances SWAPI body relative to RTN, the new
            # SWAPI→RTN map is R_extra @ R_ref.T, so RTN→SWAPI is R_ref @ R_extra.T.
            R[out_lo + _FIXTURE_BINS_PER_SWEEP + j] = R_ref @ R_extra.T
    return R


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
    _W_ROT = _real_rotation_matrices(_N_SWEEPS)
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
    csv_path = _DEFAULT_WIND_CSV
    print(f"Loading WIND ground-truth samples from {csv_path}")
    params = _load_wind_samples(csv_path)
    n_samples = len(params[0])
    print(f"  {n_samples} samples")

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
    _rot0 = _real_rotation_matrices(_N_SWEEPS)
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

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "fit_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
