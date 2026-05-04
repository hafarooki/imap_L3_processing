#!/usr/bin/env python3
"""
χ² landscape in the body-frame perpendicular plane illustrating the two-basin
problem addressed by the K-rotation grid check in `_optimize`.

For a representative case where the production LM workflow (LM-from-IG) actually
converges to the spin-axis mirror basin, this figure shows the profile-MSE surface
in (vx_body, vz_body) with two clear minima — truth and its spin-axis mirror.

Without the K=4 grid, LM from the production initial guess (which starts at
vT=−30 km/s, vN=0) converges to the mirror basin because the IG is on the wrong
side of the saddle. The K=4 grid detects this by evaluating the 180°-rotated
candidate and firing a second LM from the correct basin.

In body-frame perpendicular coordinates the spin axis is body-y, so:
- Mirror is exactly (−vx_truth, −vz_truth) — reflection across body-y
- Production IG has vT=−30 km/s, which sits on the mirror-basin side of the saddle
  when truth has large positive vT

Output: docs/swapi/figures/wrong_basin.png
Usage:  python docs/swapi/figure_src/plot_wrong_basin.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import numba
import spacepy.pycdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    EV_TO_KELVIN,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_SCIENCE_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _model_count_rates,
    apply_deadtime_correction_array,
    _residuals_njit,
    _get_initial_guess,
    SWAPI_DEADTIME_S,
)
from figure_utils import load_swapi_response


@numba.njit(parallel=True, nogil=True)
def _compute_mse_grid(
    grid_vx,
    grid_vz,
    T_candidates,
    vy_candidates,
    R_avg_T,
    tiled,
    tiled_cs,
    tiled_cea,
    at,
    ats,
    rot,
    cr,
    mass_kg,
):
    """Parallel numba grid: for each (vx, vz) grid point, profile over
    (T, vy_body) candidates and return the minimum-MSE via closed-form
    density rescale."""
    nz = len(grid_vz)
    nx = len(grid_vx)
    mse_flat = np.empty(nz * nx)
    for flat_idx in numba.prange(nz * nx):
        i = flat_idx // nx
        j = flat_idx % nx
        vx_ = grid_vx[j]
        vz_ = grid_vz[i]
        best_mse = np.inf
        for ti in range(len(T_candidates)):
            T_ = T_candidates[ti]
            for vi in range(len(vy_candidates)):
                vy_ = vy_candidates[vi]
                v_rtn = np.empty(3)
                v_rtn[0] = R_avg_T[0, 0] * vx_ + R_avg_T[0, 1] * vy_ + R_avg_T[0, 2] * vz_
                v_rtn[1] = R_avg_T[1, 0] * vx_ + R_avg_T[1, 1] * vy_ + R_avg_T[1, 2] * vz_
                v_rtn[2] = R_avg_T[2, 0] * vx_ + R_avg_T[2, 1] * vy_ + R_avg_T[2, 2] * vz_
                # Unit-density model + deadtime correction
                model_true = _model_count_rates(
                    1.0, T_, v_rtn, tiled, tiled_cs, tiled_cea, at, ats, rot, mass_kg
                )
                model_obs = model_true / (1.0 + SWAPI_DEADTIME_S * model_true)
                # Closed-form optimal density scale
                mm = 0.0
                mr = 0.0
                for k in range(len(model_obs)):
                    mm += model_obs[k] * model_obs[k]
                    mr += model_obs[k] * cr[k]
                n_opt = mr / mm if mm > 0.0 else 1.0
                # MSE at this (T, vy) candidate
                mse_sum = 0.0
                for k in range(len(model_obs)):
                    r = n_opt * model_obs[k] - cr[k]
                    mse_sum += r * r
                mse = mse_sum / len(model_obs)
                if mse < best_mse:
                    best_mse = mse
        mse_flat[flat_idx] = best_mse
    return mse_flat

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_L2_CDF = (
    _REPO_ROOT / "tests/test_data/swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf"
)
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_N_SWEEPS = 5
_N_BINS = 71
_DT_S = 12.0 / 72
_SWEEP_S = 12.0
_SPIN_S = 15.0
_R_BASE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Representative case from the synthetic benchmark (seed=7, idx=408).
# Truth has large positive vT (+45.8 km/s), so the production IG (vT=-30, vN=0)
# sits on the mirror-basin side of the saddle. LM-from-IG converges to the
# mirror at (-45.5, -10.4) — 93.5 km/s from truth, 0.8 km/s from the exact
# mirror. The K=4 grid detects this and restarts LM from the truth basin.
CASE_V_R = 440.96
CASE_T_EV = 28.28
CASE_T_K = CASE_T_EV * EV_TO_KELVIN
CASE_DENSITY = 17.63
CASE_VT = 45.7861
CASE_VN = 9.7115
CASE_SEED = 1408  # np.random.default_rng(1000 + 408)


def _spin_rotation_matrices(n: int) -> np.ndarray:
    sweep_idx = np.arange(n) // _N_BINS
    bin_in_sweep = (np.arange(n) % _N_BINS) + 1
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R[i] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ _R_BASE
    return R


def _svd_orthogonalize(M: np.ndarray) -> np.ndarray:
    """Return the nearest rotation matrix to M via SVD."""
    U, _, Vt = np.linalg.svd(M)
    # Fix determinant to +1 (ensure proper rotation, not reflection)
    d = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def main():
    t0 = time.perf_counter()
    sr = load_swapi_response()
    with spacepy.pycdf.CDF(str(_TEST_L2_CDF)) as cdf:
        voltages = (
            cdf["esa_energy"][...].mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR
        )
    all_voltages = np.tile(voltages, _N_SWEEPS)
    sr.warm_cache(all_voltages)
    tiled = numba.typed.List([sr.create_passband_grid(v) for v in all_voltages])
    tiled_cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in all_voltages]
    )
    tiled_cea = np.array([sr.get_central_effective_area(v) for v in all_voltages])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    rot = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)

    # --- Compute R_avg: SVD-orthogonalized mean rotation matrix (RTN → body) ---
    R_avg = _svd_orthogonalize(rot.mean(axis=0))
    print(f"R_avg =\n{R_avg}")
    print(f"det(R_avg) = {np.linalg.det(R_avg):.6f}")

    # --- Truth and mirror in body-frame ---
    true_vel_rtn = np.array([CASE_V_R, CASE_VT, CASE_VN])
    truth_body = R_avg @ true_vel_rtn
    # Mirror: spin-axis (body-y) reflection flips vx_body and vz_body
    mirror_rtn = np.array([CASE_V_R, -CASE_VT, -CASE_VN])
    mirror_body = R_avg @ mirror_rtn
    print(f"\ntruth_body = ({truth_body[0]:.2f}, {truth_body[1]:.2f}, {truth_body[2]:.2f}) km/s")
    print(f"mirror_body = ({mirror_body[0]:.2f}, {mirror_body[1]:.2f}, {mirror_body[2]:.2f}) km/s")
    print(f"mirror_body should ≈ (-truth_body[0], truth_body[1], -truth_body[2]) = "
          f"({-truth_body[0]:.2f}, {truth_body[1]:.2f}, {-truth_body[2]:.2f})")

    # Spin component (body-y) — held fixed in the 2D grid
    vy_truth = truth_body[1]

    # --- Synthesize data ---
    cr_clean_true = _model_count_rates(
        CASE_DENSITY,
        CASE_T_K,
        true_vel_rtn,
        tiled,
        tiled_cs,
        tiled_cea,
        at,
        ats,
        rot,
        PROTON_MASS_KG,
    )
    cr_clean = apply_deadtime_correction_array(cr_clean_true)
    cr = (
        np.random.default_rng(CASE_SEED)
        .poisson(np.maximum(cr_clean, 0.0))
        .astype(float)
    )

    # --- Get production initial guess and run LM from it (no K=4 grid) ---
    print("\nComputing production initial guess...")
    ig = _get_initial_guess(
        cr, all_voltages, tiled, tiled_cs, tiled_cea, at, ats, rot
    )
    ig_vel = np.array(ig.bulk_velocity_rtn, dtype=float)
    ig_body = R_avg @ ig_vel
    print(f"IG: vel=({ig_vel[0]:.1f}, {ig_vel[1]:.1f}, {ig_vel[2]:.1f}) km/s  "
          f"n={ig.density:.2f} T={ig.temperature/EV_TO_KELVIN:.2f} eV")
    print(f"IG body-frame: ({ig_body[0]:.2f}, {ig_body[1]:.2f}, {ig_body[2]:.2f}) km/s")

    def residuals(x):
        return _residuals_njit(x, cr, tiled, tiled_cs, tiled_cea, at, ats, rot, PROTON_MASS_KG)

    def run_lm(x0):
        return scipy.optimize.least_squares(
            residuals, x0, method="lm", diff_step=1e-4, xtol=1e-10,
        )

    print("Running LM from production IG (no K=4 grid)...")
    x_ig = np.array([np.log(ig.density), np.log(ig.temperature),
                     ig_vel[0], ig_vel[1], ig_vel[2]])
    lm_ig = run_lm(x_ig)
    lm_ig_vel_rtn = lm_ig.x[2:5]
    lm_ig_body = R_avg @ lm_ig_vel_rtn
    mse_from_ig = float(np.mean(lm_ig.fun**2))
    print(f"LM-from-IG converged: vT={lm_ig_vel_rtn[1]:+.1f} vN={lm_ig_vel_rtn[2]:+.1f}  "
          f"body=({lm_ig_body[0]:.2f}, {lm_ig_body[1]:.2f}, {lm_ig_body[2]:.2f})")
    print(f"MSE (LM-from-IG) = {mse_from_ig:.3e}")

    print("Running LM from truth IG (to find truth-basin minimum)...")
    x_truth = np.array([np.log(CASE_DENSITY), np.log(CASE_T_K),
                        CASE_V_R, CASE_VT, CASE_VN])
    lm_truth = run_lm(x_truth)
    lm_truth_vel_rtn = lm_truth.x[2:5]
    lm_truth_body = R_avg @ lm_truth_vel_rtn
    mse_truth = float(np.mean(lm_truth.fun**2))
    print(f"LM-from-truth converged: vT={lm_truth_vel_rtn[1]:+.1f} vN={lm_truth_vel_rtn[2]:+.1f}  "
          f"body=({lm_truth_body[0]:.2f}, {lm_truth_body[1]:.2f}, {lm_truth_body[2]:.2f})")
    print(f"MSE (truth basin) = {mse_truth:.3e}")

    dist_to_truth = float(np.hypot(lm_ig_vel_rtn[1] - lm_truth_vel_rtn[1],
                                   lm_ig_vel_rtn[2] - lm_truth_vel_rtn[2]))
    print(f"\nDistance from LM-from-IG to truth basin: {dist_to_truth:.1f} km/s")
    print(f"MSE ratio (mirror/truth): {mse_from_ig/mse_truth:.0f}x")

    print("\nComputing profile-MSE grid in body-frame perpendicular plane...")
    print("(parallel numba: closed-form density rescale + 5×5 T/vy_body mini-grid per point)")
    t_grid_start = time.perf_counter()

    # Grid spans truth and mirror with room for both basins
    half_range = 75.0
    grid_vx = np.linspace(-half_range, half_range, 61)
    grid_vz = np.linspace(-half_range, half_range, 61)

    # Mini grid for (T, vy_body): 5 log-spaced T values (0.5×–2× truth),
    # 5 vy_body values spanning ±50 km/s around truth.
    K_T = 5
    K_VY = 5
    T_candidates = np.exp(np.linspace(np.log(CASE_T_K * 0.5), np.log(CASE_T_K * 2.0), K_T))
    vy_candidates = np.linspace(vy_truth - 50.0, vy_truth + 50.0, K_VY)
    R_avg_T = np.ascontiguousarray(R_avg.T)

    # JIT warm-up: compile on a tiny 2×2 sub-grid before the full run
    _compute_mse_grid(
        grid_vx[:2], grid_vz[:2], T_candidates, vy_candidates,
        R_avg_T, tiled, tiled_cs, tiled_cea, at, ats, rot, cr, PROTON_MASS_KG,
    )
    print(f"  JIT warm-up done ({time.perf_counter() - t_grid_start:.1f}s)")

    # Full parallel compute: 61×61×25 model evaluations across all available threads
    mse_flat = _compute_mse_grid(
        grid_vx, grid_vz, T_candidates, vy_candidates,
        R_avg_T, tiled, tiled_cs, tiled_cea, at, ats, rot, cr, PROTON_MASS_KG,
    )
    mse_grid = mse_flat.reshape(len(grid_vz), len(grid_vx))

    t_grid_end = time.perf_counter()
    print(f"  Grid compute done in {t_grid_end - t_grid_start:.1f}s")

    # --- Read off MSE at truth and mirror grid cells ---
    ix_truth = int(np.abs(grid_vx - truth_body[0]).argmin())
    iz_truth = int(np.abs(grid_vz - truth_body[2]).argmin())
    ix_mirror = int(np.abs(grid_vx - mirror_body[0]).argmin())
    iz_mirror = int(np.abs(grid_vz - mirror_body[2]).argmin())
    mse_truth_grid = mse_grid[iz_truth, ix_truth]
    mse_mirror_grid = mse_grid[iz_mirror, ix_mirror]

    # Read off MSE at LM-from-IG converged point
    ix_lm_ig = int(np.abs(grid_vx - lm_ig_body[0]).argmin())
    iz_lm_ig = int(np.abs(grid_vz - lm_ig_body[2]).argmin())
    mse_lm_ig_grid = mse_grid[iz_lm_ig, ix_lm_ig]

    print(f"\nMSE at truth grid cell  ≈ {mse_truth_grid:.3e}  "
          f"[vx={grid_vx[ix_truth]:.1f}, vz={grid_vz[iz_truth]:.1f}]")
    print(f"MSE at mirror grid cell ≈ {mse_mirror_grid:.3e}  "
          f"[vx={grid_vx[ix_mirror]:.1f}, vz={grid_vz[iz_mirror]:.1f}]")
    print(f"MSE at LM-from-IG cell  ≈ {mse_lm_ig_grid:.3e}  "
          f"[vx={grid_vx[ix_lm_ig]:.1f}, vz={grid_vz[iz_lm_ig]:.1f}]")
    print(f"Ratio (mirror/truth): {mse_mirror_grid / mse_truth_grid:.0f}×")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    log_mse = np.log10(mse_grid)
    cf = ax.contourf(grid_vx, grid_vz, log_mse, levels=30, cmap="viridis")
    ax.contour(
        grid_vx, grid_vz, log_mse, levels=10, colors="white", linewidths=0.5, alpha=0.5
    )
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r"$\log_{10}$ MSE", fontsize=11)

    # Truth (global minimum) — body-frame perpendicular coords
    ax.plot(
        truth_body[0],
        truth_body[2],
        "*",
        color="lime",
        ms=28,
        mec="black",
        mew=2,
        label=f"Truth (global min): ({truth_body[0]:+.1f}, {truth_body[2]:+.1f}),  "
        f"MSE $\\approx {mse_truth_grid:.0f}$",
        zorder=5,
    )
    # Mirror (local minimum)
    ax.plot(
        mirror_body[0],
        mirror_body[2],
        "X",
        color="red",
        ms=20,
        mec="black",
        mew=2,
        label=f"Spin-axis mirror (wrong basin): ({mirror_body[0]:+.1f}, {mirror_body[2]:+.1f}),  "
        f"MSE $\\approx {mse_mirror_grid:.0f}$ ({mse_mirror_grid / mse_truth_grid:.0f}$\\times$ truth)",
        zorder=5,
    )

    # Production IG position in body-frame
    ax.plot(
        ig_body[0],
        ig_body[2],
        "D",
        color="orange",
        ms=14,
        mec="black",
        mew=1.5,
        label=f"Production IG: ({ig_body[0]:+.1f}, {ig_body[2]:+.1f})",
        zorder=6,
    )

    # LM-from-IG converged point
    ax.plot(
        lm_ig_body[0],
        lm_ig_body[2],
        "o",
        color="yellow",
        ms=16,
        mec="black",
        mew=1.5,
        label=f"LM-from-IG converges here (no K=4 grid): ({lm_ig_body[0]:+.1f}, {lm_ig_body[2]:+.1f}),  "
              f"MSE $\\approx {mse_from_ig:.0f}$",
        zorder=6,
    )

    # Arrow from IG to LM-converged point
    ax.annotate(
        "",
        xy=(lm_ig_body[0], lm_ig_body[2]),
        xytext=(ig_body[0], ig_body[2]),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5, connectionstyle="arc3,rad=0.1"),
        zorder=7,
    )

    # Arrow from LM-converged to truth (what K=4 grid achieves)
    ax.annotate(
        "K=4 grid\nrecovers truth",
        xy=(truth_body[0], truth_body[2]),
        xytext=(lm_ig_body[0] * 0.5 + truth_body[0] * 0.0,
                lm_ig_body[2] * 0.5 + truth_body[2] * 0.0),
        fontsize=9,
        color="lime",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="lime", lw=1.5),
        zorder=7,
    )

    ax.set_xlabel(r"$v_x$ (body, km/s)", fontsize=12)
    ax.set_ylabel(r"$v_z$ (body, km/s)", fontsize=12)
    ax.set_title(
        "Without K=4 grid, LM converges to mirror; K=4 grid recovers truth\n"
        "(body-frame perpendicular plane; spin-axis component re-optimized)\n"
        f"$v_R={CASE_V_R:.0f}$ km/s, $T={CASE_T_EV:.1f}$ eV, $n={CASE_DENSITY:.1f}$ cm$^{{-3}}$, "
        f"$v_T=+{CASE_VT:.1f}$, $v_N=+{CASE_VN:.1f}$ km/s",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.set_xlim(grid_vx[0], grid_vx[-1])
    ax.set_ylim(grid_vz[0], grid_vz[-1])
    ax.grid(alpha=0.3, color="white")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "wrong_basin.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
