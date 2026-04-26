#!/usr/bin/env python3
"""
χ² landscape in the (v_T, v_N) plane illustrating the two-basin problem
solved by the post-fit flip check in `_optimize`.

For one representative case where LM-from-(0,0) lands in the wrong basin,
this figure shows the χ² surface (n, T, v_R fixed at truth) with two clear
minima — truth and its spin-axis mirror — separated by a saddle ridge through
(0, 0). The mirror has χ² ≳ 100× the truth, so a single residual evaluation
at the flipped solution is enough to detect the wrong-basin convergence.

Output: docs/swapi/figures/wrong_basin.png
Usage:  python docs/swapi/figure_src/plot_wrong_basin.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import numba
import spacepy.pycdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_SCIENCE_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _model_count_rates,
    _residuals_njit,
    SWAPI_LIVETIME_S,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
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

# Representative case from the synthetic benchmark — large deflection,
# warm plasma, where LM-from-(0,0) ends up in the mirror basin.
CASE_V_R = 378.0
CASE_T_EV = 26.31
CASE_DENSITY = 12.7
CASE_VT = 35.3
CASE_VN = -48.1
CASE_SEED = 402


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


def main():
    sr = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )
    with spacepy.pycdf.CDF(str(_TEST_L2_CDF)) as cdf:
        voltages = (
            cdf["esa_energy"][...].mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR
        )
    base_grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    tiled = numba.typed.List()
    for _ in range(_N_SWEEPS):
        for g in base_grids:
            tiled.append(g)
    rot = _spin_rotation_matrices(_N_SWEEPS * _N_BINS)
    sc_vel = np.zeros(3)

    true_vel = np.array([CASE_V_R, CASE_VT, CASE_VN])
    cr_clean = _model_count_rates(CASE_DENSITY, CASE_T_EV, true_vel, tiled, rot, sc_vel)
    cr = (
        np.random.default_rng(CASE_SEED)
        .poisson(np.maximum(cr_clean, 0.0))
        .astype(float)
    )
    sigma = np.sqrt(np.maximum(cr * SWAPI_LIVETIME_S, 1.0)) / SWAPI_LIVETIME_S

    print("Computing χ² grid (n, T, v_R held at truth)...")
    grid_vT = np.linspace(-80, 80, 51)
    grid_vN = np.linspace(-100, 100, 51)
    chi2 = np.zeros((len(grid_vN), len(grid_vT)))
    for i, vN_ in enumerate(grid_vN):
        for j, vT_ in enumerate(grid_vT):
            x = np.array([np.log(CASE_DENSITY), np.log(CASE_T_EV), CASE_V_R, vT_, vN_])
            r = _residuals_njit(x, cr, sigma, tiled, rot, sc_vel)
            chi2[i, j] = float(np.sum(r * r))

    fig, ax = plt.subplots(figsize=(9, 7))
    log_chi2 = np.log10(chi2)
    cf = ax.contourf(grid_vT, grid_vN, log_chi2, levels=30, cmap="viridis")
    ax.contour(
        grid_vT, grid_vN, log_chi2, levels=10, colors="white", linewidths=0.5, alpha=0.5
    )
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r"$\log_{10}\chi^2$", fontsize=11)

    # Closest grid points to truth and mirror for χ² readout in legend
    iT_truth = int(np.abs(grid_vT - CASE_VT).argmin())
    iN_truth = int(np.abs(grid_vN - CASE_VN).argmin())
    iT_mirror = int(np.abs(grid_vT + CASE_VT).argmin())
    iN_mirror = int(np.abs(grid_vN + CASE_VN).argmin())
    chi2_truth = chi2[iN_truth, iT_truth]
    chi2_mirror = chi2[iN_mirror, iT_mirror]

    # Truth (global minimum)
    ax.plot(
        CASE_VT,
        CASE_VN,
        "*",
        color="lime",
        ms=28,
        mec="black",
        mew=2,
        label=f"Truth (global min): ({CASE_VT:+.1f}, {CASE_VN:+.1f}),  "
        f"$\\chi^2 \\approx {chi2_truth:.0f}$",
        zorder=5,
    )
    # Mirror (local minimum)
    ax.plot(
        -CASE_VT,
        -CASE_VN,
        "X",
        color="red",
        ms=20,
        mec="black",
        mew=2,
        label=f"Spin-axis mirror (local min): ({-CASE_VT:+.1f}, {-CASE_VN:+.1f}),  "
        f"$\\chi^2 \\approx {chi2_mirror:.0f}$ ({chi2_mirror / chi2_truth:.0f}× truth)",
        zorder=5,
    )
    # Saddle / starting point
    ax.plot(
        0,
        0,
        "s",
        color="white",
        ms=14,
        mec="black",
        mew=2,
        label="Initial guess (0, 0): saddle between basins",
        zorder=5,
    )

    # Arrows from saddle to each basin
    ax.annotate(
        "",
        xy=(CASE_VT, CASE_VN),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="lime", lw=2),
    )
    ax.annotate(
        "",
        xy=(-CASE_VT, -CASE_VN),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    ax.set_xlabel("$v_T$ (km/s)", fontsize=12)
    ax.set_ylabel("$v_N$ (km/s)", fontsize=12)
    ax.set_title(
        "Two-basin problem in the $(v_T, v_N)$ plane\n"
        f"$v_R={CASE_V_R}$ km/s, $T={CASE_T_EV:.1f}$ eV, $n={CASE_DENSITY}$ cm$^{{-3}}$ "
        f"(n, T, $v_R$ fixed at truth)",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="lower right", framealpha=0.92)
    ax.set_xlim(grid_vT[0], grid_vT[-1])
    ax.set_ylim(grid_vN[0], grid_vN[-1])
    ax.grid(alpha=0.3, color="white")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "wrong_basin.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    print(f"χ² at truth ≈ {chi2_truth:.1f}")
    print(f"χ² at mirror ≈ {chi2_mirror:.1f}")
    print(f"Ratio: {chi2_mirror / chi2_truth:.0f}×")


if __name__ == "__main__":
    main()
