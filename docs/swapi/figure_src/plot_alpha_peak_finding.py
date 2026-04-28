#!/usr/bin/env python3
"""
Plot alpha peak-finding walkthrough on a synthetic SWAPI alpha spectrum.

Two-panel figure illustrating the _alpha_initial_guess steps:

  Panel (a): 5-sweep averaged observed count rate vs the frozen proton model.
             The alpha voltage search window [2V_p*, 4V_p*] is shaded; the
             true alpha peak voltage is marked.

  Panel (b): Log-space residual ℓ_i = ln(max(C_i, 0.1)) − ln(max(R_i^p, 0.1))
             inside the alpha window.  Gaussian fit overlaid; initial-guess peak
             vs ground-truth bulk speed both marked.

Spectrum parameters:
  proton  n=5 cm⁻³, T=10 eV, v_p=[450, 0, 0] km/s
  alpha   n=0.2 cm⁻³, T=40 eV, Δv=+30 km/s along B̂=[1, 0, 0]
  5 sweeps × 62 coarse bins, Poisson noise seed=7.

Output: docs/swapi/figures/alpha_peak_finding.png
Usage:  python docs/swapi/figure_src/plot_alpha_peak_finding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numba

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_moments import (
    _alpha_peak_fit,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _model_count_rates,
    apply_deadtime_correction_array,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_K_FACTOR,
    esa_voltage_to_alpha_speed,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_N_SWEEPS = 5
_N_BINS = 62
_DT_S = 12.0 / 72
_SWEEP_S = 12.0
_SPIN_S = 15.0
# Base rotation: RTN → SWAPI frame at t=0 (spin axis = boresight = +Y_SWAPI).
_R_BASE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Spectrum parameters.
_N_P, _T_P = 5.0, 10.0
_V_P_RTN = np.array([450.0, 0.0, 0.0])
_N_A, _T_A = 0.2, 40.0
_DELTA_V = 30.0
_B_HAT = np.array([1.0, 0.0, 0.0])
_V_A_RTN = _V_P_RTN + _DELTA_V * _B_HAT
_VOLTAGES = np.geomspace(60.0, 5000.0, _N_BINS)[::-1]
_SEED = 7


def _spin_rotation_matrices(n):
    """RTN → SWAPI rotation for each of n (sweep, bin) measurements."""
    sweep_idx = np.arange(n) // _N_BINS
    bin_in_sweep = (np.arange(n) % _N_BINS) + SWAPI_COARSE_SWEEP_BINS.start
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R[i] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ _R_BASE
    return R


def main():
    print("Loading calibration data...")
    sr = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    # --- Synthesize the combined (proton + alpha) observed spectrum ---
    print("Synthesizing spectrum...")
    n_meas = _N_SWEEPS * _N_BINS
    esa_flat = np.tile(_VOLTAGES, _N_SWEEPS)
    rot = _spin_rotation_matrices(n_meas)
    sc_vel = np.zeros(3)

    grids = numba.typed.List([sr.create_passband_grid(v) for v in esa_flat])
    p_cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
    )
    a_cs = np.array(
        [sr.central_speed(v, ALPHA_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in esa_flat])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    print("Computing true count rates (first Numba run compiles JIT)...")
    p_true = _model_count_rates(
        _N_P, _T_P, _V_P_RTN, grids, p_cs, cea, at, ats, rot, sc_vel, PROTON_MASS_KG
    )
    a_true = _model_count_rates(
        _N_A,
        _T_A,
        _V_A_RTN,
        grids,
        a_cs,
        cea,
        at,
        ats,
        rot,
        sc_vel,
        ALPHA_PARTICLE_MASS_KG,
    )
    obs_clean = apply_deadtime_correction_array(p_true + a_true)
    rng = np.random.default_rng(_SEED)
    livetime = 0.145  # seconds per ESA bin
    obs = rng.poisson(np.maximum(obs_clean * livetime, 0)).astype(float) / livetime

    # Per-bin sweep averages passed to the production peak-finder.
    proton_true_avg = p_true.reshape(_N_SWEEPS, _N_BINS).mean(axis=0)
    voltage_per_sweep = esa_flat.reshape(_N_SWEEPS, _N_BINS)[0]
    count_avg = obs.reshape(_N_SWEEPS, _N_BINS).mean(axis=0)
    proton_obs_avg = apply_deadtime_correction_array(proton_true_avg)

    # ------------------------------------------------------------------
    # Call the production peak-finder — no parallel reimplementation here.
    # ------------------------------------------------------------------
    v_p_speed = float(np.linalg.norm(_V_P_RTN))
    peak_fit = _alpha_peak_fit(
        count_avg, proton_obs_avg, voltage_per_sweep, _N_SWEEPS, v_p_speed
    )
    assert peak_fit is not None, "No alpha signal detected in synthetic spectrum"

    alpha_speeds = esa_voltage_to_alpha_speed(voltage_per_sweep)
    alpha_min_speed = float(esa_voltage_to_alpha_speed(peak_fit.alpha_min_voltage))
    alpha_max_speed = float(esa_voltage_to_alpha_speed(peak_fit.alpha_max_voltage))
    proton_peak_voltage = peak_fit.alpha_min_voltage / 2.0

    true_alpha_speed = float(np.linalg.norm(_V_A_RTN))
    v_a_peak_voltage = (
        ALPHA_PARTICLE_MASS_KG
        * (true_alpha_speed * 1e3) ** 2
        / (2.0 * SWAPI_K_FACTOR * ALPHA_PARTICLE_CHARGE_COULOMBS)
    )

    print(
        f"Initial guess: v_alpha = {peak_fit.bulk_speed:.1f} km/s  "
        f"(truth = {true_alpha_speed:.1f} km/s, "
        f"error = {abs(peak_fit.bulk_speed - true_alpha_speed):.1f} km/s)"
    )
    print(
        f"               T_alpha = {peak_fit.T_alpha:.1f} eV  (truth = {_T_A:.1f} eV)"
    )

    # Sort by ascending voltage for clean line plots (panel a).
    abs_voltage = np.abs(voltage_per_sweep)
    sort_idx = np.argsort(abs_voltage)
    abs_v_s = abs_voltage[sort_idx]
    count_avg_s = count_avg[sort_idx]
    proton_obs_avg_s = proton_obs_avg[sort_idx]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Alpha peak-finding on the validation spectrum\n"
        r"$n_p=5\,\mathrm{cm}^{-3},\ T_p=10\,\mathrm{eV},\ v_p=450\,\mathrm{km/s}$;  "
        r"$n_\alpha=0.20\,\mathrm{cm}^{-3},\ T_\alpha=40\,\mathrm{eV},\ \Delta v=+30\,\mathrm{km/s}$",
        fontsize=11,
    )

    # --- Panel (a): spectrum + search window ---
    ax1.plot(
        abs_v_s,
        count_avg_s,
        ".",
        color="tab:blue",
        markersize=5,
        label="Observed (5-sweep avg)",
    )
    ax1.plot(
        abs_v_s,
        proton_obs_avg_s,
        color="tab:orange",
        lw=1.8,
        ls="--",
        label=r"Frozen proton model $R_i^p$",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.axvspan(
        peak_fit.alpha_min_voltage,
        peak_fit.alpha_max_voltage,
        alpha=0.10,
        color="tab:green",
        label=r"$\alpha$ search window $[2V_p^*,\,4V_p^*]$",
    )
    ax1.axvline(
        v_a_peak_voltage,
        color="forestgreen",
        lw=1.5,
        ls=":",
        label=rf"True $v_\alpha$ = {true_alpha_speed:.0f} km/s",
    )
    ax1.axvline(
        proton_peak_voltage,
        color="sienna",
        lw=1.0,
        ls=":",
        alpha=0.7,
        label=rf"$v_p^*$ = {v_p_speed:.0f} km/s",
    )
    ax1.set_xlabel("ESA Voltage ($|V|$) [V]", fontsize=11)
    ax1.set_ylabel("Count Rate [Hz]", fontsize=11)
    ax1.set_title("(a) 5-sweep averaged spectrum")
    ax1.set_ylim(0, 1e5)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    # --- Panel (b): log-residual + Gaussian fit ---
    ax2.plot(
        peak_fit.fit_speeds[~peak_fit.alpha_mask],
        peak_fit.fit_log_res[~peak_fit.alpha_mask],
        ".",
        color="silver",
        markersize=5,
        label="Outside search window",
        zorder=1,
    )
    ax2.plot(
        peak_fit.fit_speeds[peak_fit.alpha_mask],
        peak_fit.fit_log_res[peak_fit.alpha_mask],
        "o",
        color="tab:blue",
        markersize=7,
        label=r"Log-residual $\ell_i$ in window",
        zorder=3,
    )
    v_dense = np.linspace(alpha_min_speed * 0.97, alpha_max_speed * 1.03, 300)
    gauss_curve = peak_fit.gauss_A * np.exp(
        -((v_dense - peak_fit.bulk_speed) ** 2) / (2 * peak_fit.sigma_v**2)
    )
    ax2.plot(
        v_dense,
        gauss_curve,
        color="tab:red",
        lw=2.5,
        label=(
            rf"Gaussian fit: $\hat{{v}}_\alpha = {peak_fit.bulk_speed:.0f}$ km/s, "
            rf"$\hat{{T}}_\alpha = {peak_fit.T_alpha:.0f}$ eV"
        ),
        zorder=4,
    )
    ax2.axvline(
        true_alpha_speed,
        color="black",
        lw=2.0,
        ls="--",
        label=rf"Ground truth: $v_\alpha = {true_alpha_speed:.0f}$ km/s",
        zorder=5,
    )
    ax2.axhline(
        np.log(2.0),
        color="gray",
        lw=1.2,
        ls=":",
        label=r"Guard threshold: $\ln 2$",
    )
    ax2.axvspan(alpha_min_speed, alpha_max_speed, alpha=0.06, color="tab:green")
    ax2.set_xlabel(r"Alpha central speed $v_0^\alpha$ (km/s)", fontsize=11)
    ax2.set_ylabel(r"Log-residual $\ell_i = \ln C_i - \ln R_i^p$", fontsize=11)
    ax2.set_title("(b) Log-residual and Gaussian fit")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "alpha_peak_finding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
