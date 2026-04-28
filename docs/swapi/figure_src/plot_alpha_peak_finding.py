#!/usr/bin/env python3
"""
Plot alpha peak-finding walkthrough on a synthetic SWAPI alpha spectrum.

Single-panel figure illustrating the count-rate forward model fit:

  Observed count rate vs ESA voltage (scatter) with overlaid models:
  - Frozen proton model (solid line)
  - Fitted alpha Gaussian component (dashed line)
  - Combined (proton + alpha) with deadtime correction (thick solid)

The alpha search window [2V_p*, 4V_p*] is shaded; true and fitted alpha
peak voltages are marked.

Spectrum parameters:
  proton  n=5 cm⁻³, T=10 eV, v_p=[450, 0, 0] km/s
  alpha   n=0.25 cm⁻³ (~5% abundance), T=10 eV, Δv=+30 km/s along B̂=[1, 0, 0]
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
_N_A, _T_A = 0.25, 10.0  # ~5% abundance, same temperature as protons
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
    # Call the production peak-finder.
    # ------------------------------------------------------------------
    v_p_speed = float(np.linalg.norm(_V_P_RTN))
    peak_fit = _alpha_peak_fit(
        count_avg, proton_obs_avg, voltage_per_sweep, _N_SWEEPS, v_p_speed
    )
    assert peak_fit is not None, "No alpha signal detected in synthetic spectrum"

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

    # Compute the three model curves for visualization.
    abs_voltage = np.abs(voltage_per_sweep)
    sort_idx = np.argsort(abs_voltage)
    abs_v_s = abs_voltage[sort_idx]
    count_avg_s = count_avg[sort_idx]
    proton_obs_avg_s = peak_fit.proton_obs_avg[sort_idx]

    # Alpha Gaussian component and combined model (with deadtime).
    alpha_model = peak_fit.gauss_A * np.exp(
        -((peak_fit.alpha_speeds - peak_fit.bulk_speed) ** 2)
        / (2 * peak_fit.sigma_v**2)
    )
    combined_no_dt = peak_fit.proton_obs_avg + alpha_model
    combined_with_dt = apply_deadtime_correction_array(combined_no_dt)

    alpha_model_s = alpha_model[sort_idx]
    combined_with_dt_s = combined_with_dt[sort_idx]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        "Alpha peak-finding count-rate forward model fit\n"
        r"$n_p=5\,\mathrm{cm}^{-3},\ T_p=10\,\mathrm{eV},\ v_p=450\,\mathrm{km/s}$;  "
        r"$n_\alpha=0.25\,\mathrm{cm}^{-3}\ (\sim 5\%),\ T_\alpha=10\,\mathrm{eV},\ \Delta v=+30\,\mathrm{km/s}$",
        fontsize=11,
    )

    # Observed data.
    ax.plot(
        abs_v_s,
        count_avg_s,
        ".",
        color="tab:blue",
        markersize=6,
        label="Observed (5-sweep avg)",
        zorder=3,
    )

    # Frozen proton model.
    ax.plot(
        abs_v_s,
        proton_obs_avg_s,
        color="tab:orange",
        lw=2.0,
        label=r"Proton model $R_p(V)$",
        zorder=2,
    )

    # Alpha Gaussian component.
    ax.plot(
        abs_v_s,
        alpha_model_s,
        color="forestgreen",
        lw=2.0,
        ls="--",
        label=rf"Alpha Gaussian: $\hat{{v}}_\alpha={peak_fit.bulk_speed:.0f}$ km/s, "
        rf"$\hat{{T}}_\alpha={peak_fit.T_alpha:.0f}$ eV",
        zorder=2,
    )

    # Combined with deadtime correction.
    ax.plot(
        abs_v_s,
        combined_with_dt_s,
        color="tab:red",
        lw=2.5,
        label=r"Proton + Alpha (deadtime corrected)",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.axvspan(
        peak_fit.alpha_min_voltage,
        peak_fit.alpha_max_voltage,
        alpha=0.08,
        color="tab:green",
        label=r"$\alpha$ search window $[2V_p^*,\,4V_p^*]$",
        zorder=1,
    )
    ax.axvline(
        v_a_peak_voltage,
        color="black",
        lw=1.5,
        ls=":",
        label=rf"True $v_\alpha = {true_alpha_speed:.0f}$ km/s",
        zorder=1.5,
    )
    ax.axvline(
        proton_peak_voltage,
        color="sienna",
        lw=1.0,
        ls=":",
        alpha=0.7,
        label=rf"$v_p^*$ = {v_p_speed:.0f} km/s",
        zorder=1.5,
    )

    ax.set_xlabel("ESA Voltage ($|V|$) [V]", fontsize=12)
    ax.set_ylabel("Count Rate [Hz]", fontsize=12)
    ax.set_ylim(0, 1e5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "alpha_peak_finding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
