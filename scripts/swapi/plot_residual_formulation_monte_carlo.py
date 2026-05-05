#!/usr/bin/env python3
"""
Compare three least-squares formulations for SWAPI proton moment fitting:
  1. Unweighted log-space (current): residual = log(model) - log(data)
  2. sqrt(N)-weighted linear (previous): residual = (model - data) / sigma_poisson
  3. Unweighted linear: residual = model - data

Generates synthetic noisy spectra for several SW regimes, fits with each
formulation, and plots results.

Output: docs/swapi/figures/residual_formulation_monte_carlo.png
Usage: python scripts/swapi/plot_residual_formulation_monte_carlo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.optimize

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    EV_TO_KELVIN,
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    SWAPI_LIVETIME_S,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _model_count_rates,
    _optimize,
    apply_deadtime_correction_array,
    ProtonSolarWindMoments,
    _get_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"

_N_BINS = 71
_SWEEP_S = 12.0
_SPIN_S = 15.0
_DT_S = _SWEEP_S / 72
_R_BASE_RTN_TO_SWAPI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _thermal_speed(temperature_k):
    return float(
        np.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature_k / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )


def _realistic_rotation_matrices(n_total):
    sweep_idx = np.arange(n_total) // _N_BINS
    bin_in_sweep = (np.arange(n_total) % _N_BINS) + 1
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n_total, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        R[i] = R_spin @ _R_BASE_RTN_TO_SWAPI
    return R


def _load_swapi_response():
    return SwapiResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )


def _build_proton_arrays(sr, voltages):
    sr.warm_cache(voltages)
    grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    cs = np.array([sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in voltages])
    cea = np.array([sr.get_central_effective_area(v) for v in voltages])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    return grids, cs, cea, at, ats


def _generate_synthetic_data(sr, truth_v_rtn, truth_n, truth_T, n_sweeps=5, seed=42):
    import spacepy.pycdf
    cdf_path = _REPO_ROOT / "tests" / "test_data" / "swapi" / "imap_swapi_l2_50-sweeps_20250606_v003.cdf"
    with spacepy.pycdf.CDF(str(cdf_path)) as cdf:
        esa_energy = cdf["esa_energy"][...]
    voltages_single = esa_energy.mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR
    voltages = np.tile(voltages_single, n_sweeps)

    n_total = len(voltages)
    rot = _realistic_rotation_matrices(n_total)
    grids, cs, cea, at, ats = _build_proton_arrays(sr, voltages)

    true_rates = _model_count_rates(
        truth_n, truth_T, truth_v_rtn, grids, cs, cea, at, ats, rot, PROTON_MASS_KG
    )
    observed_rates = apply_deadtime_correction_array(true_rates)

    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum(observed_rates * SWAPI_LIVETIME_S, 0.01))
    noisy_rates = counts / SWAPI_LIVETIME_S
    noisy_rates = np.maximum(noisy_rates, 1.0 / SWAPI_LIVETIME_S)

    return voltages, noisy_rates, observed_rates, rot, grids, cs, cea, at, ats


def _apply_mask(sr, voltages, noisy_rates, clean_rates, rot):
    keep = (voltages > 0) & np.isfinite(voltages) & (noisy_rates > 0)
    cr_max = float(np.nanmax(noisy_rates[keep]))
    tail_mask = noisy_rates >= 0.1 * cr_max
    if int((keep & tail_mask).sum()) >= 5:
        keep = keep & tail_mask

    v_m = voltages[keep]
    nr_m = noisy_rates[keep]
    cr_m = clean_rates[keep]
    rot_m = rot[keep]
    sr.warm_cache(v_m)
    grids_m = numba.typed.List([sr.create_passband_grid(v) for v in v_m])
    cs_m = np.array([sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in v_m])
    cea_m = np.array([sr.get_central_effective_area(v) for v in v_m])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    return v_m, nr_m, cr_m, rot_m, grids_m, cs_m, cea_m, at, ats


def _fit_log_unweighted(noisy_rates, grids, cs, cea, at, ats, rot, initial_guess):
    return _optimize(noisy_rates, grids, cs, cea, at, ats, rot, initial_guess)


def _fit_linear_weighted(noisy_rates, grids, cs, cea, at, ats, rot, initial_guess):
    sigma = np.sqrt(np.maximum(noisy_rates * SWAPI_LIVETIME_S, 1.0)) / SWAPI_LIVETIME_S
    x0 = np.array([
        np.log(initial_guess.density), np.log(initial_guess.temperature),
        *initial_guess.bulk_velocity_rtn,
    ])

    def residuals(x):
        model_true = _model_count_rates(
            np.exp(x[0]), np.exp(x[1]), x[2:5],
            grids, cs, cea, at, ats, rot, PROTON_MASS_KG,
        )
        return (apply_deadtime_correction_array(model_true) - noisy_rates) / sigma

    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)
    spin_axis_rtn = rot[0, 1, :].copy()
    v_rtn = result.x[2:5]
    v_flipped = 2.0 * float(np.dot(v_rtn, spin_axis_rtn)) * spin_axis_rtn - v_rtn
    x_f = result.x.copy(); x_f[2:5] = v_flipped
    rf = scipy.optimize.least_squares(residuals, x_f, method="lm", diff_step=1e-4)
    if np.sum(rf.fun**2) < np.sum(result.fun**2):
        result = rf
    return ProtonSolarWindMoments(
        density=float(np.exp(result.x[0])), temperature=float(np.exp(result.x[1])),
        bulk_velocity_rtn=result.x[2:5], bad_fit_flag=0,
    )


def _fit_linear_unweighted(noisy_rates, grids, cs, cea, at, ats, rot, initial_guess):
    x0 = np.array([
        np.log(initial_guess.density), np.log(initial_guess.temperature),
        *initial_guess.bulk_velocity_rtn,
    ])

    def residuals(x):
        model_true = _model_count_rates(
            np.exp(x[0]), np.exp(x[1]), x[2:5],
            grids, cs, cea, at, ats, rot, PROTON_MASS_KG,
        )
        return apply_deadtime_correction_array(model_true) - noisy_rates

    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)
    spin_axis_rtn = rot[0, 1, :].copy()
    v_rtn = result.x[2:5]
    v_flipped = 2.0 * float(np.dot(v_rtn, spin_axis_rtn)) * spin_axis_rtn - v_rtn
    x_f = result.x.copy(); x_f[2:5] = v_flipped
    rf = scipy.optimize.least_squares(residuals, x_f, method="lm", diff_step=1e-4)
    if np.sum(rf.fun**2) < np.sum(result.fun**2):
        result = rf
    return ProtonSolarWindMoments(
        density=float(np.exp(result.x[0])), temperature=float(np.exp(result.x[1])),
        bulk_velocity_rtn=result.x[2:5], bad_fit_flag=0,
    )


# ---- Monte Carlo comparison ----

def _run_monte_carlo(sr, truth_v_rtn, truth_n, truth_T, n_trials=100, n_sweeps=5):
    results = {name: {"n": [], "T": [], "speed": [], "vR": [], "vT": [], "vN": []}
               for name in ["log", "sqrtN", "linear"]}

    for trial in range(n_trials):
        voltages, noisy, clean, rot, grids, cs, cea, at, ats = \
            _generate_synthetic_data(sr, truth_v_rtn, truth_n, truth_T, n_sweeps, seed=trial)

        vm, nrm, crm, rotm, gm, csm, ceam, at, ats = _apply_mask(sr, voltages, noisy, clean, rot)

        ig = _get_initial_guess(nrm, vm, gm, csm, ceam, at, ats, rotm)

        fitters = [
            ("log", _fit_log_unweighted),
            ("sqrtN", _fit_linear_weighted),
            ("linear", _fit_linear_unweighted),
        ]
        for name, fitter in fitters:
            try:
                fit = fitter(nrm, gm, csm, ceam, at, ats, rotm, ig)
                results[name]["n"].append(fit.density)
                results[name]["T"].append(fit.temperature)
                results[name]["speed"].append(float(np.linalg.norm(fit.bulk_velocity_rtn)))
                results[name]["vR"].append(float(fit.bulk_velocity_rtn[0]))
                results[name]["vT"].append(float(fit.bulk_velocity_rtn[1]))
                results[name]["vN"].append(float(fit.bulk_velocity_rtn[2]))
            except Exception:
                results[name]["n"].append(np.nan)
                results[name]["T"].append(np.nan)
                results[name]["speed"].append(np.nan)
                results[name]["vR"].append(np.nan)
                results[name]["vT"].append(np.nan)
                results[name]["vN"].append(np.nan)

    for name in results:
        for key in results[name]:
            results[name][key] = np.array(results[name][key])
    return results


CASES = [
    ("Nominal: 450 km/s, 10 eV, n=5",
     np.array([450.0, -30.0, 5.0]), 5.0, 10.0 * EV_TO_KELVIN),
    ("Cold: 400 km/s, 1 eV, n=5",
     np.array([400.0, -20.0, 0.0]), 5.0, 1.0 * EV_TO_KELVIN),
    ("Low density: 450 km/s, 10 eV, n=0.5",
     np.array([450.0, -30.0, 5.0]), 0.5, 10.0 * EV_TO_KELVIN),
    ("Hot: 450 km/s, 50 eV, n=5",
     np.array([450.0, -30.0, 5.0]), 5.0, 50.0 * EV_TO_KELVIN),
]

FITTER_LABELS = {
    "log": "Log unweighted",
    "sqrtN": "√N-weighted linear",
    "linear": "Unweighted linear",
}
FITTER_COLORS = {"log": "C0", "sqrtN": "C1", "linear": "C2"}


def main():
    print("Loading SWAPI response...")
    sr = _load_swapi_response()

    n_trials = 100
    fig, axes = plt.subplots(len(CASES), 4, figsize=(18, 4 * len(CASES)))

    param_labels = [
        ("n", "Density (cm⁻³)"),
        ("T", "Temperature (K)"),
        ("speed", "|v| (km/s)"),
        ("vT", "v_T (km/s)"),
    ]
    param_truths_keys = ["n", "T", "speed", "vT"]

    for i_case, (label, truth_v, truth_n, truth_T) in enumerate(CASES):
        truth_speed = float(np.linalg.norm(truth_v))
        truths = {"n": truth_n, "T": truth_T, "speed": truth_speed,
                  "vR": truth_v[0], "vT": truth_v[1], "vN": truth_v[2]}

        print(f"\n--- {label} ({n_trials} trials) ---")
        mc = _run_monte_carlo(sr, truth_v, truth_n, truth_T, n_trials=n_trials)

        # Print table
        print(f"  {'Formulation':25s} {'Param':>8s} {'Truth':>10s} {'Median':>10s} {'Bias':>10s} {'σ':>10s}")
        for pkey, plabel in param_labels:
            truth_val = truths[pkey]
            for name in ["log", "sqrtN", "linear"]:
                vals = mc[name][pkey]
                valid = vals[np.isfinite(vals)]
                if len(valid) == 0:
                    continue
                bias = np.median(valid) - truth_val
                scatter = np.std(valid)
                print(f"  {FITTER_LABELS[name]:25s} {pkey:>8s} {truth_val:10.3g} {np.median(valid):10.3g} {bias:+10.3g} {scatter:10.3g}")

        for j, (pkey, plabel) in enumerate(param_labels):
            ax = axes[i_case, j]
            truth_val = truths[pkey]

            for name in ["log", "sqrtN", "linear"]:
                vals = mc[name][pkey]
                valid = vals[np.isfinite(vals)]
                if len(valid) == 0:
                    continue
                bias = np.median(valid) - truth_val
                scatter = np.std(valid)

                ax.hist(valid, bins=25, alpha=0.4, color=FITTER_COLORS[name],
                        label=f'{FITTER_LABELS[name]}\nbias={bias:+.3g}, σ={scatter:.3g}',
                        density=True)

            ax.axvline(truth_val, color='k', lw=1.5, ls='--', label='Truth')
            ax.set_xlabel(plabel)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Monte Carlo comparison of residual formulations ({n_trials} noise realizations each)',
        fontsize=13, y=1.0
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = str(_REPO_ROOT / "docs" / "swapi" / "figures" / "residual_formulation_monte_carlo.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
