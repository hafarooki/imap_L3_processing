#!/usr/bin/env python3
"""
Fit four least-squares formulations on real SWAPI L2 data and compare:
  1. Unweighted log-space (current): residual = log(model) - log(data)
  2. 1/sqrt(N)-weighted log-space: residual = (log(model) - log(data)) / (1/sqrt(N))
  3. sqrt(N)-weighted linear (previous): residual = (model - data) / sigma_poisson
  4. Unweighted linear: residual = model - data

Uses three 5-sweep chunks from L2 CDFs with different solar wind conditions
(slow/medium/fast). Each row is a chunk, each column is a sweep; all three
fitted model curves are overlaid on each sweep's count-rate spectrum.

Output: docs/swapi/figures/residual_formulation_real_data.png
Usage: python scripts/swapi/plot_residual_formulation_real_data.py
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
import spacepy.pycdf
import spiceypy

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    SWAPI_LIVETIME_S,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    _get_initial_guess,
    _model_count_rates,
    _optimize,
    apply_deadtime_correction_array,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry
from imap_l3_processing.utils import furnish_local_spice

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_L2_DATA_DIR = Path.home() / "projects" / "swapi-calibration"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

ONE_SECOND_IN_NS = 1_000_000_000

CHUNKS = [
    {
        "label": "Slow wind (~420 km/s)",
        "file": "imap_swapi_l2_sci_20260131_v004.cdf",
        "sweep_start": 3650,
    },
    {
        "label": "Medium wind (~510 km/s)",
        "file": "imap_swapi_l2_sci_20260101_v001.cdf",
        "sweep_start": 6950,
    },
    {
        "label": "Fast wind (~660 km/s)",
        "file": "imap_swapi_l2_sci_20260117_v001.cdf",
        "sweep_start": 3150,
    },
]


def _load_swapi_response():
    return SwapiResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )


def _build_proton_arrays(sr, voltages):
    sr.warm_cache(voltages)
    grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in voltages]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in voltages])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    return grids, cs, cea, at, ats


def _load_chunk(chunk_spec):
    """Load 5 sweeps of L2 data and build measurement times for SPICE."""
    cdf_path = _L2_DATA_DIR / chunk_spec["file"]
    s = chunk_spec["sweep_start"]
    with spacepy.pycdf.CDF(str(cdf_path)) as cdf:
        cr_full = cdf["swp_coin_rate"][s : s + 5, :]
        esa_full = cdf["esa_energy"][s : s + 5, :]
        epoch_full = cdf["epoch"][s : s + 5]

    cr = cr_full[:, SWAPI_SCIENCE_BINS]
    esa = esa_full[:, SWAPI_SCIENCE_BINS]

    # Build TT2000 measurement times (same as _measurement_times_for_chunk)
    bin_indices = np.arange(SWAPI_SCIENCE_BINS.start, SWAPI_SCIENCE_BINS.stop)
    sci_start_tt2000 = np.array(
        [spiceypy.str2et(str(e)) for e in epoch_full], dtype=float
    )
    # Actually we need TT2000 nanoseconds. spacepy epochs are datetimes;
    # convert to TT2000 via unitim.
    # Easier: use the CDF epoch directly (TT2000 int64).
    with spacepy.pycdf.CDF(str(cdf_path)) as cdf:
        raw_epoch = cdf.raw_var("epoch")[s : s + 5]  # int64 TT2000

    measurement_times = (
        raw_epoch[:, np.newaxis] + bin_indices * int(12 / 72 * ONE_SECOND_IN_NS)
    ).flatten()

    return cr, esa, epoch_full, measurement_times


def _apply_mask(sr, voltages_flat, cr_flat, rot):
    keep = (voltages_flat > 0) & np.isfinite(voltages_flat) & (cr_flat > 0)
    cr_max = float(np.nanmax(cr_flat[keep])) if np.any(keep) else 0.0
    tail_mask = cr_flat >= 0.1 * cr_max
    if int((keep & tail_mask).sum()) >= 5:
        keep = keep & tail_mask

    vm = voltages_flat[keep]
    crm = cr_flat[keep]
    rotm = rot[keep]
    grids_m, cs_m, cea_m, at, ats = _build_proton_arrays(sr, vm)
    return vm, crm, rotm, grids_m, cs_m, cea_m, at, ats, keep


def _fit_log_unweighted(cr, grids, cs, cea, at, ats, rot, ig):
    return _optimize(cr, grids, cs, cea, at, ats, rot, ig)


def _fit_linear_weighted(cr, grids, cs, cea, at, ats, rot, ig):
    sigma = np.sqrt(np.maximum(cr * SWAPI_LIVETIME_S, 1.0)) / SWAPI_LIVETIME_S
    x0 = np.array([np.log(ig.density), np.log(ig.temperature), *ig.bulk_velocity_rtn])

    def residuals(x):
        model_true = _model_count_rates(
            np.exp(x[0]),
            np.exp(x[1]),
            x[2:5],
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )
        return (apply_deadtime_correction_array(model_true) - cr) / sigma

    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)
    spin_axis = rot[0, 1, :].copy()
    v = result.x[2:5]
    vf = 2.0 * float(np.dot(v, spin_axis)) * spin_axis - v
    xf = result.x.copy()
    xf[2:5] = vf
    rf = scipy.optimize.least_squares(residuals, xf, method="lm", diff_step=1e-4)
    if np.sum(rf.fun**2) < np.sum(result.fun**2):
        result = rf
    return ProtonSolarWindMoments(
        density=float(np.exp(result.x[0])),
        temperature=float(np.exp(result.x[1])),
        bulk_velocity_rtn=result.x[2:5],
        bad_fit_flag=0,
    )


def _fit_log_weighted(cr, grids, cs, cea, at, ats, rot, ig):
    sigma = 1.0 / np.sqrt(np.maximum(cr * SWAPI_LIVETIME_S, 1.0))
    x0 = np.array([np.log(ig.density), np.log(ig.temperature), *ig.bulk_velocity_rtn])

    def residuals(x):
        model_true = _model_count_rates(
            np.exp(x[0]),
            np.exp(x[1]),
            x[2:5],
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )
        model = apply_deadtime_correction_array(model_true)
        return (np.log(model) - np.log(cr)) / sigma

    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)
    spin_axis = rot[0, 1, :].copy()
    v = result.x[2:5]
    vf = 2.0 * float(np.dot(v, spin_axis)) * spin_axis - v
    xf = result.x.copy()
    xf[2:5] = vf
    rf = scipy.optimize.least_squares(residuals, xf, method="lm", diff_step=1e-4)
    if np.sum(rf.fun**2) < np.sum(result.fun**2):
        result = rf
    return ProtonSolarWindMoments(
        density=float(np.exp(result.x[0])),
        temperature=float(np.exp(result.x[1])),
        bulk_velocity_rtn=result.x[2:5],
        bad_fit_flag=0,
    )


def _fit_linear_unweighted(cr, grids, cs, cea, at, ats, rot, ig):
    x0 = np.array([np.log(ig.density), np.log(ig.temperature), *ig.bulk_velocity_rtn])

    def residuals(x):
        model_true = _model_count_rates(
            np.exp(x[0]),
            np.exp(x[1]),
            x[2:5],
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )
        return apply_deadtime_correction_array(model_true) - cr

    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)
    spin_axis = rot[0, 1, :].copy()
    v = result.x[2:5]
    vf = 2.0 * float(np.dot(v, spin_axis)) * spin_axis - v
    xf = result.x.copy()
    xf[2:5] = vf
    rf = scipy.optimize.least_squares(residuals, xf, method="lm", diff_step=1e-4)
    if np.sum(rf.fun**2) < np.sum(result.fun**2):
        result = rf
    return ProtonSolarWindMoments(
        density=float(np.exp(result.x[0])),
        temperature=float(np.exp(result.x[1])),
        bulk_velocity_rtn=result.x[2:5],
        bad_fit_flag=0,
    )


def _thermal_speed_from_T(T_K):
    return float(
        np.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * T_K / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )


def main():
    print("Furnishing SPICE kernels...")
    furnish_local_spice()

    print("Loading SWAPI response...")
    sr = _load_swapi_response()

    n_chunks = len(CHUNKS)
    n_sweeps = 5
    fig, axes = plt.subplots(
        n_chunks, n_sweeps, figsize=(4 * n_sweeps, 4 * n_chunks), squeeze=False
    )

    fitter_specs = [
        ("Log unweighted", "C0", "-", _fit_log_unweighted),
        ("1/√N-weighted log", "C3", "-.", _fit_log_weighted),
        ("√N-weighted linear", "C1", "--", _fit_linear_weighted),
        ("Unweighted linear", "C2", ":", _fit_linear_unweighted),
    ]

    for i_chunk, chunk_spec in enumerate(CHUNKS):
        print(f"\n--- {chunk_spec['label']} ---")
        print(
            f"  Loading {chunk_spec['file']} sweeps {chunk_spec['sweep_start']}–{chunk_spec['sweep_start'] + 4}..."
        )
        cr_2d, esa_2d, epochs, meas_times = _load_chunk(chunk_spec)

        n_bins = cr_2d.shape[1]
        voltages_flat = (esa_2d / SWAPI_L2_K_FACTOR).flatten()
        cr_flat = cr_2d.flatten()

        print("  Computing SPICE geometry...")
        rot = get_swapi_geometry(meas_times)

        print("  Applying 10%% mask and building passband grids...")
        vm, crm, rotm, gm, csm, ceam, at, ats, keep_mask = _apply_mask(
            sr, voltages_flat, cr_flat, rot
        )
        print(f"  Bins after mask: {len(crm)} / {len(cr_flat)}")

        print("  Computing initial guess...")
        ig = _get_initial_guess(crm, vm, gm, csm, ceam, at, ats, rotm)

        fits = {}
        for label, color, ls, fitter in fitter_specs:
            print(f"  Fitting: {label}...")
            fit = fitter(crm, gm, csm, ceam, at, ats, rotm, ig)
            fits[label] = fit
            speed = float(np.linalg.norm(fit.bulk_velocity_rtn))
            vt = _thermal_speed_from_T(fit.temperature)
            print(
                f"    n={fit.density:.3f} cm⁻³, T={fit.temperature:.0f} K "
                f"(vth={vt:.1f} km/s), |v|={speed:.1f} km/s, "
                f"v_RTN=[{fit.bulk_velocity_rtn[0]:.1f}, {fit.bulk_velocity_rtn[1]:.1f}, {fit.bulk_velocity_rtn[2]:.1f}]"
            )

        # Evaluate model on full (valid) bins with actual per-bin rotation matrices
        valid_full = (voltages_flat > 0) & np.isfinite(voltages_flat)
        v_valid = voltages_flat[valid_full]
        rot_valid = rot[valid_full]
        grids_valid, cs_valid, cea_valid, _, _ = _build_proton_arrays(sr, v_valid)

        models = {}
        for label, _, _, _ in fitter_specs:
            fit = fits[label]
            model_true = _model_count_rates(
                fit.density,
                fit.temperature,
                fit.bulk_velocity_rtn,
                grids_valid,
                cs_valid,
                cea_valid,
                at,
                ats,
                rot_valid,
                PROTON_MASS_KG,
            )
            full = np.full(len(voltages_flat), np.nan)
            full[valid_full] = apply_deadtime_correction_array(model_true)
            models[label] = full

        cr_max_chunk = float(np.nanmax(cr_flat[cr_flat > 0]))
        bin_nums = np.arange(n_bins)

        for i_sweep in range(n_sweeps):
            ax = axes[i_chunk, i_sweep]
            sl = slice(i_sweep * n_bins, (i_sweep + 1) * n_bins)

            cr_sweep = cr_flat[sl]
            ax.semilogy(
                bin_nums,
                np.maximum(cr_sweep, 0.1),
                "k.",
                ms=3,
                alpha=0.6,
                label="Data" if i_sweep == 0 else None,
            )

            for label, color, ls, _ in fitter_specs:
                model_sweep = models[label][sl]
                ok = np.isfinite(model_sweep) & (model_sweep > 0)
                ax.semilogy(
                    bin_nums[ok],
                    model_sweep[ok],
                    color=color,
                    ls=ls,
                    lw=1.5,
                    label=label if i_sweep == 0 else None,
                )

            ax.axhline(
                0.1 * cr_max_chunk,
                color="gray",
                ls="--",
                lw=0.5,
                alpha=0.5,
            )

            ax.set_xlabel("Bin index", fontsize=8)
            if i_sweep == 0:
                ax.set_ylabel("Count rate (Hz)", fontsize=8)
            ax.set_title(f"Sweep {i_sweep + 1}", fontsize=9)
            ax.tick_params(labelsize=7)

            peak_cr = float(np.nanmax(cr_sweep))
            ax.set_ylim(max(0.1, peak_cr * 1e-4), peak_cr * 3)

        # Row label
        axes[i_chunk, 0].annotate(
            chunk_spec["label"],
            xy=(0, 0.5),
            xytext=(-0.45, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    # Add legend to top-left panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.suptitle(
        "Residual formulation comparison on real L2 data (5 sweeps × 3 conditions)",
        fontsize=13,
        y=1.03,
    )
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "residual_formulation_real_data.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    spiceypy.kclear()


if __name__ == "__main__":
    main()
