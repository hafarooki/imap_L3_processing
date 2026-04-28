#!/usr/bin/env python3
"""
Scatter plot: ground truth vs optimized integral for 10000 random SW conditions.

Ground truth: fixed-limit reference integrals from tests/.../reference_integrals.csv
Optimized:    dynamic-limit integral at the N values defined below

Adjust the N constants and re-run to assess accuracy.

Output: docs/swapi/figures/scatter_benchmark.png
Usage:  python scripts/swapi/scatter_benchmark.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imap_l3_processing.constants import METERS_PER_KILOMETER, PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, PROTON_CHARGE_OVER_MASS_C_PER_KG
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    SWParams, calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"
_REFERENCE_INTEGRALS_PATH = _REPO_ROOT / "tests" / "swapi" / "l3a" / "science" / "reference_integrals.csv"


def _peak_voltage(bulk_speed_km_s):
    return (PROTON_MASS_KG * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
            / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS))


def main():
    print("Loading calibration data...")
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    df = pd.read_csv(_REFERENCE_INTEGRALS_PATH)
    df = df[df.integral > 1]  # only consider 1 Hz and above

    truths = df['integral'].to_numpy()
    optimized = np.empty(len(df))

    # TODO incorporate background into model instead maybe?
    optimized = optimized + 1e-1
    truths = truths + 1e-1

    print(f"Computing {len(df)} optimized integrals...")
    for i, row in enumerate(df.itertuples(index=False)):
        thermal_speed = float(
            np.sqrt(row.temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
        )
        sw = SWParams(
            density=float(row.density),
            bulk_speed=float(row.bulk_speed),
            bulk_azimuth=float(row.bulk_azimuth),
            bulk_elevation=float(row.bulk_elevation),
            thermal_speed=thermal_speed,
        )
        grid = swapi_response.create_passband_grid(_peak_voltage(float(row.bulk_speed)))
        optimized[i] = calculate_integral(grid, sw)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    rel_errors = (optimized - truths) / truths
    print(f"\nMax |rel error|:    {np.abs(rel_errors).max():.2%}")
    print(f"Median |rel error|: {np.median(np.abs(rel_errors)):.2%}")
    print(f"95th pct |rel err|: {np.percentile(np.abs(rel_errors), 95):.2%}")

    log_t = np.log10(df['temperature_ev'].values)
    t_norm = matplotlib.colors.Normalize(vmin=log_t.min(), vmax=log_t.max())
    cmap = matplotlib.colormaps["coolwarm"]
    colors = cmap(t_norm(log_t))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(truths, optimized, c=colors, s=8, alpha=0.7, linewidths=0)

    positive = (truths > 0) & (optimized > 0)
    lo = 1e-1  # background level in https://link.springer.com/article/10.1007/s11214-025-01229-8/figures/29
    hi = max(truths[positive].max(), optimized[positive].max())
    lo, hi = lo / 2, hi * 2
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Ground truth count rate (fixed limits)")
    ax.set_ylabel("Optimized count rate (dynamic limits)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
        N_ELEVATION, N_AZIMUTH_SG, N_SPEED,
        N_AZIMUTH_OA_TARGET_SPACING_DEG,
    )
    ax.set_title(
        f"N_el={N_ELEVATION}  N_az_sg={N_AZIMUTH_SG}  oa_dx={N_AZIMUTH_OA_TARGET_SPACING_DEG}°  N_sp={N_SPEED}\n"
        f"max|err|={np.abs(rel_errors).max():.1%}   "
        f"median={np.median(np.abs(rel_errors)):.1%}   "
        f"p95={np.percentile(np.abs(rel_errors), 95):.1%}"
    )

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=t_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Temperature (eV)")
    cbar.set_ticks([np.log10(t) for t in [1, 3, 10, 30, 100]])
    cbar.set_ticklabels(["1", "3", "10", "30", "100"])

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "scatter_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
