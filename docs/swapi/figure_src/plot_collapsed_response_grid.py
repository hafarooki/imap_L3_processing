#!/usr/bin/env python3
"""
Plot the angular-collapsed instrument response H(v', V): the dblquad shell-
integral reference (line) overlaid with the closed-form φ-inversion result
(scatter), plus a relative-error subpanel.

Fixed condition matches the CSV reference produced by
scripts/swapi/generate_collapsed_response_grid_reference.py:
ESA voltage = 5000 V, m/q = 4 m_p/e, bulk = 450 km/s @ (az=5°, el=−10°).

Output: docs/swapi/figures/collapsed_response_grid.svg
Usage:  uv run python docs/swapi/figure_src/plot_collapsed_response_grid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    build_collapsed_response_grid,
    solar_wind_frame_speed_range,
)
from figure_utils import FIGURES_DIR, REPO_ROOT, load_swapi_response

_REFERENCE_CSV_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "collapsed_response_grid_reference.csv"
)

_ESA_VOLTAGE = 5000.0
_MASS_PER_CHARGE = 4.0
_BULK_SPEED = 450.0
_BULK_AZIMUTH = 5.0
_BULK_ELEVATION = -10.0

_N_FAST_INTEGRAL_POINTS = 64


def main():
    print(f"Loading reference CSV from {_REFERENCE_CSV_PATH}...")
    reference = pd.read_csv(_REFERENCE_CSV_PATH, comment="#")
    v_prime_ref = reference["v_prime_kms"].to_numpy()
    h_ref = reference["h_truth_km3_per_s"].to_numpy()

    print("Loading SWAPI response and warming cache...")
    swapi_response = load_swapi_response()
    swapi_response.warm_cache(np.array([_ESA_VOLTAGE]))
    response_grid = swapi_response.get_response_grid(
        esa_voltage=_ESA_VOLTAGE,
        mass_per_charge_m_p_per_e=_MASS_PER_CHARGE,
    )

    print("Computing production collapsed response grid...")
    v_prime_min, v_prime_max = solar_wind_frame_speed_range(
        response_grid.central_speed, float(_BULK_SPEED)
    )
    speed_in_sw_frame = np.linspace(v_prime_min, v_prime_max, _N_FAST_INTEGRAL_POINTS)
    production = build_collapsed_response_grid(
        response_grid,
        bulk_speed=_BULK_SPEED,
        bulk_azimuth=_BULK_AZIMUTH,
        bulk_elevation=_BULK_ELEVATION,
        speed_in_sw_frame=speed_in_sw_frame,
    )

    h_ref_on_prod = np.interp(production.speed_in_sw_frame, v_prime_ref, h_ref)
    rel_err = np.abs(production.values - h_ref_on_prod) / h_ref_on_prod

    KM3_PER_S_TO_CM3_PER_S = 1e15
    h_ref_cm3 = h_ref * KM3_PER_S_TO_CM3_PER_S
    production_values_cm3 = production.values * KM3_PER_S_TO_CM3_PER_S

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(
        v_prime_ref,
        h_ref_cm3,
        color="black",
        lw=1.5,
        label="Reference",
        zorder=2,
    )
    axes[0].scatter(
        production.speed_in_sw_frame,
        production_values_cm3,
        s=8,
        color="red",
        label="Optimized",
        zorder=3,
    )
    axes[0].set_ylabel(r"$H(v', V)\ [\mathrm{cm}^3/\mathrm{s}]$")
    axes[0].set_title(
        f"Angular-collapsed response at V={_ESA_VOLTAGE:.0f} V, m/q={_MASS_PER_CHARGE:.0f}, "
        f"bulk={_BULK_SPEED:.0f} km/s @ (az={_BULK_AZIMUTH:.0f}°, el={_BULK_ELEVATION:.0f}°)",
        fontsize=10,
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(
        production.speed_in_sw_frame,
        rel_err,
        s=8,
        color="red",
        zorder=3,
    )
    axes[1].set_xlabel(r"$v'\ [\mathrm{km}/\mathrm{s}]$")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_yscale("log")
    axes[1].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    axes[1].set_ylim(1e-4, 1.5e0)
    axes[1].grid(True, alpha=0.2)

    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "collapsed_response_grid.svg"
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
