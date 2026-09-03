#!/usr/bin/env python3
"""
Plot the xarray + pint PUI forward-model reference: total coincidence count-
rate spectrum across the ESA voltage table at the parameters baked into
`scripts/swapi/pui_xarray_reference.py`.

Loads the precomputed reference from
`tests/test_data/swapi/pui_count_rate_reference.csv` (regenerate by re-running
the xarray reference script). Overlays the in-development production kernel
`calculate_coincidence_rate` for visual TDD; if that kernel raises, the figure
still saves with the reference curve alone.

Output: docs/swapi/figures/pui_xarray_reference.svg
Usage:  uv run python docs/swapi/figure_src/plot_pui_xarray_reference.py
"""

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import FIGURES_DIR

_REFERENCE_CSV_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "pui_count_rate_reference.csv"
)

# Must match scripts/swapi/pui_xarray_reference.py so the production kernel is
# evaluated at the same point as the precomputed reference CSV.
_SW_SPEED_KMS = 450.0
_SW_AZIMUTH_DEG = 0.0
_SW_ELEVATION_DEG = -10.0
_COOLING_INDEX = 2.0
_CUTOFF_SPEED_KMS = 450.0
_IONIZATION_RATE_HZ = 2e-7
_BACKGROUND_RATE_HZ = 0.0
_HELIO_DIST_AU = 1.0
_SW_SPEED_INERTIAL_KMS = 450.0
_INFLOW_PSI_DEG = 75.0
_HELIUM_MASS_PER_CHARGE_M_P_PER_E = 4.0
_HELIUM_EFFICIENCY_RATIO = 1.05

_DENSITY_LUT_PATH = (
    REPO_ROOT / "instrument_team_data" / "swapi" / "density-of-neutral-helium-lut.dat"
)


def compute_production_rates(voltages_v: np.ndarray) -> np.ndarray:
    """Build the chunk-level collapsed response and call the in-development
    production kernel. Imports live inside the function so a broken module
    doesn't break the figure script at import time."""
    from imap_l3_processing.constants import ONE_AU_IN_KM
    from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
        calculate_coincidence_rate,
    )
    from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
        build_chunk_collapsed_response,
    )
    from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
        DensityOfNeutralHeliumLookupTable,
    )
    from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
        FittingParameters,
        VasyliunasSiscoeDistribution,
    )
    from figure_utils import load_swapi_response

    fitting_params = FittingParameters(
        cooling_index=_COOLING_INDEX,
        ionization_rate=_IONIZATION_RATE_HZ,
        cutoff_speed=_CUTOFF_SPEED_KMS,
        background_count_rate=_BACKGROUND_RATE_HZ,
    )
    lut = DensityOfNeutralHeliumLookupTable.from_file(_DENSITY_LUT_PATH)
    min_speed_kms = max(
        1.0, _CUTOFF_SPEED_KMS * lut.get_minimum_distance() / _HELIO_DIST_AU
    )
    vasyliunas_siscoe_distribution = VasyliunasSiscoeDistribution(
        ephemeris_time=0.0,
        solar_wind_speed_inertial_frame=_SW_SPEED_INERTIAL_KMS,
        density_of_neutral_helium_lookup_table=lut,
        distance_km=_HELIO_DIST_AU * ONE_AU_IN_KM,
        psi=_INFLOW_PSI_DEG,
    )

    sw_az_rad = np.radians(_SW_AZIMUTH_DEG)
    sw_el_rad = np.radians(_SW_ELEVATION_DEG)
    bulk_vec = _SW_SPEED_KMS * np.array([
        -np.cos(sw_el_rad) * np.sin(sw_az_rad),
        -np.cos(sw_el_rad) * np.cos(sw_az_rad),
        -np.sin(sw_el_rad),
    ])
    bulk_sw_per_bin = np.broadcast_to(
        bulk_vec, (1, voltages_v.size, 3)
    ).copy()

    swapi_response = load_swapi_response()
    swapi_response.warm_cache(voltages_v)
    chunk_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=np.asarray(voltages_v, dtype=float),
        bulk_sw_per_bin_kms=bulk_sw_per_bin,
        mass_per_charge_m_p_per_e=_HELIUM_MASS_PER_CHARGE_M_P_PER_E,
        cutoff_speed_max_kms=_CUTOFF_SPEED_KMS,
        min_speed_kms=min_speed_kms,
        central_effective_area_scale=_HELIUM_EFFICIENCY_RATIO,
    )
    return calculate_coincidence_rate(chunk_response, vasyliunas_siscoe_distribution, fitting_params)[0]


def main():
    reference = pd.read_csv(_REFERENCE_CSV_PATH)
    voltage_v = reference.iloc[:, 0].to_numpy()
    count_rate_hz = reference.iloc[:, 1].to_numpy()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    sw_az_rad = np.radians(_SW_AZIMUTH_DEG)
    sw_el_rad = np.radians(_SW_ELEVATION_DEG)
    bulk_vec = _SW_SPEED_KMS * np.array([
        -np.cos(sw_el_rad) * np.sin(sw_az_rad),
        -np.cos(sw_el_rad) * np.cos(sw_az_rad),
        -np.sin(sw_el_rad),
    ]) + 0.0  # normalize -0.0 to +0.0 for display
    axes[0].set_title(
        f"PUI He coincidence rate — m/q={_HELIUM_MASS_PER_CHARGE_M_P_PER_E:.0f}, "
        f"efficiency ratio={_HELIUM_EFFICIENCY_RATIO:.2f}\n"
        f"bulk = {_SW_SPEED_KMS:.0f} km/s @ (az={_SW_AZIMUTH_DEG:.0f}°, el={_SW_ELEVATION_DEG:.0f}°) "
        f"= ({bulk_vec[0]:.0f}, {bulk_vec[1]:.0f}, {bulk_vec[2]:.0f}) km/s\n"
        f"cooling index={_COOLING_INDEX:.1f}, v_cutoff={_CUTOFF_SPEED_KMS:.0f} km/s, "
        f"ionization={_IONIZATION_RATE_HZ:.0e} Hz, ψ={_INFLOW_PSI_DEG:.0f}°, r={_HELIO_DIST_AU:.1f} AU",
        fontsize=9,
    )

    axes[0].loglog(voltage_v, count_rate_hz, "-", label="Reference", color="black")

    voltage_v = voltage_v[::4]
    count_rate_hz = count_rate_hz[::4]

    try:
        production_rate_hz = compute_production_rates(voltage_v)
    except Exception:
        traceback.print_exc()
        print("calculate_coincidence_rate failed — skipping production overlay.")
    else:
        axes[0].loglog(
            voltage_v,
            production_rate_hz,
            ".",
            color="red",
            label="Optimized",
            markersize=4
        )
        rel_err = np.abs(production_rate_hz - count_rate_hz) / count_rate_hz
        axes[1].semilogy(voltage_v, rel_err, ".", color="red")

    axes[0].set_ylabel("Coincidence Rate [Hz]")
    axes[0].set_ylim(1e-2, 1e2)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("ESA voltage [V]")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_yscale("log")
    axes[1].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    axes[1].set_ylim(1e-4, 1e0)
    axes[1].grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "pui_xarray_reference.svg"
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
