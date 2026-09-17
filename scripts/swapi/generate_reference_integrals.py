#!/usr/bin/env python3
"""
Generate reference ground-truth integrals for SWAPI proton solar wind.

Computes reference integrals using fixed limits covering the full passband at
high resolution. Solar wind configurations come from
`docs/swapi/figure_src/wind_solar_wind_samples_2025.csv` (10,000 WIND/SWE 2-min
samples from 2025) — the same dataset used by `plot_fit_accuracy.py` — so this
benchmark and the fit-accuracy benchmark cover identical SW conditions.

Each case is evaluated at the ESA voltage whose central proton speed matches
its bulk speed magnitude, using a single representative SWAPI→RTN rotation
matrix (figure_utils.ANCHOR_ROTATION_SWAPI_TO_RTN, the real SPICE attitude near
2026-01-01 used as the anchor for fit-accuracy plotting).

Fixed integration limits:
  elevation:  -15 to 15 deg at 0.05 deg (601 pts)
  azimuth SG: -20 to 20 deg at 0.05 deg (801 pts)
  azimuth OA: 0.05 deg in transition |az| ∈ [20, 30], 0.5 deg in bulk to ±150 (441 pts/side)
  speed: 200 samples from 0.9 to 1.1 × central_speed

Output: tests/test_data/swapi/reference_integrals.csv
Usage:  python scripts/swapi/generate_reference_integrals.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "docs" / "swapi" / "figure_src"))

import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species
from scripts.swapi.reference_integral import reference_integrals_batch
from figure_utils import ANCHOR_ROTATION_SWAPI_TO_RTN, NOMINAL_EPOCH_TT2000

_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_EFFICIENCY_TABLE_PATH = (
    _REPO_ROOT
    / "tests"
    / "test_data"
    / "swapi"
    / "imap_swapi_efficiency-lut-test_20241020_v001.dat"
)
_WIND_SAMPLES_PATH = (
    _REPO_ROOT / "docs" / "swapi" / "figure_src" / "wind_solar_wind_samples_2025.csv"
)
_OUTPUT_PATH = _REPO_ROOT / "tests" / "test_data" / "swapi" / "reference_integrals.csv"


def _peak_voltage(bulk_speed_km_s: float) -> float:
    return (
        PROTON_MASS_KG
        * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def main():
    print(f"Loading WIND samples from {_WIND_SAMPLES_PATH.relative_to(_REPO_ROOT)}...")
    wind = pd.read_csv(_WIND_SAMPLES_PATH)
    v_r = wind["v_R_km_s"].to_numpy(dtype=float)
    v_t = wind["v_T_km_s"].to_numpy(dtype=float)
    v_n = wind["v_N_km_s"].to_numpy(dtype=float)
    temperature_k = wind["proton_temperature_K"].to_numpy(dtype=float)
    density = wind["proton_density_cm3"].to_numpy(dtype=float)
    bulk_speed = np.sqrt(v_r**2 + v_t**2 + v_n**2)
    n_samples = len(wind)

    print("Loading calibration data...")
    swapi_response = SwapiResponse.from_files(
        azimuthal_transmission_path=_INSTRUMENT_DATA
        / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        central_effective_area_path=_INSTRUMENT_DATA
        / "imap_swapi_central-effective-area_20260425_v001.csv",
        passband_fit_coefficients_path=_INSTRUMENT_DATA
        / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
        efficiency_table_path=_EFFICIENCY_TABLE_PATH,
    )

    print(f"Building grids and SWParams for {n_samples} samples...")
    sws = [
        SolarWindParams(
            density=float(density[i]),
            velocity_rtn=np.array([v_r[i], v_t[i], v_n[i]]),
            temperature=float(temperature_k[i]),
            mass=PROTON_MASS_KG,
        )
        for i in range(n_samples)
    ]
    rotation_matrices = np.broadcast_to(ANCHOR_ROTATION_SWAPI_TO_RTN, (n_samples, 3, 3))
    voltages = [_peak_voltage(float(v)) for v in bulk_speed]
    swapi_response.warm_cache(voltages)
    response_grids = [
        swapi_response.get_response_grid(NOMINAL_EPOCH_TT2000, v, Species.PROTON)
        for v in voltages
    ]

    print(f"Computing {n_samples} reference integrals (JIT-parallel, fixed limits)...")
    integrals = reference_integrals_batch(response_grids, sws, rotation_matrices)

    df = pd.DataFrame(
        {
            "bulk_speed_km_s": np.round(bulk_speed, 2),
            "v_R_km_s": v_r,
            "v_T_km_s": v_t,
            "v_N_km_s": v_n,
            "temperature_K": temperature_k,
            "density_cm3": density,
            "integral_hz": np.round(integrals, 2),
        }
    )
    df.to_csv(_OUTPUT_PATH, index=False)
    print(f"Saved {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
