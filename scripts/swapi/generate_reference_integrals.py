#!/usr/bin/env python3
"""
Generate reference ground-truth integrals for SWAPI proton solar wind.

Computes 10000 reference integrals using fixed limits covering the full passband
at high resolution. These are used as ground truth in regression tests and the
scatter benchmark.

Fixed integration limits:
  elevation:  -15 to 15 deg at 0.1 deg (301 pts)
  azimuth SG: -20 to 20 deg at 0.1 deg (401 pts)
  azimuth OA: 0.1 deg in transition |az| ∈ [20, 30], 1 deg in bulk to ±150 (221 pts/side)
  speed: 200 samples from 0.9 to 1.1 × central_speed

Solar wind parameter ranges (10000 samples, seed=42):
  bulk_speed:      200–2000 km/s   (uniform)
  temperature:     1–100 eV        (log-uniform)
  bulk_azimuth:    -20 to 20 deg   (uniform)
  bulk_elevation:  -20 to 20 deg   (uniform)
  density:         1–100 cm⁻³      (uniform)

Output: tests/swapi/l3a/science/reference_integrals.csv
Usage:  python scripts/swapi/generate_reference_integrals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    SWParams,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from tests.swapi.l3a.science.reference_integral import reference_integrals_batch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_PATH = (
    _REPO_ROOT / "tests" / "swapi" / "l3a" / "science" / "reference_integrals.csv"
)

_N_SAMPLES = 10000
_RNG_SEED = 42

_SPEED_RANGE = (200.0, 2000.0)
_TEMP_RANGE = (1.0, 100.0)
_AZ_RANGE = (-20.0, 20.0)
_EL_RANGE = (-20.0, 20.0)
_DENSITY_RANGE = (1.0, 100.0)


def _peak_voltage(bulk_speed_km_s: float) -> float:
    return (
        PROTON_MASS_KG
        * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _thermal_speed(temperature_ev: float) -> float:
    return float(
        np.sqrt(temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )


def main():
    print("Loading calibration data...")
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    rng = np.random.default_rng(_RNG_SEED)
    bulk_speeds = rng.uniform(*_SPEED_RANGE, _N_SAMPLES)
    temperatures_ev = np.exp(
        rng.uniform(np.log(_TEMP_RANGE[0]), np.log(_TEMP_RANGE[1]), _N_SAMPLES)
    )
    bulk_azimuths = rng.uniform(*_AZ_RANGE, _N_SAMPLES)
    bulk_elevations = rng.uniform(*_EL_RANGE, _N_SAMPLES)
    densities = rng.uniform(*_DENSITY_RANGE, _N_SAMPLES)

    print(f"Building grids and SWParams for {_N_SAMPLES} samples...")
    sws = [
        SWParams(
            density=float(densities[i]),
            bulk_speed=float(bulk_speeds[i]),
            bulk_azimuth=float(bulk_azimuths[i]),
            bulk_elevation=float(bulk_elevations[i]),
            thermal_speed=_thermal_speed(float(temperatures_ev[i])),
        )
        for i in range(_N_SAMPLES)
    ]
    voltages = [_peak_voltage(float(v)) for v in bulk_speeds]
    swapi_response.warm_cache(voltages)
    grids = [swapi_response.create_passband_grid(v) for v in voltages]
    central_speeds = [swapi_response.central_speed(v, 1.0) for v in voltages]
    central_effective_areas = [
        swapi_response.get_central_effective_area(v) for v in voltages
    ]

    print(f"Computing {_N_SAMPLES} reference integrals (JIT-parallel, fixed limits)...")
    integrals = reference_integrals_batch(
        grids,
        sws,
        central_speeds,
        central_effective_areas,
        swapi_response.azimuthal_transmission,
        swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG,
    )

    df = pd.DataFrame(
        {
            "bulk_speed": bulk_speeds,
            "temperature_ev": temperatures_ev,
            "bulk_azimuth": bulk_azimuths,
            "bulk_elevation": bulk_elevations,
            "density": densities,
            "integral": integrals,
        }
    )
    df.to_csv(_OUTPUT_PATH, index=False)
    print(f"Saved {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
