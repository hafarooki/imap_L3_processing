from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from spacepy import pycdf

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    solar_wind_frame_speed_range,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid, SwapiResponse
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


# Calibration CSVs shipped with the repo. Loading the full SwapiResponse from
# these triggers the same code path the production pipeline uses.
SWAPI_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
SWAPI_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
SWAPI_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)
SWAPI_EFFICIENCY_TABLE_PATH = get_test_data_path(
    "swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat"
)


def load_swapi_response(
    warm_cache_voltages: Optional[np.ndarray] = None,
    efficiency_table_path: Optional[Union[str, Path]] = None,
) -> SwapiResponse:
    """Build a `SwapiResponse` from the CSV files..

    `efficiency_table_path` overrides the shipped test efficiency table, for
    callers that need the per-species relative efficiencies to match a
    synthetic fixture; `get_response_grid` applies them to the effective area.
    """
    response = SwapiResponse.from_files(
        azimuthal_transmission_path=SWAPI_AZIMUTHAL_TRANSMISSION_PATH,
        central_effective_area_path=SWAPI_CENTRAL_EFFECTIVE_AREA_PATH,
        passband_fit_coefficients_path=SWAPI_PASSBAND_FIT_COEFFICIENTS_PATH,
        efficiency_table_path=efficiency_table_path
        if efficiency_table_path is not None
        else SWAPI_EFFICIENCY_TABLE_PATH,
    )
    if warm_cache_voltages is not None:
        response.warm_cache(warm_cache_voltages)
    return response


def build_default_v_prime_grid_kms(
    response_grid: ResponseGrid, bulk_speed_kms: float, n_points: int = 128
) -> np.ndarray:
    """Per-bin v' grid for verification scripts: `linspace(v'_min, v'_max, n_points)`
    over the active range of one (response_grid, bulk_speed). Production code
    uses a chunk-wide shared grid via `build_chunk_collapsed_response`."""
    v_prime_min, v_prime_max = solar_wind_frame_speed_range(
        response_grid.central_speed, float(bulk_speed_kms)
    )
    return np.linspace(v_prime_min, v_prime_max, n_points)


# Nominal IMAP_SWAPI → IMAP_RTN rotation at spin phase 0: spin axis (+Y_SWAPI,
# the boresight) along -R_RTN (SWAPI looks sunward), X_SWAPI along +T_RTN,
# Z_SWAPI along +N_RTN. Columns are the SWAPI basis vectors expressed in RTN,
# so `R @ v_swapi = v_rtn` and `R.T @ v_rtn = v_swapi`. The default bulk
# velocity (+450, 0, 0) km/s arrives along -Y_SWAPI under this rotation.
# 71-bin realistic SWAPI science sweep (62 coarse bins log-spaced 10000→50 V
# plus 9 fine bins log-spaced 1000→500 V), matching the production sweep
# shape used by `SWAPI_COARSE_SWEEP_BINS` and `SWAPI_FINE_SWEEP_BINS`.
# Caution: The fine sweeps in real data typically start near the peak,
# this will only start near the peak if the peak is near 1 keV
REALISTIC_ESA_VOLTAGES = np.concatenate(
    [
        np.logspace(np.log10(10000.0), np.log10(50.0), 62),
        np.logspace(np.log10(1000.0), np.log10(500.0), 9),
    ]
)


NOMINAL_SWAPI_TO_RTN_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def proton_params(
    *,
    density: float = 5.0,
    velocity_rtn=(450.0, 0, 0.0),
    temperature: float = 100_000.0,
) -> SolarWindParams:
    """Solar-wind proton `SolarWindParams` at typical slow-wind values."""
    return SolarWindParams(
        density=density,
        velocity_rtn=np.asarray(velocity_rtn, dtype=float),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def synthesize_count_rates(ctx, sw_params: SolarWindParams) -> np.ndarray:
    """Forward-model deadtime-applied coincidence count rates from `sw_params`
    using the same model the fitter inverts. No Poisson noise."""
    ideal, _ = model_solar_wind_ideal_coincidence_rates(sw_params, ctx)
    return ideal * deadtime_factor(ideal)

NOMINAL_TEST_EPOCH_TT2000 = pycdf.lib.datetime_to_tt2000(datetime(2026, 1, 1))
