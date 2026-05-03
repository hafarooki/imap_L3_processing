"""Shared fixtures and assertion helpers for SWAPI tests.

Caches expensive lookup-table loads at module level so test suites that touch
multiple files only pay the parse cost once. Anything that loads a real
`SWAPIResponse`, instrument-response collection, neutral-helium LUT, or
efficiency table should funnel through here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import UFloat

from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import (
    GeometricFactorCalibrationTable,
)
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import (
    InstrumentResponseLookupTableCollection,
)
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


_cache: dict[str, object] = {}


def swapi_response() -> SWAPIResponse:
    if "swapi_response" not in _cache:
        _cache["swapi_response"] = SWAPIResponse.from_files(
            _AZIMUTHAL_TRANSMISSION_PATH,
            _CENTRAL_EFFECTIVE_AREA_PATH,
            _PASSBAND_FIT_COEFFICIENTS_PATH,
        )
    return _cache["swapi_response"]


def neutral_helium_lookup_table() -> DensityOfNeutralHeliumLookupTable:
    if "neutral_he_lut" not in _cache:
        _cache["neutral_he_lut"] = DensityOfNeutralHeliumLookupTable.from_file(
            get_test_data_path(
                "swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
            )
        )
    return _cache["neutral_he_lut"]


def efficiency_calibration_table() -> EfficiencyCalibrationTable:
    if "efficiency_table" not in _cache:
        _cache["efficiency_table"] = EfficiencyCalibrationTable(
            get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")
        )
    return _cache["efficiency_table"]


def geometric_factor_pui_table() -> GeometricFactorCalibrationTable:
    if "gf_pui_table" not in _cache:
        _cache["gf_pui_table"] = GeometricFactorCalibrationTable.from_file(
            get_test_data_path("swapi/imap_swapi_energy-gf-pui-lut_20100101_v001.csv")
        )
    return _cache["gf_pui_table"]


def geometric_factor_sw_table() -> GeometricFactorCalibrationTable:
    if "gf_sw_table" not in _cache:
        _cache["gf_sw_table"] = GeometricFactorCalibrationTable.from_file(
            get_test_data_path("swapi/imap_swapi_energy-gf-sw-lut_20100101_v001.csv")
        )
    return _cache["gf_sw_table"]


def instrument_response_collection() -> InstrumentResponseLookupTableCollection:
    if "instrument_response" not in _cache:
        _cache["instrument_response"] = (
            InstrumentResponseLookupTableCollection.from_file(
                get_test_data_path(
                    "swapi/imap_swapi_instrument-response-lut_20241023_v000.zip"
                )
            )
        )
    return _cache["instrument_response"]


def hydrogen_inflow_vector() -> InflowVector:
    return InflowVector.from_file(
        get_test_data_path("swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat")
    )


def helium_inflow_vector() -> InflowVector:
    return InflowVector.from_file(
        get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat")
    )


def assert_ufloat_close(
    actual: UFloat,
    expected_nominal: float,
    expected_sigma: float,
    rtol: float = 1e-7,
    atol: float = 0.0,
    msg: str = "",
) -> None:
    """Assert a `ufloat` matches expected nominal and sigma within tolerance."""
    np.testing.assert_allclose(
        actual.nominal_value,
        expected_nominal,
        rtol=rtol,
        atol=atol,
        err_msg=f"{msg} (nominal)",
    )
    np.testing.assert_allclose(
        actual.std_dev,
        expected_sigma,
        rtol=rtol,
        atol=atol,
        err_msg=f"{msg} (sigma)",
    )


def make_l2_chunk(
    n_sweeps: int = 5,
    n_bins: int = 72,
    epoch_start_tt2000_ns: int = 0,
    epoch_step_ns: int = 12 * 1_000_000_000,
    energies: ArrayLike | None = None,
    coincidence_count_rate: ArrayLike | None = None,
) -> SwapiL2Data:
    """Build a minimal `SwapiL2Data` of `n_sweeps × n_bins` for unit tests.

    Caller may pass `energies` and `coincidence_count_rate` of shape
    `(n_sweeps, n_bins)` to override the defaults. Defaults are a
    geometrically-decreasing energy table (as on real SWAPI sweeps) and a flat
    count-rate of 100 Hz.
    """
    if energies is None:
        # ~20 keV → ~50 eV log-spaced over 72 bins, identical for every sweep.
        per_bin = np.logspace(np.log10(20_000.0), np.log10(50.0), n_bins)
        energies = np.tile(per_bin, (n_sweeps, 1))
    energies = np.asarray(energies, dtype=float)
    if coincidence_count_rate is None:
        coincidence_count_rate = np.full((n_sweeps, n_bins), 100.0)
    coincidence_count_rate = np.asarray(coincidence_count_rate, dtype=float)
    sci_start_time = epoch_start_tt2000_ns + np.arange(n_sweeps) * epoch_step_ns
    return SwapiL2Data(
        sci_start_time=sci_start_time,
        energy=energies,
        coincidence_count_rate=coincidence_count_rate,
        coincidence_count_rate_uncertainty=np.sqrt(
            np.where(coincidence_count_rate > 0, coincidence_count_rate, 1.0)
        ),
    )
