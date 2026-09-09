"""Compare `calculate_coincidence_rate` against the independent PUI reference CSV.

The reference (`scripts/swapi/pui_xarray_reference.py`) integrates the same PUI
forward model in solar-wind-frame velocity space, with the radial grid ending at
the exact cutoff speed so the filled-shell edge is the integration boundary
rather than a step inside the integrand. At the parameters baked into that
script, the production kernel's per-ESA-voltage coincidence rate should agree
with the precomputed reference. Agreement is sub-percent across the peak and
rising edge; it loosens to ~3% only on the deep high-voltage falloff, where
production's instrument-coordinate response collapse under-counts the
bulk-aligned geometry near its 1/sin(delta_azimuth) Jacobian singularity, and at
very low rates (covered by `atol`).
"""
import unittest

import numpy as np
import pandas as pd

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
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000, load_swapi_response
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

_REFERENCE_CSV_PATH = get_test_data_path("swapi/pui_count_rate_reference.csv")
_DENSITY_LUT_PATH = get_test_instrument_team_data_path(
    "swapi/density-of-neutral-helium-lut.dat"
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


class CalculateCoincidenceRateAgainstReferenceTest(unittest.TestCase):
    def test_matches_xarray_reference_per_voltage(self):
        reference = pd.read_csv(_REFERENCE_CSV_PATH)
        voltage_v = reference.iloc[:, 0].to_numpy()
        reference_rate_hz = reference.iloc[:, 1].to_numpy()

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

        # Single (sweep, step) bulk-SW vector in IMAP_SWAPI. SWAPI v̂ convention:
        # (−cos θ sin φ, −cos θ cos φ, −sin θ).
        sw_az_rad = np.radians(_SW_AZIMUTH_DEG)
        sw_el_rad = np.radians(_SW_ELEVATION_DEG)
        bulk_vec = _SW_SPEED_KMS * np.array([
            -np.cos(sw_el_rad) * np.sin(sw_az_rad),
            -np.cos(sw_el_rad) * np.cos(sw_az_rad),
            -np.sin(sw_el_rad),
        ])
        bulk_sw_per_bin = np.broadcast_to(
            bulk_vec, (1, voltage_v.size, 3)
        ).copy()

        swapi_response = load_swapi_response(
            warm_cache_voltages=voltage_v.astype(float)
        )
        chunk_response = build_chunk_collapsed_response(
            swapi_response=swapi_response,
            voltages_v=voltage_v.astype(float),
            bulk_sw_per_bin_kms=bulk_sw_per_bin,
            time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
            species=Species.HELIUM_PLUS,
            cutoff_speed_max_kms=_CUTOFF_SPEED_KMS,
        )

        production_rate_hz = calculate_coincidence_rate(
            chunk_response, vasyliunas_siscoe_distribution, fitting_params
        )[0]

        # rtol covers the deep-falloff collapse difference (production reads a
        # few percent low there); atol covers the low-rate tail where the two
        # quadratures' relative spread is large but absolute rates are tiny.
        np.testing.assert_allclose(
            production_rate_hz, reference_rate_hz, atol=0.1, rtol=0.03
        )


if __name__ == "__main__":
    unittest.main()
