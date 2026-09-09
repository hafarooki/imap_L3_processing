import unittest
from datetime import datetime

import numpy as np
import numpy.testing as npt
from spacepy import pycdf

from imap_l3_processing import constants
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path


def _load_response() -> SwapiResponse:
    return SwapiResponse.from_files(
        azimuthal_transmission_path=get_test_instrument_team_data_path(
            "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
        ),
        central_effective_area_path=get_test_instrument_team_data_path(
            "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
        ),
        passband_fit_coefficients_path=get_test_instrument_team_data_path(
            "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
        ),
        efficiency_table_path=get_test_data_path(
            "swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat"
        )
    )


class _RealResponseFixture(unittest.TestCase):
    """Base class loading a single shared `SwapiResponse` once per class."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()


class TestWarmCacheApi(unittest.TestCase):
    """Input-handling contract of `warm_cache`: dedup, NaN/inf skip, dimensionality,
    determinism. Each test mutates the response's `_grid_cache`, so a fresh
    instance is built per test."""

    def setUp(self):
        self.response = _load_response()

    def test_populates_cache_with_unique_finite_voltages(self):
        """Duplicates collapse, NaN/inf are skipped."""
        voltages = np.array([100.0, 100.0, 200.0, 300.0, np.nan, np.inf])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._passband_grid_cache.keys()), {100.0, 200.0, 300.0})
        self.assertEqual(len(self.response._passband_grid_cache), 3)

    def test_warming_twice_at_same_voltage_is_a_noop(self):
        """A second `warm_cache` at the same voltage keeps the existing
        PassbandGrid objects — it does not rebuild them."""
        self.response.warm_cache([750.0])
        first_grids = {
            region: self.response._passband_grid_cache[750.0][region] for region in ("SG", "OA")
        }
        self.response.warm_cache([750.0])

        for region in ("SG", "OA"):
            with self.subTest(region=region):
                self.assertIs(self.response._passband_grid_cache[750.0][region], first_grids[region])

    def test_accepts_2d_voltage_array(self):
        """Multidimensional voltage inputs are flattened."""
        voltages = np.array([[100.0, 200.0], [200.0, 300.0]])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._passband_grid_cache.keys()), {100.0, 200.0, 300.0})


class TestPassbandInterpolation(_RealResponseFixture):
    """Cover `_get_passband_values`: in-range it must evaluate
    `exp(polyval(coeffs_row, log(SWAPI_K_FACTOR * V)))` with coefficients read
    along the column axis (highest degree first); outside the per-region
    calibration window the voltage is clamped before evaluation, which keeps
    values physical (non-negative, near unity) even when the raw polynomial
    extrapolation would diverge."""

    # Upper bound > 1.0 because the polynomial is fit to *un-normalized*
    # response data; normalization happens elsewhere. The fit can produce
    # values slightly above 1.0 at evaluation points between the fit nodes.
    PASSBAND_VALUE_UPPER_BOUND = 1.5

    def test_values_match_manual_polyval(self):
        for region in ("OA", "SG"):
            with self.subTest(region=region):
                v_min, v_max = self.response._passband_esa_voltage_limits[region]
                voltage = 0.5 * (v_min + v_max)

                returned = self.response._get_passband_values(voltage, region)

                coeffs = self.response._passband_fit_coefficients.xs(
                    region, level="region"
                )
                log_beam_energy = np.log(SWAPI_K_FACTOR * voltage)
                n_degrees = coeffs.shape[1]
                degrees = np.arange(n_degrees - 1, -1, -1)
                expected_exponent = (coeffs.values * log_beam_energy ** degrees).sum(
                    axis=1
                )
                expected = np.exp(expected_exponent)

                npt.assert_allclose(
                    returned["value"].to_numpy(), expected, rtol=1e-12
                )

    def _assert_grid_value_bounds(self, esa_voltage):
        self.response.warm_cache([esa_voltage])
        cached = self.response._passband_grid_cache[self.response._cache_key(esa_voltage)]
        for region in ("SG", "OA"):
            grid = cached[region]
            self.assertGreaterEqual(
                grid.values.min(),
                0.0,
                msg=f"{region} passband has negative values at {esa_voltage} V",
            )
            self.assertLessEqual(
                grid.values.max(),
                self.PASSBAND_VALUE_UPPER_BOUND,
                msg=f"{region} passband exceeds bound at {esa_voltage} V",
            )

    def test_passband_grid_bounds_at_low_voltage(self):
        self._assert_grid_value_bounds(50.0)

    def test_passband_grid_bounds_at_high_voltage(self):
        self._assert_grid_value_bounds(20000.0)


class TestGetResponseGrid(unittest.TestCase):
    """Tests for `SwapiResponse.get_response_grid`"""

    def test_get_response_grid(self):
        """Test values of response grids of various species"""
        time_as_tt2000 = pycdf.lib.datetime_to_tt2000(datetime(2026, 1, 1, 0, 0, 0))
        esa_voltage = -552.1339894
        response = _load_response()
        response.warm_cache(esa_voltage)

        response_grid = response.get_response_grid(time_as_tt2000, esa_voltage, Species.PROTON)
        self.assertAlmostEqual(response_grid.central_effective_area, 0.37553722631276887)
        self.assertIs(response_grid.azimuthal_transmission, response._azimuthal_transmission_grid)
        self.assertIs(response_grid.oa_passband, response._passband_grid_cache[round(esa_voltage, 3)]["OA"])
        self.assertIs(response_grid.sg_passband, response._passband_grid_cache[round(esa_voltage, 3)]["SG"])
        expected_central_speed_km_per_s = np.sqrt(
            2 * 1.89 * 552.1339894 * constants.PROTON_CHARGE_COULOMBS / constants.PROTON_MASS_KG
        ) / constants.METERS_PER_KILOMETER
        self.assertAlmostEqual(response_grid.central_speed, expected_central_speed_km_per_s)

        response_grid2 = response.get_response_grid(time_as_tt2000, esa_voltage, Species.PROTON)
        self.assertIs(response_grid, response_grid2, "cache miss")

        for species, mass_per_charge in [
            (Species.ALPHA, constants.ALPHA_MASS_PER_CHARGE_M_P_PER_E),
            (Species.HELIUM_PLUS, constants.HE_PUI_PARTICLE_MASS_PER_CHARGE_M_P_PER_E),
        ]:
            response_grid_species = response.get_response_grid(time_as_tt2000, esa_voltage, species)
            self.assertAlmostEqual(response_grid_species.central_effective_area / response_grid.central_effective_area, 1.05)
            self.assertAlmostEqual(
                response_grid_species.central_speed / response_grid.central_speed,
                np.sqrt(constants.PROTON_MASS_PER_CHARGE_M_P_PER_E / mass_per_charge),
                msg="central speed should be lower by the sqrt mass per charge ratio"
            )
            for attr in ["sg_passband", "oa_passband", "azimuthal_transmission"]:
                self.assertIs(getattr(response_grid, attr), getattr(response_grid_species, attr), msg="shared attribute mismatch")


if __name__ == "__main__":
    unittest.main()
