"""Tests for `calculate_combined_solar_wind_differential_flux`.

Source formula:
    flux = count / (E · G_cm² · ε)

with `G_cm² = G_km² · (METERS_PER_KILOMETER · CENTIMETERS_PER_METER)²`. The
test asserts on values computed inline from this formula against a real
`GeometricFactorCalibrationTable` rather than mocking the lookup and pinning
opaque hard-coded fluxes.
"""

import unittest

import numpy as np

from imap_l3_processing.constants import CENTIMETERS_PER_METER, METERS_PER_KILOMETER
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import (
    calculate_combined_solar_wind_differential_flux,
)
from tests.swapi._swapi_test_helpers import geometric_factor_pui_table


class TestCalculateCombinedDifferentialFlux(unittest.TestCase):
    def setUp(self):
        self.energies = np.array([1000.0, 750.0, 500.0])
        self.count_rates = np.array([10.0, 20.0, 30.0])
        self.efficiency = 0.0882
        self.gf_table = geometric_factor_pui_table()

    def test_flux_matches_closed_form(self):
        flux = calculate_combined_solar_wind_differential_flux(
            self.energies, self.count_rates, self.efficiency, self.gf_table
        )
        gf_km2 = self.gf_table.lookup_geometric_factor(self.energies)
        gf_cm2 = gf_km2 * (METERS_PER_KILOMETER * CENTIMETERS_PER_METER) ** 2
        expected = self.count_rates / (self.energies * gf_cm2 * self.efficiency)
        np.testing.assert_allclose(flux, expected, rtol=1e-12)

    def test_flux_scales_linearly_with_count_rate(self):
        flux_low = calculate_combined_solar_wind_differential_flux(
            self.energies, self.count_rates, self.efficiency, self.gf_table
        )
        flux_high = calculate_combined_solar_wind_differential_flux(
            self.energies, 5.0 * self.count_rates, self.efficiency, self.gf_table
        )
        np.testing.assert_allclose(flux_high, 5.0 * flux_low)

    def test_flux_scales_inversely_with_efficiency(self):
        flux_a = calculate_combined_solar_wind_differential_flux(
            self.energies, self.count_rates, 0.05, self.gf_table
        )
        flux_b = calculate_combined_solar_wind_differential_flux(
            self.energies, self.count_rates, 0.10, self.gf_table
        )
        np.testing.assert_allclose(flux_a, 2.0 * flux_b)

    def test_zero_count_rate_yields_zero_flux(self):
        flux = calculate_combined_solar_wind_differential_flux(
            self.energies,
            np.zeros_like(self.energies),
            self.efficiency,
            self.gf_table,
        )
        np.testing.assert_array_equal(flux, np.zeros_like(self.energies))


if __name__ == "__main__":
    unittest.main()
