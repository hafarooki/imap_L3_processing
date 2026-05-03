"""Tests for the L3b VDF calculations.

Each `calculate_*_vdf` function maps `(energies, count_rates)` to
`(velocities, probabilities)` for a given particle species via:

    v(km/s)    = sqrt(2 q E / m) / 1000
    P(s³/cm⁶) = 4π · (m·1000 / q) · count / (E · G_cm² · ε)

where `G_cm² = G_km² · 1e10`. The earlier version of these tests mocked the
geometric factor table and asserted on hard-coded probabilities. This rewrite
computes expected values inline from the formulas above so a unit-conversion
or constant change will surface a clear, derivation-traceable failure.
"""

import unittest

import numpy as np

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    CENTIMETERS_PER_METER,
    HE_PUI_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PUI_PARTICLE_CHARGE_COULOMBS,
)
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import (
    DeltaMinusPlus,
    calculate_alpha_solar_wind_vdf,
    calculate_delta_minus_plus,
    calculate_proton_solar_wind_vdf,
    calculate_pui_solar_wind_vdf,
    calculate_vdf,
)
from tests.swapi._swapi_test_helpers import geometric_factor_pui_table


def _expected_vdf(
    mass_kg, charge_c, energies_J, count_rates, efficiency, geometric_factor_km2
):
    """Closed-form V/P expected from the documented formulas."""
    velocities_km_s = (
        np.sqrt(2.0 * charge_c * energies_J / mass_kg) / METERS_PER_KILOMETER
    )
    geometric_factor_cm2 = (
        geometric_factor_km2 * (METERS_PER_KILOMETER * CENTIMETERS_PER_METER) ** 2
    )
    probabilities = (
        4.0
        * np.pi
        * (mass_kg * 1000.0 / charge_c)
        * count_rates
        / (energies_J * geometric_factor_cm2 * efficiency)
    )
    return velocities_km_s, probabilities


class TestCalculateProtonSolarWindVdf(unittest.TestCase):
    def setUp(self):
        self.energies = np.array([1000.0, 750.0, 500.0])
        self.count_rates = np.array([10.0, 20.0, 30.0])
        self.efficiency = 0.0882
        self.gf_table = geometric_factor_pui_table()

    def test_velocity_and_probability_match_closed_form(self):
        velocities, probabilities = calculate_proton_solar_wind_vdf(
            self.energies, self.count_rates, self.efficiency, self.gf_table
        )
        gf_km2 = self.gf_table.lookup_geometric_factor(self.energies)
        expected_v, expected_p = _expected_vdf(
            PROTON_MASS_KG,
            PROTON_CHARGE_COULOMBS,
            self.energies,
            self.count_rates,
            self.efficiency,
            gf_km2,
        )
        np.testing.assert_allclose(velocities, expected_v, rtol=1e-12)
        np.testing.assert_allclose(probabilities, expected_p, rtol=1e-12)

    def test_zero_count_rate_yields_zero_probability(self):
        zero_counts = np.zeros_like(self.energies)
        _, probabilities = calculate_proton_solar_wind_vdf(
            self.energies, zero_counts, self.efficiency, self.gf_table
        )
        np.testing.assert_array_equal(probabilities, np.zeros_like(self.energies))

    def test_velocity_independent_of_count_rate(self):
        v_low, _ = calculate_proton_solar_wind_vdf(
            self.energies, self.count_rates, self.efficiency, self.gf_table
        )
        v_high, _ = calculate_proton_solar_wind_vdf(
            self.energies, 100.0 * self.count_rates, self.efficiency, self.gf_table
        )
        np.testing.assert_array_equal(v_low, v_high)

    def test_probability_scales_inversely_with_efficiency(self):
        _, p_a = calculate_proton_solar_wind_vdf(
            self.energies, self.count_rates, 0.05, self.gf_table
        )
        _, p_b = calculate_proton_solar_wind_vdf(
            self.energies, self.count_rates, 0.10, self.gf_table
        )
        np.testing.assert_allclose(p_a, 2.0 * p_b)


class TestCalculateAlphaSolarWindVdf(unittest.TestCase):
    def test_alpha_velocity_uses_alpha_mass_and_charge(self):
        energies = np.array([1000.0, 500.0])
        gf_table = geometric_factor_pui_table()
        velocities, _ = calculate_alpha_solar_wind_vdf(
            energies, np.array([10.0, 30.0]), 0.0882, gf_table
        )
        expected_v = (
            np.sqrt(
                2.0 * ALPHA_PARTICLE_CHARGE_COULOMBS * energies / ALPHA_PARTICLE_MASS_KG
            )
            / METERS_PER_KILOMETER
        )
        np.testing.assert_allclose(velocities, expected_v, rtol=1e-12)


class TestCalculatePuiSolarWindVdf(unittest.TestCase):
    def test_pui_velocity_uses_he_pui_mass_and_pui_charge(self):
        energies = np.array([1000.0, 500.0])
        gf_table = geometric_factor_pui_table()
        velocities, _ = calculate_pui_solar_wind_vdf(
            energies, np.array([10.0, 30.0]), 0.0882, gf_table
        )
        expected_v = (
            np.sqrt(
                2.0 * PUI_PARTICLE_CHARGE_COULOMBS * energies / HE_PUI_PARTICLE_MASS_KG
            )
            / METERS_PER_KILOMETER
        )
        np.testing.assert_allclose(velocities, expected_v, rtol=1e-12)


class TestCalculateVdfGenericDispatch(unittest.TestCase):
    """`calculate_vdf` is the underlying implementation for all three species
    wrappers. Exercise it directly with a non-standard mass/charge pair."""

    def test_arbitrary_mass_charge_matches_formula(self):
        energies = np.array([200.0, 400.0])
        counts = np.array([5.0, 7.0])
        eff = 0.5
        # Use 3 m_p / 2 e (a hypothetical species).
        mass = 3.0 * PROTON_MASS_KG
        charge = 2.0 * PROTON_CHARGE_COULOMBS
        gf_table = geometric_factor_pui_table()
        velocities, probabilities = calculate_vdf(
            mass, charge, energies, counts, eff, gf_table
        )
        gf_km2 = gf_table.lookup_geometric_factor(energies)
        exp_v, exp_p = _expected_vdf(mass, charge, energies, counts, eff, gf_km2)
        np.testing.assert_allclose(velocities, exp_v, rtol=1e-12)
        np.testing.assert_allclose(probabilities, exp_p, rtol=1e-12)


class TestCalculateDeltaMinusPlus(unittest.TestCase):
    def test_geometric_progression_ascending(self):
        result = calculate_delta_minus_plus(np.array([8.0, 32.0, 128.0]))
        # Geometric ratio 4 → half-ratio 2 → delta_minus = E - E/2 = E/2,
        # delta_plus = E*2 - E = E.
        np.testing.assert_array_equal(result.delta_minus, [4.0, 16.0, 64.0])
        np.testing.assert_array_equal(result.delta_plus, [8.0, 32.0, 128.0])

    def test_geometric_progression_descending(self):
        result = calculate_delta_minus_plus(np.array([128.0, 32.0, 8.0]))
        np.testing.assert_array_equal(result.delta_minus, [64.0, 16.0, 4.0])
        np.testing.assert_array_equal(result.delta_plus, [128.0, 32.0, 8.0])

    def test_uneven_ratios_take_min_max_of_left_right_edges(self):
        result = calculate_delta_minus_plus(np.array([8.0, 32.0, 288.0]))
        np.testing.assert_array_equal(result.delta_minus, [4.0, 16.0, 192.0])
        np.testing.assert_array_equal(result.delta_plus, [8.0, 64.0, 576.0])

    def test_returns_delta_minus_plus_dataclass(self):
        result = calculate_delta_minus_plus(np.array([8.0, 32.0]))
        self.assertIsInstance(result, DeltaMinusPlus)
        self.assertTrue(hasattr(result, "delta_minus"))
        self.assertTrue(hasattr(result, "delta_plus"))


if __name__ == "__main__":
    unittest.main()
