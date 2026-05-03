"""Direct unit tests for `imap_l3_processing.swapi.l3a.science.speed_calculation`.

Every public function and constant in `speed_calculation.py` is exercised here
against an analytical reference. The module is otherwise tested only
transitively through the moments fitter and L3b VDF code, which masks any
regression that happens to leave end-to-end products numerically close.
"""

import unittest

import numpy as np
from uncertainties import UFloat, ufloat
from uncertainties.unumpy import nominal_values

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_DISCARDED_BIN,
    SWAPI_FINE_SWEEP_BINS,
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
    calculate_combined_sweeps,
    calculate_sw_speed,
    calculate_sw_speed_h_plus,
    esa_voltage_to_alpha_speed,
    esa_voltage_to_proton_speed,
    extract_coarse_sweep,
    find_peak_center_of_mass_index,
    get_alpha_peak_indices,
    interpolate_energy,
    times_for_sweep,
)


def _analytic_proton_speed_km_per_s(voltage: float) -> float:
    return float(
        np.sqrt(
            2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS * abs(voltage) / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def _analytic_alpha_speed_km_per_s(voltage: float) -> float:
    return float(
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * ALPHA_PARTICLE_CHARGE_COULOMBS
            * abs(voltage)
            / ALPHA_PARTICLE_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


class TestBinSliceConstants(unittest.TestCase):
    def test_discarded_bin_is_zero(self):
        self.assertEqual(SWAPI_DISCARDED_BIN, 0)

    def test_science_bins_excludes_discarded_bin(self):
        self.assertEqual(SWAPI_SCIENCE_BINS, slice(1, 72))

    def test_coarse_and_fine_partition_science_bins(self):
        # Coarse and fine bin slices must tile the science range with no overlap and no gap.
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.start, SWAPI_SCIENCE_BINS.start)
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.stop, SWAPI_FINE_SWEEP_BINS.start)
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop, SWAPI_SCIENCE_BINS.stop)

    def test_coarse_sweep_has_62_bins_and_fine_has_9(self):
        self.assertEqual(
            SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start, 62
        )
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop - SWAPI_FINE_SWEEP_BINS.start, 9)


class TestKFactors(unittest.TestCase):
    def test_simion_k_factor(self):
        self.assertAlmostEqual(SWAPI_K_FACTOR, 1.89)

    def test_l2_label_k_factor_differs_from_simion(self):
        # L2 esa_energy = SWAPI_L2_K_FACTOR × |V|, divided out by L3 to recover voltage.
        self.assertAlmostEqual(SWAPI_L2_K_FACTOR, 1.93)
        self.assertNotAlmostEqual(SWAPI_K_FACTOR, SWAPI_L2_K_FACTOR)


class TestEsaVoltageToProtonSpeed(unittest.TestCase):
    def test_matches_analytical_formula_for_typical_solar_wind(self):
        # 1000 V → ~570 km/s for protons in the SWAPI ESA.
        for V in [100.0, 500.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_proton_speed(V),
                    _analytic_proton_speed_km_per_s(V),
                    rtol=1e-12,
                )

    def test_handles_negative_voltage_via_absolute_value(self):
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-2500.0),
            esa_voltage_to_proton_speed(2500.0),
        )

    def test_array_input_returns_elementwise_speeds(self):
        voltages = np.array([100.0, 1000.0, 4000.0])
        result = esa_voltage_to_proton_speed(voltages)
        expected = np.array([_analytic_proton_speed_km_per_s(v) for v in voltages])
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestEsaVoltageToAlphaSpeed(unittest.TestCase):
    def test_matches_analytical_formula(self):
        for V in [200.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_alpha_speed(V),
                    _analytic_alpha_speed_km_per_s(V),
                    rtol=1e-12,
                )

    def test_handles_negative_voltage(self):
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )


class TestFindPeakCenterOfMassIndex(unittest.TestCase):
    def test_recovers_known_peak_in_gaussian(self):
        bins = np.arange(72)
        true_peak = 35.4
        counts = 1000.0 * np.exp(-((bins - true_peak) ** 2) / (2.0 * 1.5**2))
        peak_slice = slice(30, 41)
        com = find_peak_center_of_mass_index(peak_slice, counts)
        np.testing.assert_allclose(com, true_peak, atol=0.05)

    def test_drops_bins_below_minimum_count_rate(self):
        counts = np.array([0.0, 0.0, 100.0, 50.0, 0.0])
        # With min_count_rate=10, only indices 2 and 3 contribute → CoM = (2*100 + 3*50)/150 = 7/3.
        com = find_peak_center_of_mass_index(
            slice(0, 5), counts, minimum_count_rate=10.0
        )
        self.assertAlmostEqual(com, (2 * 100.0 + 3 * 50.0) / 150.0)

    def test_raises_when_minimum_bin_count_violated(self):
        counts = np.array([0.0, 0.0, 100.0, 0.0])
        with self.assertRaises(Exception):
            find_peak_center_of_mass_index(
                slice(0, 4), counts, minimum_count_rate=10.0, minimum_bin_count=2
            )


class TestInterpolateEnergy(unittest.TestCase):
    def test_log_interpolation_between_two_known_grid_points(self):
        # Geometric energy table — interpolation should be exact in log space.
        energies = np.array([100.0, 1000.0, 10000.0])
        # Index 0.5 in log-space → sqrt(100*1000) ≈ 316.228.
        result = interpolate_energy(0.5, energies)
        np.testing.assert_allclose(result, np.sqrt(100.0 * 1000.0), rtol=1e-10)

    def test_at_grid_point_returns_grid_value(self):
        energies = np.array([100.0, 1000.0, 10000.0])
        np.testing.assert_allclose(
            interpolate_energy(1.0, energies), 1000.0, rtol=1e-10
        )

    def test_propagates_uncertainty_for_ufloat_index(self):
        energies = np.array([100.0, 1000.0, 10000.0])
        index = ufloat(1.0, 0.1)
        result = interpolate_energy(index, energies)
        self.assertIsInstance(result, UFloat)
        self.assertAlmostEqual(result.nominal_value, 1000.0)
        # In log-space, slope is ln(10) at index 1.0; sigma_E ≈ E * ln(10) * sigma_index.
        self.assertAlmostEqual(result.std_dev, 1000.0 * np.log(10.0) * 0.1, places=2)


class TestTimesForSweep(unittest.TestCase):
    def test_returns_72_entries_with_uniform_step(self):
        start = 1234.5
        times = times_for_sweep(start)
        self.assertEqual(len(times), 72)
        # Each bin is 12 s / 72 = 1/6 s wide.
        np.testing.assert_allclose(times[0], start)
        np.testing.assert_allclose(np.diff(times), 1.0 / 6.0, rtol=1e-12)

    def test_total_sweep_duration_just_under_12_s(self):
        times = times_for_sweep(0.0)
        self.assertAlmostEqual(times[-1] - times[0], 71.0 / 6.0)


class TestExtractCoarseSweep(unittest.TestCase):
    def test_2d_input_returns_columns_1_to_63(self):
        data = np.arange(72 * 5, dtype=float).reshape(5, 72)
        result = extract_coarse_sweep(data)
        np.testing.assert_array_equal(result, data[:, 1:63])
        self.assertEqual(result.shape, (5, 62))

    def test_1d_input_returns_indices_1_to_63(self):
        data = np.arange(72, dtype=float)
        result = extract_coarse_sweep(data)
        np.testing.assert_array_equal(result, data[1:63])
        self.assertEqual(result.shape, (62,))


class TestCalculateCombinedSweeps(unittest.TestCase):
    def test_returns_average_count_rate_and_mean_energy_over_coarse_bins(self):
        n_sweeps, n_bins = 5, 72
        rng = np.random.default_rng(seed=0)
        rates = rng.uniform(50.0, 200.0, size=(n_sweeps, n_bins))
        per_sweep_energy = np.logspace(np.log10(20_000.0), np.log10(50.0), n_bins)
        # Per-sweep energies vary slightly so the column mean is a meaningful test.
        energies = per_sweep_energy * (
            1.0 + 0.001 * rng.standard_normal((n_sweeps, n_bins))
        )

        avg_rate, avg_energy = calculate_combined_sweeps(rates, energies)

        self.assertEqual(avg_rate.shape, (62,))
        self.assertEqual(avg_energy.shape, (62,))
        np.testing.assert_allclose(avg_rate, rates[:, 1:63].mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(
            avg_energy, energies[:, 1:63].mean(axis=0), rtol=1e-12
        )


class TestGetAlphaPeakIndices(unittest.TestCase):
    def _alpha_test_spectrum(self):
        # SWAPI energies are decreasing. Build a spectrum with a proton peak at idx 50
        # and a smaller alpha shoulder around idx 30 (higher energies).
        energies = np.linspace(20_000.0, 50.0, 72)
        n_proton = 1000.0 * np.exp(-((np.arange(72) - 50) ** 2) / 4.0)
        n_alpha = 200.0 * np.exp(-((np.arange(72) - 30) ** 2) / 9.0)
        return energies, n_proton, n_alpha

    def test_returns_slice_containing_alpha_peak(self):
        energies, n_proton, n_alpha = self._alpha_test_spectrum()
        residuals = n_alpha
        peak_slice = get_alpha_peak_indices(residuals, energies, proton_peak_index=50)
        # Slice should include the alpha bump centered at idx 30.
        self.assertLessEqual(peak_slice.start, 30)
        self.assertGreater(peak_slice.stop, 30)

    def test_raises_when_no_alpha_bump_present(self):
        energies = np.linspace(20_000.0, 50.0, 72)
        residuals = np.zeros(72)
        with self.assertRaises(Exception):
            get_alpha_peak_indices(residuals, energies, proton_peak_index=50)

    def test_asserts_decreasing_energies(self):
        ascending_energies = np.linspace(50.0, 20_000.0, 72)
        with self.assertRaises(AssertionError):
            get_alpha_peak_indices(
                np.zeros(72), ascending_energies, proton_peak_index=50
            )


class TestCalculateSwSpeed(unittest.TestCase):
    def test_scalar_proton_input(self):
        E_J = 1.0e-16
        expected = (
            np.sqrt(2 * E_J * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        np.testing.assert_allclose(
            calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E_J), expected
        )

    def test_array_input_returns_array(self):
        E = np.array([1.0e-16, 2.0e-16, 4.0e-16])
        expected = (
            np.sqrt(2 * E * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        np.testing.assert_allclose(
            calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E), expected
        )

    def test_2d_array_input_preserves_shape(self):
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertEqual(result.shape, E.shape)
        np.testing.assert_allclose(result, calculate_sw_speed_h_plus(E))

    def test_empty_array_input_returns_empty_array(self):
        result = calculate_sw_speed(
            PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, np.array([])
        )
        self.assertEqual(result.size, 0)

    def test_ufloat_scalar_propagates_uncertainty(self):
        E = ufloat(1.0e-16, 1.0e-18)
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertIsInstance(result, UFloat)
        # σ_v = v · σ_E / (2 E)
        expected_nom = (
            np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        expected_sigma = expected_nom * E.std_dev / (2 * E.nominal_value)
        self.assertAlmostEqual(result.nominal_value, expected_nom)
        self.assertAlmostEqual(result.std_dev, expected_sigma)

    def test_ufloat_array_propagates_uncertainty_per_element(self):
        E = np.array([ufloat(1.0e-16, 1.0e-18), ufloat(4.0e-16, 1.0e-18)])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        nominal = nominal_values(result)
        expected = (
            np.sqrt(
                2.0
                * np.array([1.0e-16, 4.0e-16])
                * PROTON_CHARGE_COULOMBS
                / PROTON_MASS_KG
            )
            / METERS_PER_KILOMETER
        )
        np.testing.assert_allclose(nominal, expected)


class TestCalculateSwSpeedHPlus(unittest.TestCase):
    def test_consistent_with_calculate_sw_speed_for_protons(self):
        E = np.array([1.0e-16, 4.0e-16])
        np.testing.assert_allclose(
            calculate_sw_speed_h_plus(E),
            calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E),
        )


if __name__ == "__main__":
    unittest.main()
