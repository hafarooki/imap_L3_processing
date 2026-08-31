import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, call

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, CodiceLoDirectEventData
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events, calculate_mass, calculate_mass_per_charge, \
    rebin_to_counts_by_species_elevation_and_spin_sector, rebin_direct_events_by_energy_and_spin_sector, \
    CODICE_LO_NUM_AZIMUTH_BINS, combine_priorities_for_species_and_convert_to_rate, \
    rebin_3d_distribution_azimuth_to_elevation, convert_count_rate_to_intensity, rebin_direct_events_for_normalization, \
    calculate_normalization_factor, lookup_normalization_per_event


class TestCodiceLoCalculations(unittest.TestCase):

    def test_calculate_total_number_of_events(self):
        priority_0_tcrs = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]],
            [[2, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]]])
        acquisition_time = np.array([1, 2, 3, 4]) * 1_000_000

        expected_total_number_of_events = [6 + 30 + 72 + 132, 7 + 30 + 72 + 132]

        actual_total_number_of_events = calculate_total_number_of_events(priority_0_tcrs, acquisition_time)
        np.testing.assert_array_equal(actual_total_number_of_events, expected_total_number_of_events)

    def test_calculate_mass(self):
        apd_energy = np.array([[np.exp(1)], [np.exp(2)]])
        tof = np.array([[np.exp(50)], [np.exp(60)]])

        lookup = MassCoefficientLookup(np.array([10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6]))
        actual_mass = calculate_mass(apd_energy, tof, lookup)

        expected_mass_1 = lookup[0] + lookup[1] * 1 + lookup[2] * 50 + lookup[3] * 1 * 50 + lookup[4] * 1 + lookup[
            5] * np.power(50, 3)
        expected_mass_2 = lookup[0] + lookup[1] * 2 + lookup[2] * 60 + lookup[3] * 2 * 60 + lookup[4] * np.power(2, 2) + \
                          lookup[5] * np.power(60, 3)

        expected_mass_calculation = np.array([
            [np.e ** expected_mass_1],
            [np.e ** expected_mass_2]])

        np.testing.assert_array_equal(actual_mass, expected_mass_calculation)

    def test_calculate_mass_per_charge(self):
        tof = np.array([[5], [6]])
        energy_per_charge = np.array([[1], [2]])

        POST_ACCELERATION_VOLTAGE_IN_KV = 15
        ENERGY_LOST_IN_CARBON_FOIL = 0
        CONVERSION_CONSTANT_K = 1.692e-5

        mass_per_charge_1 = (1 + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
                5 ** 2) * CONVERSION_CONSTANT_K

        mass_per_charge_2 = (2 + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
                6 ** 2) * CONVERSION_CONSTANT_K

        actual_mass_per_charge = calculate_mass_per_charge(energy_per_charge, tof)
        np.testing.assert_array_equal(actual_mass_per_charge, np.array([[mass_per_charge_1], [mass_per_charge_2]]))

    def test_calculate_mass_per_charge_handles_fill_value(self):
        tof = np.ma.masked_array([[432, 234], [434, 347]], mask=[[False, True], [True, False]])
        apd_energy = np.array([[np.nan, 2342.2], [4324.8, np.nan]])

        actual_mass_per_charge = calculate_mass_per_charge(apd_energy, tof)

        self.assertIsInstance(actual_mass_per_charge, np.ma.masked_array)
        actual_filled_with_nan = np.ma.filled(actual_mass_per_charge, np.nan)

        self.assertTrue(np.all(np.isnan(actual_filled_with_nan)))

    def test_calculate_mass_handles_fill_value(self):
        rng = np.random.default_rng()
        mass_coefficients = MassCoefficientLookup(coefficients=rng.random(size=6))

        tof = np.ma.masked_array([[432, 234], [434, 347]], mask=[[False, True], [True, False]])
        apd_energy = np.array([[np.nan, 2342.2], [4324.8, np.nan]])

        actual_mass = calculate_mass(apd_energy, tof, mass_coefficients)

        self.assertIsInstance(actual_mass, np.ma.masked_array)
        actual_filled_with_nan = np.ma.filled(actual_mass, np.nan)

        self.assertTrue(np.all(np.isnan(actual_filled_with_nan)))

    def test_calculate_partial_densities(self):
        rng = np.random.default_rng()
        epochs = np.array([datetime.now(), datetime.now() + timedelta(days=1)])
        energy_steps = np.geomspace(100000, 1, num=128)
        intensities = rng.random((len(epochs), len(energy_steps), 1))
        mass_per_charge = 10

        partial_densities = calculate_partial_densities(intensities, energy_steps, mass_per_charge)

        expected_partial_densities = np.sum(
            2.283e-8 * np.deg2rad(15) * np.deg2rad(15) * .4 * intensities * np.sqrt(
                energy_steps[np.newaxis, :, np.newaxis]) * np.sqrt(mass_per_charge), axis=(1, 2))

        np.testing.assert_array_equal(expected_partial_densities, partial_densities)

    def test_calculate_partial_densities_ignores_nan_intensities(self):
        energy_steps = np.array([1, 100, 10000])
        intensities = np.array([
            [[1], [2], [3]],
            [[4], [5], [np.nan]]
        ])
        mass_per_charge = 10

        partial_densities = calculate_partial_densities(intensities, energy_steps, mass_per_charge)

        expected_partial_densities = 2.283e-8 * np.deg2rad(15) * np.deg2rad(15) * .4 * np.array([
            1 + 2 * 10 + 3 * 100,
            4 + 5 * 10
        ]) * np.sqrt(mass_per_charge)

        np.testing.assert_allclose(expected_partial_densities, partial_densities)

    def test_rebin_to_counts_by_species_elevation_and_spin_sector(self):
        num_epochs = 2
        num_priorities = 2
        num_spin_sectors = 24
        num_esa_steps = 128
        num_species = 4

        # [[He+, Fe, He+, no call], [O+5, no call, no call, no call]],
        # [[Mg, no call, None, no call], [no call, no call, no call, no call]]
        expected_species_returned = [
            "He+", "Fe", "He+", "O+5",
            "Mg", None
        ]
        mock_species_mass_range_lookup = Mock(spec=MassSpeciesBinLookup)
        mock_species_mass_range_lookup.get_species.side_effect = expected_species_returned
        mock_species_mass_range_lookup.get_num_species.return_value = num_species

        def mock_get_species_index(species):
            return_values = {"He+": 0, "Fe": 1, "Mg": 2, "O+5": 3, }
            return return_values[species]

        mock_species_mass_range_lookup.get_species_index = mock_get_species_index

        num_events = np.array([[4, 2],
                               [3, 1]])
        num_events = np.ma.masked_array(num_events, mask=[[False, False], [False, True]])

        normalized_per_event = np.array([
            [[100.78, 30.9, 200.1, np.nan], [50.458, np.nan, np.nan, np.nan]],
            [[60.22, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]]
        ])

        spin_sector_indices = np.array([
            [[2, 1, 2, 0], [0, 0, 0, 0]],
            [[2, 2, 2, 0], [0, 0, 0, 0]]
        ])

        apd_id = np.ma.masked_array(data=np.array([
            [[2, 24, 2, 0], [1, 25, 255, 255]],
            [[9, 9, 9, 255], [1, 255, 255, 255]]
        ]), mask=np.array([
            [[False, False, False, False], [False, False, True, True]],
            [[False, True, False, True], [False, True, True, True]],
        ]))

        energy_step = np.ma.masked_array(data=np.array([
            [[0, 100, 0, 0], [0, 100, 255, 255]],
            [[127, 255, 127, 255], [0, 255, 255, 255]]
        ]), mask=np.array([
            [[False, False, False, False], [False, False, True, True]],
            [[False, True, False, True], [False, True, True, True]],
        ]))

        mass = np.array([
            [[6, 5, 4, np.nan], [7, 5, np.nan, np.nan]],
            [[3, 3, 3, np.nan], [7, np.nan, np.nan, np.nan]],
        ])

        mass_per_charge = np.array([
            [[1, 2, 3, np.nan], [8, 2, np.nan, np.nan]],
            [[4, 4, 4, np.nan], [9, np.nan, np.nan, np.nan]],
        ])

        direct_event_data = Mock(spec=CodiceLoDirectEventData)
        direct_event_data.mass = mass
        direct_event_data.mass_per_charge = mass_per_charge
        direct_event_data.energy_step = energy_step
        direct_event_data.spin_sector = spin_sector_indices
        direct_event_data.apd_id = apd_id
        direct_event_data.normalization_per_event = normalized_per_event
        direct_event_data.num_events = num_events

        actual_counts_3d_data = rebin_to_counts_by_species_elevation_and_spin_sector(
            direct_event_data,
            mock_species_mass_range_lookup,
        )

        mock_species_mass_range_lookup.get_species.assert_has_calls([
            call(mass[0, 0, 0], mass_per_charge[0, 0, 0]),
            call(mass[0, 0, 1], mass_per_charge[0, 0, 1]),
            call(mass[0, 0, 2], mass_per_charge[0, 0, 2]),

            call(mass[0, 1, 0], mass_per_charge[0, 1, 0]),

            call(mass[1, 0, 0], mass_per_charge[1, 0, 0]),
            call(mass[1, 0, 2], mass_per_charge[1, 0, 2]),
        ])

        self.assertIsInstance(actual_counts_3d_data, np.ndarray)

        output_shape = [num_species, num_epochs, num_priorities, num_esa_steps, num_spin_sectors,
                        CODICE_LO_NUM_AZIMUTH_BINS]
        self.assertEqual(
            tuple(output_shape),
            actual_counts_3d_data.shape)

        rebinned_shape = tuple(output_shape[1:])
        expected_he_plus_counts = np.zeros(rebinned_shape)
        expected_he_plus_counts[0, 0, 0, 2, 1] = 300.88
        np.testing.assert_array_equal(actual_counts_3d_data[0, ...], expected_he_plus_counts)

        expected_fe_counts = np.zeros(rebinned_shape)
        expected_fe_counts[0, 0, 100, 1, 23] = 30.9
        np.testing.assert_array_equal(actual_counts_3d_data[1, ...], expected_fe_counts)

        expected_mg_counts = np.zeros(rebinned_shape)
        expected_mg_counts[1, 0, 127, 2, 8] = 60.22
        np.testing.assert_array_equal(actual_counts_3d_data[2, ...], expected_mg_counts)

        expected_o_counts = np.zeros(rebinned_shape)
        expected_o_counts[0, 1, 0, 0, 0] = 50.458
        np.testing.assert_array_equal(actual_counts_3d_data[3, ...], expected_o_counts)

    def test_rebin_direct_events_by_energy_and_spin_sector(self):
        num_energy_bins = 30
        num_spin_angle_bins = 20
        num_priorities = 3
        num_epochs = 1
        event_buffer_len = 15
        num_events = np.ma.masked_array(np.array([[1, 2, 4]]), mask=[[False, True, False]])
        spin_sector = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        spin_sector[0, 0, :1] = [3]
        spin_sector[0, 2, :4] = [5, 6, 5, 7]

        energy_step = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        energy_step[0, 0, :1] = [7]
        energy_step[0, 2, :4] = [1, 3, 1, 4]

        result = rebin_direct_events_by_energy_and_spin_sector(num_events, spin_sector, energy_step,
                                                               num_spin_angle_bins, num_energy_bins)

        expected_rebinned_counts = np.zeros((num_epochs, num_priorities, num_energy_bins, num_spin_angle_bins))
        expected_rebinned_counts[0, 0, 7, 3] = 1
        expected_rebinned_counts[0, 2, 1, 5] = 2
        expected_rebinned_counts[0, 2, 3, 6] = 1
        expected_rebinned_counts[0, 2, 4, 7] = 1

        np.testing.assert_array_equal(result, expected_rebinned_counts)


    def test_rebin_direct_events_by_energy_and_spin_sector_ignores_masked_events(self):
        num_energy_bins = 30
        num_spin_angle_bins = 20
        num_priorities = 3
        num_epochs = 1
        event_buffer_len = 15
        num_events = np.ma.masked_array(np.array([[1, 2, 4]]), mask=[[False, True, False]])
        spin_sector = np.ma.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        spin_sector[0, 0, :1] = [3]
        spin_sector[0, 2, :4] = [5, 6, 255, 7]
        spin_sector[0, 2, 2] = np.ma.masked

        energy_step = np.ma.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        energy_step[0, 0, :1] = [7]
        energy_step[0, 2, :4] = [1, 255, 1, 4]
        energy_step[0, 2, 1] = np.ma.masked

        result = rebin_direct_events_by_energy_and_spin_sector(num_events, spin_sector, energy_step,
                                                               num_spin_angle_bins, num_energy_bins)

        expected_rebinned_counts = np.zeros((num_epochs, num_priorities, num_energy_bins, num_spin_angle_bins))
        expected_rebinned_counts[0, 0, 7, 3] = 1
        expected_rebinned_counts[0, 2, 1, 5] = 1
        expected_rebinned_counts[0, 2, 3, 6] = 0
        expected_rebinned_counts[0, 2, 4, 7] = 1

        np.testing.assert_array_equal(result, expected_rebinned_counts)

    def test_rebin_direct_events_for_normalization(self):
        num_energy_bins = 30
        num_spin_sectors = 24
        num_priorities = 3
        num_epochs = 1
        event_buffer_len = 15
        num_events = np.ma.masked_array(np.array([[2, 2, 4]]), mask=[[False, True, False]])
        spin_sector = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        spin_sector[0, 0, :2] = [3, 3 + 12]
        spin_sector[0, 2, :4] = [5, 6, 5, 7 + 12]

        energy_step = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=np.uint8)
        energy_step[0, 0, :2] = [7, 7]
        energy_step[0, 2, :4] = [1, 3, 1, 4]

        result = rebin_direct_events_for_normalization(num_events, spin_sector, energy_step, num_spin_sectors,
                                                       num_energy_bins)

        expected_rebinned_counts = np.zeros((num_epochs, num_priorities, num_energy_bins, num_spin_sectors))
        expected_rebinned_counts[0, 0, 7, 3] = 2
        expected_rebinned_counts[0, 2, 1, 5] = 2
        expected_rebinned_counts[0, 2, 3, 6] = 1
        expected_rebinned_counts[0, 2, 4, 7] = 1
        expected_rebinned_counts[0, 0, 7, 3 + 12] = 2
        expected_rebinned_counts[0, 2, 1, 5 + 12] = 2
        expected_rebinned_counts[0, 2, 3, 6 + 12] = 1
        expected_rebinned_counts[0, 2, 4, 7 + 12] = 1
        np.testing.assert_equal(result[0, 0, 7, 3], 2)

        np.testing.assert_array_equal(result, expected_rebinned_counts)

    def test_calculate_normalization_factor(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors
        priority_counts = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l1_spin_sectors))
        priority_counts[0, 0, 0, 0] = 4
        priority_counts[0, 0, 0, 2] = 20

        event_buffer_len = 15
        num_events = np.zeros((num_epochs, num_priorities), dtype=int)
        spin_sectors = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)
        energy_steps = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)
        

        num_events[0, 0] = 4
        spin_sectors[0, 0, :4] = [0, 2, 12, 0]
        energy_steps[0, 0, :4] = [0, 0, 0, 0]

        result = calculate_normalization_factor(priority_counts, num_events, energy_steps, spin_sectors)

        expected_shape = (num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors)
        expected = np.full(expected_shape, 0.0)
        expected[0, 0, 0, 0] = 4 / 3
        expected[0, 0, 0, 0 + 12] = 4 / 3
        expected[0, 0, 0, 2] = 20 / 1
        expected[0, 0, 0, 2 + 12] = 20 / 1

        np.testing.assert_equal(result, expected)

    def test_calculate_normalization_factor_floor_is_one(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors
        priority_counts = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l1_spin_sectors))
        priority_counts[0, 0, 0, 0] = 1.
        priority_counts[0, 0, 0, 2] = 0.

        event_buffer_len = 15
        num_events = np.zeros((num_epochs, num_priorities), dtype=int)
        spin_sectors = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)
        energy_steps = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)

        num_events[0, 0] = 3
        spin_sectors[0, 0, :3] = [0, 12, 2]
        energy_steps[0, 0, :3] = [0, 0, 0]

        result = calculate_normalization_factor(priority_counts, num_events, energy_steps, spin_sectors)

        expected_shape = (num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors)
        expected = np.full(expected_shape, 0.0)
        self.assertEqual(1.0, result[0, 0, 0, 0])
        self.assertEqual(1.0, result[0, 0, 0, 0 + 12])
        self.assertEqual(1.0, result[0, 0, 0, 2])
        self.assertEqual(1.0, result[0, 0, 0, 2 + 12])

        expected[0, 0, 0, (0, 2, 12, 14)] = 1.0
        np.testing.assert_equal(result, expected)

    def test_normalization_factor_is_fill_when_priority_nonzero_and_direct_zero(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors
        priority_counts = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l1_spin_sectors))
        priority_counts[0, 0, 0, 0] = 1.
        priority_counts[0, 0, 0, 2] = 50.

        event_buffer_len = 15
        num_events = np.zeros((num_epochs, num_priorities), dtype=int)
        spin_sectors = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)
        energy_steps = np.zeros((num_epochs, num_priorities, event_buffer_len), dtype=int)

        result = calculate_normalization_factor(priority_counts, num_events, energy_steps, spin_sectors)

        expected_shape = (num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors)
        expected = np.full(expected_shape, 0.0)
        np.testing.assert_equal(result[0, 0, 0, 0], np.nan)
        np.testing.assert_equal(result[0, 0, 0, 0 + 12], np.nan)
        np.testing.assert_equal(result[0, 0, 0, 2], np.nan)
        np.testing.assert_equal(result[0, 0, 0, 2 + 12], np.nan)

        expected[0, 0, 0, (0, 2, 12, 14)] = np.nan
        np.testing.assert_equal(result, expected)

    def test_lookup_normalization_per_event(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors

        normalization = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors))
        normalization[0, 0, 0, 0] = 123.
        normalization[0, 0, 0, 12] = 456.
        normalization[0, 0, 1, 2] = 789.

        event_buffer_len = 15
        num_events = np.zeros((num_epochs, num_priorities), dtype=int)
        spin_sectors = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)
        energy_steps = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)

        num_events[0, 0] = 3
        spin_sectors[0, 0, :3] = [0, 12, 2]
        energy_steps[0, 0, :3] = [0, 0, 1]

        normalization_per_event = lookup_normalization_per_event(normalization, num_events, energy_steps, spin_sectors)

        np.testing.assert_array_equal(normalization_per_event[0, 0, :3], [123, 456, 789])

        expected_normalization_per_event = np.full((num_epochs, num_priorities, event_buffer_len), np.nan)
        expected_normalization_per_event[0, 0, :3] = [123, 456, 789]

        np.testing.assert_array_equal(normalization_per_event, expected_normalization_per_event)

    def test_lookup_normalization_per_event_skips_masked_num_events(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors

        normalization = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors))
        normalization[0, 0, 0, 0] = 123.
        normalization[0, 0, 0, 12] = 456.
        normalization[0, 0, 1, 2] = 789.

        event_buffer_len = 15
        num_events = np.ma.array(np.full((num_epochs, num_priorities), 65535), mask=True)
        spin_sectors = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)
        energy_steps = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)

        normalization_per_event = lookup_normalization_per_event(normalization, num_events, energy_steps, spin_sectors)

        expected_normalization_per_event = np.full((num_epochs, num_priorities, event_buffer_len), np.nan)
        np.testing.assert_array_equal(normalization_per_event, expected_normalization_per_event)

    def test_lookup_normalization_per_event_ignores_masked_events(self):
        num_epochs = 2
        num_priorities = 7
        num_esa_steps = 128
        num_l1_spin_sectors = 12
        num_l2_spin_sectors = 2 * num_l1_spin_sectors

        normalization = np.zeros((num_epochs, num_priorities, num_esa_steps, num_l2_spin_sectors))
        normalization[0, 0, 0, 0] = 123.
        normalization[0, 0, 0, 12] = 456.
        normalization[0, 0, 1, 2] = 789.

        event_buffer_len = 15
        num_events = np.zeros((num_epochs, num_priorities), dtype=int)
        spin_sectors = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)
        energy_steps = np.ma.array(np.full((num_epochs, num_priorities, event_buffer_len), 255, dtype=int), mask=True)

        num_events[0, 0] = 4
        energy_steps[0, 0, 1] = 0
        energy_steps[0, 0, 3] = 0

        spin_sectors[0, 0, :4] = np.ma.masked_equal([255, 255, 0, 12], 255)
        energy_steps[0, 0, :4] = np.ma.masked_equal([255, 0, 255, 0], 255)

        normalization_per_event = lookup_normalization_per_event(normalization, num_events, energy_steps, spin_sectors)

        np.testing.assert_array_equal(normalization_per_event[0, 0, :4], [np.nan, np.nan, np.nan, 456])

        expected_normalization_per_event = np.full((num_epochs, num_priorities, event_buffer_len), np.nan)
        expected_normalization_per_event[0, 0, :4] = [np.nan, np.nan, np.nan, 456]

        np.testing.assert_array_equal(normalization_per_event, expected_normalization_per_event)

    def test_combine_priorities_for_species_and_convert_to_rate(self):
        num_epochs = 2
        num_energies = 7
        num_spin_sectors = 6
        num_azimuth_bins = 5

        rng = np.random.default_rng()
        priority_1 = rng.random((num_epochs, num_energies, num_spin_sectors, num_azimuth_bins))
        priority_2 = rng.random((num_epochs, num_energies, num_spin_sectors, num_azimuth_bins))
        priority_3 = rng.random((num_epochs, num_energies, num_spin_sectors, num_azimuth_bins))

        counts = np.stack((priority_1, priority_2, priority_3), axis=1)

        acquisition_durations_in_seconds = rng.random((num_epochs, num_energies,))

        actual_count_rates = combine_priorities_for_species_and_convert_to_rate(counts,
                                                                                acquisition_durations_in_seconds)
        expected_summed_counts = priority_1 + priority_2 + priority_3

        self.assertEqual((num_epochs, num_energies, num_spin_sectors, num_azimuth_bins),
                         actual_count_rates.shape)

        for index in np.ndindex(num_epochs, num_spin_sectors, num_azimuth_bins):
            epoch, spin, azimuth = index
            expected_count_rate_per_epoch_spin_azimuth = expected_summed_counts[epoch, :, spin, azimuth] / \
                                                         acquisition_durations_in_seconds[epoch]
            np.testing.assert_array_almost_equal(actual_count_rates[epoch, :, spin, azimuth],
                                                 expected_count_rate_per_epoch_spin_azimuth)

    def test_convert_count_rate_to_intensity(self):
        num_epochs = 3
        num_position_bins = 4
        num_spin_angles = 5
        num_energies = 6

        rng = np.random.default_rng()
        count_rates = rng.random((num_epochs, num_energies, num_spin_angles, num_position_bins))
        energy_per_charge = EnergyLookup(bin_centers=rng.random(num_energies),
                                         delta_plus=rng.random(num_energies),
                                         delta_minus=rng.random(num_energies))
        geometric_factor = rng.random((num_epochs, num_energies, num_spin_angles, num_position_bins))

        efficiency = rng.random((num_energies, num_position_bins))

        mock_efficiency_lookup = Mock()
        mock_efficiency_lookup.efficiency_data = efficiency

        intensity_data = convert_count_rate_to_intensity(count_rates, energy_per_charge, mock_efficiency_lookup,
                                                         geometric_factor)

        expected_denominator = (energy_per_charge.bin_centers[np.newaxis, :, np.newaxis, np.newaxis]
                                * geometric_factor
                                * efficiency[np.newaxis, :, np.newaxis, :])

        np.testing.assert_array_almost_equal(intensity_data * expected_denominator, count_rates)

    def test_rebin_azimuth_to_elevation(self):
        num_epochs = 3
        num_energies = 6
        num_spin_angles = 24
        num_azimuth_bins = 24

        intensity_data = np.zeros((num_epochs, num_energies, num_spin_angles, num_azimuth_bins))
        azimuth_1_intensity = 1
        azimuth_2_intensity = 2
        azimuth_3_intensity = 3
        azimuth_4_intensity = 4

        intensity_data[:, :, :, 0] = azimuth_1_intensity
        intensity_data[:, :, :, 1] = azimuth_2_intensity
        intensity_data[:, :, :, 2] = azimuth_3_intensity
        intensity_data[:, :, :, 3] = azimuth_4_intensity

        azimuths = np.array([1, 2, 3, 4])

        num_elevation_bins = 13
        mock_position_to_elevation_lookup = Mock(spec=PositionToElevationLookup)
        mock_position_to_elevation_lookup.num_bins = num_elevation_bins
        mock_position_to_elevation_lookup.bin_centers = np.arange(num_elevation_bins)
        mock_position_to_elevation_lookup.apd_to_elevation_index.return_value = np.array([1, 0, 2, 2])

        half_spin_per_esa = np.zeros((num_epochs, num_energies))
        actual_rebinned = rebin_3d_distribution_azimuth_to_elevation(intensity_data,
                                                                     azimuths,
                                                                     mock_position_to_elevation_lookup,
                                                                     half_spin_per_esa)

        mock_position_to_elevation_lookup.apd_to_elevation_index.assert_called_once_with(azimuths)

        self.assertEqual((num_epochs, num_energies, num_spin_angles, num_elevation_bins),
                         actual_rebinned.shape)

        self.assertTrue(np.all(actual_rebinned[:, :, :, 0] == azimuth_2_intensity))
        self.assertTrue(np.all(actual_rebinned[:, :, :, 1] == azimuth_1_intensity))
        self.assertTrue(np.all(actual_rebinned[:, :, :, 2] == (azimuth_3_intensity + azimuth_4_intensity)))
