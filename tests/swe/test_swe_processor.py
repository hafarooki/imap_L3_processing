import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch, call, Mock, sentinel

import numpy as np
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.models import MagData, InputMetadata
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swe.l3.models import SweL2Data, SwapiL3aProtonData, SweL1bData
from imap_l3_processing.swe.l3.models import SweL3MomentData
from imap_l3_processing.swe.l3.science.moment_calculations import MomentFitResults, ScaleDensityOutput
from imap_l3_processing.swe.l3.science.moment_calculations import Moments
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.quality_flags import SweL3Flags
from imap_l3_processing.swe.swe_processor import SweProcessor, logger
from tests.test_helpers import NumpyArrayMatcher, build_swe_configuration, create_dataclass_mock, build_moments, \
    build_moment_fit_results


class TestSweProcessor(unittest.TestCase):
    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.swe.swe_processor.save_data')
    @patch('imap_l3_processing.swe.swe_processor.SweL3Dependencies.fetch_dependencies')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_products')
    def test_process(self, mock_calculate_products, mock_fetch_dependencies, mock_save_data, mock_spiceypy):
        input_file_names = [
            "imap_swe_l2_sci_20200101_v000.cdf",
            "imap_swe_l1b_sci_20200101_v000.cdf",
            "imap_mag_l1d_norm-dsrf_20200101_v000.cdf",
            "imap_swapi_l3_proton-sw_20200101_v000.cdf",
            "imap_swe_config-json-not-cdf_20200101_v000.cdf"
        ]

        mock_spiceypy.ktotal.return_value = 0

        science_inputs = [ScienceInput(file_name) for file_name in input_file_names[0:4]]
        config_input = AncillaryInput(input_file_names[4])
        mock_dependencies = ProcessingInputCollection(*science_inputs, config_input)

        mock_input_metadata = Mock()
        calculate_products_return = Mock()
        mock_calculate_products.return_value = calculate_products_return
        swe_processor = SweProcessor(mock_dependencies, mock_input_metadata)
        product = swe_processor.process()

        self.assertEqual(calculate_products_return.parent_file_names, input_file_names)

        mock_fetch_dependencies.assert_called_once_with(mock_dependencies)
        mock_calculate_products.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_once_with(mock_calculate_products.return_value)
        self.assertEqual([mock_save_data.return_value], product)

    @patch("imap_l3_processing.swe.swe_processor.compute_epoch_delta_in_ns")
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.mec_breakpoint_finder')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_pitch_angle_products')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products')
    def test_calculate_products(self, mock_calculate_moment_products, mock_calculate_pitch_angle_products,
                                mock_breakpoint_finder, mock_average_over_look_directions,
                                mock_compute_epoch_delta_in_ns):
        mock_compute_epoch_delta_in_ns.return_value = [30e9, 40e9]
        epochs = datetime.now() + np.arange(7) * timedelta(minutes=1)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(2) * timedelta(minutes=.5)
        mock_breakpoint_finder.side_effect = [
            (1, 2, SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE | SweL3Flags.POTENTIAL_FIT_UNCONVERGED),
            (3, 4, SweL3Flags.NONE),
            (5, 6, SweL3Flags.NONE),
            (7, 8, SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE),
            (9, 10, SweL3Flags.NONE),
            (11, 12, SweL3Flags.NONE),
            (13, 14, SweL3Flags.NONE),
        ]
        expected_spacecraft_potential = [1, 3, 5, 7, 9, 11, 13]
        expected_core_halo_breakpoint = [2, 4, 6, 8, 10, 12, 14]
        geometric_fractions = [1e7, 1e6, 1e4, 1e4, 1e3, 1e1, 1e1]
        sum_geometric_fractions = np.sum(geometric_fractions)

        psd_rebinned = np.array([4e-7, 4e-6, 4e-4, 4e-4, 4e-3, 4e-1, 4e-1]) * sum_geometric_fractions

        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(21).reshape(7, 3) + 100,
            flux=np.arange(21).reshape(7, 3),
            energy=np.array([2, 4, 6]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(21).reshape(7, 3) + 200,
            acquisition_time=np.array([]),
            acquisition_duration=np.array([]),
            phase_space_density_rebinned=np.broadcast_to(psd_rebinned[np.newaxis, np.newaxis, :], (7, 3, 7))
        )

        expected_corrected_energy_bins = np.array([
            [2 - 1, 4 - 1, 6 - 1],
            [2 - 3, 4 - 3, 6 - 3],
            [2 - 5, 4 - 5, 6 - 5],
            [2 - 7, 4 - 7, 6 - 7],
            [2 - 9, 4 - 9, 6 - 9],
            [2 - 11, 4 - 11, 6 - 11],
            [2 - 13, 4 - 13, 6 - 13],
        ])

        mag_l1d_data = Mock()

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
            swp_flags=np.array([]),
        )
        mock_average_over_look_directions.return_value = np.array([5, 10, 15])


        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_delta=[45, 45, 45],
            energy_bins=[1, 10, 100],
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
            gyrophase_bins=[90, 180, 270],
            gyrophase_delta=[90, 90, 90]
        )

        mock_moment_data = create_dataclass_mock(SweL3MomentData)
        mock_calculate_moment_products.return_value = mock_moment_data

        calculate_pitch_angle_flags = np.array([SweL3Flags.NONE] * len(epochs))
        calculate_pitch_angle_flags[0] = SweL3Flags.FALLBACK_SWAPI_SPEED

        mock_calculate_pitch_angle_products.return_value = (
            sentinel.expected_phase_space_density_by_pitch_angle,
            sentinel.expected_phase_space_density_by_pitch_angle_and_gyrophase, sentinel.expected_intensity,
            sentinel.expected_phase_space_density_inward, sentinel.expected_phase_space_density_outward,
            sentinel.intensity_by_pitch_angle_and_gyrophase, sentinel.intensity_by_pitch_angle,
            sentinel.uncertainty_by_pitch_angle_and_gyrophase, sentinel.uncertainty_by_pitch_angle,
            calculate_pitch_angle_flags
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swe_l1b_data = Mock()
        swe_l3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)

        swe_processor = SweProcessor(swe_l3_dependency, input_metadata)
        swe_l3_data = swe_processor.calculate_products(swe_l3_dependency)

        mock_compute_epoch_delta_in_ns.assert_called_once_with(swe_l2_data.acquisition_duration,
                                                               swe_l1b_data.settle_duration)

        self.assertEqual(7, mock_average_over_look_directions.call_count)
        first_average_over_look_directions_call_args = mock_average_over_look_directions.call_args_list[0].args
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[0],
                                      swe_l2_data.phase_space_density[0])
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[1],
                                      swe_config["geometric_fractions"])
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[2],
                                      swe_config["minimum_phase_space_density_value"])

        middle_average_over_look_directions_call_args = mock_average_over_look_directions.call_args_list[3].args
        np.testing.assert_array_equal(middle_average_over_look_directions_call_args[0],
                                      np.mean([
                                          swe_l2_data.phase_space_density[0],
                                          swe_l2_data.phase_space_density[1],
                                          swe_l2_data.phase_space_density[2],
                                          swe_l2_data.phase_space_density[3],
                                          swe_l2_data.phase_space_density[4],
                                          swe_l2_data.phase_space_density[5],
                                          swe_l2_data.phase_space_density[6],
                                      ], axis=0))
        np.testing.assert_array_equal(middle_average_over_look_directions_call_args[1],
                                      swe_config["geometric_fractions"])
        np.testing.assert_array_equal(middle_average_over_look_directions_call_args[2],
                                      swe_config["minimum_phase_space_density_value"])

        last_average_over_look_directions_call_args = mock_average_over_look_directions.call_args_list[6].args
        np.testing.assert_array_equal(last_average_over_look_directions_call_args[0],
                                      np.mean([
                                          swe_l2_data.phase_space_density[3],
                                          swe_l2_data.phase_space_density[4],
                                          swe_l2_data.phase_space_density[5],
                                          swe_l2_data.phase_space_density[6],
                                      ], axis=0))
        np.testing.assert_array_equal(last_average_over_look_directions_call_args[1],
                                      swe_config["geometric_fractions"])
        np.testing.assert_array_equal(last_average_over_look_directions_call_args[2],
                                      swe_config["minimum_phase_space_density_value"])

        mock_breakpoint_finder.assert_has_calls([
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value),
        ])

        mag_l1d_data.rebin_to.assert_called_with(epochs, [timedelta(seconds=30), timedelta(seconds=40)])
        mock_calculate_moment_products.assert_called_once()
        self.assertEqual(swe_l2_data, mock_calculate_moment_products.call_args[0][0])
        self.assertEqual(swe_l1b_data, mock_calculate_moment_products.call_args[0][1])
        self.assertEqual(mag_l1d_data.rebin_to.return_value, mock_calculate_moment_products.call_args[0][2])
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][3], expected_spacecraft_potential)
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][4], expected_core_halo_breakpoint)
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][5], expected_corrected_energy_bins)
        self.assertEqual(swe_config, mock_calculate_moment_products.call_args[0][6])

        mock_calculate_pitch_angle_products.assert_called_once()
        self.assertEqual(swe_l3_dependency, mock_calculate_pitch_angle_products.call_args[0][0])
        np.testing.assert_array_equal(mock_calculate_pitch_angle_products.call_args[0][1],
                                      expected_corrected_energy_bins)

        # @formatter:off
        self.assertEqual(swe_l3_data.input_metadata, replace(input_metadata, descriptor="sci"))
        # pass through from l2
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, mock_compute_epoch_delta_in_ns.return_value)
        # coming from the config
        np.testing.assert_array_equal(swe_l3_data.energy, swe_config["energy_bins"])
        np.testing.assert_array_equal(swe_l3_data.energy_delta_plus, swe_config["energy_delta_plus"])
        np.testing.assert_array_equal(swe_l3_data.energy_delta_minus, swe_config["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.pitch_angle, swe_config["pitch_angle_bins"])
        np.testing.assert_array_equal(swe_l3_data.pitch_angle_delta, swe_config["pitch_angle_deltas"])
        np.testing.assert_array_equal(swe_l3_data.gyrophase_bins, swe_config["gyrophase_bins"])
        np.testing.assert_array_equal(swe_l3_data.gyrophase_delta, swe_config["gyrophase_deltas"])
        np.testing.assert_array_equal(swe_l3_data.intensity_by_pitch_angle_and_gyrophase,
                                      sentinel.intensity_by_pitch_angle_and_gyrophase)
        np.testing.assert_array_equal(swe_l3_data.intensity_by_pitch_angle, sentinel.intensity_by_pitch_angle)
        np.testing.assert_array_equal(swe_l3_data.intensity_uncertainty_by_pitch_angle_and_gyrophase,
                                      sentinel.uncertainty_by_pitch_angle_and_gyrophase)
        np.testing.assert_array_equal(swe_l3_data.intensity_uncertainty_by_pitch_angle,
                                      sentinel.uncertainty_by_pitch_angle)

        # need for both moments and pitch angle
        np.testing.assert_array_equal(swe_l3_data.spacecraft_potential, expected_spacecraft_potential)
        np.testing.assert_array_equal(swe_l3_data.core_halo_breakpoint, expected_core_halo_breakpoint)

        # pitch angle specific
        self.assertEqual(sentinel.expected_phase_space_density_by_pitch_angle,
                         swe_l3_data.phase_space_density_by_pitch_angle)
        self.assertEqual(sentinel.expected_intensity, swe_l3_data.phase_space_density_1d)
        self.assertEqual(sentinel.expected_phase_space_density_inward, swe_l3_data.phase_space_density_inward)
        self.assertEqual(sentinel.expected_phase_space_density_outward, swe_l3_data.phase_space_density_outward)
        self.assertEqual(mock_moment_data, swe_l3_data.moment_data)
        self.assertEqual(sentinel.expected_phase_space_density_by_pitch_angle_and_gyrophase,
                         swe_l3_data.phase_space_density_by_pitch_angle_and_gyrophase)
        np.testing.assert_array_equal(swe_l3_data.raw_psd_by_phi_rebinned, np.full((7, 3), 28))
        np.testing.assert_array_equal(swe_l3_data.raw_1d_psd_rebinned, np.full((7,), 28))
        np.testing.assert_allclose(swe_l3_data.raw_psd_by_theta_rebinned, np.broadcast_to(psd_rebinned, (7, 7)))

        expected_quality_flags = np.array([
            SweL3Flags.FALLBACK_SWAPI_SPEED | SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE | SweL3Flags.POTENTIAL_FIT_UNCONVERGED,
            SweL3Flags.NONE,
            SweL3Flags.NONE,
            SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE,
            SweL3Flags.NONE,
            SweL3Flags.NONE,
            SweL3Flags.NONE,
        ])

        np.testing.assert_array_equal(swe_l3_data.swe_flags, expected_quality_flags)

    @patch('imap_l3_processing.swe.swe_processor.swe_rebin_intensity_by_pitch_angle_and_gyrophase')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.mec_breakpoint_finder')
    @patch('imap_l3_processing.swe.swe_processor.calculate_solar_wind_velocity_vector')
    @patch('imap_l3_processing.swe.swe_processor.correct_and_rebin')
    @patch('imap_l3_processing.swe.swe_processor.integrate_distribution_to_get_1d_spectrum')
    @patch('imap_l3_processing.swe.swe_processor.integrate_distribution_to_get_inbound_and_outbound_1d_spectrum')
    @patch('imap_l3_processing.swe.swe_processor.find_closest_neighbor')
    def test_calculate_pitch_angle_and_gyrophase_products(self, mock_find_closest_neighbor,
                                                          mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum,
                                                          mock_integrate_distribution_to_get_1d_spectrum,
                                                          mock_correct_and_rebin,
                                                          mock_calculate_solar_wind_velocity_vector,
                                                          _,
                                                          mock_average_over_look_directions,
                                                          mock_calculate_velocities, mock_swe_rebin_intensity):
        epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)
        mag_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        spacecraft_potential = np.array([12, 16, 19])
        energies = np.array([2, 4, 6])

        corrected_energy_bins = energies.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        pitch_angle_bins = [0, 90, 180]
        gyrophase_bins = [0, 360]

        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(27).reshape(3, 3, 3),
            energy=energies,
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=(np.arange(9).reshape(3, 3) + 5) * 1e6,
            phase_space_density_rebinned=np.array([])
        )

        swe_l1b_data = SweL1bData(
            epoch=epochs,
            count_rates=np.arange(27).reshape(3, 3, 3) + 10,
            settle_duration=np.array([])
        )

        mag_l1d_data = MagData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
            swp_flags=np.array([0, 0, 0, SwapiL3Flags.BAD_FIT, SwapiL3Flags.BAD_FIT, SwapiL3Flags.NONE, 0, 0, 0, 0]),
        )
        counts = swe_l1b_data.count_rates * swe_l2_data.acquisition_duration[:, :, np.newaxis] / 1e6
        mock_average_over_look_directions.return_value = np.array([5, 10, 15])
        closest_mag_data = np.arange(9).reshape(3, 3)
        closest_swapi_data = np.arange(8, 17).reshape(3, 3)
        mock_find_closest_neighbor.side_effect = [
            (closest_mag_data, sentinel.mag_best_indices),
            (closest_swapi_data, np.array([3,4,5]))
        ]

        rebinned_by_pitch_list = [
            i + np.arange(len(swe_l2_data.energy) * len(pitch_angle_bins)).reshape(len(swe_l2_data.energy),
                                                                                   len(pitch_angle_bins)) for i in
            range(len(epochs))]

        rebinned_by_pitch_and_gyrophase_list = [
            i + np.arange(len(swe_l2_data.energy) * len(pitch_angle_bins) * len(gyrophase_bins)).reshape(
                len(swe_l2_data.energy),
                len(pitch_angle_bins), len(gyrophase_bins)) for i in
            range(len(epochs))]

        rebinned_by_pitch_and_gyrophase = [(pa, pa_and_gyro) for pa, pa_and_gyro in
                                           zip(rebinned_by_pitch_list, rebinned_by_pitch_and_gyrophase_list)]
        mock_correct_and_rebin.side_effect = rebinned_by_pitch_and_gyrophase
        integrated_spectrum = np.arange(9).reshape(3, 3) + 11

        expected_intensity_by_pa_and_gyro = np.arange(9).reshape(3, 3) + 25
        expected_intensity_by_pa = np.arange(9).reshape(3, 3) + 26
        expected_uncertainty_by_pa_and_gyro = np.arange(27).reshape(3, 3, 3) + 27
        expected_uncertainty_by_pa = np.arange(27).reshape(3, 3, 3) + 28
        mock_swe_rebin_intensity.side_effect = [
            (expected_intensity_by_pa_and_gyro[0], expected_intensity_by_pa[0], expected_uncertainty_by_pa_and_gyro[0],
             expected_uncertainty_by_pa[0]),
            (expected_intensity_by_pa_and_gyro[1], expected_intensity_by_pa[1], expected_uncertainty_by_pa_and_gyro[1],
             expected_uncertainty_by_pa[1]),
            (expected_intensity_by_pa_and_gyro[2], expected_intensity_by_pa[2], expected_uncertainty_by_pa_and_gyro[2],
             expected_uncertainty_by_pa[2])]

        mock_integrate_distribution_to_get_1d_spectrum.side_effect = integrated_spectrum

        expected_inbound_spectrum = np.arange(9).reshape(3, 3) + 12
        expected_outbound_spectrum = np.arange(9).reshape(3, 3) + 13
        mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum.side_effect = [
            (expected_inbound_spectrum[0], expected_outbound_spectrum[0]),
            (expected_inbound_spectrum[1], expected_outbound_spectrum[1]),
            (expected_inbound_spectrum[2], expected_outbound_spectrum[2])]

        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_delta=[45, 45, 45],
            energy_bins=[1, 10, 100],
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            gyrophase_bins=gyrophase_bins,
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)

        actual_phase_space_density_by_pitch_angle, actual_phase_space_density_by_pa_and_gyrophase, actual_energy_spectrum, actual_energy_spectrum_inbound, actual_energy_spectrum_outbound, \
            actual_intensity_by_pa_and_gyro, actual_intensity_by_pa, actual_uncertainty_by_pa_and_gyro, actual_uncertainty_by_pa, actual_swe_flags \
            = swe_processor.calculate_pitch_angle_products(swel3_dependency, corrected_energy_bins)

        self.assertEqual(3, mock_correct_and_rebin.call_count)
        self.assertEqual(3, mock_integrate_distribution_to_get_1d_spectrum.call_count)
        mock_calculate_solar_wind_velocity_vector.assert_called_once_with(
            swel3_dependency.swapi_l3a_proton_data.proton_sw_speed,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_clock_angle,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_deflection_angle)
        mock_find_closest_neighbor.assert_has_calls([
            call(
                from_epoch=mag_epochs,
                from_data=mag_l1d_data.mag_data,
                to_epoch=swe_l2_data.acquisition_time,
                maximum_distance=np.timedelta64(1, 'm')
            ),
            call(
                from_epoch=swapi_epochs,
                from_data=mock_calculate_solar_wind_velocity_vector.return_value,
                to_epoch=epochs,
                maximum_distance=np.timedelta64(5, 'm')
            )
        ])
        expected_flags = np.array([SweL3Flags.FALLBACK_SWAPI_SPEED, SweL3Flags.FALLBACK_SWAPI_SPEED, SweL3Flags.NONE])

        np.testing.assert_array_equal(actual_phase_space_density_by_pitch_angle, rebinned_by_pitch_list)
        np.testing.assert_array_equal(actual_phase_space_density_by_pa_and_gyrophase,
                                      rebinned_by_pitch_and_gyrophase_list)
        np.testing.assert_array_equal(actual_energy_spectrum, integrated_spectrum)
        np.testing.assert_array_equal(actual_energy_spectrum_inbound, expected_inbound_spectrum)
        np.testing.assert_array_equal(actual_energy_spectrum_outbound, expected_outbound_spectrum)
        np.testing.assert_array_equal(actual_energy_spectrum_outbound, expected_outbound_spectrum)
        np.testing.assert_array_equal(actual_intensity_by_pa_and_gyro, expected_intensity_by_pa_and_gyro)
        np.testing.assert_array_equal(actual_intensity_by_pa, expected_intensity_by_pa)
        np.testing.assert_array_equal(actual_uncertainty_by_pa_and_gyro, expected_uncertainty_by_pa_and_gyro)
        np.testing.assert_array_equal(actual_uncertainty_by_pa, expected_uncertainty_by_pa)
        np.testing.assert_array_equal(actual_swe_flags, expected_flags)


        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        actual_calc_velocity_calls = mock_calculate_velocities.call_args_list
        expected_calc_velocity_calls = [
            call_with_array_matchers(swe_l2_data.energy - 12, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0]),
            call_with_array_matchers(swe_l2_data.energy - 16, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1]),
            call_with_array_matchers(swe_l2_data.energy - 19, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2])
        ]
        self.assertEqual(expected_calc_velocity_calls, actual_calc_velocity_calls)

        correct_and_rebin_actual_calls = mock_correct_and_rebin.call_args_list

        correct_and_rebin_expected_calls = [
            call_with_array_matchers(swe_l2_data.phase_space_density[0], closest_swapi_data[0],
                                     mock_calculate_velocities.return_value, closest_mag_data[0],
                                     swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[1], closest_swapi_data[1],
                                     mock_calculate_velocities.return_value, closest_mag_data[1],
                                     swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[2], closest_swapi_data[2],
                                     mock_calculate_velocities.return_value, closest_mag_data[2],
                                     swe_config)
        ]
        self.assertEqual(correct_and_rebin_expected_calls, correct_and_rebin_actual_calls)

        mock_rebin_intensity_expected_calls = [
            call_with_array_matchers(swe_l2_data.flux[0], counts[0], mock_calculate_velocities.return_value,
                                     closest_mag_data[0], swe_config),
            call_with_array_matchers(swe_l2_data.flux[1], counts[1], mock_calculate_velocities.return_value,
                                     closest_mag_data[1], swe_config),
            call_with_array_matchers(swe_l2_data.flux[2], counts[2], mock_calculate_velocities.return_value,
                                     closest_mag_data[2], swe_config)
        ]
        self.assertEqual(mock_rebin_intensity_expected_calls, mock_swe_rebin_intensity.call_args_list)

        mock_integrate_distribution_to_get_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch_list[0], swe_config),
            call(rebinned_by_pitch_list[1], swe_config),
            call(rebinned_by_pitch_list[2], swe_config)
        ])
        mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch_list[0], swe_config),
            call(rebinned_by_pitch_list[1], swe_config),
            call(rebinned_by_pitch_list[2], swe_config)
        ])

    @patch("imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products")
    def test_calculate_pitch_angle_products_makes_nan_if_no_mag_close_enough(self, _):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_epochs = np.array([
            datetime(2025, 3, 6, 0, 1, 30),
            datetime(2025, 3, 6, 0, 2, 30),
            datetime(2025, 3, 6, 0, 3, 30),
            datetime(2025, 3, 6, 0, 4, 30),
            datetime(2025, 3, 6, 0, 5, 30),
            datetime(2025, 3, 6, 0, 6, 30),
            datetime(2025, 3, 6, 0, 7, 30),
            datetime(2025, 3, 6, 0, 8, 30),
            datetime(2025, 3, 6, 0, 9, 30),
            datetime(2025, 3, 6, 0, 10, 30),
        ])
        swapi_epochs = np.array([datetime(2025, 3, 6)])

        pitch_angle_bins = [70, 100, 130]

        gyrophase_bins = [0, 180, 360]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 20,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_duration=np.full((num_epochs, num_energies, 5), 80000),
            phase_space_density_rebinned=np.arange(0,392,7).reshape(2, 4, 7)
        )

        swe_l1b_data = SweL1bData(
            epoch=epochs,
            count_rates=np.full((num_epochs, num_energies, 5, 7), 350),
            settle_duration=np.full((num_epochs, 3), 333)
        )

        mag_l1d_data = MagData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.full(len(swapi_epochs), 400),
            proton_sw_clock_angle=np.full(len(swapi_epochs), 0),
            proton_sw_deflection_angle=np.full(len(swapi_epochs), 0),
            swp_flags=np.full(len(swapi_epochs), 0)
        )
        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        energy_bins = [8, 10, 13]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_deltas=[15, 15, 15],
            gyrophase_bins=gyrophase_bins,
            gyrophase_deltas=[180, 180],
            energy_bins=energy_bins,
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
            in_vs_out_energy_index=len(energy_bins) - 1
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(ProcessingInputCollection(), input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_products(swel3_dependency)

        self.assertEqual(replace(input_metadata, descriptor="sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_deltas"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_by_pitch_angle,
                                      np.full((len(epochs), len(energy_bins), len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_1d, np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_inward,
                                      np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_outward,
                                      np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.intensity_by_pitch_angle,
                                      np.full((len(epochs), num_energies, len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.intensity_by_pitch_angle_and_gyrophase,
                                      np.full((len(epochs), num_energies, len(pitch_angle_bins), len(gyrophase_bins)),
                                              np.nan))
        np.testing.assert_array_equal(swe_l3_data.intensity_uncertainty_by_pitch_angle,
                                      np.full((len(epochs), num_energies, len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.intensity_uncertainty_by_pitch_angle_and_gyrophase,
                                      np.full((len(epochs), num_energies, len(pitch_angle_bins), len(gyrophase_bins)),
                                              np.nan))

    @patch("imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products")
    def test_calculate_pitch_angle_products_without_mocks(self, _):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_start_time = datetime(2025, 3, 6, 0, 1, 0)
        mag_epochs = np.array([mag_start_time + i * timedelta(seconds=1) for i in range(10)])
        swapi_epochs = np.array([datetime(2025, 3, 6), datetime(2025, 3, 10)])
        swp_flags = np.array([SwapiL3Flags.BAD_FIT, SwapiL3Flags.BAD_FIT])

        pitch_angle_bins = [70, 100, 130]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 5,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_duration=np.full((num_epochs, num_energies, 5), 80000),
            phase_space_density_rebinned=np.arange(0,392, 7).reshape(2, 4, 7)
        )

        swe_l1b_data = SweL1bData(
            epoch=epochs,
            count_rates=(np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                              7) + 350) * 1e6,
            settle_duration=np.full((num_epochs, 3), 333)

        )

        mag_l1d_data = MagData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.full(len(swapi_epochs), 400),
            proton_sw_clock_angle=np.full(len(swapi_epochs), 0),
            proton_sw_deflection_angle=np.full(len(swapi_epochs), 0),
            swp_flags=swp_flags
        )
        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        energy_bins = [8, 10, 13]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_deltas=[15, 15, 15],
            energy_bins=energy_bins,
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
            in_vs_out_energy_index=len(energy_bins) - 1,
            gyrophase_bins=[90, 180, 270],
            gyrophase_deltas=[45, 45, 45]
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(ProcessingInputCollection(), input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_products(swel3_dependency)

        self.assertEqual(replace(input_metadata, descriptor="sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_deltas"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        self.assertEqual(swe_l3_data.gyrophase_bins, swel3_dependency.configuration["gyrophase_bins"])
        self.assertEqual(swe_l3_data.gyrophase_delta, swel3_dependency.configuration["gyrophase_deltas"])
        np.testing.assert_allclose(swe_l3_data.phase_space_density_by_pitch_angle,
                                   np.array([[[np.nan, 249.548992, 310.137124],
                                              [np.nan, 308.858857, 363.635632],
                                              [np.nan, 394.907904, 437.401188]]]))
        np.testing.assert_allclose(swe_l3_data.epoch_delta, np.array([1807492500]))
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_allclose(swe_l3_data.phase_space_density_1d, np.array([[126.537227, 152.557743, 189.536765]]))
        np.testing.assert_allclose(swe_l3_data.phase_space_density_inward, np.array([[0., 0., 0.]]))
        np.testing.assert_allclose(swe_l3_data.phase_space_density_outward,
                                   np.array([[253.074453, 305.115485, 379.073531]]))
        self.assertEqual((1, 9, 3, 3), swe_l3_data.intensity_by_pitch_angle_and_gyrophase.shape)
        self.assertEqual((1, 9, 3), swe_l3_data.intensity_by_pitch_angle.shape)
        np.testing.assert_allclose(swe_l3_data.intensity_by_pitch_angle[0, 1, 1:3],
                                   np.array([49.5, 53. ]))
        np.testing.assert_allclose(swe_l3_data.intensity_by_pitch_angle_and_gyrophase[0, 0, 1, 2],
                                   np.array([15.]))

        np.testing.assert_allclose(swe_l3_data.intensity_uncertainty_by_pitch_angle[0, 0, 1:3],
                                   np.array([0.000717, 0.000762]), atol=1e-6)
        np.testing.assert_allclose(swe_l3_data.intensity_uncertainty_by_pitch_angle_and_gyrophase[0, 0, 1:3, 2],
                                   np.array([0.000717, 0.000762]), atol=1e-6)
        expected_swe_flags = np.array([
            SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE
            |SweL3Flags.BACKUP_SPLINE_UNRESOLVED
            |SweL3Flags.BREAKPOINT_FIT_UNCONVERGED
            |SweL3Flags.FALLBACK_SWAPI_SPEED
        ])
        np.testing.assert_array_equal(swe_l3_data.swe_flags, expected_swe_flags)

    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature_tensor_to_mag')
    @patch('imap_l3_processing.swe.swe_processor.calculate_primary_eigenvector')
    @patch('imap_l3_processing.swe.swe_processor.rotate_vector_to_rtn_spherical_coordinates')
    @patch('imap_l3_processing.swe.swe_processor.scale_halo_density')
    @patch('imap_l3_processing.swe.swe_processor.scale_core_density')
    @patch('imap_l3_processing.swe.swe_processor.integrate')
    @patch('imap_l3_processing.swe.swe_processor.halo_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.core_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    def test_calculate_moment_products(self, mock_rotate_temperature,
                                       mock_rotate_dps_vector_to_rtn,
                                       mock_calculate_velocity_in_dsp_frame_km_s,
                                       mock_compute_maxwellian_weight_factors,
                                       mock_core_fit_moments_retrying_on_failure: Mock,
                                       mock_halo_fit_moments_retrying_on_failure: Mock,
                                       mock_integrate: Mock,
                                       mock_scale_core_density: Mock,
                                       mock_scale_halo_density: Mock,
                                       mock_rotate_vector_to_rtn_spherical_coordinates,
                                       mock_calculate_primary_eigenvector,
                                       mock_rotate_temperature_tensor_to_mag):
        epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)

        instrument_elevation = np.array([-63, -42, -21, 0, 21, 42, 63])
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156, 172]),
            inst_el=instrument_elevation,
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=[1e7, 2e7, 3e7],
            phase_space_density_rebinned=np.array([]),
        )
        expected_sin_theta = np.sin(np.deg2rad(90 - instrument_elevation))
        expected_cos_theta = np.cos(np.deg2rad(90 - instrument_elevation))
        np.testing.assert_allclose(expected_cos_theta,
                                   [-0.9034, -0.6947, -0.3730, 0.0, 0.3714, 0.6896, 0.8996], atol=0.03)
        swe_l1_data = SweL1bData(epoch=epochs,
                                 count_rates=[sentinel.l1b_count_rates_1, sentinel.l1b_count_rates_2,
                                              sentinel.l1b_count_rates_3],
                                 settle_duration=Mock())

        spacecraft_potential = np.array([12, 14, 16])
        core_halo_breakpoint = np.array([96, 54, 103])

        corrected_energy_bins = swe_l2_data.energy.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        velocity_in_dsp_frame_km_s = [
            np.full(shape=(24, 30, 7, 3), fill_value=1),
            np.full(shape=(24, 30, 7, 3), fill_value=2),
            np.full(shape=(24, 30, 7, 3), fill_value=3)
        ]

        mock_calculate_velocity_in_dsp_frame_km_s.side_effect = velocity_in_dsp_frame_km_s

        maxwellian_weight_factors = [np.full(shape=(24, 30, 7), fill_value=1),
                                     np.full(shape=(24, 30, 7), fill_value=2),
                                     np.full(shape=(24, 30, 7), fill_value=3)]

        mock_compute_maxwellian_weight_factors.side_effect = maxwellian_weight_factors

        core_moments1 = create_dataclass_mock(Moments, density=15, velocity_x=5, velocity_y=6, velocity_z=7,
                                              t_parallel=1e5, t_perpendicular=1e5)
        halo_moments1 = create_dataclass_mock(Moments, density=22, velocity_x=5, velocity_y=6,
                                              velocity_z=7, t_parallel=1e5, t_perpendicular=1e5)
        core_moments2 = create_dataclass_mock(Moments, density=12, velocity_x=8, velocity_y=9, velocity_z=10,
                                              t_parallel=1e5, t_perpendicular=1e5)
        halo_moments2 = create_dataclass_mock(Moments, density=19, velocity_x=8, velocity_y=9,
                                              velocity_z=10, t_parallel=1e5, t_perpendicular=1e5)

        core_moment_fit_results_1 = MomentFitResults(moments=core_moments1, chisq=4730912.0, number_of_points=10,
                                                     regress_result=sentinel.core_regress_result_1)
        core_moment_fit_results_2 = MomentFitResults(moments=core_moments2, chisq=4705988.0, number_of_points=11,
                                                     regress_result=sentinel.core_regress_result_2)
        halo_moment_fit_results_1 = MomentFitResults(moments=halo_moments1, chisq=3412.0, number_of_points=12,
                                                     regress_result=sentinel.halo_regress_result_1)
        halo_moment_fit_results_2 = MomentFitResults(moments=halo_moments2, chisq=3214.0, number_of_points=13,
                                                     regress_result=sentinel.halo_regress_result_2)

        mock_core_fit_moments_retrying_on_failure.side_effect = [core_moment_fit_results_1, core_moment_fit_results_2,
                                                                 None]
        mock_halo_fit_moments_retrying_on_failure.side_effect = [halo_moment_fit_results_1, halo_moment_fit_results_2,
                                                                 None]

        core_rtn_theta_1, core_rtn_phi_1 = (62, 190)
        halo_rtn_theta_1, halo_rtn_phi_1 = (54, 222)
        core_rtn_theta_2, core_rtn_phi_2 = (63, 191)
        halo_rtn_theta_2, halo_rtn_phi_2 = (55, 223)
        mock_rotate_temperature.side_effect = [
            np.deg2rad([62, 190]),
            np.deg2rad([54, 222]),
            np.deg2rad([63, 191]),
            np.deg2rad([55, 223]),
        ]

        core_fit_velocity_1 = [0, 0, 1]
        core_fit_velocity_2 = [0, 1, 1]
        halo_fit_velocity_1 = [0, 1, 0]
        halo_fit_velocity_2 = [1, 0, 0]
        core_integrated_velocity_rtn = [2, 3, 4]
        total_integrated_velocity_rtn = [5, 8, 9]
        halo_integrated_velocity_rtn = [6, 0, 6]
        mock_rotate_dps_vector_to_rtn.side_effect = [
            core_fit_velocity_1,
            core_integrated_velocity_rtn,
            total_integrated_velocity_rtn,
            halo_fit_velocity_1,
            halo_integrated_velocity_rtn,
            core_fit_velocity_2,
            halo_fit_velocity_2
        ]

        core_integrate_output = Mock()
        total_integrate_output = Mock()
        total_integrate_output.density = 5
        total_temperature = [101000, 102000, 103000, 104000, 105000, 106000]
        total_integrate_output.temperature = total_temperature
        halo_integrate_output = Mock()

        mock_integrate.side_effect = [core_integrate_output, total_integrate_output, halo_integrate_output, None, None]

        scaled_core_velocity = [900, 800, 700]
        scaled_core_temperature = [90000, 80000, 70000, 80000, 90000, 100000]
        core_cdelnv = Mock()
        core_cdelt = Mock()
        scale_core_density_output = ScaleDensityOutput(density=400, velocity=scaled_core_velocity,
                                                       temperature=scaled_core_temperature,
                                                       cdelnv=core_cdelnv,
                                                       cdelt=core_cdelt)

        mock_scale_core_density.side_effect = [scale_core_density_output]

        scaled_halo_velocity = [2000, 1800, 1700]
        scaled_halo_temperature = [180000, 190000, 200000, 201000, 202000, 203000]
        scale_halo_density_output = ScaleDensityOutput(density=500, velocity=scaled_halo_velocity,
                                                       temperature=scaled_halo_temperature,
                                                       cdelnv=None,
                                                       cdelt=None)

        mock_scale_halo_density.side_effect = [scale_halo_density_output]

        mock_rotate_vector_to_rtn_spherical_coordinates.side_effect = [
            (10, np.deg2rad(11), np.deg2rad(12)),
            (16, np.deg2rad(17), np.deg2rad(18)),
            (101, np.deg2rad(102), np.deg2rad(103)),
            (104, np.deg2rad(105), np.deg2rad(106)),
            (13, np.deg2rad(14), np.deg2rad(15)),
            (19, np.deg2rad(20), np.deg2rad(21))]

        core_primary_evec = Mock()
        total_primary_evec = Mock()
        halo_primary_evec = Mock()
        t_par_1 = 100001
        t_par_2 = 100002
        t_par_3 = 100003
        t_perp_1 = 200001
        t_perp_2 = 200002
        t_perp_3 = 200003
        gyro_1 = 1.1
        gyro_2 = 1.2
        gyro_3 = 1.3
        mock_calculate_primary_eigenvector.side_effect = [
            (core_primary_evec, np.array([t_par_1, t_perp_1, gyro_1])),
            (total_primary_evec, np.array([t_par_3, t_perp_3, gyro_3])),
            (halo_primary_evec, np.array([t_par_2, t_perp_2, gyro_2])),
        ]

        mock_rotate_temperature_tensor_to_mag.side_effect = [(11.4, 12.4, 13.4), (14.4, 15.4, 16.4), (17.4, 18.4, 19.4)]

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        config = build_swe_configuration()

        rebinned_mag_data = [sentinel.mag_data_1, sentinel.mag_data_2, sentinel.mag_data_3]

        swe_moment_data = swe_processor.calculate_moment_products(swe_l2_data, swe_l1_data, rebinned_mag_data,
                                                                  spacecraft_potential,
                                                                  core_halo_breakpoint,
                                                                  corrected_energy_bins,
                                                                  config)

        calculate_velocity_call_1 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[0]
        np.testing.assert_array_equal(corrected_energy_bins[0], calculate_velocity_call_1.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[0], calculate_velocity_call_1.args[2])

        calculate_velocity_call_2 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1], calculate_velocity_call_2.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[1], calculate_velocity_call_2.args[2])

        calculate_velocity_call_3 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2], calculate_velocity_call_3.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[2], calculate_velocity_call_3.args[2])

        self.assertEqual(3, mock_compute_maxwellian_weight_factors.call_count)
        mock_compute_maxwellian_weight_factors.assert_has_calls(
            [call(sentinel.l1b_count_rates_1, 10.0),
             call(sentinel.l1b_count_rates_2, 20.0),
             call(sentinel.l1b_count_rates_3, 30.0),
             ])

        self.assertEqual(3, mock_core_fit_moments_retrying_on_failure.call_count)
        self.assertEqual(3, mock_halo_fit_moments_retrying_on_failure.call_count)

        core_fit_moments_call_1 = mock_core_fit_moments_retrying_on_failure.mock_calls[0]
        np.testing.assert_array_equal(core_fit_moments_call_1.args[0], corrected_energy_bins[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0] * 100000,
                                      core_fit_moments_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], core_fit_moments_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      core_fit_moments_call_1.args[3])
        self.assertEqual(3, core_fit_moments_call_1.args[4])
        self.assertEqual(7, core_fit_moments_call_1.args[5])
        np.testing.assert_array_equal(core_fit_moments_call_1.args[6], [100, 100, 100])

        halo_fit_moments_call_1 = mock_halo_fit_moments_retrying_on_failure.mock_calls[0]
        np.testing.assert_array_equal(corrected_energy_bins[0],
                                      halo_fit_moments_call_1.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0] * 100000,
                                      halo_fit_moments_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], halo_fit_moments_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      halo_fit_moments_call_1.args[3])
        self.assertEqual(6, halo_fit_moments_call_1.args[4])
        self.assertEqual(11, halo_fit_moments_call_1.args[5])
        np.testing.assert_array_equal(halo_fit_moments_call_1.args[6], [25, 25, 25])
        self.assertEqual(spacecraft_potential[0], halo_fit_moments_call_1.args[7])
        self.assertEqual(core_halo_breakpoint[0], halo_fit_moments_call_1.args[8])

        core_fit_moments_2 = mock_core_fit_moments_retrying_on_failure.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1],
                                      core_fit_moments_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1] * 100000,
                                      core_fit_moments_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], core_fit_moments_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      core_fit_moments_2.args[3])
        self.assertEqual(4, core_fit_moments_2.args[4])
        self.assertEqual(6, core_fit_moments_2.args[5])
        np.testing.assert_array_equal(core_fit_moments_2.args[6], [100, 100, core_moments1.density])

        halo_fit_moments_2 = mock_halo_fit_moments_retrying_on_failure.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1],
                                      halo_fit_moments_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1] * 100000,
                                      halo_fit_moments_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], halo_fit_moments_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      halo_fit_moments_2.args[3])
        self.assertEqual(5, halo_fit_moments_2.args[4])
        self.assertEqual(10, halo_fit_moments_2.args[5])
        np.testing.assert_array_equal(halo_fit_moments_2.args[6], [25, 25, halo_moments1.density])
        self.assertEqual(spacecraft_potential[1], halo_fit_moments_2.args[7])
        self.assertEqual(core_halo_breakpoint[1], halo_fit_moments_2.args[8])

        core_fit_moments_3 = mock_core_fit_moments_retrying_on_failure.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2],
                                      core_fit_moments_3.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[2] * 100000,
                                      core_fit_moments_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[2], core_fit_moments_3.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[2],
                                      core_fit_moments_3.args[3])
        self.assertEqual(5, core_fit_moments_3.args[4])
        self.assertEqual(9, core_fit_moments_3.args[5])
        np.testing.assert_array_equal(core_fit_moments_3.args[6], [100, core_moments1.density, core_moments2.density])

        halo_fit_moments_3 = mock_halo_fit_moments_retrying_on_failure.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2],
                                      halo_fit_moments_3.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[2] * 100000,
                                      halo_fit_moments_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[2], halo_fit_moments_3.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[2],
                                      halo_fit_moments_3.args[3])
        self.assertEqual(8, halo_fit_moments_3.args[4])
        self.assertEqual(11, halo_fit_moments_3.args[5])
        np.testing.assert_array_equal(halo_fit_moments_3.args[6], [25, halo_moments1.density, halo_moments2.density])
        self.assertEqual(spacecraft_potential[2], halo_fit_moments_3.args[7])
        self.assertEqual(core_halo_breakpoint[2], halo_fit_moments_3.args[8])

        self.assertEqual(7, mock_rotate_dps_vector_to_rtn.call_count)

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[0].args[0])
        np.testing.assert_array_equal(
            np.array([core_moments1.velocity_x, core_moments1.velocity_y, core_moments1.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[0].args[1])
        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[1].args[0])
        np.testing.assert_array_equal(
            scaled_core_velocity,
            mock_rotate_dps_vector_to_rtn.call_args_list[1].args[1])

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[2].args[0])
        np.testing.assert_array_equal(
            total_integrate_output.velocity,
            mock_rotate_dps_vector_to_rtn.call_args_list[2].args[1])

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[3].args[0])
        np.testing.assert_array_equal(
            np.array([halo_moments1.velocity_x, halo_moments1.velocity_y, halo_moments1.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[3].args[1])

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[4].args[0])
        np.testing.assert_array_equal(
            scaled_halo_velocity,
            mock_rotate_dps_vector_to_rtn.call_args_list[4].args[1])

        self.assertEqual(epochs[1], mock_rotate_dps_vector_to_rtn.call_args_list[5].args[0])
        np.testing.assert_array_equal(
            np.array([core_moments2.velocity_x, core_moments2.velocity_y, core_moments2.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[5].args[1])

        self.assertEqual(epochs[1], mock_rotate_dps_vector_to_rtn.call_args_list[6].args[0])
        np.testing.assert_array_equal(
            np.array([halo_moments2.velocity_x, halo_moments2.velocity_y, halo_moments2.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[6].args[1])

        self.assertEqual(4, mock_rotate_temperature.call_count)
        mock_rotate_temperature.assert_has_calls(
            [call(epochs[0], core_moments1.alpha, core_moments1.beta),
             call(epochs[0], halo_moments1.alpha, halo_moments1.beta),
             call(epochs[1], core_moments2.alpha, core_moments2.beta),
             call(epochs[1], halo_moments2.alpha, halo_moments2.beta), ])

        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        self.assertEqual(5, mock_integrate.call_count)
        mock_integrate.assert_has_calls(
            [call_with_array_matchers(4, 5,
                                      corrected_energy_bins[0], expected_sin_theta, expected_cos_theta,
                                      config["aperture_field_of_view_radians"],
                                      swe_l2_data.phase_space_density[0],
                                      swe_l2_data.inst_az_spin_sector[0],
                                      12, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])),
             call_with_array_matchers(4, 5,
                                      corrected_energy_bins[0], expected_sin_theta, expected_cos_theta,
                                      config["aperture_field_of_view_radians"],
                                      swe_l2_data.phase_space_density[0],
                                      swe_l2_data.inst_az_spin_sector[0],
                                      12, core_cdelnv, core_cdelt),
             call_with_array_matchers(6, 5,
                                      corrected_energy_bins[0], expected_sin_theta, expected_cos_theta,
                                      config["aperture_field_of_view_radians"],
                                      swe_l2_data.phase_space_density[0],
                                      swe_l2_data.inst_az_spin_sector[0],
                                      12, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])),
             call_with_array_matchers(5, 4,
                                      corrected_energy_bins[1], expected_sin_theta, expected_cos_theta,
                                      config["aperture_field_of_view_radians"],
                                      swe_l2_data.phase_space_density[1],
                                      swe_l2_data.inst_az_spin_sector[1],
                                      14, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])),
             call_with_array_matchers(5, 5,
                                      corrected_energy_bins[1], expected_sin_theta, expected_cos_theta,
                                      config["aperture_field_of_view_radians"],
                                      swe_l2_data.phase_space_density[1],
                                      swe_l2_data.inst_az_spin_sector[1],
                                      14, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])),

             ]
        )

        self.assertEqual(1, mock_scale_core_density.call_count)
        mock_scale_core_density.assert_has_calls([
            call_with_array_matchers(core_integrate_output.density, core_integrate_output.velocity,
                                     core_integrate_output.temperature, core_moments1, 3, corrected_energy_bins[0],
                                     spacecraft_potential[0], expected_cos_theta,
                                     config["aperture_field_of_view_radians"],
                                     swe_l2_data.inst_az_spin_sector[0],
                                     sentinel.core_regress_result_1,
                                     core_integrate_output.base_energy)
        ])

        self.assertEqual(1, mock_scale_halo_density.call_count)

        mock_scale_halo_density.assert_has_calls([
            call_with_array_matchers(halo_integrate_output.density, halo_integrate_output.velocity,
                                     halo_integrate_output.temperature, halo_moments1, spacecraft_potential[0],
                                     core_halo_breakpoint[0], expected_cos_theta,
                                     config["aperture_field_of_view_radians"],
                                     swe_l2_data.inst_az_spin_sector[0], sentinel.halo_regress_result_1,
                                     halo_integrate_output.base_energy)
        ])

        self.assertEqual(6, mock_rotate_vector_to_rtn_spherical_coordinates.call_count)

        mock_rotate_vector_to_rtn_spherical_coordinates.assert_has_calls([
            call(epochs[0], core_integrate_output.heat_flux),
            call(epochs[0], core_primary_evec),
            call(epochs[0], total_integrate_output.heat_flux),
            call(epochs[0], total_primary_evec),
            call(epochs[0], halo_integrate_output.heat_flux),
            call(epochs[0], halo_primary_evec),
        ])

        self.assertEqual(3, mock_calculate_primary_eigenvector.call_count)
        mock_calculate_primary_eigenvector.assert_has_calls([
            call(scaled_core_temperature),
            call(total_integrate_output.temperature),
            call(scaled_halo_temperature),
        ])

        self.assertEqual(3, mock_rotate_temperature_tensor_to_mag.call_count)

        mock_rotate_temperature_tensor_to_mag.assert_has_calls([
            call(scale_core_density_output.temperature, sentinel.mag_data_1),
            call(total_integrate_output.temperature, sentinel.mag_data_1),
            call(scale_halo_density_output.temperature, sentinel.mag_data_1),
        ])

        # @formatter:off
        self.assertIsInstance(swe_moment_data, SweL3MomentData)
        np.testing.assert_array_equal(swe_moment_data.core_fit_num_points, [core_moment_fit_results_1.number_of_points,
                                                                            core_moment_fit_results_2.number_of_points,
                                                                            np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_chisq,
                                      [core_moment_fit_results_1.chisq, core_moment_fit_results_2.chisq, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_chisq,
                                      [halo_moment_fit_results_1.chisq, halo_moment_fit_results_2.chisq, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_density_fit,
                                      [core_moments1.density, core_moments2.density, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_density_fit,
                                      [halo_moments1.density, halo_moments2.density, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_t_parallel_fit,
                                      [core_moments1.t_parallel, core_moments2.t_parallel, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_t_parallel_fit,
                                      [halo_moments1.t_parallel, halo_moments2.t_parallel, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_t_perpendicular_fit,
                                      [core_moments1.t_perpendicular, core_moments2.t_perpendicular, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_t_perpendicular_fit,
                                      [halo_moments1.t_perpendicular, halo_moments2.t_perpendicular, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_temperature_phi_rtn_fit,
                                      [core_rtn_phi_1, core_rtn_phi_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_phi_rtn_fit,
                                      [halo_rtn_phi_1, halo_rtn_phi_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_temperature_theta_rtn_fit,
                                      [core_rtn_theta_1, core_rtn_theta_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_theta_rtn_fit,
                                      [halo_rtn_theta_1, halo_rtn_theta_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_speed_fit,
                                      [np.linalg.norm(core_fit_velocity_1), np.linalg.norm(core_fit_velocity_2),
                                       np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_speed_fit,
                                      [np.linalg.norm(halo_fit_velocity_1), np.linalg.norm(halo_fit_velocity_2),
                                       np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_velocity_vector_rtn_fit,
                                      [core_fit_velocity_1, core_fit_velocity_2, [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.halo_velocity_vector_rtn_fit,
                                      [halo_fit_velocity_1, halo_fit_velocity_2, [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.core_density_integrated,
                                      [scale_core_density_output.density, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_density_integrated,
                                      [scale_halo_density_output.density, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.total_density_integrated,
                                      [total_integrate_output.density, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_speed_integrated,
                                      [np.linalg.norm(core_integrated_velocity_rtn), np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_speed_integrated,
                                      [np.linalg.norm(halo_integrated_velocity_rtn), np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.total_speed_integrated,
                                      [np.linalg.norm(total_integrated_velocity_rtn), np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_velocity_vector_rtn_integrated,
                                      [core_integrated_velocity_rtn, [np.nan, np.nan, np.nan],
                                       [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.halo_velocity_vector_rtn_integrated,
                                      [halo_integrated_velocity_rtn, [np.nan, np.nan, np.nan],
                                       [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.total_velocity_vector_rtn_integrated,
                                      [total_integrated_velocity_rtn, [np.nan, np.nan, np.nan],
                                       [np.nan, np.nan, np.nan]])

        np.testing.assert_allclose(swe_moment_data.core_heat_flux_magnitude_integrated, [10, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.core_heat_flux_theta_integrated, [11, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.core_heat_flux_phi_integrated, [12, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.halo_heat_flux_magnitude_integrated, [13, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.halo_heat_flux_theta_integrated, [14, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.halo_heat_flux_phi_integrated, [15, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.total_heat_flux_magnitude_integrated, [101, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.total_heat_flux_theta_integrated, [102, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.total_heat_flux_phi_integrated, [103, np.nan, np.nan])

        np.testing.assert_array_equal(swe_moment_data.core_t_parallel_integrated, [t_par_1, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_t_perpendicular_integrated,
                                      [[t_perp_1, gyro_1], [np.nan, np.nan], [np.nan, np.nan]])

        np.testing.assert_array_equal(swe_moment_data.halo_t_parallel_integrated, [t_par_2, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_t_perpendicular_integrated,
                                      [[t_perp_2, gyro_2], [np.nan, np.nan], [np.nan, np.nan]])

        np.testing.assert_array_equal(swe_moment_data.total_t_parallel_integrated, [t_par_3, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.total_t_perpendicular_integrated,
                                      [[t_perp_3, gyro_3], [np.nan, np.nan], [np.nan, np.nan]])
        np.testing.assert_allclose(swe_moment_data.core_temperature_theta_rtn_integrated, [17, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.core_temperature_phi_rtn_integrated, [18, np.nan, np.nan])

        np.testing.assert_allclose(swe_moment_data.halo_temperature_theta_rtn_integrated, [20, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.halo_temperature_phi_rtn_integrated, [21, np.nan, np.nan])

        np.testing.assert_allclose(swe_moment_data.total_temperature_theta_rtn_integrated, [105, np.nan, np.nan])
        np.testing.assert_allclose(swe_moment_data.total_temperature_phi_rtn_integrated, [106, np.nan, np.nan])

        np.testing.assert_array_equal(swe_moment_data.core_temperature_parallel_to_mag, [11.4, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_temperature_perpendicular_to_mag,
                                      [[12.4, 13.4], [np.nan, np.nan], [np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_parallel_to_mag, [17.4, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_perpendicular_to_mag,
                                      [[18.4, 19.4], [np.nan, np.nan], [np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.total_temperature_parallel_to_mag, [14.4, np.nan, np.nan])
        np.testing.assert_array_equal(swe_moment_data.total_temperature_perpendicular_to_mag,
                                      [[15.4, 16.4], [np.nan, np.nan], [np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.total_temperature_perpendicular_to_mag,
                                      [[15.4, 16.4], [np.nan, np.nan], [np.nan, np.nan]])

        np.testing.assert_array_equal(swe_moment_data.core_temperature_tensor_integrated,
                                      [scaled_core_temperature, np.full(6, np.nan), np.full(6, np.nan)])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_tensor_integrated,
                                      [scaled_halo_temperature, np.full(6, np.nan), np.full(6, np.nan)])
        np.testing.assert_array_equal(swe_moment_data.total_temperature_tensor_integrated,
                                      [total_temperature, np.full(6, np.nan), np.full(6, np.nan)])

        # @formatter:on

    @patch('imap_l3_processing.swe.swe_processor.rotate_vector_to_rtn_spherical_coordinates')
    @patch('imap_l3_processing.swe.swe_processor.scale_halo_density')
    @patch('imap_l3_processing.swe.swe_processor.scale_core_density')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.integrate')
    @patch('imap_l3_processing.swe.swe_processor.halo_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.core_fit_moments_retrying_on_failure')
    def test_calculate_moment_products_only_integrates_temperatures_in_a_range(self,
                                                                               mock_core_fit_moments_retrying_on_failure,
                                                                               mock_halo_fit_moments_retrying_on_failure,
                                                                               mock_integrate,
                                                                               _, mock_rotate_dps_vector_to_rtn,
                                                                               mock_rotate_temperature,
                                                                               mock_scale_core_density,
                                                                               mock_scale_halo_density,
                                                                               mock_rotate_vector_to_rtn_spherical_coordinates):
        mock_core_fit_moments_retrying_on_failure.side_effect = [
            build_moment_fit_results(moments=build_moments(t_parallel=1e3 - 1, t_perpendicular=1e3 - 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e3 + 1, t_perpendicular=1e3 + 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e7 - 1, t_perpendicular=1e7 - 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e7 + 1, t_perpendicular=1e7 + 1)),
        ]

        mock_halo_fit_moments_retrying_on_failure.side_effect = [
            build_moment_fit_results(moments=build_moments(t_parallel=1e4 - 1, t_perpendicular=1e4 - 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e4 + 1, t_perpendicular=1e4 + 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e8 - 1, t_perpendicular=1e8 - 1)),
            build_moment_fit_results(moments=build_moments(t_parallel=1e8 + 1, t_perpendicular=1e8 + 1)),
        ]

        mock_rotate_temperature.return_value = ((4, 5))
        mock_rotate_vector_to_rtn_spherical_coordinates.return_value = ((1, 2, 3))

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        config = build_swe_configuration()
        epochs = datetime.now() + np.arange(4) * timedelta(minutes=1)

        instrument_elevation = np.array([-70, -50, -30, 0, 30, 50, 70])
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(16).reshape(4, 4) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156]),
            inst_el=instrument_elevation,
            inst_az_spin_sector=np.arange(16, 32).reshape(4, 4),
            acquisition_time=np.array([]),
            acquisition_duration=[2e7, 2e7, 2e7, 2e7],
            phase_space_density_rebinned=np.array([])
        )
        swe_l1_data = SweL1bData(epoch=epochs,
                                 count_rates=[Mock(), Mock(), Mock(), Mock()],
                                 settle_duration=Mock())

        spacecraft_potential = np.array([12, 14, 16, 18])
        core_halo_breakpoint = np.array([96, 54, 103, 124])

        corrected_energy_bins = swe_l2_data.energy.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        mock_integrate.return_value.temperature = np.full(6, 0)

        mock_scale_core_density.return_value.density = 1
        mock_scale_core_density.return_value.temperature = np.full(6, 1)
        mock_scale_halo_density.return_value.density = 1
        mock_scale_halo_density.return_value.temperature = np.full(6, 1)

        mock_rotate_dps_vector_to_rtn.return_value = (2, 3, 4)

        rebinned_mag_data = np.full((3, 3), 1)
        swe_moment_data = swe_processor.calculate_moment_products(swe_l2_data, swe_l1_data, rebinned_mag_data,
                                                                  spacecraft_potential,
                                                                  core_halo_breakpoint,
                                                                  corrected_energy_bins,
                                                                  config)

        self.assertEqual(6, mock_integrate.call_count)
        self.assertEqual(2, mock_scale_core_density.call_count)
        self.assertEqual(2, mock_scale_halo_density.call_count)

        np.testing.assert_array_equal([np.nan, 1, 1, np.nan], swe_moment_data.core_density_integrated)
        self.assertEqual(4, len(swe_moment_data.core_velocity_vector_rtn_integrated))
        np.testing.assert_array_equal([np.nan, np.nan, np.nan], swe_moment_data.core_velocity_vector_rtn_integrated[0])
        np.testing.assert_array_equal([np.nan, np.nan, np.nan], swe_moment_data.core_velocity_vector_rtn_integrated[3])

        np.testing.assert_array_equal([np.nan, 1, 1, np.nan], swe_moment_data.halo_density_integrated)
        self.assertEqual(4, len(swe_moment_data.halo_velocity_vector_rtn_integrated))

    def test_calculate_moment_products_handles_bad_fit_indices_and_continues(self):
        epochs = np.array([datetime.now()])

        instrument_elevation = np.array([-63, -42, -21, 0, 21, 42, 63])
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156, 172]),
            inst_el=instrument_elevation,
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=np.full((1, 11, 3), 1e7),
            phase_space_density_rebinned=np.array([])
        )
        swe_l1_data = SweL1bData(epoch=epochs,
                                 count_rates=np.full((1, 11, 3, 7), 10.5),
                                 settle_duration=Mock())
        spacecraft_potential = np.array([12, 14])
        core_halo_breakpoint = np.array([12, 54])
        corrected_energy_bins = swe_l2_data.energy.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swe_processor = SweProcessor(ProcessingInputCollection(), input_metadata=input_metadata)
        config = build_swe_configuration()

        rebinned_mag_data = [sentinel.mag_data_1]

        with self.assertLogs(logger, level='INFO') as log_context:
            swe_moment_data = swe_processor.calculate_moment_products(swe_l2_data, swe_l1_data, rebinned_mag_data,
                                                                      spacecraft_potential,
                                                                      core_halo_breakpoint,
                                                                      corrected_energy_bins,
                                                                      config)

            self.assertIsInstance(swe_moment_data, SweL3MomentData)
            self.assertEqual(log_context.output,
                             [
                                 "INFO:imap_l3_processing.swe.swe_processor:Bad core-halo breakpoint value at index 0. Continuing."])

    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature_tensor_to_mag')
    @patch('imap_l3_processing.swe.swe_processor.calculate_primary_eigenvector')
    @patch('imap_l3_processing.swe.swe_processor.rotate_vector_to_rtn_spherical_coordinates')
    @patch('imap_l3_processing.swe.swe_processor.scale_halo_density')
    @patch('imap_l3_processing.swe.swe_processor.scale_core_density')
    @patch('imap_l3_processing.swe.swe_processor.integrate')
    @patch('imap_l3_processing.swe.swe_processor.halo_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.core_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    def test_calculate_moment_products_handles_errors_with_fill(self, mock_rotate_temperature,
                                                                mock_rotate_dps_vector_to_rtn,
                                                                mock_calculate_velocity_in_dsp_frame_km_s,
                                                                mock_compute_maxwellian_weight_factors,
                                                                mock_core_fit_moments_retrying_on_failure: Mock,
                                                                mock_halo_fit_moments_retrying_on_failure: Mock,
                                                                mock_integrate: Mock,
                                                                mock_scale_core_density: Mock,
                                                                mock_scale_halo_density: Mock,
                                                                mock_rotate_vector_to_rtn_spherical_coordinates,
                                                                mock_calculate_primary_eigenvector,
                                                                mock_rotate_temperature_tensor_to_mag):
        mock_rotate_dps_vector_to_rtn.return_value = np.full(3, np.nan)
        mock_rotate_temperature.return_value = 1, 2
        mock_core_fit_moments_retrying_on_failure.return_value = MomentFitResults(
            moments=Moments(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            chisq=2.5,
            number_of_points=100,
            regress_result=np.array([1, 2, 3, 4])
        )
        cases = [
            mock_rotate_temperature,
            mock_rotate_dps_vector_to_rtn,
            mock_calculate_velocity_in_dsp_frame_km_s,
            mock_compute_maxwellian_weight_factors,
            mock_core_fit_moments_retrying_on_failure,
            mock_halo_fit_moments_retrying_on_failure,
            mock_integrate,
            mock_scale_core_density,
            mock_scale_halo_density,
            mock_rotate_vector_to_rtn_spherical_coordinates,
            mock_calculate_primary_eigenvector,
            mock_rotate_temperature_tensor_to_mag
        ]

        for failing_function in cases:
            with self.subTest(failing_function):
                failing_function.side_effect = Exception

                epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)

                instrument_elevation = np.array([-63, -42, -21, 0, 21, 42, 63])
                swe_l2_data = SweL2Data(
                    epoch=epochs,
                    phase_space_density=np.arange(9).reshape(3, 3) + 100,
                    flux=np.arange(9).reshape(3, 3),
                    energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156, 172]),
                    inst_el=instrument_elevation,
                    inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
                    acquisition_time=np.array([]),
                    acquisition_duration=[1e7, 2e7, 3e7],
                    phase_space_density_rebinned=np.array([])
                )
                swe_l1_data = SweL1bData(epoch=epochs,
                                         count_rates=[sentinel.l1b_count_rates_1, sentinel.l1b_count_rates_2,
                                                      sentinel.l1b_count_rates_3],
                                         settle_duration=Mock())

                spacecraft_potential = np.array([12, 14, 16])
                core_halo_breakpoint = np.array([96, 54, 103])

                corrected_energy_bins = swe_l2_data.energy.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

                input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                               datetime(2025, 2, 22), "v001")

                swe_processor = SweProcessor(ProcessingInputCollection(), input_metadata=input_metadata)
                config = build_swe_configuration()

                rebinned_mag_data = [sentinel.mag_data_1, sentinel.mag_data_2, sentinel.mag_data_3]

                with self.assertLogs():
                    swe_moment_data = swe_processor.calculate_moment_products(swe_l2_data, swe_l1_data,
                                                                              rebinned_mag_data,
                                                                              spacecraft_potential,
                                                                              core_halo_breakpoint,
                                                                              corrected_energy_bins,
                                                                              config)

                self.assertIsInstance(swe_moment_data, SweL3MomentData)

                failing_function.side_effect = None
