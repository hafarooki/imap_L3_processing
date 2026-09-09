from dataclasses import replace
from datetime import datetime, timedelta, date
from unittest import TestCase
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
from imap_data_access import config
from imap_data_access.file_validation import Version
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, \
    FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.swapi.descriptors import DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, \
    GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR, \
    HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, HELIUM_INFLOW_VECTOR_DESCRIPTOR, \
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR, CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, \
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SWAPI_L2_DESCRIPTOR, SwapiL3ADependencies
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import DeltaMinusPlus
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.species import Species
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from tests.test_helpers import create_mock_version_map


class TestSwapiProcessor(TestCase):
    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3PickupIonData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.PuiChunkFitter')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_pui(self, mock_spicepy,
                             mock_swapi_l3_dependencies_class,
                             mock_parallel_chunk_runner_class,
                             mock_pui_chunk_fitter_class,
                             mock_chunk_l2_data, mock_write_cdf,
                             mock_pickup_ion_data_constructor, mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 9, 26)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 9, 25)
        descriptor = "pui-he"
        input_version = VersionMap({descriptor: Version(None, 123)})
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        returned_chunk_epoch = 10 + THIRTY_SECONDS_IN_NANOSECONDS
        returned_velocity_rtn_sc = np.array([370.0, 10.0, 5.0])

        proton_runner_result = dict(
            epoch=np.array([returned_chunk_epoch]),
            proton_sw_velocity_rtn=np.array([returned_velocity_rtn_sc]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        epoch_for_fifty_sweeps = np.arange(initial_epoch, initial_epoch + 50)
        energy_1d = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        chunk_of_five = SwapiL2Data(epoch, energy_1d, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        energy_fifty = np.tile(energy_1d * 2, (50, 1))
        coincidence_count_rate_fifty = np.full((50, 5), 7.0)
        coincidence_count_rate_uncertainty_fifty = np.full((50, 5), 0.2)
        chunk_of_fifty = SwapiL2Data(epoch_for_fifty_sweeps, energy_fifty,
                                     coincidence_count_rate_fifty,
                                     coincidence_count_rate_uncertainty_fifty)

        pui_runner_result = dict(
            epoch=np.array([initial_epoch + FIVE_MINUTES_IN_NANOSECONDS]),
            cooling_index=np.array([1]),
            ionization_rate=np.array([2]),
            cutoff_speed=np.array([3]),
            background_rate=np.array([4]),
            density=np.array([5]),
            temperature=np.array([6]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )
        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.side_effect = [proton_runner_result, pui_runner_result]

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        pickup_ion_data = mock_pickup_ion_data_constructor.return_value
        expected_pickup_ion_metadata = replace(input_metadata, descriptor=descriptor)
        pickup_ion_data.input_metadata = expected_pickup_ion_metadata

        input_metadata.descriptor = descriptor

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "09" /
                             f"imap_swapi_l3a_pui-he_{start_date_as_str}_v123.cdf")

        mock_chunk_l2_data.side_effect = [
            [chunk_of_five],
            [chunk_of_fifty],
        ]

        mock_l3a_dependencies = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value
        mock_l3a_dependencies.data = Mock(energy=energy_1d)

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        actual_science_input = swapi_processor.dependencies.get_science_inputs()[0]
        self.assertEqual(actual_science_input.get_time_range()[0].strftime("%Y%m%d"), dependency_start_date)

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_swapi_response = mock_l3a_dependencies.swapi_response
        mock_efficiency_lut = mock_l3a_dependencies.efficiency_calibration_table
        mock_density_of_neutral_helium_calibration_table = mock_l3a_dependencies.density_of_neutral_helium_calibration_table
        mock_hydrogen_inflow_vector = mock_l3a_dependencies.hydrogen_inflow_vector
        mock_helium_inflow_vector = mock_l3a_dependencies.helium_inflow_vector

        mock_chunk_l2_data.assert_has_calls([call(mock_l3a_dependencies.data, 5),
                                             call(mock_l3a_dependencies.data, 50)])

        mock_swapi_response.warm_cache.assert_called_once()
        mock_parallel_chunk_runner_class.assert_called_once_with(
            mock_swapi_response, mock_efficiency_lut,
        )
        self.assertEqual(mock_runner.run.call_count, 2)

        first_chunks, first_fitter = mock_runner.run.call_args_list[0].args
        self.assertEqual([chunk_of_five], first_chunks)
        self.assertEqual(first_fitter.__class__.__name__, "ProtonChunkFitter")

        second_chunks, second_fitter = mock_runner.run.call_args_list[1].args
        self.assertEqual([chunk_of_fifty], second_chunks)
        self.assertIs(second_fitter, mock_pui_chunk_fitter_class.return_value)

        pui_fitter_kwargs = mock_pui_chunk_fitter_class.call_args.kwargs
        self.assertIs(
            pui_fitter_kwargs["density_of_neutral_helium_lookup_table"],
            mock_density_of_neutral_helium_calibration_table,
        )
        self.assertIs(pui_fitter_kwargs["hydrogen_inflow_vector"], mock_hydrogen_inflow_vector)
        self.assertIs(pui_fitter_kwargs["helium_inflow_vector"], mock_helium_inflow_vector)
        self.assertIs(pui_fitter_kwargs["proton_results"], proton_runner_result)

        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_pui-he"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_pui-he_{start_date_as_str}_v123"),
                                                            ])

        actual_positional = mock_pickup_ion_data_constructor.call_args.args
        actual_kwargs = mock_pickup_ion_data_constructor.call_args.kwargs
        self.assertEqual(expected_pickup_ion_metadata, actual_positional[0])
        for key, expected_val in pui_runner_result.items():
            np.testing.assert_array_equal(expected_val, actual_kwargs[key])

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", descriptor)

        self.assertEqual(input_file_names, pickup_ion_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), pickup_ion_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_proton(self, mock_spicepy,
                                mock_swapi_l3_dependencies_class,
                                mock_chunk_l2_data,
                                mock_parallel_chunk_runner_class,
                                mock_write_cdf,
                                mock_proton_solar_wind_data_constructor,
                                mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 6, 13)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 6, 12)
        descriptor = "proton-sw"
        input_version = VersionMap({descriptor: Version(None, 123)})
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        runner_result = dict(
            epoch=np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
            proton_sw_speed=np.array([400000.0]),
            proton_sw_speed_uncert=np.array([2.0]),
            proton_sw_speed_sun=np.array([401000.0]),
            proton_sw_speed_sun_uncert=np.array([2.5]),
            proton_sw_temperature=np.array([99000.0]),
            proton_sw_temperature_uncert=np.array([1000.0]),
            proton_sw_density=np.array([4.97]),
            proton_sw_density_uncert=np.array([0.25]),
            proton_sw_clock_angle=np.array([200.0]),
            proton_sw_clock_angle_uncert=np.array([0.25]),
            proton_sw_deflection_angle=np.array([5.0]),
            proton_sw_deflection_angle_uncert=np.array([0.001]),
            proton_sw_velocity_rtn_sun=np.array([[400.0, 10.0, 5.0]]),
            proton_sw_velocity_rtn=np.array([[370.0, 10.0, 5.0]]),
            proton_sw_velocity_rtn_covariance=np.array([np.eye(3)]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )
        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.return_value = runner_result

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{HYDROGEN_INFLOW_VECTOR_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{HELIUM_INFLOW_VECTOR_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value

        expected_proton_metadata = replace(input_metadata, descriptor=descriptor)
        proton_solar_wind_data.input_metadata = expected_proton_metadata

        input_metadata.descriptor = descriptor

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "06" /
                             f"imap_swapi_l3a_proton-sw_{start_date_as_str}_v123.cdf")

        mock_chunk_l2_data.return_value = [chunk_of_five]

        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies.data = Mock(energy=energy)
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = swapi_l3a_dependencies

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        swapi_l3a_dependencies.swapi_response.warm_cache.assert_called_once()

        mock_parallel_chunk_runner_class.assert_called_once_with(
            swapi_l3a_dependencies.swapi_response,
            swapi_l3a_dependencies.efficiency_calibration_table,
        )
        mock_runner.run.assert_called_once()
        actual_chunks = mock_runner.run.call_args.args[0]
        self.assertEqual([chunk_of_five], actual_chunks)

        actual_kwargs = mock_proton_solar_wind_data_constructor.call_args.kwargs
        actual_positional = mock_proton_solar_wind_data_constructor.call_args.args
        self.assertEqual(expected_proton_metadata, actual_positional[0])
        for key, expected_val in runner_result.items():
            np.testing.assert_array_equal(expected_val, actual_kwargs[key])

        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_proton-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_proton-sw_{start_date_as_str}_v123"),
                                                            ])

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", descriptor)

        self.assertEqual(input_file_names, proton_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), proton_solar_wind_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_alpha(self, mock_spicepy,
                               mock_swapi_l3_dependencies_class,
                               mock_chunk_l2_data,
                               mock_parallel_chunk_runner_class,
                               mock_write_cdf,
                               mock_alpha_solar_wind_data_constructor,
                               mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 8, 29)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 8, 28)
        l3a_descriptor = "alpha-sw"
        input_version = create_mock_version_map(descriptor=l3a_descriptor, minor_version=123)
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        initial_epoch = 10
        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]] * 4)

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        runner_result = dict(
            epoch=np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
            alpha_sw_speed=np.array([450.0]),
            alpha_sw_speed_uncert=np.array([1.0]),
            alpha_sw_speed_sun=np.array([480.0]),
            alpha_sw_speed_sun_uncert=np.array([1.4]),
            alpha_sw_density=np.array([0.15]),
            alpha_sw_density_uncert=np.array([0.01]),
            alpha_sw_temperature=np.array([400000.0]),
            alpha_sw_temperature_uncert=np.array([2000.0]),
            alpha_sw_velocity_rtn_sun=np.array([[480.0, 5.0, 1.0]]),
            alpha_sw_velocity_rtn=np.array([[450.0, 5.0, 1.0]]),
            alpha_sw_velocity_rtn_covariance=np.array([np.eye(3)]),
            quality_flags=np.array([int(SwapiL3Flags.NONE)]),
        )
        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.return_value = runner_result

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]
        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        expected_alpha_metadata = replace(input_metadata, descriptor=l3a_descriptor)
        alpha_solar_wind_data.input_metadata = expected_alpha_metadata

        input_metadata.descriptor = l3a_descriptor

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "08" /
                             f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_v123.cdf")

        mock_chunk_l2_data.return_value = [chunk_of_five]

        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies.data = Mock(energy=energy)
        swapi_l3a_dependencies.mag_data = Mock()
        swapi_l3a_dependencies.mag_is_preliminary = False
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = swapi_l3a_dependencies

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        swapi_l3a_dependencies.swapi_response.warm_cache.assert_called_once()

        mock_parallel_chunk_runner_class.assert_called_once_with(
            swapi_l3a_dependencies.swapi_response,
            swapi_l3a_dependencies.efficiency_calibration_table,
        )
        mock_runner.run.assert_called_once()
        actual_chunks = mock_runner.run.call_args.args[0]
        self.assertEqual([chunk_of_five], actual_chunks)

        actual_kwargs = mock_alpha_solar_wind_data_constructor.call_args.kwargs
        actual_positional = mock_alpha_solar_wind_data_constructor.call_args.args
        self.assertEqual(expected_alpha_metadata, actual_positional[0])
        for key, expected_val in runner_result.items():
            np.testing.assert_array_equal(expected_val, actual_kwargs[key])

        mock_manager.add_global_attribute.assert_has_calls([
            call("Data_version", outgoing_version),
            call("Generation_date", date.today().strftime("%Y%m%d")),
            call("Logical_source", "imap_swapi_l3a_alpha-sw"),
            call("Logical_file_id", f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_v123"),
        ])
        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", l3a_descriptor)

        self.assertEqual(input_file_names, alpha_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), alpha_solar_wind_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.swapi.swapi_processor.calculate_delta_minus_plus')
    @patch('imap_l3_processing.swapi.swapi_processor.save_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3BCombinedVDF')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3BDependencies')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_alpha_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_pui_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_combined_solar_wind_differential_flux')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3b(self, mock_spiceypy, mock_calculate_combined_solar_wind_differential_flux,
                         mock_calculate_pui_solar_wind_vdf,
                         mock_calculate_proton_solar_wind_vdf,
                         mock_calculate_alpha_solar_wind_vdf,
                         mock_swapi_l3b_dependencies_class,
                         mock_chunk_l2_data,
                         mock_combined_vdf_data,
                         mock_save_data,
                         mock_calculate_delta_minus_plus):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        end_date = datetime.now()
        version = 'v001'
        outgoing_data_level = "l3b"
        dependency_start_date = "20250101"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

        mock_spiceypy.ktotal.return_value = 0
        mock_calculate_proton_solar_wind_vdf.side_effect = [
            (sentinel.proton_calculated_velocities1, sentinel.proton_calculated_probabilities1),
            (sentinel.proton_calculated_velocities2, sentinel.proton_calculated_probabilities2),
        ]

        mock_calculate_alpha_solar_wind_vdf.side_effect = [
            (sentinel.alpha_calculated_velocities1, sentinel.alpha_calculated_probabilities1),
            (sentinel.alpha_calculated_velocities2, sentinel.alpha_calculated_probabilities2),
        ]

        mock_calculate_pui_solar_wind_vdf.side_effect = [
            (sentinel.pui_calculated_velocities1, sentinel.pui_calculated_probabilities1),
            (sentinel.pui_calculated_velocities2, sentinel.pui_calculated_probabilities2),
        ]

        mock_calculate_combined_solar_wind_differential_flux.side_effect = [
            sentinel.calculated_diffential_flux1,
            sentinel.calculated_diffential_flux2
        ]

        mock_calculate_delta_minus_plus.side_effect = [
            DeltaMinusPlus(sentinel.proton_velocity_delta_minus1, sentinel.proton_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.alpha_velocity_delta_minus1, sentinel.alpha_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.pui_velocity_delta_minus1, sentinel.pui_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.energy_delta_minus1, sentinel.energy_delta_plus1),
            DeltaMinusPlus(sentinel.proton_velocity_delta_minus2, sentinel.proton_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.alpha_velocity_delta_minus2, sentinel.alpha_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.pui_velocity_delta_minus2, sentinel.pui_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.energy_delta_minus2, sentinel.energy_delta_plus2),
        ]

        n_sweeps, n_bins = 4, 72
        coarse_count_rates = np.linspace(14.0, 75.0, 62)
        coarse_count_rate_uncertainties = np.linspace(0.1, 6.2, 62)
        coarse_energies = np.linspace(15000.0, 76000.0, 62)

        # All sweeps identical so the per-bin mean across sweeps equals the per-bin pattern.
        coincidence_count_rate = np.zeros((n_sweeps, n_bins))
        coincidence_count_rate[:, 1:63] = coarse_count_rates
        coincidence_count_rate_uncertainty = np.zeros((n_sweeps, n_bins))
        coincidence_count_rate_uncertainty[:, 1:63] = coarse_count_rate_uncertainties
        energy_per_sweep = np.zeros((n_sweeps, n_bins))
        energy_per_sweep[:, 1:63] = coarse_energies

        average_coincident_count_rates = coarse_count_rates
        # σ propagation through Σ/N over n independent sweeps shrinks the per-sweep σ by √n.
        average_coincident_count_rate_uncertainties = coarse_count_rate_uncertainties / np.sqrt(n_sweeps)
        energy = coarse_energies

        first_chunk_initial_epoch = 10
        first_l2_data_chunk = SwapiL2Data(np.array([first_chunk_initial_epoch, 11, 12, 13]), energy_per_sweep,
                                          coincidence_count_rate,
                                          coincidence_count_rate_uncertainty)

        second_chunk_initial_epoch = 60
        second_l2_data_chunk = SwapiL2Data(np.array([second_chunk_initial_epoch, 11, 12, 13]), energy_per_sweep,
                                           coincidence_count_rate,
                                           coincidence_count_rate_uncertainty)

        mock_chunk_l2_data.return_value = [first_l2_data_chunk, second_l2_data_chunk]

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{incoming_data_level}_{GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{incoming_data_level}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        science_inputs = [ScienceInput(file_name) for file_name in input_file_names]

        dependencies = ProcessingInputCollection(*science_inputs)

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3b_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_geometric_factor_calibration_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.geometric_factor_calibration_table
        mock_efficiency_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.efficiency_calibration_table

        mock_chunk_l2_data.assert_called_with(mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.data,
                                              50)

        mock_efficiency_table.relative_efficiency.assert_has_calls(
            [call(first_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS, Species.PROTON),
             call(second_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS, Species.PROTON)])

        expected_count_rate_with_uncertainties = uarray(average_coincident_count_rates,
                                                        average_coincident_count_rate_uncertainties)

        np.testing.assert_array_equal(energy, mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[0])
        nominal_count_rates = nominal_values(expected_count_rate_with_uncertainties)
        std_devs_count_rates = std_devs(expected_count_rate_with_uncertainties)

        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.relative_efficiency.return_value,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))

        self.assertEqual(mock_efficiency_table.relative_efficiency.return_value,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.relative_efficiency.return_value,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy,
                                      mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates, nominal_values(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates, std_devs(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.relative_efficiency.return_value,
                         mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[3])
        delta_calls = mock_calculate_delta_minus_plus.call_args_list
        self.assertEqual(sentinel.proton_calculated_velocities1, delta_calls[0].args[0])
        self.assertEqual(sentinel.alpha_calculated_velocities1, delta_calls[1].args[0])
        self.assertEqual(sentinel.pui_calculated_velocities1, delta_calls[2].args[0])
        np.testing.assert_array_equal(energy, delta_calls[3].args[0])
        self.assertEqual(sentinel.proton_calculated_velocities2, delta_calls[4].args[0])
        self.assertEqual(sentinel.alpha_calculated_velocities2, delta_calls[5].args[0])
        self.assertEqual(sentinel.pui_calculated_velocities2, delta_calls[6].args[0])
        np.testing.assert_array_equal(energy, delta_calls[7].args[0])

        expected_combined_metadata = InputMetadata(descriptor="combined", data_level=outgoing_data_level,
                                                   start_date=start_date, end_date=end_date, instrument=instrument,
                                                   version=outgoing_version)

        self.assertEqual(expected_combined_metadata, mock_combined_vdf_data.call_args_list[0].kwargs["input_metadata"])

        np.testing.assert_array_equal(np.array([first_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS,
                                                second_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS]),
                                      mock_combined_vdf_data.call_args_list[0].kwargs["epoch"])

        np.testing.assert_array_equal([sentinel.proton_calculated_velocities1, sentinel.proton_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["proton_sw_velocities"])
        np.testing.assert_array_equal([sentinel.proton_velocity_delta_minus1, sentinel.proton_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "proton_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.proton_velocity_delta_plus1, sentinel.proton_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "proton_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.proton_calculated_probabilities1, sentinel.proton_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["proton_sw_combined_vdf"])

        np.testing.assert_array_equal([sentinel.alpha_calculated_velocities1, sentinel.alpha_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["alpha_sw_velocities"])
        np.testing.assert_array_equal([sentinel.alpha_velocity_delta_minus1, sentinel.alpha_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "alpha_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.alpha_velocity_delta_plus1, sentinel.alpha_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "alpha_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.alpha_calculated_probabilities1, sentinel.alpha_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["alpha_sw_combined_vdf"])

        np.testing.assert_array_equal([sentinel.pui_calculated_velocities1, sentinel.pui_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["pui_sw_velocities"])
        np.testing.assert_array_equal([sentinel.pui_velocity_delta_minus1, sentinel.pui_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "pui_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.pui_velocity_delta_plus1, sentinel.pui_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "pui_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.pui_calculated_probabilities1, sentinel.pui_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["pui_sw_combined_vdf"])

        np.testing.assert_array_equal([energy, energy],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy"])
        np.testing.assert_array_equal([sentinel.energy_delta_minus1, sentinel.energy_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy_delta_minus"])
        np.testing.assert_array_equal([sentinel.energy_delta_plus1, sentinel.energy_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.calculated_diffential_flux1, sentinel.calculated_diffential_flux2],
            mock_combined_vdf_data.call_args_list[0].kwargs["combined_differential_flux"])

        self.assertEqual(input_file_names, mock_combined_vdf_data.return_value.parent_file_names)
        mock_save_data.assert_called_once_with(mock_combined_vdf_data.return_value)
        self.assertEqual([mock_save_data.return_value], product)


    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_raises_for_unknown_descriptor(self, mock_spicepy,
                                                       mock_swapi_l3_dependencies_class):
        mock_spicepy.ktotal.return_value = 0
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = create_swapi_l3a_dependencies_with_mocks()

        input_metadata = InputMetadata('swapi', 'l3a', datetime(2025, 6, 12), datetime(2025, 6, 13), 'v123')
        input_metadata.descriptor = "not-a-real-descriptor"

        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as cm:
            swapi_processor.process()
        self.assertEqual(("unknown descriptor", "not-a-real-descriptor"), cm.exception.args)

    def test_process_l3a_alpha_raises_when_mag_data_missing(self):
        input_metadata = InputMetadata('swapi', 'l3a', datetime(2025, 8, 28), datetime(2025, 8, 29), 'v123')
        input_metadata.descriptor = "alpha-sw"

        dependencies = create_swapi_l3a_dependencies_with_mocks()
        dependencies.mag_data = None

        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        with self.assertRaises(ValueError) as cm:
            swapi_processor.process_l3a_alpha(dependencies.data, dependencies)
        self.assertIn("alpha-sw requires MAG RTN data", str(cm.exception))

        
def create_swapi_l3a_dependencies_with_mocks():
    return SwapiL3ADependencies(
        data=Mock(),
        efficiency_calibration_table=Mock(),
        density_of_neutral_helium_calibration_table=Mock(),
        hydrogen_inflow_vector=Mock(),
        helium_inflow_vector=Mock(),
        swapi_response=Mock(),
    )
