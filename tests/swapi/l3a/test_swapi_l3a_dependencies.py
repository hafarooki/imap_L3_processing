import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, \
    HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, HELIUM_INFLOW_VECTOR_DESCRIPTOR, \
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR, CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR, \
    MAG_RTN_DESCRIPTOR
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies


class TestSwapiL3ADependencies(unittest.TestCase):

    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.select_mag_path")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_file_paths, mock_select_mag_path):
        input_collection = ProcessingInputCollection()

        start_date = '20100105'
        mission = 'imap'
        instrument = 'swapi'
        data_level = 'l2'
        version = 'v010'

        mock_download.side_effect = [
            sentinel.swapi_l2_data,
            sentinel.efficiency_file,
            sentinel.neutral_helium_table,
            sentinel.hydrogen_vector,
            sentinel.helium_vector,
            sentinel.azimuthal_transmission,
            sentinel.central_effective_area,
            sentinel.passband_fit_coefficients,
        ]
        mock_select_mag_path.return_value = (sentinel.mag_path, "l2")

        swapi_science_file_download_path = f"{mission}_{instrument}_{data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_efficiency_file_name = f"{mission}_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_density_of_neutral_helium_lookup = f"{mission}_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_hydrogen_inflow_filename = f"{mission}_{instrument}_{HYDROGEN_INFLOW_VECTOR_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_helium_inflow_filename = f"{mission}_{instrument}_{HELIUM_INFLOW_VECTOR_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_azimuthal_transmission_filename = f"{mission}_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_central_effective_area_filename = f"{mission}_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_passband_fit_coefficients_filename = f"{mission}_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{start_date}_{version}.cdf"

        science_input = ScienceInput(swapi_science_file_download_path)
        efficiency_ancillary = AncillaryInput(swapi_efficiency_file_name)
        density_of_neutral_helium_ancillary = AncillaryInput(swapi_density_of_neutral_helium_lookup)
        hydrogen_inflow_ancillary = AncillaryInput(swapi_hydrogen_inflow_filename)
        helium_inflow_ancillary = AncillaryInput(swapi_helium_inflow_filename)
        azimuthal_transmission_ancillary = AncillaryInput(swapi_azimuthal_transmission_filename)
        central_effective_area_ancillary = AncillaryInput(swapi_central_effective_area_filename)
        passband_fit_coefficients_ancillary = AncillaryInput(swapi_passband_fit_coefficients_filename)

        input_collection.add(
            [science_input, efficiency_ancillary,
             density_of_neutral_helium_ancillary,
             hydrogen_inflow_ancillary,
             helium_inflow_ancillary,
             azimuthal_transmission_ancillary,
             central_effective_area_ancillary,
             passband_fit_coefficients_ancillary,
             ])

        actual_swapi_l3_dependencies = SwapiL3ADependencies.fetch_dependencies(input_collection)

        science_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'swapi' / 'l2' / '2010' / '01'
        ancillary_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'ancillary' / 'swapi'

        mock_download.assert_has_calls([
            call(science_data_dir / swapi_science_file_download_path),
            call(ancillary_data_dir / swapi_efficiency_file_name),
            call(ancillary_data_dir / swapi_density_of_neutral_helium_lookup),
            call(ancillary_data_dir / swapi_hydrogen_inflow_filename),
            call(ancillary_data_dir / swapi_helium_inflow_filename),
            call(ancillary_data_dir / swapi_azimuthal_transmission_filename),
            call(ancillary_data_dir / swapi_central_effective_area_filename),
            call(ancillary_data_dir / swapi_passband_fit_coefficients_filename),
        ])

        mock_select_mag_path.assert_called_once_with(input_collection, MAG_RTN_DESCRIPTOR)

        mock_from_file_paths.assert_called_with(
            sentinel.swapi_l2_data,
            sentinel.efficiency_file,
            sentinel.neutral_helium_table,
            sentinel.hydrogen_vector,
            sentinel.helium_vector,
            sentinel.azimuthal_transmission,
            sentinel.central_effective_area,
            sentinel.passband_fit_coefficients,
            sentinel.mag_path,
            False,
        )

        self.assertEqual(mock_from_file_paths.return_value, actual_swapi_l3_dependencies)

    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.select_mag_path")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_fetch_dependencies_with_l1d_mag(self, mock_download, mock_from_file_paths, mock_select_mag_path):
        input_collection = ProcessingInputCollection()

        mock_download.side_effect = [
            sentinel.swapi_l2_data,
            sentinel.efficiency_file,
            sentinel.neutral_helium_table,
            sentinel.hydrogen_vector,
            sentinel.helium_vector,
            sentinel.azimuthal_transmission,
            sentinel.central_effective_area,
            sentinel.passband_fit_coefficients,
        ]
        mock_select_mag_path.return_value = (sentinel.mag_path, "l1d")

        start_date = '20100105'
        version = 'v010'
        descriptors = [
            ("imap_swapi_l2", SWAPI_L2_DESCRIPTOR, ScienceInput),
            ("imap_swapi", EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", HELIUM_INFLOW_VECTOR_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", AZIMUTHAL_TRANSMISSION_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, AncillaryInput),
            ("imap_swapi", PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR, AncillaryInput),
        ]
        for prefix, desc, cls in descriptors:
            input_collection.add([cls(f"{prefix}_{desc}_{start_date}_{version}.cdf")])

        SwapiL3ADependencies.fetch_dependencies(input_collection)

        _, kwargs = mock_from_file_paths.call_args
        args = mock_from_file_paths.call_args.args
        self.assertEqual(sentinel.mag_path, args[8])
        self.assertEqual(True, args[9])

    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_mag_rtn_data')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiResponse.from_files')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.InflowVector.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.DensityOfNeutralHeliumLookupTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_l2_swapi_data')
    def test_from_file_paths(self, mock_read_l2_swapi, mock_neutral_helium_from_file,
                             mock_efficiency_lookup_class,
                             mock_inflow_vector_from_file, mock_CDF,
                             mock_swapi_response_from_files, mock_read_mag_rtn_data):
        science_path = Path("imap_swapi_l2_sci_20100105_v010.cdf")
        efficiency_path = Path("imap_swapi_efficiency_20100105_v010.cdf")
        neutral_helium_path = Path("imap_swapi_neutral_he_20100105_v010.cdf")
        hydrogen_vector_path = Path("imap_swapi_h_inflow_20100105_v010.dat")
        helium_vector_path = Path("imap_swapi_he_inflow_20100105_v010.dat")
        azimuthal_transmission_path = Path("imap_swapi_azimuthal_transmission_20100105_v010.cdf")
        central_effective_area_path = Path("imap_swapi_central_effective_area_20100105_v010.cdf")
        passband_fit_coefficients_path = Path("imap_swapi_passband_fit_coefficients_20100105_v010.cdf")
        mag_path = Path("imap_mag_l2_norm-rtn_20100105_v010.cdf")

        mock_read_l2_swapi.return_value = sentinel.swapi_l2_data
        mock_efficiency_lookup_class.return_value = sentinel.efficiency_lookup
        mock_neutral_helium_from_file.return_value = sentinel.neutral_helium_data
        mock_inflow_vector_from_file.side_effect = [sentinel.hydrogen_vector, sentinel.helium_vector]
        mock_swapi_response_from_files.return_value = sentinel.swapi_response
        mock_read_mag_rtn_data.return_value = sentinel.mag_data

        expected_dependencies = SwapiL3ADependencies(
            data=sentinel.swapi_l2_data,
            efficiency_calibration_table=sentinel.efficiency_lookup,
            density_of_neutral_helium_calibration_table=sentinel.neutral_helium_data,
            hydrogen_inflow_vector=sentinel.hydrogen_vector,
            helium_inflow_vector=sentinel.helium_vector,
            swapi_response=sentinel.swapi_response,
            mag_data=sentinel.mag_data,
            mag_is_preliminary=True,
        )

        actual_dependencies = SwapiL3ADependencies.from_file_paths(
            science_path,
            efficiency_path,
            neutral_helium_path,
            hydrogen_vector_path,
            helium_vector_path,
            azimuthal_transmission_path,
            central_effective_area_path,
            passband_fit_coefficients_path,
            mag_path,
            True,
        )

        mock_CDF.assert_called_once_with(str(science_path))
        mock_read_l2_swapi.assert_called_once_with(mock_CDF.return_value)
        mock_efficiency_lookup_class.assert_called_once_with(efficiency_path)
        mock_neutral_helium_from_file.assert_called_once_with(neutral_helium_path)
        mock_inflow_vector_from_file.assert_has_calls(
            [call(hydrogen_vector_path), call(helium_vector_path)])
        mock_swapi_response_from_files.assert_called_once_with(
            azimuthal_transmission_path=azimuthal_transmission_path,
            central_effective_area_path=central_effective_area_path,
            passband_fit_coefficients_path=passband_fit_coefficients_path,
            efficiency_table_path=efficiency_path)
        mock_read_mag_rtn_data.assert_called_once_with(mag_path)

        self.assertEqual(expected_dependencies, actual_dependencies)

    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_mag_rtn_data')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiResponse.from_files')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.InflowVector.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.DensityOfNeutralHeliumLookupTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_l2_swapi_data')
    def test_from_file_paths_without_mag(self, mock_read_l2_swapi, mock_neutral_helium_from_file,
                                         mock_efficiency_lookup_class, mock_inflow_vector_from_file, mock_CDF,
                                         mock_swapi_response_from_files, mock_read_mag_rtn_data):
        mock_inflow_vector_from_file.side_effect = [sentinel.hydrogen_vector, sentinel.helium_vector]

        actual = SwapiL3ADependencies.from_file_paths(
            Path("science.cdf"),
            Path("efficiency.cdf"),
            Path("neutral_helium.cdf"),
            Path("h_inflow.dat"),
            Path("he_inflow.dat"),
            Path("azimuthal_transmission.cdf"),
            Path("central_effective_area.cdf"),
            Path("passband_fit_coefficients.cdf"),
        )

        self.assertIsNone(actual.mag_data)
        self.assertFalse(actual.mag_is_preliminary)
        mock_read_mag_rtn_data.assert_not_called()
