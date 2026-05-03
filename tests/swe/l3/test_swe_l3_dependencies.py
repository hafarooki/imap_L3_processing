import unittest
from pathlib import Path
from unittest.mock import patch, Mock, call

from imap_data_access.processing_input import ScienceInput, AncillaryInput, ProcessingInputCollection

from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies

MODULE = "imap_l3_processing.swe.l3.swe_l3_dependencies"


class TestSweL3Dependencies(unittest.TestCase):
    @patch(f"{MODULE}.read_l1b_swe_data")
    @patch(f"{MODULE}.read_mag_data")
    @patch(f"{MODULE}.read_l2_swe_data")
    @patch(f"{MODULE}.read_l3a_swapi_proton_data")
    @patch(f"{MODULE}.read_swe_config")
    def test_from_file_paths(self, mock_read_swe_config, mock_read_swapi_data, mock_read_swe_data, mock_read_mag_data,
                             mock_read_l1b_swe_data):
        swe_path = Path("test_swe_cdf.cdf")
        swe_l1b_path = Path("test_swe_l1b_cdf.cdf")
        mag_path = Path("test_mag_cdf.cdf")
        swapi_path = Path("test_swapi_cdf.cdf")
        config_path = Path("test_config.json")

        result = SweL3Dependencies.from_file_paths(swe_path, swe_l1b_path, mag_path, swapi_path, config_path)

        mock_read_l1b_swe_data.assert_called_once_with(swe_l1b_path)
        mock_read_swe_data.assert_called_once_with(swe_path)
        mock_read_mag_data.assert_called_once_with(mag_path)
        mock_read_swapi_data.assert_called_once_with(swapi_path)
        mock_read_swe_config.assert_called_once_with(config_path)

        self.assertEqual(mock_read_l1b_swe_data.return_value, result.swe_l1b_data)
        self.assertEqual(mock_read_swe_data.return_value, result.swe_l2_data)
        self.assertEqual(mock_read_mag_data.return_value, result.mag_data)
        self.assertEqual(mock_read_swapi_data.return_value, result.swapi_l3a_proton_data)
        self.assertEqual(mock_read_swe_config.return_value, result.configuration)

    @patch(f"{MODULE}.download")
    @patch(f"{MODULE}.SweL3Dependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download_dependency):
        swe_l2_dependency = ScienceInput("imap_swe_l2_sci_20200101_v000.cdf")
        swe_l1b_dependency = ScienceInput("imap_swe_l1b_sci_20200101_v000.cdf")
        mag_l1d_dependency = ScienceInput("imap_mag_l1d_norm-dsrf_20200101_v000.cdf")
        swapi_l3a_dependency = ScienceInput("imap_swapi_l3_proton-sw_20200101_v000.cdf")
        config_dependency = AncillaryInput("imap_swe_config_20250101_v000.json")

        processing_input_collection = ProcessingInputCollection(swe_l2_dependency, swe_l1b_dependency,
                                                                mag_l1d_dependency, swapi_l3a_dependency,
                                                                config_dependency)

        expected_swe_path = Mock()
        expected_swe_l1b_path = Mock()
        expected_mag_path = Mock()
        expected_swapi_path = Mock()
        expected_config_path = Mock()
        mock_download_dependency.side_effect = [expected_swe_path, expected_swe_l1b_path, expected_mag_path,
                                                expected_swapi_path,
                                                expected_config_path]

        result = SweL3Dependencies.fetch_dependencies(processing_input_collection)

        mock_download_dependency.assert_has_calls([call(swe_l2_dependency.imap_file_paths[0].construct_path()),
                                                   call(swe_l1b_dependency.imap_file_paths[0].construct_path()),
                                                   call(mag_l1d_dependency.imap_file_paths[0].construct_path()),
                                                   call(swapi_l3a_dependency.imap_file_paths[0].construct_path()),
                                                   call(config_dependency.imap_file_paths[0].construct_path())],
                                                  any_order=False)

        mock_from_file_paths.assert_called_with(expected_swe_path, expected_swe_l1b_path, expected_mag_path,
                                                expected_swapi_path,
                                                expected_config_path)
        self.assertEqual(mock_from_file_paths.return_value, result)

    @patch(f"{MODULE}.download")
    @patch(f"{MODULE}.SweL3Dependencies.from_file_paths")
    def test_fetch_dependencies_prioritizes_mag_l2_data(self, mock_from_file_paths, mock_download):
        swe_l2_dependency = ScienceInput("imap_swe_l2_sci_20200101_v000.cdf")
        swe_l1b_dependency = ScienceInput("imap_swe_l1b_sci_20200101_v000.cdf")
        mag_l1d_dependency = ScienceInput("imap_mag_l1d_norm-dsrf_20200101_v000.cdf")
        mag_l2_dependency = ScienceInput("imap_mag_l2_norm-dsrf_20200101_v000.cdf")
        swapi_l3a_dependency = ScienceInput("imap_swapi_l3_proton-sw_20200101_v000.cdf")
        config_dependency = AncillaryInput("imap_swe_config_20250101_v000.json")

        processing_input_collection = ProcessingInputCollection(
            swe_l2_dependency,
            swe_l1b_dependency,
            mag_l1d_dependency,
            mag_l2_dependency,
            swapi_l3a_dependency,
            config_dependency,
        )

        expected_swe_path = Mock()
        expected_swe_l1b_path = Mock()
        expected_mag_path = Mock()
        expected_swapi_path = Mock()
        expected_config_path = Mock()
        mock_download.side_effect = [
            expected_swe_path,
            expected_swe_l1b_path,
            expected_mag_path,
            expected_swapi_path,
            expected_config_path,
        ]

        result = SweL3Dependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([call(swe_l2_dependency.imap_file_paths[0].construct_path()),
                                        call(swe_l1b_dependency.imap_file_paths[0].construct_path()),
                                        call(mag_l2_dependency.imap_file_paths[0].construct_path()),
                                        call(swapi_l3a_dependency.imap_file_paths[0].construct_path()),
                                        call(config_dependency.imap_file_paths[0].construct_path())],
                                       any_order=False)

        mock_from_file_paths.assert_called_with(
            expected_swe_path,
            expected_swe_l1b_path,
            expected_mag_path,
            expected_swapi_path,
            expected_config_path,
        )
        self.assertEqual(mock_from_file_paths.return_value, result)
