import unittest
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies


class TestCodicePitchAngleDependencies(unittest.TestCase):

    @patch(
        "imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies.CodicePitchAngleDependencies.from_file_paths")
    @patch("imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_files):
        expected_codice_science_file_download_path = "imap/codice/l2/2010/01/imap_codice_l2_hi-sectored_20100105_v010.cdf"
        codice_sectored_intensities_input_file_name = "imap_codice_l2_hi-sectored_20100105_v010.cdf"

        expected_mag_download_path = "imap/mag/l1d/2010/01/imap_mag_l1d_norm-dsrf_20100105_v010.cdf"
        mag_input_file_name = "imap_mag_l1d_norm-dsrf_20100105_v010.cdf"

        science_input_codice_l2 = ScienceInput(codice_sectored_intensities_input_file_name)
        science_input_mag_l1d = ScienceInput(mag_input_file_name)

        process_input_collection = ProcessingInputCollection()
        process_input_collection.add(science_input_codice_l2)
        process_input_collection.add(science_input_mag_l1d)

        dependencies = CodicePitchAngleDependencies.fetch_dependencies(process_input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]

        expected_codice_science_file_path = data_dir / expected_codice_science_file_download_path
        expected_mag_science_file_path = data_dir / expected_mag_download_path

        mock_download.assert_has_calls([
            call(expected_codice_science_file_path),
            call(expected_mag_science_file_path)
        ])

        mock_from_files.assert_called_with(expected_mag_science_file_path, expected_codice_science_file_path)
        self.assertEqual(mock_from_files.return_value, dependencies)

    @patch('imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies.read_mag_data')
    @patch(
        'imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies.CodiceHiL2SectoredIntensitiesData.read_from_cdf')
    def test_from_file_paths(self, mock_codice_l2_data, mock_read_mag_data):
        codice_file_path = sentinel.codice_file_path
        mag_file_path = sentinel.mag_file_path
        dependencies = CodicePitchAngleDependencies.from_file_paths(mag_file_path, codice_file_path)

        mock_codice_l2_data.assert_called_once_with(codice_file_path)
        mock_read_mag_data.assert_called_once_with(mag_file_path)

        self.assertEqual(mock_codice_l2_data.return_value, dependencies.codice_sectored_intensities_data)
        self.assertEqual(mock_read_mag_data.return_value, dependencies.mag_l1d_data)
