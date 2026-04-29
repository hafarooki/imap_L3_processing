from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies, HIT_L2_DESCRIPTOR


class TestHITL3SectoredDependencies(TestCase):
    @patch("imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.read_mag_data")
    @patch("imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.read_l2_hit_data")
    @patch('imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.download')
    def test_fetch_dependencies(self, mock_download, mock_read_hit_data, mock_read_mag_data):
        hit_l2_data_dependency = ScienceInput(f"imap_hit_l2_{HIT_L2_DESCRIPTOR}_20240908_v001.cdf")
        mag_data_dependency = ScienceInput(f"imap_mag_l1d_norm-dsrf_20240906_v001.cdf")

        hit_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'hit' / 'l2' / '2024' / '09'
        mag_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'mag' / 'l1d' / '2024' / '09'

        expected_file_paths = [hit_data_dir / f"imap_hit_l2_{HIT_L2_DESCRIPTOR}_20240908_v001.cdf",
                               mag_data_dir / f"imap_mag_l1d_norm-dsrf_20240906_v001.cdf"]
        processing_input_collection = ProcessingInputCollection(hit_l2_data_dependency, mag_data_dependency)

        hit_data_path = Path("hit")
        mag_data_path = Path("mag")
        mock_download.side_effect = [
            hit_data_path,
            mag_data_path
        ]

        deps = HITL3SectoredDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(expected_file_paths[0]),
            call(expected_file_paths[1])
        ])

        mock_read_hit_data.assert_called_with(hit_data_path)
        mock_read_mag_data.assert_called_with(mag_data_path)

        self.assertEqual(mock_read_hit_data.return_value, deps.data)
        self.assertEqual(mock_read_mag_data.return_value, deps.mag_l1d_data)
