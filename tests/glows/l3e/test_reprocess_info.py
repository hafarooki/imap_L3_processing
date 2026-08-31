import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from imap_data_access import ProcessingInputCollection, AncillaryInput, RepointInput
from imap_processing.spice.repoint import set_global_repoint_table_paths, get_repoint_data

from imap_l3_processing.glows.descriptors import GLOWS_L3E_HI_45_DESCRIPTOR, GLOWS_L3E_HI_90_DESCRIPTOR, \
    GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3E_ULTRA_HF_DESCRIPTOR, GLOWS_L3B_DESCRIPTOR, \
    GLOWS_L3C_DESCRIPTOR, GLOWS_L3D_DESCRIPTOR, GLOWS_REPROCESSING_DESCRIPTOR
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo, ReprocessTargets, fetch_reprocess_info
from tests.test_helpers import get_test_data_folder, get_test_data_path, create_mock_query_results


class TestReprocessInfo(unittest.TestCase):
    @patch("imap_l3_processing.glows.l3e.reprocess_info.datetime")
    def test_parse_from_ancillary(self, mock_datetime):
        products = {
            "survival-probability-lo": ReprocessTargets([245, 246], [2310]),
            "survival-probability-hi-90": ReprocessTargets([], [2313]),
            "survival-probability-hi-45": ReprocessTargets([], [2313, 2314]),
            "survival-probability-ul-sf": ReprocessTargets([246, 247], []),
            "survival-probability-ul-hf": ReprocessTargets([243], []),
            "solar-hist": ReprocessTargets([], []),
        }
        test_file: Path = (
            get_test_data_folder() / "glows" / "glows_reprocessing_ancillary_file.dat"
        )
        mock_datetime.now.return_value = datetime(2026, 8, 11, 11, 40, tzinfo=timezone.utc)
        cases = [
            (datetime(2026, 8, 11, 11, 41, tzinfo=timezone.utc), products),
            (datetime(2026, 8, 11, 11, 41), products),
            (datetime(2026, 8, 11, 11, 39, tzinfo=timezone.utc), {}),
            (datetime(2026, 8, 11, 11, 39), {}),
        ]
        for ignore_time, expected_products in cases:
            with self.subTest(ignore_time):
                mock_datetime.fromisoformat.return_value = ignore_time

                reprocess_info = ReprocessInfo.parse_from_ancillary(test_file)

                self.assertEqual(expected_products, reprocess_info.products_to_reprocess)
                mock_datetime.fromisoformat.assert_called_with("2226-08-11 11:40")
                mock_datetime.now.assert_called_with(timezone.utc)

    def test_should_reprocess_l3d_if_l3d_or_l3e_products_are_specified(self):
        cases = [
            ("should reprocess l3e hi45",
             {GLOWS_L3E_HI_45_DESCRIPTOR: ReprocessTargets([], []), GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, True),
            ("should reprocess l3e hi90",
             {GLOWS_L3E_HI_90_DESCRIPTOR: ReprocessTargets([], []), GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, True),
            ("should reprocess l3e lo",
             {GLOWS_L3E_LO_DESCRIPTOR: ReprocessTargets([], []), GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, True),
            ("should reprocess l3e ultra sf",
             {GLOWS_L3E_ULTRA_SF_DESCRIPTOR: ReprocessTargets([], []), GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, True),
            ("should reprocess l3e ultra hf",
             {GLOWS_L3E_ULTRA_HF_DESCRIPTOR: ReprocessTargets([], []), GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, True),
            ("should not reprocess l3b alone", {GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}, False),
            ("should not reprocess l3c alone", {GLOWS_L3C_DESCRIPTOR: ReprocessTargets([], [])}, False),
            ("should reprocess l3d alone", {GLOWS_L3D_DESCRIPTOR: ReprocessTargets([], [])}, True),
        ]

        for case, products_to_reprocess, expected_result in cases:
            with self.subTest(case):
                reprocess_info = ReprocessInfo(products_to_reprocess)

                self.assertEqual(expected_result, reprocess_info.should_reprocess_l3d())

    def test_get_repoints(self):
        reprocess_info = ReprocessInfo(
            {
                GLOWS_L3E_LO_DESCRIPTOR: ReprocessTargets(
                    [2045, 2046, 3700], [2093, 2094]
                ),
                GLOWS_L3E_HI_45_DESCRIPTOR: ReprocessTargets([2050], []),
            }
        )

        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        set_global_repoint_table_paths([repointing_path])
        repointing_data = get_repoint_data()

        repoints_for_lo = reprocess_info.get_repoints_for_descriptor(
            GLOWS_L3E_LO_DESCRIPTOR, repointing_data
        )
        repoints_for_hi = reprocess_info.get_repoints_for_descriptor(
            GLOWS_L3E_HI_45_DESCRIPTOR, repointing_data
        )
        repoints_for_ultra = reprocess_info.get_repoints_for_descriptor(
            GLOWS_L3E_ULTRA_HF_DESCRIPTOR, repointing_data
        )

        repoints_for_cr2093 = list(range(3682, 3710))
        repoints_for_cr2094 = list(range(3710, 3736))

        expected_repoints_lo = [2045, 2046] + repoints_for_cr2093 + repoints_for_cr2094
        expected_repoints_hi = [2050]
        expected_repoints_ultra = []
        self.assertEqual(expected_repoints_lo, repoints_for_lo)
        self.assertEqual(expected_repoints_hi, repoints_for_hi)
        self.assertEqual(expected_repoints_ultra, repoints_for_ultra)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access.download')
    def test_fetch_reprocess_info_downloads_and_parses_reprocess_info(self, mock_download):
        processing_input = ProcessingInputCollection(
            AncillaryInput(f"imap_glows_{GLOWS_REPROCESSING_DESCRIPTOR}_20250101_v000.dat"),
            RepointInput("imap_2026_269_12.repoint.csv"),
        )
        path_to_downloaded_ancillary = get_test_data_folder() / 'glows' / 'glows_reprocessing_ancillary_file.dat'
        mock_download.return_value = path_to_downloaded_ancillary

        reprocess_info = fetch_reprocess_info(processing_input)

        mock_download.assert_called_with(f'imap_glows_{GLOWS_REPROCESSING_DESCRIPTOR}_20250101_v000.dat')

        expected_products_to_reprocess = {
            GLOWS_L3E_LO_DESCRIPTOR: ReprocessTargets([245, 246], [2310]),
            GLOWS_L3E_HI_90_DESCRIPTOR: ReprocessTargets([], [2313]),
            GLOWS_L3E_HI_45_DESCRIPTOR: ReprocessTargets([], [2313, 2314]),
            GLOWS_L3E_ULTRA_SF_DESCRIPTOR: ReprocessTargets([246, 247], []),
            GLOWS_L3E_ULTRA_HF_DESCRIPTOR: ReprocessTargets([243], []),
            GLOWS_L3D_DESCRIPTOR: ReprocessTargets([], []),
        }
        self.assertEqual(expected_products_to_reprocess, reprocess_info.products_to_reprocess)
