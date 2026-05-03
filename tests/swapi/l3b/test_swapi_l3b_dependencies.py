import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, AZIMUTHAL_TRANSMISSION_DESCRIPTOR, \
    CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies


class TestSwapiL3BDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imap_patcher = patch('imap_l3_processing.utils.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()

    def tearDown(self) -> None:
        self.mock_imap_patcher.stop()

    @patch(
        "imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.SwapiL3BDependencies.from_file_paths")
    @patch(
        "imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_file_paths):
        incoming_data_level = 'l2'
        version = 'v002'
        start_date = datetime(2025, 1, 1).strftime("%Y%m%d")

        science_file_path = f'imap_swapi_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf'
        efficiency_table_path = f'imap_swapi_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf'
        az_path = f'imap_swapi_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{start_date}_{version}.csv'
        ea_path = f'imap_swapi_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{start_date}_{version}.csv'
        pb_path = f'imap_swapi_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{start_date}_{version}.csv'

        science_input = ScienceInput(science_file_path)
        dependencies = ProcessingInputCollection(
            science_input,
            AncillaryInput(efficiency_table_path),
            AncillaryInput(az_path),
            AncillaryInput(ea_path),
            AncillaryInput(pb_path),
        )

        actual_swapi_l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(dependencies)

        sci_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'swapi' / 'l2' / '2025' / '01'
        ancillary_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'ancillary' / 'swapi'

        mock_download.assert_has_calls([
            call(sci_data_dir / science_file_path),
            call(ancillary_data_dir / efficiency_table_path),
            call(ancillary_data_dir / az_path),
            call(ancillary_data_dir / ea_path),
            call(ancillary_data_dir / pb_path),
        ])

        mock_from_file_paths.assert_called_with(
            sci_data_dir / science_file_path,
            ancillary_data_dir / efficiency_table_path,
            ancillary_data_dir / az_path,
            ancillary_data_dir / ea_path,
            ancillary_data_dir / pb_path,
        )

        self.assertEqual(mock_from_file_paths.return_value, actual_swapi_l3b_dependencies)

    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.SWAPIResponse.from_files')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.read_l2_swapi_data')
    def test_from_file_paths(self, mock_read_l2_swapi, mock_swapi_response_from_files,
                             mock_efficiency_calibration, mock_cdf):
        l2 = Path("imap_swapi_l2_sci_20100105_v010.cdf")
        efficiency = Path("imap_swapi_efficiency-lut_20100105_v010.cdf")
        az = Path("imap_swapi_azimuthal-transmission_20100105_v010.csv")
        ea = Path("imap_swapi_central-effective-area_20100105_v010.csv")
        pb = Path("imap_swapi_passband-fit-coefficients_20100105_v010.csv")

        mock_read_l2_swapi.return_value = sentinel.swapi_l2_data
        mock_swapi_response_from_files.return_value = sentinel.swapi_response
        mock_efficiency_calibration.return_value = sentinel.efficiency_calibration_data

        expected_dependencies = SwapiL3BDependencies(
            sentinel.swapi_l2_data,
            sentinel.swapi_response,
            sentinel.efficiency_calibration_data,
        )

        actual_dependencies = SwapiL3BDependencies.from_file_paths(l2, efficiency, az, ea, pb)

        mock_read_l2_swapi.assert_called_once_with(mock_cdf.return_value)
        mock_swapi_response_from_files.assert_called_once_with(az, ea, pb)
        mock_efficiency_calibration.assert_called_once_with(efficiency)
        mock_cdf.assert_called_once_with(str(l2))

        self.assertEqual(expected_dependencies, actual_dependencies)
