import json
import logging
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from imap_data_access import ProcessingInputCollection, ScienceInput
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF
from spiceypy import spiceypy

import imap_l3_data_processor
from imap_l3_processing.hi.hi_combined_initializer import HI_COMBINED_DESCRIPTORS
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, run_periodically, get_test_data_path

INTEGRATION_TEST_DATA_PATH = Path(__file__).parent / "test_data"


class TestMapIntegration(unittest.TestCase):
    def setUp(self):
        spiceypy.kclear()

    def tearDown(self):
        spiceypy.kclear()

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_lo_l3_multiple_arcs(self, mock_parse_cli_arguments):
        lo_multiple_arcs_test_data_dir = INTEGRATION_TEST_DATA_PATH / "lo/multiple_arcs"
        lo_imap_data_dir = get_run_local_data_path("lo/integration_data")

        input_files = list(lo_multiple_arcs_test_data_dir.glob("*.cdf")) + [
            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_recon_20250415_20260415_v01.bsp"]

        with mock_imap_data_access(lo_imap_data_dir, input_files):
            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            mock_arguments = Mock()
            mock_arguments.instrument = "lo"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "all-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_map_path = ScienceFilePath(
                'imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-1yr_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_map_path = ScienceFilePath(
                'imap_lo_l3_l090-ena-h-hf-sp-ram-hae-6deg-1yr_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hi_all_sp_maps_single_sensor(self, mock_parse_cli_arguments):
        hi_test_data_dir = INTEGRATION_TEST_DATA_PATH / "hi"
        hi_imap_data_dir = get_run_local_data_path("hi/integration_data")

        input_files = [
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20250415_v971.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l1c_45sensor-pset_20250415-repoint01000_v971.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l1c_45sensor-pset_20251015-repoint01183_v971.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l1c_90sensor-pset_20251015-repoint01183_v971.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-45_20260418-repoint02000_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-90_20250415-repoint01000_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-90_20260418-repoint02000_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-90_20251015-repoint01183_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_glows_l3e_survival-probability-hi-45_20251015-repoint01183_v001.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-6mo_20250415_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-6mo_20250415_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-6mo_20251015_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-6mo_20251015_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-6mo_20250415_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-6mo_20250415_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-6mo_20251015_v000.cdf",
            hi_test_data_dir / 'single-sensor' / "imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-6mo_20251015_v000.cdf",

            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_recon_20250415_20260415_v01.bsp",
        ]

        with mock_imap_data_access(hi_imap_data_dir, input_files):
            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            mock_arguments = Mock()
            mock_arguments.instrument = "hi"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "sp-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h45-ena-h-sf-sp-ram-hae-4deg-1yr_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {
                "imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf",
                "imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf",
                "imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf",
            }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-4deg-6mo_20250415_v001.cdf').construct_path()

            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {'imap_hi_l1c_45sensor-pset_20250415-repoint01000_v971.cdf',
                                'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-6mo_20250415_v000.cdf',
                                'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-6mo_20250415_v000.cdf',
                                'imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf',
                                }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-4deg-6mo_20251015_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {'imap_hi_l1c_45sensor-pset_20251015-repoint01183_v971.cdf',
                                'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-6mo_20251015_v000.cdf',
                                'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-6mo_20251015_v000.cdf',
                                'imap_glows_l3e_survival-probability-hi-45_20251015-repoint01183_v001.cdf',
                                }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-4deg-6mo_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {'imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf',
                                'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-6mo_20250415_v000.cdf',
                                'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-6mo_20250415_v000.cdf',
                                'imap_glows_l3e_survival-probability-hi-90_20250415-repoint01000_v001.cdf',
                                }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-4deg-6mo_20251015_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {'imap_hi_l1c_90sensor-pset_20251015-repoint01183_v971.cdf',
                                'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-6mo_20251015_v000.cdf',
                                'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-6mo_20251015_v000.cdf',
                                'imap_glows_l3e_survival-probability-hi-90_20251015-repoint01183_v001.cdf',
                                }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))
                self.assertTrue(np.any(cdf['quality_flags'][...] > 0))

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hi_combined_sensor(self, mock_parse_cli_arguments):
        hi_test_data_dir = INTEGRATION_TEST_DATA_PATH / "hi"
        hi_imap_data_dir = get_run_local_data_path("hi/integration_data")

        sp_files = [
            'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250415_v001.cdf',
            'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250415_v001.cdf',
            'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20250415_v001.cdf',
            'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-6deg-1yr_20250415_v001.cdf',
        ]

        nsp_files = [
            'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-6deg-1yr_20250415_v000.cdf',
            'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20250415_v000.cdf',
            'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-6deg-1yr_20250415_v000.cdf',
            'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-6deg-1yr_20250415_v000.cdf',
        ]

        input_files = [hi_test_data_dir / 'combined' / f for f in [*sp_files, *nsp_files]]

        with mock_imap_data_access(hi_imap_data_dir, input_files):
            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            mock_arguments = Mock()
            mock_arguments.instrument = "hi"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "hic-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"

            version_descriptors = {descriptor: {"major_version": 4, "minor_version": 7} for descriptor in HI_COMBINED_DESCRIPTORS}
            dependency_with_version_info = {
                "version": version_descriptors,
                "dependency": []
            }
            mock_arguments.dependency = json.dumps(dependency_with_version_info)
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250415_v004.0001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(set(sp_files), set(cdf.attrs["Parents"]))

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_hic-ena-h-hf-nsp-full-hae-6deg-1yr_20250415_v004.0001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(set(nsp_files), set(cdf.attrs["Parents"]))

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_lo_all_sp_maps(self, mock_parse_cli_arguments):
        lo_test_data_dir = INTEGRATION_TEST_DATA_PATH / "lo"
        lo_imap_data_dir = get_run_local_data_path("lo/integration_data")

        input_files = [
            lo_test_data_dir
            / "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr_20260101_v900.cdf",
            lo_test_data_dir
            / "imap_lo_l2_l090-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v900.cdf",
            get_test_data_path("lo/imap_lo_l1c_pset_20260101-repoint01261_v001.cdf"),
            lo_test_data_dir
            / "imap_glows_l3e_survival-probability-lo_20260101-repoint01261_v001.cdf",
            lo_test_data_dir
            / "imap_glows_l3e_survival-probability-lo_20270418-repoint03003_v001.cdf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH
            / "spice"
            / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA_PATH
            / "spice"
            / "imap_recon_20250415_20260415_v01.bsp",
        ]

        with mock_imap_data_access(lo_imap_data_dir, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "lo"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "all-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_ena_path = ScienceFilePath(
                "imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-1yr_20260101_v001.cdf").construct_path()
            expected_hf_ena_path = ScienceFilePath(
                "imap_lo_l3_l090-ena-h-hf-sp-ram-hae-6deg-1yr_20260101_v001.cdf").construct_path()
            self.assertTrue(expected_ena_path.exists(), f"Expected file {expected_ena_path.name} not found")
            self.assertTrue(expected_hf_ena_path.exists(), f"Expected file {expected_hf_ena_path.name} not found")
            expected_ena_parents = {
                "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr_20260101_v900.cdf",
                "imap_lo_l1c_pset_20260101-repoint01261_v001.cdf",
                "imap_glows_l3e_survival-probability-lo_20260101-repoint01261_v001.cdf",
            }
            expected_hf_ena_parents = {
                "imap_lo_l2_l090-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v900.cdf",
                "imap_lo_l1c_pset_20260101-repoint01261_v001.cdf",
                "imap_glows_l3e_survival-probability-lo_20260101-repoint01261_v001.cdf",
            }

            with CDF(str(expected_ena_path)) as cdf:
                self.assertEqual(expected_ena_parents, set(cdf.attrs["Parents"]))
                self.assertIn("survival_probability", cdf)

            with CDF(str(expected_hf_ena_path)) as cdf:
                self.assertEqual(expected_hf_ena_parents, set(cdf.attrs["Parents"]))
                self.assertIn("survival_probability", cdf)

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_lo_isn_maps(self, mock_parse_cli_arguments):
        lo_test_data_dir = INTEGRATION_TEST_DATA_PATH / "lo"
        lo_imap_data_dir = get_run_local_data_path("lo/integration_data")

        input_file = "imap_lo_l2_l090-enanbs-h-sf-nsp-ram-hae-6deg-1yr_20250415_v000.cdf"
        descriptor = "l090-isn-h-sf-nsp-ram-hae-6deg-1yr"
        expected_output_file = "imap_lo_l3_l090-isn-h-sf-nsp-ram-hae-6deg-1yr_20250415_v001.cdf"

        processing_input_collection = ProcessingInputCollection(ScienceInput(input_file))

        with (mock_imap_data_access(lo_imap_data_dir, [lo_test_data_dir / input_file])):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "lo"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = descriptor
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = processing_input_collection.serialize()
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_ena_path = ScienceFilePath(
                expected_output_file).construct_path()

            self.assertTrue(
                expected_ena_path.exists(),
                f"Expected file {expected_ena_path.name} not found",
            )

            expected_ena_parents = {
                input_file,
            }

            with CDF(str(expected_ena_path)) as cdf:
                self.assertEqual(expected_ena_parents, set(cdf.attrs["Parents"]))
                self.assertEqual("ECLIPJ2000", str(cdf.attrs["Spice_reference_frame"]))

    @run_periodically(timedelta(days=3))
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_ultra_all_sp_maps(self, mock_parse_cli_arguments):
        ultra_test_data_dir = INTEGRATION_TEST_DATA_PATH / "ultra"
        ultra_imap_data_dir = get_run_local_data_path("ultra/integration_data")

        input_files = [
            ultra_test_data_dir / "imap_glows_l3e_survival-probability-ul-sf_20250415-repoint00000_v001.cdf",
            ultra_test_data_dir / "imap_glows_l3e_survival-probability-ul-sf_20261020-repoint02000_v001.cdf",
            ultra_test_data_dir / "imap_ultra_l1c_45sensor-spacecraftpset_20250416-repoint00000_v000.cdf",
            ultra_test_data_dir / "imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",

            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",

            get_test_data_path("ultra/imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv"),
        ]

        with mock_imap_data_access(ultra_imap_data_dir, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "ultra"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "u45-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_map_path = ScienceFilePath(
                "imap_ultra_l3_u45-ena-h-sf-sp-full-hae-6deg-3mo_20250416_v001.cdf").construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {
                "imap_glows_l3e_survival-probability-ul-sf_20250415-repoint00000_v001.cdf",
                "imap_ultra_l1c_45sensor-spacecraftpset_20250416-repoint00000_v000.cdf",
                "imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-6deg-3mo_20250416_v001.cdf"
            }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))
                self.assertIn("survival_probability", cdf)

    @run_periodically(timedelta(days=7))
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_ultra_combined_sp_maps(self, mock_parse_cli_arguments):
        ultra_imap_data_dir = get_run_local_data_path("ultra/integration_data")
        ultra_path = INTEGRATION_TEST_DATA_PATH / "ultra"

        ancil_files = [
            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",

            get_test_data_path("ultra/imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv"),
        ]
        input_files = [
            ultra_path / "imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            ultra_path / "imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            ultra_path / "imap_ultra_l1c_45sensor-heliopset_20250416-repoint00000_v000.cdf",
            ultra_path / "imap_ultra_l1c_45sensor-heliopset_20251017-repoint00184_v000.cdf",
            ultra_path / "imap_ultra_l1c_90sensor-heliopset_20250416-repoint00000_v000.cdf",
            ultra_path / "imap_ultra_l1c_90sensor-heliopset_20251017-repoint00184_v000.cdf",
            ultra_path / "imap_glows_l3e_survival-probability-ul-hf_20250415-repoint00000_v001.cdf",
            ultra_path / "imap_glows_l3e_survival-probability-ul-hf_20261020-repoint02000_v001.cdf",
        ]

        with mock_imap_data_access(ultra_imap_data_dir, input_files + ancil_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "ultra"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "ulc-sp-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_sp_map_path = ScienceFilePath(
                "imap_ultra_l3_ulc-ena-h-hf-sp-full-hae-6deg-3mo_20250416_v001.cdf").construct_path()
            self.assertTrue(expected_sp_map_path.exists(), f"Expected file {expected_sp_map_path.name} not found")

            expected_parents = {
                "imap_glows_l3e_survival-probability-ul-hf_20250415-repoint00000_v001.cdf",
                "imap_ultra_l1c_45sensor-heliopset_20250416-repoint00000_v000.cdf",
                "imap_ultra_l1c_90sensor-heliopset_20250416-repoint00000_v000.cdf",
                "imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
                "imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            }

            with CDF(str(expected_sp_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

    @run_periodically(timedelta(days=7))
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_ultra_combined_nsp_maps(self, mock_parse_cli_arguments):
        ultra_imap_data_dir = get_run_local_data_path("ultra/integration_data")
        ultra_path = INTEGRATION_TEST_DATA_PATH / "ultra"

        ancil_files = [
            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",

            get_test_data_path("ultra/imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv"),
        ]
        input_files = [
            ultra_path / "imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            ultra_path / "imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            ultra_path / "imap_ultra_l1c_45sensor-heliopset_20250416-repoint00000_v000.cdf",
            ultra_path / "imap_ultra_l1c_45sensor-heliopset_20251017-repoint00184_v000.cdf",
            ultra_path / "imap_ultra_l1c_90sensor-heliopset_20250416-repoint00000_v000.cdf",
            ultra_path / "imap_ultra_l1c_90sensor-heliopset_20251017-repoint00184_v000.cdf",
        ]

        with mock_imap_data_access(ultra_imap_data_dir, input_files + ancil_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "ultra"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "ulc-nsp-maps"
            mock_arguments.start_date = "20250415"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "[]"
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_nsp_map_path = ScienceFilePath(
                "imap_ultra_l3_ulc-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf").construct_path()
            self.assertTrue(expected_nsp_map_path.exists(), f"Expected file {expected_nsp_map_path.name} not found")

            expected_parents = {
                "imap_ultra_l1c_45sensor-heliopset_20250416-repoint00000_v000.cdf",
                "imap_ultra_l1c_90sensor-heliopset_20250416-repoint00000_v000.cdf",
                "imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
                "imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-6deg-3mo_20250416_v001.cdf",
            }

            with CDF(str(expected_nsp_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))
