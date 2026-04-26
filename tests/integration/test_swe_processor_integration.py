import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import skipUnless

import imap_data_access
from imap_data_access import (
    ScienceFilePath,
)

import imap_l3_processing
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path, get_run_local_data_path


class SweProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_swe_processor_with_production_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"
        expected_file_path = ScienceFilePath(
            "imap_swe_l3_sci_20260120_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument",
                "swe",
                "--data-level",
                "l3",
                "--descriptor",
                "sci",
                "--start-date",
                "20260120",
                "--version",
                "v001",
                "--dependency",
                "imap_swe_l3_sci-e979d33c_20260120_v001.json",
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

    def test_swe_processor_with_local_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        OUTPUT_DATA_DIR = get_run_local_data_path("swe_integration")
        expected_file_path = (
            OUTPUT_DATA_DIR / "imap/swe/l3/2026/01/imap_swe_l3_sci_20260120_v001.cdf"
        )
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        input_files = [f for f in Path("tests/integration/test_data/swe").iterdir() if f.is_file()]
        # pck00011.tpc and imap_130.tf were moved to the shared SPICE dir; pull them in here.
        spice_dir = Path("tests/integration/test_data/spice")
        input_files += [spice_dir / "pck00011.tpc", spice_dir / "imap_130.tf"]
        os.environ["IMAP_DATA_DIR"] = str(OUTPUT_DATA_DIR)
        with mock_imap_data_access(OUTPUT_DATA_DIR, input_files):
            result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swe",
                    "--data-level",
                    "l3",
                    "--descriptor",
                    "sci",
                    "--start-date",
                    "20260120",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_swe_l3_sci-e979d33c_20260120_v001.json",
                ],
            )

            self.assertEqual(0, result.returncode)
            self.assertTrue(expected_file_path.exists())
