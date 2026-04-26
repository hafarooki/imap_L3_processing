"""End-to-end subprocess integration test for the SWAPI L3a proton-sw processor.

Mirrors test_swe_processor_integration.py: runs ``imap_l3_data_processor.py`` as
a subprocess against staged test data, exercising the full real path —
dependency manifest deserialization → SPICE furnishing → SwapiL3ADependencies
loading of all 13 ancillaries → SwapiProcessor.process() → process_l3a_proton
→ save_data → CDF written to disk.

SWAPI-specific inputs live in ``tests/integration/test_data/swapi/``. SPICE
kernels are pulled from the shared ``tests/integration/test_data/spice/`` dir.
The L2 science CDF is generated on the fly by retiming an existing synthetic
spectrum into the SPICE coverage window — keeping a date-shifted copy on disk
would just be duplicate data.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

import imap_l3_processing
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path

_L2_SOURCE = "swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf"
_L2_SDC_NAME = "imap_swapi_l2_sci_20260120_v001.cdf"
_L2_START = datetime(2026, 1, 20, 12, 0, 0)


def _materialize_retimed_l2(dest_dir: Path) -> Path:
    """Copy the synthetic L2 spectrum into ``dest_dir`` under its SDC name and
    shift its epoch fields to ``_L2_START`` (within the integration SPICE
    coverage window). Returns the new path."""
    dest = dest_dir / _L2_SDC_NAME
    shutil.copy(get_test_data_path(_L2_SOURCE), dest)
    with CDF(str(dest)) as cdf:
        cdf.readonly(False)
        n = len(cdf["sci_start_time"])
        cdf["sci_start_time"][:] = [
            (_L2_START + timedelta(seconds=12 * i)).isoformat(timespec="seconds") for i in range(n)
        ]
        cdf["epoch"][:] = [_L2_START + timedelta(seconds=12 * i) for i in range(n)]
    return dest


class SwapiProcessorIntegration(unittest.TestCase):
    def test_swapi_processor_with_synthetic_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        output_data_dir = get_run_local_data_path("swapi_integration")
        expected_file_path = (
            output_data_dir / "imap/swapi/l3a/2026/01/imap_swapi_l3a_proton-sw_20260120_v001.cdf"
        )
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        spice_dir = Path("tests/integration/test_data/spice")
        spice_files = [
            spice_dir / "naif020.tls",
            spice_dir / "pck00011.tpc",
            spice_dir / "imap_130.tf",  # defines IMAP_SWAPI frame
            spice_dir / "imap_science_108.tf",
            spice_dir / "imap_sclk_008.tsc",
            spice_dir / "de440.bsp",
            spice_dir / "imap_recon_20250415_20260415_v01.bsp",
            spice_dir / "imap_2025_105_2026_105_01.ah.bc",
            spice_dir / "imap_dps_2025_105_2026_105_009.ah.bc",
        ]
        swapi_inputs = [f for f in Path("tests/integration/test_data/swapi").iterdir() if f.is_file()]

        os.environ["IMAP_DATA_DIR"] = str(output_data_dir)
        with tempfile.TemporaryDirectory() as tmp:
            l2_cdf = _materialize_retimed_l2(Path(tmp))
            input_files = spice_files + swapi_inputs + [l2_cdf]
            with mock_imap_data_access(output_data_dir, input_files):
                result = subprocess.run([
                    sys.executable, "imap_l3_data_processor.py",
                    "--instrument", "swapi",
                    "--data-level", "l3a",
                    "--descriptor", "proton-sw",
                    "--start-date", "20260120",
                    "--version", "v001",
                    "--dependency", "imap_swapi_l3a_proton-sw_20260120_v001.json",
                ])

                self.assertEqual(0, result.returncode)
                self.assertTrue(expected_file_path.exists(),
                                f"expected output CDF not found at {expected_file_path}")

                # process_l3a_proton swallows per-chunk exceptions and writes fill values,
                # so returncode==0 alone doesn't prove the fitter ran. Open the CDF and
                # check that at least one chunk produced a finite, physical solar-wind speed.
                with CDF(str(expected_file_path)) as cdf:
                    speeds = np.asarray(cdf["proton_sw_speed"][...], dtype=float)
                finite = speeds[np.isfinite(speeds)]
                self.assertGreater(len(finite), 0, "no chunk produced a finite proton_sw_speed")
                self.assertTrue(np.all((finite > 200.0) & (finite < 1500.0)),
                                f"finite speeds outside plausible heliospheric range: {finite}")


if __name__ == "__main__":
    unittest.main()
