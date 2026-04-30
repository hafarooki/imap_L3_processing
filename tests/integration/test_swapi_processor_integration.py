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
from tests.test_helpers import (
    get_run_local_data_path,
    get_test_data_path,
    get_test_instrument_team_data_path,
)

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
            (_L2_START + timedelta(seconds=12 * i)).isoformat(timespec="seconds")
            for i in range(n)
        ]
        cdf["epoch"][:] = [_L2_START + timedelta(seconds=12 * i) for i in range(n)]
    return dest


class SwapiProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_swapi_processor_with_production_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"

        anc_dir = root_dir / "data" / "imap" / "ancillary" / "swapi"
        anc_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "imap_swapi_azimuthal-transmission_20260425_v001.csv",
            "imap_swapi_central-effective-area_20260425_v001.csv",
            "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
        ]:
            dest = anc_dir / name
            if not dest.exists():
                shutil.copy(root_dir / "instrument_team_data" / "swapi" / name, dest)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3a_proton-sw_20260101_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument",
                "swapi",
                "--data-level",
                "l3a",
                "--descriptor",
                "proton-sw",
                "--start-date",
                "20260101",
                "--version",
                "v001",
                "--dependency",
                # TODO switch to dependency file
                """
                [{"type":"science","files":["imap_swapi_l2_sci_20260101_v001.cdf"]},{"type":"science","files":["imap_mag_l1d_norm-dsrf_20260101_v010.cdf"]},{"type":"ancillary","files":["imap_swapi_alpha-density-temperature-lut_20250125_v001.dat"]},{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v001.dat"]},{"type":"ancillary","files":["imap_swapi_energy-gf-pui-lut_20100101_v003.csv"]},{"type":"ancillary","files":["imap_swapi_instrument-response-lut_20241023_v001.zip"]},{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v002.dat"]},{"type":"ancillary","files":["imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"]},{"type":"ancillary","files":["imap_swapi_helium-inflow-vector_20100101_v001.dat"]},{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]},{"type":"spice","files":["naif0012.tls","pck00011.tpc","imap_130.tf","imap_science_120.tf","imap_sclk_0161.tsc","de440.bsp","imap_recon_20250925_20260420_v01.bsp","imap_2025_358_2026_085_004.ah.bc","imap_dps_2025_363_2025_365_001.ah.bc","imap_dps_2025_359_2026_115_002.ah.bc"]}]
                """,
                # "imap_swapi_l3a_proton-sw_20260425_v001.json",
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        # process_l3a_proton swallows per-chunk exceptions and writes fill values,
        # so returncode==0 alone doesn't prove the fitter ran. Open the CDF and
        # check that at least one chunk produced a finite, physical solar-wind speed.
        with CDF(str(expected_file_path)) as cdf:
            speed_fill = float(cdf["proton_sw_speed"].attrs["FILLVAL"])
            speeds = np.asarray(cdf["proton_sw_speed"][...], dtype=float)
            temperatures = np.asarray(cdf["proton_sw_temperature"][...], dtype=float)
            densities = np.asarray(cdf["proton_sw_density"][...], dtype=float)
            clock_angles = np.asarray(cdf["proton_sw_clock_angle"][...], dtype=float)
            deflection_angles = np.asarray(
                cdf["proton_sw_deflection_angle"][...], dtype=float
            )
            bulk_vel_sun = np.asarray(
                cdf["proton_sw_bulk_velocity_rtn_sun"][...], dtype=float
            )
            bulk_vel_sc = np.asarray(
                cdf["proton_sw_bulk_velocity_rtn_sc"][...], dtype=float
            )
            speed_uncerts = np.asarray(
                cdf["proton_sw_speed_uncert"][...], dtype=float
            )
            temp_uncerts = np.asarray(
                cdf["proton_sw_temperature_uncert"][...], dtype=float
            )
            density_uncerts = np.asarray(
                cdf["proton_sw_density_uncert"][...], dtype=float
            )
            flags = np.asarray(cdf["swp_flags"][...])

        valid = np.isfinite(speeds) & (speeds != speed_fill)
        finite = speeds[valid]
        self.assertGreater(len(finite), 0, "no chunk produced a finite proton_sw_speed")
        self.assertTrue(
            np.all((finite > 200.0) & (finite < 1500.0)),
            f"finite speeds outside plausible heliospheric range: {finite}",
        )

        # Regression check: spot-check 3 time indices against hardcoded values
        # from a known-good run. Tolerances are 1% relative.
        chk = [0, 4, -1]
        np.testing.assert_allclose(
            speeds[chk],
            [474.635, 475.007, 515.447],
            rtol=0.01,
            err_msg="proton_sw_speed regression",
        )
        np.testing.assert_allclose(
            temperatures[chk],
            [55962.0, 68086.8, 185338.6],
            rtol=0.01,
            err_msg="proton_sw_temperature regression",
        )
        np.testing.assert_allclose(
            densities[chk],
            [2.6931, 3.2717, 4.4771],
            rtol=0.01,
            err_msg="proton_sw_density regression",
        )
        np.testing.assert_allclose(
            clock_angles[chk],
            [56.872, 76.194, 144.707],
            rtol=0.01,
            err_msg="proton_sw_clock_angle regression",
        )
        np.testing.assert_allclose(
            deflection_angles[chk],
            [4.652, 5.220, 6.826],
            rtol=0.01,
            err_msg="proton_sw_deflection_angle regression",
        )
        np.testing.assert_allclose(
            bulk_vel_sun[chk],
            [
                [474.122, 30.273, 17.336],
                [474.777, 38.661, 5.568],
                [512.929, 22.772, -53.293],
            ],
            rtol=0.01,
            err_msg="proton_sw_bulk_velocity_rtn_sun regression",
        )
        np.testing.assert_allclose(
            bulk_vel_sc[chk],
            [
                [474.181, 0.720, 20.762],
                [474.835, 9.108, 8.993],
                [512.981, -6.781, -49.897],
            ],
            rtol=0.01,
            err_msg="proton_sw_bulk_velocity_rtn_sc regression",
        )
        np.testing.assert_allclose(
            speed_uncerts[chk],
            [0.357867, 0.776051, 0.753184],
            rtol=0.01,
            err_msg="proton_sw_speed_uncert regression",
        )
        np.testing.assert_allclose(
            temp_uncerts[chk],
            [1763.507, 4369.083, 6559.847],
            rtol=0.01,
            err_msg="proton_sw_temperature_uncert regression",
        )
        np.testing.assert_allclose(
            density_uncerts[chk],
            [0.049187, 0.134292, 0.126813],
            rtol=0.01,
            err_msg="proton_sw_density_uncert regression",
        )
        np.testing.assert_array_equal(
            flags,
            np.zeros(len(flags), dtype=np.uint16),
            err_msg="swp_flags should all be 0",
        )

    def test_swapi_processor_with_synthetic_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        output_data_dir = get_run_local_data_path("swapi_integration")
        expected_file_path = (
            output_data_dir
            / "imap/swapi/l3a/2026/01/imap_swapi_l3a_proton-sw_20260120_v001.cdf"
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
        swapi_inputs = [
            f
            for f in Path("tests/integration/test_data/swapi").iterdir()
            if f.is_file()
        ]
        swapi_inputs += [
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
            ),
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
            ),
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
            ),
        ]

        os.environ["IMAP_DATA_DIR"] = str(output_data_dir)
        with tempfile.TemporaryDirectory() as tmp:
            l2_cdf = _materialize_retimed_l2(Path(tmp))
            input_files = spice_files + swapi_inputs + [l2_cdf]
            with mock_imap_data_access(output_data_dir, input_files):
                result = subprocess.run(
                    [
                        sys.executable,
                        "imap_l3_data_processor.py",
                        "--instrument",
                        "swapi",
                        "--data-level",
                        "l3a",
                        "--descriptor",
                        "proton-sw",
                        "--start-date",
                        "20260120",
                        "--version",
                        "v001",
                        "--dependency",
                        "imap_swapi_l3a_proton-sw_20260425_v001.json",
                    ]
                )

                self.assertEqual(0, result.returncode)
                self.assertTrue(
                    expected_file_path.exists(),
                    f"expected output CDF not found at {expected_file_path}",
                )

                # process_l3a_proton swallows per-chunk exceptions and writes fill values,
                # so returncode==0 alone doesn't prove the fitter ran. Open the CDF and
                # check that at least one chunk produced a finite, physical solar-wind speed.
                with CDF(str(expected_file_path)) as cdf:
                    speeds = np.asarray(cdf["proton_sw_speed"][...], dtype=float)
                    temperatures = np.asarray(
                        cdf["proton_sw_temperature"][...], dtype=float
                    )
                    densities = np.asarray(cdf["proton_sw_density"][...], dtype=float)
                    clock_angles = np.asarray(
                        cdf["proton_sw_clock_angle"][...], dtype=float
                    )
                    deflection_angles = np.asarray(
                        cdf["proton_sw_deflection_angle"][...], dtype=float
                    )
                    bulk_vel_sun = np.asarray(
                        cdf["proton_sw_bulk_velocity_rtn_sun"][...], dtype=float
                    )
                    bulk_vel_sc = np.asarray(
                        cdf["proton_sw_bulk_velocity_rtn_sc"][...], dtype=float
                    )
                    speed_uncerts = np.asarray(
                        cdf["proton_sw_speed_uncert"][...], dtype=float
                    )
                    temp_uncerts = np.asarray(
                        cdf["proton_sw_temperature_uncert"][...], dtype=float
                    )
                    density_uncerts = np.asarray(
                        cdf["proton_sw_density_uncert"][...], dtype=float
                    )
                    flags = np.asarray(cdf["swp_flags"][...])

                finite = speeds[np.isfinite(speeds)]
                self.assertGreater(
                    len(finite), 0, "no chunk produced a finite proton_sw_speed"
                )
                self.assertTrue(
                    np.all((finite > 200.0) & (finite < 1500.0)),
                    f"finite speeds outside plausible heliospheric range: {finite}",
                )

                # Regression check: spot-check 3 time indices against hardcoded values
                # from a known-good run. Tolerances are 1% relative.
                chk = [0, 4, 9]
                np.testing.assert_allclose(
                    speeds[chk],
                    [492.264, 492.264, 492.264],
                    rtol=0.01,
                    err_msg="proton_sw_speed regression",
                )
                np.testing.assert_allclose(
                    temperatures[chk],
                    [87690.3, 87690.2, 87690.2],
                    rtol=0.01,
                    err_msg="proton_sw_temperature regression",
                )
                np.testing.assert_allclose(
                    densities[chk],
                    [0.30525, 0.30525, 0.30525],
                    rtol=0.01,
                    err_msg="proton_sw_density regression",
                )
                np.testing.assert_allclose(
                    clock_angles[chk],
                    [320.43, 322.47, 320.21],
                    rtol=0.01,
                    err_msg="proton_sw_clock_angle regression",
                )
                np.testing.assert_allclose(
                    deflection_angles[chk],
                    [0, 0, 0],
                    rtol=0.01,
                    err_msg="proton_sw_deflection_angle regression",
                )
                np.testing.assert_allclose(
                    bulk_vel_sun[chk],
                    [
                        [490.866, -8.775, 0.5805],
                        [490.864, -8.799, 0.5827],
                        [490.862, -8.829, 0.5854],
                    ],
                    rtol=0.01,
                    err_msg="proton_sw_bulk_velocity_rtn_sun regression",
                )
                np.testing.assert_allclose(
                    bulk_vel_sc[chk],
                    [
                        [490.741, -38.548, 3.327],
                        [490.739, -38.572, 3.329],
                        [490.737, -38.602, 3.331],
                    ],
                    rtol=0.01,
                    err_msg="proton_sw_bulk_velocity_rtn_sc regression",
                )
                np.testing.assert_allclose(
                    speed_uncerts[chk],
                    [0.5277, 0.5277, 0.5277],
                    rtol=0.01,
                    err_msg="proton_sw_speed_uncert regression",
                )
                np.testing.assert_allclose(
                    temp_uncerts[chk],
                    [3218.9, 3218.9, 3218.9],
                    rtol=0.01,
                    err_msg="proton_sw_temperature_uncert regression",
                )
                np.testing.assert_allclose(
                    density_uncerts[chk],
                    [0.005735, 0.005735, 0.005735],
                    rtol=0.01,
                    err_msg="proton_sw_density_uncert regression",
                )
                np.testing.assert_array_equal(
                    flags,
                    np.zeros(10, dtype=np.uint16),
                    err_msg="swp_flags should all be 0",
                )


if __name__ == "__main__":
    unittest.main()
