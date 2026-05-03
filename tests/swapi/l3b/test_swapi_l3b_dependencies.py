"""Tests for `SwapiL3BDependencies`.

Replaces the previous "mock everything" tests with real fixture loads:
`from_file_paths` is invoked with real file paths and `fetch_dependencies` is
exercised through `mock_imap_data_access` so the returned dependency objects
are *used*, not just checked for type.
"""

import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
)
from spacepy.pycdf import lib as pycdf_lib

from imap_l3_processing.swapi.descriptors import (
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR,
    GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR,
    SWAPI_L2_DESCRIPTOR,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import (
    GeometricFactorCalibrationTable,
)
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path


_L2_SCIENCE = get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")
_GEOMETRIC_FACTOR_SW = get_test_data_path(
    "swapi/imap_swapi_energy-gf-sw-lut_20100101_v001.csv"
)
_EFFICIENCY = get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")

_STAGE_DIR = Path(tempfile.gettempdir()) / "claude" / "swapi_l3b_dep_test_staging"


def _stage_under_name(source: Path, dest_name: str) -> Path:
    _STAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _STAGE_DIR / dest_name
    if not dest.exists() or dest.stat().st_mtime < source.stat().st_mtime:
        shutil.copy(source, dest)
    return dest


class TestFromFilePaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = SwapiL3BDependencies.from_file_paths(
            _L2_SCIENCE, _GEOMETRIC_FACTOR_SW, _EFFICIENCY
        )

    def test_l2_data_is_loaded(self):
        self.assertIsInstance(self.deps.data, SwapiL2Data)
        self.assertEqual(self.deps.data.energy.shape[1], 72)

    def test_geometric_factor_table_is_loaded_and_interpolates(self):
        self.assertIsInstance(
            self.deps.geometric_factor_calibration_table,
            GeometricFactorCalibrationTable,
        )
        gf = self.deps.geometric_factor_calibration_table.lookup_geometric_factor(
            np.array([1000.0, 8000.0])
        )
        self.assertTrue(np.all(np.isfinite(gf)))
        self.assertTrue(np.all(gf > 0))

    def test_efficiency_table_returns_real_value_for_valid_epoch(self):
        self.assertIsInstance(
            self.deps.efficiency_calibration_table, EfficiencyCalibrationTable
        )
        epoch = pycdf_lib.datetime_to_tt2000(datetime(2024, 11, 1))
        result = self.deps.efficiency_calibration_table.get_proton_efficiency_for(epoch)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0.0)


class TestFetchDependenciesEndToEnd(unittest.TestCase):
    def test_fetch_dependencies_returns_usable_dependencies(self):
        start_date = "20100105"
        version = "v010"
        staged_files = [
            _stage_under_name(
                _L2_SCIENCE, f"imap_swapi_l2_sci_{start_date}_{version}.cdf"
            ),
            _stage_under_name(
                _GEOMETRIC_FACTOR_SW,
                f"imap_swapi_energy-gf-sw-lut_{start_date}_{version}.csv",
            ),
            _stage_under_name(
                _EFFICIENCY,
                f"imap_swapi_efficiency-lut_{start_date}_{version}.dat",
            ),
        ]
        collection = ProcessingInputCollection(
            ScienceInput(
                f"imap_swapi_l2_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.csv"
            ),
            AncillaryInput(
                f"imap_swapi_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.dat"
            ),
        )

        with tempfile.TemporaryDirectory() as data_dir:
            with mock_imap_data_access(Path(data_dir), staged_files):
                deps = SwapiL3BDependencies.fetch_dependencies(collection)

        self.assertIsInstance(deps.data, SwapiL2Data)
        self.assertIsInstance(
            deps.geometric_factor_calibration_table, GeometricFactorCalibrationTable
        )
        self.assertIsInstance(
            deps.efficiency_calibration_table, EfficiencyCalibrationTable
        )
        # Lookup tables must be functional.
        epoch = pycdf_lib.datetime_to_tt2000(datetime(2024, 11, 1))
        self.assertGreater(
            deps.efficiency_calibration_table.get_proton_efficiency_for(epoch), 0
        )
        self.assertGreater(
            deps.geometric_factor_calibration_table.lookup_geometric_factor(
                np.array([1000.0])
            )[0],
            0,
        )


if __name__ == "__main__":
    unittest.main()
