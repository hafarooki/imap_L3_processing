"""Tests for `EfficiencyCalibrationTable`.

The class tracks both proton and alpha efficiency vs. time-since-launch and
exposes a special `eps_p_lab` property that pins the lab calibration epoch
to the first entry on/after 2025-11-01 (with a documented fallback to the first
row when no such entry exists). The `_eff_scale` callers in
`chunk_fits.py` divide by this property, so a regression there silently
shifts every L3a moment fit's density by the inverse ratio.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf

from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from tests.test_helpers import get_test_data_path


_TEST_LUT_PATH = get_test_data_path(
    "swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat"
)


def _tt2000(year, month=1, day=1):
    return pycdf.lib.datetime_to_tt2000(datetime(year, month, day))


class TestProtonEfficiencyLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = EfficiencyCalibrationTable(_TEST_LUT_PATH)

    def test_returns_first_entry_for_time_at_first_record(self):
        # First record is 2000-01-01T12:00; queries on 2001-02-01 fall AFTER it,
        # so the loop walks back from the end and finds an earlier match.
        # Last entry strictly before 2001-02-01 → first row with H+ = 0.1.
        self.assertEqual(self.table.get_proton_efficiency_for(_tt2000(2001, 2, 1)), 0.1)

    def test_returns_value_from_immediately_preceding_record(self):
        self.assertEqual(
            self.table.get_proton_efficiency_for(_tt2000(2014, 10, 3)), 0.09
        )

    def test_returns_value_from_third_record(self):
        self.assertEqual(
            self.table.get_proton_efficiency_for(_tt2000(2024, 10, 3)), 0.0882
        )

    def test_handles_float_input(self):
        self.assertEqual(
            self.table.get_proton_efficiency_for(float(_tt2000(2001, 2, 1))), 0.1
        )

    def test_raises_for_time_before_table_start(self):
        with self.assertRaises(ValueError) as ctx:
            self.table.get_proton_efficiency_for(_tt2000(1999, 1, 4))
        self.assertIn("No efficiency data", str(ctx.exception))


class TestAlphaEfficiencyLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = EfficiencyCalibrationTable(_TEST_LUT_PATH)

    def test_returns_first_entry_value(self):
        self.assertEqual(self.table.get_alpha_efficiency_for(_tt2000(2001, 2, 1)), 0.9)

    def test_returns_value_from_second_record(self):
        self.assertEqual(
            self.table.get_alpha_efficiency_for(_tt2000(2014, 10, 3)), 0.95
        )

    def test_returns_value_from_third_record(self):
        self.assertEqual(
            self.table.get_alpha_efficiency_for(_tt2000(2024, 10, 3)), 0.99
        )

    def test_handles_float_input(self):
        self.assertEqual(
            self.table.get_alpha_efficiency_for(float(_tt2000(2001, 2, 1))), 0.9
        )

    def test_raises_for_time_before_table_start(self):
        with self.assertRaises(ValueError):
            self.table.get_alpha_efficiency_for(_tt2000(1999, 1, 4))


class TestEpsPLabProperty(unittest.TestCase):
    """`eps_p_lab` returns the first proton efficiency on/after 2025-11-01,
    falling back to `data[0]['proton efficiency']` when no later row exists.
    Exercised by both branches with synthetic LUT files."""

    def _write_lut(self, rows):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("# Time (UTC)  MET  H+ Efficiency  He++ Efficiency\n")
            for row in rows:
                f.write(row + "\n")
            return Path(f.name)

    def test_returns_post_cutoff_value_when_present(self):
        path = self._write_lut(
            [
                "2024-10-02T16:45:21.000  14013924050000  0.0882  0.99",
                "2025-11-01T00:00:00.000  16000000000000  0.105   0.95",
                "2026-01-01T00:00:00.000  17000000000000  0.110   0.97",
            ]
        )
        try:
            table = EfficiencyCalibrationTable(path)
            self.assertAlmostEqual(table.eps_p_lab, 0.105)
        finally:
            path.unlink()

    def test_returns_value_at_cutoff_boundary(self):
        # 2025-11-01 exactly satisfies `>= cutoff`.
        path = self._write_lut(
            [
                "2025-10-31T23:59:59.999  15999000000000  0.099  0.97",
                "2025-11-01T00:00:00.000  16000000000000  0.105  0.95",
            ]
        )
        try:
            table = EfficiencyCalibrationTable(path)
            self.assertAlmostEqual(table.eps_p_lab, 0.105)
        finally:
            path.unlink()

    def test_fallback_to_first_entry_when_all_pre_cutoff(self):
        # Production v000 LUT (all pre-2025-11-01) — verified directly.
        v000 = EfficiencyCalibrationTable(
            get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")
        )
        # First row's H+ efficiency = 0.02348.
        self.assertAlmostEqual(v000.eps_p_lab, 0.02348)


class TestTableData(unittest.TestCase):
    """Sanity checks on the loaded structured array — pins schema if loadtxt dtype
    declaration changes."""

    def test_dtype_field_names(self):
        table = EfficiencyCalibrationTable(_TEST_LUT_PATH)
        names = table.data.dtype.names
        self.assertEqual(
            names,
            ("time", "MET", "proton efficiency", "alpha efficiency"),
        )

    def test_data_is_sorted_by_time(self):
        table = EfficiencyCalibrationTable(_TEST_LUT_PATH)
        times = table.data["time"]
        self.assertTrue(np.all(times[:-1] <= times[1:]))


if __name__ == "__main__":
    unittest.main()
