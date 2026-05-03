"""Tests for `GeometricFactorCalibrationTable`."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import (
    GeometricFactorCalibrationTable,
)
from tests.test_helpers import get_test_data_path


_SW_LUT_PATH = get_test_data_path("swapi/imap_swapi_energy-gf-sw-lut_20100101_v001.csv")


class TestGeometricFactorTableFromFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = GeometricFactorCalibrationTable.from_file(_SW_LUT_PATH)

    def test_grid_has_62_entries(self):
        # SW geometric-factor LUT covers 62 ESA bins (matches SWAPI_COARSE_SWEEP_BINS).
        self.assertEqual(len(self.table.grid), 62)
        self.assertEqual(self.table.geometric_factor_grid.shape, (62,))

    def test_lookup_returns_exact_value_at_grid_point(self):
        # First grid energy → first geometric factor value (np.interp returns the
        # tabulated value when input lands on a grid point).
        first_energy = float(self.table.grid[0])
        first_gf = float(self.table.geometric_factor_grid[0])
        np.testing.assert_allclose(
            self.table.lookup_geometric_factor(first_energy), first_gf, rtol=1e-12
        )

    def test_known_lookup_values(self):
        # Pinned values from the calibration file.
        self.assertEqual(
            self.table.lookup_geometric_factor(8165.393844536367), 6.419796603112413e-13
        )
        np.testing.assert_allclose(
            self.table.lookup_geometric_factor(14194.87288073211),
            5.711128783363629e-13,
            rtol=1e-6,
        )

    def test_below_grid_clamps_to_first_value(self):
        # `np.interp` clamps to endpoint values for inputs outside the grid range.
        below_min = self.table.grid[0] - 50.0
        np.testing.assert_allclose(
            self.table.lookup_geometric_factor(float(below_min)),
            float(self.table.geometric_factor_grid[0]),
        )

    def test_above_grid_clamps_to_last_value(self):
        above_max = self.table.grid[-1] + 1000.0
        np.testing.assert_allclose(
            self.table.lookup_geometric_factor(float(above_max)),
            float(self.table.geometric_factor_grid[-1]),
        )

    def test_array_input_returns_per_element_lookup(self):
        sample_energies = self.table.grid[[0, 10, 30, 60]]
        result = self.table.lookup_geometric_factor(sample_energies)
        np.testing.assert_allclose(
            result, self.table.geometric_factor_grid[[0, 10, 30, 60]]
        )


class TestGeometricFactorTableSyntheticInputs(unittest.TestCase):
    def _write_csv(self, energies, factors):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write('"en","gf"\n')
            for e, g in zip(energies, factors):
                f.write(f"{e},{g}\n")
            return Path(f.name)

    def test_linear_interpolation_between_two_points(self):
        path = self._write_csv([100.0, 200.0], [1.0e-12, 2.0e-12])
        try:
            table = GeometricFactorCalibrationTable.from_file(path)
            # Midpoint → 1.5e-12.
            np.testing.assert_allclose(
                table.lookup_geometric_factor(150.0), 1.5e-12, rtol=1e-12
            )
        finally:
            path.unlink()

    def test_single_row_table_is_not_supported_via_from_file(self):
        # `np.loadtxt` collapses a single data row to 1D, which trips the (N,2) indexing
        # in `__init__`. The CSV format is documented as multi-row; this regression test
        # pins the failure mode so a future change that adds support is intentional.
        path = self._write_csv([1000.0], [3.5e-12])
        try:
            with self.assertRaises(IndexError):
                GeometricFactorCalibrationTable.from_file(path)
        finally:
            path.unlink()

    def test_two_row_table_at_boundary_returns_endpoints(self):
        path = self._write_csv([100.0, 200.0], [1.0e-12, 2.0e-12])
        try:
            table = GeometricFactorCalibrationTable.from_file(path)
            np.testing.assert_allclose(table.lookup_geometric_factor(100.0), 1.0e-12)
            np.testing.assert_allclose(table.lookup_geometric_factor(200.0), 2.0e-12)
        finally:
            path.unlink()

    def test_constructor_accepts_raw_array(self):
        # `from_file` parses CSV; the bare constructor takes (N, 2) array directly.
        table = GeometricFactorCalibrationTable(
            np.array([[100.0, 1.0e-12], [200.0, 2.0e-12]])
        )
        np.testing.assert_array_equal(table.grid, [100.0, 200.0])
        np.testing.assert_array_equal(table.geometric_factor_grid, [1.0e-12, 2.0e-12])


if __name__ == "__main__":
    unittest.main()
