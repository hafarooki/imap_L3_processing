"""Direct tests for `passband_grid.py`.

`SWAPIResponse` exercises the full module via its real CSV-loaded grid, but the
boundary-evaluation helpers and `build_passband_grid` are public and load-bearing
on their own — and the latter is built from a pandas `DataFrame` with a strict
`(elevation, energy_ratio)` index that's easy to break silently. These tests
exercise both with synthetic inputs whose expected outputs are computable by
hand.
"""

import unittest

import numpy as np
import pandas as pd

from imap_l3_processing.swapi.l3a.science.passband_grid import (
    _PASSBAND_BOUNDARY_THRESHOLD,
    _TARGET_ELEVATIONS,
    _TARGET_SPEED_RATIOS,
    PassbandGrid,
    build_passband_grid,
    eval_boundary_max,
    eval_boundary_min,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR


class TestEvalBoundaryMin(unittest.TestCase):
    """`eval_boundary_min` returns the most expansive (smallest) of the two nearest
    grid points so the integration window always brackets the active passband."""

    def setUp(self):
        # Two-point boundary at elevations [-2, 2] with min speed-ratios [0.95, 0.92].
        self.boundary = np.array([[-2.0, 2.0], [0.95, 0.92]])

    def test_at_grid_point_returns_min_of_pair(self):
        # At elevation 2.0, idx clamps to last row; idx_next clamps to last; min(0.92, 0.92).
        np.testing.assert_allclose(
            eval_boundary_min(self.boundary, np.array([2.0])), [0.92]
        )

    def test_below_grid_clamps_to_first_row_pair(self):
        np.testing.assert_allclose(
            eval_boundary_min(self.boundary, np.array([-5.0])),
            [min(0.95, 0.92)],
        )

    def test_between_grid_points_takes_smaller(self):
        # Between -2 and 2: idx=0 (val 0.95), idx_next=1 (val 0.92) → 0.92.
        np.testing.assert_allclose(
            eval_boundary_min(self.boundary, np.array([0.0])), [0.92]
        )

    def test_above_grid_clamps_to_last_row(self):
        np.testing.assert_allclose(
            eval_boundary_min(self.boundary, np.array([10.0])), [0.92]
        )

    def test_array_input_returns_per_element_evaluation(self):
        np.testing.assert_allclose(
            eval_boundary_min(self.boundary, np.array([-3.0, 0.0, 5.0])),
            [0.92, 0.92, 0.92],
        )


class TestEvalBoundaryMax(unittest.TestCase):
    """Mirror of `eval_boundary_min` — returns the largest of the two nearest grid points."""

    def setUp(self):
        self.boundary = np.array([[-2.0, 2.0], [1.05, 1.08]])

    def test_at_grid_point_returns_max_of_pair(self):
        np.testing.assert_allclose(
            eval_boundary_max(self.boundary, np.array([-2.0])), [1.08]
        )

    def test_between_grid_points_takes_larger(self):
        np.testing.assert_allclose(
            eval_boundary_max(self.boundary, np.array([0.0])), [1.08]
        )

    def test_above_grid_clamps_to_last_row(self):
        np.testing.assert_allclose(
            eval_boundary_max(self.boundary, np.array([10.0])), [1.08]
        )


class TestBuildPassbandGrid(unittest.TestCase):
    """`build_passband_grid` consumes a per-region `(energy_ratio, elevation)` -indexed
    DataFrame whose `value` column is the passband response. Verify the resulting
    grid has the documented shape, that boundary arrays bracket the active region,
    and that elevation ranges correspond to the rows above the threshold."""

    def _values_df(self, peak_at_elevation: float, peak_at_speed_ratio: float = 1.0):
        """Build a synthetic per-region passband DataFrame with a single Gaussian
        bump centered at `(peak_at_elevation, peak_at_speed_ratio)`. The DataFrame
        is indexed on `energy_ratio = SWAPI_K_FACTOR * speed_ratio**2` (the convention
        the source uses to map back to speed ratio internally)."""
        elevations = np.arange(-15.0, 15.0 + 0.5, 0.5)
        # Speed ratios spanning the production target [0.9, 1.1] with margin.
        speed_ratios = np.linspace(0.85, 1.15, 41)
        peak_er = SWAPI_K_FACTOR * peak_at_speed_ratio**2
        rows = []
        for elev in elevations:
            for sr in speed_ratios:
                er = SWAPI_K_FACTOR * sr**2
                v = np.exp(
                    -((elev - peak_at_elevation) ** 2) / (2.0 * 1.0**2)
                    - ((er - peak_er) ** 2) / (2.0 * 0.1**2)
                )
                rows.append(
                    {
                        "elevation": float(elev),
                        "energy_ratio": float(er),
                        "value": float(v),
                    }
                )
        return pd.DataFrame(rows).set_index(["energy_ratio", "elevation"])

    def test_grid_has_target_shape(self):
        oa = self._values_df(peak_at_elevation=0.0)
        sg = self._values_df(peak_at_elevation=0.0)
        grid = build_passband_grid(oa_values=oa, sg_values=sg)
        self.assertIsInstance(grid, PassbandGrid)
        self.assertEqual(
            grid.values_open_aperture.shape,
            (len(_TARGET_ELEVATIONS), len(_TARGET_SPEED_RATIOS)),
        )
        self.assertEqual(grid.values_sunglasses.shape, grid.values_open_aperture.shape)

    def test_grid_origin_and_spacing_match_targets(self):
        oa = self._values_df(0.0)
        sg = self._values_df(0.0)
        grid = build_passband_grid(oa_values=oa, sg_values=sg)
        self.assertAlmostEqual(grid.min_elevation, float(_TARGET_ELEVATIONS[0]))
        self.assertAlmostEqual(
            grid.elevation_spacing,
            float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
        )
        self.assertAlmostEqual(grid.min_speed_ratio, float(_TARGET_SPEED_RATIOS[0]))
        self.assertAlmostEqual(
            grid.speed_ratio_spacing,
            float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]),
        )

    def test_boundary_brackets_above_threshold_region(self):
        oa = self._values_df(peak_at_elevation=0.0)
        sg = self._values_df(peak_at_elevation=0.0)
        grid = build_passband_grid(oa_values=oa, sg_values=sg)
        cutoff = _PASSBAND_BOUNDARY_THRESHOLD * grid.values_open_aperture.max()
        center_idx = int(round((0.0 - grid.min_elevation) / grid.elevation_spacing))
        row = grid.values_open_aperture[center_idx]
        sr_min = float(eval_boundary_min(grid.min_OA_boundary, np.array([0.0]))[0])
        sr_max = float(eval_boundary_max(grid.max_OA_boundary, np.array([0.0]))[0])
        for ratio, val in zip(_TARGET_SPEED_RATIOS, row):
            if ratio < sr_min - 1e-9 or ratio > sr_max + 1e-9:
                self.assertLessEqual(
                    val,
                    cutoff + 1e-9,
                    msg=f"row value {val} at speed_ratio {ratio} exceeds cutoff {cutoff}",
                )

    def test_elevation_range_brackets_active_rows(self):
        oa = self._values_df(peak_at_elevation=2.5)
        sg = self._values_df(peak_at_elevation=-3.5)
        grid = build_passband_grid(oa_values=oa, sg_values=sg)
        oa_lo, oa_hi = grid.oa_elevation_range
        sg_lo, sg_hi = grid.sg_elevation_range
        self.assertLess(oa_lo, 2.5)
        self.assertGreater(oa_hi, 2.5)
        self.assertLess(sg_lo, -3.5)
        self.assertGreater(sg_hi, -3.5)

    def test_inactive_grid_collapses_elevation_range(self):
        # All-zero values → no row exceeds the cutoff (which is also zero); active
        # range collapses to a single point at target_elevations[0].
        elevations = np.arange(-15.0, 15.0 + 0.5, 0.5)
        speed_ratios = np.linspace(0.85, 1.15, 41)
        rows = []
        for elev in elevations:
            for sr in speed_ratios:
                rows.append(
                    {
                        "elevation": float(elev),
                        "energy_ratio": float(SWAPI_K_FACTOR * sr**2),
                        "value": 0.0,
                    }
                )
        oa = pd.DataFrame(rows).set_index(["energy_ratio", "elevation"])
        grid = build_passband_grid(oa_values=oa, sg_values=oa)
        self.assertEqual(grid.oa_elevation_range[0], grid.oa_elevation_range[1])
        self.assertEqual(grid.oa_elevation_range, grid.sg_elevation_range)

    def test_speed_ratio_indexing_via_k_factor(self):
        # Source converts energy_ratio → speed_ratio via sqrt(er/k). Build a DF whose
        # peak in energy_ratio is exactly k * (1.0)^2 = k, and confirm the resulting grid
        # peaks at speed_ratio = 1.0 — the central elevation column should peak near
        # speed_ratio = 1.0 (target index ~50 of 101).
        oa = self._values_df(peak_at_elevation=0.0, peak_at_speed_ratio=1.0)
        grid = build_passband_grid(oa_values=oa, sg_values=oa)
        center_row_idx = int(round((0.0 - grid.min_elevation) / grid.elevation_spacing))
        row = grid.values_open_aperture[center_row_idx]
        peak_idx = int(np.argmax(row))
        peak_speed_ratio = _TARGET_SPEED_RATIOS[peak_idx]
        self.assertAlmostEqual(peak_speed_ratio, 1.0, places=2)
        self.assertGreater(row.max(), 0.5)


if __name__ == "__main__":
    unittest.main()
