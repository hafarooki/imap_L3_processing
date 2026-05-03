"""Tests for `imap_l3_processing.swapi.l3a.utils`.

Covers L2/MAG file readers, the data-chunk helpers, the SPICE-frame helpers
(`get_swapi_geometry`, `get_spacecraft_velocity_rtn`, `rotate_rtn_to_dps`),
and the per-bin time helpers (`measurement_times`, `chunk_epoch`). SPICE
helpers run against the locally furnished kernels in `spice_kernels/`.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.constants import (
    ONE_SECOND_IN_NANOSECONDS,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.models import MagL1dData
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.l3a.utils import (
    chunk_epoch,
    chunk_l2_data,
    compute_b_hat_rtn,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
    read_l1d_mag_data,
    read_l2_swapi_data,
    rotate_rtn_to_dps,
)
from tests.spice_test_case import SpiceTestCase


def _tt2000(year, month=1, day=1, hour=0, minute=0, second=0):
    return pycdf.lib.datetime_to_tt2000(
        datetime(year, month, day, hour, minute, second)
    )


def _make_l2_data(n=4, n_bins=5):
    return SwapiL2Data(
        sci_start_time=np.arange(n, dtype=np.int64),
        energy=np.tile(
            np.arange(15_000.0, 15_000.0 + n_bins * 1_000.0, 1_000.0), (n, 1)
        )
        + np.arange(n)[:, None] * 10_000.0,
        coincidence_count_rate=np.arange(n * n_bins, dtype=float).reshape(n, n_bins),
        coincidence_count_rate_uncertainty=np.full((n, n_bins), 0.5),
    )


class TestChunkL2Data(TestCase):
    def test_yields_chunks_of_requested_size_with_matching_arrays(self):
        data = _make_l2_data(n=4)
        chunks = list(chunk_l2_data(data, 2))
        self.assertEqual(len(chunks), 2)
        for chunk_idx, chunk in enumerate(chunks):
            self.assertEqual(len(chunk.sci_start_time), 2)
            np.testing.assert_array_equal(
                chunk.sci_start_time, [chunk_idx * 2, chunk_idx * 2 + 1]
            )
            np.testing.assert_array_equal(
                chunk.energy, data.energy[chunk_idx * 2 : chunk_idx * 2 + 2]
            )
            np.testing.assert_array_equal(
                chunk.coincidence_count_rate,
                data.coincidence_count_rate[chunk_idx * 2 : chunk_idx * 2 + 2],
            )
            np.testing.assert_array_equal(
                chunk.coincidence_count_rate_uncertainty,
                data.coincidence_count_rate_uncertainty[
                    chunk_idx * 2 : chunk_idx * 2 + 2
                ],
            )

    def test_drops_trailing_partial_chunk(self):
        data = _make_l2_data(n=5)
        chunks = list(chunk_l2_data(data, 2))
        # 5 sweeps with chunk_size=2 → only 2 full chunks (the trailing single sweep is dropped).
        self.assertEqual(len(chunks), 2)

    def test_chunk_size_one_yields_one_per_sweep(self):
        data = _make_l2_data(n=3)
        chunks = list(chunk_l2_data(data, 1))
        self.assertEqual(len(chunks), 3)


class TestChunkEpoch(TestCase):
    def test_returns_first_sweep_start_plus_30_seconds(self):
        chunk = SwapiL2Data(
            sci_start_time=np.array([1_000_000_000, 2_000_000_000]),
            energy=np.zeros((2, 5)),
            coincidence_count_rate=np.zeros((2, 5)),
            coincidence_count_rate_uncertainty=np.zeros((2, 5)),
        )
        # Chunk epoch is centered 30 s after the first sweep — half of a 5-sweep × 12 s window.
        self.assertEqual(
            chunk_epoch(chunk), 1_000_000_000 + THIRTY_SECONDS_IN_NANOSECONDS
        )


class TestMeasurementTimes(TestCase):
    """`measurement_times(chunk, bin_slice)` returns one TT2000 timestamp per
    `(sweep, bin)` pair: `t_i = sweep_start + bin_index · (12/72) s`."""

    def test_returns_per_sweep_per_bin_timestamps(self):
        chunk = SwapiL2Data(
            sci_start_time=np.array(
                [0, 12 * ONE_SECOND_IN_NANOSECONDS], dtype=np.int64
            ),
            energy=np.zeros((2, 72)),
            coincidence_count_rate=np.zeros((2, 72)),
            coincidence_count_rate_uncertainty=np.zeros((2, 72)),
        )
        times = measurement_times(chunk, SWAPI_SCIENCE_BINS)
        self.assertEqual(times.shape, (2 * 71,))
        # First bin of first sweep: 0 + 1 * (12/72)s = 1/6 s in nanoseconds.
        bin_step_ns = 12 / 72 * ONE_SECOND_IN_NANOSECONDS
        np.testing.assert_allclose(times[0], 1 * bin_step_ns)
        # Last bin of first sweep: 71 * bin_step.
        np.testing.assert_allclose(times[70], 71 * bin_step_ns)
        # First bin of second sweep: 12 s + 1 * bin_step.
        np.testing.assert_allclose(
            times[71], 12 * ONE_SECOND_IN_NANOSECONDS + 1 * bin_step_ns
        )

    def test_coarse_slice_yields_n_sweeps_times_62_entries(self):
        chunk = SwapiL2Data(
            sci_start_time=np.zeros(3, dtype=np.int64),
            energy=np.zeros((3, 72)),
            coincidence_count_rate=np.zeros((3, 72)),
            coincidence_count_rate_uncertainty=np.zeros((3, 72)),
        )
        times = measurement_times(chunk, SWAPI_COARSE_SWEEP_BINS)
        self.assertEqual(times.shape, (3 * 62,))


class TestReadL2SwapiData(TestCase):
    def tearDown(self):
        for p in [Path("temp_cdf.cdf")]:
            if p.exists():
                p.unlink()

    def test_replaces_fillval_with_nan(self):
        path = Path("temp_cdf.cdf")
        if path.exists():
            os.remove(path)
        cdf = CDF("temp_cdf", "")
        cdf["sci_start_time"] = np.array(["2010-01-01T00:00:46.000"])
        cdf["esa_energy"] = np.array([1, -1e31, 3, 4], dtype=float)
        cdf["swp_coin_rate"] = np.array([5, 6, 7, -1e31], dtype=float)
        cdf["swp_coin_rate_stat_uncert_plus"] = np.array(
            [2, 2, -1e31, 2, 2, 2, 2, 2], dtype=float
        )
        for var in [
            "sci_start_time",
            "esa_energy",
            "swp_coin_rate",
            "swp_coin_rate_stat_uncert_plus",
        ]:
            cdf[var].attrs["FILLVAL"] = -1e31 if var != "sci_start_time" else "0"
        cdf.close()

        data = read_l2_swapi_data(CDF("temp_cdf.cdf"))
        np.testing.assert_array_equal(
            data.sci_start_time, np.array([315576112184000000])
        )
        np.testing.assert_array_equal(data.energy, [1, np.nan, 3, 4])
        np.testing.assert_array_equal(data.coincidence_count_rate, [5, 6, 7, np.nan])
        np.testing.assert_array_equal(
            data.coincidence_count_rate_uncertainty, [2, 2, np.nan, 2, 2, 2, 2, 2]
        )


class TestReadL1dMagData(TestCase):
    def tearDown(self):
        for p in [Path("temp_cdf.cdf")]:
            if p.exists():
                p.unlink()

    def test_validmin_and_validmax_become_nan(self):
        path = Path("temp_cdf.cdf")
        if path.exists():
            os.remove(path)
        cdf = CDF("temp_cdf", "")
        cdf["epoch"] = [datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1)]
        cdf["b_rtn"] = np.array([[1.0, 2.0, 3.0], [2.0e5, -2.0e5, 4.0]])
        cdf["b_rtn"].attrs["FILLVAL"] = -1e31
        cdf["b_rtn"].attrs["VALIDMIN"] = -1.0e5
        cdf["b_rtn"].attrs["VALIDMAX"] = 1.0e5
        cdf.close()
        data = read_l1d_mag_data(path)
        np.testing.assert_allclose(
            data.mag_data, [[1.0, 2.0, 3.0], [np.nan, np.nan, 4.0]]
        )


class TestComputeBHatRtn(TestCase):
    def test_averages_in_window_and_normalizes(self):
        mag = MagL1dData(
            epoch=np.array([0, 6, 10, 20]),
            mag_data=np.array(
                [
                    [100.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 100.0],
                ]
            ),
        )
        # Window [10-5, 10+5) = [5, 15). Includes epochs 6 and 10 → mean (1, 1, 0), normalized.
        np.testing.assert_allclose(
            compute_b_hat_rtn(mag, 10, 5), np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        )

    def test_returns_nan_when_mag_is_none(self):
        result = compute_b_hat_rtn(None, 10, 5)
        self.assertTrue(np.all(np.isnan(result)))

    def test_returns_nan_when_window_empty(self):
        mag = MagL1dData(
            epoch=np.array([0, 100]),
            mag_data=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        # Window [50-1, 50+1) excludes both samples.
        self.assertTrue(np.all(np.isnan(compute_b_hat_rtn(mag, 50, 1))))

    def test_returns_nan_when_in_window_samples_have_nan(self):
        mag = MagL1dData(
            epoch=np.array([5, 10]),
            mag_data=np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        self.assertTrue(np.all(np.isnan(compute_b_hat_rtn(mag, 10, 10))))

    def test_returns_nan_when_averaged_b_is_below_norm_threshold(self):
        mag = MagL1dData(
            epoch=np.array([5, 10]),
            mag_data=np.array([[1e-15, 0.0, 0.0], [-1e-15, 0.0, 0.0]]),
        )
        self.assertTrue(np.all(np.isnan(compute_b_hat_rtn(mag, 10, 10))))


class TestSpiceFrameHelpers(SpiceTestCase):
    """Tests for the SPICE-backed frame transforms. Uses kernels furnished from
    `spice_kernels/` (matching the production pipeline) to exercise actual
    transforms — not mocks of `spiceypy.sxform`."""

    EPOCH_TT2000 = _tt2000(2025, 9, 25, 12, 0, 0)

    def test_get_swapi_geometry_returns_orthonormal_rotation_per_epoch(self):
        epochs = np.array(
            [self.EPOCH_TT2000, self.EPOCH_TT2000 + ONE_SECOND_IN_NANOSECONDS]
        )
        rotations = get_swapi_geometry(epochs)
        self.assertEqual(rotations.shape, (2, 3, 3))
        for i in range(2):
            with self.subTest(epoch_idx=i):
                # Orthonormal: R @ R.T == I (within numerical tolerance).
                np.testing.assert_allclose(
                    rotations[i] @ rotations[i].T, np.eye(3), atol=1e-12
                )
                # Determinant +1 (proper rotation, not a reflection).
                np.testing.assert_allclose(np.linalg.det(rotations[i]), 1.0, atol=1e-12)

    def test_get_swapi_geometry_returns_smoothly_varying_rotation(self):
        # Two epochs spaced by 1 s: the matrices should differ but only slightly,
        # since the spacecraft spin period is 15 s and the rotation per second is
        # ~24 deg around the spin axis.
        epochs = np.array(
            [self.EPOCH_TT2000, self.EPOCH_TT2000 + ONE_SECOND_IN_NANOSECONDS]
        )
        rotations = get_swapi_geometry(epochs)
        diff = rotations[1] - rotations[0]
        self.assertGreater(np.linalg.norm(diff), 1e-6)
        self.assertLess(np.linalg.norm(diff), 2.0)

    def test_get_spacecraft_velocity_rtn_returns_three_component_velocity(self):
        sc_velocity = get_spacecraft_velocity_rtn(self.EPOCH_TT2000)
        self.assertEqual(sc_velocity.shape, (3,))
        # IMAP orbits the Sun-Earth L1 region → magnitude on the order of Earth orbital
        # velocity (~30 km/s) but in the kinematic RTN frame the radial component is
        # near zero and the tangential is small. Sanity-check that components are finite
        # and within plausible physical magnitudes.
        self.assertTrue(np.all(np.isfinite(sc_velocity)))
        self.assertLess(np.linalg.norm(sc_velocity), 100.0)

    def test_rotate_rtn_to_dps_round_trips_via_inverse(self):
        # rotate_rtn_to_dps applied to an arbitrary vector then inverted via
        # imap_processing.spice.geometry.frame_transform with the inverse arguments
        # should recover the original vector.
        from imap_processing.spice.geometry import SpiceFrame, frame_transform
        from imap_processing.spice.time import ttj2000ns_to_et

        v_rtn = np.array([1.0, 2.0, -3.0])
        v_dps = rotate_rtn_to_dps(v_rtn, self.EPOCH_TT2000)
        et = float(ttj2000ns_to_et(self.EPOCH_TT2000))
        v_round_trip = frame_transform(
            et, v_dps, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_RTN
        )
        np.testing.assert_allclose(v_round_trip, v_rtn, atol=1e-12)

    def test_rotate_rtn_to_dps_preserves_norm(self):
        for v in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([3.0, -1.0, 5.0]),
        ]:
            with self.subTest(vector=v):
                rotated = rotate_rtn_to_dps(v, self.EPOCH_TT2000)
                np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v))


if __name__ == "__main__":
    import unittest

    unittest.main()
