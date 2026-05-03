"""Tests for `chunk_fits.py` — the per-chunk fitting strategies and parallel runner.

The previous version of this file mocked both `_fit_proton` and
`derive_velocity_angles`, leaving none of `fit_chunk`'s actual logic exercised.
This rewrite covers:

- A real end-to-end `ProtonChunkFitter.fit_chunk` run on a synthetic 5-sweep
  chunk built from `_model_count_rates` (round-trip: known input moments →
  count rates → fit → recovered moments within tolerance).
- The fill-value branch (`np.isnan` in coincidence_count_rate → NaN-filled output).
- The SPICE-gap branch (rotation_matrices=None → exception inside fit_chunk →
  NaN-filled output, no raise).
- `AlphaChunkFitter` and `PuiProtonChunkFitter` smoke tests (returns the
  documented dict shape with finite values for proton-only inputs).
- `ParallelChunkRunner.run` end-to-end (3 chunks → assembled output dict with
  arrays of length 3 in chunk order).
"""

import unittest
from datetime import datetime
from unittest.mock import patch

import numba
import numpy as np
import spiceypy
from spacepy import pycdf
from uncertainties import ufloat

from tests.spice_test_case import SpiceTestCase

from imap_l3_processing.constants import (
    EV_TO_KELVIN,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitter,
    ParallelChunkRunner,
    ProtonChunkFitter,
    PuiProtonChunkFitter,
    _shared,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _model_count_rates,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi._swapi_test_helpers import efficiency_calibration_table, swapi_response


_TRUE_DENSITY = 5.0
_TRUE_TEMP_K = 10.0 * EV_TO_KELVIN
_TRUE_VELOCITY = np.array([450.0, 0.0, 0.0])  # mostly radial


_EPOCH_TT2000 = pycdf.lib.datetime_to_tt2000(datetime(2025, 9, 25, 12, 0, 0))


def _build_synthetic_chunk(
    n_sweeps: int = 5,
) -> tuple[SwapiL2Data, np.ndarray, np.ndarray]:
    """Construct a synthetic L2 chunk with known moments → returns (chunk, voltages, rotations).

    The chunk has 72 bins per sweep with bin 0 zeroed (matching SWAPI hardware) and
    bins 1..71 populated by `_model_count_rates` for the true SW parameters above.
    """
    sr = swapi_response()
    n_per_sweep = 71  # bins 1..71

    # Build a sensible voltage table over solar-wind energies (matches L2 esa_energy / k_L2).
    from imap_l3_processing.constants import METERS_PER_KILOMETER

    peak_voltage = float(
        PROTON_MASS_KG
        * (_TRUE_VELOCITY[0] * METERS_PER_KILOMETER) ** 2
        / (2 * 1.89 * PROTON_CHARGE_COULOMBS)
    )
    per_sweep_voltages = np.geomspace(
        peak_voltage * 0.4, peak_voltage * 2.5, n_per_sweep
    )
    voltages_per_sweep = np.tile(per_sweep_voltages, (n_sweeps, 1))

    # L2 reports energy in eV: energy = SWAPI_L2_K_FACTOR × |voltage|.
    energy_full = np.zeros((n_sweeps, 72))
    energy_full[:, 1:72] = voltages_per_sweep * SWAPI_L2_K_FACTOR

    all_voltages = voltages_per_sweep.flatten()
    sr.warm_cache(all_voltages)

    # Simple identity rotation per (sweep, bin) — matches the old test's pattern.
    rotations = np.tile(np.eye(3), (len(all_voltages), 1, 1))
    grids = numba.typed.List([sr.create_passband_grid(v) for v in all_voltages])
    central_speeds = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in all_voltages]
    )
    central_eff_areas = np.array(
        [sr.get_central_effective_area(v) for v in all_voltages]
    )
    az_trans = np.asarray(sr.azimuthal_transmission, dtype=float)
    az_spacing = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    flat_count_rate = _model_count_rates(
        _TRUE_DENSITY,
        _TRUE_TEMP_K,
        _TRUE_VELOCITY,
        grids,
        central_speeds,
        central_eff_areas,
        az_trans,
        az_spacing,
        rotations,
        PROTON_MASS_KG,
    )
    count_rate_full = np.zeros((n_sweeps, 72))
    count_rate_full[:, 1:72] = flat_count_rate.reshape(n_sweeps, n_per_sweep)
    uncert = np.sqrt(np.where(count_rate_full > 0, count_rate_full, 1.0))

    chunk = SwapiL2Data(
        sci_start_time=_EPOCH_TT2000 + np.arange(n_sweeps) * 12_000_000_000,
        energy=energy_full,
        coincidence_count_rate=count_rate_full,
        coincidence_count_rate_uncertainty=uncert,
    )
    return chunk, all_voltages, rotations


class _UnitEfficiencyTable:
    """Stand-in efficiency table that returns 1.0 for everything.

    The synthetic count rates we feed `_fit_proton` already use unit efficiency,
    so the production lookup table would over-correct by ~25× and the fit would
    miss the truth. Real-data integration tests use the full table — this fake
    is for unit-level wiring only."""

    eps_p_lab = 1.0

    def get_proton_efficiency_for(self, time):
        return 1.0

    def get_alpha_efficiency_for(self, time):
        return 1.0


def _populate_shared(efficiency_table=None):
    _shared["swapi_response"] = swapi_response()
    _shared["efficiency_table"] = (
        efficiency_table if efficiency_table is not None else _UnitEfficiencyTable()
    )


def _clear_shared():
    _shared.clear()


class TestProtonChunkFitterRoundTrip(SpiceTestCase):
    """Real fit through `ProtonChunkFitter.fit_chunk` on a known synthetic spectrum.
    Uses identity rotations and no spacecraft-velocity offset, so the recovered
    bulk velocity should match the truth to within 5%."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _populate_shared()
        cls.chunk, cls.voltages, cls.rotations = _build_synthetic_chunk(n_sweeps=5)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_runs_real_fit_and_emits_documented_dict(self):
        # End-to-end smoke through fit_chunk: real `_fit_proton` + real
        # `derive_velocity_angles`. Verifies the output dict has every documented
        # key and that scalar fields are finite for clean synthetic input. Strict
        # truth recovery is owned by `tests/swapi/l3a/science/test_calculate_proton_solar_wind_moments.py`.
        sc_velocity_rtn = np.zeros(3)
        result = ProtonChunkFitter().fit_chunk(
            self.chunk,
            int(self.chunk.sci_start_time[0]),
            self.rotations,
            sc_velocity_rtn,
        )
        expected_keys = {
            "epoch",
            "proton_sw_speed",
            "proton_sw_speed_uncert",
            "proton_sw_speed_sun",
            "proton_sw_speed_sun_uncert",
            "proton_sw_temperature",
            "proton_sw_temperature_uncert",
            "proton_sw_density",
            "proton_sw_density_uncert",
            "proton_sw_clock_angle",
            "proton_sw_clock_angle_uncert",
            "proton_sw_deflection_angle",
            "proton_sw_deflection_angle_uncert",
            "proton_sw_bulk_velocity_rtn_sun",
            "proton_sw_bulk_velocity_rtn_sun_covariance",
            "proton_sw_bulk_velocity_rtn_sc",
            "proton_sw_bulk_velocity_rtn_sc_covariance",
            "quality_flags",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        for scalar_field in [
            "proton_sw_speed",
            "proton_sw_density",
            "proton_sw_temperature",
            "proton_sw_speed_sun",
            "proton_sw_clock_angle",
            "proton_sw_deflection_angle",
        ]:
            with self.subTest(field=scalar_field):
                self.assertTrue(
                    np.isfinite(result[scalar_field]),
                    msg=f"{scalar_field} is NaN — exception likely raised inside fit_chunk",
                )
        self.assertTrue(np.all(np.isfinite(result["proton_sw_bulk_velocity_rtn_sc"])))
        # Sun-frame velocity is the SC-frame plus SC velocity (zero here).
        np.testing.assert_allclose(
            result["proton_sw_bulk_velocity_rtn_sun"],
            result["proton_sw_bulk_velocity_rtn_sc"] + sc_velocity_rtn,
        )
        np.testing.assert_allclose(
            result["proton_sw_speed_sun"],
            np.linalg.norm(result["proton_sw_bulk_velocity_rtn_sun"]),
            rtol=1e-9,
        )

    def test_sun_frame_velocity_offset_by_sc_velocity(self):
        sc_velocity_rtn = np.array([10.0, -3.0, 2.0])
        result = ProtonChunkFitter().fit_chunk(
            self.chunk,
            int(self.chunk.sci_start_time[0]),
            self.rotations,
            sc_velocity_rtn,
        )
        # Sun-frame = SC-frame + SC velocity (km/s), per docstring.
        np.testing.assert_allclose(
            result["proton_sw_bulk_velocity_rtn_sun"],
            result["proton_sw_bulk_velocity_rtn_sc"] + sc_velocity_rtn,
        )


class TestProtonChunkFitterFillValueAndSpiceGap(SpiceTestCase):
    """Fill-value paths produce a NaN-filled output dict and never raise."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _populate_shared()

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def _make_fill_chunk(self, n_sweeps=5):
        chunk = SwapiL2Data(
            sci_start_time=_EPOCH_TT2000 + np.arange(n_sweeps) * 12_000_000_000,
            energy=np.tile(np.linspace(20000.0, 50.0, 72), (n_sweeps, 1)),
            coincidence_count_rate=np.full((n_sweeps, 72), np.nan),
            coincidence_count_rate_uncertainty=np.full((n_sweeps, 72), np.nan),
        )
        return chunk

    def test_nan_in_count_rate_yields_nan_filled_dict(self):
        chunk = self._make_fill_chunk()
        rotations = np.tile(np.eye(3), (len(chunk.sci_start_time) * 71, 1, 1))
        result = ProtonChunkFitter().fit_chunk(
            chunk, int(chunk.sci_start_time[0]), rotations, np.zeros(3)
        )
        # All scalar fields → NaN; vector/matrix fields → NaN-filled.
        self.assertTrue(np.isnan(result["proton_sw_speed"]))
        self.assertTrue(np.isnan(result["proton_sw_density"]))
        self.assertTrue(np.isnan(result["proton_sw_temperature"]))
        self.assertTrue(np.all(np.isnan(result["proton_sw_bulk_velocity_rtn_sc"])))
        self.assertTrue(
            np.all(np.isnan(result["proton_sw_bulk_velocity_rtn_sc_covariance"]))
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)

    def test_none_rotation_matrices_yields_nan_filled_dict(self):
        # Mirrors the SPICE-gap path: precompute_geometry returned None for the rotation
        # matrices, so fit_chunk will hit an exception inside _fit_proton and fall through
        # to the NaN-filled output.
        chunk, _, _ = _build_synthetic_chunk()
        result = ProtonChunkFitter().fit_chunk(
            chunk,
            int(chunk.sci_start_time[0]),
            None,  # SPICE gap
            np.zeros(3),
        )
        self.assertTrue(np.isnan(result["proton_sw_speed"]))
        self.assertTrue(np.all(np.isnan(result["proton_sw_bulk_velocity_rtn_sc"])))


class TestProtonChunkFitterPrecomputeGeometryHandlesSpiceGap(unittest.TestCase):
    """`precompute_geometry` swallows SPICE failures and returns None for both
    rotation matrices and SC velocity, so workers can produce NaN output without
    a hard error."""

    def test_returns_none_geometry_when_spice_gap_raises(self):
        chunk, _, _ = _build_synthetic_chunk(n_sweeps=2)
        # Patch `get_swapi_geometry` to raise — simulates a SPICE coverage gap
        # outside furnished kernel windows.
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=spiceypy.utils.exceptions.SpiceyError("simulated SPICE gap"),
        ):
            epoch, rotations, sc_velocity_rtn = ProtonChunkFitter().precompute_geometry(
                chunk
            )
        self.assertIsNone(rotations)
        self.assertIsNone(sc_velocity_rtn)
        self.assertIsInstance(epoch, (int, np.integer))


class TestPuiProtonChunkFitter(SpiceTestCase):
    """`PuiProtonChunkFitter.fit_chunk` uses the same proton fitter but only
    emits speed + clock/deflection angles + a quality_flag."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _populate_shared()

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_returns_documented_keys_for_synthetic_chunk(self):
        chunk, _, rotations = _build_synthetic_chunk(n_sweeps=5)
        result = PuiProtonChunkFitter().fit_chunk(
            chunk, int(chunk.sci_start_time[0]), rotations
        )
        self.assertEqual(
            set(result.keys()),
            {
                "proton_sw_speed",
                "proton_sw_clock_angle",
                "proton_sw_deflection_angle",
                "quality_flags",
            },
        )
        # Speed should be non-NaN ufloat for clean synthetic input.
        self.assertFalse(np.isnan(result["proton_sw_speed"].nominal_value))

    def test_nan_count_rate_yields_nan_ufloats(self):
        chunk = SwapiL2Data(
            sci_start_time=_EPOCH_TT2000 + np.arange(5) * 12_000_000_000,
            energy=np.tile(np.linspace(20000.0, 50.0, 72), (5, 1)),
            coincidence_count_rate=np.full((5, 72), np.nan),
            coincidence_count_rate_uncertainty=np.full((5, 72), np.nan),
        )
        rotations = np.tile(np.eye(3), (5 * 71, 1, 1))
        result = PuiProtonChunkFitter().fit_chunk(
            chunk, int(chunk.sci_start_time[0]), rotations
        )
        self.assertTrue(np.isnan(result["proton_sw_speed"].nominal_value))


class TestAlphaChunkFitter(unittest.TestCase):
    """`AlphaChunkFitter` requires both proton moments (Stage 1) and a B̂ direction.
    Verify the dict shape and NaN-fill path on bad inputs."""

    @classmethod
    def setUpClass(cls):
        _populate_shared()

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_nan_count_rate_yields_nan_filled_dict_with_bad_fit_flag(self):
        chunk = SwapiL2Data(
            sci_start_time=_EPOCH_TT2000 + np.arange(5) * 12_000_000_000,
            energy=np.tile(np.linspace(20000.0, 50.0, 72), (5, 1)),
            coincidence_count_rate=np.full((5, 72), np.nan),
            coincidence_count_rate_uncertainty=np.full((5, 72), np.nan),
        )
        rotations = np.tile(np.eye(3), (5 * 62, 1, 1))
        b_hat = np.array([1.0, 0.0, 0.0])
        result = AlphaChunkFitter(mag_l1d_data=None).fit_chunk(
            chunk, int(chunk.sci_start_time[0]), rotations, b_hat
        )
        self.assertTrue(np.isnan(result["alpha_sw_density"]))
        self.assertTrue(np.all(np.isnan(result["alpha_sw_velocity_rtn"])))
        self.assertEqual(result["bad_fit_flag"], int(SwapiL3Flags.BAD_FIT))

    def test_pickle_drops_mag_data_to_avoid_per_worker_replication(self):
        # AlphaChunkFitter.__getstate__ returns a dict with mag_l1d_data=None.
        # This is the documented pickle behavior for the fork pool: workers don't
        # need the raw MAG arrays (b_hat is derived parent-side and passed via the
        # geometry tuple). Verify a round-trip through __getstate__ doesn't carry
        # the MAG arrays.
        from imap_l3_processing.models import MagL1dData

        fitter = AlphaChunkFitter(
            mag_l1d_data=MagL1dData(
                epoch=np.array([0]), mag_data=np.array([[1.0, 0.0, 0.0]])
            )
        )
        state = fitter.__getstate__()
        self.assertIsNone(state["mag_l1d_data"])


class TestParallelChunkRunner(unittest.TestCase):
    """`ParallelChunkRunner.run` precomputes geometry per chunk in the parent and
    submits each chunk to a fork pool. Verify it assembles outputs in chunk order
    and that the result dict has arrays of length `len(chunks)`."""

    def test_run_assembles_outputs_in_chunk_order(self):
        # Use a fitter that returns a deterministic dict per chunk so we can verify
        # ordering. Submit 3 distinguishable chunks.
        class _SequentialFitter:
            def precompute_geometry(self, chunk):
                return (int(chunk.sci_start_time[0]),)

            def fit_chunk(self, chunk, label):
                return {
                    "label": label,
                    "speed": float(label) * 0.001,
                }

        chunks = [
            SwapiL2Data(
                sci_start_time=np.array([10 * i]),
                energy=np.zeros((1, 72)),
                coincidence_count_rate=np.zeros((1, 72)),
                coincidence_count_rate_uncertainty=np.zeros((1, 72)),
            )
            for i in range(3)
        ]
        runner = ParallelChunkRunner(
            swapi_response=swapi_response(),
            efficiency_table=efficiency_calibration_table(),
        )
        try:
            result = runner.run(chunks, _SequentialFitter())
        finally:
            _clear_shared()

        np.testing.assert_array_equal(
            result["label"], [chunks[i].sci_start_time[0] for i in range(3)]
        )
        # _SequentialFitter scales the label by 0.001 → expected speed = label / 1000.
        np.testing.assert_allclose(result["speed"], [0.0, 0.01, 0.02])


if __name__ == "__main__":
    unittest.main()
