import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import numpy as np
import spacepy.pycdf
from spacepy.pycdf import CDF
from uncertainties import UFloat, ufloat

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_BIN_PERIOD_S,
    SWAPI_K_FACTOR,
    SWAPI_LIVETIME_CENTER_OFFSET_S,
    SWAPI_LIVETIME_S,
    SWAPI_SWEEP_BIN_COUNT,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import (
    velocity_components_to_angles_in_instrument_frame,
    calculate_sw_speed,
    chunk_l2_data,
    esa_voltage_to_alpha_speed,
    esa_voltage_to_proton_speed,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
    pickup_ion_chunk_epoch,
    read_l2_swapi_data,
    read_mag_rtn_data,
    velocity_to_angles_in_instrument_frame,
)
from tests.spice_test_case import SpiceTestCase
from tests.swapi._helpers import proton_params


def _analytic_speed_km_per_s(
    voltage: float, mass_kg: float, charge_c: float
) -> float:
    return float(
        np.sqrt(2 * SWAPI_K_FACTOR * charge_c * abs(voltage) / mass_kg)
        / METERS_PER_KILOMETER
    )


class TestChunkL2Data(TestCase):
    """Tests for `chunk_l2_data`."""

    def test_chunk_l2_data(self):
        """chunk_l2_data splits a 4-sweep L2 dataset into two 2-sweep chunks, copying epoch, energy, count rate, and uncertainty into each chunk."""
        epoch = np.array([0, 1, 2, 3])
        energy = np.array([[15000, 16000, 17000, 18000, 19000],
                           [25000, 26000, 27000, 28000, 29000],
                           [35000, 36000, 37000, 38000, 39000],
                           [45000, 46000, 47000, 48000, 49000], ]
                          )
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        data = SwapiL2Data(epoch, energy, coincidence_count_rate, coincidence_count_rate_uncertainty)
        chunks = list(chunk_l2_data(data, 2))

        expected_energy_chunk_1 = np.array([[15000, 16000, 17000, 18000, 19000],
                                            [25000, 26000, 27000, 28000, 29000]])
        expected_energy_chunk_2 = np.array([[35000, 36000, 37000, 38000, 39000],
                                            [45000, 46000, 47000, 48000, 49000]])

        expected_count_rate_chunk_1 = np.array([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        expected_count_rate_uncertainty_chunk_1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        first_chunk = chunks[0]

        np.testing.assert_array_equal(first_chunk.sci_start_time, np.array([0, 1]))
        np.testing.assert_array_equal(expected_energy_chunk_1, first_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_1, first_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_1,
                                      first_chunk.coincidence_count_rate_uncertainty)

        expected_count_rate_chunk_2 = np.array([[14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        expected_count_rate_uncertainty_chunk_2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        second_chunk = chunks[1]
        np.testing.assert_array_equal(np.array([2, 3]), second_chunk.sci_start_time)
        np.testing.assert_array_equal(expected_energy_chunk_2, second_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_2, second_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_2,
                                      second_chunk.coincidence_count_rate_uncertainty)

    def test_chunk_l2_data_partial_trailing_chunk_is_dropped(self):
        """When the sweep count is not a multiple of the chunk size, the trailing partial chunk is dropped rather than yielded short."""
        rates = np.arange(7 * 4, dtype=float).reshape(7, 4)
        data = SwapiL2Data(
            sci_start_time=np.arange(7, dtype=np.int64),
            energy=rates.copy(),
            coincidence_count_rate=rates.copy(),
            coincidence_count_rate_uncertainty=rates.copy(),
        )
        chunks = list(chunk_l2_data(data, 5))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0].sci_start_time, np.arange(5))


class TestMeasurementTimes(TestCase):
    """Tests for `measurement_times`, which timestamps each ESA step at the center of its livetime."""

    def test_each_bin_is_offset_from_its_sweep_start_by_the_livetime_center(self):
        """Every bin of the sweep is timestamped, one row per sweep, at its sweep start plus the bin index times the bin period plus the fixed livetime-center offset."""
        sweep_period_ns = 12 * ONE_SECOND_IN_NANOSECONDS
        sci_start_time = np.array([1_000, 1_000 + sweep_period_ns], dtype=np.int64)

        times = measurement_times(sci_start_time)

        def expected(sweep_start, bin_index):
            seconds_into_sweep = bin_index * SWAPI_BIN_PERIOD_S + SWAPI_LIVETIME_CENTER_OFFSET_S
            return sweep_start + seconds_into_sweep * ONE_SECOND_IN_NANOSECONDS

        self.assertEqual(times.shape, (2, SWAPI_SWEEP_BIN_COUNT))
        np.testing.assert_allclose(times[0, 0], expected(1_000, 0))
        np.testing.assert_allclose(times[0, 71], expected(1_000, 71))
        np.testing.assert_allclose(times[1, 0], expected(1_000 + sweep_period_ns, 0))

    def test_offset_is_the_livetime_center_not_its_start(self):
        """The per-bin offset lands half a livetime past the end of the ramp-up, i.e. at the center of the livetime window rather than its start."""
        ramp_up_s = SWAPI_BIN_PERIOD_S - SWAPI_LIVETIME_S
        self.assertAlmostEqual(SWAPI_LIVETIME_CENTER_OFFSET_S, ramp_up_s + SWAPI_LIVETIME_S / 2)
        self.assertAlmostEqual(SWAPI_LIVETIME_CENTER_OFFSET_S, SWAPI_BIN_PERIOD_S - SWAPI_LIVETIME_S / 2)


class TestReadL2SwapiData(TestCase):
    """Tests for `read_l2_swapi_data`."""

    def tearDown(self) -> None:
        if os.path.exists('temp_cdf.cdf'):
            os.remove('temp_cdf.cdf')

    def test_reading_l2_data_into_model(self):
        """read_l2_swapi_data parses a CDF into SwapiL2Data, decoding the start time to TT2000 and replacing each variable's FILLVAL entries with NaN."""
        path = Path('temp_cdf.cdf')
        if path.exists():
            os.remove(path)

        temp_cdf = CDF('temp_cdf', '')
        temp_cdf["sci_start_time"] = np.array(['2010-01-01T00:00:46.000'])
        temp_cdf["esa_energy"] = np.array([1, -1e31, 3, 4], dtype=float)
        temp_cdf["swp_coin_rate"] = np.array([5, 6, 7, -1e31], dtype=float)
        temp_cdf["swp_coin_rate_stat_uncert_plus"] = np.array([2, 2, -1e31, 2, 2, 2, 2, 2], dtype=float)

        temp_cdf["sci_start_time"].attrs["FILLVAL"] = '0'
        temp_cdf["esa_energy"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate_stat_uncert_plus"].attrs["FILLVAL"] = -1e31

        temp_cdf.close()

        actual_swapi_l2_data = read_l2_swapi_data(CDF("temp_cdf.cdf"))

        epoch_as_tt2000 = 315576112184000000
        np.testing.assert_array_equal(np.array(epoch_as_tt2000), actual_swapi_l2_data.sci_start_time)
        np.testing.assert_array_equal(np.array([1, np.nan, 3, 4]), actual_swapi_l2_data.energy)
        np.testing.assert_array_equal(np.array([5, 6, 7, np.nan]), actual_swapi_l2_data.coincidence_count_rate)
        np.testing.assert_array_equal(np.array([2, 2, np.nan, 2, 2, 2, 2, 2]),
                                      actual_swapi_l2_data.coincidence_count_rate_uncertainty)


class TestCalculateSwSpeed(TestCase):
    """Tests for `calculate_sw_speed`."""

    def test_2d_array_matches_analytic_formula_per_element(self):
        """calculate_sw_speed on a 2D energy array matches the analytic v = sqrt(2qE/m) (in km/s) element-by-element."""
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        expected = (
            np.sqrt(2 * E * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        np.testing.assert_allclose(result, expected)

    def test_2d_array_input_preserves_shape(self):
        """calculate_sw_speed preserves the input array shape (no flattening or broadcasting collapse)."""
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertEqual(result.shape, E.shape)

    def test_empty_array_input_returns_empty_array(self):
        """calculate_sw_speed returns an empty array when given one, without errors."""
        result = calculate_sw_speed(
            PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, np.array([])
        )
        self.assertEqual(result.size, 0)

    def test_ufloat_scalar_propagates_uncertainty(self):
        """A scalar ufloat energy returns a ufloat speed with uncertainty σ_v = v · σ_E / (2E) per first-order error propagation."""
        E = ufloat(1.0e-16, 1.0e-18)
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertIsInstance(result, UFloat)
        # σ_v = v · σ_E / (2 E)
        expected_nom = (
            np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        expected_sigma = expected_nom * E.std_dev / (2 * E.nominal_value)
        self.assertAlmostEqual(result.nominal_value, expected_nom)
        self.assertAlmostEqual(result.std_dev, expected_sigma)

    def test_ufloat_array_input_propagates_uncertainty_per_element(self):
        """A numpy array of UFloat energies returns per-element UFloat speeds with each element's own σ_E correctly propagated."""
        E_values = np.array([ufloat(1.0e-16, 1.0e-18), ufloat(4.0e-16, 2.0e-18)])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E_values)
        self.assertEqual(result.shape, E_values.shape)
        for r, E in zip(result, E_values):
            self.assertIsInstance(r, UFloat)
            expected_nom = (
                np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
                / METERS_PER_KILOMETER
            )
            self.assertAlmostEqual(r.nominal_value, expected_nom)
            self.assertAlmostEqual(
                r.std_dev, expected_nom * E.std_dev / (2 * E.nominal_value)
            )


class TestReadMagRtnData(TestCase):
    """Tests for `read_mag_rtn_data`."""

    def setUp(self) -> None:
        self.cdf_path = Path('temp_mag_cdf.cdf')
        if self.cdf_path.exists():
            os.remove(self.cdf_path)

    def tearDown(self) -> None:
        if self.cdf_path.exists():
            os.remove(self.cdf_path)

    def test_reads_b_rtn_and_epoch_into_mag_data(self):
        """read_mag_rtn_data converts CDF epochs to TT2000 and keeps only the leading three vector components of b_rtn (dropping the magnitude column)."""
        epochs = np.array([datetime(2026, 1, 1, 0, 0, 0),
                           datetime(2026, 1, 1, 0, 0, 1)])
        b_rtn = np.array(
            [[1.0, 2.0, 3.0, 0.0],
             [4.0, 5.0, 6.0, 0.0]],
        )
        cdf = CDF(str(self.cdf_path.with_suffix("")), '')
        cdf["epoch"] = epochs
        cdf["b_rtn"] = b_rtn
        cdf["b_rtn"].attrs["FILLVAL"] = -1e31
        cdf.close()

        result = read_mag_rtn_data(self.cdf_path)

        expected_epoch_tt2000 = spacepy.pycdf.lib.v_datetime_to_tt2000(epochs)
        np.testing.assert_array_equal(result.epoch, expected_epoch_tt2000)
        np.testing.assert_array_equal(
            result.mag_data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )


class TestSwapiSpiceHelpers(SpiceTestCase):
    """Tests for the SPICE helpers `get_swapi_geometry` and `get_spacecraft_velocity_rtn`."""

    # 2025-06-06 12:00 UTC — inside the IMAP SPK and attitude coverage windows
    # used by the shipped `spice_kernels/` set.
    _EPOCH_TT2000_NS = spacepy.pycdf.lib.datetime_to_tt2000(
        datetime(2025, 6, 6, 12, 0, 0)
    )

    def test_get_swapi_geometry_returns_orthonormal_rotation_per_time(self):
        """get_swapi_geometry returns one proper-orthogonal (right-handed) rotation matrix per input time across a 30 s window."""
        # Three TT2000 ns samples spanning ~30 s.
        # IMAP's spin period is 15 s,
        # so these should produce distinct rotations.
        times = self._EPOCH_TT2000_NS + np.array([0, 10_000_000_000, 20_000_000_000])

        rotation_matrices = get_swapi_geometry(times)

        self.assertEqual(rotation_matrices.shape, (3, 3, 3))
        for matrix in rotation_matrices:
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=10)

    def test_get_spacecraft_velocity_rtn_returns_finite_orbital_velocity(self):
        """get_spacecraft_velocity_rtn returns a finite 3-vector with magnitude in the 10–60 km/s band, ruling out unit-conversion mistakes for IMAP near L1."""
        velocity_rtn = get_spacecraft_velocity_rtn(self._EPOCH_TT2000_NS)

        self.assertEqual(velocity_rtn.shape, (3,))
        self.assertTrue(np.all(np.isfinite(velocity_rtn)))
        # IMAP sits near L1; its barycentric speed is dominated by Earth's
        # ~30 km/s orbital motion. A loose bound rules out unit-conversion
        # mistakes (m/s vs km/s) without pinning a frame-specific value.
        speed = float(np.linalg.norm(velocity_rtn))
        self.assertGreater(speed, 10.0)
        self.assertLess(speed, 60.0)


class TestVelocityComponentsToAnglesInInstrumentFrame(TestCase):
    """Tests for `velocity_components_to_angles_in_instrument_frame`. The Cartesian input
    is the flow direction; the returned (azimuth, elevation) is the look
    direction (opposite the flow)."""

    def test_flow_along_minus_y_returns_zero_angles(self):
        """A flow along -Y means SWAPI looks toward +Y, which is the (azimuth=0, elevation=0) bore-sight in the instrument frame."""
        azimuth, elevation = velocity_components_to_angles_in_instrument_frame(0.0, -450.0, 0.0)
        self.assertAlmostEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_positive_x_flow_yields_negative_azimuth(self):
        """A flow with a positive X component (look direction in -X) gives a negative azimuth, since azimuth = atan2(-vx, -vy) < 0 when vx > 0 and vy < 0."""
        azimuth, _ = velocity_components_to_angles_in_instrument_frame(50.0, -450.0, 0.0)
        self.assertLess(azimuth, 0.0)

    def test_positive_z_flow_yields_negative_elevation(self):
        """A flow with a positive Z component (look direction in -Z) gives a negative elevation, since elevation = asin(-vz/|v|) < 0 when vz > 0."""
        _, elevation = velocity_components_to_angles_in_instrument_frame(0.0, -450.0, 50.0)
        self.assertLess(elevation, 0.0)

    def test_negative_z_flow_yields_positive_elevation(self):
        """A flow with a negative Z component (look direction in +Z) gives a positive elevation."""
        _, elevation = velocity_components_to_angles_in_instrument_frame(0.0, -450.0, -50.0)
        self.assertGreater(elevation, 0.0)

    def test_pure_plus_x_flow_yields_azimuth_minus_90_elevation_zero(self):
        """A pure +X flow has look direction along -X; azimuth = atan2(-450, 0) = -90 deg, elevation = 0."""
        azimuth, elevation = velocity_components_to_angles_in_instrument_frame(450.0, 0.0, 0.0)
        self.assertAlmostEqual(azimuth, -90.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_pure_plus_y_flow_yields_azimuth_180(self):
        """A pure +Y flow (away from SWAPI) has look direction along -Y; azimuth = atan2(0, -450) = +/-180 deg, elevation = 0."""
        azimuth, elevation = velocity_components_to_angles_in_instrument_frame(0.0, 450.0, 0.0)
        self.assertAlmostEqual(abs(azimuth), 180.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_pure_plus_z_flow_yields_elevation_minus_90(self):
        """A pure +Z flow has look direction along -Z; elevation = asin(-1) = -90 deg."""
        _, elevation = velocity_components_to_angles_in_instrument_frame(0.0, 0.0, 450.0)
        self.assertAlmostEqual(elevation, -90.0)


class TestBulkVelocityToAnglesInInstrumentFrame(TestCase):
    """Tests for `velocity_to_angles_in_instrument_frame`, which rotates a
    SolarWindParams bulk velocity from RTN into the SWAPI XYZ frame and returns
    look-direction (azimuth_deg, elevation_deg). See
    `velocity_to_angles_in_instrument_frame` for the flow-vs-look convention."""

    def test_velocity_along_minus_y_inst_returns_zero_angles(self):
        """Under identity rotation, an RTN flow along -Y becomes v_inst = (0, -450, 0); look direction is +Y so azimuth = 0 and elevation = 0."""
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0))
        azimuth, elevation = velocity_to_angles_in_instrument_frame(sw, np.eye(3))
        self.assertAlmostEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_velocity_with_negative_x_inst_component_yields_positive_azimuth(self):
        """A negative instrument-X flow component (v_inst_x = -50) gives a strictly positive azimuth, since azimuth = atan2(+50, +450) > 0."""
        sw = proton_params(velocity_rtn=(-50.0, -450.0, 0.0))
        azimuth, _ = velocity_to_angles_in_instrument_frame(sw, np.eye(3))
        self.assertGreater(azimuth, 0.0)

    def test_velocity_with_positive_z_inst_component_yields_negative_elevation(self):
        """A positive instrument-Z flow component (v_inst_z = +50) gives a strictly negative elevation, since elevation = asin(-50/|v|) < 0."""
        sw = proton_params(velocity_rtn=(0.0, -450.0, 50.0))
        _, elevation = velocity_to_angles_in_instrument_frame(sw, np.eye(3))
        self.assertLess(elevation, 0.0)

    def test_uses_rotation_argument_to_transform_into_instrument_frame(self):
        """When the rotation maps +Y_inst to -X_RTN, an RTN flow along -X_RTN lands along +Y_inst and produces azimuth = atan2(0, -1) = +/-180 deg, catching regressions that drop the rotation."""
        rotation_xyz_to_rtn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        sw = proton_params(velocity_rtn=(-450.0, 0.0, 0.0))
        azimuth, _ = velocity_to_angles_in_instrument_frame(sw, rotation_xyz_to_rtn)
        self.assertAlmostEqual(abs(azimuth), 180.0, places=10)


class TestEsaVoltageToProtonSpeed(TestCase):
    """Tests for `esa_voltage_to_proton_speed`."""

    def test_matches_analytical_formula(self):
        """A sweep of representative ESA voltages reproduces the closed-form `sqrt(2 K q V / m)` proton speed to machine precision."""
        for V in [200.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_proton_speed(V),
                    _analytic_speed_km_per_s(
                        V, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS
                    ),
                    rtol=1e-12,
                )

    def test_matches_hardcoded_reference_value_at_1000_volts(self):
        """At V = 1000 V the function returns ~601.730748 km/s, the value pre-computed offline from `sqrt(2 · 1.89 · q_p · 1000 / m_p) / 1000`. This pins the absolute magnitude against a manually verifiable number, not just the live formula."""
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(1000.0), 601.7307477647, rtol=1e-9
        )

    def test_handles_negative_voltage(self):
        """A negative ESA voltage yields the same proton speed as its positive counterpart, since the conversion depends on magnitude."""
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-1000.0), esa_voltage_to_proton_speed(1000.0)
        )


class TestEsaVoltageToAlphaSpeed(TestCase):
    """Tests for `esa_voltage_to_alpha_speed`."""

    def test_matches_analytical_formula(self):
        """A sweep of representative ESA voltages reproduces the closed-form `sqrt(2 K q V / m)` alpha speed to machine precision."""
        for V in [200.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_alpha_speed(V),
                    _analytic_speed_km_per_s(
                        V, ALPHA_PARTICLE_MASS_KG, ALPHA_PARTICLE_CHARGE_COULOMBS
                    ),
                    rtol=1e-12,
                )

    def test_matches_hardcoded_reference_value_at_1000_volts(self):
        """At V = 1000 V the function returns ~426.952735 km/s, the value pre-computed offline from `sqrt(2 · 1.89 · 2·q_p · 1000 / m_alpha) / 1000`. This pins the absolute magnitude against a manually verifiable number, not just the live formula."""
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(1000.0), 426.9527347202, rtol=1e-9
        )

    def test_handles_negative_voltage(self):
        """A negative ESA voltage yields the same alpha speed as its positive counterpart, since the conversion depends on magnitude."""
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )


class TestPuiChunkEpoch(TestCase):
    def test_ten_minute_chunk_is_centered_five_minutes_in(self):
        """A PUI chunk's fifty 12 s sweeps span ten minutes from the first sweep
        start, so its center is 5 minutes in — matching the +/- 5 minute
        `epoch_delta` the chunk is reported with."""
        first_start = 800_000_000_000_000_000
        sweep_starts = (
            first_start
            + np.arange(50, dtype=np.int64) * int(12 * ONE_SECOND_IN_NANOSECONDS)
        )
        empty_sweeps = np.zeros((50, SWAPI_SWEEP_BIN_COUNT))
        chunk = SwapiL2Data(
            sci_start_time=sweep_starts,
            energy=empty_sweeps,
            coincidence_count_rate=empty_sweeps,
            coincidence_count_rate_uncertainty=empty_sweeps,
        )

        self.assertEqual(
            pickup_ion_chunk_epoch(chunk), first_start + 5 * 60 * ONE_SECOND_IN_NANOSECONDS
        )

    def test_center_is_half_a_sweep_past_the_midpoint_of_the_sweep_starts(self):
        """The chunk runs one whole sweep past its last sweep start, so its
        center sits half a sweep beyond the midpoint of the start times — the
        half sweep that separates a sweep's start from its own center."""
        first_start = 800_000_000_000_000_000
        sweep_starts = (
            first_start
            + np.arange(50, dtype=np.int64) * int(12 * ONE_SECOND_IN_NANOSECONDS)
        )
        empty_sweeps = np.zeros((50, SWAPI_SWEEP_BIN_COUNT))
        chunk = SwapiL2Data(
            sci_start_time=sweep_starts,
            energy=empty_sweeps,
            coincidence_count_rate=empty_sweeps,
            coincidence_count_rate_uncertainty=empty_sweeps,
        )
        midpoint_of_starts = int((sweep_starts[0] + sweep_starts[-1]) // 2)

        self.assertEqual(
            pickup_ion_chunk_epoch(chunk) - midpoint_of_starts,
            6 * ONE_SECOND_IN_NANOSECONDS,
        )
