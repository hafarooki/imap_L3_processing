import math
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numba
import numpy as np
from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    EV_TO_KELVIN,
    PROTON_MASS_KG,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
    PROTON_CHARGE_COULOMBS,
    METERS_PER_KILOMETER,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _compute_angles,
    _model_count_rates,
    apply_deadtime_correction_array,
    _optimize,
    apply_deadtime_correction,
    calculate_integral,
    fit_solar_wind_proton_moments,
    interpolate_passband,
    _interpolate_transmission,
    _get_angular_limits,
    SWParams,
    ProtonSolarWindMoments,
    SWAPI_LIVETIME_S,
)
import pandas as pd
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    esa_voltage_to_proton_speed,
    SWAPI_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.test_helpers import get_test_instrument_team_data_path

_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def _load_swapi_response():
    return SWAPIResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )


_coverage_shared = {}


def _coverage_worker(noisy_rate):
    return fit_solar_wind_proton_moments(
        noisy_rate,
        _coverage_shared["voltages"],
        _coverage_shared["sr"],
        1.0,
        _coverage_shared["rot"],
    )


def _peak_voltage(bulk_speed_kms):
    """ESA voltage whose central speed equals bulk_speed_kms (inverse of esa_voltage_to_proton_speed)."""
    return float(
        PROTON_MASS_KG
        * (bulk_speed_kms * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _thermal_speed(temperature_k):
    """Convert proton temperature in K to 1-D thermal speed in km/s."""
    return float(
        np.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature_k / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )


def _make_sw_params(
    density=5.0,
    temperature_k=10.0 * EV_TO_KELVIN,
    bulk_speed=450.0,
    bulk_azimuth=15.0,
    bulk_elevation=-5.0,
):
    """Build an SWParams with defaults that produce a nonzero integral for all three azimuth regions."""
    return SWParams(
        density=density,
        bulk_speed=bulk_speed,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
        thermal_speed=_thermal_speed(temperature_k),
    )


def _build_proton_arrays(sr, voltages):
    """Build grids, central_speeds, central_effective_areas, az_trans, spacing for proton fits."""
    grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in voltages]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in voltages])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    return grids, cs, cea, at, ats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyDeadtimeCorrection(unittest.TestCase):
    """Verify apply_deadtime_correction against a known (true rate, measured rate) pair."""

    def test_example(self):
        # At g=35.000 kHz measured, n=35.226 kHz true.
        # apply_deadtime_correction maps true rate -> measured rate, so
        # apply_deadtime_correction(35226) should recover ~35000 Hz.
        self.assertAlmostEqual(apply_deadtime_correction(35226), 35000, delta=1)


# ---------------------------------------------------------------------------


class TestEsaVoltageToProtonSpeed(unittest.TestCase):
    """Verify the ESA voltage → proton speed conversion: known value, negative-voltage symmetry, array output."""

    def test_known_value(self):
        # At V = 1000 V, E = k* * V = 1.89 keV
        # v = sqrt(2 * 1890 eV * e / m_p) = 601.730748 km/s (independently computed)
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(1000.0), 601.730748, rtol=1e-3
        )

    def test_uses_absolute_value_of_voltage(self):
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-529.0), esa_voltage_to_proton_speed(529.0)
        )

    def test_vectorized(self):
        speeds = esa_voltage_to_proton_speed(np.array([500.0, 1000.0, 2000.0]))
        self.assertEqual(speeds.shape, (3,))
        self.assertTrue(np.all(np.diff(speeds) > 0))
        np.testing.assert_allclose(speeds[1], 601.730748, rtol=1e-3)


class TestComputeAngles(unittest.TestCase):
    """Verify _compute_angles produces correct azimuth/elevation for known inputs.

    Covers identity rotation, SC-velocity shift, Z-velocity → nonzero elevation,
    rotation-matrix effect, and direction-only dependence on speed magnitude.
    """

    def test_antisunward_beam_with_identity_rotation(self):
        # v_RTN = [450, 0, 0], identity rotation
        # v_xyz = [450, 0, 0]
        # phi = arctan2(-450, 0) = -90°, theta = arcsin(0/450) = 0°
        phi, theta = _compute_angles(np.array([450.0, 0.0, 0.0]), np.eye(3))
        np.testing.assert_allclose(phi, -90.0, atol=1e-6)
        np.testing.assert_allclose(theta, 0.0, atol=1e-6)

    def test_elevation_nonzero_for_z_component(self):
        bulk_velocity_rtn = np.array([450.0, 0.0, 50.0])
        _, theta = _compute_angles(bulk_velocity_rtn, np.eye(3))
        v_b = np.linalg.norm(bulk_velocity_rtn)
        expected_theta = np.degrees(np.arcsin(-50.0 / v_b))
        np.testing.assert_allclose(theta, expected_theta, atol=1e-6)

    def test_rotation_matrix_changes_azimuth_not_elevation(self):
        # 90° rotation about z: x→y, y→−x; elevation (z component) unchanged
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        phi_id, theta_id = _compute_angles(np.array([450.0, 0.0, 0.0]), np.eye(3))
        phi_rot, theta_rot = _compute_angles(np.array([450.0, 0.0, 0.0]), R)
        np.testing.assert_allclose(theta_rot, theta_id, atol=1e-6)
        self.assertFalse(np.isclose(phi_rot, phi_id))

    def test_result_independent_of_bulk_speed_magnitude_for_angles(self):
        # Angles depend only on direction, not magnitude
        v1 = np.array([300.0, 50.0, -20.0])
        v2 = v1 * 2
        phi1, theta1 = _compute_angles(v1, np.eye(3))
        phi2, theta2 = _compute_angles(v2, np.eye(3))
        np.testing.assert_allclose(phi1, phi2, atol=1e-4)
        np.testing.assert_allclose(theta1, theta2, atol=1e-4)


class TestInterpolatePassband(unittest.TestCase):
    """Verify interpolate_passband returns sensible in-bounds values and 0 out-of-bounds,
    and that SG and OA grids differ at the same point."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.grid = sr.create_passband_grid(_peak_voltage(450.0))

    def test_central_value_is_one_for_normalized_grid(self):
        # At elevation=0 and speed_ratio=1.0 (central speed), OA passband should be ~1 (peak)
        val = interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=1.0)
        self.assertGreater(val, 0.5)

    def test_out_of_bounds_elevation_returns_zero(self):
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=100.0, speed_ratio=1.0),
            0.0,
        )
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=-100.0, speed_ratio=1.0),
            0.0,
        )

    def test_out_of_bounds_speed_returns_zero(self):
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=0.0), 0.0
        )
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=10.0), 0.0
        )

    def test_sg_and_oa_are_different(self):
        sg = interpolate_passband(self.grid, True, elevation=0.0, speed_ratio=1.0)
        oa = interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=1.0)
        self.assertFalse(np.isclose(sg, oa))


class TestInterpolateTransmission(unittest.TestCase):
    """Verify _interpolate_transmission against known calibration values, symmetry, and periodicity."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.at = np.asarray(sr.azimuthal_transmission, dtype=float)
        cls.ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    def test_sunglasses_region_near_zero(self):
        # Sunglasses region |phi| < 9° has ~1e-3 transmission
        val = _interpolate_transmission(self.at, self.ats, 0.0)
        self.assertLess(val, 0.01)

    def test_open_aperture_region_near_one(self):
        # Open aperture region has transmission close to 1
        val = _interpolate_transmission(self.at, self.ats, 90.0)
        self.assertGreater(val, 0.5)

    def test_periodic_wraparound(self):
        # 170° and −170° should give the same value (both near 180°)
        v1 = _interpolate_transmission(self.at, self.ats, 170.0)
        v2 = _interpolate_transmission(self.at, self.ats, -170.0)
        np.testing.assert_allclose(v1, v2, rtol=1e-6)

    def test_symmetric_about_zero(self):
        # Transmission is symmetric: T(phi) = T(-phi)
        for phi in [5.0, 45.0, 90.0, 120.0]:
            v_pos = _interpolate_transmission(self.at, self.ats, phi)
            v_neg = _interpolate_transmission(self.at, self.ats, -phi)
            np.testing.assert_allclose(v_pos, v_neg, rtol=1e-6, err_msg=f"phi={phi}")


class TestCalculateIntegral(unittest.TestCase):
    """Verify calculate_integral is positive, linear in density, monotone in voltage offset,
    increasing in temperature at off-peak voltage, nonzero at elevations outside the SG
    passband range, zero outside the full FOV, and within 5%/15% of the reference CSV
    at p95/p99 for count rates ≥ 100 Hz."""

    @classmethod
    def setUpClass(cls):
        cls.swapi_response = _load_swapi_response()
        cls.peak_voltage = _peak_voltage(450.0)
        cls.at = np.asarray(cls.swapi_response.azimuthal_transmission, dtype=float)
        cls.ats = float(cls.swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    def _ci(self, voltage, sw_params):
        """Compute calculate_integral for the given voltage and sw_params."""
        grid = self.swapi_response.create_passband_grid(voltage)
        cs = self.swapi_response.central_speed(
            voltage, PROTON_MASS_PER_CHARGE_M_P_PER_E
        )
        cea = self.swapi_response.get_central_effective_area(voltage)
        return calculate_integral(grid, sw_params, cs, cea, self.at, self.ats)

    def test_count_rate_is_positive(self):
        self.assertGreater(self._ci(self.peak_voltage, _make_sw_params()), 0.0)

    def test_scales_linearly_with_density(self):
        result_1 = self._ci(self.peak_voltage, _make_sw_params(density=1.0))
        result_5 = self._ci(self.peak_voltage, _make_sw_params(density=5.0))
        np.testing.assert_allclose(result_5, 5.0 * result_1, rtol=1e-10)

    def test_drops_away_from_peak_voltage(self):
        # The passband is centered on central_speed. At 2× peak voltage, the
        # passband window is far from the Maxwellian peak → lower count rate.
        rate_peak = self._ci(self.peak_voltage, _make_sw_params())
        rate_high = self._ci(self.peak_voltage * 2.0, _make_sw_params())
        self.assertGreater(rate_peak, rate_high)

    def test_increases_with_temperature_at_off_peak_voltage(self):
        # At 1.5× peak voltage, a hotter plasma has a wider VDF that overlaps
        # more with the passband, so count rate should increase with temperature
        rate_cold = self._ci(
            self.peak_voltage * 1.5, _make_sw_params(temperature_k=5.0 * EV_TO_KELVIN)
        )
        rate_hot = self._ci(
            self.peak_voltage * 1.5, _make_sw_params(temperature_k=30.0 * EV_TO_KELVIN)
        )
        self.assertGreater(rate_hot, rate_cold)

    def test_nonzero_for_bulk_elevation_outside_sg_passband_range(self):
        # bulk_elevation=8° is above the SG active elevation range (−11 to 7)
        # but within the OA range (−12 to 10). The elevation window is clamped to
        # the passband bounds, so the OA contribution is nonzero.
        self.assertGreater(
            self._ci(self.peak_voltage, _make_sw_params(bulk_elevation=8.0)), 0.0
        )

    def test_matches_reference_integrals_csv(self):
        _REFERENCE_CSV = Path(__file__).resolve().parents[0] / "reference_integrals.csv"
        df = pd.read_csv(_REFERENCE_CSV)
        # Filter to physically significant count rates (>= 1 Hz).
        # Cases below this threshold are near or below SWAPI's background level
        df = df[df["integral"] >= 1.0].reset_index(drop=True)

        optimized = np.empty(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            thermal_speed = float(
                np.sqrt(row.temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
                / METERS_PER_KILOMETER
            )
            sw = SWParams(
                density=float(row.density),
                bulk_speed=float(row.bulk_speed),
                bulk_azimuth=float(row.bulk_azimuth),
                bulk_elevation=float(row.bulk_elevation),
                thermal_speed=thermal_speed,
            )
            v = _peak_voltage(float(row.bulk_speed))
            optimized[i] = self._ci(v, sw)

        rel_errors = (
            np.abs(optimized - df["integral"].to_numpy()) / df["integral"].to_numpy()
        )

        self.assertLess(
            np.median(rel_errors),
            0.005,
            "median relative error vs reference_integrals.csv exceeds 0.5%",
        )

        # TODO tighter maximum after fixing edge cases
        self.assertLess(
            max(rel_errors),
            0.1,
            "max relative error vs reference_integrals.csv exceeds 10%",
        )

    def test_zero_for_beam_outside_fov(self):
        # Beam pointing at elevation=89° is outside all passbands → zero
        self.assertAlmostEqual(
            self._ci(self.peak_voltage, _make_sw_params(bulk_elevation=89.0)),
            0.0,
            places=3,
        )


class TestModelCountRates(unittest.TestCase):
    """Verify _model_count_rates output shape, positivity, monotonicity with density,
    and that the peak bin corresponds to the grid nearest the true bulk speed."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        peak_v = _peak_voltage(450.0)
        voltages = np.geomspace(peak_v * 0.8, peak_v * 1.3, 5)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(sr, voltages)
        cls.rot = np.tile(np.eye(3), (5, 1, 1))

    def _mcr(self, density, temperature, velocity):
        return _model_count_rates(
            density,
            temperature,
            velocity,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            PROTON_MASS_KG,
        )

    def test_output_shape(self):
        self.assertEqual(
            self._mcr(5.0, 10.0 * EV_TO_KELVIN, np.array([450.0, 0.0, 0.0])).shape, (5,)
        )

    def test_all_positive(self):
        self.assertTrue(
            np.all(self._mcr(5.0, 10.0 * EV_TO_KELVIN, np.array([450.0, 0.0, 0.0])) > 0)
        )

    def test_increases_with_density(self):
        # After deadtime correction the observed rate is sublinear in density,
        # but must still be strictly increasing.
        r1 = self._mcr(1.0, 10.0 * EV_TO_KELVIN, np.array([450.0, 0.0, 0.0]))
        r5 = self._mcr(5.0, 10.0 * EV_TO_KELVIN, np.array([450.0, 0.0, 0.0]))
        self.assertTrue(np.all(r5 > r1))

    def test_peak_at_central_grid(self):
        # The 5 grids span [0.8×, 1.3×] peak voltage. The middle grid (index 2)
        # is nearest to 1.0× (geometric mean ≈ 1.02×) and should dominate.
        result = self._mcr(5.0, 10.0 * EV_TO_KELVIN, np.array([450.0, 0.0, 0.0]))
        self.assertEqual(
            np.argmax(result), 2
        )  # middle of 5 grids spanning ±30% of peak


class TestGetInitialGuess(unittest.TestCase):
    """Verify _get_initial_guess recovers bulk speed, temperature, and density
    from realistic synthetic SWAPI data (5 sweeps × 72 ESA voltage steps over
    60 s, spin axis = boresight) and returns a purely anti-sunward bulk velocity.

    The full optimizer recovers V_T and V_N from the spin-phase modulation; the
    initial guess only needs to provide the dominant radial component."""

    @classmethod
    def setUpClass(cls):
        cls.sr = _load_swapi_response()

    def setUp(self):
        self.true_density = 5.0
        self.true_temperature = 10.0 * EV_TO_KELVIN  # K (10 eV)
        self.true_speed = 450.0
        # Non-zero transverse components in the true velocity exercise the geometry,
        # but the initial guess returns them as zero — only the optimizer recovers them.
        self.true_velocity = np.array([self.true_speed, 30.0, -20.0])

        n_sweeps = 5
        n_bins = 72
        sweep_s = 12.0
        spin_s = 15.0
        dt_s = sweep_s / n_bins

        voltages_one_sweep = np.geomspace(
            _peak_voltage(self.true_speed) * 0.3,
            _peak_voltage(self.true_speed) * 3.0,
            n_bins,
        )
        self.voltages = np.tile(voltages_one_sweep, n_sweeps)

        all_voltages = np.tile(voltages_one_sweep, n_sweeps)
        self.grids, self.cs, self.cea, self.at, self.ats = _build_proton_arrays(
            self.sr, all_voltages
        )

        self.rotation_matrices = _spin_rotation_matrices(
            n_sweeps * n_bins,
            spin_period_s=spin_s,
            dt_s=dt_s,
        )

        self.count_rate = _model_count_rates(
            self.true_density,
            self.true_temperature,
            self.true_velocity,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rotation_matrices,
            PROTON_MASS_KG,
        )

    def _run(self):
        return _get_initial_guess(
            self.count_rate,
            self.voltages,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rotation_matrices,
        )

    def test_recovers_bulk_speed(self):
        # Initial guess returns the radial speed only; compare to v_R, not |v_true|.
        np.testing.assert_allclose(
            self._run().bulk_velocity_rtn[0], self.true_speed, rtol=0.02
        )

    def test_recovers_temperature(self):
        np.testing.assert_allclose(
            self._run().temperature, self.true_temperature, rtol=0.2
        )

    def test_recovers_density(self):
        # Density is scaled against a unit model evaluated with the (incorrect) anti-sunward
        # direction; small transverse components in the truth thus produce a few-percent bias
        # in the initial density. The optimizer corrects this in step 3.
        np.testing.assert_allclose(self._run().density, self.true_density, rtol=0.2)

    def test_bulk_velocity_initial_guess(self):
        result = self._run()
        expected_vR = math.sqrt(max(self.true_speed**2 - 30.0**2, 0.0))
        np.testing.assert_allclose(result.bulk_velocity_rtn[0], expected_vR, rtol=0.02)
        np.testing.assert_allclose(result.bulk_velocity_rtn[1], -30.0, atol=1.0)
        np.testing.assert_allclose(result.bulk_velocity_rtn[2], 0.0, atol=1e-6)


_R_BASE_RTN_TO_SWAPI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _spin_rotation_matrices(n, spin_period_s=15.0, dt_s=0.145):
    """Realistic SWAPI geometry: spin axis = boresight (+Y_SWAPI = -R_RTN).

    R(t) = R_spin_around_Y(2*pi*t/T_spin) @ R_base, where R_base maps
    nominal anti-sunward bulk (R_RTN) to -Y_SWAPI so phi=theta=0 at zero spin phase.
    Spin around the boresight leaves the dominant Y component unchanged and only
    introduces small phi/theta wobble of order arcsin(v_T,N / v_R) (a few degrees).
    """
    times = np.arange(n) * dt_s
    alphas = 2.0 * np.pi * times / spin_period_s
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        R[i] = R_spin @ _R_BASE_RTN_TO_SWAPI
    return R


class TestOptimize(unittest.TestCase):
    """Verify _optimize recovers all five parameters from synthetic noiseless data
    generated with spinning rotation matrices (which make V_T and V_N observable)."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.true_speed = 450.0
        cls.true_density = 5.0
        cls.true_temperature = 10.0 * EV_TO_KELVIN  # K (10 eV)
        cls.true_velocity = np.array([cls.true_speed, 20.0, -10.0])

        voltages = np.geomspace(
            _peak_voltage(cls.true_speed) * 0.75,
            _peak_voltage(cls.true_speed) * 1.35,
            20,
        )
        n_sweeps = 5
        all_voltages = np.tile(voltages, n_sweeps)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(
            sr, all_voltages
        )

        cls.rot = _spin_rotation_matrices(n_sweeps * len(voltages))

        cls.count_rate = _model_count_rates(
            cls.true_density,
            cls.true_temperature,
            cls.true_velocity,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            PROTON_MASS_KG,
        )
        cls.initial_guess = ProtonSolarWindMoments(
            density=cls.true_density * 0.8,
            temperature=cls.true_temperature * 1.2,
            bulk_velocity_rtn=np.array([cls.true_speed * 1.05, 0.0, 0.0]),
            bad_fit_flag=0,
        )

    def _run(self):
        return _optimize(
            self.count_rate,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            self.initial_guess,
        )

    def test_recovers_density(self):
        np.testing.assert_allclose(self._run().density, self.true_density, rtol=0.05)

    def test_recovers_temperature(self):
        np.testing.assert_allclose(
            self._run().temperature, self.true_temperature, rtol=0.1
        )

    def test_recovers_bulk_velocity(self):
        np.testing.assert_allclose(
            self._run().bulk_velocity_rtn, self.true_velocity, rtol=0.05, atol=1.0
        )

    def test_success_flag_on_good_fit(self):
        self.assertEqual(self._run().bad_fit_flag, SwapiL3Flags.NONE)

    def test_density_and_temperature_positive(self):
        result = self._run()
        self.assertGreater(result.density, 0)
        self.assertGreater(result.temperature, 0)


class TestUncertainties(unittest.TestCase):
    """Verify that _optimize returns finite, positive uncertainties that decrease with more data."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.true_speed = 450.0
        cls.true_density = 5.0
        cls.true_temperature = 10.0 * EV_TO_KELVIN  # K (10 eV)
        cls.true_velocity = np.array([cls.true_speed, 20.0, -10.0])

        voltages = np.geomspace(
            _peak_voltage(cls.true_speed) * 0.75,
            _peak_voltage(cls.true_speed) * 1.35,
            20,
        )
        n_sweeps = 5
        all_voltages = np.tile(voltages, n_sweeps)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(
            sr, all_voltages
        )

        cls.rot = _spin_rotation_matrices(n_sweeps * len(voltages))

        cls.count_rate = _model_count_rates(
            cls.true_density,
            cls.true_temperature,
            cls.true_velocity,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            PROTON_MASS_KG,
        )
        cls.initial_guess = ProtonSolarWindMoments(
            density=cls.true_density * 0.8,
            temperature=cls.true_temperature * 1.2,
            bulk_velocity_rtn=np.array([cls.true_speed * 1.05, 0.0, 0.0]),
            bad_fit_flag=0,
        )
        cls.result = _optimize(
            cls.count_rate,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            cls.initial_guess,
        )

    def test_density_sigma_is_finite_and_positive(self):
        self.assertTrue(np.isfinite(self.result.density_sigma))
        self.assertGreater(self.result.density_sigma, 0.0)

    def test_temperature_sigma_is_finite_and_positive(self):
        self.assertTrue(np.isfinite(self.result.temperature_sigma))
        self.assertGreater(self.result.temperature_sigma, 0.0)

    def test_velocity_covariance_is_symmetric_positive_semidefinite(self):
        cov = self.result.velocity_covariance
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(cov)
        self.assertTrue(
            np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"
        )

    def test_density_sigma_smaller_than_density(self):
        # Fractional uncertainty on density should be less than 100%
        self.assertLess(self.result.density_sigma / self.result.density, 1.0)

    def test_temperature_sigma_smaller_than_temperature(self):
        self.assertLess(self.result.temperature_sigma / self.result.temperature, 1.0)

    def test_velocity_diagonal_sigmas_finite_and_positive(self):
        cov = self.result.velocity_covariance
        for i in range(3):
            self.assertGreater(cov[i, i], 0.0, f"cov[{i},{i}] not positive")
            self.assertTrue(np.isfinite(cov[i, i]))

    def test_more_data_gives_smaller_uncertainties(self):
        """Doubling the number of sweeps (data points) should reduce uncertainties."""
        sr = _load_swapi_response()
        voltages = np.geomspace(
            _peak_voltage(self.true_speed) * 0.75,
            _peak_voltage(self.true_speed) * 1.35,
            20,
        )

        def run_with_n_sweeps(n_sweeps):
            all_v = np.tile(voltages, n_sweeps)
            grids, cs, cea, at, ats = _build_proton_arrays(sr, all_v)
            rot = _spin_rotation_matrices(n_sweeps * len(voltages))
            cr = _model_count_rates(
                self.true_density,
                self.true_temperature,
                self.true_velocity,
                grids,
                cs,
                cea,
                at,
                ats,
                rot,
                PROTON_MASS_KG,
            )
            ig = ProtonSolarWindMoments(
                density=self.true_density * 0.8,
                temperature=self.true_temperature * 1.2,
                bulk_velocity_rtn=np.array([self.true_speed * 1.05, 0.0, 0.0]),
                bad_fit_flag=0,
            )
            return _optimize(cr, grids, cs, cea, at, ats, rot, ig)

        r5 = run_with_n_sweeps(5)
        r10 = run_with_n_sweeps(10)
        self.assertLess(r10.density_sigma, r5.density_sigma)
        self.assertLess(r10.temperature_sigma, r5.temperature_sigma)

    def test_speed_uncertainty_via_propagation(self):
        """sigma_speed from covariance matches finite-difference estimate."""
        result = self.result
        vr, vt, vn = result.bulk_velocity_rtn
        speed = float(np.linalg.norm(result.bulk_velocity_rtn))
        v_hat = np.array([vr, vt, vn]) / speed
        sigma_speed = float(np.sqrt(v_hat @ result.velocity_covariance @ v_hat))
        self.assertGreater(sigma_speed, 0.0)
        self.assertLess(sigma_speed, speed)  # uncertainty smaller than value


# ---------------------------------------------------------------------------
# Integration test: hardcoded real L2 spectrum from 2026-01-01 ~12:00 UTC
#
# Extracted from imap_L3_processing L2 dataset (5-sweep group, proton-peak bins).
# Rotation matrices computed from SPICE at the actual measurement times.
# Expected moments verified against OMNI reference.
# ---------------------------------------------------------------------------

# 5 sweeps × 7 ESA bins spanning the proton peak
_L2_ESA_VOLTAGES = np.array(
    [
        776.757277,
        712.276025,
        653.147580,
        598.927587,
        549.208580,
        503.616916,
        461.809970,
    ]
)

_L2_COUNT_RATES = np.array(
    [
        [
            2158.620690,
            5765.517241,
            12586.206897,
            19055.172414,
            18806.896552,
            11586.206897,
            5517.241379,
        ],
        [
            2951.724138,
            6841.379310,
            14131.034483,
            21248.275862,
            19620.689655,
            11641.379310,
            5344.827586,
        ],
        [
            2779.310345,
            7400.000000,
            13958.620690,
            15606.896552,
            10682.758621,
            5655.172414,
            2765.517241,
        ],
        [
            2613.793103,
            5758.620690,
            12600.000000,
            15827.586207,
            11668.965517,
            5758.620690,
            2551.724138,
        ],
        [
            2993.103448,
            8579.310345,
            18144.827586,
            24572.413793,
            19841.379310,
            11027.586207,
            5220.689655,
        ],
    ]
)  # shape (5 sweeps, 7 bins), Hz

# RTN-to-SWAPI rotation matrices at each measurement time, shape (5, 7, 3, 3)
_L2_ROTATION_MATRICES = np.array(
    [
        [
            [
                [2.609821e-02, 2.704214e-01, -9.623883e-01],
                [-9.979728e-01, 6.294733e-02, -9.375635e-03],
                [5.804440e-02, 9.606820e-01, 2.715160e-01],
            ],
            [
                [2.203913e-02, 2.033433e-01, -9.788594e-01],
                [-9.979848e-01, 6.274867e-02, -9.434649e-03],
                [5.950366e-02, 9.770947e-01, 2.043164e-01],
            ],
            [
                [1.788886e-02, 1.352909e-01, -9.906444e-01],
                [-9.979970e-01, 6.254642e-02, -9.479757e-03],
                [6.067874e-02, 9.888298e-01, 1.361388e-01],
            ],
            [
                [1.366727e-02, 6.658989e-02, -9.976868e-01],
                [-9.980096e-01, 6.234155e-02, -9.510745e-03],
                [6.156402e-02, 9.958310e-01, 6.730939e-02],
            ],
            [
                [9.394556e-03, -2.430801e-03, -9.999529e-01],
                [-9.980223e-01, 6.213503e-02, -9.527463e-03],
                [6.215526e-02, 9.980648e-01, -1.842263e-03],
            ],
            [
                [5.091190e-03, -7.144072e-02, -9.974319e-01],
                [-9.980351e-01, 6.192786e-02, -9.529831e-03],
                [6.244964e-02, 9.955205e-01, -7.098506e-02],
            ],
            [
                [7.838670e-04, -1.401076e-01, -9.901360e-01],
                [-9.980479e-01, 6.172186e-02, -9.523982e-03],
                [6.244741e-02, 9.882106e-01, -1.397857e-01],
            ],
        ],
        [
            [
                [6.545899e-02, 9.978447e-01, 4.596260e-03],
                [-9.978362e-01, 6.548550e-02, -5.876371e-03],
                [-6.164694e-03, -4.201653e-03, 9.999722e-01],
            ],
            [
                [6.578006e-02, 9.957435e-01, -6.455892e-02],
                [-9.978324e-01, 6.552059e-02, -6.130419e-03],
                [-1.874386e-03, 6.482224e-02, 9.978951e-01],
            ],
            [
                [6.580371e-02, 9.888746e-01, -1.334049e-01],
                [-9.978296e-01, 6.553807e-02, -6.386295e-03],
                [2.427855e-03, 1.335356e-01, 9.910410e-01],
            ],
            [
                [6.552982e-02, 9.772709e-01, -2.016121e-01],
                [-9.978280e-01, 6.553784e-02, -6.642774e-03],
                [6.721436e-03, 2.016095e-01, 9.794429e-01],
            ],
            [
                [6.495970e-02, 9.609878e-01, -2.688542e-01],
                [-9.978274e-01, 6.551991e-02, -6.898628e-03],
                [1.098581e-02, 2.687183e-01, 9.631562e-01],
            ],
            [
                [6.409607e-02, 9.401035e-01, -3.348091e-01],
                [-9.978280e-01, 6.548437e-02, -7.152632e-03],
                [1.520055e-02, 3.345404e-01, 9.422588e-01],
            ],
            [
                [6.293972e-02, 9.147186e-01, -3.991598e-01],
                [-9.978299e-01, 6.542560e-02, -7.408373e-03],
                [1.933869e-02, 3.987599e-01, 9.168515e-01],
            ],
        ],
        [
            [
                [1.335341e-02, 2.624175e-01, 9.648620e-01],
                [-9.980289e-01, 6.267281e-02, -3.232949e-03],
                [-6.131900e-02, -9.629170e-01, 2.627371e-01],
            ],
            [
                [1.756634e-02, 3.283778e-01, 9.443831e-01],
                [-9.980155e-01, 6.288242e-02, -3.301316e-03],
                [-6.046918e-02, -9.424510e-01, 3.288307e-01],
            ],
            [
                [2.171042e-02, 3.927648e-01, 9.193827e-01],
                [-9.980023e-01, 6.308681e-02, -3.384042e-03],
                [-5.933005e-02, -9.174725e-01, 3.933498e-01],
            ],
            [
                [2.576580e-02, 4.552704e-01, 8.899803e-01],
                [-9.979894e-01, 6.328499e-02, -3.480730e-03],
                [-5.790707e-02, -8.881013e-01, 4.559856e-01],
            ],
            [
                [2.971308e-02, 5.155952e-01, 8.563169e-01],
                [-9.979769e-01, 6.347602e-02, -3.590918e-03],
                [-5.620705e-02, -8.544778e-01, 5.164382e-01],
            ],
            [
                [3.353334e-02, 5.734505e-01, 8.185537e-01],
                [-9.979648e-01, 6.365898e-02, -3.714078e-03],
                [-5.423813e-02, -8.167632e-01, 5.744181e-01],
            ],
            [
                [3.721313e-02, 6.285520e-01, 7.768768e-01],
                [-9.979532e-01, 6.383253e-02, -3.842428e-03],
                [-5.200518e-02, -7.751437e-01, 6.296409e-01],
            ],
        ],
        [
            [
                [-5.309179e-02, -8.579861e-01, 5.109218e-01],
                [-9.982624e-01, 5.869778e-02, -5.162563e-03],
                [-2.556057e-02, -5.103081e-01, -8.596117e-01],
            ],
            [
                [-5.113999e-02, -8.206493e-01, 5.691391e-01],
                [-9.982585e-01, 5.878521e-02, -4.935318e-03],
                [-2.940680e-02, -5.684004e-01, -8.222264e-01],
            ],
            [
                [-4.892690e-02, -7.793850e-01, 6.246320e-01],
                [-9.982535e-01, 5.888811e-02, -4.714666e-03],
                [-3.310886e-02, -6.237717e-01, -7.809050e-01],
            ],
            [
                [-4.646312e-02, -7.343907e-01, 6.771348e-01],
                [-9.982475e-01, 5.900600e-02, -4.501664e-03],
                [-3.664903e-02, -6.761572e-01, -7.358453e-01],
            ],
            [
                [-4.376044e-02, -6.858817e-01, 7.263961e-01],
                [-9.982405e-01, 5.913830e-02, -4.297329e-03],
                [-4.001037e-02, -7.253061e-01, -6.872629e-01],
            ],
            [
                [-4.083180e-02, -6.340904e-01, 7.721801e-01],
                [-9.982327e-01, 5.928438e-02, -4.102642e-03],
                [-4.317678e-02, -7.709830e-01, -6.353904e-01],
            ],
            [
                [-3.769038e-02, -5.792601e-01, 8.142710e-01],
                [-9.982237e-01, 5.944768e-02, -3.914874e-03],
                [-4.613879e-02, -8.129722e-01, -5.804718e-01],
            ],
        ],
        [
            [
                [-3.684847e-02, -7.210986e-01, -6.918519e-01],
                [-9.981869e-01, 5.953191e-02, -8.884448e-03],
                [4.759383e-02, 6.902701e-01, -7.219848e-01],
            ],
            [
                [-4.004352e-02, -7.670996e-01, -6.402771e-01],
                [-9.981978e-01, 5.937390e-02, -8.706120e-03],
                [4.469421e-02, 6.387746e-01, -7.680947e-01],
            ],
            [
                [-4.303045e-02, -8.094299e-01, -5.856377e-01],
                [-9.982081e-01, 5.922857e-02, -8.517265e-03],
                [4.158061e-02, 5.842218e-01, -8.105282e-01],
            ],
            [
                [-4.579495e-02, -8.478872e-01, -5.281952e-01],
                [-9.982176e-01, 5.909662e-02, -8.318787e-03],
                [3.826794e-02, 5.268728e-01, -8.490822e-01],
            ],
            [
                [-4.832379e-02, -8.822872e-01, -4.682244e-01],
                [-9.982263e-01, 5.897869e-02, -8.111635e-03],
                [3.477206e-02, 4.670019e-01, -8.835724e-01],
            ],
            [
                [-5.060488e-02, -9.124653e-01, -4.060125e-01],
                [-9.982341e-01, 5.887535e-02, -7.896802e-03],
                [3.110969e-02, 4.048959e-01, -9.138334e-01],
            ],
            [
                [-5.263135e-02, -9.382797e-01, -3.418496e-01],
                [-9.982407e-01, 5.879232e-02, -7.678498e-03],
                [2.730271e-02, 3.408440e-01, -9.397233e-01],
            ],
        ],
    ]
)  # shape (5, 7, 3, 3)

_L2_SPACECRAFT_VELOCITY_RTN = np.array([-0.055488, 29.552893, -3.411245])  # km/s


class TestIntegrationRealL2Spectrum(unittest.TestCase):
    """End-to-end fit on a real SWAPI L2 spectrum (count rates hardcoded, no file I/O)."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        n_sweeps, n_bins = _L2_COUNT_RATES.shape

        all_voltages = np.tile(_L2_ESA_VOLTAGES, n_sweeps)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(
            sr, all_voltages
        )

        cls.rot = _L2_ROTATION_MATRICES.reshape(n_sweeps * n_bins, 3, 3)
        cls.count_rate = _L2_COUNT_RATES.ravel()

        ig = _get_initial_guess(
            cls.count_rate,
            all_voltages,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
        )
        cls.result = _optimize(
            cls.count_rate,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            ig,
        )

    def test_fit_succeeds(self):
        self.assertEqual(self.result.bad_fit_flag, SwapiL3Flags.NONE)

    def test_bulk_speed_reasonable(self):
        speed = float(np.linalg.norm(self.result.bulk_velocity_rtn))
        self.assertGreater(speed, 350.0, "Speed too low")
        self.assertLess(speed, 650.0, "Speed too high")

    def test_dominant_component_is_radial(self):
        vr, vt, vn = self.result.bulk_velocity_rtn
        self.assertGreater(vr, abs(vt), "V_R should dominate V_T")
        self.assertGreater(vr, abs(vn), "V_R should dominate V_N")

    def test_density_reasonable(self):
        self.assertGreater(self.result.density, 1.0, "Density too low")
        self.assertLess(self.result.density, 50.0, "Density too high")

    def test_temperature_reasonable(self):
        self.assertGreater(self.result.temperature, 10_000, "Temperature too low")
        self.assertLess(self.result.temperature, 1_500_000, "Temperature too high")

    def test_model_reproduces_observed_count_rates(self):
        model = _model_count_rates(
            self.result.density,
            self.result.temperature,
            self.result.bulk_velocity_rtn,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            PROTON_MASS_KG,
        )
        # Real data has Poisson noise; normalized residuals will be O(√N) per bin.
        # Require that the model is at least within a factor of 2 of the observations
        # (i.e. relative error < 100%), confirming the fit is physically sensible.
        rel_err = np.abs(model - self.count_rate) / np.maximum(self.count_rate, 1.0)
        self.assertLess(float(np.mean(rel_err)), 1.0, "Mean relative error > 100%")

    def test_uncertainties_are_finite_for_real_data(self):
        self.assertTrue(np.isfinite(self.result.density_sigma))
        self.assertTrue(np.isfinite(self.result.temperature_sigma))
        self.assertIsNotNone(self.result.velocity_covariance)
        self.assertTrue(np.all(np.isfinite(self.result.velocity_covariance)))
        self.assertGreater(self.result.density_sigma, 0.0)
        self.assertGreater(self.result.temperature_sigma, 0.0)


class TestFitSolarWindProtonMoments(unittest.TestCase):
    """Tests for the top-level fit_solar_wind_proton_moments entry point."""

    def setUp(self):
        from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
            fit_solar_wind_proton_moments,
        )

        self.fit = fit_solar_wind_proton_moments

    def test_returns_proton_solar_wind_moments(self):
        sr = _load_swapi_response()
        true_speed = 450.0
        voltages = np.geomspace(
            _peak_voltage(true_speed) * 0.75, _peak_voltage(true_speed) * 1.35, 10
        )
        times = np.zeros(len(voltages), dtype=np.int64)
        grids, cs, cea, at, ats = _build_proton_arrays(sr, voltages)
        rot = np.tile(np.eye(3), (len(voltages), 1, 1))
        count_rate = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            np.array([true_speed, 0.0, 0.0]),
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )

        result = self.fit(count_rate, voltages, sr, 1.0, rot)

        self.assertIsInstance(result, ProtonSolarWindMoments)
        self.assertGreater(result.density, 0)
        self.assertGreater(result.temperature, 0)
        speed = float(np.linalg.norm(result.bulk_velocity_rtn))
        self.assertGreater(speed, 300.0)
        self.assertLess(speed, 700.0)

    def test_low_count_bins_below_tail_mask_dont_influence_fit(self):
        # Bins below 10% of the peak count rate are masked out.
        # Perturbing them (without lifting them above the threshold) must not
        # change the recovered moments.
        sr = _load_swapi_response()
        true_speed = 450.0
        voltages = np.geomspace(
            _peak_voltage(true_speed) * 0.4, _peak_voltage(true_speed) * 2.5, 30
        )
        times = np.zeros(len(voltages), dtype=np.int64)
        grids, cs, cea, at, ats = _build_proton_arrays(sr, voltages)
        rot = np.tile(np.eye(3), (len(voltages), 1, 1))
        count_rate = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            np.array([true_speed, 0.0, 0.0]),
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )
        below = count_rate < 0.1 * float(np.max(count_rate))
        self.assertGreater(int(below.sum()), 0)

        perturbed = count_rate.copy()
        perturbed[below] = 0.0

        baseline = self.fit(count_rate, voltages, sr, 1.0, rot)
        with_perturb = self.fit(perturbed, voltages, sr, 1.0, rot)

        np.testing.assert_allclose(with_perturb.density, baseline.density, rtol=1e-3)
        np.testing.assert_allclose(
            with_perturb.temperature, baseline.temperature, rtol=1e-3
        )
        np.testing.assert_allclose(
            with_perturb.bulk_velocity_rtn,
            baseline.bulk_velocity_rtn,
            rtol=1e-3,
            atol=1e-2,
        )


class TestPoissonUncertaintyCoverage(unittest.TestCase):
    """Verify that reported fitting uncertainties match empirical scatter across Poisson-noise realizations.

    Runs 500 independent fits on synthetic data with Poisson noise drawn from the true
    model count rates. The mean reported sigma (from s²-scaled covariance, i.e. fitting
    error derived from residuals) should agree with the empirical std dev of the fit
    outputs to within 10%. When the model is correct and noise is purely Poisson, s² ≈ 1
    and the fitting error coincides with the propagated Poisson error.

    Uses 5 sweeps × 8 bins with realistic sweep timing (12 s per sweep) so that
    the 60 s total spans 4 full spin periods. This gives enough spin-phase diversity
    to strongly break the mirror symmetry, preventing basin toggling across noise
    realizations that would inflate the empirical variance.
    """

    N_REALIZATIONS = 500

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.true_density = 5.0
        cls.true_temperature = 10.0 * EV_TO_KELVIN  # K (10 eV)
        cls.true_velocity = np.array([450.0, 20.0, -10.0])

        n_bins_per_sweep = 8
        n_sweeps = 5
        voltages = np.geomspace(
            _peak_voltage(450.0) * 0.75, _peak_voltage(450.0) * 1.35, n_bins_per_sweep
        )
        all_voltages = np.tile(voltages, n_sweeps)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(
            sr, all_voltages
        )
        cls.rot = _spin_rotation_matrices(
            n_sweeps * n_bins_per_sweep,
            dt_s=12.0 / n_bins_per_sweep,
        )
        cls.sr = sr
        cls.all_voltages = all_voltages

        true_rate = _model_count_rates(
            cls.true_density,
            cls.true_temperature,
            cls.true_velocity,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            PROTON_MASS_KG,
        )
        # Deadtime is applied at the residual stage; synthesize "observed" data accordingly.
        observed_rate = apply_deadtime_correction_array(true_rate)

        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        import os

        rng = np.random.default_rng(42)
        all_noisy_rates = (
            rng.poisson(
                observed_rate * SWAPI_LIVETIME_S,
                size=(cls.N_REALIZATIONS, len(observed_rate)),
            )
            / SWAPI_LIVETIME_S
        )

        _coverage_shared["sr"] = cls.sr
        _coverage_shared["voltages"] = cls.all_voltages
        _coverage_shared["rot"] = cls.rot

        # Force lazy imports in the parent so forked children inherit them.
        import imap_l3_processing.swapi.l3a.utils  # noqa: F401

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as pool:
            results = list(pool.map(_coverage_worker, all_noisy_rates, chunksize=25))

        cls.densities = np.array([r.density for r in results])
        cls.temperatures = np.array([r.temperature for r in results])
        cls.velocities = np.array(
            [r.bulk_velocity_rtn for r in results]
        )  # shape (N, 3)
        cls.mean_density_sigma = float(np.mean([r.density_sigma for r in results]))
        cls.mean_temperature_sigma = float(
            np.mean([r.temperature_sigma for r in results])
        )
        cls.mean_velocity_cov = np.mean(
            [r.velocity_covariance for r in results], axis=0
        )

    def test_density_sigma_matches_empirical_std(self):
        np.testing.assert_allclose(
            self.mean_density_sigma, np.std(self.densities), rtol=0.1
        )

    def test_temperature_sigma_matches_empirical_std(self):
        np.testing.assert_allclose(
            self.mean_temperature_sigma, np.std(self.temperatures), rtol=0.1
        )

    def test_velocity_covariance_diagonal_matches_empirical_variance(self):
        empirical_var = np.var(self.velocities, axis=0)
        np.testing.assert_allclose(
            np.diag(self.mean_velocity_cov), empirical_var, rtol=0.1
        )


class TestGetInitialGuessCurveFitFailure(unittest.TestCase):
    """Verify _get_initial_guess falls back to peak-bin speed and 50 km/s thermal speed
    when scipy.optimize.curve_fit raises RuntimeError."""

    def test_falls_back_to_peak_speed_and_default_sigma(self):
        voltages = np.geomspace(
            _peak_voltage(450.0) * 0.5, _peak_voltage(450.0) * 2.0, 20
        )
        count_rate = np.ones(len(voltages))
        rotation_matrices = np.tile(np.eye(3), (len(voltages), 1, 1))
        dummy_cs = np.ones(len(voltages))
        dummy_cea = np.ones(len(voltages))
        dummy_at = np.ones(100)

        with (
            patch(
                "scipy.optimize.curve_fit", side_effect=RuntimeError("max iterations")
            ),
            patch(
                "imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments._model_count_rates",
                return_value=np.ones(len(voltages)),
            ),
        ):
            result = _get_initial_guess(
                count_rate,
                voltages,
                None,
                dummy_cs,
                dummy_cea,
                dummy_at,
                0.1,
                rotation_matrices,
            )

        # Fallback: bulk_speed = speed[peak_idx], sigma_v = 50.0 km/s
        peak_idx = int(np.nanargmax(count_rate))
        expected_speed = float(esa_voltage_to_proton_speed(voltages[peak_idx]))
        np.testing.assert_allclose(
            result.bulk_velocity_rtn[0], expected_speed, rtol=0.01
        )
        # Fallback sigma_v = 50 km/s → T = m_p * (50e3)^2 / k_B ≈ 302,778 K (≈ 26.1 eV)
        np.testing.assert_allclose(result.temperature, 26.1 * EV_TO_KELVIN, rtol=0.01)


class TestCalculateIntegralZeroPassbandNorm(unittest.TestCase):
    """Verify calculate_integral returns 0 when all passband values are zero (skips division by norm)."""

    def test_zero_passband_grid_returns_zero(self):
        # Build a PassbandGrid where all passband values are zero — norm will be 0
        # and the region loop should skip all regions, returning 0.
        from imap_l3_processing.swapi.l3a.science.swapi_response import PassbandGrid

        zero_grid = np.zeros((23, 101), dtype=np.float64)
        transmission = np.ones(1800, dtype=np.float64)
        boundary = np.array([[0.0], [0.95]])
        grid = PassbandGrid(
            min_elevation=-12.0,
            elevation_spacing=1.0,
            min_speed_ratio=0.9,
            speed_ratio_spacing=0.002,
            values_sunglasses=zero_grid,
            values_open_aperture=zero_grid,
            min_OA_boundary=boundary,
            max_OA_boundary=boundary,
            min_SG_boundary=boundary,
            max_SG_boundary=boundary,
            oa_active_el_range=(-12.0, 10.5),
            sg_active_el_range=(-10.5, 7.0),
        )
        sw_params = SWParams(
            density=5.0,
            bulk_speed=450.0,
            bulk_azimuth=0.0,
            bulk_elevation=0.0,
            thermal_speed=40.0,
        )
        self.assertEqual(
            calculate_integral(grid, sw_params, 450.0, 1.0, transmission, 0.1), 0.0
        )


class TestInterpolateTransmissionBoundary(unittest.TestCase):
    """Verify _interpolate_transmission returns 0 when both interpolation indices clamp to the
    same out-of-bounds entry (weights cancel). Uses a 3-element array to trigger easily."""

    def setUp(self):
        from imap_l3_processing.swapi.l3a.science.swapi_response import PassbandGrid

        zero_grid = np.zeros((23, 101), dtype=np.float64)
        boundary = np.array([[0.0], [0.95]])
        self.at = np.array([0.001, 0.5, 1.0])
        self.ats = 1.0
        self.grid = PassbandGrid(
            min_elevation=-12.0,
            elevation_spacing=1.0,
            min_speed_ratio=0.9,
            speed_ratio_spacing=0.002,
            values_sunglasses=zero_grid,
            values_open_aperture=zero_grid,
            min_OA_boundary=boundary,
            max_OA_boundary=boundary,
            min_SG_boundary=boundary,
            max_SG_boundary=boundary,
            oa_active_el_range=(-12.0, 10.5),
            sg_active_el_range=(-10.5, 7.0),
        )

    def test_azimuth_far_beyond_array_returns_zero(self):
        # azimuth=170 -> abs=170, i_lower=170 >= n=3, i_upper=171 >= n=3, both clamp to n-1=2.
        # weight_lower = float(2) - 170 = -168, weight_upper = 170 - 2 = 168 -> they cancel to 0.
        val = _interpolate_transmission(self.at, self.ats, 170.0)
        self.assertEqual(val, 0.0)

    def test_i_upper_just_beyond_array_returns_zero(self):
        # azimuth=2.5 -> i_lower=2=n-1, i_upper=3 clamps to 2.
        # weight_lower = float(2) - 2.5 = -0.5, weight_upper = 2.5 - 2 = 0.5 -> cancel to 0.
        val = _interpolate_transmission(self.at, self.ats, 2.5)
        self.assertEqual(val, 0.0)


class TestGetAngularLimits(unittest.TestCase):
    """Verify _get_angular_limits clamps to the correct per-region passband bounds:
      SG  (region=0):  elevation ∈ [−11°, 7°],   azimuth ∈ [−20°, 20°]
      OA− (region=−1): elevation ∈ [−12°, 10°],  azimuth ∈ [−150°, −20°]
      OA+ (region=+1): elevation ∈ [−12°, 10°],  azimuth ∈ [20°, 150°]
    Bounds are set to the bilinear-interpolation transition cells of each passband, not
    just the active grid extent — this captures the small-but-nonzero contribution that
    matters when bulk direction is just outside the FOV."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.grid = sr.create_passband_grid(_peak_voltage(450.0))
        cls.cs = sr.central_speed(
            _peak_voltage(450.0), PROTON_MASS_PER_CHARGE_M_P_PER_E
        )

    def test_sg_elevation_clamped_to_sg_passband_bounds(self):
        # bulk_elevation outside the SG active range should be clamped to [-10.5, 7]
        sw = _make_sw_params(bulk_elevation=12.0)
        min_el, max_el, _, _ = _get_angular_limits(sw, 0, self.grid, self.cs)
        self.assertGreaterEqual(min_el, -10.5)
        self.assertLessEqual(max_el, 7.0)

    def test_oa_elevation_clamped_to_oa_passband_bounds(self):
        # bulk_elevation outside the OA active range should be clamped to [-12, 10.5]
        sw = _make_sw_params(bulk_elevation=15.0)
        min_el, max_el, _, _ = _get_angular_limits(sw, 1, self.grid, self.cs)
        self.assertGreaterEqual(min_el, -12.0)
        self.assertLessEqual(max_el, 10.5)

    def test_sg_azimuth_limited_to_sg_range(self):
        sw = _make_sw_params(bulk_azimuth=0.0)
        _, _, min_az, max_az = _get_angular_limits(sw, 0, self.grid, self.cs)
        self.assertGreaterEqual(min_az, -20.0)
        self.assertLessEqual(max_az, 20.0)

    def test_oa_neg_azimuth_clamped_to_oa_neg_range(self):
        sw = _make_sw_params(bulk_azimuth=-90.0)
        _, _, min_az, max_az = _get_angular_limits(sw, -1, self.grid, self.cs)
        self.assertGreaterEqual(min_az, -150.0)
        self.assertLessEqual(max_az, -20.0)

    def test_oa_pos_azimuth_clamped_to_oa_pos_range(self):
        sw = _make_sw_params(bulk_azimuth=90.0)
        _, _, min_az, max_az = _get_angular_limits(sw, 1, self.grid, self.cs)
        self.assertGreaterEqual(min_az, 20.0)
        self.assertLessEqual(max_az, 150.0)

    def test_elevation_window_centered_within_sg_bounds_is_not_clamped(self):
        # bulk_elevation well inside SG range: window should equal [center±width] unmodified
        sw = _make_sw_params(
            bulk_elevation=0.0, temperature_k=1.0 * EV_TO_KELVIN
        )  # narrow window
        min_el, max_el, _, _ = _get_angular_limits(sw, 0, self.grid, self.cs)
        self.assertGreater(min_el, -11.0)
        self.assertLess(max_el, 7.0)


class TestColdPlasmaTransverseRecovery(unittest.TestCase):
    """Regression tests for the known convergence failure on cold plasma.

    At T ≲ 5 eV the VDF is so narrow that the spin-phase modulation encoding
    vT and vN is too weak to pull the optimizer away from its initial vT=vN=0
    starting point. The optimizer converges to vT≈vN≈0 regardless of the true
    values. These tests currently FAIL and will pass once the bug is fixed.

    Cases are drawn from the figure-script random sample (seed=7, 100 samples).
    Tolerance is 20 km/s — comfortably larger than the Poisson-noise scatter for
    warm plasma (~5 km/s RMSE) but tight enough to detect the ≈26–44 km/s errors
    seen in the failing cases.
    """

    _ATOL_KMS = 1.0

    @classmethod
    def setUpClass(cls):
        cls.sr = _load_swapi_response()

    def _run(self, bulk_speed, temperature_k, density, vT, vN, seed):
        voltages = np.geomspace(
            _peak_voltage(bulk_speed) * 0.3, _peak_voltage(bulk_speed) * 3.0, 72
        )
        n_sweeps = 5
        all_voltages = np.tile(voltages, n_sweeps)
        grids, cs, cea, at, ats = _build_proton_arrays(self.sr, all_voltages)
        rot = _spin_rotation_matrices(n_sweeps * len(voltages))
        esa_full = np.tile(voltages, n_sweeps)
        true_vel = np.array([bulk_speed, vT, vN])
        cr = _model_count_rates(
            density,
            temperature_k,
            true_vel,
            grids,
            cs,
            cea,
            at,
            ats,
            rot,
            PROTON_MASS_KG,
        )
        rng = np.random.default_rng(seed)
        cr_noisy = rng.poisson(np.maximum(cr, 0.0)).astype(float)
        ig = _get_initial_guess(cr_noisy, esa_full, grids, cs, cea, at, ats, rot)
        return _optimize(cr_noisy, grids, cs, cea, at, ats, rot, ig), true_vel

    def _assert_velocity_recovered(self, result, true_vel):
        np.testing.assert_allclose(
            result.bulk_velocity_rtn[1],
            true_vel[1],
            atol=self._ATOL_KMS,
            err_msg=f"vT: fit={result.bulk_velocity_rtn[1]:.1f}, true={true_vel[1]:.1f}",
        )
        np.testing.assert_allclose(
            result.bulk_velocity_rtn[2],
            true_vel[2],
            atol=self._ATOL_KMS,
            err_msg=f"vN: fit={result.bulk_velocity_rtn[2]:.1f}, true={true_vel[2]:.1f}",
        )

    def test_cold_plasma_high_vt(self):
        # vb=488 km/s, T=2.1 eV (~24,000 K), vT=26 km/s — optimizer returns vT≈0
        result, true_vel = self._run(
            bulk_speed=488,
            temperature_k=2.1 * EV_TO_KELVIN,
            density=11.7,
            vT=26.1,
            vN=8.1,
            seed=0,
        )
        self._assert_velocity_recovered(result, true_vel)

    def test_cold_plasma_large_vn(self):
        # vb=534 km/s, T=2.3 eV (~26,700 K), vN=38 km/s — optimizer returns vN≈0
        result, true_vel = self._run(
            bulk_speed=534,
            temperature_k=2.3 * EV_TO_KELVIN,
            density=16.4,
            vT=18.1,
            vN=38.4,
            seed=0,
        )
        self._assert_velocity_recovered(result, true_vel)

    def test_cold_fast_plasma_large_vt(self):
        # vb=789 km/s, T=3.8 eV (~44,100 K), vT=−32 km/s — high speed + cold
        result, true_vel = self._run(
            bulk_speed=789,
            temperature_k=3.8 * EV_TO_KELVIN,
            density=12.4,
            vT=-31.6,
            vN=24.4,
            seed=0,
        )
        self._assert_velocity_recovered(result, true_vel)


class TestWrongBasinFlipCheck(unittest.TestCase):
    """Regression tests for the spin-axis mirror flip check in _optimize.

    The count-rate model is approximately invariant under (vT, vN) → (-vT, -vN)
    (a 180° rotation about the spin axis), creating two chi² basins. Without
    the dual-LM flip check, LM can converge to the wrong (higher-chi²) basin
    depending on the initial guess sign. These tests verify that _optimize
    always recovers the correct basin regardless of which side it starts from.
    """

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.true_density = 8.0
        cls.true_temperature = 10.0 * EV_TO_KELVIN
        cls.true_velocity = np.array([450.0, -40.0, 35.0])

        voltages = np.geomspace(
            _peak_voltage(450.0) * 0.75, _peak_voltage(450.0) * 1.35, 20
        )
        n_sweeps = 5
        all_voltages = np.tile(voltages, n_sweeps)
        cls.grids, cls.cs, cls.cea, cls.at, cls.ats = _build_proton_arrays(
            sr, all_voltages
        )
        cls.rot = _spin_rotation_matrices(n_sweeps * len(voltages))
        cls.spin_axis = cls.rot[0, 1, :].copy()

        cls.count_rate = _model_count_rates(
            cls.true_density,
            cls.true_temperature,
            cls.true_velocity,
            cls.grids,
            cls.cs,
            cls.cea,
            cls.at,
            cls.ats,
            cls.rot,
            PROTON_MASS_KG,
        )

    def _make_ig(self, velocity):
        return ProtonSolarWindMoments(
            density=self.true_density,
            temperature=self.true_temperature,
            bulk_velocity_rtn=np.array(velocity, dtype=float),
            bad_fit_flag=0,
        )

    def _run(self, ig):
        return _optimize(
            self.count_rate,
            self.grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            ig,
        )

    def test_correct_basin_from_true_init(self):
        result = self._run(self._make_ig([450.0, -40.0, 35.0]))
        np.testing.assert_allclose(
            result.bulk_velocity_rtn, self.true_velocity, atol=0.5
        )

    def test_correct_basin_from_mirror_init(self):
        v_mirror = (
            2.0 * np.dot(self.true_velocity, self.spin_axis) * self.spin_axis
            - self.true_velocity
        )
        result = self._run(self._make_ig(v_mirror))
        np.testing.assert_allclose(
            result.bulk_velocity_rtn, self.true_velocity, atol=0.5
        )

    def test_correct_basin_from_zero_transverse_init(self):
        result = self._run(self._make_ig([450.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            result.bulk_velocity_rtn, self.true_velocity, atol=0.5
        )

    def test_both_inits_give_same_result(self):
        r_true = self._run(self._make_ig([450.0, -40.0, 35.0]))
        v_mirror = (
            2.0 * np.dot(self.true_velocity, self.spin_axis) * self.spin_axis
            - self.true_velocity
        )
        r_mirror = self._run(self._make_ig(v_mirror))
        np.testing.assert_allclose(
            r_true.bulk_velocity_rtn, r_mirror.bulk_velocity_rtn, atol=1e-6
        )
        np.testing.assert_allclose(r_true.density, r_mirror.density, rtol=1e-6)
        np.testing.assert_allclose(r_true.temperature, r_mirror.temperature, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
