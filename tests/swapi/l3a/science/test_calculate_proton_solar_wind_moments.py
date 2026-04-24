import math
import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _compute_angles,
    calculate_integral,
    _get_angular_limits,
    _dynamic_limits,
    SWParams,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import esa_voltage_to_proton_speed, SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from tests.test_helpers import get_test_instrument_team_data_path

_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-azimuthal-transmission_20250101_v001.csv")
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-central-effective-area_20250101_v001.csv")
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-passband-fit-coefficients_20250101_v001.csv")


def _trapz_weights(a, b, n):
    w = np.full(n, (b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _passband_vectorized(grid, is_sg, el_arr, sp_arr):
    """Vectorized bilinear passband interpolation. Returns (n_el, n_sp)."""
    gv = grid.values_sunglasses if is_sg else grid.values_open_aperture
    i = (el_arr[:, None] - grid.min_elevation) / grid.elevation_spacing
    j = sp_arr[None, :] / grid.central_speed
    j = (j - grid.min_speed_ratio) / grid.speed_ratio_spacing
    valid = (i >= 0) & (i + 1 < gv.shape[0]) & (j >= 0) & (j + 1 < gv.shape[1])
    i0 = np.clip(i.astype(int), 0, gv.shape[0] - 2)
    j0 = np.clip(j.astype(int), 0, gv.shape[1] - 2)
    wi = i - i0
    wj = j - j0
    result = (
        (1 - wi) * ((1 - wj) * gv[i0, j0] + wj * gv[i0, j0 + 1])
        + wi * ((1 - wj) * gv[i0 + 1, j0] + wj * gv[i0 + 1, j0 + 1])
    )
    return np.where(valid, result, 0.0)


def _transmission_vectorized(azimuthal_transmission, spacing, az_arr):
    """Vectorized azimuthal transmission interpolation."""
    az = (az_arr + 180) % 360 - 180
    i = np.abs(az) / spacing
    i0 = np.clip(i.astype(int), 0, len(azimuthal_transmission) - 1)
    i1 = np.clip(i0 + 1, 0, len(azimuthal_transmission) - 1)
    return azimuthal_transmission[i0] * (i1 - i) + azimuthal_transmission[i1] * (i - i0)


def _calculate_integral_highres(grid, sw_params, n=200):
    """High-resolution trapezoid reference for benchmarking calculate_integral."""
    sin_be = math.sin(math.radians(sw_params.bulk_elevation))
    cos_be = math.cos(math.radians(sw_params.bulk_elevation))
    count_rate = 0.0

    for region in (0, -1, 1):
        is_sg = region == 0
        passband_norm = float(
            _passband_vectorized(grid, is_sg, np.array([0.0]), np.array([grid.central_speed]))[0, 0]
        )
        if passband_norm == 0:
            continue

        min_el, max_el, min_az, max_az = _get_angular_limits(sw_params, region, grid)
        if max_el <= min_el or max_az <= min_az:
            continue

        el = np.linspace(min_el, max_el, n)
        az = np.linspace(min_az, max_az, n)

        # Speed range: passband grid bounds intersected with ±5 v_th
        sp_lo, sp_hi = _dynamic_limits(
            sw_params.bulk_speed, sw_params.thermal_speed * 5,
            grid.central_speed * grid.min_speed_ratio,
            grid.central_speed * (grid.min_speed_ratio + (101 - 1) * grid.speed_ratio_spacing),
        )
        sp = np.linspace(sp_lo, sp_hi, n)

        el_w = _trapz_weights(min_el, max_el, n)
        az_w = _trapz_weights(min_az, max_az, n)
        sp_w = _trapz_weights(sp_lo, sp_hi, n)

        trans = _transmission_vectorized(grid.azimuthal_transmission, grid.azimuthal_transmission_spacing, az)

        # passband × speed³, shape (n_el, n_sp)
        pb_sp3 = _passband_vectorized(grid, is_sg, el, sp) * sp[None, :] ** 3 / passband_norm

        sin_el = np.sin(np.radians(el))
        cos_el = np.cos(np.radians(el))

        # cos_alpha (n_el, n_az)
        cos_alpha = (
            sin_be * sin_el[:, None]
            + cos_be * cos_el[:, None] * np.cos(np.radians(az[None, :] - sw_params.bulk_azimuth))
        )

        # exponential term (n_el, n_az, n_sp)
        exponent = -(
            sp[None, None, :] ** 2
            + sw_params.bulk_speed ** 2
            - 2 * sp[None, None, :] * sw_params.bulk_speed * cos_alpha[:, :, None]
        ) / (2 * sw_params.thermal_speed ** 2)
        exp_vals = np.exp(exponent)

        # integrate: speed → azimuth → elevation
        sp_integral = np.einsum('s,es,eas->ea', sp_w, pb_sp3, exp_vals)
        az_integral = np.einsum('a,a,ea->e', az_w, trans, sp_integral)
        el_integral = np.dot(el_w * cos_el, az_integral)

        count_rate += (
            el_integral
            * grid.central_effective_area
            * sw_params.density
            * (np.sqrt(2 * np.pi) * sw_params.thermal_speed) ** -3
            * 1e5
            * (math.pi / 180) ** 2
        )

    return count_rate


def _make_sw_params(grid, density=5.0, temperature_ev=10.0, bulk_speed=450.0, bulk_azimuth=15.0, bulk_elevation=-5.0):
    thermal_speed = float(
        np.sqrt(temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
    )
    return SWParams(
        density=density,
        bulk_speed=bulk_speed,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
        thermal_speed=thermal_speed,
    )


class TestEsaVoltageToProtonSpeed(unittest.TestCase):
    def test_known_value(self):
        # At V = 1000 V, E = k* * V = 1.89 keV
        # v = sqrt(2 * 1890 eV * e / m_p) = 601.730748 km/s (independently computed)
        np.testing.assert_allclose(esa_voltage_to_proton_speed(1000.0), 601.730748, rtol=1e-3)

    def test_uses_absolute_value_of_voltage(self):
        np.testing.assert_allclose(esa_voltage_to_proton_speed(-529.0), esa_voltage_to_proton_speed(529.0))

    def test_vectorized(self):
        speeds = esa_voltage_to_proton_speed(np.array([500.0, 1000.0, 2000.0]))
        self.assertEqual(speeds.shape, (3,))
        self.assertTrue(np.all(np.diff(speeds) > 0))
        np.testing.assert_allclose(speeds[1], 601.730748, rtol=1e-3)


class TestComputeAngles(unittest.TestCase):
    def test_antisunward_beam_with_identity_rotation(self):
        # v_xyz = R @ (v_RTN - v_sc) = [450, 0, 0]
        # phi = arctan2(-v_x, -v_y) = arctan2(-450, 0) = -90°
        # theta = arcsin(-v_z / |v_RTN|) = 0°
        phi, theta = _compute_angles(np.array([450.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
        np.testing.assert_allclose(phi, -90.0, atol=1e-6)
        np.testing.assert_allclose(theta, 0.0, atol=1e-6)

    def test_spacecraft_velocity_shifts_apparent_direction(self):
        # Subtracting SC velocity changes the apparent flow direction in instrument frame
        bulk_velocity_rtn = np.array([450.0, 0.0, 0.0])
        sc_velocity = np.array([0.0, 30.0, 0.0])
        phi_no_sc, _ = _compute_angles(bulk_velocity_rtn, np.eye(3), np.zeros(3))
        phi_with_sc, _ = _compute_angles(bulk_velocity_rtn, np.eye(3), sc_velocity)
        # With SC lateral velocity, apparent beam direction shifts in azimuth
        self.assertFalse(np.isclose(phi_no_sc, phi_with_sc))

    def test_elevation_nonzero_for_z_component(self):
        bulk_velocity_rtn = np.array([450.0, 0.0, 50.0])
        _, theta = _compute_angles(bulk_velocity_rtn, np.eye(3), np.zeros(3))
        v_b = np.linalg.norm(bulk_velocity_rtn)
        expected_theta = np.degrees(np.arcsin(-50.0 / v_b))
        np.testing.assert_allclose(theta, expected_theta, atol=1e-6)

    def test_rotation_matrix_applied(self):
        # A 90° rotation about z: x→y, y→-x
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        phi_identity, theta_identity = _compute_angles(np.array([450.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
        phi_rotated, theta_rotated = _compute_angles(np.array([450.0, 0.0, 0.0]), R, np.zeros(3))
        # Elevation unchanged (no z component), but azimuth shifts
        np.testing.assert_allclose(theta_rotated, theta_identity, atol=1e-6)
        self.assertFalse(np.isclose(phi_rotated, phi_identity))


class TestCalculateIntegral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.swapi_response = SWAPIResponse.from_files(
            _AZIMUTHAL_TRANSMISSION_PATH,
            _CENTRAL_EFFECTIVE_AREA_PATH,
            _PASSBAND_FIT_COEFFICIENTS_PATH,
        )
        # Peak voltage for 450 km/s
        cls.peak_voltage = float(
            PROTON_MASS_KG * (450.0 * METERS_PER_KILOMETER) ** 2
            / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
        )

    def test_count_rate_is_positive(self):
        grid = self.swapi_response.create_passband_grid(self.peak_voltage)
        sw_params = _make_sw_params(grid)
        self.assertGreater(calculate_integral(grid, sw_params), 0.0)

    def test_count_rate_scales_linearly_with_density(self):
        grid = self.swapi_response.create_passband_grid(self.peak_voltage)
        result_1 = calculate_integral(grid, _make_sw_params(grid, density=1.0))
        result_5 = calculate_integral(grid, _make_sw_params(grid, density=5.0))
        np.testing.assert_allclose(result_5, 5.0 * result_1, rtol=1e-10)

    def test_count_rate_drops_away_from_peak(self):
        # Count rate should be much lower at 2x peak voltage (central speed = sqrt(2) * bulk_speed,
        # passband well above the Maxwellian peak) than at peak voltage
        grid_peak = self.swapi_response.create_passband_grid(self.peak_voltage)
        grid_high = self.swapi_response.create_passband_grid(self.peak_voltage * 2.0)
        rate_peak = calculate_integral(grid_peak, _make_sw_params(grid_peak))
        rate_high = calculate_integral(grid_high, _make_sw_params(grid_high))
        self.assertGreater(rate_peak, rate_high)

    def test_matches_highres_trapezoid_at_peak_voltage(self):
        grid = self.swapi_response.create_passband_grid(self.peak_voltage)
        sw_params = _make_sw_params(grid)
        result = calculate_integral(grid, sw_params)
        reference = _calculate_integral_highres(grid, sw_params)
        np.testing.assert_allclose(result, reference, rtol=0.01)

    def test_matches_highres_trapezoid_across_sweep(self):
        # Check accuracy at voltages near the proton peak where the integral is well-resolved
        for voltage in np.geomspace(self.peak_voltage * 0.8, self.peak_voltage * 1.3, 5):
            grid = self.swapi_response.create_passband_grid(voltage)
            sw_params = _make_sw_params(grid)
            result = calculate_integral(grid, sw_params)
            reference = _calculate_integral_highres(grid, sw_params)
            np.testing.assert_allclose(result, reference, rtol=0.01,
                                       err_msg=f"Failed at {voltage:.0f} V")

    def test_matches_highres_trapezoid_off_axis_beam(self):
        # Non-zero azimuth and elevation stress test angular integration
        grid = self.swapi_response.create_passband_grid(self.peak_voltage)
        sw_params = _make_sw_params(grid, bulk_azimuth=80.0, bulk_elevation=5.0)
        result = calculate_integral(grid, sw_params)
        reference = _calculate_integral_highres(grid, sw_params)
        np.testing.assert_allclose(result, reference, rtol=0.01)


class TestGetInitialGuess(unittest.TestCase):
    def setUp(self):
        true_bulk_speed = 450.0    # km/s
        true_thermal_speed = 40.0  # km/s
        self.true_density = 5.0    # cm^-3
        self.true_temperature = PROTON_MASS_KG * (true_thermal_speed * METERS_PER_KILOMETER) ** 2 / PROTON_CHARGE_COULOMBS

        peak_voltage = PROTON_MASS_KG * (true_bulk_speed * METERS_PER_KILOMETER) ** 2 / (
            2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
        self.voltages = np.geomspace(peak_voltage * 0.5, peak_voltage * 2.0, 62)
        speeds = esa_voltage_to_proton_speed(self.voltages)

        # TODO update with a more realistic model input
        self.count_rate = self.true_density * np.exp(
            -(speeds - true_bulk_speed) ** 2 / (2 * true_thermal_speed ** 2))
        self.rotation_matrices = np.tile(np.eye(3), (len(self.voltages), 1, 1))
        self.true_bulk_speed = true_bulk_speed

    def _run(self, spacecraft_velocity_rtn=None):
        if spacecraft_velocity_rtn is None:
            spacecraft_velocity_rtn = np.zeros(3)
        with patch('imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments._model_count_rates',
                   return_value=self.count_rate / self.true_density):
            return _get_initial_guess(self.count_rate, self.voltages, None, self.rotation_matrices, spacecraft_velocity_rtn)

    def test_recovers_bulk_speed(self):
        result = self._run()
        np.testing.assert_allclose(result.bulk_velocity_rtn[0], self.true_bulk_speed, rtol=0.01)

    def test_recovers_temperature(self):
        result = self._run()
        np.testing.assert_allclose(result.temperature, self.true_temperature, rtol=0.05)

    def test_recovers_density(self):
        result = self._run()
        np.testing.assert_allclose(result.density, self.true_density, rtol=0.01)

    def test_bulk_velocity_is_anti_sunward(self):
        result = self._run()
        self.assertGreater(result.bulk_velocity_rtn[0], 0)
        np.testing.assert_allclose(result.bulk_velocity_rtn[1:], 0.0)

    def test_spacecraft_velocity_does_not_affect_initial_bulk_velocity(self):
        sc_velocity = np.array([10.0, 5.0, -3.0])
        result = self._run(spacecraft_velocity_rtn=sc_velocity)
        np.testing.assert_allclose(result.bulk_velocity_rtn,
                                   np.array([self.true_bulk_speed, 0.0, 0.0]), rtol=0.01)


if __name__ == '__main__':
    unittest.main()
