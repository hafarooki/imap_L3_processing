import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import _get_initial_guess
from imap_l3_processing.swapi.l3a.science.speed_calculation import esa_voltage_to_proton_speed, SWAPI_K_FACTOR


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
