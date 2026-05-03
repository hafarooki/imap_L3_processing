from unittest import TestCase
from unittest.mock import create_autospec, patch, MagicMock

import numpy as np

from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    calculate_alpha_solar_wind_vdf, calculate_pui_solar_wind_vdf, \
    calculate_delta_minus_plus


class TestCalculateSolarWindVDF(TestCase):
    @patch('imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf.calculate_geometric_factor')
    def test_calculate_proton_solar_wind_vdf(self, mock_calc_gf):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        eff_correction = 0.0882
        mock_swapi_response = MagicMock(spec=SWAPIResponse)

        mock_calc_gf.side_effect = [100.0, 75.0, 50.0]

        velocities, probabilities = calculate_proton_solar_wind_vdf(energies, count_rates, eff_correction,
                                                                    mock_swapi_response)

        expected_velocities = [437.6947142244463, 379.05474162054054, 309.496900517614]
        np.testing.assert_array_equal(velocities, expected_velocities)
        self.assertEqual(mock_calc_gf.call_count, 3)

    def test_calculate_delta_minus_plus(self):
        energies = np.array([8, 32, 128])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([4, 16, 64], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([8, 32, 128], delta_minus_plus.delta_plus)

    def test_calculate_delta_minus_plus_reversed(self):
        energies = np.array([128, 32, 8])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([64, 16, 4], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([128, 32, 8], delta_minus_plus.delta_plus)

    def test_calculate_delta_minus_plus_uneven(self):
        energies = np.array([8, 32, 288])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([4, 16, 192], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([8, 64, 576], delta_minus_plus.delta_plus)

    @patch('imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf.calculate_geometric_factor')
    def test_calculate_alpha_solar_wind_vdf(self, mock_calc_gf):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        eff_correction = 0.0882
        mock_swapi_response = MagicMock(spec=SWAPIResponse)

        mock_calc_gf.side_effect = [100.0, 75.0, 50.0]

        velocities, probabilities = calculate_alpha_solar_wind_vdf(energies, count_rates, eff_correction,
                                                                   mock_swapi_response)

        expected_velocities = [310.5624166704235, 268.95494229727456, 219.6007908093385]
        np.testing.assert_array_equal(velocities, expected_velocities)

    @patch('imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf.calculate_geometric_factor')
    def test_calculate_pickup_ion_solar_wind_vdf(self, mock_calc_gf):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        eff_correction = 0.0882
        mock_swapi_response = MagicMock(spec=SWAPIResponse)

        mock_calc_gf.side_effect = [100.0, 75.0, 50.0]

        velocities, probabilities = calculate_pui_solar_wind_vdf(energies, count_rates, eff_correction,
                                                                 mock_swapi_response)

        expected_velocities = [219.58573945228636, 190.16682867447082, 155.27056541857408]
        np.testing.assert_array_equal(velocities, expected_velocities)
