from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np

from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import \
    calculate_combined_solar_wind_differential_flux


class TestCalculateSolarWindDifferentialFlux(TestCase):
    @patch('imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux.calculate_geometric_factor')
    def test_calculate_combined_differential_flux(self, mock_calc_gf):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        eff_correction = 0.0882
        mock_swapi_response = MagicMock(spec=SWAPIResponse)

        mock_calc_gf.side_effect = [100.0, 75.0, 50.0]

        differential_flux = calculate_combined_solar_wind_differential_flux(
            energies, count_rates, eff_correction, mock_swapi_response)

        expected_flux = count_rates / (np.array([100.0, 75.0, 50.0]) * eff_correction)
        np.testing.assert_array_almost_equal(expected_flux, differential_flux)
        self.assertEqual(mock_calc_gf.call_count, 3)
