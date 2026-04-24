import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from tests.test_helpers import get_test_instrument_team_data_path

AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-azimuthal-transmission_20250101_v001.csv")
CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-central-effective-area_20250101_v001.csv")
PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_proton-sw-passband-fit-coefficients_20250101_v001.csv")


class TestSWAPIResponseFromFiles(unittest.TestCase):
    def setUp(self):
        self.response = SWAPIResponse.from_files(
            AZIMUTHAL_TRANSMISSION_PATH,
            CENTRAL_EFFECTIVE_AREA_PATH,
            PASSBAND_FIT_COEFFICIENTS_PATH,
        )

    def test_azimuthal_transmission_loaded(self):
        # CSV is spaced at 0.1 deg; 0.0 to 180.0 inclusive = 1801 rows
        self.assertEqual(len(self.response.azimuthal_transmission), 1801)

    def test_azimuthal_transmission_at_zero_azimuth(self):
        # sunward direction is heavily blocked
        npt.assert_equal(self.response.azimuthal_transmission[0], 0.001)

    def test_central_effective_area_loaded(self):
        self.assertEqual(len(self.response.central_effective_area), len(self.response.central_effective_area_voltage))
        self.assertGreater(len(self.response.central_effective_area), 0)

    def test_passband_fit_coefficients_loaded(self):
        self.assertIn('OA', self.response.passband_fit_coefficients.index.get_level_values('region'))
        self.assertIn('SG', self.response.passband_fit_coefficients.index.get_level_values('region'))
        self.assertEqual(list(self.response.passband_fit_coefficients.columns), [2, 1, 0])


class TestGetCentralEffectiveArea(unittest.TestCase):
    def setUp(self):
        self.response = SWAPIResponse.from_files(
            AZIMUTHAL_TRANSMISSION_PATH,
            CENTRAL_EFFECTIVE_AREA_PATH,
            PASSBAND_FIT_COEFFICIENTS_PATH,
        )

    def test_known_value_at_1kev(self):
        # 1 keV proton ESA voltage = 1000 / 1.89 ≈ 529 V
        npt.assert_approx_equal(self.response.get_central_effective_area(1000 / 1.89), 0.38, significant=2)

    def test_uses_absolute_value_of_voltage(self):
        v = 1000 / 1.89
        npt.assert_equal(
            self.response.get_central_effective_area(-v),
            self.response.get_central_effective_area(v),
        )


class TestGetPassbandValues(unittest.TestCase):
    def setUp(self):
        self.response = SWAPIResponse.from_files(
            AZIMUTHAL_TRANSMISSION_PATH,
            CENTRAL_EFFECTIVE_AREA_PATH,
            PASSBAND_FIT_COEFFICIENTS_PATH,
        )

    def test_returns_correct_number_of_rows_for_oa(self):
        result = self.response.get_passband_values(529.0, 'OA')
        self.assertEqual(len(result), 156)

    def test_returns_correct_number_of_rows_for_sg(self):
        result = self.response.get_passband_values(529.0, 'SG')
        self.assertEqual(len(result), 112)

    def test_values_are_non_negative(self):
        result = self.response.get_passband_values(529.0, 'OA')
        self.assertTrue((result['value'] >= 0).all())

    def test_peak_oa_value_near_one_at_central_energy(self):
        # At the voltage corresponding to the SIMION calibration energy (1 keV / 1.89),
        # the passband peak should be close to 1 (it's normalized per beam energy)
        result = self.response.get_passband_values(1000 / 1.89, 'OA')
        npt.assert_approx_equal(result['value'].max(), 1.0, significant=1)

    def test_index_has_energy_ratio_and_elevation(self):
        result = self.response.get_passband_values(529.0, 'OA')
        self.assertEqual(result.index.names, ['energy_ratio', 'elevation'])


if __name__ == '__main__':
    unittest.main()
