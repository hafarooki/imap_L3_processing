import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from imap_l3_processing.swapi.l3a.science.swapi_response import (
    SWAPIResponse, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS,
    eval_boundary_min, eval_boundary_max,
)
from tests.test_helpers import get_test_instrument_team_data_path

AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv")
CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv")
PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv")


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
        self.assertEqual(len(result), 679)

    def test_returns_correct_number_of_rows_for_sg(self):
        result = self.response.get_passband_values(529.0, 'SG')
        self.assertEqual(len(result), 467)

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


class TestCreatePassbandGridExtremeVoltages(unittest.TestCase):
    def setUp(self):
        self.response = SWAPIResponse.from_files(
            AZIMUTHAL_TRANSMISSION_PATH,
            CENTRAL_EFFECTIVE_AREA_PATH,
            PASSBAND_FIT_COEFFICIENTS_PATH,
        )

    def _assert_grid_value_bounds(self, esa_voltage):
        grid = self.response.create_passband_grid(esa_voltage)
        for values, label in [
            (grid.values_sunglasses, 'SG'),
            (grid.values_open_aperture, 'OA'),
        ]:
            self.assertGreaterEqual(values.min(), 0.0,
                msg=f"{label} passband has negative values at {esa_voltage} V")
            self.assertLessEqual(values.max(), 1.5,
                msg=f"{label} passband exceeds 1.5 at {esa_voltage} V")

    def test_passband_grid_bounds_at_low_voltage(self):
        self._assert_grid_value_bounds(50.0)

    def test_passband_grid_bounds_at_high_voltage(self):
        self._assert_grid_value_bounds(20000.0)


class TestPassbandPolynomialBoundaries(unittest.TestCase):
    """Verify that the fitted polynomial boundaries land where the passband is zero."""

    @classmethod
    def setUpClass(cls):
        response = SWAPIResponse.from_files(
            AZIMUTHAL_TRANSMISSION_PATH,
            CENTRAL_EFFECTIVE_AREA_PATH,
            PASSBAND_FIT_COEFFICIENTS_PATH,
        )
        cls.grid = response.create_passband_grid(2000.0 / 1.89)  # 2 keV beam

    def _passband_at_speed_ratio(self, grid_values, elevation, speed_ratio):
        """Interpolate passband at (elevation, speed_ratio) using the grid arrays."""
        el_idx = (elevation - self.grid.min_elevation) / self.grid.elevation_spacing
        row = int(round(el_idx))
        return float(np.interp(speed_ratio, _TARGET_SPEED_RATIOS, grid_values[row],
                               left=0.0, right=0.0))

    def _check_boundaries(self, grid_values, bnd_min, bnd_max, region_name):
        for elevation in _TARGET_ELEVATIONS:
            with self.subTest(region=region_name, elevation=elevation):
                row = grid_values[int(round((elevation - self.grid.min_elevation)
                                           / self.grid.elevation_spacing))]
                if not np.any(row > 0):
                    continue  # no passband at this elevation — nothing to check

                min_ratio = float(eval_boundary_min(bnd_min, np.array([elevation]))[0])
                max_ratio = float(eval_boundary_max(bnd_max, np.array([elevation]))[0])

                val_at_min = self._passband_at_speed_ratio(grid_values, elevation, min_ratio)
                val_at_max = self._passband_at_speed_ratio(grid_values, elevation, max_ratio)

                self.assertAlmostEqual(
                    val_at_min, 0.0, places=2,
                    msg=f"{region_name} min boundary at el={elevation} deg: "
                        f"passband={val_at_min:.4f} (speed ratio {min_ratio:.4f})",
                )
                self.assertAlmostEqual(
                    val_at_max, 0.0, places=2,
                    msg=f"{region_name} max boundary at el={elevation} deg: "
                        f"passband={val_at_max:.4f} (speed ratio {max_ratio:.4f})",
                )

    def test_oa_boundary_is_zero(self):
        self._check_boundaries(
            self.grid.values_open_aperture,
            self.grid.min_OA_boundary, self.grid.max_OA_boundary, "OA",
        )

    def test_sg_boundary_is_zero(self):
        self._check_boundaries(
            self.grid.values_sunglasses,
            self.grid.min_SG_boundary, self.grid.max_SG_boundary, "SG",
        )


if __name__ == '__main__':
    unittest.main()
