import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import imap_l3_processing
from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.uniform_speed_grid import (
    UniformSpeedGrid,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import \
    vasyliunas_siscoe_vdf
from tests.test_helpers import NumpyArrayMatcher

_VASYLIUNAS_SISCOE_MODULE = "imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution"

_IONIZATION_RATE_HZ = 0.47
_SOLAR_WIND_SPEED_INERTIAL_KMS = 456.0
_DISTANCE_KM = 0.99 * ONE_AU_IN_KM
_INFLOW_ANGLE_DEG = 13.0

_SPEED_GRID = UniformSpeedGrid(600.0)


class VasyliunasSiscoeVdfTest(unittest.TestCase):
    def setUp(self) -> None:
        density_lut_path = (
            Path(imap_l3_processing.__file__).parent.parent
            / "tests"
            / "test_data"
            / "swapi"
            / "imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
        )
        self.density_of_neutral_helium_lookup_table = (
            DensityOfNeutralHeliumLookupTable.from_file(density_lut_path)
        )

    def _evaluate(self, cutoff_speed: float) -> np.ndarray:
        return vasyliunas_siscoe_vdf(
            _SPEED_GRID,
            ionization_rate=_IONIZATION_RATE_HZ,
            cutoff_speed=cutoff_speed,
            distance=_DISTANCE_KM,
            inflow_angle=_INFLOW_ANGLE_DEG,
            solar_wind_speed_inertial_frame=_SOLAR_WIND_SPEED_INERTIAL_KMS,
            density_of_neutral_helium_lookup_table=self.density_of_neutral_helium_lookup_table,
        )

    @staticmethod
    def _expected_uncut_filled_shell(cutoff_speed: float) -> np.ndarray:
        # closed form of the filled shell, with the mocked unit neutral
        # helium density converted from cm^-3 to km^-3.
        return (
            SWAPI_PUI_COOLING_INDEX / (4 * np.pi)
            * (_IONIZATION_RATE_HZ * ONE_AU_IN_KM**2)
            / (_DISTANCE_KM * _SOLAR_WIND_SPEED_INERTIAL_KMS * cutoff_speed**3)
            * (_SPEED_GRID.centers / cutoff_speed) ** (SWAPI_PUI_COOLING_INDEX - 3)
            * 1e15
        )

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_scales_the_cutoff_cell_by_the_fraction_below_it_and_zeros_the_cells_above(
        self, mock_density
    ):
        mock_density.return_value = 1
        cutoff_index = 300
        fraction_below_cutoff = 0.3
        cutoff_speed = float(
            _SPEED_GRID.centers[cutoff_index]
            + (fraction_below_cutoff - 0.5) * _SPEED_GRID.spacing
        )

        result = self._evaluate(cutoff_speed)

        expected = self._expected_uncut_filled_shell(cutoff_speed)
        expected[cutoff_index] *= fraction_below_cutoff
        expected[cutoff_index + 1:] = 0.0
        np.testing.assert_allclose(result, expected, rtol=1e-12)

        mock_density.assert_called_with(
            _INFLOW_ANGLE_DEG,
            NumpyArrayMatcher(
                _DISTANCE_KM
                / ONE_AU_IN_KM
                * (_SPEED_GRID.centers / cutoff_speed) ** SWAPI_PUI_COOLING_INDEX,
            ),
        )

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_treats_a_cutoff_on_the_last_node_as_a_half_cell(self, mock_density):
        mock_density.return_value = 1
        cutoff_speed = float(_SPEED_GRID.centers[-1])

        result = self._evaluate(cutoff_speed)

        expected = self._expected_uncut_filled_shell(cutoff_speed)
        expected[-1] *= 0.5
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_returns_the_uncut_filled_shell_when_the_cutoff_is_above_the_grid(
        self, mock_density
    ):
        mock_density.return_value = 1
        cutoff_speed = float(_SPEED_GRID.centers[-1] + _SPEED_GRID.spacing)

        result = self._evaluate(cutoff_speed)

        np.testing.assert_allclose(
            result, self._expected_uncut_filled_shell(cutoff_speed), rtol=1e-12
        )

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_returns_zeros_when_the_cutoff_is_below_the_grid(self, mock_density):
        mock_density.return_value = 1
        # half way from zero up to the first cell's lower edge, so no
        # part of any cell lies below the cutoff.
        cutoff_speed = float(_SPEED_GRID.centers[0] - _SPEED_GRID.spacing / 2) / 2

        result = self._evaluate(cutoff_speed)

        np.testing.assert_array_equal(result, np.zeros(_SPEED_GRID.size))


if __name__ == "__main__":
    unittest.main()
