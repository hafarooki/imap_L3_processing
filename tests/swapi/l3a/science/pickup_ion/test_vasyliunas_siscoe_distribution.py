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
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)
from tests.test_helpers import NumpyArrayMatcher

_VASYLIUNAS_SISCOE_MODULE = "imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution"


class VasyliunasSiscoeDistributionFTest(unittest.TestCase):
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

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_evaluates_filled_shell_vdf_with_heaviside_cutoff(self, mock_density):
        mock_density.return_value = 1

        fitting_parameters = FittingParameters(
            ionization_rate=0.47,
            cutoff_speed=500,
        )
        ephemeris_time = 1_234_567.1
        solar_wind_speed_inertial_frame = 456
        distance_km = 0.99 * ONE_AU_IN_KM
        psi = 13

        vasyliunas_siscoe_distribution = VasyliunasSiscoeDistribution(
            ephemeris_time,
            solar_wind_speed_inertial_frame,
            self.density_of_neutral_helium_lookup_table,
            distance_km,
            psi,
        )
        speed_grid = np.array([485.45, 200, 585.45, 755.45])

        result = vasyliunas_siscoe_distribution.f(speed_grid, fitting_parameters)

        expected_term_1 = SWAPI_PUI_COOLING_INDEX / (4 * np.pi)
        expected_term_2 = (0.47 * ONE_AU_IN_KM**2) / (
            distance_km * solar_wind_speed_inertial_frame * 500**3
        )
        expected_term_3 = (speed_grid / 500) ** (SWAPI_PUI_COOLING_INDEX - 3)
        expected_term_4 = 1 * 1e15  # cm^-3 → km^-3
        expected_term_5 = np.array([1, 1, 0, 0])

        expected = (
            expected_term_1
            * expected_term_2
            * expected_term_3
            * expected_term_4
            * expected_term_5
        )
        np.testing.assert_array_equal(result, expected)

        mock_density.assert_called_with(
            psi,
            NumpyArrayMatcher(
                distance_km / ONE_AU_IN_KM * (speed_grid / 500) ** SWAPI_PUI_COOLING_INDEX,
            ),
        )

    @patch(f"{_VASYLIUNAS_SISCOE_MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_apply_cutoff_false_omits_heaviside_so_speeds_above_cutoff_stay_nonzero(
        self, mock_density
    ):
        """With apply_cutoff=False the w<1 Heaviside is dropped, so the
        returned distribution is the full product even for speeds above the
        cutoff — used by the forward model so the cutoff can be applied as a
        grid-corrected partial cell instead of a hard grid step."""
        mock_density.return_value = 1

        fitting_parameters = FittingParameters(
            ionization_rate=0.47,
            cutoff_speed=500,
        )
        distance_km = 0.99 * ONE_AU_IN_KM
        vasyliunas_siscoe_distribution = VasyliunasSiscoeDistribution(
            1_234_567.1, 456, self.density_of_neutral_helium_lookup_table, distance_km, 13
        )
        speed_grid = np.array([485.45, 200, 585.45, 755.45])

        result = vasyliunas_siscoe_distribution.f(
            speed_grid, fitting_parameters, apply_cutoff=False
        )

        expected = (
            SWAPI_PUI_COOLING_INDEX / (4 * np.pi)
            * (0.47 * ONE_AU_IN_KM**2)
            / (distance_km * 456 * 500**3)
            * (speed_grid / 500) ** (SWAPI_PUI_COOLING_INDEX - 3)
            * 1e15
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        self.assertTrue(np.all(result > 0))


if __name__ == "__main__":
    unittest.main()
