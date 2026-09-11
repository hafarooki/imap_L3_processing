import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from astropy import constants, units

from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_fit_energy_range,
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000


class CalculatePickupIonValuesGoodnessOfFitTest(unittest.TestCase):
    _MODULE_PATH = (
        "imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values"
    )

    _N_SWEEPS = 50
    _N_COARSE_BINS = 62

    _SW_VELOCITY_RTN_KMS = np.array([400.0, 0.0, 0.0])
    _FITTED_IONIZATION_RATE = 1e-7
    _FITTED_CUTOFF_SPEED_KMS = 450.0

    _VOLTAGE_PER_STEP = np.geomspace(100.0, 10000.0, _N_COARSE_BINS)
    _ESA_ENERGIES = _VOLTAGE_PER_STEP * SWAPI_L2_K_FACTOR
    _OBSERVED_RATES = np.full((_N_SWEEPS, _N_COARSE_BINS), 2.0)

    # narrower than the full sweep, 
    _FIT_ENERGY_RANGE = (5000.0, 18000.0)

    # response and goodness of fit built on steps above lower cutoff
    _MODELED_STEPS = _ESA_ENERGIES > _FIT_ENERGY_RANGE[0]
    _MODEL_RATES = np.full((_N_SWEEPS, int(_MODELED_STEPS.sum())), 3.0)

    def _run_calculate_with_mocked_fit(self, fit_is_good=True):
        fake_result = MagicMock()
        fake_result.var_names = ["ionization_rate", "cutoff_speed"]
        fake_result.x = np.zeros(2)
        fake_result.params.valuesdict.return_value = {
            "ionization_rate": self._FITTED_IONIZATION_RATE,
            "cutoff_speed": self._FITTED_CUTOFF_SPEED_KMS,
        }
        fake_minimizer = MagicMock()
        fake_minimizer.minimize.return_value = fake_result
        fake_minimizer._int2ext_cov_x.return_value = np.eye(2)

        with patch(f"{self._MODULE_PATH}.build_chunk_collapsed_response"), patch(
            f"{self._MODULE_PATH}.lmfit.Minimizer", return_value=fake_minimizer
        ), patch(
            f"{self._MODULE_PATH}.ndt.Hessian", return_value=lambda _: np.eye(2)
        ), patch(
            f"{self._MODULE_PATH}.calculate_coincidence_rate",
            return_value=self._MODEL_RATES,
        ), patch(
            f"{self._MODULE_PATH}.calculate_pickup_ion_fit_energy_range",
            return_value=self._FIT_ENERGY_RANGE,
        ), patch(
            f"{self._MODULE_PATH}.is_good_fit", return_value=fit_is_good
        ) as mock_is_good_fit:
            fit_result = calculate_pickup_ion_values(
                swapi_response=MagicMock(),
                voltages=np.tile(self._VOLTAGE_PER_STEP, (self._N_SWEEPS, 1)),
                count_rates=self._OBSERVED_RATES,
                sw_velocity_rtn_kms=self._SW_VELOCITY_RTN_KMS,
                bulk_sw_per_bin_swapi_kms=np.tile(
                    self._SW_VELOCITY_RTN_KMS,
                    (self._N_SWEEPS, self._N_COARSE_BINS, 1),
                ),
                density_of_neutral_helium_lookup_table=MagicMock(),
                vasyliunas_siscoe_distribution=MagicMock(
                    spec=VasyliunasSiscoeDistribution
                ),
                time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
            )

        return fit_result.fitting_params, mock_is_good_fit

    def test_is_good_fit_is_passed_correct_inputs(self):
        """Ensure is_good_fit is given the right inputs:
            - background-free model rates
            - the instrument background as a separate scalar
            - every step above the lower fitting cutoff, not just the steps
              inside the narrower fit window"""
        _, mock_is_good_fit = self._run_calculate_with_mocked_fit()

        kwargs = mock_is_good_fit.call_args.kwargs
        np.testing.assert_array_equal(kwargs["model_rates"], self._MODEL_RATES)
        np.testing.assert_array_equal(
            kwargs["observed_rates"], self._OBSERVED_RATES[:, self._MODELED_STEPS]
        )
        np.testing.assert_allclose(
            kwargs["esa_energies"], self._ESA_ENERGIES[self._MODELED_STEPS]
        )
        self.assertEqual(kwargs["background_rate"], SWAPI_BACKGROUND_RATE)
        self.assertEqual(kwargs["ionization_rate"], self._FITTED_IONIZATION_RATE)
        self.assertEqual(kwargs["cutoff_speed_kms"], self._FITTED_CUTOFF_SPEED_KMS)
        self.assertEqual(
            kwargs["sw_speed_kms"], np.linalg.norm(self._SW_VELOCITY_RTN_KMS)
        )

    def test_rejected_fit_fills_all_params_with_bad_fit(self):
        """When the the goodness-of-fit check fials, fill values are reported
            and BAD_FIT is set."""
        fitting_params, _ = self._run_calculate_with_mocked_fit(fit_is_good=False)

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        for value in (fitting_params.ionization_rate, fitting_params.cutoff_speed):
            self.assertTrue(np.isnan(value.nominal_value))
            self.assertTrue(np.isnan(value.std_dev))


class CalculatePickupIonFitEnergyRangeTest(unittest.TestCase):
    """Tests for `calculate_pickup_ion_fit_energy_range`."""

    def test_edges_are_scaled_from_the_proton_energy_of_the_given_bulk_speed(self):
        for label, solar_wind_bulk_speed_kms in [
            ("slow wind", 300.0),
            ("nominal wind", 400.0),
            ("fast wind", 750.0),
        ]:
            with self.subTest(case=label):
                bulk_speed = solar_wind_bulk_speed_kms * units.km / units.s
                proton_energy_per_charge_volts = (
                    (0.5 * constants.m_p * bulk_speed**2 / constants.e.si)
                    .to(units.V)
                    .value
                )
                nominal_alpha_peak = 2 * proton_energy_per_charge_volts
                nominal_pickup_ion_cutoff = 16 * proton_energy_per_charge_volts
                expected_upper_edge = nominal_pickup_ion_cutoff
                expected_lower_edge = np.sqrt(
                    nominal_alpha_peak * nominal_pickup_ion_cutoff
                )

                lower_edge, upper_edge = calculate_pickup_ion_fit_energy_range(
                    solar_wind_bulk_speed_kms
                )

                self.assertAlmostEqual(
                    lower_edge,
                    expected_lower_edge,
                    delta=expected_lower_edge * 1e-6,
                    msg=label,
                )
                self.assertAlmostEqual(
                    upper_edge,
                    expected_upper_edge,
                    delta=expected_upper_edge * 1e-6,
                    msg=label,
                )

    def test_edges_scale_with_the_square_of_the_bulk_speed(self):
        """Doubling the bulk speed quadruples both window edges, since the edges
        are fixed multiples of a kinetic energy."""
        lower_edge, upper_edge = calculate_pickup_ion_fit_energy_range(350.0)
        doubled_lower_edge, doubled_upper_edge = calculate_pickup_ion_fit_energy_range(
            700.0
        )

        self.assertAlmostEqual(doubled_lower_edge / lower_edge, 4.0)
        self.assertAlmostEqual(doubled_upper_edge / upper_edge, 4.0)

    def test_lower_edge_is_the_geometric_mean_of_the_alpha_peak_and_the_cutoff(self):
        """The lower edge sits at the logarithmic midpoint between the nominal
        alpha peak (2 E_p) and the nominal He+ cutoff (16 E_p), so it is
        `sqrt(2/16)` of the upper edge."""
        lower_edge, upper_edge = calculate_pickup_ion_fit_energy_range(425.0)

        self.assertAlmostEqual(lower_edge / upper_edge, np.sqrt(2.0 / 16.0))


if __name__ == "__main__":
    unittest.main()
