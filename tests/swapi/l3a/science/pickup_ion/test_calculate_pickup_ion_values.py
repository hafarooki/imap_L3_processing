import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from astropy import constants, units

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_fit_energy_range,
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

_MODULE_PATH = (
    "imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values"
)
_N_SWEEPS = 5
_N_COARSE_BINS = 62
_SW_VELOCITY_RTN_KMS = np.array([400.0, 0.0, 0.0])
_VOLTAGE_PER_STEP = np.linspace(100.0, 8000.0, _N_COARSE_BINS)


_ENERGY_RANGE_ADMITTING_EVERY_BIN = (0.0, 1.0e9)


def _vasyliunas_siscoe_distribution():
    """A Vasyliunas-Siscoe distribution with a finite distance so the
    `min_speed_kms` calculation runs."""
    distribution = MagicMock(spec=VasyliunasSiscoeDistribution)
    distribution.distance_km = ONE_AU_IN_KM
    return distribution


def _density_lookup_table():
    table = MagicMock()
    table.get_minimum_distance.return_value = 1.0
    return table


def _good_nominal(**overrides):
    base = {
        "cooling_index": 1.5,
        "ionization_rate": 1e-7,
        "cutoff_speed": 450.0,
        "background_count_rate": 0.1,
    }
    base.update(overrides)
    return base


def _run_calculate_with_mocked_fit(
    *,
    nominal,
    observed_per_step,
    modeled_per_step,
    cov_external_diag=(1.0, 1.0, 1.0, 1.0),
):
    """Drive `calculate_pickup_ion_values` through the post-fit branches.

    `observed_per_step` and `modeled_per_step` are tiled across `_N_SWEEPS`
    sweeps; differences between them set the residual sum of squares that
    feeds the R² guard. `cov_external_diag` becomes the diagonal of the
    mocked external-coordinate covariance — passing negative entries yields
    NaN σ̂ and exercises the non-positive-definite Hessian branch."""
    voltages = np.tile(_VOLTAGE_PER_STEP, (_N_SWEEPS, 1))
    observed_per_step = np.broadcast_to(
        np.asarray(observed_per_step, dtype=float), (_N_COARSE_BINS,)
    )
    modeled_per_step = np.broadcast_to(
        np.asarray(modeled_per_step, dtype=float), (_N_COARSE_BINS,)
    )
    count_rates = np.tile(observed_per_step, (_N_SWEEPS, 1))
    modeled_rates = np.tile(modeled_per_step, (_N_SWEEPS, 1))
    bulk_sw_per_bin_swapi_kms = np.tile(
        _SW_VELOCITY_RTN_KMS, (_N_SWEEPS, _N_COARSE_BINS, 1)
    )

    fake_result = MagicMock()
    fake_result.var_names = [
        "cooling_index",
        "ionization_rate",
        "cutoff_speed",
        "background_count_rate",
    ]
    fake_result.x = np.zeros(4)
    fake_result.params.valuesdict.return_value = nominal
    fake_minimizer = MagicMock()
    fake_minimizer.minimize.return_value = fake_result
    fake_minimizer._int2ext_cov_x.return_value = np.diag(cov_external_diag)

    with patch(f"{_MODULE_PATH}.build_chunk_collapsed_response") as mock_build, patch(
        f"{_MODULE_PATH}.lmfit.Minimizer", return_value=fake_minimizer
    ), patch(
        f"{_MODULE_PATH}.ndt.Hessian", return_value=lambda _: np.eye(4)
    ), patch(
        f"{_MODULE_PATH}.calculate_coincidence_rate", return_value=modeled_rates
    ), patch(
        f"{_MODULE_PATH}.calculate_pickup_ion_fit_energy_range",
        return_value=_ENERGY_RANGE_ADMITTING_EVERY_BIN,
    ):
        mock_build.return_value = MagicMock()
        return calculate_pickup_ion_values(
            swapi_response=MagicMock(),
            voltages=voltages,
            count_rates=count_rates,
            sw_velocity_rtn_kms=_SW_VELOCITY_RTN_KMS,
            bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
            density_of_neutral_helium_lookup_table=_density_lookup_table(),
            vasyliunas_siscoe_distribution=_vasyliunas_siscoe_distribution(),
        )


def _assert_all_nan_params(tc, fitting_params):
    for value in (
        fitting_params.cooling_index,
        fitting_params.ionization_rate,
        fitting_params.cutoff_speed,
        fitting_params.background_count_rate,
    ):
        tc.assertTrue(np.isnan(value.nominal_value))
        tc.assertTrue(np.isnan(value.std_dev))


class CalculatePickupIonValuesFillTest(unittest.TestCase):
    """Tests for the post-fit guards in `calculate_pickup_ion_values`, with the
    optimizer, Hessian and coincidence-rate seams mocked so each test isolates
    one guard."""

    def test_zero_variance_observations_fill_all_params_with_bad_fit(self):
        """When every observed count rate is identical the total sum of
        squares is zero and R² is undefined; `BAD_FIT` is set and every
        parameter is reported as NaN ± NaN."""
        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=5.0,
            modeled_per_step=5.0,
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_low_r_squared_fills_all_params_with_bad_fit(self):
        """When the model misses non-constant observations badly enough that
        R² < 0.9, `BAD_FIT` is set and every parameter is reported as NaN ±
        NaN — values are not retained."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)
        modeled_per_step = np.zeros(_N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=modeled_per_step,
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_non_positive_definite_hessian_fills_all_params_with_bad_fit(self):
        """A non-positive-definite Hessian gives a covariance with negative
        diagonal entries; `np.sqrt(np.diag(cov))` then yields NaN σ̂. The
        guard sets `BAD_FIT` and every parameter is NaN ± NaN."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
            cov_external_diag=(-1.0, -1.0, -1.0, -1.0),
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_background_above_one_hz_fills_background_only(self):
        """When the fitted background exceeds 1 Hz the flat term is absorbing
        real signal; the background is reported as NaN ± NaN, the other three
        parameters are unchanged, and the fit flag stays NONE."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.5),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        self.assertTrue(np.isnan(fitting_params.background_count_rate.nominal_value))
        self.assertTrue(np.isnan(fitting_params.background_count_rate.std_dev))
        for value in (
            fitting_params.cooling_index,
            fitting_params.ionization_rate,
            fitting_params.cutoff_speed,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))

    def test_background_at_one_hz_is_not_filled(self):
        """The background guard uses a strict inequality (`> 1.0`); a fit
        sitting exactly at 1 Hz is retained."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.0),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        self.assertEqual(fitting_params.background_count_rate.nominal_value, 1.0)
        self.assertTrue(np.isfinite(fitting_params.background_count_rate.std_dev))

    def test_clean_fit_returns_all_finite_params_with_no_flag(self):
        """A perfect fit (R² = 1) with a background ≤ 1 Hz returns all four
        parameters with finite nominal and σ̂ and the fit flag is NONE — the
        baseline against which the fill-value branches above are deviations."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        for value in (
            fitting_params.cooling_index,
            fitting_params.ionization_rate,
            fitting_params.cutoff_speed,
            fitting_params.background_count_rate,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))


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
