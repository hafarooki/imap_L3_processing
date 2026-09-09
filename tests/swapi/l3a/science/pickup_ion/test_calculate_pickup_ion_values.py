import unittest
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
from astropy import constants, units

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.constants import SWAPI_L2_K_FACTOR
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    PickupIonFitInputData,
    PickupIonFitResult,
    calculate_pickup_ion_fit_energy_range,
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000

_MODULE_PATH = (
    "imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values"
)
_N_SWEEPS = 50
_SWEEP_LEN = 72
_N_COARSE_BINS = 62
_BULK_SW_VELOCITY_SWAPI_KMS = np.array([400.0, 0.0, 0.0])
_SOLAR_WIND_VELOCITY_RTN_SUN_KMS = np.array([-400.0, 30.0, 0.0])
_SOLAR_WIND_SPEED_INERTIAL_KMS = float(
    np.linalg.norm(_SOLAR_WIND_VELOCITY_RTN_SUN_KMS)
)
_DISTANCE_KM = ONE_AU_IN_KM
_INFLOW_ANGLE_DEG = 75.0
_ENERGY_PER_COARSE_STEP = np.linspace(100.0, 8000.0, _N_COARSE_BINS)

_ENERGY_RANGE_ADMITTING_EVERY_BIN = (0.0, 1.0e9)
_MODELED_RATES = np.full((_N_SWEEPS, _N_COARSE_BINS), 3.0)
_CHUNK_CENTER_TT2000 = NOMINAL_TEST_EPOCH_TT2000


def _good_nominal(**overrides):
    base = {
        "ionization_rate": 1e-7,
        "cutoff_speed": 450.0,
    }
    base.update(overrides)
    return base


class _MockedFitRun(NamedTuple):
    fit_result: PickupIonFitResult
    build_collapsed_response_mock: MagicMock
    is_good_fit_mock: MagicMock
    calculate_coincidence_rate_mock: MagicMock
    calculate_density_mock: MagicMock
    calculate_temperature_mock: MagicMock


def _run_calculate_with_mocked_fit(
    *,
    nominal,
    observed_per_step=1.0,
    bulk_sw_per_bin_swapi_kms=None,
    log_parameter_variances=(0.04, 0.01),
    fit_succeeded=True,
    fit_is_good=True,
):
    """Drive `calculate_pickup_ion_values` through its post-fit branches.

    The response builder, optimizer, HC3 covariance, coincidence-rate model,
    goodness-of-fit check and the two moment integrals are all mocked, so
    `nominal` sets what the fit reports and `fit_is_good` picks which post-fit
    branch is taken.
    `log_parameter_variances` becomes the diagonal of the mocked log-space
    parameter covariance — passing NaN entries exercises the branch where the
    uncertainty estimate is unusable."""
    esa_energies = _ENERGY_PER_COARSE_STEP
    count_rates = np.broadcast_to(
        np.asarray(observed_per_step, dtype=float), (_N_SWEEPS, _N_COARSE_BINS)
    )
    if bulk_sw_per_bin_swapi_kms is None:
        bulk_sw_per_bin_swapi_kms = np.tile(
            _BULK_SW_VELOCITY_SWAPI_KMS, (_N_SWEEPS, _N_COARSE_BINS, 1)
        )

    fake_result = MagicMock()
    fake_result.x = np.log(
        [nominal["ionization_rate"], nominal["cutoff_speed"]]
    )
    fake_result.success = fit_succeeded
    fake_result.jac = np.eye(2)
    fake_result.fun = np.zeros(2)

    with patch(f"{_MODULE_PATH}.build_chunk_collapsed_response") as mock_build, patch(
        f"{_MODULE_PATH}.scipy.optimize.least_squares", return_value=fake_result
    ), patch(
        f"{_MODULE_PATH}.compute_hc3_parameter_covariance",
        return_value=np.diag(log_parameter_variances),
    ), patch(
        f"{_MODULE_PATH}.calculate_coincidence_rate", return_value=_MODELED_RATES
    ) as mock_calculate_coincidence_rate, patch(
        f"{_MODULE_PATH}.calculate_pickup_ion_fit_energy_range",
        return_value=_ENERGY_RANGE_ADMITTING_EVERY_BIN,
    ), patch(
        f"{_MODULE_PATH}.is_good_fit", return_value=fit_is_good
    ) as mock_is_good_fit, patch(
        f"{_MODULE_PATH}.calculate_helium_pui_density"
    ) as mock_calculate_density, patch(
        f"{_MODULE_PATH}.calculate_helium_pui_temperature"
    ) as mock_calculate_temperature:
        mock_build.return_value = MagicMock()
        fit_result = calculate_pickup_ion_values(
            fit_input=PickupIonFitInputData(
                time_as_tt2000=_CHUNK_CENTER_TT2000,
                esa_energies=esa_energies,
                coincidence_count_rates=count_rates,
                bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
                solar_wind_velocity_rtn_sun=_SOLAR_WIND_VELOCITY_RTN_SUN_KMS,
                distance=_DISTANCE_KM,
                inflow_angle=_INFLOW_ANGLE_DEG,
            ),
            swapi_response=MagicMock(),
            density_of_neutral_helium_lookup_table=MagicMock(),
        )

    return _MockedFitRun(
        fit_result=fit_result,
        build_collapsed_response_mock=mock_build,
        is_good_fit_mock=mock_is_good_fit,
        calculate_coincidence_rate_mock=mock_calculate_coincidence_rate,
        calculate_density_mock=mock_calculate_density,
        calculate_temperature_mock=mock_calculate_temperature,
    )


def _assert_all_nan_params(tc, fit_result):
    for value in (
        fit_result.ionization_rate,
        fit_result.cutoff_speed,
    ):
        tc.assertTrue(np.isnan(value.nominal_value))
        tc.assertTrue(np.isnan(value.std_dev))


class PickupIonFitInputDataTest(unittest.TestCase):
    """Tests for the shape and dtype guards on `PickupIonFitInputData`."""

    def test_wrong_input_shapes_are_rejected(self):
        """Reject an input that does not arrive on the coarse-sweep layout the fit assumes.

        A mis-shaped input raises at construction rather than being silently
        reshaped.
        """
        coarse_sweep = np.zeros((_N_SWEEPS, _N_COARSE_BINS))
        coarse_sweep_vectors = np.zeros((_N_SWEEPS, _N_COARSE_BINS, 3))
        cases = [
            (
                "esa_energies given per sweep instead of averaged over the chunk",
                {"esa_energies": coarse_sweep},
            ),
            (
                "esa_energies given on the full sweep instead of the coarse sweep",
                {"esa_energies": np.zeros(_SWEEP_LEN)},
            ),
            (
                "count rates given on the full sweep instead of the coarse sweep",
                {"coincidence_count_rates": np.zeros((_N_SWEEPS, _SWEEP_LEN))},
            ),
            (
                "count rates holding a number of sweeps other than the 50 in a window",
                {"coincidence_count_rates": np.zeros((_N_SWEEPS - 1, _N_COARSE_BINS))},
            ),
            (
                "bulk velocities given on the full sweep instead of the coarse sweep",
                {"bulk_sw_per_bin_swapi_kms": np.zeros((_N_SWEEPS, _SWEEP_LEN, 3))},
            ),
            (
                "bulk velocities missing their component axis",
                {"bulk_sw_per_bin_swapi_kms": coarse_sweep},
            ),
            (
                "a solar wind velocity that is not a single RTN vector",
                {"solar_wind_velocity_rtn_sun": np.zeros(2)},
            ),
        ]

        for label, override in cases:
            with self.subTest(case=label):
                fields = {
                    "time_as_tt2000": _CHUNK_CENTER_TT2000,
                    "esa_energies": np.zeros(_N_COARSE_BINS),
                    "coincidence_count_rates": coarse_sweep,
                    "bulk_sw_per_bin_swapi_kms": coarse_sweep_vectors,
                    "solar_wind_velocity_rtn_sun": _SOLAR_WIND_VELOCITY_RTN_SUN_KMS,
                    "distance": _DISTANCE_KM,
                    "inflow_angle": _INFLOW_ANGLE_DEG,
                }
                fields.update(override)
                with self.assertRaises(ValueError):
                    PickupIonFitInputData(**fields)


class CalculatePickupIonValuesFillTest(unittest.TestCase):
    """Tests for fill value logic in `calculate_pickup_ion_values`."""

    def test_non_finite_covariance_fills_all_params_with_bad_fit(self):
        """When the HC3 covariance is not finite the uncertainty estimate is
        unusable, so `BAD_FIT` is set, every parameter is NaN ± NaN, and the
        goodness-of-fit check never runs."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            log_parameter_variances=(np.nan, np.nan),
        )

        self.assertEqual(
            int(run.fit_result.flags), int(SwapiL3Flags.BAD_FIT)
        )
        _assert_all_nan_params(self, run.fit_result)
        run.is_good_fit_mock.assert_not_called()

    def test_unconverged_optimization_fills_all_params_with_bad_fit(self):
        """An optimizer that reports failure yields `BAD_FIT` and NaN ± NaN
        parameters, even though the covariance itself came back finite."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            fit_succeeded=False,
        )

        self.assertEqual(
            int(run.fit_result.flags), int(SwapiL3Flags.BAD_FIT)
        )
        _assert_all_nan_params(self, run.fit_result)
        run.is_good_fit_mock.assert_not_called()

    def test_rejected_goodness_of_fit_fills_all_params_with_bad_fit(self):
        """When the goodness-of-fit check rejects a converged fit, `BAD_FIT` is
        set and every parameter is reported as NaN ± NaN rather than retained."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            fit_is_good=False,
        )

        self.assertEqual(
            int(run.fit_result.flags), int(SwapiL3Flags.BAD_FIT)
        )
        _assert_all_nan_params(self, run.fit_result)

    def test_coarse_quantities_are_used_as_given(self):
        """The fit works on the coarse-sweep arrays it is handed.

        The energies become the response's voltage axis, while count rates and
        bulk velocities keep their per-sweep resolution.
        """
        count_rates_per_step = np.arange(_N_COARSE_BINS, dtype=float)

        bulk_sw_per_bin_swapi_kms = np.broadcast_to(
            np.arange(_N_COARSE_BINS, dtype=float)[np.newaxis, :, np.newaxis],
            (_N_SWEEPS, _N_COARSE_BINS, 3),
        )

        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=count_rates_per_step,
            bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
        )

        response_args = run.build_collapsed_response_mock.call_args.kwargs
        np.testing.assert_allclose(
            response_args["voltages_v"], _ENERGY_PER_COARSE_STEP / SWAPI_L2_K_FACTOR
        )
        np.testing.assert_allclose(
            response_args["bulk_sw_per_bin_kms"], bulk_sw_per_bin_swapi_kms
        )
        np.testing.assert_allclose(
            run.is_good_fit_mock.call_args.kwargs["observed_rates"],
            np.tile(count_rates_per_step, (_N_SWEEPS, 1)),
        )

    def test_response_is_built_at_the_chunk_center_time(self):
        """The instrument response is evaluated at the chunk's center time, so
        it picks up the He+ efficiency in force at that time."""
        run = _run_calculate_with_mocked_fit(nominal=_good_nominal())

        self.assertEqual(
            run.build_collapsed_response_mock.call_args.kwargs["time_as_tt2000"],
            _CHUNK_CENTER_TT2000,
        )

    def test_is_good_fit_receives_correct_input(
        self,
    ):
        """Ensure is_good_fit receives the correct input."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(ionization_rate=1e-7, cutoff_speed=450.0),
            observed_per_step=2.0,
        )

        modeled_params = run.calculate_coincidence_rate_mock.call_args.kwargs
        np.testing.assert_allclose(modeled_params["ionization_rate"], 1e-7)
        np.testing.assert_allclose(modeled_params["cutoff_speed"], 450.0)

        goodness_of_fit_args = run.is_good_fit_mock.call_args.kwargs
        np.testing.assert_array_equal(
            goodness_of_fit_args["model_rates"], _MODELED_RATES
        )
        np.testing.assert_allclose(goodness_of_fit_args["cutoff_speed_kms"], 450.0)
        np.testing.assert_allclose(goodness_of_fit_args["ionization_rate"], 1e-7)
        np.testing.assert_allclose(
            goodness_of_fit_args["sw_speed_kms"], _SOLAR_WIND_SPEED_INERTIAL_KMS
        )
        np.testing.assert_allclose(
            goodness_of_fit_args["observed_rates"], np.full((_N_SWEEPS, _N_COARSE_BINS), 2.0)
        )
        np.testing.assert_allclose(
            goodness_of_fit_args["esa_energies"], _ENERGY_PER_COARSE_STEP
        )


    def test_bad_fit_skips_the_moment_integrals_and_fills_density_and_temperature(self):
        """A `BAD_FIT` leaves the parameters NaN, so the moments are never
        integrated (a NaN cutoff speed would otherwise crash them) and density
        and temperature come back as NaN fill."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            fit_is_good=False,
        )

        run.calculate_density_mock.assert_not_called()
        run.calculate_temperature_mock.assert_not_called()
        self.assertTrue(np.isnan(run.fit_result.density.nominal_value))
        self.assertTrue(np.isnan(run.fit_result.temperature.nominal_value))

    def test_accepted_fit_reports_the_moment_integrals(self):
        """On an accepted fit the density and temperature reported are the
        moment integrals evaluated at the fitted parameters."""
        run = _run_calculate_with_mocked_fit(nominal=_good_nominal())

        self.assertIs(run.fit_result.density, run.calculate_density_mock.return_value)
        self.assertIs(
            run.fit_result.temperature, run.calculate_temperature_mock.return_value
        )

    def test_solar_wind_speed_in_the_sun_frame_is_the_rtn_vector_sum(self):
        """The model is defined against the Sun-frame solar wind speed, which is
        the norm of the spacecraft-frame velocity plus IMAP's own velocity —
        both already in IMAP_RTN, so no frame transform is needed."""
        run = _run_calculate_with_mocked_fit(nominal=_good_nominal())

        for call_kwargs in (
            run.calculate_coincidence_rate_mock.call_args.kwargs,
            run.calculate_density_mock.call_args.kwargs,
        ):
            np.testing.assert_allclose(
                call_kwargs["solar_wind_speed_inertial_frame"],
                _SOLAR_WIND_SPEED_INERTIAL_KMS,
            )

    def test_accepted_fit_returns_all_finite_params_with_no_flag(self):
        """A successful fit is all green."""
        run = _run_calculate_with_mocked_fit(nominal=_good_nominal())
        fit_result = run.fit_result

        self.assertEqual(int(fit_result.flags), int(SwapiL3Flags.NONE))
        for value in (
            fit_result.ionization_rate,
            fit_result.cutoff_speed,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))

    def test_accepted_fit_reports_parameters_exponentiated_out_of_log_space(self):
        """The optimizer works in log space, so the reported nominal values are
        the exponentiated solution."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(ionization_rate=3e-7, cutoff_speed=475.0)
        )
        fit_result = run.fit_result

        self.assertAlmostEqual(fit_result.ionization_rate.nominal_value, 3e-7)
        self.assertAlmostEqual(fit_result.cutoff_speed.nominal_value, 475.0)

    def test_accepted_fit_scales_log_space_sigmas_by_the_delta_method(self):
        """A log-space standard deviation is converted to the parameter itself
        by the delta method, σ(x) = x σ(ln x)."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(ionization_rate=2e-7, cutoff_speed=500.0),
            log_parameter_variances=(0.04, 0.01),
        )
        fit_result = run.fit_result

        self.assertAlmostEqual(fit_result.ionization_rate.std_dev, 2e-7 * 0.2)
        self.assertAlmostEqual(fit_result.cutoff_speed.std_dev, 500.0 * 0.1)


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
