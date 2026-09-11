import unittest
from unittest.mock import patch

import numpy as np
import scipy.optimize

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.optimize_solar_wind_proton_params import (
    OptimizeSolarWindProtonParamsResult,
    optimize_solar_wind_proton_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    N_STATE,
    SolarWindParams,
)
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import (
    NOMINAL_TEST_EPOCH_TT2000,
    NOMINAL_SWAPI_TO_RTN_ROTATION,
    REALISTIC_ESA_VOLTAGES,
    load_swapi_response,
    proton_params,
    synthesize_count_rates,
)


def _build_proton_fit_context(count_rate: np.ndarray):
    response = load_swapi_response(warm_cache_voltages=REALISTIC_ESA_VOLTAGES)
    n_sweeps = len(REALISTIC_ESA_VOLTAGES)
    rotation_matrices = np.broadcast_to(
        NOMINAL_SWAPI_TO_RTN_ROTATION, (n_sweeps, 3, 3)
    ).copy()
    return build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=REALISTIC_ESA_VOLTAGES,
        swapi_response=response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=rotation_matrices,
    )


def _synthetic_count_rate_for(sw_params: SolarWindParams) -> np.ndarray:
    """Forward-model deadtime-applied count rates from `sw_params` against
    the fixture voltages. Used to seed both the "recovers known params" and
    cache-counting tests with a realistic measurement vector.

    Two-step: build a placeholder context with zero counts, run the forward
    model to get ideal rates, then apply the deadtime factor exactly the way
    `_Evaluator._eval` does (so the residual at the truth is zero)."""
    ctx = _build_proton_fit_context(np.zeros_like(REALISTIC_ESA_VOLTAGES))
    return synthesize_count_rates(ctx, sw_params)


class TestOptimizeSolarWindProtonParamsResultMSE(unittest.TestCase):
    """Tests for `OptimizeSolarWindProtonParamsResult.mse`."""

    def test_mse_is_mean_of_squared_residuals(self):
        """Given a residuals vector [1, -2, 3], mse returns 14/3 — the per-bin mean of squared residuals."""
        result = OptimizeSolarWindProtonParamsResult(
            sw_params=proton_params(),
            residuals=np.array([1.0, -2.0, 3.0]),
            jacobian=np.zeros((3, N_STATE)),
            success=True,
        )
        self.assertAlmostEqual(result.mse, 14.0 / 3.0)

    def test_mse_handles_all_zero_residuals_without_dividing_by_zero(self):
        """At the truth (noise-free fit) all residuals are zero and mse reports exactly 0.0 rather than NaN/Inf, so the wrong-basin chi^2 comparator can be applied unconditionally."""
        result = OptimizeSolarWindProtonParamsResult(
            sw_params=proton_params(),
            residuals=np.zeros(5),
            jacobian=np.zeros((5, N_STATE)),
            success=True,
        )
        self.assertEqual(result.mse, 0.0)


class TestOptimizeSolarWindParamsRecoversTruth(unittest.TestCase):
    """End-to-end tests for `optimize_solar_wind_proton_params` driving an LM fit to a known synthetic solar wind state."""

    @classmethod
    def setUpClass(cls):
        cls.true_params = proton_params()
        cls.count_rate = _synthetic_count_rate_for(cls.true_params)
        cls.ctx = _build_proton_fit_context(count_rate=cls.count_rate)
        # Initial guess perturbed in density (+10%), speed (+3%), and
        # temperature (+20%). Sized so LM converges in one basin without
        # invoking the wrong-basin flip; the basin-of-attraction bounds live
        # in docs/swapi/solar-wind-moments.md § Wrong-basin detection.
        cls.initial_guess = proton_params(
            density=cls.true_params.density * 1.1,
            velocity_rtn=cls.true_params.velocity_rtn * 1.03,
            temperature=cls.true_params.temperature * 1.2,
        )
        cls.result = optimize_solar_wind_proton_params(cls.initial_guess, cls.ctx)

    def test_optimizer_recovers_density(self):
        """Starting from a +10% density perturbation against noise-free synthetic data, the fitted density returns to the truth within 0.1%."""
        np.testing.assert_allclose(
            self.result.sw_params.density,
            self.true_params.density,
            rtol=1e-3,
        )

    def test_optimizer_recovers_temperature(self):
        """Starting from a +20% temperature perturbation against noise-free synthetic data, the fitted temperature returns to the truth within 0.1%."""
        np.testing.assert_allclose(
            self.result.sw_params.temperature,
            self.true_params.temperature,
            rtol=1e-3,
        )

    def test_optimizer_recovers_velocity(self):
        """Starting from a +3% speed perturbation against noise-free synthetic data, the fitted bulk velocity returns to the truth within 0.5 km/s on each RTN component."""
        np.testing.assert_allclose(
            self.result.sw_params.velocity_rtn,
            self.true_params.velocity_rtn,
            atol=0.5,
        )

    def test_optimizer_reports_success(self):
        """LM converges cleanly on the well-posed synthetic problem and the result's success flag is True."""
        self.assertTrue(self.result.success)

    def test_residuals_at_solution_are_small(self):
        """At the converged solution against noise-free data, mse is below (peak_rate)^2 * 1e-4 — residuals are limited only by LM's xtol=1e-4 tolerance."""
        peak_rate_squared = float(np.max(self.count_rate)) ** 2
        self.assertLess(self.result.mse, peak_rate_squared * 1e-4)


class TestOptimizeSolarWindProtonParamsResultShape(unittest.TestCase):
    """Tests for `optimize_solar_wind_proton_params` result-object shape and type contracts the wrong-basin detector and uncertainty derivation depend on."""

    @classmethod
    def setUpClass(cls):
        true_params = proton_params()
        count_rate = _synthetic_count_rate_for(true_params)
        cls.ctx = _build_proton_fit_context(count_rate=count_rate)
        cls.result = optimize_solar_wind_proton_params(true_params, cls.ctx)

    def test_sw_params_is_a_solar_wind_params(self):
        """The returned `sw_params` field is a `SolarWindParams` instance, not a raw state vector."""
        self.assertIsInstance(self.result.sw_params, SolarWindParams)

    def test_residuals_length_matches_ctx_count_rate(self):
        """The residuals array has one entry per observed count-rate bin in the fit context."""
        self.assertEqual(self.result.residuals.shape, self.ctx.count_rate.shape)

    def test_jacobian_shape_is_n_residuals_by_n_state(self):
        """The jacobian has shape (n_residuals, N_STATE) — one row per residual, one column per fit parameter."""
        self.assertEqual(
            self.result.jacobian.shape, (self.ctx.count_rate.size, N_STATE)
        )

    def test_success_is_bool(self):
        """The success field is a plain Python bool rather than numpy's bool-like wrapper."""
        self.assertIsInstance(self.result.success, bool)

    def test_sw_params_carries_context_mass(self):
        """The fitted `SolarWindParams.mass` is propagated from `ctx.mass_kg` rather than defaulting to proton mass."""
        self.assertEqual(self.result.sw_params.mass, self.ctx.mass_kg)


class TestOptimizerLeastSquaresKwargs(unittest.TestCase):
    """Tests that `optimize_solar_wind_proton_params` invokes scipy with the doc-pinned LM kwargs."""

    def test_uses_lm_with_xtol_from_doc_spec(self):
        """When the optimizer runs, scipy.optimize.least_squares is called with `method='lm'` and `xtol=1e-4` per the doc spec."""
        true_params = proton_params()
        count_rate = _synthetic_count_rate_for(true_params)
        ctx = _build_proton_fit_context(count_rate=count_rate)

        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton.optimize_solar_wind_proton_params.scipy.optimize.least_squares"
        ) as mock_least_squares:
            mock_least_squares.return_value = scipy.optimize.OptimizeResult(
                x=true_params.to_vector(),
                fun=np.zeros_like(count_rate),
                jac=np.zeros((count_rate.size, N_STATE)),
                success=True,
            )

            optimize_solar_wind_proton_params(true_params, ctx)

            kwargs = mock_least_squares.call_args.kwargs
            self.assertEqual(kwargs["method"], "lm")
            self.assertEqual(kwargs["xtol"], 1e-4)




if __name__ == "__main__":
    unittest.main()
