import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.optimize_solar_wind_proton_params import (
    OptimizeSolarWindProtonParamsResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    N_STATE,
    SolarWindParams,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi._helpers import load_swapi_response, synthesize_count_rates

# Mean SWAPI L2 coarse-sweep voltages (V), descending — a 62-bin sweep that
# covers the proton speed range densely. Identical to the set used by
# `docs/swapi/figure_src/plot_fit_accuracy.py`.
_VOLTAGES_PER_SWEEP = np.array(
    [
        9895.52, 9088.69, 8348.80, 7667.55, 7042.16, 6469.31, 5941.77, 5457.31,
        5013.22, 4603.65, 4230.77, 3886.92, 3569.16, 3278.72, 3011.13, 2766.25,
        2539.54, 2333.83, 2144.24, 1969.31, 1808.74, 1660.86, 1525.75, 1401.82,
        1287.58, 1182.24, 1085.15, 995.55, 914.31, 839.94, 771.70, 709.46,
        651.59, 598.47, 549.91, 505.12, 463.89, 425.92, 391.18, 359.35, 329.94,
        303.02, 278.25, 255.55, 234.77, 215.61, 197.95, 181.82, 167.04, 153.46,
        140.91, 129.50, 118.91, 109.20, 100.30, 92.11, 84.61, 77.73, 71.40,
        65.59, 60.23, 55.34,
    ]
)
_N_BINS_PER_SWEEP = len(_VOLTAGES_PER_SWEEP)
_N_SWEEPS = 5
_SWEEP_DURATION_S = 12.0
_SAMPLE_TIME_PER_BIN_S = _SWEEP_DURATION_S / 72
# Typical IMAP spin period; matches `docs/swapi/figure_src/plot_fit_accuracy.py`.
_SPIN_PERIOD_S = 15.13

# Anchor SWAPI→RTN rotation matrix near 2026-01-01, lifted from
# `docs/swapi/figure_src/plot_fit_accuracy.py`. The literal block is in
# RTN→SWAPI orientation; transposing converts to SWAPI→RTN. Per-bin matrices
# are produced by spinning this anchor about its own +Y column (the spin axis
# in RTN) at the SWAPI spin period.
_ANCHOR_ROTATION_MATRIX = np.array(
    [
        [+0.0705, +0.9157, +0.3955],
        [-0.9968, +0.0792, -0.0057],
        [-0.0365, -0.3939, +0.9184],
    ]
).T
_ANCHOR_TIME_S = 0.5 * _SWEEP_DURATION_S
# Negative sign chosen so R(t) = anchor @ Rot(δφ, spin_axis_RTN) reproduces
# independent SPICE-derived sweep midpoints over a 5-sweep cycle (see
# `docs/swapi/figure_src/plot_fit_accuracy.py`).
_SPIN_OMEGA_RAD_S = -2.0 * np.pi / _SPIN_PERIOD_S

# The +Y column of `_ANCHOR_ROTATION_MATRIX` is the SWAPI spin axis expressed
# in RTN. For a proper rotation matrix it must be unit-norm; assert that here
# so the bulk-velocity construction below can use it directly.
assert np.isclose(
    np.linalg.norm(_ANCHOR_ROTATION_MATRIX[:, 1]), 1.0, atol=1e-3
), "expected +Y column of anchor rotation to be unit-norm"

# Ground-truth solar-wind parameters used for the parameter-recovery test.
# Moderate-speed slow-stream proton population — well inside the SWAPI energy
# range, well within the SG passband elevation, and warm enough that the LM
# fit is not numerically marginal.
#
# The bulk velocity is anti-parallel to the synthetic spin axis (the +Y column
# of `_ANCHOR_ROTATION_MATRIX`, which lies near -R̂_RTN). This puts the wind
# *into* the SWAPI aperture; using arbitrary RTN-frame velocity components
# would point the wind at the back of the instrument and zero the synthetic
# count rates.
_TRUE_DENSITY_CM3 = 5.0
_TRUE_TEMPERATURE_K = 1.0e5
_TRUE_BULK_SPEED_KM_S = 450.0
_TRUE_VELOCITY_RTN_KM_S = (
    -_TRUE_BULK_SPEED_KM_S * _ANCHOR_ROTATION_MATRIX[:, 1]
)


def _per_bin_rotation_matrices() -> np.ndarray:
    """Synthesize plausible per-bin SWAPI→RTN matrices for `_N_SWEEPS` sweeps;
    details don't affect what's being tested."""
    sweep_index = np.repeat(
        np.arange(_N_SWEEPS), _N_BINS_PER_SWEEP
    )
    bin_index_in_sweep = np.tile(
        np.arange(1, _N_BINS_PER_SWEEP + 1), _N_SWEEPS
    )
    sample_times_s = (
        sweep_index * _SWEEP_DURATION_S
        + bin_index_in_sweep * _SAMPLE_TIME_PER_BIN_S
    )

    spin_axis = _ANCHOR_ROTATION_MATRIX[:, 1] / np.linalg.norm(
        _ANCHOR_ROTATION_MATRIX[:, 1]
    )
    delta_phi = _SPIN_OMEGA_RAD_S * (sample_times_s - _ANCHOR_TIME_S)

    ax, ay, az = spin_axis
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return rot @ _ANCHOR_ROTATION_MATRIX


def _build_context(count_rate, esa_voltage, swapi_response, rotation_matrices):
    return build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )


def _build_synthetic_fit_context(truth_params: SolarWindParams):
    """Build a SwapiResponse, per-bin rotation matrices, and a populated
    `SolarWindFitContext` whose count rates are forward-modelled from
    `truth_params`. Returns `(swapi_response, rotation_matrices, fit_ctx)`."""
    all_voltages_2d = np.tile(_VOLTAGES_PER_SWEEP, (_N_SWEEPS, 1))
    swapi_response = load_swapi_response(
        warm_cache_voltages=all_voltages_2d.ravel()
    )
    rotation_matrices = _per_bin_rotation_matrices()

    # Build a context with placeholder rates first, then forward-model the
    # synthetic rates at the truth, then rebuild the context with those rates.
    # Two-step pattern keeps the rotation-matrix / response-grid wiring
    # identical between forward-model and fit.
    placeholder_ctx = _build_context(
        count_rate=np.ones_like(all_voltages_2d),
        esa_voltage=all_voltages_2d,
        swapi_response=swapi_response,
        rotation_matrices=rotation_matrices,
    )
    synthesized_rates = synthesize_count_rates(placeholder_ctx, truth_params)
    fit_ctx = _build_context(
        count_rate=synthesized_rates.reshape(all_voltages_2d.shape),
        esa_voltage=all_voltages_2d,
        swapi_response=swapi_response,
        rotation_matrices=rotation_matrices,
    )
    return swapi_response, rotation_matrices, fit_ctx


def _truth_params() -> SolarWindParams:
    return SolarWindParams(
        density=_TRUE_DENSITY_CM3,
        velocity_rtn=_TRUE_VELOCITY_RTN_KM_S.copy(),
        temperature=_TRUE_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )


class _ProtonFitFixture(unittest.TestCase):
    """Loads the SwapiResponse and a populated `SolarWindFitContext` with
    forward-modelled count rates for the moderate-speed truth case, then runs
    the fitter once and stores `cls.result`."""

    @classmethod
    def setUpClass(cls):
        cls.truth_params = _truth_params()
        (
            cls.swapi_response,
            cls.rotation_matrices,
            cls.fit_ctx,
        ) = _build_synthetic_fit_context(truth_params=cls.truth_params)
        cls.result = fit_solar_wind_proton_model(cls.fit_ctx)


class TestFitSolarWindProtonModelEndToEnd(_ProtonFitFixture):
    """Tests for `fit_solar_wind_proton_model`; noise-free parameter-recovery sanity check against truth."""

    def test_recovers_density(self):
        """Fitting noise-free synthesized rates recovers the truth density within 1%."""
        self.assertAlmostEqual(
            self.result.density.nominal_value,
            _TRUE_DENSITY_CM3,
            delta=0.01 * _TRUE_DENSITY_CM3,
        )

    def test_recovers_temperature(self):
        """Fitting noise-free synthesized rates recovers the truth temperature within 1%."""
        self.assertAlmostEqual(
            self.result.temperature.nominal_value,
            _TRUE_TEMPERATURE_K,
            delta=0.01 * _TRUE_TEMPERATURE_K,
        )

    def test_recovers_velocity_components(self):
        """Fitting noise-free synthesized rates recovers all three RTN bulk-velocity components within 1 km/s."""
        nominal = self.result.velocity_rtn_nominal()
        np.testing.assert_allclose(
            nominal, _TRUE_VELOCITY_RTN_KM_S, atol=1.0
        )

    def test_quality_flag_is_none_on_successful_convergence(self):
        """A clean LM convergence on noise-free data leaves the bad-fit flag cleared (`SwapiL3Flags.NONE`)."""
        self.assertEqual(self.result.quality_flag, SwapiL3Flags.NONE)


class TestFitSolarWindProtonModelUncertainties(_ProtonFitFixture):
    """Tests for `fit_solar_wind_proton_model`; exercises the internal HC3 sandwich uncertainty path via the public fit result."""

    def test_density_std_dev_is_finite_and_positive(self):
        """A successful fit emits a finite, strictly positive density std_dev from the sandwich estimator."""
        self.assertTrue(np.isfinite(self.result.density.std_dev))
        self.assertGreater(self.result.density.std_dev, 0.0)

    def test_temperature_std_dev_is_finite_and_positive(self):
        """A successful fit emits a finite, strictly positive temperature std_dev from the sandwich estimator."""
        self.assertTrue(np.isfinite(self.result.temperature.std_dev))
        self.assertGreater(self.result.temperature.std_dev, 0.0)

    def test_velocity_rtn_components_have_finite_positive_std_devs(self):
        """Each RTN bulk-velocity component carries a finite, strictly positive std_dev from the sandwich estimator."""
        for component in self.result.velocity_rtn:
            self.assertTrue(np.isfinite(component.std_dev))
            self.assertGreater(component.std_dev, 0.0)


class TestProtonSolarWindFitResultPublicAPI(_ProtonFitFixture):
    """Tests for `ProtonSolarWindFitResult.velocity_rtn_nominal` and `velocity_rtn_covariance` accessors."""

    def test_velocity_rtn_nominal_is_three_component_vector(self):
        """The nominal-velocity accessor returns a length-3 RTN vector."""
        nominal = self.result.velocity_rtn_nominal()
        self.assertEqual(nominal.shape, (3,))

    def test_velocity_rtn_nominal_matches_per_component_nominal_values(
        self,
    ):
        """The nominal-velocity vector matches the per-component `.nominal_value` of the stored UFloat triple."""
        per_component = np.array(
            [v.nominal_value for v in self.result.velocity_rtn]
        )
        np.testing.assert_array_equal(
            self.result.velocity_rtn_nominal(), per_component
        )

    def test_velocity_rtn_covariance_is_three_by_three(self):
        """The covariance accessor returns a 3x3 matrix matching the RTN component count."""
        covariance = self.result.velocity_rtn_covariance()
        self.assertEqual(covariance.shape, (3, 3))

    def test_velocity_rtn_covariance_is_symmetric(self):
        """The returned velocity covariance matrix is symmetric to numerical tolerance."""
        covariance = self.result.velocity_rtn_covariance()
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)

    def test_velocity_rtn_covariance_is_positive_semidefinite(self):
        """The returned velocity covariance matrix is positive semidefinite (all eigenvalues non-negative)."""
        covariance = self.result.velocity_rtn_covariance()
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertGreaterEqual(eigenvalues.min(), -1e-9)

    def test_covariance_diagonal_matches_per_component_variance(self):
        """Each diagonal entry of the velocity covariance equals the square of the corresponding UFloat `std_dev`."""
        covariance = self.result.velocity_rtn_covariance()
        for i, ufloat_value in enumerate(self.result.velocity_rtn):
            self.assertAlmostEqual(
                covariance[i, i], ufloat_value.std_dev ** 2, places=10
            )


class TestQualityFlagBranches(unittest.TestCase):
    """Tests for `fit_solar_wind_proton_model`; quality-flag branches: optimizer failure → `FIT_ERROR` (NaN moments), high fitted temperature → `BAD_FIT` (moments kept)."""

    @classmethod
    def setUpClass(cls):
        _, _, cls.fit_ctx = _build_synthetic_fit_context(
            truth_params=_truth_params()
        )

    def _patch_optimizer_with_result(self, optimize_result):
        return patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model.optimize_solar_wind_proton_params",
            return_value=optimize_result,
        )

    def test_fit_error_flag_when_optimizer_reports_failure(self):
        """When the underlying optimizer returns `success=False`, the result's bad-fit flag is `SwapiL3Flags.FIT_ERROR` and the moments are NaN-filled."""
        failed_result = OptimizeSolarWindProtonParamsResult(
            sw_params=SolarWindParams(
                density=1.0,
                velocity_rtn=np.array([-1.0, 0.0, 0.0]),
                temperature=1.0,
                mass=PROTON_MASS_KG,
            ),
            residuals=np.zeros(_N_BINS_PER_SWEEP * _N_SWEEPS),
            jacobian=np.zeros(
                (_N_BINS_PER_SWEEP * _N_SWEEPS, N_STATE)
            ),
            success=False,
        )
        with self._patch_optimizer_with_result(failed_result):
            result = fit_solar_wind_proton_model(self.fit_ctx)
        self.assertEqual(result.quality_flag, SwapiL3Flags.FIT_ERROR)
        self.assertTrue(np.isnan(result.density.nominal_value))
        self.assertTrue(np.isnan(result.temperature.nominal_value))

    def test_quality_flag_when_temperature_above_threshold(self):
        """A converged fit whose temperature exceeds 5e5 K is flagged `BAD_FIT` and its moments are NaN-filled, distinguishing it from a clean fit but matching `FIT_ERROR`'s fill-value contract."""
        too_hot_temperature = 6.0e5
        too_hot_result = OptimizeSolarWindProtonParamsResult(
            sw_params=SolarWindParams(
                density=_TRUE_DENSITY_CM3,
                velocity_rtn=_TRUE_VELOCITY_RTN_KM_S.copy(),
                temperature=too_hot_temperature,
                mass=PROTON_MASS_KG,
            ),
            residuals=np.zeros(_N_BINS_PER_SWEEP * _N_SWEEPS),
            jacobian=np.zeros(
                (_N_BINS_PER_SWEEP * _N_SWEEPS, N_STATE)
            ),
            success=True,
        )
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model.escape_local_minimum",
            return_value=too_hot_result,
        ):
            result = fit_solar_wind_proton_model(self.fit_ctx)
        self.assertEqual(result.quality_flag, int(SwapiL3Flags.BAD_FIT))
        self.assertTrue(np.isnan(result.density.nominal_value))
        self.assertTrue(np.isnan(result.temperature.nominal_value))
        for component in result.velocity_rtn:
            self.assertTrue(np.isnan(component.nominal_value))


class TestPipelineOrder(unittest.TestCase):
    """Tests for `fit_solar_wind_proton_model`; verifies basin-hopping output supersedes the LM-1 result."""

    @classmethod
    def setUpClass(cls):
        _, _, cls.fit_ctx = _build_synthetic_fit_context(
            truth_params=_truth_params()
        )

    def test_construct_fit_result_uses_post_escape_local_minimum_result(self):
        """When `escape_local_minimum` returns a result distinct from LM-1, the final fit result carries the post-escape parameters."""
        # Synthetic post-escape density chosen well above _TRUE_DENSITY_CM3=5.0
        # so the round-tripped density is unambiguously the patched value.
        # Velocity is far from the truth bulk velocity for the same reason.
        post_escape_density = 50.0
        post_escape_velocity = np.array([-321.0, 0.0, 0.0])
        post_escape_temperature = 2.5e5

        post_escape_result = OptimizeSolarWindProtonParamsResult(
            sw_params=SolarWindParams(
                density=post_escape_density,
                velocity_rtn=post_escape_velocity,
                temperature=post_escape_temperature,
                mass=PROTON_MASS_KG,
            ),
            residuals=np.zeros(_N_BINS_PER_SWEEP * _N_SWEEPS),
            jacobian=np.zeros(
                (_N_BINS_PER_SWEEP * _N_SWEEPS, N_STATE)
            ),
            success=True,
        )

        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model.escape_local_minimum",
            return_value=post_escape_result,
        ):
            result = fit_solar_wind_proton_model(self.fit_ctx)

        # density alone is decisive: it, temperature, and velocity_rtn
        # propagate together from the same OptimizeSolarWindProtonParamsResult.
        self.assertAlmostEqual(
            result.density.nominal_value, post_escape_density
        )


if __name__ == "__main__":
    unittest.main()
