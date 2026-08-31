import unittest
from unittest.mock import MagicMock, patch

import numba
import numpy as np
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha import (
    fit_solar_wind_alpha_model as alpha_module,
    calculate_initial_guess as alpha_initial_guess_module,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    AlphaSolarWindFitResult,
    _AlphaEvaluator,
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.swapi._helpers import load_swapi_response


# ----- module-level fixture constants --------------------------------------

# 5-sweep coarse-only voltage axis: 62 bins/sweep × 5 sweeps = 310 measurements,
# matching the Stage-2 axis described in the doc. The voltage values themselves
# are arbitrary log-spaced decreasing — the alpha peak finder requires
# strictly decreasing energies, but the exact endpoints are not load-bearing
# here. The L2 voltage range divided by `SWAPI_L2_K_FACTOR` would give the
# physical voltages SWAPI sweeps in flight.
_N_BINS_PER_SWEEP = 62
_N_SWEEPS = 5
_ONE_SWEEP_VOLTAGE = np.logspace(
    np.log10(3500.0), np.log10(140.0), _N_BINS_PER_SWEEP
)
_FIVE_SWEEP_VOLTAGE = np.broadcast_to(_ONE_SWEEP_VOLTAGE, (_N_SWEEPS, _N_BINS_PER_SWEEP)).copy()
_N_MEAS = _FIVE_SWEEP_VOLTAGE.size

# Slow-wind ground-truth moments. RTN +R points sunward, so a sunward solar
# wind has v_R = -450 km/s. B̂ is taken along -R̂ so a positive Δv pushes
# alphas toward more-negative v_R (along +B̂).
_TRUE_PROTON_DENSITY_CM3 = 5.0
_TRUE_PROTON_TEMPERATURE_K = 1.0e5
_TRUE_PROTON_VELOCITY_RTN = np.array([-450.0, 0.0, 0.0])
_TRUE_ALPHA_DENSITY_CM3 = 0.2
_TRUE_ALPHA_TEMPERATURE_K = 4.0e5
_TRUE_DELTA_V_KM_S = 30.0
_B_HAT_RTN = np.array([-1.0, 0.0, 0.0])

# Stage-1 proton uncertainties used to seed `ProtonSolarWindFitResult`.
# Values are chosen small but nonzero so `velocity_rtn_covariance`
# (used downstream by `fit_solar_wind_alpha_model` to add
# `σ_Δv²·B̂B̂ᵀ`) is well-defined. The exact values are not pinned by any
# test — they only need to be finite and positive.
_STAGE1_PROTON_DENSITY_SIGMA_CM3 = 0.05
_STAGE1_PROTON_TEMPERATURE_SIGMA_K = 1.0e3
_STAGE1_PROTON_VELOCITY_SIGMA_KM_S = 1.0

# Predicted ESA voltage of the alpha bump on the fixture spectrum. SWAPI's
# central-speed conversion is `v² = 2·k·V·(e/m_p) / (m/q in m_p/e)`, so
# inverting for V at the alpha truth speed (|v_p| + Δv along -R̂, ≈ 480 km/s)
# gives V_α ≈ 1264 V on this fixture.
_ALPHA_TRUTH_SPEED_M_S = (
    float(np.linalg.norm(_TRUE_PROTON_VELOCITY_RTN)) + _TRUE_DELTA_V_KM_S
) * 1.0e3
_ALPHA_PEAK_VOLTAGE = (
    _ALPHA_TRUTH_SPEED_M_S**2
    * ALPHA_MASS_PER_CHARGE_M_P_PER_E
    / (2.0 * SWAPI_K_FACTOR * (PROTON_CHARGE_COULOMBS / PROTON_MASS_KG))
)


# ----- helpers --------------------------------------------------------------


def _load_swapi_response_with_warm_cache() -> SwapiResponse:
    """Load `SwapiResponse` from the shipped instrument-team CSVs and warm
    its passband cache for the fixture voltages.

    `create_response_grid` raises if the passband cache isn't warm for the
    requested voltage, so callers building a fit context off this response
    must warm it first."""
    return load_swapi_response(warm_cache_voltages=_FIVE_SWEEP_VOLTAGE)


def _identity_rotation_matrices(n: int = _N_MEAS) -> np.ndarray:
    """Identity SWAPI→RTN rotations so the instrument frame coincides with
    RTN — keeps the wind-direction interpretation straightforward.

    `np.broadcast_to` produces a read-only view; `.copy()` materialises a
    contiguous writeable array, which the JIT'd Stage-2 forward model
    expects."""
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _build_proton_and_alpha_contexts(
    *,
    response: SwapiResponse,
    count_rate: np.ndarray,
    voltage: np.ndarray = _FIVE_SWEEP_VOLTAGE,
    rotation_matrices: np.ndarray | None = None,
    proton_effective_area_scale: float = 1.0,
    alpha_effective_area_scale: float = 1.0,
) -> tuple[SolarWindFitContext, SolarWindFitContext]:
    """Build the (proton, alpha) context pair the new `fit_solar_wind_alpha_model`
    signature expects. Tests previously passed raw count_rate / response /
    eff-scale to the fitter; the fitter now requires precomputed contexts."""
    if rotation_matrices is None:
        rotation_matrices = _identity_rotation_matrices(voltage.size)
    proton_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=voltage,
        swapi_response=response,
        central_effective_area_scale=proton_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=voltage,
        swapi_response=response,
        central_effective_area_scale=alpha_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )
    return proton_ctx, alpha_ctx


def _build_proton_fit_result(
    *,
    density: float = _TRUE_PROTON_DENSITY_CM3,
    temperature: float = _TRUE_PROTON_TEMPERATURE_K,
    velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
    bad_fit_flag: int = int(SwapiL3Flags.NONE),
) -> ProtonSolarWindFitResult:
    """Build a `ProtonSolarWindFitResult` (Stage-1 output) with small but
    nonzero uncertainties so `velocity_rtn_covariance` is well-defined
    and Stage-2 has something to add `σ_Δv²·B̂B̂ᵀ` to."""
    return ProtonSolarWindFitResult(
        density=ufloat(density, _STAGE1_PROTON_DENSITY_SIGMA_CM3),
        temperature=ufloat(temperature, _STAGE1_PROTON_TEMPERATURE_SIGMA_K),
        velocity_rtn=(
            ufloat(velocity_rtn[0], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
            ufloat(velocity_rtn[1], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
            ufloat(velocity_rtn[2], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
        ),
        quality_flag=int(bad_fit_flag),
    )


def _assert_moments_are_nan_filled(test, result) -> None:
    """Assert that every moment field on an `AlphaSolarWindFitResult`
    fill-valued result is NaN. Used by the two pre-fit guard branches
    (Stage-1 failed, MAG B̂ NaN) which both short-circuit to fill values."""
    test.assertTrue(np.isnan(result.density.nominal_value))
    test.assertTrue(np.isnan(result.temperature.nominal_value))
    test.assertTrue(np.isnan(result.delta_v.nominal_value))
    for component in result.velocity_rtn:
        test.assertTrue(np.isnan(component.nominal_value))


def _synthesize_proton_plus_alpha_count_rate(
    *,
    response: SwapiResponse,
    voltage: np.ndarray,
    rotation_matrices: np.ndarray,
    proton_density: float = _TRUE_PROTON_DENSITY_CM3,
    proton_temperature: float = _TRUE_PROTON_TEMPERATURE_K,
    proton_velocity_rtn: np.ndarray = _TRUE_PROTON_VELOCITY_RTN,
    alpha_density: float = _TRUE_ALPHA_DENSITY_CM3,
    alpha_temperature: float = _TRUE_ALPHA_TEMPERATURE_K,
    delta_v: float = _TRUE_DELTA_V_KM_S,
    b_hat: np.ndarray = _B_HAT_RTN,
):
    """Forward-model deadtime-applied count rates for a (proton + alpha)
    spectrum on the fixture voltage axis. Pass `alpha_density=0.0` for a
    proton-only spectrum.

    Returns the observed count rate, the (pre-deadtime) proton-only true
    rate, and the alpha velocity v_α = v_p + Δv·B̂. The proton true rate is
    the same Stage-2 calls "frozen proton model rate" — useful for
    initial-guess setup as well."""
    proton_params = SolarWindParams(
        density=proton_density,
        velocity_rtn=np.asarray(proton_velocity_rtn, dtype=float),
        temperature=proton_temperature,
        mass=PROTON_MASS_KG,
    )
    alpha_velocity = np.asarray(proton_velocity_rtn, dtype=float) + delta_v * b_hat
    alpha_params = SolarWindParams(
        density=alpha_density,
        velocity_rtn=alpha_velocity,
        temperature=alpha_temperature,
        mass=ALPHA_PARTICLE_MASS_KG,
    )
    proton_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(voltage.shape),
        esa_voltage=voltage,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    proton_true, _ = model_solar_wind_ideal_coincidence_rates(proton_params, proton_ctx)
    if alpha_density > 0.0:
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=np.zeros(voltage.shape),
            esa_voltage=voltage,
            swapi_response=response,
            central_effective_area_scale=1.0,
            rotation_matrices=rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_true, _ = model_solar_wind_ideal_coincidence_rates(alpha_params, alpha_ctx)
        total_true = proton_true + alpha_true
    else:
        total_true = proton_true
    observed = (total_true * deadtime_factor(total_true)).reshape(voltage.shape)
    return observed, proton_true, alpha_velocity


class _SyntheticAlphaSpectrumFixture:
    """Cached forward-model fixture: builds the SwapiResponse, warm-caches
    its passband, and synthesizes a (proton + alpha) count-rate spectrum
    once for all subclass tests. Class-level so JIT compile + the pandas
    pivot inside `_build_passband_array` are paid one time."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache()
        cls.voltage = _FIVE_SWEEP_VOLTAGE
        cls.rotation_matrices = _identity_rotation_matrices()
        observed, proton_true, alpha_v = _synthesize_proton_plus_alpha_count_rate(
            response=cls.response,
            voltage=cls.voltage,
            rotation_matrices=cls.rotation_matrices,
        )
        cls.observed_count_rate = observed
        cls.proton_true_rate = proton_true
        cls.alpha_velocity_rtn = alpha_v


# ----- fit_solar_wind_alpha_model — guard conditions ---------------------


class TestFitAlphaMomentsGuardBranches(unittest.TestCase):
    """Tests for `fit_solar_wind_alpha_model` — pre-fit guard branches (proton fill values, missing/non-finite MAG B̂) must short-circuit to a NaN-filled moments result with the proton's flag propagated before any forward-model evaluation."""

    @classmethod
    def setUpClass(cls):
        # Build the (proton, alpha) ctx pair once — the guard branches
        # short-circuit before touching them, so any positive-voltage
        # response is enough.
        cls.response = _load_swapi_response_with_warm_cache()
        cls.proton_ctx, cls.alpha_ctx = _build_proton_and_alpha_contexts(
            response=cls.response,
            count_rate=np.zeros(_FIVE_SWEEP_VOLTAGE.shape),
        )

    def test_proton_fill_values_propagate_proton_flag_to_alpha(self):
        """When the Stage-1 proton fit returned NaN moments (fill values), the alpha result inherits the proton's `bad_fit_flag` unchanged — no separate alpha-side flag is added for "stage 1 failed"."""
        proton_moments = _build_proton_fit_result(
            velocity_rtn=np.array([np.nan, np.nan, np.nan]),
            bad_fit_flag=int(SwapiL3Flags.FIT_ERROR),
        )
        result = fit_solar_wind_alpha_model(
            proton_ctx=self.proton_ctx,
            alpha_ctx=self.alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
        )
        self.assertEqual(result.quality_flag, int(SwapiL3Flags.FIT_ERROR))

    def test_proton_fill_values_return_nan_filled_alpha_moments(self):
        """When the Stage-1 proton fit returned NaN moments, every alpha moment field is filled with NaN so downstream consumers can distinguish "no fit attempted" from "fit succeeded with degenerate values"."""
        proton_moments = _build_proton_fit_result(
            velocity_rtn=np.array([np.nan, np.nan, np.nan]),
            bad_fit_flag=int(SwapiL3Flags.FIT_ERROR),
        )
        result = fit_solar_wind_alpha_model(
            proton_ctx=self.proton_ctx,
            alpha_ctx=self.alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
        )
        _assert_moments_are_nan_filled(self, result)

    def test_mag_gap_propagates_proton_flag_with_no_dedicated_bit(self):
        """A NaN component in `magnetic_field_direction` is treated as an ordinary data gap: the fitter short-circuits and propagates the proton's flag unchanged with no dedicated MAG-gap bit added."""
        proton_moments = _build_proton_fit_result()
        nan_b_hat = np.array([np.nan, 0.0, 0.0])
        result = fit_solar_wind_alpha_model(
            proton_ctx=self.proton_ctx,
            alpha_ctx=self.alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=nan_b_hat,
        )
        self.assertEqual(result.quality_flag, int(SwapiL3Flags.NONE))

    def test_nan_magnetic_field_direction_returns_nan_filled_moments(self):
        """A NaN B̂ short-circuits before any forward-model call and every moment field is filled with NaN, mirroring the Stage-1-failure guard."""
        proton_moments = _build_proton_fit_result()
        nan_b_hat = np.array([np.nan, 0.0, 0.0])
        result = fit_solar_wind_alpha_model(
            proton_ctx=self.proton_ctx,
            alpha_ctx=self.alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=nan_b_hat,
        )
        _assert_moments_are_nan_filled(self, result)


# ----- fit_solar_wind_alpha_model — end-to-end recovery -----------------


class TestFitAlphaMomentsRecoversTruth(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """Tests for `fit_solar_wind_alpha_model` — end-to-end Stage-2 recovery: synthesize a proton+alpha spectrum from known truth, hand the fitter the exact proton moments, and verify it recovers (n_α, T_α, Δv) and the field-aligned alpha velocity."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.proton_moments = _build_proton_fit_result()
        proton_ctx, alpha_ctx = _build_proton_and_alpha_contexts(
            response=cls.response,
            count_rate=cls.observed_count_rate,
            voltage=cls.voltage,
            rotation_matrices=cls.rotation_matrices,
        )
        cls.result = fit_solar_wind_alpha_model(
            proton_ctx=proton_ctx,
            alpha_ctx=alpha_ctx,
            proton_moments=cls.proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
        )

    def test_fit_succeeds_with_no_quality_flags(self):
        """A clean synthetic spectrum + exact proton moments must yield `bad_fit_flag == NONE` — no LM convergence issues, no missing inputs."""
        self.assertEqual(self.result.quality_flag, int(SwapiL3Flags.NONE))

    def test_recovers_alpha_density(self):
        """The fitted alpha density matches the synthesis truth to ~1% (LM termination + deadtime rounding)."""
        np.testing.assert_allclose(
            self.result.density.nominal_value,
            _TRUE_ALPHA_DENSITY_CM3,
            rtol=1e-2,
        )

    def test_recovers_alpha_temperature(self):
        """The fitted alpha temperature matches the synthesis truth to ~1%."""
        np.testing.assert_allclose(
            self.result.temperature.nominal_value,
            _TRUE_ALPHA_TEMPERATURE_K,
            rtol=1e-2,
        )

    def test_recovers_signed_delta_v(self):
        """The fitted Δv matches the truth in magnitude and sign — LM recovers the correct basin from the Δv=0 seed."""
        np.testing.assert_allclose(
            self.result.delta_v.nominal_value, _TRUE_DELTA_V_KM_S, atol=0.5
        )

    def test_alpha_velocity_equals_proton_velocity_plus_delta_v_along_bhat(self):
        """The post-fit `velocity_rtn` satisfies the algebraic identity v_α = v_p* + Δv·B̂ exactly (not approximately) — the dataclass stores the constraint, not a free vector."""
        v_alpha = self.result.velocity_rtn_nominal()
        expected = (
            _TRUE_PROTON_VELOCITY_RTN
            + self.result.delta_v.nominal_value * _B_HAT_RTN
        )
        np.testing.assert_allclose(v_alpha, expected, atol=1e-9)

    def test_alpha_velocity_recovers_truth(self):
        """The recovered alpha velocity vector matches the synthesis truth (≈480 km/s) within ~1 km/s — combines Δv recovery + field-aligned constraint."""
        np.testing.assert_allclose(
            self.result.velocity_rtn_nominal(),
            self.alpha_velocity_rtn,
            atol=1.0,
        )


class TestFitAlphaMomentsAlphaVelocityFollowsBHat(unittest.TestCase):
    """Tests for `fit_solar_wind_alpha_model` — the field-aligned drift constraint v_α = v_p + Δv·B̂ must hold for any B̂ direction, including B̂ not parallel to -R̂."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache()
        cls.rotation_matrices = _identity_rotation_matrices()
        cls.proton_velocity_rtn = _TRUE_PROTON_VELOCITY_RTN
        # B̂ tilted away from -R̂ by `tilt_deg` in the R-T plane.
        tilt_deg = 10.0
        tilt_rad = np.deg2rad(tilt_deg)
        cls.b_hat = np.array([-np.cos(tilt_rad), np.sin(tilt_rad), 0.0])
        observed, _proton_true, alpha_v = _synthesize_proton_plus_alpha_count_rate(
            response=cls.response,
            voltage=_FIVE_SWEEP_VOLTAGE,
            rotation_matrices=cls.rotation_matrices,
            proton_velocity_rtn=cls.proton_velocity_rtn,
            delta_v=_TRUE_DELTA_V_KM_S,
            b_hat=cls.b_hat,
        )
        cls.alpha_velocity_truth = alpha_v
        cls.proton_moments = _build_proton_fit_result(
            velocity_rtn=cls.proton_velocity_rtn
        )
        proton_ctx, alpha_ctx = _build_proton_and_alpha_contexts(
            response=cls.response,
            count_rate=observed,
            rotation_matrices=cls.rotation_matrices,
        )
        cls.result = fit_solar_wind_alpha_model(
            proton_ctx=proton_ctx,
            alpha_ctx=alpha_ctx,
            proton_moments=cls.proton_moments,
            magnetic_field_direction=cls.b_hat,
        )

    def test_alpha_velocity_minus_proton_velocity_is_parallel_to_bhat(self):
        """For a tilted B̂, the recovered (v_α − v_p) lies along ±B̂ — equivalently, the cross product (v_α − v_p) × B̂ is zero up to numerical noise."""
        delta = (
            self.result.velocity_rtn_nominal() - self.proton_velocity_rtn
        )
        np.testing.assert_allclose(np.cross(delta, self.b_hat), 0.0, atol=1e-9)

    def test_recovered_delta_v_matches_dot_product_of_velocity_offset(self):
        """The stored `delta_v` equals (v_α − v_p)·B̂ since B̂ is unit-normed — the scalar matches the projection of the vector offset onto the field."""
        delta_along = float(
            np.dot(
                self.result.velocity_rtn_nominal() - self.proton_velocity_rtn,
                self.b_hat,
            )
        )
        np.testing.assert_allclose(
            self.result.delta_v.nominal_value, delta_along, atol=1e-9
        )

    def test_alpha_velocity_recovers_truth_under_tilted_bhat(self):
        """With a 10° tilted B̂, the recovered alpha velocity matches the synthesis truth vector to ~1 km/s."""
        np.testing.assert_allclose(
            self.result.velocity_rtn_nominal(),
            self.alpha_velocity_truth,
            atol=1.0,
        )


# ----- AlphaSolarWindFitResult accessors --------------------------------------


class TestAlphaSolarWindFitResultAccessors(unittest.TestCase):
    """Tests for `AlphaSolarWindFitResult.velocity_rtn_nominal` and `velocity_rtn_covariance` — accessors that extract the nominal vector and covariance matrix from the correlated UFloat triple."""

    def setUp(self):
        # Build moments from a known correlated velocity covariance so the
        # accessor outputs are predictable. Using `make_correlated_velocity`
        # would require reconstructing the same covariance the alpha
        # pipeline produces; building the UFloat triple directly is enough
        # to exercise the accessors.
        from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
            make_correlated_velocity,
        )

        self.expected_nominal = np.array([-450.0, 10.0, -5.0])
        # Symmetric, positive-definite — `correlated_values` requires PSD.
        self.expected_covariance = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 9.0, 0.25],
                [0.5, 0.25, 16.0],
            ]
        )
        velocity_triple = make_correlated_velocity(
            self.expected_nominal, self.expected_covariance
        )
        self.moments = AlphaSolarWindFitResult(
            density=ufloat(0.2, 0.01),
            temperature=ufloat(4.0e5, 1.0e3),
            velocity_rtn=velocity_triple,
            delta_v=ufloat(30.0, 1.0),
            quality_flag=int(SwapiL3Flags.NONE),
        )

    def test_velocity_rtn_nominal_returns_per_component_nominals(self):
        """The nominal-vector accessor returns the per-component nominal_values as a length-3 ndarray."""
        np.testing.assert_array_equal(
            self.moments.velocity_rtn_nominal(), self.expected_nominal
        )

    def test_velocity_rtn_covariance_is_three_by_three(self):
        """The covariance accessor returns a 3×3 matrix matching the RTN velocity dimensionality."""
        covariance = self.moments.velocity_rtn_covariance()
        self.assertEqual(covariance.shape, (3, 3))

    def test_velocity_rtn_covariance_matches_input_covariance(self):
        """The covariance accessor round-trips: a UFloat triple built from a known PSD covariance returns that same matrix to machine precision."""
        np.testing.assert_allclose(
            self.moments.velocity_rtn_covariance(),
            self.expected_covariance,
            atol=1e-12,
        )


# ----- fit_solar_wind_alpha_model — non-converged LM ---------------------


class TestFitAlphaMomentsLMFailureFlag(unittest.TestCase):
    """Tests for `fit_solar_wind_alpha_model` — when LM does not converge (`result.success=False`), the chunk is rejected: every moment is NaN-filled and `FIT_ERROR` is set."""

    def test_fit_error_flag_set_when_least_squares_does_not_converge(self):
        """A non-converged LM result triggers the fit-quality guard: density and the other moments are NaN-filled and `FIT_ERROR` is reported alone."""
        proton_moments = _build_proton_fit_result()

        x_fit = np.array([np.log(0.2), np.log(4.0e5), 30.0])
        residual_norm = np.full(_N_MEAS, 1.0)
        jac = np.zeros((_N_MEAS, 3))
        jac[0, 0] = 1.0
        jac[0, 1] = 1.0
        jac[0, 2] = 1.0
        non_converged = MagicMock()
        non_converged.x = x_fit
        non_converged.fun = residual_norm
        non_converged.jac = jac
        non_converged.success = False

        response = _load_swapi_response_with_warm_cache()
        proton_ctx, alpha_ctx = _build_proton_and_alpha_contexts(
            response=response,
            count_rate=np.full(_FIVE_SWEEP_VOLTAGE.shape, 100.0),
        )

        with patch.object(
            alpha_module,
            "calculate_initial_guess",
            return_value=(0.2, 4.0e5, 0.0, np.array([10, 11, 12])),
        ), patch.object(
            alpha_module._AlphaEvaluator,
            "residuals",
            return_value=np.full(_N_MEAS, 1.0),
        ), patch.object(
            alpha_module.scipy.optimize,
            "least_squares",
            return_value=non_converged,
        ):
            result = fit_solar_wind_alpha_model(
                proton_ctx=proton_ctx,
                alpha_ctx=alpha_ctx,
                proton_moments=proton_moments,
                magnetic_field_direction=_B_HAT_RTN,
            )

        self.assertEqual(result.quality_flag, int(SwapiL3Flags.FIT_ERROR))
        self.assertTrue(np.isnan(result.density.nominal_value))


# ----- fit_solar_wind_alpha_model — initial-guess failure branches -------


class TestFitAlphaMomentsInitialGuessFailures(unittest.TestCase):
    """Tests for `fit_solar_wind_alpha_model` — failure branches inside `calculate_initial_guess` and `_infer_sweep_layout` that return None, causing the public function to short-circuit to a `FIT_ERROR` NaN-filled result."""

    def _call(self, *, count_rate, esa_voltage, rotation_matrices=None):
        n_meas = esa_voltage.size
        if rotation_matrices is None:
            rotation_matrices = np.broadcast_to(np.eye(3), (n_meas, 3, 3)).copy()
        warm_voltages = esa_voltage if n_meas > 0 else None
        response = load_swapi_response(warm_cache_voltages=warm_voltages)
        proton_ctx, alpha_ctx = _build_proton_and_alpha_contexts(
            response=response,
            count_rate=count_rate,
            voltage=esa_voltage,
            rotation_matrices=rotation_matrices,
        )
        return fit_solar_wind_alpha_model(
            proton_ctx=proton_ctx,
            alpha_ctx=alpha_ctx,
            proton_moments=_build_proton_fit_result(),
            magnetic_field_direction=_B_HAT_RTN,
        )

    def _assert_fit_error_nan(self, result):
        self.assertEqual(result.quality_flag, int(SwapiL3Flags.FIT_ERROR))
        _assert_moments_are_nan_filled(self, result)

    def test_one_dimensional_inputs_raise_value_error(self):
        """`fit_solar_wind_alpha_model` requires a 2D (n_sweeps, n_bins) count_rate for its per-sweep aggregations and raises `ValueError` on a 1D context rather than silently degrading. The chunk-level try/except is what converts the raise into a `FIT_ERROR` quality flag (covered in `test_chunk_fits.py`)."""
        voltage = np.logspace(np.log10(3500.0), np.log10(140.0), _N_MEAS)
        with self.assertRaises(ValueError):
            self._call(
                count_rate=np.zeros(_N_MEAS),
                esa_voltage=voltage,
            )

    def test_non_monotonic_voltage_raises_in_peak_finder(self):
        """A non-monotonic per-sweep voltage axis violates the alpha peak finder's decreasing-energies assertion. The exception propagates out of `fit_solar_wind_alpha_model`; the chunk-level try/except is what converts it into a `FIT_ERROR` quality flag (covered in `test_chunk_fits.py`)."""
        one_sweep = _ONE_SWEEP_VOLTAGE.copy()
        one_sweep[10], one_sweep[11] = one_sweep[11], one_sweep[10]
        voltage = np.broadcast_to(one_sweep, (_N_SWEEPS, _N_BINS_PER_SWEEP)).copy()
        with self.assertRaises(AssertionError):
            self._call(
                count_rate=np.ones(voltage.shape),
                esa_voltage=voltage,
            )

    def test_short_peak_window_returns_fit_error(self):
        """When the alpha peak finder returns a slice with fewer than 3 bins, `calculate_initial_guess` returns `None` and the fitter reports `FIT_ERROR`."""
        with patch.object(
            alpha_initial_guess_module,
            "_get_alpha_peak_indices",
            return_value=slice(10, 12),
        ):
            result = self._call(
                count_rate=np.ones(_FIVE_SWEEP_VOLTAGE.shape),
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
            )
        self._assert_fit_error_nan(result)

    def test_no_positive_residual_at_peak_returns_fit_error(self):
        """When the residual `count_avg − 2·proton_bg_avg` has no positive entry inside the peak window (e.g. count_rate=0 everywhere), `calculate_initial_guess` returns `None`."""
        with patch.object(
            alpha_initial_guess_module,
            "_get_alpha_peak_indices",
            return_value=slice(10, 20),
        ):
            result = self._call(
                count_rate=np.zeros(_FIVE_SWEEP_VOLTAGE.shape),
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
            )
        self._assert_fit_error_nan(result)

    def test_non_positive_unit_alpha_denominator_returns_fit_error(self):
        """When the unit-density alpha forward model evaluates to ~0 at every peak bin (peak slice placed in the highest-voltage bins, far above the alpha resonance), `denom <= 0` and `calculate_initial_guess` returns `None`."""
        # Inject a positive residual at the chosen peak bins by overriding
        # count_rate at those locations, then place the peak slice in the
        # high-voltage tail where unit_alpha ~ 0 (alpha truth voltage is
        # ~1264 V, voltage bin 0 is 3500 V).
        count_rate = np.zeros((_N_SWEEPS, _N_BINS_PER_SWEEP))
        peak_slice = slice(0, 3)
        count_rate[:, peak_slice] = 1.0e3
        with patch.object(
            alpha_initial_guess_module,
            "_get_alpha_peak_indices",
            return_value=peak_slice,
        ):
            result = self._call(
                count_rate=count_rate,
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
            )
        self._assert_fit_error_nan(result)


# Doc contracts intentionally not pinned here: upstream-set quality flags
# (`PRELIMINARY_MAG`) belong in chunk-fitter / processor tests, and the
# LM-jacobian-derived sigmas (`σ_n_α`, `σ_T_α`,
# `σ_Δv`) plus the `Σ_v_α = Σ_v_p + σ_Δv²·B̂B̂ᵀ` covariance update would
# require an independent numerical reference to pin without coupling to
# scipy version drift.


# ----- _AlphaEvaluator analytic Jacobian -----------------------------------


class TestAlphaEvaluatorAnalyticJacobianMatchesFiniteDifference(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """Tests for `_AlphaEvaluator.jacobian` — verifies the analytic 3-column Jacobian (∂residuals/∂[log n_α, log T_α, Δv]) matches a central-difference numerical Jacobian within the same tolerance used for the proton-fit forward-model Jacobian (5% rtol). The small mismatch comes from the dynamic-quadrature limits shifting when the state is perturbed for finite differences, per `docs/swapi/solar-wind-moments.md` § Boundary terms."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.alpha_ctx = build_solar_wind_fit_context(
            count_rate=cls.observed_count_rate,
            esa_voltage=cls.voltage,
            swapi_response=cls.response,
            central_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        cls.evaluator = _AlphaEvaluator(
            proton_bulk=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
            proton_true_rate=cls.proton_true_rate,
            alpha_ctx=cls.alpha_ctx,
        )
        cls.x0 = np.array(
            [
                np.log(_TRUE_ALPHA_DENSITY_CM3),
                np.log(_TRUE_ALPHA_TEMPERATURE_K),
                _TRUE_DELTA_V_KM_S,
            ]
        )
        cls.analytic_jacobian = cls.evaluator.jacobian(cls.x0).copy()
        steps = np.array([1.0e-5, 1.0e-5, 1.0e-3])
        n_residuals = cls.analytic_jacobian.shape[0]
        cls.numerical_jacobian = np.empty((n_residuals, 3))
        for j in range(3):
            x_plus = cls.x0.copy()
            x_minus = cls.x0.copy()
            x_plus[j] += steps[j]
            x_minus[j] -= steps[j]
            r_plus = cls.evaluator.residuals(x_plus).copy()
            r_minus = cls.evaluator.residuals(x_minus).copy()
            cls.numerical_jacobian[:, j] = (r_plus - r_minus) / (2.0 * steps[j])

    def test_jacobian_shape_is_n_residuals_by_three(self):
        """The analytic Jacobian has shape (N_residuals, 3) — one row per measurement bin, one column per fit parameter (log n_α, log T_α, Δv)."""
        self.assertEqual(self.analytic_jacobian.shape, (self.alpha_ctx.count_rate.size, 3))

    def test_log_density_column_equals_alpha_only_rate_times_deadtime_squared(self):
        """Because the rate is linear in n_α, the analytic ∂(observable)/∂(log n_α) column equals 𝒟²(R_total) · R_α exactly with no quadrature slack — an identity check independent of finite-difference noise."""
        from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
            model_solar_wind_ideal_coincidence_rates,
        )
        alpha_only_params = SolarWindParams(
            density=_TRUE_ALPHA_DENSITY_CM3,
            velocity_rtn=_TRUE_PROTON_VELOCITY_RTN + _TRUE_DELTA_V_KM_S * _B_HAT_RTN,
            temperature=_TRUE_ALPHA_TEMPERATURE_K,
            mass=ALPHA_PARTICLE_MASS_KG,
        )
        alpha_rate, _ = model_solar_wind_ideal_coincidence_rates(
            alpha_only_params, self.alpha_ctx
        )
        total_rate = self.proton_true_rate + alpha_rate
        deadtime_squared = deadtime_factor(total_rate) ** 2
        np.testing.assert_allclose(
            self.analytic_jacobian[:, 0],
            deadtime_squared * alpha_rate,
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_log_temperature_column_matches_finite_difference(self):
        """The analytic ∂residuals/∂(log T_α) column matches central differences within 5% — the threshold matches the proton forward-model Jacobian tolerance."""
        np.testing.assert_allclose(
            self.analytic_jacobian[:, 1],
            self.numerical_jacobian[:, 1],
            rtol=5.0e-2,
            atol=1.0e-3,
        )

    def test_delta_v_column_matches_finite_difference(self):
        """The analytic ∂residuals/∂(Δv) column matches central differences within 5% — derived from the 5-D forward-model Jacobian projected onto B̂, inherits the same boundary-term residual."""
        np.testing.assert_allclose(
            self.analytic_jacobian[:, 2],
            self.numerical_jacobian[:, 2],
            rtol=5.0e-2,
            atol=1.0e-3,
        )


class TestAlphaEvaluatorCachesEvaluation(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """Tests for `_AlphaEvaluator._refresh` — caches the last (residuals, jacobian) so scipy.least_squares' separate `fun` and `jac` callbacks share one forward-model evaluation per state."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.alpha_ctx = build_solar_wind_fit_context(
            count_rate=cls.observed_count_rate,
            esa_voltage=cls.voltage,
            swapi_response=cls.response,
            central_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        cls.x0 = np.array(
            [
                np.log(_TRUE_ALPHA_DENSITY_CM3),
                np.log(_TRUE_ALPHA_TEMPERATURE_K),
                _TRUE_DELTA_V_KM_S,
            ]
        )

    def test_residuals_then_jacobian_at_same_state_share_one_forward_model_call(self):
        """Calling residuals(x) and then jacobian(x) at the same x triggers exactly one underlying forward-model call — the second call hits the cache."""
        evaluator = _AlphaEvaluator(
            proton_bulk=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
            proton_true_rate=self.proton_true_rate,
            alpha_ctx=self.alpha_ctx,
        )
        with patch.object(
            alpha_module,
            "model_solar_wind_ideal_coincidence_rates",
            wraps=alpha_module.model_solar_wind_ideal_coincidence_rates,
        ) as wrapped:
            evaluator.residuals(self.x0)
            evaluator.jacobian(self.x0)
        self.assertEqual(wrapped.call_count, 1)

    def test_residuals_at_different_states_each_trigger_forward_model_call(self):
        """Calling residuals at two distinct states triggers two forward-model calls — the cache is keyed on the exact state vector."""
        evaluator = _AlphaEvaluator(
            proton_bulk=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
            proton_true_rate=self.proton_true_rate,
            alpha_ctx=self.alpha_ctx,
        )
        x1 = self.x0.copy()
        x2 = self.x0.copy()
        x2[2] += 1.0
        with patch.object(
            alpha_module,
            "model_solar_wind_ideal_coincidence_rates",
            wraps=alpha_module.model_solar_wind_ideal_coincidence_rates,
        ) as wrapped:
            evaluator.residuals(x1)
            evaluator.residuals(x2)
        self.assertEqual(wrapped.call_count, 2)


# ----- LM call wiring: analytic Jacobian passed to scipy -------------------


class TestFitAlphaMomentsPassesAnalyticJacobianToLM(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """Tests for `fit_solar_wind_alpha_model` — verifies scipy.optimize.least_squares is invoked with `jac=...` (rather than the previous finite-difference `diff_step=...` path)."""

    def test_least_squares_receives_jac_callable_and_no_diff_step(self):
        """LM is called with a `jac` callable (the evaluator's analytic Jacobian) and without `diff_step` — finite-difference Jacobian estimation is no longer needed."""
        proton_moments = _build_proton_fit_result()

        peak_bin_idx = np.array([10, 11, 12])
        n_peak_residuals = _N_SWEEPS * peak_bin_idx.size

        mock_result = MagicMock()
        mock_result.x = np.array(
            [np.log(_TRUE_ALPHA_DENSITY_CM3), np.log(_TRUE_ALPHA_TEMPERATURE_K), 0.0]
        )
        mock_result.fun = np.zeros(n_peak_residuals)
        mock_result.jac = np.zeros((n_peak_residuals, 3))
        mock_result.success = True

        proton_ctx, alpha_ctx = _build_proton_and_alpha_contexts(
            response=self.response,
            count_rate=self.observed_count_rate,
            voltage=self.voltage,
            rotation_matrices=self.rotation_matrices,
        )

        with patch.object(
            alpha_module,
            "calculate_initial_guess",
            return_value=(
                _TRUE_ALPHA_DENSITY_CM3,
                _TRUE_ALPHA_TEMPERATURE_K,
                0.0,
                peak_bin_idx,
            ),
        ), patch.object(
            alpha_module._AlphaEvaluator,
            "residuals",
            return_value=np.zeros(n_peak_residuals),
        ), patch.object(
            alpha_module.scipy.optimize,
            "least_squares",
            return_value=mock_result,
        ) as mock_lm:
            fit_solar_wind_alpha_model(
                proton_ctx=proton_ctx,
                alpha_ctx=alpha_ctx,
                proton_moments=proton_moments,
                magnetic_field_direction=_B_HAT_RTN,
            )

        kwargs = mock_lm.call_args.kwargs
        self.assertIn("jac", kwargs)
        self.assertTrue(callable(kwargs["jac"]))
        self.assertNotIn("diff_step", kwargs)
        self.assertEqual(kwargs["method"], "lm")


if __name__ == "__main__":
    unittest.main()
