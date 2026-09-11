import math
import unittest

import numpy as np

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    N_STATE,
    SolarWindParams,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000, proton_params
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


# Reference bulk speed for the slow-wind fixture state shared by every test
# in this module (5 cm⁻³, 100 kK protons under the default `proton_params`).
_REF_BULK_SPEED_KM_S = 450.0

# Off-axis bulk velocity used for the velocity-component Jacobian tests.
# Adding small components in R and N breaks the symmetry that makes vR/vN
# Jacobian entries near zero, so finite-difference noise doesn't dominate.
_OFF_AXIS_VELOCITY_RTN = np.array([30.0, -_REF_BULK_SPEED_KM_S, 20.0])

# Index aliases for the RTN velocity components inside the LM state vector.
_VELOCITY_R_IDX, _VELOCITY_T_IDX, _VELOCITY_N_IDX = (
    VELOCITY_SLICE.start,
    VELOCITY_SLICE.start + 1,
    VELOCITY_SLICE.start + 2,
)


def _esa_voltage_at_speed(speed_km_s: float) -> float:
    """ESA voltage (V) whose central proton speed equals `speed_km_s`. The
    relation is `V = ½ m_p v² / (e · k_SWAPI)`."""
    return (
        0.5
        * PROTON_MASS_KG
        * (speed_km_s * 1e3) ** 2
        / PROTON_CHARGE_COULOMBS
        / SWAPI_K_FACTOR
    )


# Putting the bulk speed at the passband center maximizes the rate and
# minimizes truncation effects in the quadrature.
_ESA_VOLTAGE_AT_REF_SPEED_V = _esa_voltage_at_speed(_REF_BULK_SPEED_KM_S)

# Numerical-differentiation parameters. `1e-5` is small enough that truncation
# from finite differences is well below the GL quadrature noise floor for the
# scales involved (≈10⁴ Hz rates, ≈10² km/s velocities) but large enough that
# round-off error stays well below 1%.
_FD_RELATIVE_STEP = 1e-5

# Tolerances for analytic-vs-finite-difference Jacobian comparison. The Jacobian
# is exact in the continuum but is evaluated with the same Gauss-Legendre
# quadrature as the rate; the small mismatch comes from the dynamic-quadrature
# limits shifting (slightly) when the state is perturbed for finite differences,
# which moves the GL nodes off the exact integrand peaks. Measured worst case
# on this fixture (off-axis bulk, 100 kK, 450 km/s) is ~2.74% on the vR column;
# 5% leaves headroom for the same machinery to stay green on adjacent fixtures
# while still flagging an analytic-derivative bug (sign flip or factor-of-2,
# which would shift the column by ~100% or ~50%). To remeasure: evaluate
# `model_solar_wind_ideal_coincidence_rates` at the test fixture state and
# compare each column against `_finite_difference_jacobian_column`.
_JACOBIAN_RTOL = 0.05
_JACOBIAN_ATOL = 1e-3




def _build_fit_context_for_voltages(
    response: SwapiResponse,
    esa_voltages: np.ndarray,
    rotation_matrices: np.ndarray = None,
):
    """Wrap `build_solar_wind_fit_context` with sensible defaults for these
    tests: identity rotations and dummy count rates."""
    voltages_array = np.asarray(esa_voltages, dtype=float)
    if rotation_matrices is None:
        rotation_matrices = np.stack([np.eye(3)] * len(voltages_array))
    response.warm_cache(voltages_array)
    return build_solar_wind_fit_context(
        count_rate=np.full(len(voltages_array), 100.0),
        esa_voltage=voltages_array,
        swapi_response=response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=rotation_matrices,
    )


class _ForwardModelFixture(unittest.TestCase):
    """Loads the SwapiResponse once per class — the CSV parsing is not free
    and none of these tests mutate the response."""

    @classmethod
    def setUpClass(cls):
        cls.response = SwapiResponse.from_files(
            azimuthal_transmission_path=get_test_instrument_team_data_path(
                "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
            ),
            central_effective_area_path=get_test_instrument_team_data_path(
                "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
            ),
            passband_fit_coefficients_path=get_test_instrument_team_data_path(
                "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
            ),
            efficiency_table_path=get_test_data_path(
                "swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat"
            ),
        )


class TestModelSolarWindIdealCoincidenceRatesShape(_ForwardModelFixture):
    """Tests for `model_solar_wind_ideal_coincidence_rates`, output-shape contract."""

    def test_returns_one_rate_and_one_jacobian_row_per_sweep(self):
        """Given three sweeps spanning the passband-center voltage, the model returns a length-3 rate vector and a (3, N_STATE) Jacobian."""
        ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 0.95,
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        rates, jacobian = model_solar_wind_ideal_coincidence_rates(
            proton_params(), ctx
        )
        self.assertEqual(rates.shape, (3,))
        self.assertEqual(jacobian.shape, (3, N_STATE))


class TestCalculateIntegralRateBehavior(_ForwardModelFixture):
    """Tests for `calculate_integral`, rate-value behavior (Jacobian asserted elsewhere)."""

    def setUp(self):
        self.ctx = _build_fit_context_for_voltages(
            self.response, np.array([_ESA_VOLTAGE_AT_REF_SPEED_V])
        )
        self.response_grid = self.ctx.response_grids[0]
        self.rotation_matrix = self.ctx.rotation_matrices[0]

    def test_rate_is_positive_when_velocity_aligns_with_boresight(self):
        """With bulk speed at the passband center and the flow aimed along boresight, the predicted rate is strictly positive."""
        rate, _ = calculate_integral(
            proton_params(), self.response_grid, self.rotation_matrix
        )
        self.assertGreater(rate, 0.0)

    def test_rate_is_linear_in_density(self):
        """Doubling the proton density at fixed temperature and geometry doubles the predicted rate exactly, since the rate is linear in `n`."""
        rate_at_5, _ = calculate_integral(
            proton_params(density=5.0),
            self.response_grid,
            self.rotation_matrix,
        )
        rate_at_10, _ = calculate_integral(
            proton_params(density=10.0),
            self.response_grid,
            self.rotation_matrix,
        )
        np.testing.assert_allclose(rate_at_10, 2.0 * rate_at_5, rtol=1e-12)

    def test_returns_zero_rate_and_jacobian_when_bulk_speed_misses_passband_above(self):
        """When the bulk speed (2000 km/s) sits hundreds of sigma above the passband at this voltage, the model short-circuits to rate=0 and a zero Jacobian row."""
        very_fast = proton_params(velocity_rtn=np.array([0.0, -2000.0, 0.0]))
        rate, jacobian_row = calculate_integral(
            very_fast, self.response_grid, self.rotation_matrix
        )
        self.assertEqual(rate, 0.0)
        np.testing.assert_array_equal(jacobian_row, np.zeros(N_STATE))

    def test_returns_zero_rate_and_jacobian_when_bulk_speed_misses_passband_below(self):
        """Mirror of the above on the low side: a 50 km/s bulk sits far below the passband at this voltage, so rate=0 and the Jacobian row is zero."""
        very_slow = proton_params(velocity_rtn=np.array([0.0, -50.0, 0.0]))
        rate, jacobian_row = calculate_integral(
            very_slow, self.response_grid, self.rotation_matrix
        )
        self.assertEqual(rate, 0.0)
        np.testing.assert_array_equal(jacobian_row, np.zeros(N_STATE))


class TestAnalyticJacobianIdentities(_ForwardModelFixture):
    """Tests for `model_solar_wind_ideal_coincidence_rates`, exact closed-form Jacobian identities that hold independent of quadrature."""

    def test_log_density_jacobian_column_equals_the_rate(self):
        """Because the rate is linear in `n`, the analytic log-density Jacobian column equals the rate vector exactly with no quadrature slack."""
        ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 0.95,
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        rates, jacobian = model_solar_wind_ideal_coincidence_rates(
            proton_params(velocity_rtn=_OFF_AXIS_VELOCITY_RTN),
            ctx,
        )
        np.testing.assert_array_equal(jacobian[:, LOG_DENSITY_IDX], rates)


def _rate_at_state(state: np.ndarray, ctx) -> np.ndarray:
    """Evaluate the rate vector at a state-vector point (no Jacobian)."""
    sw_local = SolarWindParams.from_vector(state, mass=PROTON_MASS_KG)
    rates, _ = model_solar_wind_ideal_coincidence_rates(sw_local, ctx)
    return rates.copy()


def _finite_difference_jacobian_column(state: np.ndarray, ctx, j: int) -> np.ndarray:
    """Central-difference column `j` of the Jacobian: `(C(x+ε e_j) − C(x−ε e_j)) / 2ε`."""
    eps = _FD_RELATIVE_STEP * max(abs(state[j]), 1.0)
    forward = state.copy()
    forward[j] += eps
    backward = state.copy()
    backward[j] -= eps
    return (_rate_at_state(forward, ctx) - _rate_at_state(backward, ctx)) / (2 * eps)


class TestAnalyticJacobianAgainstFiniteDifferences(_ForwardModelFixture):
    """Tests for `model_solar_wind_ideal_coincidence_rates`, analytic Jacobian columns vs. central finite differences."""

    def setUp(self):
        # Two-sweep context with a small voltage gradient so each sweep
        # contributes a different, non-degenerate row to the Jacobian.
        self.ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        # Off-axis bulk velocity so all three velocity components produce
        # nonzero Jacobian entries (boresight-aligned bulk has vR/vN ≈ 0
        # by symmetry, and the FD signal-to-noise on those columns is poor).
        self.sw = proton_params(velocity_rtn=_OFF_AXIS_VELOCITY_RTN)
        self.state = self.sw.to_vector()

    def test_log_temperature_jacobian_is_correct(self):
        """At an off-axis bulk state, the analytic log-temperature Jacobian column matches a central finite difference of the rate to within GL quadrature noise."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, LOG_TEMPERATURE_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, LOG_TEMPERATURE_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_r_jacobian_is_correct(self):
        """At an off-axis bulk state, the analytic v_R Jacobian column matches a central finite difference of the rate to within GL quadrature noise."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_R_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_R_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_t_jacobian_is_correct(self):
        """At an off-axis bulk state, the analytic v_T Jacobian column matches a central finite difference of the rate to within GL quadrature noise."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_T_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_T_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_n_jacobian_is_correct(self):
        """At an off-axis bulk state, the analytic v_N Jacobian column matches a central finite difference of the rate to within GL quadrature noise."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_N_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_N_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )


class TestNonIdentityRotation(_ForwardModelFixture):
    """Tests for `calculate_integral`, invariance under joint rotation of bulk velocity and SWAPI->RTN matrix."""

    def test_rate_is_invariant_under_joint_rotation_of_bulk_and_response(self):
        """Rotating both the bulk velocity (in RTN) and the SWAPI->RTN matrix by the same 30 degree rotation reproduces the identity-rotation baseline rate."""
        baseline_ctx = _build_fit_context_for_voltages(
            self.response, np.array([_ESA_VOLTAGE_AT_REF_SPEED_V])
        )
        baseline_rate, _ = calculate_integral(
            proton_params(velocity_rtn=_OFF_AXIS_VELOCITY_RTN),
            baseline_ctx.response_grids[0],
            baseline_ctx.rotation_matrices[0],
        )

        # 30° rotation about RTN +R. Rotating both the bulk velocity and the
        # instrument-to-RTN matrix by the same Q preserves the bulk direction
        # in the instrument frame, which is all the integrand sees.
        angle_rad = math.radians(30.0)
        rotation_about_r = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle_rad), -math.sin(angle_rad)],
                [0.0, math.sin(angle_rad), math.cos(angle_rad)],
            ]
        )
        rotated_velocity_rtn = rotation_about_r @ _OFF_AXIS_VELOCITY_RTN
        rotated_xyz_to_rtn = rotation_about_r
        rotated_ctx = _build_fit_context_for_voltages(
            self.response,
            np.array([_ESA_VOLTAGE_AT_REF_SPEED_V]),
            rotation_matrices=np.stack([rotated_xyz_to_rtn]),
        )
        rotated_rate, _ = calculate_integral(
            proton_params(velocity_rtn=rotated_velocity_rtn),
            rotated_ctx.response_grids[0],
            rotated_ctx.rotation_matrices[0],
        )

        # Agreement is limited only by GL quadrature node positions, which
        # are placed on the same bulk-frame angular window in both cases.
        np.testing.assert_allclose(rotated_rate, baseline_rate, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
