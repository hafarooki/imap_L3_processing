from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import SolarWindFitContext
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    SWAPI_DEADTIME_S,
    SolarWindParams,
    model_solar_wind_ideal_coincidence_rates,
)


@dataclass
class OptimizeSolarWindParamsResult:
    sw_params: SolarWindParams
    residuals: ndarray  # count-rate residuals at the solution
    jacobian: ndarray   # ∂residuals/∂state, columns ordered per the state vector
    success: bool

    @property
    def mse(self) -> float:
        return float(np.mean(self.residuals ** 2))


class _ResidJacEvaluator:
    """Caches the most recent (residuals, jacobian) so scipy.least_squares' separate
    `fun` and `jac` callbacks share a single forward-model evaluation per state."""

    def __init__(self, ctx: SolarWindFitContext):
        self.ctx = ctx
        self._last_state: ndarray | None = None
        self._last_resid: ndarray | None = None
        self._last_jac: ndarray | None = None

    def _eval(self, state: ndarray) -> None:
        sw = SolarWindParams.from_state_vector(state, self.ctx.mass_kg)
        rate_true, partial = model_solar_wind_ideal_coincidence_rates(sw, self.ctx)
        deadtime_factor = 1.0 / (1.0 + SWAPI_DEADTIME_S * rate_true)
        jac = np.empty((rate_true.shape[0], 5))
        jac[:, 0] = rate_true       # ∂C_true / ∂ ln n  (Maxwellian is linear in n)
        jac[:, 1:] = partial        # [d/d ln T, d/d v_R, d/d v_T, d/d v_N]
        jac *= (deadtime_factor * deadtime_factor)[:, None]  # chain rule onto C_obs
        self._last_state = state.copy()
        self._last_resid = rate_true * deadtime_factor - self.ctx.count_rate
        self._last_jac = jac

    def _refresh(self, state: ndarray) -> None:
        if self._last_state is None or not np.array_equal(state, self._last_state):
            self._eval(state)

    def resid(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_resid

    def jac(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_jac


def optimize_solar_wind_params(
    initial_guess: SolarWindParams, ctx: SolarWindFitContext
) -> OptimizeSolarWindParamsResult:
    evaluator = _ResidJacEvaluator(ctx)

    raw: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
        evaluator.resid, initial_guess.to_state_vector(),
        jac=evaluator.jac, method="lm", xtol=1e-4
    )

    return OptimizeSolarWindParamsResult(
        sw_params=SolarWindParams.from_state_vector(raw.x, ctx.mass_kg),
        residuals=raw.fun,
        jacobian=raw.jac,
        success=bool(raw.success),
    )
