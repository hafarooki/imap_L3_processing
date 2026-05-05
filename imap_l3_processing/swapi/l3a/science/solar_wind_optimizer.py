from dataclasses import dataclass

import numba
import numpy as np
import scipy.optimize
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import SolarWindFitContext
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    apply_deadtime_correction_array,
    model_solar_wind_ideal_coincidence_rates,
    SolarWindParams,
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


def optimize_solar_wind_params(
    initial_guess: SolarWindParams, ctx: SolarWindFitContext
) -> OptimizeSolarWindParamsResult:
    def wrapper(state):
        return _calculate_residuals(
            SolarWindParams.from_state_vector(state, ctx.mass_kg), ctx
        )

    raw: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
        wrapper, initial_guess.to_state_vector(),
        method="lm", diff_step=1e-4, xtol=1e-4
    )

    return OptimizeSolarWindParamsResult(
        sw_params=SolarWindParams.from_state_vector(raw.x, ctx.mass_kg),
        residuals=raw.fun,
        jacobian=raw.jac,
        success=bool(raw.success),
    )


@numba.njit
def _calculate_residuals(sw_params: SolarWindParams, ctx: SolarWindFitContext) -> ndarray:
    model_true = model_solar_wind_ideal_coincidence_rates(sw_params, ctx)
    model_obs = apply_deadtime_correction_array(model_true)
    return model_obs - ctx.count_rate
