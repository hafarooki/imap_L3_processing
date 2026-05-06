import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    SolarWindFitContext,
    average_spin_axis_rtn,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    apply_deadtime_correction_array,
    model_solar_wind_ideal_coincidence_rates,
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.science.proton_initial_guess import (
    optimal_density_scale,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_optimizer import (
    OptimizeSolarWindParamsResult,
    optimize_solar_wind_params,
)


_MAX_BASIN_REFINE_ITERS = 6
_ROTATED_RMSE_RATIO_THRESHOLD = 10


def escape_local_minimum(
    first_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    spin_axis_rtn = average_spin_axis_rtn(ctx.rotation_matrices)

    current_result = first_result
    for _ in range(_MAX_BASIN_REFINE_ITERS):
        flipped_velocity, flipped_density, flipped_mse = _flipped_seed(
            current_result,
            ctx,
            spin_axis_rtn,
        )

        if flipped_mse >= current_result.mse * _ROTATED_RMSE_RATIO_THRESHOLD**2:
            break

        restart_result = _restart_from_rotated_seed(
            current_result,
            flipped_velocity,
            flipped_density,
            ctx,
        )

        if restart_result.mse > current_result.mse:
            break

        current_result = restart_result

    return current_result


def _flipped_seed(
    lm_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
    spin_axis_rtn: ndarray,
) -> tuple[ndarray, float, float]:
    sw = lm_result.sw_params
    flipped_velocity = _flip_vector_about_axis(sw.bulk_velocity_rtn, spin_axis_rtn)
    predicted_obs_rate = apply_deadtime_correction_array(
        model_solar_wind_ideal_coincidence_rates(
            SolarWindParams(sw.density, flipped_velocity, sw.temperature, sw.mass_kg),
            ctx,
        )
    )
    density_scale = optimal_density_scale(predicted_obs_rate, ctx.count_rate)
    flipped_mse = float(
        np.mean((density_scale * predicted_obs_rate - ctx.count_rate) ** 2)
    )
    return flipped_velocity, density_scale * sw.density, flipped_mse


def _flip_vector_about_axis(v: ndarray, axis: ndarray) -> ndarray:
    return 2.0 * axis * float(np.dot(axis, v)) - v


def _restart_from_rotated_seed(
    current_result: OptimizeSolarWindParamsResult,
    rotated_velocity: ndarray,
    rotated_density: float,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    sw = current_result.sw_params
    density = (
        rotated_density
        if rotated_density > 0.0 and np.isfinite(rotated_density)
        else sw.density
    )
    restart_guess = SolarWindParams(
        density=density,
        bulk_velocity_rtn=rotated_velocity,
        temperature=sw.temperature,
        mass_kg=sw.mass_kg,
    )
    return optimize_solar_wind_params(restart_guess, ctx=ctx)
