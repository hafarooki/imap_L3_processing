import math

import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import SolarWindFitContext
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    apply_deadtime_correction_array,
    model_solar_wind_coincidence_rates,
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.science.proton_initial_guess import (
    optimal_density_scale,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_optimizer import (
    OptimizeSolarWindParamsResult,
    optimize_solar_wind_params,
)


_GRID_K = 4
_MAX_BASIN_REFINE_ITERS = 6
_ROTATED_RMSE_RATIO_THRESHOLD = 10


def escape_local_minimum(
    first_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    spin_axis_rtn = _average_spin_axis_in_rtn(ctx.rotation_matrices)

    current_result = first_result
    for _ in range(_MAX_BASIN_REFINE_ITERS):
        rotated_velocity, rotated_density, rotated_mse = _best_k_rotation_seed(
            current_result, ctx, spin_axis_rtn,
        )

        if rotated_mse >= current_result.mse * _ROTATED_RMSE_RATIO_THRESHOLD ** 2:
            continue

        restart_result = _restart_from_rotated_seed(
            current_result, rotated_velocity, rotated_density, ctx
        )

        if restart_result.mse < current_result.mse:
            current_result = restart_result

    return current_result


def _average_spin_axis_in_rtn(rotation_matrices: ndarray) -> ndarray:
    axis = rotation_matrices[:, 1, :].mean(axis=0)
    return axis / np.linalg.norm(axis)


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


def _best_k_rotation_seed(
    lm_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
    spin_axis_rtn: ndarray,
) -> tuple[ndarray, float, float]:
    sw = lm_result.sw_params
    best_velocity = sw.bulk_velocity_rtn
    best_density = sw.density
    best_mse = math.inf
    for rotation_index in range(1, _GRID_K):
        rotation_angle_rad = 2.0 * math.pi * rotation_index / _GRID_K
        rotated_velocity = _rotate_vector_about_axis(
            sw.bulk_velocity_rtn, spin_axis_rtn, rotation_angle_rad
        )
        predicted_obs_rate = apply_deadtime_correction_array(
            model_solar_wind_coincidence_rates(
                SolarWindParams(sw.density, rotated_velocity, sw.temperature, sw.mass_kg),
                ctx,
            )
        )
        density_scale = optimal_density_scale(predicted_obs_rate, ctx.count_rate)
        rotated_mse = float(
            np.mean((density_scale * predicted_obs_rate - ctx.count_rate) ** 2)
        )
        if rotated_mse < best_mse:
            best_mse = rotated_mse
            best_velocity = rotated_velocity
            best_density = density_scale * sw.density
    return best_velocity, best_density, best_mse


def _rotate_vector_about_axis(v: ndarray, axis: ndarray, angle: float) -> ndarray:
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    return (
        v * cos_t
        + np.cross(axis, v) * sin_t
        + axis * float(np.dot(axis, v)) * (1.0 - cos_t)
    )
