import math

import numba
import numpy as np
import scipy.optimize
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


_MAX_BASIN_REFINE_ITERS = 6
_ROTATED_RMSE_RATIO_THRESHOLD = 10
# Fractional bounds on |v_perp| / |v|. The endpoints are clipped off zero/one
# so |v_par| stays real and sgn(v_par·axis) is unambiguous.
_PERP_FRACTION_BOUNDS = (1e-3, 1.0 - 1e-3)


def escape_local_minimum(
    first_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    spin_axis_rtn = _average_spin_axis_in_rtn(ctx.rotation_matrices)

    current_result = first_result
    for _ in range(_MAX_BASIN_REFINE_ITERS):
        flipped_velocity, flipped_density, flipped_mse = _flipped_seed(
            current_result, ctx, spin_axis_rtn,
        )

        if flipped_mse >= current_result.mse * _ROTATED_RMSE_RATIO_THRESHOLD ** 2:
            break

        refined_velocity, refined_density = _refine_perp_distance(
            current_result.sw_params, flipped_velocity, spin_axis_rtn, ctx,
        )

        restart_result = _restart_from_rotated_seed(
            current_result, refined_velocity, refined_density, ctx
        )

        if restart_result.mse > current_result.mse:
            break

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


def _flipped_seed(
    lm_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
    spin_axis_rtn: ndarray,
) -> tuple[ndarray, float, float]:
    sw = lm_result.sw_params
    flipped_velocity = _flip_vector_about_axis(sw.bulk_velocity_rtn, spin_axis_rtn)
    predicted_obs_rate = apply_deadtime_correction_array(
        model_solar_wind_coincidence_rates(
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
    # Rodrigues' rotation formula for a 180-degree rotation
    return 2.0 * axis * float(np.dot(axis, v)) - v


def _refine_perp_distance(
    current_params: SolarWindParams,
    flipped_velocity: ndarray,
    spin_axis_rtn: ndarray,
    ctx: SolarWindFitContext,
) -> tuple[ndarray, float]:
    """Slide the flipped seed along an iso-|v| arc through the spin axis.

    Bulk speed is well-constrained by the count-rate peak voltage, so we hold
    |v| fixed at the LM value and sweep the perpendicular fraction f =
    |v_perp| / |v|. The parallel magnitude is then |v_par| = |v|·sqrt(1−f²),
    keeping its sign aligned with the LM solution's parallel projection (the
    spin axis is mostly anti-radial, so this preserves the inflow direction).
    Density is rescaled optimally at each f so the cost is purely geometric.
    """
    bulk_speed = float(np.linalg.norm(flipped_velocity))
    if bulk_speed <= 0.0:
        return flipped_velocity, current_params.density

    parallel_dot = float(np.dot(spin_axis_rtn, flipped_velocity))
    perpendicular = flipped_velocity - parallel_dot * spin_axis_rtn
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    if perpendicular_norm == 0.0:
        return flipped_velocity, current_params.density
    perpendicular_unit = perpendicular / perpendicular_norm
    parallel_unit = np.sign(parallel_dot) * spin_axis_rtn

    def cost(perp_fraction: float) -> float:
        return _arc_perp_fraction_mse(
            float(perp_fraction), bulk_speed,
            parallel_unit, perpendicular_unit,
            current_params.temperature, current_params.mass_kg, ctx,
        )

    result = scipy.optimize.minimize_scalar(
        cost, bounds=_PERP_FRACTION_BOUNDS, method="bounded",
        options={"xatol": 1e-3},
    )
    optimal_fraction = float(result.x)
    refined_velocity = (
        bulk_speed * np.sqrt(1.0 - optimal_fraction ** 2) * parallel_unit
        + bulk_speed * optimal_fraction * perpendicular_unit
    )
    predicted_obs_rate = apply_deadtime_correction_array(
        model_solar_wind_coincidence_rates(
            SolarWindParams(
                1.0, refined_velocity,
                current_params.temperature, current_params.mass_kg,
            ),
            ctx,
        )
    )
    refined_density = optimal_density_scale(predicted_obs_rate, ctx.count_rate)
    return refined_velocity, refined_density


@numba.njit(nogil=True)
def _arc_perp_fraction_mse(
    perp_fraction: float,
    bulk_speed: float,
    parallel_unit: ndarray,
    perpendicular_unit: ndarray,
    temperature: float,
    mass_kg: float,
    ctx: SolarWindFitContext,
) -> float:
    parallel_mag = bulk_speed * math.sqrt(max(1.0 - perp_fraction * perp_fraction, 0.0))
    perp_mag = bulk_speed * perp_fraction
    velocity = parallel_mag * parallel_unit + perp_mag * perpendicular_unit
    sw_params = SolarWindParams(1.0, velocity, temperature, mass_kg)
    predicted_obs_rate = apply_deadtime_correction_array(
        model_solar_wind_coincidence_rates(sw_params, ctx)
    )
    pp = 0.0
    po = 0.0
    for i in range(predicted_obs_rate.size):
        pp += predicted_obs_rate[i] * predicted_obs_rate[i]
        po += predicted_obs_rate[i] * ctx.count_rate[i]
    if pp <= 0.0:
        return float(np.mean(ctx.count_rate ** 2))
    density_scale = po / pp
    sse = 0.0
    for i in range(predicted_obs_rate.size):
        diff = density_scale * predicted_obs_rate[i] - ctx.count_rate[i]
        sse += diff * diff
    return sse / predicted_obs_rate.size
