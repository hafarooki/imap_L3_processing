import math

import numpy as np
import scipy.optimize
from numpy import ndarray

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    SolarWindParams,
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    esa_voltage_to_proton_speed,
)


INITIAL_TEMPERATURE_FLOOR_K = (
    PROTON_CHARGE_COULOMBS / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
)


def calculate_initial_guess(ctx: SolarWindFitContext) -> SolarWindParams:
    speed = esa_voltage_to_proton_speed(ctx.esa_voltage)

    peak_idx = np.nanargmax(ctx.count_rate)
    bulk_speed_seed = float(speed[peak_idx])

    temperature_seed = max(
        60000.0 * (bulk_speed_seed / 400.0) ** 2,
        INITIAL_TEMPERATURE_FLOOR_K,
    )

    bulk_speed_init, temperature = _gaussian_refine_bulk_speed_and_temperature(
        speed, ctx.count_rate, bulk_speed_seed, temperature_seed, ctx.mass_kg,
    )

    spin_axis_rtn = ctx.rotation_matrices[:, 1, :].mean(axis=0)
    spin_axis_rtn = spin_axis_rtn / np.linalg.norm(spin_axis_rtn)
    bulk_velocity_rtn = -bulk_speed_init * spin_axis_rtn

    unit_density_rate = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, bulk_velocity_rtn, temperature, ctx.mass_kg), ctx,
    )
    density = optimal_density_scale(unit_density_rate, ctx.count_rate)

    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=bulk_velocity_rtn,
        temperature=temperature,
        mass_kg=ctx.mass_kg,
    )


def optimal_density_scale(predicted: ndarray, observed: ndarray) -> float:
    mm = float(np.dot(predicted, predicted))
    return float(np.dot(predicted, observed)) / mm if mm > 0.0 else 1.0


def _gaussian_refine_bulk_speed_and_temperature(
    speed: ndarray,
    count_rate: ndarray,
    bulk_speed_seed: float,
    temperature_seed: float,
    mass_kg: float,
) -> tuple[float, float]:
    """Refine the bulk-speed and temperature seeds with a Gaussian fit.

    Fits A·exp(-(v - v0)² / (2σ²)) to the per-bin count rate (no per-bin
    aggregation — multiple sweeps over the same voltage just produce repeated
    samples that curve_fit treats independently). Falls back to the seed
    values if the fit fails or yields a non-positive σ.
    """
    sigma_v_seed = (
        math.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature_seed / mass_kg)
        / METERS_PER_KILOMETER
    )
    amplitude_seed = float(np.nanmax(count_rate))
    valid = np.isfinite(speed) & np.isfinite(count_rate)
    if valid.sum() < 4:
        return bulk_speed_seed, temperature_seed

    def gaussian(v, amplitude, mean, sigma):
        return amplitude * np.exp(-0.5 * ((v - mean) / sigma) ** 2)

    try:
        params, _ = scipy.optimize.curve_fit(
            gaussian, speed[valid], count_rate[valid],
            p0=[amplitude_seed, bulk_speed_seed, sigma_v_seed],
            maxfev=200,
        )
    except (RuntimeError, ValueError):
        return bulk_speed_seed, temperature_seed

    _, mean_fit, sigma_fit = params
    sigma_fit = abs(float(sigma_fit))
    if not (np.isfinite(mean_fit) and sigma_fit > 0.0):
        return bulk_speed_seed, temperature_seed
    temperature_fit = max(
        mass_kg * (sigma_fit * METERS_PER_KILOMETER) ** 2
        / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
        INITIAL_TEMPERATURE_FLOOR_K,
    )
    return float(mean_fit), float(temperature_fit)
