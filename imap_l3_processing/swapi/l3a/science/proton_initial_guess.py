import math

import numpy as np
from numpy import ndarray

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    PROTON_CHARGE_COULOMBS,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    SolarWindParams,
    model_solar_wind_coincidence_rates,
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
    bulk_speed_init = float(speed[peak_idx])

    temperature = max(
        60000.0 * (bulk_speed_init / 400.0) ** 2,
        INITIAL_TEMPERATURE_FLOOR_K,
    )

    nominal_earth_speed = -30
    derived_radial_speed = math.sqrt(
        max(bulk_speed_init ** 2 - nominal_earth_speed ** 2, 0.0)
    )

    bulk_velocity_rtn = np.array([derived_radial_speed, nominal_earth_speed, 0.0])

    unit_density_rate = model_solar_wind_coincidence_rates(
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
