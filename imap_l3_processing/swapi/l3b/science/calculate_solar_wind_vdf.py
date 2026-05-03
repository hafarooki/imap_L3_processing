from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_CHARGE_COULOMBS,
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    HE_PUI_PARTICLE_MASS_KG,
    PUI_PARTICLE_CHARGE_COULOMBS,
    METERS_PER_KILOMETER,
    CENTIMETERS_PER_METER,
)
from imap_l3_processing.swapi.l3a.science.geometric_factor import (
    calculate_geometric_factor,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    calculate_sw_speed,
    SWAPI_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse


def calculate_vdf(
    particle_mass,
    particle_charge,
    energies: ndarray,
    average_count_rates: ndarray,
    eff_correction: float,
    swapi_response: SWAPIResponse,
):
    velocities = calculate_sw_speed(particle_mass, particle_charge, energies)

    q_per_e = particle_charge / PROTON_CHARGE_COULOMBS
    esa_voltages = np.asarray(energies, dtype=float) / (SWAPI_K_FACTOR * q_per_e)
    geometric_factors = np.array(
        [
            calculate_geometric_factor(swapi_response, float(v), q_per_e)
            for v in esa_voltages
        ]
    )

    proton_mass_per_charge = particle_mass * 1000 / particle_charge

    numerator = 4 * np.pi * proton_mass_per_charge * average_count_rates
    denominator = geometric_factors * eff_correction

    probabilities = numerator / denominator

    return velocities, probabilities


def calculate_proton_solar_wind_vdf(
    energies: ndarray,
    average_count_rates: ndarray,
    eff_correction: float,
    swapi_response: SWAPIResponse,
):
    return calculate_vdf(
        PROTON_MASS_KG,
        PROTON_CHARGE_COULOMBS,
        energies,
        average_count_rates,
        eff_correction,
        swapi_response,
    )


def calculate_alpha_solar_wind_vdf(
    energies: ndarray,
    average_count_rates: ndarray,
    eff_correction: float,
    swapi_response: SWAPIResponse,
):
    return calculate_vdf(
        ALPHA_PARTICLE_MASS_KG,
        ALPHA_PARTICLE_CHARGE_COULOMBS,
        energies,
        average_count_rates,
        eff_correction,
        swapi_response,
    )


def calculate_pui_solar_wind_vdf(
    energies: ndarray,
    average_count_rates: ndarray,
    eff_correction: float,
    swapi_response: SWAPIResponse,
):
    return calculate_vdf(
        HE_PUI_PARTICLE_MASS_KG,
        PUI_PARTICLE_CHARGE_COULOMBS,
        energies,
        average_count_rates,
        eff_correction,
        swapi_response,
    )


@dataclass
class DeltaMinusPlus:
    delta_minus: ndarray
    delta_plus: ndarray


def calculate_delta_minus_plus(nominal_values: ndarray) -> DeltaMinusPlus:
    ratios = nominal_values[1:] / nominal_values[:-1]
    half_ratios = np.sqrt(ratios)
    left_edges = nominal_values / [half_ratios[0], *half_ratios]
    right_edges = nominal_values * [*half_ratios, half_ratios[-1]]
    lower_bounds = np.minimum(left_edges, right_edges)
    upper_bounds = np.maximum(left_edges, right_edges)
    return DeltaMinusPlus(
        delta_minus=nominal_values - lower_bounds,
        delta_plus=upper_bounds - nominal_values,
    )
