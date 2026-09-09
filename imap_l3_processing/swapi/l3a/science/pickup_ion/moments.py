from __future__ import annotations

import numpy as np
import uncertainties
from numpy.typing import NDArray
from uncertainties import UFloat

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    CENTIMETERS_PER_METER,
    HE_PUI_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    apply_partial_heaviside_at_cutoff,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    vasyliunas_siscoe_vdf,
)


def calculate_helium_pui_density(
    speed_in_sw_frame: NDArray,
    *,
    ionization_rate: UFloat,
    cutoff_speed: UFloat,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> UFloat:
    """
    Calculate He+ PUI density through integration of the Vasyliunas & Siscoe distribution.

    Uncertainty on the fit parameters is propagated through the integral.
    """
    integration_weights = _moment_integration_weights(speed_in_sw_frame)

    @uncertainties.wrap
    def calculate(ionization_rate, cutoff_speed):
        f_pui = _evaluate_vdf_with_cutoff(
            speed_in_sw_frame,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        )
        integral = float(np.sum(integration_weights * f_pui))
        return (
            4 * np.pi * integral / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        )

    return calculate(ionization_rate, cutoff_speed)


def calculate_helium_pui_temperature(
    speed_in_sw_frame: NDArray,
    *,
    ionization_rate: UFloat,
    cutoff_speed: UFloat,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> UFloat:
    """
    Calculate He+ PUI temperature from the second moment of the Vasyliunas & Siscoe distribution.

    Uncertainty on the fit parameters is propagated through the moments.
    """
    integration_weights = _moment_integration_weights(speed_in_sw_frame)

    @uncertainties.wrap
    def calculate(ionization_rate, cutoff_speed):
        f_pui = _evaluate_vdf_with_cutoff(
            speed_in_sw_frame,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        )
        weighted = integration_weights * f_pui
        denominator = float(np.sum(weighted))
        numerator = float(np.sum(speed_in_sw_frame * speed_in_sw_frame * weighted))
        return (
            HE_PUI_PARTICLE_MASS_KG
            / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
            * numerator
            / denominator
            * METERS_PER_KILOMETER**2
        )

    return calculate(ionization_rate, cutoff_speed)


def _evaluate_vdf_with_cutoff(
    speed_in_sw_frame: NDArray, *, cutoff_speed: float, **vdf_kwargs
) -> NDArray:
    f_pui = np.asarray(
        vasyliunas_siscoe_vdf(
            speed_in_sw_frame,
            cutoff_speed=cutoff_speed,
            apply_cutoff=False,
            **vdf_kwargs,
        ),
        dtype=float,
    ).copy()
    apply_partial_heaviside_at_cutoff(f_pui, speed_in_sw_frame, cutoff_speed)
    return f_pui


def _moment_integration_weights(speed_in_sw_frame: NDArray) -> NDArray:
    delta_v_prime = speed_in_sw_frame[1] - speed_in_sw_frame[0]
    return speed_in_sw_frame**2 * delta_v_prime
