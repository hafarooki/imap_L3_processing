from __future__ import annotations

import numpy as np
import uncertainties
from uncertainties import UFloat

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    CENTIMETERS_PER_METER,
    HE_PUI_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.uniform_speed_grid import (
    UniformSpeedGrid,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    vasyliunas_siscoe_vdf,
)


def calculate_helium_pui_density(
    speed_grid: UniformSpeedGrid,
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
    @uncertainties.wrap
    def calculate(ionization_rate, cutoff_speed):
        f_pui = vasyliunas_siscoe_vdf(
            speed_grid,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        )
        integral = float(np.sum(speed_grid.spherical_shell_integration_weights * f_pui))
        return (
            4 * np.pi * integral / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        )

    return calculate(ionization_rate, cutoff_speed)


def calculate_helium_pui_temperature(
    speed_grid: UniformSpeedGrid,
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
    integration_weights = speed_grid.spherical_shell_integration_weights

    @uncertainties.wrap
    def calculate(ionization_rate, cutoff_speed):
        f_pui = vasyliunas_siscoe_vdf(
            speed_grid,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        )
        weighted = integration_weights * f_pui
        denominator = float(np.sum(weighted))
        numerator = float(np.sum(speed_grid.centers**2 * weighted))
        return (
            HE_PUI_PARTICLE_MASS_KG
            / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
            * numerator
            / denominator
            * METERS_PER_KILOMETER**2
        )

    return calculate(ionization_rate, cutoff_speed)
