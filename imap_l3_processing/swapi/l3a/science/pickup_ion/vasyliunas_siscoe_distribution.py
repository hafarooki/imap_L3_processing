from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.constants import ONE_AU_IN_KM, CENTIMETERS_PER_METER, METERS_PER_KILOMETER
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.pickup_ion.uniform_speed_grid import UniformSpeedGrid


def vasyliunas_siscoe_vdf(
    speed_grid: UniformSpeedGrid,
    *,
    ionization_rate: float,
    cutoff_speed: float,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> NDArray:
    """
    Vasyliunas-Siscoe He+ pickup ion filled shell distribution.

    Parameters
    ----------
    speed_grid : UniformSpeedGrid
        The grid to evaluate the VDF on.
    ionization_rate : float [1/s]
        The He+ pickup ion ionization rate (VDF parameter).
    cutoff_speed : float [km/s]
        The He+ pickup ion cutoff speed (VDF parameter).
    distance : float [km]
        The heliocentric distance to evaluate the VDF at.
    inflow_angle : float [deg]
        The angle to the interstellar neutral helium inflow vector.
    solar_wind_speed_inertial_frame : float [km/s]
        The speed of the solar wind in the Sun's reference frame.
    density_of_neutral_helium_lookup_table : DensityOfNeutralHeliumLookupTable
        The density of neutral helium lookup table, used to compute the VDF.

    Returns
    -------
    VDF value per grid cell [s^3/km^6].
    """

    distribution = _filled_shell_vdf_without_cutoff(
        speed_grid.centers,
        ionization_rate=ionization_rate,
        cutoff_speed=cutoff_speed,
        distance=distance,
        inflow_angle=inflow_angle,
        solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
        density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
    )
    _apply_partial_heaviside_at_cutoff(distribution, speed_grid, cutoff_speed)
    return distribution


def _filled_shell_vdf_without_cutoff(
    speed_in_sw_frame: NDArray,
    *,
    ionization_rate: float,
    cutoff_speed: float,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> NDArray:
    w = speed_in_sw_frame / cutoff_speed
    radius_in_au = distance / ONE_AU_IN_KM
    neutral_helium_density_per_cm3 = (
        density_of_neutral_helium_lookup_table.density(
            inflow_angle, radius_in_au * w**SWAPI_PUI_COOLING_INDEX
        )
    )
    neutral_helium_density_per_km3 = (
        neutral_helium_density_per_cm3
        * (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
    )
    term1 = SWAPI_PUI_COOLING_INDEX / (4 * np.pi)
    term2 = (ionization_rate * ONE_AU_IN_KM**2) / (
        distance
        * solar_wind_speed_inertial_frame
        * cutoff_speed**3
    )
    term3 = w ** (SWAPI_PUI_COOLING_INDEX - 3)
    term4 = neutral_helium_density_per_km3
    return term1 * term2 * term3 * term4


def _apply_partial_heaviside_at_cutoff(
    f_pui: NDArray, speed_grid: UniformSpeedGrid, cutoff_speed: float
) -> None:
    delta_v_prime = speed_grid.spacing
    cutoff_index = round((cutoff_speed - speed_grid.centers[0]) / delta_v_prime)

    if cutoff_index < 0:
        f_pui[:] = 0.0
        return

    if cutoff_index > speed_grid.size - 1:
        return

    # Scale the cutoff cell by the fraction of its width below the cutoff and
    # zero everything above. This also covers cutoff_index == size - 1 (the
    # cutoff landing on the last grid node), where the cell is still partial
    # (~half) and `f_pui[cutoff_index + 1:]` is an empty slice.
    bin_min = speed_grid.centers[cutoff_index] - delta_v_prime / 2
    fraction_below_cutoff = (cutoff_speed - bin_min) / delta_v_prime
    f_pui[cutoff_index] *= fraction_below_cutoff
    f_pui[cutoff_index + 1:] = 0.0
