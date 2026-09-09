from __future__ import annotations

import numpy as np
from numpy import ndarray

from imap_l3_processing.constants import ONE_AU_IN_KM, CENTIMETERS_PER_METER, METERS_PER_KILOMETER
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable


def vasyliunas_siscoe_vdf(
    speed_in_sw_frame: ndarray,
    *,
    ionization_rate: float,
    cutoff_speed: float,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
    apply_cutoff: bool
):
    """
    Vasyliunas-Siscoe He+ pickup ion filled shell distribution.
    
    Parameters
    ----------
    speed_in_sw_frame : ndarray of floats [km/s]
        The speed(s) in the solar wind frame at which to evaluate the VDF.
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
    apply_cutoff: bool
        Whether to apply the Heaviside cutoff in the VDF. If not, must be applied afterwards.

    Returns
    -------
    VDF value [s^3/km^6].
    """
    
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
    distribution = term1 * term2 * term3 * term4
    if apply_cutoff:
        distribution = distribution * np.heaviside(1 - w, 0.5)
    return distribution
