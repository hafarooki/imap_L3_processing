from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spiceypy
from numpy import ndarray

from imap_l3_processing.constants import ONE_AU_IN_KM, CENTIMETERS_PER_METER, METERS_PER_KILOMETER
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import convert_velocity_relative_to_imap
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector

from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


@dataclass
class FittingParameters:
    ionization_rate: float
    cutoff_speed: float
    flags: int = SwapiL3Flags.NONE


@dataclass
class VasyliunasSiscoeDistribution:
    ephemeris_time: float
    solar_wind_speed_inertial_frame: float
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    distance_km: float
    psi: float

    def f(self, pickup_ion_speed, fitting_params: FittingParameters, apply_cutoff: bool = True):
        w = pickup_ion_speed / fitting_params.cutoff_speed
        radius_in_au = self.distance_km / ONE_AU_IN_KM
        neutral_helium_density_per_cm3 = (
            self.density_of_neutral_helium_lookup_table.density(
                self.psi, radius_in_au * w**SWAPI_PUI_COOLING_INDEX
            )
        )
        neutral_helium_density_per_km3 = (
            neutral_helium_density_per_cm3
            * (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        )
        term1 = SWAPI_PUI_COOLING_INDEX / (4 * np.pi)
        term2 = (fitting_params.ionization_rate * ONE_AU_IN_KM**2) / (
            self.distance_km
            * self.solar_wind_speed_inertial_frame
            * fitting_params.cutoff_speed**3
        )
        term3 = w ** (SWAPI_PUI_COOLING_INDEX - 3)
        term4 = neutral_helium_density_per_km3
        distribution = term1 * term2 * term3 * term4
        if apply_cutoff:
            distribution = distribution * np.heaviside(1 - w, 0.5)
        return distribution


def build_vasyliunas_siscoe_distribution(
    ephemeris_time: float,
    solar_wind_vector_rtn_kms: ndarray,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
    helium_inflow_vector: InflowVector,
) -> VasyliunasSiscoeDistribution:
    solar_wind_vector_eclipj2000_frame = convert_velocity_relative_to_imap(
        solar_wind_vector_rtn_kms, ephemeris_time, "IMAP_RTN", "ECLIPJ2000"
    )
    imap_position_eclip2000_frame_state = spiceypy.spkezr(
        "IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN"
    )[0][0:3]
    distance_km, longitude, latitude = spiceypy.reclat(
        imap_position_eclip2000_frame_state
    )
    psi = np.rad2deg(longitude) - helium_inflow_vector.longitude_deg_eclipj2000

    return VasyliunasSiscoeDistribution(
        ephemeris_time,
        np.linalg.norm(solar_wind_vector_eclipj2000_frame),
        density_of_neutral_helium_lookup_table,
        distance_km,
        psi,
    )
