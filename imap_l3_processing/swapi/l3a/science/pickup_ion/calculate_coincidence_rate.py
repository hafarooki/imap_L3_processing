import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    vasyliunas_siscoe_vdf,
)


def calculate_coincidence_rate(
    chunk_response: ChunkCollapsedResponse,
    *,
    ionization_rate: float,
    cutoff_speed: float,
    distance: float,
    inflow_angle: float,
    solar_wind_speed_inertial_frame: float,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> NDArray:
    f_pui = vasyliunas_siscoe_vdf(
        chunk_response.speed_grid,
        ionization_rate=ionization_rate,
        cutoff_speed=cutoff_speed,
        distance=distance,
        inflow_angle=inflow_angle,
        solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
        density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
    )

    return np.tensordot(chunk_response.bin_weights, f_pui, axes=([2], [0]))
