from typing import NamedTuple

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.response.swapi_response import SwapiResponse


class SolarWindFitContext(NamedTuple):
    count_rate: ndarray
    esa_voltage: ndarray
    response_grids: numba.typed.List
    rotation_matrices: ndarray
    mass_kg: float

    def subset(self, indices: ndarray) -> "SolarWindFitContext":
        return self._replace(
            count_rate=self.count_rate[indices],
            esa_voltage=self.esa_voltage[indices],
            response_grids=numba.typed.List([self.response_grids[i] for i in indices]),
            rotation_matrices=self.rotation_matrices[indices],
        )


def build_solar_wind_fit_context(
    count_rate: ndarray,
    esa_voltage: ndarray,
    swapi_response: SwapiResponse,
    central_effective_area_scale: float,
    rotation_matrices: ndarray,
    mass_kg: float,
    mass_per_charge_m_p_per_e: float,
) -> SolarWindFitContext:
    keep = (esa_voltage > 0) & np.isfinite(esa_voltage)
    if not np.all(keep):
        esa_voltage = esa_voltage[keep]
        count_rate = count_rate[keep]
        rotation_matrices = rotation_matrices[keep]

    response_grids = numba.typed.List(
        [
            swapi_response.create_response_grid(
                v, mass_per_charge_m_p_per_e, central_effective_area_scale
            )
            for v in esa_voltage
        ]
    )

    return SolarWindFitContext(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        response_grids=response_grids,
        rotation_matrices=rotation_matrices,
        mass_kg=float(mass_kg),
    )


def average_spin_axis_rtn(rotation_matrices: ndarray) -> ndarray:
    """Unit-normalized mean of the per-sweep body +Y axes in RTN.

    The second row of each RTN→SWAPI rotation matrix is SWAPI's boresight
    in RTN, which is parallel to the spacecraft spin axis.
    """
    axis = rotation_matrices[:, 1, :].mean(axis=0)
    return axis / np.linalg.norm(axis)
