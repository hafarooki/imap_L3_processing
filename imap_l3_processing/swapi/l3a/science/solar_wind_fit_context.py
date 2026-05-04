from typing import NamedTuple

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.passband_grid import PassbandGrid
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse


class SwapiResponseGrid(NamedTuple):
    """V-and-species-specific instrument response evaluated at one ESA step.

    `azimuthal_transmission` and `azimuthal_transmission_spacing` are the same
    array reference across all grids in a sweep — bundling them per-grid is
    pointer-cheap and keeps the integration call signature compact."""

    passband_grid: PassbandGrid
    central_speed: float
    central_effective_area: float
    azimuthal_transmission: ndarray
    azimuthal_transmission_spacing: float


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
            response_grids=numba.typed.List(
                [self.response_grids[i] for i in indices]
            ),
            rotation_matrices=self.rotation_matrices[indices],
        )

    @classmethod
    def from_l2_data(
        cls,
        count_rate: ndarray,
        esa_voltage: ndarray,
        swapi_response: SWAPIResponse,
        central_effective_area_scale: float,
        rotation_matrices: ndarray,
        mass_kg: float,
        mass_per_charge_m_p_per_e: float,
    ) -> "SolarWindFitContext":
        keep = (esa_voltage > 0) & np.isfinite(esa_voltage) & (count_rate > 0)
        if not np.all(keep):
            esa_voltage = esa_voltage[keep]
            count_rate = count_rate[keep]
            rotation_matrices = rotation_matrices[keep]

        az_trans = np.asarray(swapi_response.azimuthal_transmission, dtype=float)
        az_trans_spacing = float(swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
        eff_area_scale = float(central_effective_area_scale)

        response_grids = numba.typed.List(
            [
                SwapiResponseGrid(
                    passband_grid=swapi_response.create_passband_grid(v),
                    central_speed=float(
                        swapi_response.central_speed(v, mass_per_charge_m_p_per_e)
                    ),
                    central_effective_area=float(
                        swapi_response.get_central_effective_area(v)
                    ) * eff_area_scale,
                    azimuthal_transmission=az_trans,
                    azimuthal_transmission_spacing=az_trans_spacing,
                )
                for v in esa_voltage
            ]
        )

        return cls(
            count_rate=count_rate,
            esa_voltage=esa_voltage,
            response_grids=response_grids,
            rotation_matrices=rotation_matrices,
            mass_kg=float(mass_kg),
        )
