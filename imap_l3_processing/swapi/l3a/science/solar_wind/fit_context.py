from typing import NamedTuple, Self

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species


class SolarWindFitContext(NamedTuple):
    count_rate: ndarray
    esa_voltage: ndarray
    response_grids: numba.typed.List
    rotation_matrices: ndarray
    mass_kg: float

    def subset(self, bin_indices: ndarray) -> Self:
        n_sweeps, n_bins = self.count_rate.shape
        flat_indices = np.concatenate(
            [bin_indices + s * n_bins for s in range(n_sweeps)]
        )
        return self._replace(
            count_rate=self.count_rate[:, bin_indices],
            esa_voltage=self.esa_voltage[:, bin_indices],
            response_grids=numba.typed.List(
                [self.response_grids[i] for i in flat_indices]
            ),
            rotation_matrices=self.rotation_matrices[flat_indices],
        )


def build_solar_wind_fit_context(
    count_rate: ndarray,
    esa_voltage: ndarray,
    swapi_response: SwapiResponse,
    time_as_tt2000: int,
    species: Species,
    rotation_matrices: ndarray,
) -> SolarWindFitContext:
    flat_voltage = esa_voltage.ravel()
    if not np.all((flat_voltage > 0) & np.isfinite(flat_voltage)):
        raise ValueError(
            "esa_voltage must be strictly positive and finite at every bin. "
            "Filter invalid voltages (and the matching count_rate / "
            "rotation_matrices entries) at the call site before building the "
            "context."
        )

    response_grids = numba.typed.List(
        [
            swapi_response.get_response_grid(time_as_tt2000, v, species)
            for v in flat_voltage
        ]
    )

    return SolarWindFitContext(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        response_grids=response_grids,
        rotation_matrices=rotation_matrices,
        mass_kg=species.mass_kg,
    )
