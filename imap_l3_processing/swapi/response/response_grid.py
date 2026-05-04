from typing import NamedTuple

from numpy import ndarray

from imap_l3_processing.swapi.response.passband_grid import PassbandGrid


class ResponseGrid(NamedTuple):
    """V-and-species-specific instrument response evaluated at one ESA step.

    `azimuthal_transmission` and `azimuthal_transmission_spacing` are the same
    array reference across all grids in a sweep — bundling them per-grid is
    pointer-cheap and keeps the integration call signature compact."""

    passband_grid: PassbandGrid
    central_speed: float
    central_effective_area: float
    azimuthal_transmission: ndarray
    azimuthal_transmission_spacing: float
