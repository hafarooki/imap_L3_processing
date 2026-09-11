from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_CELL_COUNT = 512
_LOWEST_SPEED_AS_FRACTION_OF_MAX = 1e-3


class UniformSpeedGrid:
    """
    A uniformly spaced, ascending grid of solar wind frame speeds [km/s].

    Attributes
    ----------
    centers : ndarray of floats [km/s]
        Center speeds, ascending.
    spacing : float [km/s]
        Cell width.
    size : int
        Number of cells.
    spherical_shell_integration_weights : ndarray of floats [km^3/s^3]
        Weights ``v'^2 dv'`` for integration.
    """

    def __init__(self, max_center_speed: float) -> None:
        """
        Build the grid spanning from just above zero up to ``max_center_speed`` [km/s].
        """
        min_speed = max_center_speed * _LOWEST_SPEED_AS_FRACTION_OF_MAX
        self.centers: NDArray = np.linspace(
            min_speed, max_center_speed, _CELL_COUNT
        )
        self.spacing: float = float(self.centers[1] - self.centers[0])
        self.size: int = int(self.centers.size)
        self.spherical_shell_integration_weights: NDArray = (
            self.centers**2 * self.spacing
        )
