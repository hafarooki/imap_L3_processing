from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.constants import (
    CENTIMETERS_PER_METER,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.utils import velocity_components_to_angles_in_instrument_frame
from imap_l3_processing.swapi.response.passband_grid import interpolate_passband
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid, SwapiResponse
from imap_l3_processing.swapi.species import Species
from imap_l3_processing.swapi.response.azimuthal_transmission import interpolate_azimuthal_transmission


class ChunkCollapsedResponse(NamedTuple):
    speed_in_sw_frame: NDArray[float]  # (N,) shared v' grid
    bin_weights: NDArray[float]        # (n_sweeps, n_steps, N); count_rate = bin_weights @ f(v')


class CollapsedResponseGrid(NamedTuple):
    speed_in_sw_frame: NDArray[float]  # (N,) v' samples
    values: NDArray[float]             # (N,) H(v', V) [km^3/s] at each v'


_ELEVATION_RESOLUTION = 32
_SPEED_RATIO_RESOLUTION = 32
_CHUNK_GRID_POINTS = 256


def build_chunk_collapsed_response(
    swapi_response: SwapiResponse,
    voltages_v: NDArray,
    bulk_sw_per_bin_kms: NDArray,
    time_as_tt2000: int,
    species: Species,
    cutoff_speed_max_kms: float,
) -> ChunkCollapsedResponse:
    """
    Input shapes:
        voltages_v: (n_steps,), ESA voltage setting [V].
        bulk_sw_per_bin_kms: (n_sweeps, n_steps, 3), bulk SW velocity vector
            (x, y, z) [km/s] in the SWAPI instrument frame.
    """
    voltages_v = np.asarray(voltages_v, dtype=float)
    bulk_sw_per_bin_kms = np.asarray(bulk_sw_per_bin_kms, dtype=float)
    n_sweeps, n_steps, _ = bulk_sw_per_bin_kms.shape
    if voltages_v.shape != (n_steps,):
        raise ValueError(
            f"voltages_v shape {voltages_v.shape} must match n_steps={n_steps}"
        )
    bulk_speeds = np.linalg.norm(bulk_sw_per_bin_kms, axis=-1)  # (n_sweeps, n_steps)

    speed_in_sw_frame = np.linspace(
        cutoff_speed_max_kms * 1e-3, cutoff_speed_max_kms, _CHUNK_GRID_POINTS
    )
    delta_v_prime = speed_in_sw_frame[1] - speed_in_sw_frame[0]
    integration_weights = speed_in_sw_frame ** 2 * delta_v_prime

    bin_weights = np.zeros((n_sweeps, n_steps, _CHUNK_GRID_POINTS))
    for sweep_index in range(n_sweeps):
        for step_index in range(n_steps):
            voltage = float(voltages_v[step_index])
            bulk_vec = bulk_sw_per_bin_kms[sweep_index, step_index]
            bulk_speed = float(bulk_speeds[sweep_index, step_index])
            bulk_azimuth_deg, bulk_elevation_deg = velocity_components_to_angles_in_instrument_frame(
                bulk_vec[0], bulk_vec[1], bulk_vec[2]
            )
            response_grid = swapi_response.get_response_grid(
                time_as_tt2000, voltage, species
            )
            collapsed = build_collapsed_response_grid(
                response_grid,
                bulk_speed,
                bulk_azimuth_deg,
                bulk_elevation_deg,
                speed_in_sw_frame=speed_in_sw_frame,
            )
            bin_weights[sweep_index, step_index, :] = (
                collapsed.values * integration_weights
            )

    return ChunkCollapsedResponse(
        speed_in_sw_frame=speed_in_sw_frame, bin_weights=bin_weights
    )


def build_collapsed_response_grid(
    response_grid: ResponseGrid,
    bulk_speed: float,
    bulk_azimuth: float,
    bulk_elevation: float,
    speed_in_sw_frame: NDArray,
) -> CollapsedResponseGrid:
    speed_in_sw_frame = np.asarray(speed_in_sw_frame, dtype=float)
    if speed_in_sw_frame.ndim != 1:
        raise ValueError(
            f"speed_in_sw_frame shape {speed_in_sw_frame.shape} must be 1D"
        )

    v_prime_min, v_prime_max = solar_wind_frame_speed_range(
        response_grid.central_speed, float(bulk_speed)
    )
    i_start = int(np.searchsorted(speed_in_sw_frame, v_prime_min, side="left"))
    i_end = int(np.searchsorted(speed_in_sw_frame, v_prime_max, side="right"))

    values = np.zeros_like(speed_in_sw_frame)
    if i_end > i_start:
        values[i_start:i_end] = _collapse_response(
            speed_in_sw_frame[i_start:i_end],
            response_grid,
            bulk_speed,
            bulk_azimuth,
            bulk_elevation,
        )

    return CollapsedResponseGrid(
        speed_in_sw_frame=speed_in_sw_frame, values=values
    )


def solar_wind_frame_speed_range(
    central_speed_kms: float, bulk_speed_kms: float
) -> tuple[float, float]:
    v_prime_min = max(
        0.0,
        0.9 * central_speed_kms - bulk_speed_kms,
        bulk_speed_kms - 1.1 * central_speed_kms,
    )
    v_prime_max = 1.1 * central_speed_kms + bulk_speed_kms
    return v_prime_min, v_prime_max


@numba.njit(fastmath=True)
def _collapse_response(
    speed_in_sw_frame: NDArray,
    response_grid: ResponseGrid,
    bulk_speed: float,
    bulk_azimuth: float,
    bulk_elevation: float,
):
    central_effective_area_km2 = (
        response_grid.central_effective_area
        / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 2
    )
    central_speed = float(response_grid.central_speed)
    cos_bulk_elevation = math.cos(math.radians(bulk_elevation))
    sin_bulk_elevation = math.sin(math.radians(bulk_elevation))

    elevation_points = np.linspace(-15, 15, _ELEVATION_RESOLUTION)
    speed_ratio_points = np.linspace(0.9, 1.1, _SPEED_RATIO_RESOLUTION)
    delta_elevation = math.radians(elevation_points[1] - elevation_points[0])
    delta_speed = central_speed * (speed_ratio_points[1] - speed_ratio_points[0])

    integral = np.zeros_like(speed_in_sw_frame)

    for elevation in elevation_points:
        cos_elevation = math.cos(math.radians(elevation))
        sin_elevation = math.sin(math.radians(elevation))
        for speed_ratio in speed_ratio_points:
            speed = central_speed * speed_ratio
            sg_passband_value = interpolate_passband(
                response_grid.sg_passband, elevation, speed_ratio
            )
            oa_passband_value = interpolate_passband(
                response_grid.oa_passband, elevation, speed_ratio
            )

            if oa_passband_value < 1e-3 and sg_passband_value < 1e-3:
                continue

            cell_coeff = (
                delta_speed * delta_elevation * speed**2 * central_effective_area_km2
                / (bulk_speed * cos_bulk_elevation)
            )

            for i in range(speed_in_sw_frame.shape[0]):
                v_prime = speed_in_sw_frame[i]
                cos_angle = (
                    (speed**2 + bulk_speed**2 - v_prime**2)
                    / (2.0 * speed * bulk_speed)
                )

                cos_delta_azimuth = (
                    (cos_angle - sin_elevation * sin_bulk_elevation)
                    / (cos_elevation * cos_bulk_elevation)
                )

                if abs(cos_delta_azimuth) >= 1.0 - 1e-5:
                    continue

                sin_delta_azimuth = math.sqrt(1.0 - cos_delta_azimuth**2)
                delta_azimuth = math.degrees(math.acos(cos_delta_azimuth))
                integrand_coeff = cell_coeff / (v_prime * sin_delta_azimuth)
                for azimuth in (
                    bulk_azimuth - delta_azimuth,
                    bulk_azimuth + delta_azimuth,
                ):
                    passband_value = (
                        oa_passband_value if abs(azimuth) > 20.0 else sg_passband_value
                    )
                    transmission = interpolate_azimuthal_transmission(
                        response_grid.azimuthal_transmission, azimuth
                    )
                    integral[i] += integrand_coeff * passband_value * transmission

    return integral
