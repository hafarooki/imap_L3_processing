from __future__ import annotations

import numpy as np
import spiceypy
from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix
from imap_processing.spice.time import ttj2000ns_to_et
from numpy import ndarray

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.utils import measurement_times
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


def convert_velocity_to_reference_frame(
    velocity: ndarray, ephemeris_time: float, from_frame: str, to_frame: str
) -> ndarray:
    rotation_matrix = spiceypy.sxform(from_frame, to_frame, ephemeris_time)

    state = velocity[..., np.newaxis]

    state_in_target_frame = np.matmul(rotation_matrix[3:6, 3:6], state)
    return state_in_target_frame[..., 0]


def convert_velocity_relative_to_imap(velocity, ephemeris_time, from_frame, to_frame):
    velocity_in_target_frame_relative_to_imap = convert_velocity_to_reference_frame(
        velocity, ephemeris_time, from_frame, to_frame
    )
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, to_frame, "NONE", "SUN")[0][
        3:6
    ]

    return velocity_in_target_frame_relative_to_imap + imap_velocity


def calculate_ten_minute_velocities(
    bulk_solar_wind_velocities_rtn: ndarray,
    quality_flags: list[SwapiL3Flags],
) -> (ndarray, ndarray):
    """Average the per-1-minute bulk SW velocity vectors (in IMAP_RTN)
    over consecutive 10-minute windows. The corresponding 10-minute quality
    flag is the bitwise-OR of the per-minute flags."""
    left_slice = 0
    chunked_velocities = []
    chunked_quality_flags = []
    while left_slice < len(bulk_solar_wind_velocities_rtn):
        ten_min_slice = slice(left_slice, left_slice + 10)
        ten_min_quality_flag = np.bitwise_or.reduce(quality_flags[ten_min_slice])

        chunked_velocities.append(
            np.mean(bulk_solar_wind_velocities_rtn[ten_min_slice], axis=0)
        )
        chunked_quality_flags.append(ten_min_quality_flag)

        left_slice += 10

    return np.array(chunked_velocities), np.array(chunked_quality_flags)


def rotate_rtn_velocity_to_swapi_per_bin(
    chunk: SwapiL2Data,
    sw_velocity_rtn_kms: ndarray,
) -> ndarray:
    """Apply the IMAP_RTN → IMAP_SWAPI rotation at every coarse-bin
    measurement time in the PUI chunk and return the per-bin bulk SW velocity
    in SWAPI XYZ. Output shape: (n_sweeps, n_coarse_bins, 3).

    Spacecraft spin makes the SWAPI frame orientation sweep-by-sweep and
    bin-by-bin, so the per-bin rotation captures the spin phase that the
    instrument's effective area depends on.
    """
    measurement_times_tt2000_ns = measurement_times(chunk, SWAPI_COARSE_SWEEP_BINS)
    n_sweeps = chunk.sci_start_time.shape[0]
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    ephemeris_times = ttj2000ns_to_et(measurement_times_tt2000_ns)
    rotation_matrices = get_rotation_matrix(
        ephemeris_times, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI
    ).reshape(n_sweeps, n_coarse_bins, 3, 3)
    return np.einsum(
        "swij,j->swi", rotation_matrices, np.asarray(sw_velocity_rtn_kms, dtype=float)
    )
