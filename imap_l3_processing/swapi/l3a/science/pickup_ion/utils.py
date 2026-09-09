from __future__ import annotations

import numpy as np
import spiceypy
from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix
from imap_processing.spice.time import ttj2000ns_to_et
from numpy import ndarray

from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import measurement_times


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


def rotate_rtn_velocity_to_swapi_per_bin(
    chunk: SwapiL2Data,
    sw_velocity_rtn_kms: ndarray,
) -> ndarray:
    """Apply the IMAP_RTN to IMAP_SWAPI rotation at each point in time.
    """
    measurement_times_tt2000_ns = measurement_times(chunk.sci_start_time)
    n_sweeps, n_bins = measurement_times_tt2000_ns.shape
    ephemeris_times = ttj2000ns_to_et(measurement_times_tt2000_ns.ravel())
    rotation_matrices = get_rotation_matrix(
        ephemeris_times, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI
    ).reshape(n_sweeps, n_bins, 3, 3)
    return np.einsum(
        "swij,j->swi", rotation_matrices, np.asarray(sw_velocity_rtn_kms, dtype=float)
    )
