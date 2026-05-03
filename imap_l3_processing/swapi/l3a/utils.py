from datetime import datetime
from typing import Iterable

import numpy as np
from numpy import ndarray
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS, THIRTY_SECONDS_IN_NANOSECONDS
from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    get_rotation_matrix,
    imap_state,
)
from imap_processing.spice.time import ttj2000ns_to_et


def read_mag_rtn_data(cdf_path) -> MagData:
    with CDF(str(cdf_path)) as cdf:
        var = cdf["b_rtn"]
        data = read_numeric_variable(var)[:, :3]
        attrs = var.attrs
        if "VALIDMIN" in attrs:
            data = np.where(data < float(attrs["VALIDMIN"]), np.nan, data)
        if "VALIDMAX" in attrs:
            data = np.where(data > float(attrs["VALIDMAX"]), np.nan, data)
        return MagData(
            epoch=pycdf.lib.v_datetime_to_tt2000(cdf["epoch"][...]),
            mag_data=data,
        )


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    sci_start_times = pycdf.lib.v_datetime_to_tt2000(
        [datetime.fromisoformat(x) for x in cdf["sci_start_time"][...]]
    )
    return SwapiL2Data(
        sci_start_times,
        read_numeric_variable(cdf["esa_energy"]),
        read_numeric_variable(cdf["swp_coin_rate"]),
        read_numeric_variable(cdf["swp_coin_rate_stat_uncert_plus"]),
    )


def get_swapi_geometry(measurement_time: ndarray) -> ndarray:
    """Resolve SPICE geometry for a chunk of SWAPI measurements.

    Returns:
        rotation_matrices: shape (N, 3, 3) — RTN→SWAPI rotation at each measurement time.
    """
    et_times = ttj2000ns_to_et(np.atleast_1d(measurement_time))
    return get_rotation_matrix(et_times, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI)


def rotate_rtn_to_dps(vector_rtn, epoch_tt2000_ns: float):
    """Rotate a 3-vector from IMAP_RTN into IMAP_DPS at the given TT2000 ns epoch.

    Accepts plain float or `uncertainties.UFloat` components — for object-dtype
    arrays of correlated UFloats, numpy's matmul preserves correlation tracking
    so downstream covariance propagation in `derive_velocity_angles` stays
    consistent with the prior matrix-multiplication implementation."""
    et = float(ttj2000ns_to_et(epoch_tt2000_ns))
    return frame_transform(
        et, np.asarray(vector_rtn), SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_DPS
    )


def get_spacecraft_velocity_rtn(epoch_tt2000_ns: float) -> ndarray:
    """Return the spacecraft velocity at `epoch_tt2000_ns` (TT2000 ns) in RTN, km/s.

    Uses the SPKEZR-backed `imap_state` with `IMAP_RTN` as the reference frame,
    so the returned velocity is the kinematic 6D state transform (includes the
    rotation-rate term of the dynamic RTN frame), not just the inertial velocity
    rotated into RTN axes."""
    et = float(ttj2000ns_to_et(epoch_tt2000_ns))
    return imap_state(et, SpiceFrame.IMAP_RTN)[3:]


def compute_direction_of_mean_magnetic_field_over_chunk(
    mag_data,
    chunk_epoch_center_tt2000_ns: int,
    chunk_epoch_delta_ns: int,
) -> np.ndarray:
    start = chunk_epoch_center_tt2000_ns - chunk_epoch_delta_ns
    end = chunk_epoch_center_tt2000_ns + chunk_epoch_delta_ns
    left = np.searchsorted(mag_data.epoch, start, side="left")
    right = np.searchsorted(mag_data.epoch, end, side="left")
    if right == left:
        return np.full(3, np.nan)
    b_mean = mag_data.mag_data[left:right].mean(axis=0)
    if not np.all(np.isfinite(b_mean)):
        return np.full(3, np.nan)
    return b_mean / np.linalg.norm(b_mean)


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    n = len(data.sci_start_time)
    for i in range(0, n - n % chunk_size, chunk_size):
        yield SwapiL2Data(
            data.sci_start_time[i : i + chunk_size],
            data.energy[i : i + chunk_size],
            data.coincidence_count_rate[i : i + chunk_size],
            data.coincidence_count_rate_uncertainty[i : i + chunk_size],
        )


def chunk_epoch(chunk: SwapiL2Data) -> float:
    return chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS


def measurement_times(chunk: SwapiL2Data, bin_slice: slice) -> ndarray:
    bins = np.arange(bin_slice.start, bin_slice.stop)
    return (
        chunk.sci_start_time[:, np.newaxis]
        + bins * (12 / 72 * ONE_SECOND_IN_NANOSECONDS)
    ).flatten()
