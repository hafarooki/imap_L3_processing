from datetime import datetime
from threading import Lock
from typing import Iterable

import numpy as np
import spiceypy
from numpy import ndarray
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.models import MagL1dData
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix, imap_state

# CSPICE is not thread-safe: concurrent sxform/pxform/unitim calls corrupt the
# global trace stack and trigger SIGABRT on the next SPICE error. Serialize all
# SPICE calls across threads via this lock.
_spice_lock = Lock()


def read_l1d_mag_data(cdf_path) -> MagL1dData:
    with CDF(str(cdf_path)) as cdf:
        var = cdf["b_dsrf"]
        data = read_numeric_variable(var)[:, :3]
        attrs = var.attrs
        if "VALIDMIN" in attrs:
            data = np.where(data < float(attrs["VALIDMIN"]), np.nan, data)
        if "VALIDMAX" in attrs:
            data = np.where(data > float(attrs["VALIDMAX"]), np.nan, data)
        return MagL1dData(
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
    with _spice_lock:
        # spiceypy.unitim is scalar-only, so vectorize over the input.
        et_times = np.array(
            [
                spiceypy.unitim(float(t), "TT", "ET")
                for t in np.atleast_1d(measurement_time) / ONE_SECOND_IN_NANOSECONDS
            ]
        )
        rotation_matrices = get_rotation_matrix(
            et_times, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI
        )

    return rotation_matrices


def get_swapi_dsrf_to_rtn(measurement_time_tt2000_ns: ndarray) -> ndarray:
    """DSRF→RTN rotation at each measurement time (TT2000 ns). Returns shape (N, 3, 3).

    Mirrors the TT2000→ET conversion in get_swapi_geometry; used by the alpha moments
    fitter to rotate MAG L1D unit-B vectors (DSRF) into RTN before applying the
    field-aligned-drift constraint v_α = v_p* + Δv·B̂."""
    with _spice_lock:
        et_times = np.array(
            [
                spiceypy.unitim(float(t), "TT", "ET")
                for t in np.atleast_1d(measurement_time_tt2000_ns)
                / ONE_SECOND_IN_NANOSECONDS
            ]
        )
        return get_rotation_matrix(et_times, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_RTN)


def get_spacecraft_velocity_rtn(epoch_tt2000_ns: float) -> ndarray:
    """Return the spacecraft velocity at `epoch_tt2000_ns` (TT2000 ns) in RTN, km/s."""
    with _spice_lock:
        et = spiceypy.unitim(float(epoch_tt2000_ns) / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
        state_eclipj2000 = imap_state(et, SpiceFrame.ECLIPJ2000)
        rtn_from_eclipj2000 = get_rotation_matrix(et, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN)
    return np.einsum("ij,j->i", rtn_from_eclipj2000, state_eclipj2000[3:])


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    i = 0
    while i < len(data.sci_start_time):
        yield SwapiL2Data(
            data.sci_start_time[i : i + chunk_size],
            data.energy[i : i + chunk_size],
            data.coincidence_count_rate[i : i + chunk_size],
            data.coincidence_count_rate_uncertainty[i : i + chunk_size],
        )
        i += chunk_size
