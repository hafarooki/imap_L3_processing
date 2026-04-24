from datetime import datetime
from typing import Iterable

import numpy as np
import spiceypy
from numpy import ndarray
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix, imap_state


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    sci_start_times = pycdf.lib.v_datetime_to_tt2000(
        [datetime.fromisoformat(x) for x in cdf["sci_start_time"][...]])
    return SwapiL2Data(sci_start_times,
                       read_numeric_variable(cdf["esa_energy"]),
                       read_numeric_variable(cdf["swp_coin_rate"]),
                       read_numeric_variable(cdf["swp_coin_rate_stat_uncert_plus"]))

def get_rotation_matrices(measurement_time: ndarray) -> ndarray:
    # Returns array of shape (N, 3, 3), RTN-to-SWAPI rotation at each measurement time.
    et_times = spiceypy.unitim(measurement_time / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    return get_rotation_matrix(et_times, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI)


def get_spacecraft_velocity_rtn(measurement_time: ndarray) -> ndarray:
    # Returns shape (3,), km/s — spacecraft velocity in RTN at the middle measurement time.
    middle_et = spiceypy.unitim(float(np.median(measurement_time)) / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    state_eclipj2000 = imap_state(middle_et, SpiceFrame.ECLIPJ2000)
    rtn_from_eclipj2000 = get_rotation_matrix(middle_et, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN)
    return np.einsum("ij,j->i", rtn_from_eclipj2000, state_eclipj2000[3:])


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    i = 0
    while i < len(data.sci_start_time):
        yield SwapiL2Data(
            data.sci_start_time[i:i + chunk_size],
            data.energy[i:i + chunk_size],
            data.coincidence_count_rate[i:i + chunk_size],
            data.coincidence_count_rate_uncertainty[i:i + chunk_size]
        )
        i += chunk_size
