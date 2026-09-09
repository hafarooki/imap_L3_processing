import math
from datetime import datetime
from typing import Iterable

import numba
import numpy as np
import scipy.optimize
from numpy import ndarray
from numpy.typing import ArrayLike
from spacepy import pycdf
from spacepy.pycdf import CDF
from uncertainties import UFloat, umath, unumpy

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    FIVE_MINUTES_IN_NANOSECONDS,
    METERS_PER_KILOMETER,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.constants import (
    SWAPI_BIN_PERIOD_S,
    SWAPI_K_FACTOR,
    SWAPI_LIVETIME_CENTER_OFFSET_S,
    SWAPI_SWEEP_BIN_COUNT,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_rotation_matrix,
    imap_state,
)
from imap_processing.spice.time import ttj2000ns_to_et

from imap_l3_processing.swapi.response.deadtime import deadtime_factor


def calculate_sw_speed(particle_mass, particle_charge, energy):
    speed_squared = 2 * energy * particle_charge / particle_mass
    if isinstance(energy, UFloat):
        return umath.sqrt(speed_squared) / METERS_PER_KILOMETER
    if isinstance(energy, np.ndarray) and energy.dtype == object:
        return unumpy.sqrt(speed_squared) / METERS_PER_KILOMETER
    return np.sqrt(speed_squared) / METERS_PER_KILOMETER


def esa_voltage_to_proton_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * PROTON_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def esa_voltage_to_alpha_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * ALPHA_PARTICLE_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / ALPHA_PARTICLE_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


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
    et_times = ttj2000ns_to_et(np.atleast_1d(measurement_time))
    return get_rotation_matrix(et_times, SpiceFrame.IMAP_SWAPI, SpiceFrame.IMAP_RTN)


def get_spacecraft_velocity_rtn(epoch_tt2000_ns: float) -> ndarray:
    et = float(ttj2000ns_to_et(epoch_tt2000_ns))
    state_eclipj2000 = imap_state(et, SpiceFrame.ECLIPJ2000)
    rtn_from_eclipj2000 = get_rotation_matrix(
        et, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN
    )
    return np.einsum("ij,j->i", rtn_from_eclipj2000, state_eclipj2000[3:])


@numba.njit
def velocity_components_to_angles_in_instrument_frame(vx: float, vy: float, vz: float):
    """Convert a Cartesian flow-direction velocity to (azimuth_deg, elevation_deg)
    look angles in the SWAPI instrument frame.

    Sign convention: the Cartesian input (vx, vy, vz) is the *flow direction* —
    the direction the particles are moving. The returned spherical angles are
    the *look direction* — the direction SWAPI points to see those particles,
    i.e. opposite the flow. The leading minus signs perform that flow→look
    inversion.

    Inverts the SWAPI v̂ convention v̂ = (−cosθ sinφ, −cosθ cosφ, −sinθ).
    """
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    return (
        math.degrees(math.atan2(-vx, -vy)),
        math.degrees(math.asin(-vz / speed)),
    )


@numba.njit
def velocity_to_angles_in_instrument_frame(
    sw_params: SolarWindParams, rotation_xyz_to_rtn
):
    """SolarWindParams bulk flow velocity → (azimuth_deg, elevation_deg) look
    angles in the SWAPI instrument frame.

    `sw_params.velocity_rtn` is the bulk flow direction in RTN. It is
    rotated into the SWAPI XYZ frame and converted by
    `velocity_components_to_angles_in_instrument_frame` — see that function for the
    flow-vs-look sign convention.
    """
    v_xyz = rotation_xyz_to_rtn.T @ sw_params.velocity_rtn
    return velocity_components_to_angles_in_instrument_frame(v_xyz[0], v_xyz[1], v_xyz[2])


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


def solar_wind_chunk_epoch(chunk: SwapiL2Data) -> float:
    """Center of a one-minute (five-sweep) chunk [ns since J2000 TT]."""
    return int(chunk.sci_start_time[0]) + THIRTY_SECONDS_IN_NANOSECONDS


def pickup_ion_chunk_epoch(chunk: SwapiL2Data) -> int:
    """Center of a ten-minute (fifty-sweep) PUI chunk [ns since J2000 TT].

    `sci_start_time` marks when a sweep starts and the chunk runs one sweep past
    its last start, so the center is half the ten minutes after the first start
    -- matching the +/- 5 minute `epoch_delta` the chunk is reported with.
    """
    return int(chunk.sci_start_time[0]) + FIVE_MINUTES_IN_NANOSECONDS


def measurement_times(sweep_start_times_tt2000_ns: ndarray) -> ndarray:
    """
    Central measurement time of each ESA step within each sweep times.

    Parameters
    ----------
    sweep_start_times_tt2000_ns : (N,) ndarray of ints [ns]
        The start TT2000 time of each sweep.

    Returns
    -------
    (N, 72) array of the measurement times for each ESA step within each sweep.
    """
    bins = np.arange(SWAPI_SWEEP_BIN_COUNT)
    seconds_into_sweep = bins * SWAPI_BIN_PERIOD_S + SWAPI_LIVETIME_CENTER_OFFSET_S
    return (
        np.asarray(sweep_start_times_tt2000_ns)[:, np.newaxis]
        + seconds_into_sweep * ONE_SECOND_IN_NANOSECONDS
    )


def optimal_density_scale(unit_ideal_rates: ndarray, observed_rates: ndarray) -> float:
    def predicted_observed_rate(unit_rate, density):
        true_rate = density * unit_rate
        return true_rate * deadtime_factor(true_rate)

    popt, _ = scipy.optimize.curve_fit(
        f=predicted_observed_rate, xdata=unit_ideal_rates, ydata=observed_rates, p0=[1]
    )
    return float(popt[0])
