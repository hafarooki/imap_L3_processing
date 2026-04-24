import math
from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np
import scipy.optimize
from numpy import ndarray

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import esa_voltage_to_proton_speed
from imap_l3_processing.swapi.l3a.science.swapi_response import PassbandGrid, SWAPIResponse


class SWParams(NamedTuple):
    density: float
    bulk_speed: float
    bulk_azimuth: float
    bulk_elevation: float
    thermal_speed: float


N_ELEVATION = 21
N_AZIMUTH_SG = 21
N_AZIMUTH_OA = 21
N_SPEED = 21

EPSILON_OA = 1e-3
EPSILON_SG = 1e-3


@dataclass
class ProtonSolarWindMoments:
    density: float  # cm^-3
    temperature: float  # eV
    bulk_velocity_rtn: ndarray  # shape (3,), km/s, [R, T, N]; inertial frame
    bad_fit_flag: int


def fit_solar_wind_proton_moments(
    count_rate: ndarray, esa_voltage: ndarray, measurement_time: ndarray,
    swapi_response: SWAPIResponse,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.l3a.utils import get_rotation_matrices, get_spacecraft_velocity_rtn
    # Algorithm described in docs/swapi/solar-wind-moments.md
    # Step 1: Get RTN-to-SWAPI rotation matrices and spacecraft velocity from SPICE
    rotation_matrices = get_rotation_matrices(measurement_time)
    spacecraft_velocity_rtn = get_spacecraft_velocity_rtn(measurement_time)

    # Precompute passband grids (one per measurement) for use in model evaluation
    passband_grids = numba.typed.List([swapi_response.create_passband_grid(v) for v in esa_voltage])

    # Step 2: Initial guess from Gaussian fit to count rate vs speed
    initial_guess = _get_initial_guess(
        count_rate, esa_voltage, passband_grids, rotation_matrices, spacecraft_velocity_rtn
    )

    # Step 3: Optimize solar wind parameters to best match model count rate to observed
    return _optimize(count_rate, passband_grids, rotation_matrices, spacecraft_velocity_rtn, initial_guess)


def _get_initial_guess(
    count_rate: ndarray,
    esa_voltage: ndarray,
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
) -> ProtonSolarWindMoments:
    speed = esa_voltage_to_proton_speed(esa_voltage)

    peak_idx = np.nanargmax(count_rate)
    try:
        (_, bulk_speed, sigma_v), _ = scipy.optimize.curve_fit(
            lambda v, A, mu, sigma: A * np.exp(-((v - mu) ** 2) / (2 * sigma**2)),
            speed,
            count_rate,
            p0=[count_rate[peak_idx], speed[peak_idx], 50.0],
            # TODO constraints to avoid bad fits
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError:
        bulk_speed = speed[peak_idx]
        sigma_v = 50.0

    temperature = float(
        PROTON_MASS_KG * (sigma_v * METERS_PER_KILOMETER) ** 2 / PROTON_CHARGE_COULOMBS
    )

    # Assume purely anti-sunward in the inertial frame
    bulk_velocity_rtn = np.array([bulk_speed, 0.0, 0.0])

    # Scale density so that the unit model count rate matches the mean observed count rate
    unit_model = _model_count_rates(
        1.0, temperature, bulk_velocity_rtn, passband_grids, rotation_matrices, spacecraft_velocity_rtn
    )
    density = float(np.nanmean(count_rate) / np.nanmean(unit_model))

    return ProtonSolarWindMoments(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=0,
    )


@numba.njit(nogil=True)
def _compute_angles(bulk_velocity_rtn: ndarray, rotation_matrix: ndarray, spacecraft_velocity_rtn: ndarray):
    sc_frame_velocity = bulk_velocity_rtn - spacecraft_velocity_rtn
    bulk_velocity_xyz = rotation_matrix @ sc_frame_velocity
    bulk_speed = np.linalg.norm(bulk_velocity_rtn)
    phi = np.degrees(np.arctan2(-bulk_velocity_xyz[0], -bulk_velocity_xyz[1]))
    theta = np.degrees(np.arcsin(-bulk_velocity_xyz[2] / bulk_speed))
    return phi, theta


@numba.njit(nogil=True)
def _model_count_rates(
    density: float,
    temperature: float,  # eV
    bulk_velocity_rtn: ndarray,  # shape (3,), inertial RTN, km/s
    passband_grids: numba.typed.List,  # PassbandGrid per measurement, length N
    rotation_matrices: ndarray,  # shape (N, 3, 3), RTN-to-SWAPI at each measurement time
    spacecraft_velocity_rtn: ndarray,  # shape (3,), km/s
) -> ndarray:
    thermal_speed = np.sqrt(temperature * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
    bulk_speed = np.linalg.norm(bulk_velocity_rtn)
    n = len(passband_grids)
    result = np.empty(n)
    for i in range(n):
        phi, theta = _compute_angles(bulk_velocity_rtn, rotation_matrices[i], spacecraft_velocity_rtn)
        sw_params = SWParams(
            density=density,
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            thermal_speed=thermal_speed,
        )
        result[i] = calculate_integral(passband_grids[i], sw_params)
    return result


@numba.njit(nogil=True)
def _model_count_rates_time_varying(
    density_arr: ndarray,  # (N,) per-measurement density
    temperature_arr: ndarray,  # (N,) per-measurement temperature [eV]
    bulk_velocity_rtn_arr: ndarray,  # (N, 3) per-measurement bulk velocity [km/s, RTN]
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,  # (N, 3, 3)
    spacecraft_velocity_rtn: ndarray,  # (3,)
) -> ndarray:
    """Allow density/T/bulk to vary across measurements — for fitting groups of
    sweeps during non-stationary solar wind."""
    n = len(passband_grids)
    result = np.empty(n)
    for i in range(n):
        bulk_i = bulk_velocity_rtn_arr[i]
        phi, theta = _compute_angles(bulk_i, rotation_matrices[i], spacecraft_velocity_rtn)
        thermal_speed = np.sqrt(temperature_arr[i] * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
        bulk_speed = np.linalg.norm(bulk_i)
        sw_params = SWParams(
            density=density_arr[i],
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            thermal_speed=thermal_speed,
        )
        result[i] = calculate_integral(passband_grids[i], sw_params)
    return result


@numba.njit(nogil=True)
def _trapz_weights(a, b, n):
    step = (b - a) / (n - 1)
    weights = np.full(n, step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


@numba.njit(fastmath=True, nogil=True)
def calculate_integral(grid: PassbandGrid, sw_params: SWParams):
    sin_bulk_elevation = math.sin((math.pi / 180) * sw_params.bulk_elevation)
    cos_bulk_elevation = math.cos((math.pi / 180) * sw_params.bulk_elevation)

    count_rate = 0.0

    for region in (0, -1, 1):
        is_sunglasses = region == 0
        passband_norm = interpolate_passband(grid, is_sunglasses, elevation=0, speed=grid.central_speed)
        if passband_norm == 0.0:
            continue

        min_elevation, max_elevation, min_azimuth, max_azimuth = _get_angular_limits(sw_params, region, grid)

        if max_elevation <= min_elevation or max_azimuth <= min_azimuth:
            continue

        n_azimuth = N_AZIMUTH_SG if is_sunglasses else N_AZIMUTH_OA

        elevation_points = np.linspace(min_elevation, max_elevation, N_ELEVATION)
        elevation_weights = _trapz_weights(min_elevation, max_elevation, N_ELEVATION)

        azimuth_points = np.linspace(min_azimuth, max_azimuth, n_azimuth)
        azimuth_weights = _trapz_weights(min_azimuth, max_azimuth, n_azimuth)

        interpolated_transmission = np.array([
            _interpolate_transmission(grid.azimuthal_transmission, grid.azimuthal_transmission_spacing, x)
            for x in azimuth_points
        ])

        if is_sunglasses:
            min_speed_ratio_poly = grid.min_SG_poly
            max_speed_ratio_poly = grid.max_SG_poly
        else:
            min_speed_ratio_poly = grid.min_OA_poly
            max_speed_ratio_poly = grid.max_OA_poly

        elevation_integral = 0.0
        for i_elevation in range(len(elevation_points)):
            elevation = elevation_points[i_elevation]
            sin_elevation = math.sin((math.pi / 180) * elevation)
            cos_elevation = math.cos((math.pi / 180) * elevation)

            passband_lower_speed = grid.central_speed * _polyval(min_speed_ratio_poly, elevation)
            passband_upper_speed = grid.central_speed * _polyval(max_speed_ratio_poly, elevation)

            min_speed, max_speed = _dynamic_limits(
                sw_params.bulk_speed,
                sw_params.thermal_speed * 5,
                passband_lower_speed,
                passband_upper_speed,
            )

            if max_speed <= min_speed:
                continue

            speed_points = np.linspace(min_speed, max_speed, N_SPEED)
            speed_weights = _trapz_weights(min_speed, max_speed, N_SPEED)

            passband_times_speed3_row = np.array([
                x ** 3 * interpolate_passband(grid, is_sunglasses, elevation, x)
                for x in speed_points
            ]) / passband_norm

            azimuth_integral = 0.0
            for i_azimuth in range(len(azimuth_points)):
                azimuth = azimuth_points[i_azimuth]
                cos_angle = (
                    sin_bulk_elevation * sin_elevation
                    + cos_bulk_elevation * cos_elevation
                    * np.cos((math.pi / 180) * (azimuth - sw_params.bulk_azimuth))
                )

                speed_integral = 0.0
                for i_speed in range(len(speed_points)):
                    speed_integral += (
                        speed_weights[i_speed]
                        * passband_times_speed3_row[i_speed]
                        * _exponential_term(sw_params, cos_angle, speed_points[i_speed])
                    )

                azimuth_integral += (
                    azimuth_weights[i_azimuth]
                    * interpolated_transmission[i_azimuth]
                    * speed_integral
                )

            elevation_integral += (
                elevation_weights[i_elevation]
                * cos_elevation
                * azimuth_integral
            )

        count_rate += (
            elevation_integral
            * grid.central_effective_area
            * sw_params.density
            * (np.sqrt(2 * np.pi) * sw_params.thermal_speed) ** -3
            * 1e5  # km/cm/s -> 1/s
            * (math.pi / 180) ** 2  # deg^2 -> rad^2
        )

    return count_rate


@numba.njit(nogil=True)
def _polyval(coeffs, x):
    answer = 0.0
    for c in coeffs:
        answer = answer * x + c
    return answer


@numba.njit(nogil=True)
def _exponential_term(sw_params: SWParams, cos_angle: float, speed: float) -> float:
    return math.exp(
        -(speed ** 2 + sw_params.bulk_speed ** 2 - 2 * speed * sw_params.bulk_speed * cos_angle)
        / (2 * sw_params.thermal_speed ** 2)
    )


@numba.njit(nogil=True)
def _get_angular_limits(sw_params: SWParams, region: int, grid: PassbandGrid):
    epsilon = EPSILON_SG if region == 0 else EPSILON_OA
    angular_width = (180 / np.pi) * np.arccos(_clamp(
        sw_params.thermal_speed ** 2 * np.log(epsilon) / (grid.central_speed * sw_params.bulk_speed) + 1,
        -1, 1,
    ))

    if region == 0:
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, grid.min_SG_elevation, grid.max_SG_elevation)
        min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, -20, 20)
    elif region == -1:
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, grid.min_OA_elevation, grid.max_OA_elevation)
        min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, -150, -20)
    else:
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, grid.min_OA_elevation, grid.max_OA_elevation)
        min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, 20, 150)

    return min_elevation, max_elevation, min_azimuth, max_azimuth


@numba.njit(nogil=True)
def _dynamic_limits(center: float, width: float, lower_bound: float, upper_bound: float):
    return _clamp(center - width, lower_bound, upper_bound), _clamp(center + width, lower_bound, upper_bound)


@numba.njit(nogil=True)
def _clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


@numba.njit(nogil=True)
def _interpolate_transmission(azimuthal_transmission: ndarray, spacing: float, azimuth: float) -> float:
    azimuth = (azimuth + 180) % 360 - 180
    i_float = abs(azimuth) / spacing
    i_lower = int(math.floor(i_float))
    i_upper = i_lower + 1

    n = len(azimuthal_transmission)
    if i_lower < 0:
        i_lower = 0
    elif i_lower >= n:
        i_lower = n - 1
    if i_upper < 0:
        i_upper = 0
    elif i_upper >= n:
        i_upper = n - 1

    weight_lower = float(i_upper) - i_float
    weight_upper = i_float - float(i_lower)
    return azimuthal_transmission[i_lower] * weight_lower + azimuthal_transmission[i_upper] * weight_upper


@numba.njit(nogil=True)
def interpolate_passband(grid: PassbandGrid, is_sunglasses: bool, elevation: float, speed: float) -> float:
    grid_values = grid.values_sunglasses if is_sunglasses else grid.values_open_aperture

    i_float = (elevation - grid.min_elevation) / grid.elevation_spacing
    if i_float < 0 or i_float + 1 >= grid_values.shape[0]:
        return 0.0

    j_float = (speed / grid.central_speed - grid.min_speed_ratio) / grid.speed_ratio_spacing
    if j_float < 0 or j_float + 1 >= grid_values.shape[1]:
        return 0.0

    i_lower = int(i_float)
    i_upper = i_lower + 1
    i_weight = i_float - i_lower

    j_lower = int(j_float)
    j_upper = j_lower + 1
    j_weight = j_float - j_lower

    return (
        (1 - i_weight) * ((1 - j_weight) * grid_values[i_lower, j_lower] + j_weight * grid_values[i_lower, j_upper])
        + i_weight * ((1 - j_weight) * grid_values[i_upper, j_lower] + j_weight * grid_values[i_upper, j_upper])
    )


@numba.njit(nogil=True)
def _residuals_njit(x, count_rate, sigma, passband_grids,
                    rotation_matrices, spacecraft_velocity_rtn):
    density = np.exp(x[0])
    temperature = np.exp(x[1])
    bulk_velocity_rtn = x[2:5]
    model = _model_count_rates(
        density, temperature, bulk_velocity_rtn,
        passband_grids, rotation_matrices, spacecraft_velocity_rtn,
    )
    return (model - count_rate) / sigma


@numba.njit(nogil=True)
def _residuals_njit_delta(x, fractional_times, count_rate, sigma,
                          passband_grids, rotation_matrices,
                          spacecraft_velocity_rtn):
    """10-parameter residuals: 5 center + 5 linear-in-time delta.

    x[0:5] = (log n, log T, v_R, v_T, v_N) at the group center.
    x[5:10] = linear drift over the group time span (same units).
    fractional_times in [-0.5, 0.5] maps each measurement onto its delta weight.
    """
    n = len(passband_grids)
    density_arr = np.empty(n)
    temperature_arr = np.empty(n)
    bulk_arr = np.empty((n, 3))
    for i in range(n):
        ft = fractional_times[i]
        density_arr[i] = np.exp(x[0] + ft * x[5])
        temperature_arr[i] = np.exp(x[1] + ft * x[6])
        bulk_arr[i, 0] = x[2] + ft * x[7]
        bulk_arr[i, 1] = x[3] + ft * x[8]
        bulk_arr[i, 2] = x[4] + ft * x[9]
    model = _model_count_rates_time_varying(
        density_arr, temperature_arr, bulk_arr,
        passband_grids, rotation_matrices, spacecraft_velocity_rtn,
    )
    return (model - count_rate) / sigma


def _optimize(
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
    initial_guess: ProtonSolarWindMoments,
    fractional_times: ndarray | None = None,
) -> ProtonSolarWindMoments:
    """Fit 5 (or 10) parameter Maxwellian to count rates.

    If ``fractional_times`` is provided (shape ``(N,)``, values in [-0.5, 0.5]),
    density / temperature / bulk velocity are allowed to vary linearly across
    the group time span — adding 5 "delta" parameters — so that non-stationary
    solar wind within the 5-sweep window is absorbed by the drift rather than
    distorting the central moments.
    """
    from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

    vr0, vt0, vn0 = initial_guess.bulk_velocity_rtn
    sigma = np.sqrt(np.maximum(count_rate, 1.0))

    if fractional_times is None:
        x0 = np.array([
            np.log(initial_guess.density),
            np.log(initial_guess.temperature),
            vr0, vt0, vn0,
        ])

        def residuals(x):
            return _residuals_njit(x, count_rate, sigma, passband_grids,
                                   rotation_matrices, spacecraft_velocity_rtn)
    else:
        x0 = np.array([
            np.log(initial_guess.density),
            np.log(initial_guess.temperature),
            vr0, vt0, vn0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        def residuals(x):
            return _residuals_njit_delta(x, fractional_times, count_rate, sigma,
                                         passband_grids, rotation_matrices,
                                         spacecraft_velocity_rtn)

    result = scipy.optimize.least_squares(residuals, x0, method='lm',
                                           xtol=1e-4, ftol=1e-4, gtol=1e-4)

    density = float(np.exp(result.x[0]))
    temperature = float(np.exp(result.x[1]))
    bulk_velocity_rtn = result.x[2:5]
    bad_fit_flag = SwapiL3Flags.NONE if result.success else SwapiL3Flags.HI_CHI_SQ

    return ProtonSolarWindMoments(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=bad_fit_flag,
    )
