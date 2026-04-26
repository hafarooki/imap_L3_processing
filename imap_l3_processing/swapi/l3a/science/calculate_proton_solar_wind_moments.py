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

from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import (
    PassbandGrid,
    SWAPIResponse,
)


class SWParams(NamedTuple):
    density: float
    bulk_speed: float
    bulk_azimuth: float
    bulk_elevation: float
    thermal_speed: float


N_ELEVATION = 21
N_AZIMUTH_SG = 21
N_AZIMUTH_OA = 41
N_SPEED = 11

# Gauss-Legendre quadrature nodes and weights on the standard interval [-1, 1].
# Precomputed once at module load and rescaled to each integration window inside
# `calculate_integral`. Read as module-level globals from inside the JIT region.
_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(
    N_ELEVATION
)
_GL_NODES_AZIMUTH_SG, _GL_WEIGHTS_AZIMUTH_SG = np.polynomial.legendre.leggauss(
    N_AZIMUTH_SG
)
_GL_NODES_AZIMUTH_OA, _GL_WEIGHTS_AZIMUTH_OA = np.polynomial.legendre.leggauss(
    N_AZIMUTH_OA
)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(N_SPEED)

EPSILON_OA = 1e-6
EPSILON_SG = 1e-6

# Non-paralyzable detector deadtime (Tsoulfanidis 1995, p. 74): n = g / (1 - g*tau)
# Rearranged for forward model (true rate -> measured rate): g = n / (1 + n*tau)
SWAPI_DEADTIME_S = 183.7e-9

# Duration of one ESA energy step measurement; converts count rate to counts for
# Poisson sigma estimation: sigma(rate) = sqrt(max(rate * T, 1)) / T
SWAPI_LIVETIME_S = 0.145

# Floor for the initial-guess temperature, applied when the fitted spectral width
# is below the value implied by this floor.
INITIAL_TEMPERATURE_FLOOR_EV = 1.0


@dataclass
class ProtonSolarWindMoments:
    density: float  # cm^-3
    temperature: float  # eV
    bulk_velocity_rtn: ndarray  # shape (3,), km/s, [R, T, N]; inertial frame
    bad_fit_flag: int
    density_sigma: float = np.nan
    temperature_sigma: float = np.nan
    velocity_covariance: ndarray = (
        None  # shape (3, 3), km^2/s^2; covariance of [vR, vT, vN]
    )


def fit_solar_wind_proton_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    measurement_time: ndarray,
    swapi_response: SWAPIResponse,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

    # Algorithm described in docs/swapi/solar-wind-moments.md
    # Step 1: Get RTN-to-SWAPI rotation matrices and spacecraft velocity from SPICE
    rotation_matrices, spacecraft_velocity_rtn = get_swapi_geometry(measurement_time)

    # Precompute passband grids (one per measurement) for use in model evaluation
    passband_grids = numba.typed.List(
        [swapi_response.create_passband_grid(v) for v in esa_voltage]
    )

    # Step 2: Initial guess — Gaussian fit to count rate vs speed, anti-sunward velocity
    initial_guess = _get_initial_guess(
        count_rate,
        esa_voltage,
        passband_grids,
        rotation_matrices,
        spacecraft_velocity_rtn,
    )

    # Step 3: Optimize solar wind parameters to best match model count rate to observed
    return _optimize(
        count_rate,
        passband_grids,
        rotation_matrices,
        spacecraft_velocity_rtn,
        initial_guess,
    )


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
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError:
        bulk_speed = speed[peak_idx]
        sigma_v = 50.0

    sigma_floor_v = (
        math.sqrt(
            INITIAL_TEMPERATURE_FLOOR_EV * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )
    sigma_thermal_v = max(sigma_v, sigma_floor_v)
    temperature = float(
        PROTON_MASS_KG
        * (sigma_thermal_v * METERS_PER_KILOMETER) ** 2
        / PROTON_CHARGE_COULOMBS
    )

    # Initial transverse velocity is zero; `_optimize` handles the wrong-basin trap.
    bulk_velocity_rtn = np.array([float(bulk_speed), 0.0, 0.0])

    # Scale density so that the unit model count rate matches the mean observed count rate
    unit_model = _model_count_rates(
        1.0,
        temperature,
        bulk_velocity_rtn,
        passband_grids,
        rotation_matrices,
        spacecraft_velocity_rtn,
    )
    density = float(np.nanmean(count_rate) / np.nanmean(unit_model))

    return ProtonSolarWindMoments(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=0,
    )


@numba.njit(nogil=True)
def _compute_angles(
    bulk_velocity_rtn: ndarray,
    rotation_matrix: ndarray,
    spacecraft_velocity_rtn: ndarray,
):
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
    thermal_speed = (
        np.sqrt(temperature * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )
    bulk_speed = np.linalg.norm(bulk_velocity_rtn)
    n = len(passband_grids)
    result = np.empty(n)
    for i in range(n):
        phi, theta = _compute_angles(
            bulk_velocity_rtn, rotation_matrices[i], spacecraft_velocity_rtn
        )
        sw_params = SWParams(
            density=density,
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            thermal_speed=thermal_speed,
        )
        result[i] = apply_deadtime_correction(
            calculate_integral(passband_grids[i], sw_params)
        )
    return result


@numba.njit(nogil=True)
def apply_deadtime_correction(true_rate: float) -> float:
    return true_rate / (1.0 + SWAPI_DEADTIME_S * true_rate)


@numba.njit(fastmath=True, nogil=True)
def calculate_integral(grid: PassbandGrid, sw_params: SWParams):
    sin_bulk_elevation = math.sin((math.pi / 180) * sw_params.bulk_elevation)
    cos_bulk_elevation = math.cos((math.pi / 180) * sw_params.bulk_elevation)

    count_rate = 0.0

    for region in (0, -1, +1):
        is_sunglasses = region == 0
        passband_norm = interpolate_passband(
            grid, is_sunglasses, elevation=0, speed=grid.central_speed
        )

        min_elevation, max_elevation, min_azimuth, max_azimuth = _get_angular_limits(
            sw_params, region, grid
        )

        # skip region if completely out of FOV
        if max_elevation <= min_elevation or max_azimuth <= min_azimuth:
            continue

        # TODO choose speed points dynamically for each elevation by zero-trimming;
        # must fit a linear model to the minimum and maximum point for integration

        # Gauss-Legendre points/weights, transformed from [-1, 1] to the integration window.
        half_el = 0.5 * (max_elevation - min_elevation)
        mid_el = 0.5 * (max_elevation + min_elevation)
        elevation_points = mid_el + half_el * _GL_NODES_ELEVATION
        elevation_weights = half_el * _GL_WEIGHTS_ELEVATION

        if is_sunglasses:
            az_nodes = _GL_NODES_AZIMUTH_SG
            az_weights = _GL_WEIGHTS_AZIMUTH_SG
        else:
            az_nodes = _GL_NODES_AZIMUTH_OA
            az_weights = _GL_WEIGHTS_AZIMUTH_OA
        half_az = 0.5 * (max_azimuth - min_azimuth)
        mid_az = 0.5 * (max_azimuth + min_azimuth)
        azimuth_points = mid_az + half_az * az_nodes
        azimuth_weights = half_az * az_weights

        interpolated_transmission = np.array(
            [_interpolate_transmission(grid, x) for x in azimuth_points]
        )

        elevation_integral = 0
        for i_elevation, elevation in enumerate(elevation_points):
            sin_elevation = math.sin((math.pi / 180) * elevation)
            cos_elevation = math.cos((math.pi / 180) * elevation)

            passband_lower_speed = grid.central_speed * _eval_boundary(
                grid, is_sunglasses, elevation, True
            )
            passband_upper_speed = grid.central_speed * _eval_boundary(
                grid, is_sunglasses, elevation, False
            )

            min_speed, max_speed = _dynamic_limits(
                sw_params.bulk_speed,
                sw_params.thermal_speed * 10,
                passband_lower_speed,
                passband_upper_speed,
            )
            if max_speed <= min_speed:
                continue

            half_sp = 0.5 * (max_speed - min_speed)
            mid_sp = 0.5 * (max_speed + min_speed)
            speed_points = mid_sp + half_sp * _GL_NODES_SPEED
            speed_weights = half_sp * _GL_WEIGHTS_SPEED

            passband_times_speed3_row = (
                np.array(
                    [
                        x**3 * interpolate_passband(grid, is_sunglasses, elevation, x)
                        for x in speed_points
                    ]
                )
                / passband_norm
            )

            azimuth_integral = 0
            for i_azimuth, azimuth in enumerate(azimuth_points):
                cos_angle = (
                    sin_bulk_elevation * sin_elevation
                    + cos_bulk_elevation
                    * cos_elevation
                    * np.cos((math.pi / 180) * (azimuth - sw_params.bulk_azimuth))
                )

                speed_integral = 0.0
                for i_speed, speed in enumerate(speed_points):
                    speed_integral += (
                        speed_weights[i_speed]
                        * passband_times_speed3_row[i_speed]
                        * _exponential_term(sw_params, cos_angle, speed)
                    )

                azimuth_integral += (
                    azimuth_weights[i_azimuth]
                    * interpolated_transmission[i_azimuth]
                    * speed_integral
                )

            elevation_integral += (
                elevation_weights[i_elevation] * cos_elevation * azimuth_integral
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
def _eval_boundary(
    grid: PassbandGrid, is_sunglasses, elevation: float, take_min: bool
) -> float:
    """Evaluate a `min_*_boundary` (take_min=True) or `max_*_boundary` (take_min=False)
    at `elevation`, choosing the more expansive of the two adjacent stored grid points
    so the returned interval brackets the nonzero passband region."""

    if is_sunglasses:
        boundary = grid.min_SG_boundary if take_min else grid.max_SG_boundary
    else:
        boundary = grid.min_OA_boundary if take_min else grid.max_OA_boundary
    elevs = boundary[0]
    vals = boundary[1]
    n = elevs.shape[0]
    idx = 0
    for i in range(n):
        if elevs[i] <= elevation:
            idx = i
        else:
            break
    idx_next = idx + 1 if idx + 1 < n else n - 1
    a = vals[idx]
    b = vals[idx_next]
    if take_min:
        return a if a < b else b
    return a if a > b else b


@numba.njit(nogil=True)
def _exponential_term(sw_params: SWParams, cos_angle: float, speed: float) -> float:
    return math.exp(
        -(
            speed**2
            + sw_params.bulk_speed**2
            - 2 * speed * sw_params.bulk_speed * cos_angle
        )
        / (2 * sw_params.thermal_speed**2)
    )


@numba.njit(nogil=True)
def _get_angular_limits(sw_params: SWParams, region: int, grid: PassbandGrid):
    epsilon = EPSILON_SG if region == 0 else EPSILON_OA
    angular_width = (180 / np.pi) * np.arccos(
        _clamp(
            sw_params.thermal_speed**2
            * np.log(epsilon)
            / (grid.central_speed * sw_params.bulk_speed)
            + 1,
            -1,
            1,
        )
    )

    if region == 0:
        sg_lo, sg_hi = grid.sg_active_el_range
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, sg_lo, sg_hi
        )
        min_azimuth, max_azimuth = _dynamic_limits(
            sw_params.bulk_azimuth, angular_width, -20.0, 20.0
        )
    else:
        oa_lo, oa_hi = grid.oa_active_el_range
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, oa_lo, oa_hi
        )
        if region == -1:
            min_azimuth, max_azimuth = _dynamic_limits(
                sw_params.bulk_azimuth, angular_width, -150.0, -20.0
            )
        else:
            min_azimuth, max_azimuth = _dynamic_limits(
                sw_params.bulk_azimuth, angular_width, 20.0, 150.0
            )

    return min_elevation, max_elevation, min_azimuth, max_azimuth


@numba.njit(nogil=True)
def _dynamic_limits(
    center: float, width: float, lower_bound: float, upper_bound: float
):
    return _clamp(center - width, lower_bound, upper_bound), _clamp(
        center + width, lower_bound, upper_bound
    )


@numba.njit(nogil=True)
def _clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


@numba.njit(nogil=True)
def _interpolate_transmission(grid: PassbandGrid, azimuth: float) -> float:
    azimuth = (azimuth + 180) % 360 - 180
    i_float = abs(azimuth) / grid.azimuthal_transmission_spacing
    i_lower = int(math.floor(i_float))
    i_upper = i_lower + 1

    n = len(grid.azimuthal_transmission)
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
    return (
        grid.azimuthal_transmission[i_lower] * weight_lower
        + grid.azimuthal_transmission[i_upper] * weight_upper
    )


@numba.njit(nogil=True)
def interpolate_passband(
    grid: PassbandGrid, is_sunglasses: bool, elevation: float, speed: float
) -> float:
    grid_values = grid.values_sunglasses if is_sunglasses else grid.values_open_aperture

    i_float = (elevation - grid.min_elevation) / grid.elevation_spacing
    if i_float < 0 or i_float + 1 >= grid_values.shape[0]:
        return 0.0

    j_float = (
        speed / grid.central_speed - grid.min_speed_ratio
    ) / grid.speed_ratio_spacing
    if j_float < 0 or j_float + 1 >= grid_values.shape[1]:
        return 0.0

    i_lower = int(i_float)
    i_upper = i_lower + 1
    i_weight = i_float - i_lower

    j_lower = int(j_float)
    j_upper = j_lower + 1
    j_weight = j_float - j_lower

    return (1 - i_weight) * (
        (1 - j_weight) * grid_values[i_lower, j_lower]
        + j_weight * grid_values[i_lower, j_upper]
    ) + i_weight * (
        (1 - j_weight) * grid_values[i_upper, j_lower]
        + j_weight * grid_values[i_upper, j_upper]
    )


@numba.njit(nogil=True)
def _residuals_njit(
    x, count_rate, sigma, passband_grids, rotation_matrices, spacecraft_velocity_rtn
):
    density = np.exp(x[0])
    temperature = np.exp(x[1])
    bulk_velocity_rtn = x[2:5]
    model = _model_count_rates(
        density,
        temperature,
        bulk_velocity_rtn,
        passband_grids,
        rotation_matrices,
        spacecraft_velocity_rtn,
    )
    return (model - count_rate) / sigma


def _optimize(
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
    initial_guess: ProtonSolarWindMoments,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

    vr0, vt0, vn0 = initial_guess.bulk_velocity_rtn
    sigma = np.sqrt(np.maximum(count_rate * SWAPI_LIVETIME_S, 1.0)) / SWAPI_LIVETIME_S

    x0 = np.array(
        [
            np.log(initial_guess.density),
            np.log(initial_guess.temperature),
            vr0,
            vt0,
            vn0,
        ]
    )

    def residuals(x):
        return _residuals_njit(
            x,
            count_rate,
            sigma,
            passband_grids,
            rotation_matrices,
            spacecraft_velocity_rtn,
        )

    # See docs/swapi/solar-wind-moments.md for diff_step rationale.
    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)

    # Wrong-basin detection via spin-axis mirror flip; see docs/swapi/solar-wind-moments.md.
    chi2 = float(np.sum(result.fun**2))
    x_flipped = result.x.copy()
    x_flipped[3] = -x_flipped[3]
    x_flipped[4] = -x_flipped[4]
    chi2_flipped = float(np.sum(residuals(x_flipped) ** 2))
    if chi2_flipped < chi2:
        result = scipy.optimize.least_squares(
            residuals, x_flipped, method="lm", diff_step=1e-4
        )

    density = float(np.exp(result.x[0]))
    temperature = float(np.exp(result.x[1]))
    bulk_velocity_rtn = result.x[2:5]
    bad_fit_flag = SwapiL3Flags.NONE if result.success else SwapiL3Flags.HI_CHI_SQ

    # Covariance in (log n, log T, vR, vT, vN) space via Moore-Penrose pseudoinverse
    # Assumes normalized residuals r_i = (model_i - data_i) / sigma_i are i.i.d. N(0,1)
    cov_x = np.linalg.pinv(result.jac.T @ result.jac)

    # Propagate log-space uncertainties to physical quantities
    density_sigma = float(density * np.sqrt(max(cov_x[0, 0], 0.0)))
    temperature_sigma = float(temperature * np.sqrt(max(cov_x[1, 1], 0.0)))
    velocity_covariance = cov_x[2:5, 2:5]

    return ProtonSolarWindMoments(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=bad_fit_flag,
        density_sigma=density_sigma,
        temperature_sigma=temperature_sigma,
        velocity_covariance=velocity_covariance,
    )
