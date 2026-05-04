import math
from typing import NamedTuple

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.response.passband_grid import PassbandGrid
from imap_l3_processing.swapi.response.response_grid import ResponseGrid


# LM state-vector layout, also dictating jacobian column order:
#   [log(density), log(temperature), v_R, v_T, v_N]
LOG_DENSITY_IDX = 0
LOG_TEMPERATURE_IDX = 1
VELOCITY_SLICE = slice(2, 5)


class SolarWindParams(NamedTuple):
    """Solar-wind state input. Bulk velocity is in RTN; per-measurement
    SWAPI-frame angles are derived per response grid inside the forward model.

    Packs/unpacks against the LM state vector defined by
    `STATE_LOG_DENSITY_IDX`, `STATE_LOG_TEMPERATURE_IDX`, `STATE_VELOCITY_SLICE`."""

    density: float
    bulk_velocity_rtn: ndarray  # shape (3,), km/s, inertial RTN
    temperature: float  # K
    mass_kg: float

    def to_state_vector(self) -> ndarray:
        state = np.empty(5)
        state[LOG_DENSITY_IDX] = math.log(self.density)
        state[LOG_TEMPERATURE_IDX] = math.log(self.temperature)
        state[VELOCITY_SLICE] = self.bulk_velocity_rtn
        return state

    @classmethod
    def from_state_vector(
        cls, state: ndarray, mass_kg: float
    ) -> "SolarWindParams":
        return cls(
            density=math.exp(state[LOG_DENSITY_IDX]),
            bulk_velocity_rtn=state[VELOCITY_SLICE],
            temperature=math.exp(state[LOG_TEMPERATURE_IDX]),
            mass_kg=mass_kg,
        )


class _LocalSWParams(NamedTuple):
    """Per-measurement projection of `SWParams` into the SWAPI instrument frame.
    Constructed inside the forward-model loop; consumed by integration helpers."""

    density: float
    bulk_speed: float
    bulk_azimuth: float
    bulk_elevation: float
    temperature: float  # K
    mass_kg: float


@numba.njit(nogil=True, inline="always")
def _speed_std(sw_params: _LocalSWParams) -> float:
    """Per-axis Maxwellian speed standard deviation σ = sqrt(kT/m), km/s."""
    return (
        math.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * sw_params.temperature
                  / sw_params.mass_kg)
        / METERS_PER_KILOMETER
    )


N_ELEVATION = 21
N_AZIMUTH = 21
N_SPEED = 15

_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(
    N_ELEVATION
)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(N_AZIMUTH)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(N_SPEED)

OA_SCAN_THRESHOLD = 1e-6

EPSILON_OA = 1e-6
EPSILON_SG = 1e-6

SPEED_HALF_WIDTH_VTH = 6.0

VV_OUTER_DEG = 26.0

SWAPI_DEADTIME_S = 183.7e-9

OA_SKIP_FRACTION = 1e-3

N_OA_SCAN = 64

_REGION_SUNGLASSES = 0
_REGION_OA_NEG = -1
_REGION_OA_POS = +1
_REGION_VV_NEG = -2
_REGION_VV_POS = +2


@numba.njit(nogil=True)
def model_solar_wind_coincidence_rates(
    sw_params: SolarWindParams,
    ctx: SolarWindFitContext,
) -> ndarray:
    bulk_speed = float(np.linalg.norm(sw_params.bulk_velocity_rtn))
    n = len(ctx.response_grids)
    result = np.empty(n)
    for i in range(n):
        phi, theta = _compute_angles(
            sw_params.bulk_velocity_rtn, ctx.rotation_matrices[i]
        )
        local_params = _LocalSWParams(
            density=sw_params.density,
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            temperature=sw_params.temperature,
            mass_kg=sw_params.mass_kg,
        )
        result[i] = calculate_integral(ctx.response_grids[i], local_params)
    return result


@numba.njit(nogil=True)
def _compute_angles(bulk_velocity_rtn: ndarray, rotation_matrix: ndarray):
    bulk_velocity_xyz = rotation_matrix @ bulk_velocity_rtn
    bulk_speed = np.linalg.norm(bulk_velocity_rtn)
    phi = np.degrees(np.arctan2(-bulk_velocity_xyz[0], -bulk_velocity_xyz[1]))
    theta = np.degrees(np.arcsin(-bulk_velocity_xyz[2] / bulk_speed))
    return phi, theta


@numba.njit(fastmath=True, nogil=True)
def calculate_integral(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
):
    sg_rate = _integrate_sunglasses_region(response_grid, sw_params)
    vv_rate = _integrate_vanes_vignetting_regions(response_grid, sw_params)
    oa_rate = _integrate_open_aperture_regions(response_grid, sw_params, sg_rate)
    return sg_rate + vv_rate + oa_rate


@numba.njit(fastmath=True, nogil=True)
def _integrate_sunglasses_region(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
) -> float:
    min_el, max_el, min_az, max_az = _integration_window(
        sw_params, _REGION_SUNGLASSES, response_grid
    )
    if max_el <= min_el or max_az <= min_az:
        return 0.0
    return _integrate_region(
        response_grid, sw_params, True, min_el, max_el, min_az, max_az
    )


@numba.njit(fastmath=True, nogil=True)
def _integrate_vanes_vignetting_regions(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
) -> float:
    total = 0.0
    for region in (_REGION_VV_NEG, _REGION_VV_POS):
        min_el, max_el, min_az, max_az = _integration_window(
            sw_params, region, response_grid
        )
        if max_el <= min_el or max_az <= min_az:
            continue
        total += _integrate_region(
            response_grid, sw_params, False, min_el, max_el, min_az, max_az
        )
    return total


@numba.njit(fastmath=True, nogil=True)
def _integrate_open_aperture_regions(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
    sg_rate: float,
) -> float:
    total = 0.0
    for region in (_REGION_OA_NEG, _REGION_OA_POS):
        min_el, max_el, min_az, max_az = _integration_window(
            sw_params, region, response_grid
        )
        if max_el <= min_el or max_az <= min_az:
            continue

        min_az, max_az, transmission_maxwellian_az_integral = (
            _trim_oa_azimuth_by_integrand(
                response_grid, sw_params, min_el, max_el, min_az, max_az
            )
        )
        if max_az <= min_az:
            continue

        oa_upper_bound = _oa_rate_upper_bound(
            response_grid,
            transmission_maxwellian_az_integral,
            min_el,
            max_el,
        )
        if oa_upper_bound < max(0.1, OA_SKIP_FRACTION * sg_rate):
            continue

        total += _integrate_region(
            response_grid, sw_params, False, min_el, max_el, min_az, max_az
        )
    return total


@numba.njit(nogil=True)
def _integration_window(
    sw_params: _LocalSWParams, region: int, response_grid: ResponseGrid
):
    grid = response_grid.passband_grid
    central_speed = response_grid.central_speed
    epsilon = EPSILON_SG if region == _REGION_SUNGLASSES else EPSILON_OA

    speed_std = _speed_std(sw_params)
    cos_angular_width = (
         speed_std**2 * np.log(epsilon)
         / (central_speed * sw_params.bulk_speed)
         + 1
    )
    cos_angular_width = _clamp(cos_angular_width, -1, +1)
    angular_width = np.degrees(np.arccos(cos_angular_width))

    if region == _REGION_SUNGLASSES:
        sg_lo, sg_hi = grid.sg_elevation_range
        min_elevation, max_elevation = _clamp_window(
            sw_params.bulk_elevation, angular_width, sg_lo, sg_hi
        )
        min_azimuth, max_azimuth = _clamp_window(
            sw_params.bulk_azimuth, angular_width, -20.0, 20.0
        )
    else:
        oa_lo, oa_hi = grid.oa_elevation_range
        min_elevation, max_elevation = _clamp_window(
            sw_params.bulk_elevation, angular_width, oa_lo, oa_hi
        )

        is_vv_band = abs(region) == 2
        inner = 20.0 if is_vv_band else VV_OUTER_DEG
        outer = VV_OUTER_DEG if is_vv_band else 150.0
        az_lo = inner if region > 0 else -outer
        az_hi = outer if region > 0 else -inner
        min_azimuth, max_azimuth = _clamp_window(
            sw_params.bulk_azimuth, angular_width, az_lo, az_hi
        )

    return min_elevation, max_elevation, min_azimuth, max_azimuth


@numba.njit(nogil=True)
def _clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


@numba.njit(nogil=True)
def _clamp_window(
    center: float, half_width: float, lower_bound: float, upper_bound: float
):
    return (
        _clamp(center - half_width, lower_bound, upper_bound),
        _clamp(center + half_width, lower_bound, upper_bound),
    )


@numba.njit(fastmath=True, nogil=True)
def _integrate_region(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
    is_sunglasses: bool,
    min_elevation: float,
    max_elevation: float,
    min_azimuth: float,
    max_azimuth: float,
):
    grid = response_grid.passband_grid
    central_speed = response_grid.central_speed
    central_effective_area = response_grid.central_effective_area
    azimuthal_transmission = response_grid.azimuthal_transmission
    azimuthal_transmission_spacing = response_grid.azimuthal_transmission_spacing

    sin_bulk_elevation = math.sin((math.pi / 180) * sw_params.bulk_elevation)
    cos_bulk_elevation = math.cos((math.pi / 180) * sw_params.bulk_elevation)

    passband_norm = interpolate_passband(
        grid, is_sunglasses, elevation=0, speed_ratio=1.0
    )

    half_el = 0.5 * (max_elevation - min_elevation)
    mid_el = 0.5 * (max_elevation + min_elevation)
    elevation_points = mid_el + half_el * _GL_NODES_ELEVATION
    elevation_weights = half_el * _GL_WEIGHTS_ELEVATION

    half_az = 0.5 * (max_azimuth - min_azimuth)
    mid_az = 0.5 * (max_azimuth + min_azimuth)
    azimuth_points = mid_az + half_az * _GL_NODES_AZIMUTH
    azimuth_weights = half_az * _GL_WEIGHTS_AZIMUTH

    interpolated_transmission = np.array(
        [
            _interpolate_transmission(
                azimuthal_transmission, azimuthal_transmission_spacing, x
            )
            for x in azimuth_points
        ]
    )

    elevation_integral = 0.0
    for i_elevation, elevation in enumerate(elevation_points):
        cos_elevation = math.cos((math.pi / 180) * elevation)

        passband_lower_speed = central_speed * _min_passband_speed_ratio_at_elevation(
            grid, is_sunglasses, elevation
        )
        passband_upper_speed = central_speed * _max_passband_speed_ratio_at_elevation(
            grid, is_sunglasses, elevation
        )

        min_speed, max_speed = _clamp_window(
            sw_params.bulk_speed,
            _speed_std(sw_params) * SPEED_HALF_WIDTH_VTH,
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
                    x**3
                    * interpolate_passband(
                        grid, is_sunglasses, elevation, x / central_speed
                    )
                    for x in speed_points
                ]
            )
            / passband_norm
        )

        sin_elevation = math.sin((math.pi / 180) * elevation)
        azimuth_integral = 0.0
        for i_azimuth, azimuth in enumerate(azimuth_points):
            cos_view_angle_to_bulk = (
                sin_bulk_elevation * sin_elevation
                + cos_bulk_elevation
                * cos_elevation
                * math.cos((math.pi / 180) * (azimuth - sw_params.bulk_azimuth))
            )

            speed_integral = 0.0
            for i_speed, speed in enumerate(speed_points):
                speed_integral += (
                    speed_weights[i_speed]
                    * passband_times_speed3_row[i_speed]
                    * _maxwellian_exponential(sw_params, cos_view_angle_to_bulk, speed)
                )

            azimuth_integral += (
                azimuth_weights[i_azimuth]
                * interpolated_transmission[i_azimuth]
                * speed_integral
            )

        elevation_integral += (
            elevation_weights[i_elevation] * cos_elevation * azimuth_integral
        )

    return (
        elevation_integral
        * (math.pi / 180) ** 2  # deg^2 -> rad^2
        * _phase_space_integral_to_count_rate_factor(sw_params, central_effective_area)
    )


@numba.njit(nogil=True)
def interpolate_passband(
    grid: PassbandGrid, is_sunglasses: bool, elevation: float, speed_ratio: float
) -> float:
    grid_values = grid.values_sunglasses if is_sunglasses else grid.values_open_aperture

    i_float = (elevation - grid.min_elevation) / grid.elevation_spacing
    if i_float < 0 or i_float + 1 >= grid_values.shape[0]:
        return 0.0

    j_float = (speed_ratio - grid.min_speed_ratio) / grid.speed_ratio_spacing
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
def _interpolate_transmission(
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    azimuth: float,
) -> float:
    azimuth = (azimuth + 180) % 360 - 180
    i_float = abs(azimuth) / azimuthal_transmission_spacing
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
    return (
        azimuthal_transmission[i_lower] * weight_lower
        + azimuthal_transmission[i_upper] * weight_upper
    )


@numba.njit(nogil=True)
def _min_passband_speed_ratio_at_elevation(
    grid: PassbandGrid, is_sunglasses, elevation: float
) -> float:
    boundary = grid.min_SG_boundary if is_sunglasses else grid.min_OA_boundary
    a, b = _bracketing_boundary_values(boundary, elevation)
    return a if a < b else b


@numba.njit(nogil=True)
def _max_passband_speed_ratio_at_elevation(
    grid: PassbandGrid, is_sunglasses, elevation: float
) -> float:
    boundary = grid.max_SG_boundary if is_sunglasses else grid.max_OA_boundary
    a, b = _bracketing_boundary_values(boundary, elevation)
    return a if a > b else b


@numba.njit(nogil=True)
def _bracketing_boundary_values(boundary, elevation: float):
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
    return vals[idx], vals[idx_next]


@numba.njit(nogil=True)
def _maxwellian_exponential(
    sw_params: _LocalSWParams, cos_view_angle_to_bulk: float, speed: float
) -> float:
    speed_std = _speed_std(sw_params)
    return math.exp(
        -(
            speed**2
            + sw_params.bulk_speed**2
            - 2 * speed * sw_params.bulk_speed * cos_view_angle_to_bulk
        )
        / (2 * speed_std**2)
    )


@numba.njit(nogil=True, inline="always")
def _phase_space_integral_to_count_rate_factor(
    sw_params: _LocalSWParams, central_effective_area: float
) -> float:
    return (
        central_effective_area
        * sw_params.density
        * (np.sqrt(2 * np.pi) * _speed_std(sw_params)) ** -3
        * 1e5
    )


@numba.njit(nogil=True)
def _trim_oa_azimuth_by_integrand(
    response_grid: ResponseGrid,
    sw_params: _LocalSWParams,
    min_elevation: float,
    max_elevation: float,
    gaussian_az_lo: float,
    gaussian_az_hi: float,
):
    central_speed = response_grid.central_speed
    azimuthal_transmission = response_grid.azimuthal_transmission
    azimuthal_transmission_spacing = response_grid.azimuthal_transmission_spacing
    if gaussian_az_hi <= gaussian_az_lo:
        return 0.0, 0.0, 0.0

    scan_azimuths = np.linspace(gaussian_az_lo, gaussian_az_hi, N_OA_SCAN)
    scan_elevation = _clamp(sw_params.bulk_elevation, min_elevation, max_elevation)

    deg2rad = math.pi / 180.0
    sin_bulk_el = math.sin(deg2rad * sw_params.bulk_elevation)
    cos_bulk_el = math.cos(deg2rad * sw_params.bulk_elevation)
    sin_scan_el = math.sin(deg2rad * scan_elevation)
    cos_scan_el = math.cos(deg2rad * scan_elevation)

    cos_delta_azimuth = np.cos(deg2rad * (scan_azimuths - sw_params.bulk_azimuth))
    cos_view_angle_to_bulk = (
        sin_bulk_el * sin_scan_el + cos_bulk_el * cos_scan_el * cos_delta_azimuth
    )
    relative_velocity_sq = (
        central_speed**2
        + sw_params.bulk_speed**2
        - 2.0 * central_speed * sw_params.bulk_speed * cos_view_angle_to_bulk
    )
    speed_std = _speed_std(sw_params)
    transmission_times_maxwellian = np.exp(
        -relative_velocity_sq / (2.0 * speed_std**2)
    )
    for i in range(N_OA_SCAN):
        transmission_times_maxwellian[i] *= _interpolate_transmission(
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            scan_azimuths[i],
        )

    threshold_value = OA_SCAN_THRESHOLD * np.max(transmission_times_maxwellian)

    lower_index = 0
    for i in range(N_OA_SCAN):
        if transmission_times_maxwellian[i] > threshold_value:
            lower_index = max(i - 1, 0)
            break

    upper_index = N_OA_SCAN - 1
    for i in range(N_OA_SCAN - 1, -1, -1):
        if transmission_times_maxwellian[i] > threshold_value:
            upper_index = min(i + 1, N_OA_SCAN - 1)
            break

    dphi_rad = (scan_azimuths[1] - scan_azimuths[0]) * deg2rad
    maxwellian_norm = (
        sw_params.density / (np.sqrt(2.0 * np.pi) * speed_std) ** 3
    )
    transmission_maxwellian_az_integral = (
        np.trapezoid(
            transmission_times_maxwellian[lower_index : upper_index + 1], dx=dphi_rad
        )
        * maxwellian_norm
    )

    return (
        scan_azimuths[lower_index],
        scan_azimuths[upper_index],
        transmission_maxwellian_az_integral,
    )


@numba.njit(nogil=True)
def _oa_rate_upper_bound(
    response_grid: ResponseGrid,
    transmission_maxwellian_az_integral: float,
    min_elevation: float,
    max_elevation: float,
) -> float:
    grid = response_grid.passband_grid
    central_speed = response_grid.central_speed
    central_effective_area = response_grid.central_effective_area
    delta_theta_rad = (math.pi / 180.0) * (max_elevation - min_elevation)
    delta_v = central_speed * (
        _max_passband_speed_ratio_at_elevation(grid, False, 0.0)
        - _min_passband_speed_ratio_at_elevation(grid, False, 0.0)
    )
    return (
        central_effective_area
        * central_speed**3
        * delta_theta_rad
        * delta_v
        * transmission_maxwellian_az_integral
        * 1e5  # km -> cm
    )


@numba.njit(nogil=True)
def apply_deadtime_correction_array(true_rates: ndarray) -> ndarray:
    return true_rates / (1.0 + SWAPI_DEADTIME_S * true_rates)


@numba.njit(nogil=True)
def apply_deadtime_correction(true_rate: float) -> float:
    return true_rate / (1.0 + SWAPI_DEADTIME_S * true_rate)


