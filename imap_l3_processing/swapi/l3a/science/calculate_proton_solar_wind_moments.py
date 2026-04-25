import math
from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np
import scipy.optimize
import spiceypy
from numpy import ndarray

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    METERS_PER_KILOMETER,
    ONE_SECOND_IN_NANOSECONDS,
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
N_AZIMUTH_OA = 41
N_SPEED = 11

EPSILON_OA = 1e-6
EPSILON_SG = 1e-3

# Non-paralyzable detector deadtime (Tsoulfanidis 1995, p. 74): n = g / (1 - g*tau)
# Rearranged for forward model (true rate -> measured rate): g = n / (1 + n*tau)
SWAPI_DEADTIME_S = 183.7e-9


@dataclass
class ProtonSolarWindMoments:
    density: float          # cm^-3
    temperature: float      # eV
    bulk_velocity_rtn: ndarray   # shape (3,), km/s, [R, T, N]; inertial frame
    bad_fit_flag: int
    density_sigma: float = np.nan
    temperature_sigma: float = np.nan
    velocity_covariance: ndarray = None  # shape (3, 3), km^2/s^2; covariance of [vR, vT, vN]


def fit_solar_wind_proton_moments(
    count_rate: ndarray, esa_voltage: ndarray, measurement_time: ndarray,
    swapi_response: SWAPIResponse,
    sweep_coarse_count_rates: ndarray = None,
    sweep_coarse_energies: ndarray = None,
    sweep_epoch: ndarray = None,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.l3a.utils import get_rotation_matrices, get_spacecraft_velocity_rtn
    # Algorithm described in docs/swapi/solar-wind-moments.md
    # Step 1: Get RTN-to-SWAPI rotation matrices and spacecraft velocity from SPICE
    rotation_matrices = get_rotation_matrices(measurement_time)
    spacecraft_velocity_rtn = get_spacecraft_velocity_rtn(measurement_time)

    # Precompute passband grids (one per measurement) for use in model evaluation
    passband_grids = numba.typed.List([swapi_response.create_passband_grid(v) for v in esa_voltage])

    # Step 2: Initial guess — sine-fit (preferred) or Gaussian fallback
    initial_guess = _get_initial_guess(
        count_rate, esa_voltage, passband_grids, rotation_matrices, spacecraft_velocity_rtn,
        sweep_coarse_count_rates, sweep_coarse_energies, sweep_epoch,
    )

    # Step 3: Optimize solar wind parameters to best match model count rate to observed
    return _optimize(count_rate, passband_grids, rotation_matrices, spacecraft_velocity_rtn, initial_guess)


def _get_initial_guess(
    count_rate: ndarray,
    esa_voltage: ndarray,
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
    sweep_coarse_count_rates: ndarray = None,
    sweep_coarse_energies: ndarray = None,
    sweep_epoch: ndarray = None,
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

    temperature = float(
        PROTON_MASS_KG * (sigma_v * METERS_PER_KILOMETER) ** 2 / PROTON_CHARGE_COULOMBS
    )

    # Prefer sine-fit velocity direction when per-sweep coarse data is available;
    # fall back to purely anti-sunward if the sine fit fails for any reason.
    bulk_velocity_rtn = np.array([bulk_speed, 0.0, 0.0])
    if (sweep_coarse_count_rates is not None
            and sweep_coarse_energies is not None
            and sweep_epoch is not None):
        try:
            bulk_velocity_rtn = _sine_fit_velocity_rtn(
                sweep_coarse_count_rates, sweep_coarse_energies, sweep_epoch
            )
        except Exception:
            pass  # keep anti-sunward fallback

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


def _sine_fit_velocity_rtn(
    coarse_count_rates: ndarray,  # shape (n_sweeps, n_coarse_bins)
    coarse_energies: ndarray,     # shape (n_sweeps, n_coarse_bins), eV
    sweep_epoch: ndarray,         # shape (n_sweeps,), TT2000 ns
) -> ndarray:                     # shape (3,), bulk velocity in RTN km/s
    """Estimate initial RTN bulk velocity from the Rankin et al. (2025) sine-fit method.

    Fits E = A*sin(-psi + phi) + B to the ESA peak energy vs spin-phase across sweeps
    (Eq. 13 of the SWAPI instrument paper), then rotates the resulting direction vector
    from the IMAP despun frame (DPS) to RTN via SPICE.

    The transverse wind direction in DPS at clock angle phi is (cos(phi), sin(phi), 0)
    and the dominant anti-sunward component is along -Z_DPS (DPS Z points sunward).
    Deflection angle theta is estimated geometrically as arcsin(A / 2B), the first-order
    Doppler approximation (A/B ≈ 2*sin(theta) for a narrowly peaked distribution).
    """
    from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import (
        calculate_proton_centers_of_mass,
        fit_energy_per_charge_peak_variations,
    )
    from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix

    # Get ESA peak energy and corresponding spin-phase for each sweep
    energies_at_com, spin_angles_at_com = calculate_proton_centers_of_mass(
        coarse_count_rates, coarse_energies, sweep_epoch
    )

    # Sine fit: E = A*sin(-psi + phi) + B → (a, phi, b) with uncertainties
    (a_uval, phi_uval, b_uval), _ = fit_energy_per_charge_peak_variations(
        energies_at_com, spin_angles_at_com
    )
    a_val = float(a_uval.nominal_value)    # energy amplitude, eV
    phi_val = float(phi_uval.nominal_value)  # clock angle in DPS frame, degrees
    b_val = float(b_uval.nominal_value)    # mean energy, eV

    # Bulk speed from mean energy B: v = sqrt(2*E*e/m_p)
    v_b = float(
        np.sqrt(2.0 * b_val * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
    )

    # Deflection angle from sinusoidal amplitude ratio (Doppler approximation):
    # A/B ≈ 2*sin(theta)  →  theta = arcsin(A / 2B)
    deflection_rad = float(np.arcsin(np.clip(a_val / max(2.0 * b_val, 1e-10), -1.0, 1.0)))

    # Velocity direction in DPS frame.
    # The transverse component lies along (cos(phi), sin(phi), 0) in DPS X-Y;
    # the anti-sunward component is along -Z_DPS because DPS Z points sunward.
    phi_rad = np.radians(phi_val)
    v_dir_dps = np.array([
        np.sin(deflection_rad) * np.cos(phi_rad),
        np.sin(deflection_rad) * np.sin(phi_rad),
        -np.cos(deflection_rad),
    ])

    # Rotate DPS → RTN via SPICE at the median sweep epoch
    median_et = spiceypy.unitim(
        float(np.median(sweep_epoch)) / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"
    )
    R_dps_to_rtn = get_rotation_matrix(median_et, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_RTN)

    return v_b * (R_dps_to_rtn @ v_dir_dps)


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
        result[i] = apply_deadtime_correction(calculate_integral(passband_grids[i], sw_params))
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
            grid, is_sunglasses,
            elevation=0, speed=grid.central_speed
        )
        
        min_elevation, max_elevation, min_azimuth, max_azimuth = _get_angular_limits(sw_params, region, grid)
        
        # skip region if completely out of FOV
        if max_elevation <= min_elevation or max_azimuth <= min_azimuth:
            continue
        
        # TODO choose speed points dynamically for each elevation by zero-trimming;
        # must fit a linear model to the minimum and maximum point for integration

        elevation_points = np.linspace(min_elevation, max_elevation, N_ELEVATION)
        elevation_weights = _trapz_weights(min_elevation, max_elevation, N_ELEVATION)

        N_AZIMUTH = N_AZIMUTH_SG if is_sunglasses else N_AZIMUTH_OA
        azimuth_points = np.linspace(min_azimuth, max_azimuth, N_AZIMUTH)
        azimuth_weights = _trapz_weights(min_azimuth, max_azimuth, N_AZIMUTH)

        interpolated_transmission = np.array([_interpolate_transmission(grid, x)
                                              for x in azimuth_points])

        elevation_integral = 0
        for i_elevation, elevation in enumerate(elevation_points):
            sin_elevation = math.sin((math.pi / 180) * elevation)
            cos_elevation = math.cos((math.pi / 180) * elevation)

            passband_lower_speed = grid.central_speed * _eval_boundary(grid, is_sunglasses, elevation, True)
            passband_upper_speed = grid.central_speed * _eval_boundary(grid, is_sunglasses, elevation, False)
            
            min_speed, max_speed = _dynamic_limits(
                sw_params.bulk_speed,
                sw_params.thermal_speed * 5,
                passband_lower_speed,
                passband_upper_speed
            )
            speed_points = np.linspace(min_speed, max_speed, N_SPEED)
            speed_weights = _trapz_weights(min_speed, max_speed, N_SPEED)

            if max_speed <= min_speed:
                continue

            passband_times_speed3_row = np.array([
                x ** 3 * interpolate_passband(grid, is_sunglasses, elevation, x)
                for x in speed_points
            ]) / passband_norm
            
            azimuth_integral = 0
            for i_azimuth, azimuth in enumerate(azimuth_points):
                cos_angle = (
                    sin_bulk_elevation * sin_elevation
                    + cos_bulk_elevation * cos_elevation * np.cos((math.pi / 180) * (azimuth - sw_params.bulk_azimuth))
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


@numba.njit
def _trapz_weights(a, b, n):
    step = (b - a) / (n - 1)
    weights = np.full(n, step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


@numba.njit(nogil=True)
def _eval_boundary(grid: PassbandGrid, is_sunglasses, elevation: float, take_min: bool) -> float:
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
        min_elevation, max_elevation = _dynamic_limits(sw_params.bulk_elevation, angular_width, -10.5, 6.5)
        min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, -20.0, 20.0)
    else:
        min_elevation, max_elevation = _dynamic_limits(sw_params.bulk_elevation, angular_width, -12.0, 10.5)
        if region == -1:
            min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, -150.0, -20.0)
        else:
            min_azimuth, max_azimuth = _dynamic_limits(sw_params.bulk_azimuth, angular_width, 20.0, 150.0)

    return min_elevation, max_elevation, min_azimuth, max_azimuth


@numba.njit(nogil=True)
def _dynamic_limits(center: float, width: float, lower_bound: float, upper_bound: float):
    return _clamp(center - width, lower_bound, upper_bound), _clamp(center + width, lower_bound, upper_bound)


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
    return grid.azimuthal_transmission[i_lower] * weight_lower + grid.azimuthal_transmission[i_upper] * weight_upper


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


def _optimize(
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
    initial_guess: ProtonSolarWindMoments,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

    vr0, vt0, vn0 = initial_guess.bulk_velocity_rtn
    sigma = np.sqrt(np.maximum(count_rate, 1.0))

    x0 = np.array([
        np.log(initial_guess.density),
        np.log(initial_guess.temperature),
        vr0, vt0, vn0,
    ])

    def residuals(x):
        return _residuals_njit(x, count_rate, sigma, passband_grids,
                               rotation_matrices, spacecraft_velocity_rtn)

    result = scipy.optimize.least_squares(residuals, x0, method='lm',
                                          xtol=1e-4, ftol=1e-4, gtol=1e-4)

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
