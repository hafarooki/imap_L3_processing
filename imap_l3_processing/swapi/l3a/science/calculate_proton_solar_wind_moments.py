import math
from dataclasses import dataclass, field
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
N_AZIMUTH_OA = 21
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

# OA azimuth integration: a scan of `density × transmission` at the in-passband
# elevation peak across the full OA range finds where the integrand is non-
# negligible. Then the fixed-N GL above runs over that trimmed range. The
# previous gaussian-only `angular_width` was wasteful because OA transmission is
# essentially zero from 20°–25° — the typical proton at |bulk_az| < 6° opened a
# 1° OA sliver entirely in the dead zone. See `_trim_oa_azimuth_by_integrand`.
OA_SCAN_THRESHOLD = 1e-3  # trim where g < threshold × max(g)

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
    velocity_covariance: ndarray = field(
        default_factory=lambda: np.full((3, 3), np.nan)
    )  # shape (3, 3), km^2/s^2; covariance of [vR, vT, vN]


def fit_solar_wind_proton_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    measurement_time: ndarray,
    swapi_response: SWAPIResponse,
    central_effective_area_scale: float = 1.0,
    rotation_matrices: ndarray = None,
    spacecraft_velocity_rtn: ndarray = None,
) -> ProtonSolarWindMoments:
    """Fit proton solar wind moments. ``central_effective_area_scale`` should be
    ``ε_p(t)/ε_p(t_lab)`` from the efficiency LUT — it's applied to each measurement's
    lab-derived central effective area before integration.

    ``rotation_matrices`` and ``spacecraft_velocity_rtn`` may be precomputed and reused
    across stage 1/stage 2 fits to avoid duplicate SPICE calls; if ``None``, the function
    computes them internally from ``measurement_time``."""
    from imap_l3_processing.constants import PROTON_MASS_PER_CHARGE_M_P_PER_E
    from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

    # Algorithm described in docs/swapi/solar-wind-moments.md
    # Step 1: Get RTN-to-SWAPI rotation matrices and spacecraft velocity from SPICE
    if rotation_matrices is None or spacecraft_velocity_rtn is None:
        rotation_matrices, spacecraft_velocity_rtn = get_swapi_geometry(
            measurement_time
        )

    # Spin axis (body +Y in RTN) for the wrong-basin flip check in _optimize.
    # Captured here, before the half-mean mask below may drop the bin at index 0.
    spin_axis_rtn = rotation_matrices[0, 1, :].copy()

    # Drop any 0V (or non-finite) ESA steps. Some sweeps include a zero-energy
    # step that carries no useful information and would make central_speed = 0,
    # producing divide-by-zero deep inside the JIT integrator.
    keep = (esa_voltage > 0) & np.isfinite(esa_voltage)

    # FWHM mask: keep only bins at or above half the peak count rate.  Bins in
    # the deep tails carry almost no proton signal but add noise to the fit.
    cr_max = float(np.nanmax(count_rate[keep])) if np.any(keep) else 0.0
    fwhm_mask = count_rate >= 0.5 * cr_max
    if int((keep & fwhm_mask).sum()) >= 5:
        keep = keep & fwhm_mask

    if not np.all(keep):
        esa_voltage = esa_voltage[keep]
        count_rate = count_rate[keep]
        rotation_matrices = rotation_matrices[keep]
        if measurement_time is not None:
            measurement_time = np.asarray(measurement_time)[keep]

    # V-only passband grids (cached by V across calls), plus per-measurement species/V-
    # dependent scalars for v_0 and lab-derived central effective area times the time scale.
    passband_grids = numba.typed.List(
        [swapi_response.create_passband_grid(v) for v in esa_voltage]
    )
    central_speeds = np.array(
        [
            swapi_response.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E)
            for v in esa_voltage
        ]
    )
    central_effective_areas = np.array(
        [swapi_response.get_central_effective_area(v) for v in esa_voltage]
    ) * float(central_effective_area_scale)
    az_trans = np.asarray(swapi_response.azimuthal_transmission, dtype=float)
    az_trans_spacing = float(swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    # Step 2: Initial guess — Gaussian fit to count rate vs speed, anti-sunward velocity
    initial_guess = _get_initial_guess(
        count_rate,
        esa_voltage,
        passband_grids,
        central_speeds,
        central_effective_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
        spacecraft_velocity_rtn,
    )

    # Step 3: Optimize solar wind parameters to best match model count rate to observed
    return _optimize(
        count_rate,
        passband_grids,
        central_speeds,
        central_effective_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
        spacecraft_velocity_rtn,
        initial_guess,
        spin_axis_rtn=spin_axis_rtn,
    )


def _get_initial_guess(
    count_rate: ndarray,
    esa_voltage: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
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

    # Scale density so that the unit model count rate matches the mean observed count rate.
    unit_model = _model_count_rates(
        1.0,
        temperature,
        bulk_velocity_rtn,
        passband_grids,
        central_speeds,
        central_effective_areas,
        azimuthal_transmission,
        azimuthal_transmission_spacing,
        rotation_matrices,
        spacecraft_velocity_rtn,
        PROTON_MASS_KG,
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
    passband_grids: numba.typed.List,  # PassbandGrid per measurement, length N (V-only)
    central_speeds: ndarray,  # shape (N,), km/s, species/V-dependent v_0
    central_effective_areas: ndarray,  # shape (N,), cm^2, V-dependent and includes species/time scale
    azimuthal_transmission: ndarray,  # shape (M,), constant lookup table
    azimuthal_transmission_spacing: float,  # deg, constant
    rotation_matrices: ndarray,  # shape (N, 3, 3), RTN-to-SWAPI at each measurement time
    spacecraft_velocity_rtn: ndarray,  # shape (3,), km/s
    mass_kg: float,
) -> ndarray:
    """Pre-deadtime model count rate per measurement bin.

    `passband_grids[i]` carries the V-only passband shape; `central_speeds[i]` and
    `central_effective_areas[i]` encode the species/V/time-dependent scalars. The
    azimuthal transmission table is constant across measurements and passed once.
    Deadtime is applied at the residual stage so it acts on the combined
    (proton + alpha) rate."""
    # Thermal speed uses elementary charge (PROTON_CHARGE_COULOMBS = e), NEVER species charge,
    # because "temperature in eV" means k_B T = T_eV * e Joules regardless of species.
    thermal_speed = (
        np.sqrt(temperature * PROTON_CHARGE_COULOMBS / mass_kg) / METERS_PER_KILOMETER
    )
    bulk_speed = np.linalg.norm(bulk_velocity_rtn)
    n = len(passband_grids)
    result = np.empty(n)
    for i in range(n):
        ii = numba.int64(i)
        phi, theta = _compute_angles(
            bulk_velocity_rtn, rotation_matrices[ii], spacecraft_velocity_rtn
        )
        sw_params = SWParams(
            density=density,
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            thermal_speed=thermal_speed,
        )
        result[ii] = calculate_integral(
            passband_grids[ii],
            sw_params,
            central_speeds[ii],
            central_effective_areas[ii],
            azimuthal_transmission,
            azimuthal_transmission_spacing,
        )
    return result


@numba.njit(nogil=True)
def apply_deadtime_correction(true_rate: float) -> float:
    """Scalar deadtime correction (kept for back-compat with tests).
    Use `apply_deadtime_correction_array` for the vectorized form used by the residual."""
    return true_rate / (1.0 + SWAPI_DEADTIME_S * true_rate)


@numba.njit(nogil=True)
def apply_deadtime_correction_array(true_rates: ndarray) -> ndarray:
    """Vectorized deadtime correction. Applied at the residual stage so that for the
    two-species fit it acts on the *combined* (proton + alpha) true rate."""
    return true_rates / (1.0 + SWAPI_DEADTIME_S * true_rates)


@numba.njit(fastmath=True, nogil=True)
def calculate_integral(
    grid: PassbandGrid,
    sw_params: SWParams,
    central_speed: float,
    central_effective_area: float,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
):
    """Pre-deadtime model count rate at one ESA voltage step.

    The V-only `grid` carries passband-shape arrays + boundaries; species/V/time-dependent
    quantities are passed as scalars (`central_speed`, `central_effective_area`) plus the
    constant azimuthal transmission table. `central_effective_area` should already include
    any species/time efficiency correction (e.g. ε_species(t)/ε_p(t_lab) × A_lab(V))."""
    sin_bulk_elevation = math.sin((math.pi / 180) * sw_params.bulk_elevation)
    cos_bulk_elevation = math.cos((math.pi / 180) * sw_params.bulk_elevation)

    count_rate = 0.0

    for region in (0, -1, +1):
        is_sunglasses = region == 0
        # Normalization point: speed_ratio = central_speed / central_speed = 1.
        passband_norm = interpolate_passband(
            grid, is_sunglasses, elevation=0, speed_ratio=1.0
        )

        min_elevation, max_elevation, min_azimuth, max_azimuth = _get_angular_limits(
            sw_params, region, grid, central_speed
        )

        # skip region if completely out of FOV
        if max_elevation <= min_elevation or max_azimuth <= min_azimuth:
            continue

        # For OA, replace the gaussian-only azimuth bounds with a transmission-aware
        # scan. The original [bulk_az − width, bulk_az + width] clamp wastes nodes
        # in the OA dead zone (T < 1e-3 from 20°–25°). The scan estimates
        # density × T at the elevation peak inside the OA passband, sweeping
        # azimuth, and trims to where the integrand exceeds OA_SCAN_THRESHOLD × max.
        if not is_sunglasses:
            min_azimuth, max_azimuth = _trim_oa_azimuth_by_integrand(
                sw_params,
                region,
                central_speed,
                min_elevation,
                max_elevation,
                azimuthal_transmission,
                azimuthal_transmission_spacing,
            )
            if max_azimuth <= min_azimuth:
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
            [
                _interpolate_transmission(
                    azimuthal_transmission, azimuthal_transmission_spacing, x
                )
                for x in azimuth_points
            ]
        )

        elevation_integral = 0
        for i_elevation, elevation in enumerate(elevation_points):
            sin_elevation = math.sin((math.pi / 180) * elevation)
            cos_elevation = math.cos((math.pi / 180) * elevation)

            passband_lower_speed = central_speed * _eval_boundary(
                grid, is_sunglasses, elevation, True
            )
            passband_upper_speed = central_speed * _eval_boundary(
                grid, is_sunglasses, elevation, False
            )

            min_speed, max_speed = _dynamic_limits(
                sw_params.bulk_speed,
                sw_params.thermal_speed * 10,
                passband_lower_speed,
                passband_upper_speed,
            )

            # skip if out of passband
            if max_speed <= min_speed:
                continue

            half_sp = 0.5 * (max_speed - min_speed)
            mid_sp = 0.5 * (max_speed + min_speed)
            speed_points = mid_sp + half_sp * _GL_NODES_SPEED
            speed_weights = half_sp * _GL_WEIGHTS_SPEED

            # TODO create array outside of loop and just update it; or just flip azimuth and speed integrals
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
            * central_effective_area
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
def _get_angular_limits(
    sw_params: SWParams, region: int, grid: PassbandGrid, central_speed: float
):
    epsilon = EPSILON_SG if region == 0 else EPSILON_OA

    angular_width = (180 / np.pi) * np.arccos(
        _clamp(
            sw_params.thermal_speed**2
            * np.log(epsilon)
            / (central_speed * sw_params.bulk_speed)
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


OA_FULL_AZ_LO = 20.0
OA_FULL_AZ_HI = 150.0
OA_SCAN_SPACING_MAX_DEG = 1.0  # ceiling — coarse for typical (T~10 eV)
OA_SCAN_SPACING_MIN_DEG = 0.1  # floor — fine for cold-plasma extremes
OA_SKIP_ABS_THRESHOLD = 1e-9  # max of density × T below this → skip OA region


@numba.njit(nogil=True)
def _trim_oa_azimuth_by_integrand(
    sw_params: SWParams,
    region: int,
    central_speed: float,
    min_elevation: float,
    max_elevation: float,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
):
    """Find the OA azimuth subrange where the integrand is non-negligible.

    Scans `density × T` at (elevation=peak-within-passband, speed=central_speed)
    on a grid across the full OA passband. Spacing is adapted to the gaussian
    angular width σ_ang (= thermal_speed / bulk_speed) so cold plasma peaks
    aren't missed by a coarse sample.

    Returns `(0.0, 0.0)` to signal "skip OA region entirely"."""
    if region == +1:
        scan_lo, scan_hi = OA_FULL_AZ_LO, OA_FULL_AZ_HI
    else:
        scan_lo, scan_hi = -OA_FULL_AZ_HI, -OA_FULL_AZ_LO

    # Adaptive resolution: ~half of σ_ang so the scan resolves the gaussian peak
    # in azimuth even when it's < 1° wide.
    sigma_ang_deg = (180.0 / math.pi) * sw_params.thermal_speed / sw_params.bulk_speed
    spacing = 0.5 * sigma_ang_deg
    if spacing < OA_SCAN_SPACING_MIN_DEG:
        spacing = OA_SCAN_SPACING_MIN_DEG
    elif spacing > OA_SCAN_SPACING_MAX_DEG:
        spacing = OA_SCAN_SPACING_MAX_DEG
    n_scan = int(math.ceil((scan_hi - scan_lo) / spacing)) + 1
    if n_scan < 2:
        n_scan = 2

    # Pick the elevation closest to bulk_el that's inside the integration window.
    # That's where density (gaussian peak vs el at fixed az) is maximal.
    scan_el = sw_params.bulk_elevation
    if scan_el < min_elevation:
        scan_el = min_elevation
    elif scan_el > max_elevation:
        scan_el = max_elevation

    sin_be = math.sin((math.pi / 180) * sw_params.bulk_elevation)
    cos_be = math.cos((math.pi / 180) * sw_params.bulk_elevation)
    sin_se = math.sin((math.pi / 180) * scan_el)
    cos_se = math.cos((math.pi / 180) * scan_el)

    bulk_az = sw_params.bulk_azimuth
    bs = sw_params.bulk_speed
    cs = central_speed
    inv_2thermal2 = 1.0 / (2.0 * sw_params.thermal_speed**2)

    g = np.empty(n_scan)
    g_max = 0.0
    step = (scan_hi - scan_lo) / (n_scan - 1)
    for i in range(n_scan):
        az = scan_lo + i * step
        cos_da = math.cos((math.pi / 180) * (az - bulk_az))
        cos_angle = sin_be * sin_se + cos_be * cos_se * cos_da
        d2 = cs * cs + bs * bs - 2.0 * cs * bs * cos_angle
        density = math.exp(-d2 * inv_2thermal2)
        T = _interpolate_transmission(
            azimuthal_transmission, azimuthal_transmission_spacing, az
        )
        gi = density * T
        g[i] = gi
        if gi > g_max:
            g_max = gi

    if g_max < OA_SKIP_ABS_THRESHOLD:
        return 0.0, 0.0

    threshold_val = OA_SCAN_THRESHOLD * g_max
    # Always anchor to the OA inner boundary (SG/OA transition at ±20°): T = 0
    # there, so g(boundary) is below threshold even when the peak is right next
    # to it. Trimming the boundary side cuts off the rising-edge contribution.
    # Trim only the FAR end, where density has decayed.
    if region == +1:
        az_lo = scan_lo  # = 20°
        az_hi = scan_lo
        for i in range(n_scan - 1, -1, -1):
            if g[i] > threshold_val:
                az_hi = scan_lo + i * step
                break
    else:
        az_hi = scan_hi  # = -20°
        az_lo = scan_hi
        for i in range(n_scan):
            if g[i] > threshold_val:
                az_lo = scan_lo + i * step
                break

    if az_hi <= az_lo:
        return 0.0, 0.0
    return az_lo, az_hi


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
def interpolate_passband(
    grid: PassbandGrid, is_sunglasses: bool, elevation: float, speed_ratio: float
) -> float:
    """Interpolate the V-only passband at the given (elevation, speed_ratio = v / v_0).

    The grid contains passband values on a (elevation, speed_ratio) lattice so this
    function is species-independent — convert v -> speed_ratio at the call site using
    the species-specific `central_speed`."""
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


@numba.njit
def _residuals_njit(
    x,
    count_rate,
    sigma,
    passband_grids,
    central_speeds,
    central_effective_areas,
    azimuthal_transmission,
    azimuthal_transmission_spacing,
    rotation_matrices,
    spacecraft_velocity_rtn,
    mass_kg,
):
    density = np.exp(x[0])
    temperature = np.exp(x[1])
    bulk_velocity_rtn = x[2:5]
    model_true = _model_count_rates(
        density,
        temperature,
        bulk_velocity_rtn,
        passband_grids,
        central_speeds,
        central_effective_areas,
        azimuthal_transmission,
        azimuthal_transmission_spacing,
        rotation_matrices,
        spacecraft_velocity_rtn,
        mass_kg,
    )
    # Deadtime acts on the observed true rate (here, proton-only). For two-species
    # joint observation, the alpha fitter applies it to (proton+alpha) true.
    return (apply_deadtime_correction_array(model_true) - count_rate) / sigma


def _optimize(
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    rotation_matrices: ndarray,
    spacecraft_velocity_rtn: ndarray,
    initial_guess: ProtonSolarWindMoments,
    spin_axis_rtn: ndarray = None,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

    vr0, vt0, vn0 = initial_guess.bulk_velocity_rtn

    sigma = np.sqrt(np.maximum(count_rate * SWAPI_LIVETIME_S, 1.0)) / SWAPI_LIVETIME_S

    if spin_axis_rtn is None:
        spin_axis_rtn = rotation_matrices[0, 1, :].copy()

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
            central_speeds,
            central_effective_areas,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            rotation_matrices,
            spacecraft_velocity_rtn,
            PROTON_MASS_KG,
        )

    # See docs/swapi/solar-wind-moments.md for diff_step rationale.
    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)

    # Wrong-basin detection via spin-axis mirror flip; see docs/swapi/solar-wind-moments.md.
    # Spin axis in RTN = body-Y direction expressed in RTN coords = row 1 of R[i].
    # 180° rotation about that axis: v' = 2(v·s)s − v.
    chi2 = float(np.sum(result.fun**2))
    v_rtn = result.x[2:5]
    v_flipped = 2.0 * float(np.dot(v_rtn, spin_axis_rtn)) * spin_axis_rtn - v_rtn
    x_flipped = result.x.copy()
    x_flipped[2:5] = v_flipped
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
