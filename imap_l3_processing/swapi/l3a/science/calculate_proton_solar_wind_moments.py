import math
from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np
import scipy.optimize
from numpy import ndarray
import uncertainties.umath as umath
import uncertainties.unumpy as unp
from uncertainties import UFloat, correlated_values, covariance_matrix, ufloat

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    METERS_PER_KILOMETER,
)

from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.l3a.science.passband_grid import PassbandGrid
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse


class SWParams(NamedTuple):
    density: float
    bulk_speed: float
    bulk_azimuth: float
    bulk_elevation: float
    thermal_speed: float


N_ELEVATION = 21
N_AZIMUTH = 21
N_SPEED = 15

# Gauss-Legendre quadrature nodes and weights on the standard interval [-1, 1].
# Precomputed once at module load and rescaled to each integration window inside
# `calculate_integral`. Read as module-level globals from inside the JIT region.
_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(
    N_ELEVATION
)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(N_AZIMUTH)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(N_SPEED)

# OA azimuth integration: the gaussian Δα window is refined by scanning
# `density × T` and trimming both ends where it falls below threshold.
# See `_trim_oa_azimuth_by_integrand`.
OA_SCAN_THRESHOLD = 1e-6  # trim where g < threshold × max(g)

EPSILON_OA = 1e-6
EPSILON_SG = 1e-6

# Speed integration half-width in units of thermal speed. The Maxwellian at
# 6σ is exp(-18) ≈ 1.5e-8 of peak, already negligible. With N_SPEED=15 GL
# nodes the per-node spacing is ~0.8σ, sufficient for the bilinear-interpolated
# integrand. Empirically swept k ∈ {3, 4, 5, 6, 7, 8} × N ∈ {11, 15, 21}
# against `reference_integrals.csv`: k=6 N=15 minimizes the worst-case error
# at high count rate (≥10⁴ Hz) by ~0.5% vs k=5 same N. k=3 fails
# catastrophically (worst-case ~70% at >10 Hz) because it can clip the
# Maxwellian peak when the angular geometry shifts the speed peak off-center.
SPEED_HALF_WIDTH_VTH = 6.0

# Outer edge of the vanes-vignetting (VV) sub-region of the open aperture, in
# degrees of azimuth. The transmission table T(|φ|) is identically zero at
# |φ|=20° (vanes fully blocking) and rises smoothly to ~1 by |φ|≈30°. Splitting
# the OA azimuth integration at ±VV_OUTER_DEG anchors the GL boundary
# clustering at the inflection of T(φ) so the steep rise is well resolved.
# VV_OUTER_DEG=26 was chosen by sweeping 23–30 against `reference_integrals.csv`
# (best high-rate failure count). Going to 27° puts the steepest dT/dφ point
# right at the boundary, undoing the benefit.
VV_OUTER_DEG = 26.0

# Wrong-basin recovery: K-rotation analytic verifier + restart LM + iter-flip-LM.
# The K-rotation grid evaluates analytic chi² at K-1 spin-axis rotations of the
# LM-1 velocity; the cheapest (best chi²) seeds an LM restart only when it sits
# within `_VERIFIER_CHI_RATIO_THRESHOLD`× LM-1's chi². After that one restart,
# iter-flip-LM repeatedly mirrors across the spin axis (180°) and re-runs LM,
# accepting on each chi² improvement until convergence (max `_MAX_BASIN_FLIPS`
# iterations). Each acceptance test uses a relative tolerance to break ties
# between near-equivalent basins so the iter-flip loop terminates.
_GRID_K = 4
_MAX_BASIN_FLIPS = 6
_BASIN_FLIP_REL_TOL = 1e-3
_VERIFIER_CHI_RATIO_THRESHOLD = 5000.0

# Non-paralyzable detector deadtime (Tsoulfanidis 1995, p. 74): n = g / (1 - g*tau)
# Rearranged for forward model (true rate -> measured rate): g = n / (1 + n*tau)
SWAPI_DEADTIME_S = 183.7e-9

# Duration of one ESA energy step measurement (seconds). Used by test harnesses
# for Poisson noise generation (rate × T → counts → Poisson → ÷ T).
SWAPI_LIVETIME_S = 0.145

# Floor for the initial-guess temperature, applied when the fitted spectral width
# is below the value implied by this floor (1 eV expressed in Kelvin).
INITIAL_TEMPERATURE_FLOOR_K = (
    PROTON_CHARGE_COULOMBS / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
)

# Number of Monte Carlo samples used by `derive_velocity_angles` to propagate
# the velocity covariance through the nonlinear speed/clock/deflection-angle
# transforms. The angle gradients scale as 1/v_xy² and 1/(s²·v_xy), so the
# first-order delta method underestimates σ when σ_xy is comparable to v_xy
# (typical for SWAPI: bulk velocity is dominated by the spin-axis component).
N_VELOCITY_ANGLE_MC_SAMPLES = 1000

# Loosened from scipy default; xtol fires before ftol on this problem so loosening ftol is redundant.
_LM_XTOL = 1e-3

OA_SKIP_FRACTION = 1e-3

KM_TO_CM = 1e5

N_OA_SCAN = 64


@dataclass
class ProtonSolarWindMoments:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    bad_fit_flag: int

    def bulk_velocity_rtn_nominal(self) -> ndarray:
        """Nominal RTN velocity vector (km/s); shape (3,)."""
        return np.array([v.nominal_value for v in self.bulk_velocity_rtn])

    def bulk_velocity_rtn_covariance(self) -> ndarray:
        """3×3 RTN velocity covariance (km²/s²)."""
        return np.array(covariance_matrix(self.bulk_velocity_rtn))


def fit_solar_wind_proton_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    swapi_response: SWAPIResponse,
    central_effective_area_scale: float,
    rotation_matrices: ndarray,
) -> ProtonSolarWindMoments:
    """Fit proton solar wind moments. ``central_effective_area_scale`` should be
    ``ε_p(t)/ε_p(t_lab)`` from the efficiency LUT — it's applied to each measurement's
    lab-derived central effective area before integration.

    ``rotation_matrices`` must be precomputed and reused
    across stage 1/stage 2 fits to avoid duplicate SPICE calls."""
    from imap_l3_processing.constants import PROTON_MASS_PER_CHARGE_M_P_PER_E

    # Drop any 0V (or non-finite) ESA steps. Some sweeps include a zero-energy
    # step that carries no useful information and would make central_speed = 0,
    # producing divide-by-zero deep inside the JIT integrator.
    keep = (esa_voltage > 0) & np.isfinite(esa_voltage) & (count_rate > 0)

    # 10%-of-peak mask: keep only bins above 10% of the peak count rate so the
    # deep tails (PUI/alpha contamination in production data) can't bias the
    # proton fit, but enough bins remain that the spin-axis-mirror basins are
    # discriminable. Tighter masks (e.g. FWHM at 0.5×max) leave the cold-plasma
    # chi² landscape too noise-degenerate to pick the right basin.
    cr_max = float(np.nanmax(count_rate[keep])) if np.any(keep) else 0.0
    tail_mask = count_rate >= 0.1 * cr_max
    if int((keep & tail_mask).sum()) >= 5:
        keep = keep & tail_mask

    if not np.all(keep):
        esa_voltage = esa_voltage[keep]
        count_rate = count_rate[keep]
        rotation_matrices = rotation_matrices[keep]

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

    initial_guess = _get_initial_guess(
        count_rate,
        esa_voltage,
        passband_grids,
        central_speeds,
        central_effective_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
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
        initial_guess
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
) -> ProtonSolarWindMoments:
    speed = esa_voltage_to_proton_speed(esa_voltage)

    peak_idx = np.nanargmax(count_rate)
    try:
        (_, bulk_speed_init, sigma_v), _ = scipy.optimize.curve_fit(
            lambda v, A, mu, sigma: A * np.exp(-((v - mu) ** 2) / (2 * sigma**2)),
            speed,
            count_rate,
            p0=[count_rate[peak_idx], speed[peak_idx], 50.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError:
        bulk_speed_init = speed[peak_idx]
        sigma_v = 50.0

    sigma_floor_v = (
        math.sqrt(
            INITIAL_TEMPERATURE_FLOOR_K
            * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
            / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )
    sigma_thermal_v = max(sigma_v, sigma_floor_v)
    temperature = float(
        PROTON_MASS_KG
        * (sigma_thermal_v * METERS_PER_KILOMETER) ** 2
        / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    )

    bulk_velocity_rtn = np.array(
        [
            math.sqrt(max(float(bulk_speed_init) ** 2 - 30.0**2, 0.0)),
            -30.0,
            0.0,
        ]
    )

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
        PROTON_MASS_KG,
    )
    density = _optimal_density_scale(unit_model, count_rate)

    return ProtonSolarWindMoments(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=0,
    )


@numba.njit(nogil=True)
def _model_count_rates(
    density: float,
    temperature: float,  # K
    bulk_velocity_rtn: ndarray,  # shape (3,), inertial RTN, km/s
    passband_grids: numba.typed.List,  # PassbandGrid per measurement, length N (V-only)
    central_speeds: ndarray,  # shape (N,), km/s, species/V-dependent v_0
    central_effective_areas: ndarray,  # shape (N,), cm^2, V-dependent and includes species/time scale
    azimuthal_transmission: ndarray,  # shape (M,), constant lookup table
    azimuthal_transmission_spacing: float,  # deg, constant
    rotation_matrices: ndarray,  # shape (N, 3, 3), RTN-to-SWAPI at each measurement time
    mass_kg: float,
) -> ndarray:
    """Pre-deadtime model count rate per measurement bin.

    `passband_grids[i]` carries the V-only passband shape; `central_speeds[i]` and
    `central_effective_areas[i]` encode the species/V/time-dependent scalars. The
    azimuthal transmission table is constant across measurements and passed once.
    Deadtime is applied at the residual stage so it acts on the combined
    (proton + alpha) rate."""
    thermal_speed = (
        np.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature / mass_kg)
        / METERS_PER_KILOMETER
    )
    bulk_speed = float(np.linalg.norm(bulk_velocity_rtn))
    n = len(passband_grids)
    result = np.empty(n)
    for i in range(n):
        phi, theta = _compute_angles(bulk_velocity_rtn, rotation_matrices[i])
        sw_params = SWParams(
            density=density,
            bulk_speed=bulk_speed,
            bulk_azimuth=phi,
            bulk_elevation=theta,
            thermal_speed=thermal_speed,
        )
        result[i] = calculate_integral(
            passband_grids[i],
            sw_params,
            central_speeds[i],
            central_effective_areas[i],
            azimuthal_transmission,
            azimuthal_transmission_spacing,
        )
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
    grid: PassbandGrid,
    sw_params: SWParams,
    central_speed: float,
    central_effective_area: float,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
):
    """Pre-deadtime model count rate at one ESA voltage step.

    Five-region azimuth scheme (0=SG, ±2=VV, ±1=OA):
        SG | VV- | OA- | VV+ | OA+
    The OA azimuth integration is split at ±VV_OUTER_DEG so the steep T(φ)
    rise (the vanes-vignetting transition between |φ|=20° and ~30°) is
    captured by GL boundary clustering inside the VV regions, instead of
    being smeared across a single wide OA window."""

    # SG first — used as the reference for the OA skip decision.
    min_el, max_el, min_az, max_az = _integration_window(
        sw_params, 0, grid, central_speed
    )
    sg_rate = 0.0
    if max_el > min_el and max_az > min_az:
        sg_rate = _integrate_region(
            grid,
            sw_params,
            central_speed,
            central_effective_area,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            True,
            min_el,
            max_el,
            min_az,
            max_az,
        )

    count_rate = sg_rate

    # VV± regions: small OA azimuth bands [-VV_OUTER, -20] and [20, VV_OUTER].
    # T is small (≤~0.05) but rises steeply; a few GL nodes here resolve it
    # well. No trim or skip — these regions are bounded and cheap.
    for region in (-2, +2):
        min_el, max_el, min_az, max_az = _integration_window(
            sw_params, region, grid, central_speed
        )
        if max_el <= min_el or max_az <= min_az:
            continue
        count_rate += _integrate_region(
            grid,
            sw_params,
            central_speed,
            central_effective_area,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            False,
            min_el,
            max_el,
            min_az,
            max_az,
        )

    # OA± regions: full-transmission azimuth bands beyond ±VV_OUTER.
    for region in (-1, +1):
        min_el, max_el, min_az, max_az = _integration_window(
            sw_params, region, grid, central_speed
        )
        if max_el <= min_el or max_az <= min_az:
            continue

        min_az, max_az, integral_Tg = _trim_oa_azimuth_by_integrand(
            sw_params,
            central_speed,
            min_el,
            max_el,
            min_az,
            max_az,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
        )
        if max_az <= min_az:
            continue

        oa_upper_bound = _oa_rate_upper_bound(
            grid,
            central_speed,
            central_effective_area,
            integral_Tg,
            min_el,
            max_el,
        )
        if oa_upper_bound < max(0.1, OA_SKIP_FRACTION * sg_rate):
            continue

        count_rate += _integrate_region(
            grid,
            sw_params,
            central_speed,
            central_effective_area,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            False,
            min_el,
            max_el,
            min_az,
            max_az,
        )

    return count_rate


@numba.njit(nogil=True)
def _integration_window(
    sw_params: SWParams, region: int, grid: PassbandGrid, central_speed: float
):
    epsilon = EPSILON_SG if region == 0 else EPSILON_OA

    cos_angular_width = (
         sw_params.thermal_speed**2 * np.log(epsilon)
         / (central_speed * sw_params.bulk_speed)
         + 1
    )
    cos_angular_width = _clamp(cos_angular_width, -1, +1)
    angular_width = np.degrees(np.arccos(cos_angular_width))

    if region == 0:
        # region = 0 -> sunglasses passband
        sg_lo, sg_hi = grid.sg_elevation_range
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, sg_lo, sg_hi
        )

        min_azimuth, max_azimuth = _dynamic_limits(
            sw_params.bulk_azimuth, angular_width, -20.0, 20.0
        )
    else:
        # region != 0 -> open aperture passband
        oa_lo, oa_hi = grid.oa_elevation_range
        min_elevation, max_elevation = _dynamic_limits(
            sw_params.bulk_elevation, angular_width, oa_lo, oa_hi
        )

        is_vv_band = abs(region) == 2
        inner = 20.0 if is_vv_band else VV_OUTER_DEG
        outer = VV_OUTER_DEG if is_vv_band else 150.0
        az_lo = inner if region > 0 else -outer
        az_hi = outer if region > 0 else -inner
        min_azimuth, max_azimuth = _dynamic_limits(
            sw_params.bulk_azimuth, angular_width, az_lo, az_hi
        )

    return min_elevation, max_elevation, min_azimuth, max_azimuth


@numba.njit(nogil=True)
def _clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


@numba.njit(nogil=True)
def _dynamic_limits(
    center: float, width: float, lower_bound: float, upper_bound: float
):
    return _clamp(center - width, lower_bound, upper_bound), _clamp(
        center + width, lower_bound, upper_bound
    )


@numba.njit(fastmath=True, nogil=True)
def _integrate_region(
    grid: PassbandGrid,
    sw_params: SWParams,
    central_speed: float,
    central_effective_area: float,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    is_sunglasses: bool,
    min_elevation: float,
    max_elevation: float,
    min_azimuth: float,
    max_azimuth: float,
):
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

        passband_lower_speed = central_speed * _eval_boundary(
            grid, is_sunglasses, elevation, True
        )
        passband_upper_speed = central_speed * _eval_boundary(
            grid, is_sunglasses, elevation, False
        )

        min_speed, max_speed = _dynamic_limits(
            sw_params.bulk_speed,
            sw_params.thermal_speed * SPEED_HALF_WIDTH_VTH,
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
            cos_angle = (
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

    return (
        elevation_integral
        * (math.pi / 180) ** 2  # deg^2 -> rad^2
        * _density_area_norm(sw_params, central_effective_area)
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


@numba.njit(nogil=True, inline="always")
def _density_area_norm(sw_params: SWParams, central_effective_area: float) -> float:
    """A_eff · n / (√2π v_th)³ · (km→cm). Multiplied by a phase-space integral in
    (km/s × rad²) units, gives a count rate in Hz."""
    return (
        central_effective_area
        * sw_params.density
        * (np.sqrt(2 * np.pi) * sw_params.thermal_speed) ** -3
        * 1e5
    )


@numba.njit(nogil=True)
def _trim_oa_azimuth_by_integrand(
    sw_params: SWParams,
    central_speed: float,
    min_elevation: float,
    max_elevation: float,
    gaussian_az_lo: float,
    gaussian_az_hi: float,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
):
    """Trim the OA azimuth window by walking both ends inward until the
    integrand exceeds threshold. Returns `(0.0, 0.0, 0.0)` to skip the region.
    Third return value is ∫ T(φ) g(φ) dφ over the trimmed window in radians,
    where g(φ) = f(v₀, θ_b', φ) is the full Maxwellian (with normalization)."""
    if gaussian_az_hi <= gaussian_az_lo:
        return 0.0, 0.0, 0.0

    azimuths = np.linspace(gaussian_az_lo, gaussian_az_hi, N_OA_SCAN)
    scan_el = _clamp(sw_params.bulk_elevation, min_elevation, max_elevation)

    deg2rad = math.pi / 180.0
    sin_be = math.sin(deg2rad * sw_params.bulk_elevation)
    cos_be = math.cos(deg2rad * sw_params.bulk_elevation)
    sin_se = math.sin(deg2rad * scan_el)
    cos_se = math.cos(deg2rad * scan_el)

    cos_da = np.cos(deg2rad * (azimuths - sw_params.bulk_azimuth))
    cos_angle = sin_be * sin_se + cos_be * cos_se * cos_da
    d2 = (
        central_speed**2
        + sw_params.bulk_speed**2
        - 2.0 * central_speed * sw_params.bulk_speed * cos_angle
    )
    kT = np.exp(-d2 / (2.0 * sw_params.thermal_speed**2))
    for i in range(N_OA_SCAN):
        kT[i] *= _interpolate_transmission(
            azimuthal_transmission, azimuthal_transmission_spacing, azimuths[i]
        )

    kT_max = np.max(kT)
    threshold_val = OA_SCAN_THRESHOLD * kT_max

    lo_i = 0
    for i in range(N_OA_SCAN):
        if kT[i] > threshold_val:
            lo_i = max(i - 1, 0)
            break

    hi_i = N_OA_SCAN - 1
    for i in range(N_OA_SCAN - 1, -1, -1):
        if kT[i] > threshold_val:
            hi_i = min(i + 1, N_OA_SCAN - 1)
            break

    dphi_rad = (azimuths[1] - azimuths[0]) * deg2rad
    maxwellian_norm = (
        sw_params.density / (np.sqrt(2.0 * np.pi) * sw_params.thermal_speed) ** 3
    )
    integral_Tg = np.trapezoid(kT[lo_i : hi_i + 1], dx=dphi_rad) * maxwellian_norm

    return azimuths[lo_i], azimuths[hi_i], integral_Tg


@numba.njit(nogil=True)
def _oa_rate_upper_bound(
    grid: PassbandGrid,
    central_speed: float,
    central_effective_area: float,
    integral_Tg: float,
    min_elevation: float,
    max_elevation: float,
) -> float:
    """Heuristic upper estimate of the OA region rate (Hz):
        Ĉ_OA = A₀ · v₀³ · Δθ · Δv · ∫ T(φ) g(φ) dφ
    bounding cos(θ) ≤ 1 (giving Δθ) and v³ P(v/v₀, θ) ≤ v₀³ supported on the
    passband (giving Δv = (r_max(0) - r_min(0))·v₀ at θ = 0°). The azimuth
    integral is supplied by `_trim_oa_azimuth_by_integrand` with full Maxwellian
    normalization (i.e. g(φ) = f(v₀, θ_b', φ))."""
    delta_theta_rad = (math.pi / 180.0) * (max_elevation - min_elevation)
    delta_v = central_speed * (
        _eval_boundary(grid, False, 0.0, False) - _eval_boundary(grid, False, 0.0, True)
    )
    return (
        central_effective_area
        * central_speed**3
        * delta_theta_rad
        * delta_v
        * integral_Tg
        * KM_TO_CM
    )


def _optimize(
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    rotation_matrices: ndarray,
    initial_guess: ProtonSolarWindMoments,
) -> ProtonSolarWindMoments:
    from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

    def residuals(state):
        return _residuals_njit(
            state,
            count_rate,
            passband_grids,
            central_speeds,
            central_effective_areas,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            rotation_matrices,
            PROTON_MASS_KG,
        )

    # See docs/swapi/solar-wind-moments.md for diff_step and xtol rationale.
    def run_lm(initial_state):
        return scipy.optimize.least_squares(
            residuals, initial_state, method="lm", diff_step=1e-4, xtol=_LM_XTOL,
        )

    initial_velocity_r, initial_velocity_t, initial_velocity_n = initial_guess.bulk_velocity_rtn
    first_lm_result = run_lm(np.array([
        np.log(initial_guess.density),
        np.log(initial_guess.temperature),
        initial_velocity_r, initial_velocity_t, initial_velocity_n,
    ]))

    spin_axis_rtn = _chunk_spin_axis_rtn(rotation_matrices)
    final_result = _refine_basin_via_k_rotation_and_iter_flip(
        first_lm_result,
        run_lm,
        count_rate,
        passband_grids,
        central_speeds,
        central_effective_areas,
        azimuthal_transmission,
        azimuthal_transmission_spacing,
        rotation_matrices,
        spin_axis_rtn,
    )

    density, temperature, bulk_velocity_rtn = _unpack_state(final_result.x)
    density_sigma, temperature_sigma, velocity_covariance = (
        _estimate_parameter_uncertainties(final_result, density, temperature)
    )

    return ProtonSolarWindMoments(
        density=ufloat(density, density_sigma),
        temperature=ufloat(temperature, temperature_sigma),
        bulk_velocity_rtn=make_correlated_velocity(
            bulk_velocity_rtn, velocity_covariance
        ),
        bad_fit_flag=SwapiL3Flags.NONE if final_result.success else SwapiL3Flags.BAD_FIT,
    )


def _refine_basin_via_k_rotation_and_iter_flip(
    first_lm_result,
    run_lm,
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    rotation_matrices: ndarray,
    spin_axis_rtn: ndarray,
):
    """Wrong-basin recovery (see solar-wind-moments.md → "Wrong-basin detection").

    The proton chi² landscape has a near-degenerate basin obtained by reflecting
    the bulk velocity across the spin axis, plus secondary basins reachable by
    other rotations around that axis. Recovery has three phases:

    1. K-rotation analytic verifier — for each of the K-1 non-zero rotations,
       analytically rescale density and evaluate chi². If the best rotated chi²
       is too far above LM-1's chi² (ratio ≥ `_VERIFIER_CHI_RATIO_THRESHOLD`),
       LM-1 has no nearby competing basin and we keep it as-is.
    2. K-rotation restart — when the verifier triggers, run LM seeded from the
       best rotated state. Adopt it if it improves chi² past the relative
       tolerance.
    3. Iter-flip-LM — from the current best, repeatedly flip 180° around the
       spin axis and re-run LM, accepting on each chi² improvement until the
       loop converges (max `_MAX_BASIN_FLIPS` iterations). Each flip is the
       Rodrigues identity for π-rotation: v' = 2(v·ŝ)ŝ − v."""
    first_lm_chi2 = float(np.sum(first_lm_result.fun**2))

    rotated_velocity, rotated_density, rotated_chi2 = _best_k_rotation_seed(
        first_lm_result,
        count_rate,
        passband_grids,
        central_speeds,
        central_effective_areas,
        azimuthal_transmission,
        azimuthal_transmission_spacing,
        rotation_matrices,
        spin_axis_rtn,
    )

    if rotated_chi2 >= first_lm_chi2 * _VERIFIER_CHI_RATIO_THRESHOLD:
        return first_lm_result

    restart_state = first_lm_result.x.copy()
    restart_state[2:5] = rotated_velocity
    if rotated_density > 0.0 and np.isfinite(rotated_density):
        restart_state[0] = math.log(rotated_density)
    rotated_lm_result = run_lm(restart_state)
    rotated_lm_chi2 = float(np.sum(rotated_lm_result.fun**2))

    if rotated_lm_chi2 < first_lm_chi2 * (1.0 - _BASIN_FLIP_REL_TOL):
        current_result = rotated_lm_result
        current_chi2 = rotated_lm_chi2
    else:
        current_result = first_lm_result
        current_chi2 = first_lm_chi2

    for _ in range(_MAX_BASIN_FLIPS):
        v_rtn = current_result.x[2:5]
        v_flipped = 2.0 * float(np.dot(v_rtn, spin_axis_rtn)) * spin_axis_rtn - v_rtn
        flipped_state = current_result.x.copy()
        flipped_state[2:5] = v_flipped
        flipped_lm_result = run_lm(flipped_state)
        flipped_lm_chi2 = float(np.sum(flipped_lm_result.fun**2))
        if flipped_lm_chi2 < current_chi2 * (1.0 - _BASIN_FLIP_REL_TOL):
            current_result = flipped_lm_result
            current_chi2 = flipped_lm_chi2
        else:
            break

    return current_result


def _best_k_rotation_seed(
    lm_result,
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    rotation_matrices: ndarray,
    spin_axis_rtn: ndarray,
) -> tuple[ndarray, float, float]:
    """Search the K-1 non-zero spin-axis rotations of the LM bulk velocity.
    For each rotated state, density is analytically rescaled to minimize chi²
    in closed form (no LM here — just a forward-model evaluation per rotation).
    Returns (rotated_velocity, rotated_density, rotated_chi2) of the rotation
    with the lowest chi²."""
    density, temperature, bulk_velocity = _unpack_state(lm_result.x)
    best_velocity = bulk_velocity
    best_density = density
    best_chi2 = math.inf
    n_residuals = len(count_rate)
    for rotation_index in range(1, _GRID_K):
        rotation_angle_rad = 2.0 * math.pi * rotation_index / _GRID_K
        rotated_velocity = _rotate_about_axis(
            bulk_velocity, spin_axis_rtn, rotation_angle_rad
        )
        predicted_obs_rate = apply_deadtime_correction_array(
            _model_count_rates(
                density,
                temperature,
                rotated_velocity,
                passband_grids,
                central_speeds,
                central_effective_areas,
                azimuthal_transmission,
                azimuthal_transmission_spacing,
                rotation_matrices,
                PROTON_MASS_KG,
            )
        )
        density_scale = _optimal_density_scale(predicted_obs_rate, count_rate)
        rotated_chi2 = float(
            np.sum((density_scale * predicted_obs_rate - count_rate) ** 2)
        )
        if rotated_chi2 < best_chi2:
            best_chi2 = rotated_chi2
            best_velocity = rotated_velocity
            best_density = density_scale * density
    return best_velocity, best_density, best_chi2


def _estimate_parameter_uncertainties(
    lm_result, density: float, temperature: float
) -> tuple[float, float, ndarray]:
    """Estimate (σ_density, σ_temperature, velocity_covariance) from the LM Jacobian.

    Uses the reduced-χ² scaled covariance Σ = s² (JᵀJ)⁺ where s² = Σrᵢ²/(N−p),
    equivalent to scipy.optimize.curve_fit with absolute_sigma=False (residuals
    are unweighted, so s² absorbs measurement noise + model imperfection).
    See `Parameter uncertainties` in solar-wind-moments.md.

    The state ordering is [log n, log T, v_R, v_T, v_N], so log-space variances
    on indices 0,1 transform to absolute σ via σ_n = n·√Σ₀₀ and σ_T = T·√Σ₁₁."""
    try:
        n_residuals, n_parameters = len(lm_result.fun), len(lm_result.x)
        reduced_chi_squared = (
            float(np.sum(lm_result.fun**2)) / max(n_residuals - n_parameters, 1)
        )
        parameter_covariance = (
            reduced_chi_squared * np.linalg.pinv(lm_result.jac.T @ lm_result.jac)
        )
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.full((3, 3), np.nan)
    log_density_variance = max(parameter_covariance[0, 0], 0.0)
    log_temperature_variance = max(parameter_covariance[1, 1], 0.0)
    return (
        float(density * np.sqrt(log_density_variance)),
        float(temperature * np.sqrt(log_temperature_variance)),
        parameter_covariance[2:5, 2:5],
    )


@numba.njit
def _residuals_njit(
    x,
    count_rate,
    passband_grids,
    central_speeds,
    central_effective_areas,
    azimuthal_transmission,
    azimuthal_transmission_spacing,
    rotation_matrices,
    mass_kg,
):
    density, temperature, bulk_velocity_rtn = _unpack_state(x)
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
        mass_kg,
    )
    model_obs = apply_deadtime_correction_array(model_true)
    return model_obs - count_rate


@numba.njit(inline="always")
def _unpack_state(x: ndarray) -> tuple[float, float, ndarray]:
    """Decode the LM parameter vector [log n, log T, v_R, v_T, v_N] into (n, T, v)."""
    return math.exp(x[0]), math.exp(x[1]), x[2:5]


def _pack_state(density: float, temperature: float, bulk_velocity: ndarray) -> ndarray:
    return np.array([math.log(density), math.log(temperature), *bulk_velocity])


@numba.njit(nogil=True)
def apply_deadtime_correction_array(true_rates: ndarray) -> ndarray:
    """Vectorized deadtime correction. Applied at the residual stage so that for the
    two-species fit it acts on the *combined* (proton + alpha) true rate."""
    return true_rates / (1.0 + SWAPI_DEADTIME_S * true_rates)


def _optimal_density_scale(predicted: ndarray, observed: ndarray) -> float:
    # Minimizes ‖scale·predicted − observed‖²; solution is scale = (m·r)/(m·m).
    mm = float(np.dot(predicted, predicted))
    return float(np.dot(predicted, observed)) / mm if mm > 0.0 else 1.0


def _evaluate_rotated_chi(
    lm_result,
    theta: float,
    count_rate: ndarray,
    passband_grids: numba.typed.List,
    central_speeds: ndarray,
    central_effective_areas: ndarray,
    azimuthal_transmission: ndarray,
    azimuthal_transmission_spacing: float,
    rotation_matrices: ndarray,
) -> tuple:
    density, temperature, bulk_velocity = _unpack_state(lm_result.x)

    spin_axis = _chunk_spin_axis_rtn(rotation_matrices)
    rotated_velocity = _rotate_about_axis(bulk_velocity, spin_axis, theta)

    predicted_count_rate = apply_deadtime_correction_array(
        _model_count_rates(
            density,
            temperature,
            rotated_velocity,
            passband_grids,
            central_speeds,
            central_effective_areas,
            azimuthal_transmission,
            azimuthal_transmission_spacing,
            rotation_matrices,
            PROTON_MASS_KG,
        )
    )

    density_scale = _optimal_density_scale(predicted_count_rate, count_rate)
    mse = float(np.mean((density_scale * predicted_count_rate - count_rate) ** 2))
    return mse, rotated_velocity, density_scale * density


def _chunk_spin_axis_rtn(rotation_matrices: ndarray) -> ndarray:
    axis = rotation_matrices[:, 1, :].mean(axis=0)
    return axis / np.linalg.norm(axis)


def _rotate_about_axis(v: ndarray, axis: ndarray, angle: float) -> ndarray:
    # Rodrigues' formula: v cosθ + (axis × v) sinθ + axis (axis·v)(1 − cosθ)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    return (
        v * cos_t
        + np.cross(axis, v) * sin_t
        + axis * float(np.dot(axis, v)) * (1.0 - cos_t)
    )


def make_correlated_velocity(
    v_mean: ndarray, v_cov: ndarray
) -> tuple[UFloat, UFloat, UFloat]:
    """Build a correlated 3-tuple of UFloats from (mean, covariance). Falls
    back to independent NaN-σ UFloats if the covariance is non-finite or
    `correlated_values` rejects it (LinAlg failures, NaN fills, etc.)."""
    if np.all(np.isfinite(v_cov)):
        try:
            return tuple(correlated_values(v_mean, v_cov))
        except Exception:
            pass
    return tuple(ufloat(float(v), np.nan) for v in v_mean)


def derive_velocity_angles(
    fitting_result: "ProtonSolarWindMoments",
    epoch_tt2000_ns: float,
) -> tuple:
    """Return (speed, clock_angle, deflection_angle) as ufloats in the DPS frame.

    Speed uncertainty is propagated automatically by the ``uncertainties``
    package (first-order delta method via ``umath.sqrt(Σ xᵢ²)``), which is
    essentially exact whenever ``|u| >> σ``. Clock and deflection angles, by contrast, have
    arctan2/arccos gradients that scale as ``1/v_xy²`` and ``1/(s²·v_xy)``,
    which underestimate σ severely whenever ``σ_xy`` is comparable to
    ``v_xy`` — the typical SWAPI regime, since the bulk velocity is
    dominated by the spin-axis component. Their σ are computed by drawing
    ``N_VELOCITY_ANGLE_MC_SAMPLES`` velocity samples from
    ``MultivariateNormal(u, cov)`` and applying the transforms per-sample.
    Clock-angle σ uses residuals wrapped to (-180°, 180°] so the 0°/360°
    branch cut doesn't inflate the spread.
    """
    from imap_l3_processing.swapi.l3a.utils import rotate_rtn_to_dps

    u_unc = rotate_rtn_to_dps(
        np.array(fitting_result.bulk_velocity_rtn), epoch_tt2000_ns
    )
    u = unp.nominal_values(u_unc)
    cov_DPS = np.array(covariance_matrix(u_unc))

    speed_nom = float(np.linalg.norm(u))
    clock_nom = float(np.degrees(np.arctan2(u[1], u[0])) % 360)
    defl_nom = float(np.degrees(np.arccos(-u[2] / speed_nom)))

    if not np.all(np.isfinite(cov_DPS)):
        return (
            ufloat(speed_nom, np.nan),
            ufloat(clock_nom, np.nan),
            ufloat(defl_nom, np.nan),
        )

    speed = umath.sqrt(sum(x**2 for x in u_unc))

    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(
        u, cov_DPS, size=N_VELOCITY_ANGLE_MC_SAMPLES, check_valid="ignore"
    )

    clocks = np.degrees(np.arctan2(samples[:, 1], samples[:, 0])) % 360.0
    clock_resid = ((clocks - clock_nom + 180.0) % 360.0) - 180.0
    clock_sigma = float(np.std(clock_resid, ddof=1))

    sample_speeds = np.linalg.norm(samples, axis=1)
    defls = np.degrees(np.arccos(np.clip(-samples[:, 2] / sample_speeds, -1.0, 1.0)))
    defl_sigma = float(np.std(defls, ddof=1))

    return (
        speed,
        ufloat(clock_nom, clock_sigma),
        ufloat(defl_nom, defl_sigma),
    )


@numba.njit(nogil=True)
def apply_deadtime_correction(true_rate: float) -> float:
    """Scalar deadtime correction (kept for back-compat with tests).
    Use `apply_deadtime_correction_array` for the vectorized form used by the residual."""
    return true_rate / (1.0 + SWAPI_DEADTIME_S * true_rate)
