"""
Numpy reference implementation of the SWAPI proton solar wind integral.

Used by unit/integration tests and the integration_benchmark script to validate
the numba optimized implementation in calculate_proton_solar_wind_moments.py.
"""
import math

import numpy as np

from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_angular_limits,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import (
    eval_boundary_min,
    eval_boundary_max,
)

_N_DEFAULT = 200

# Fixed integration limits covering the full passband.
# OA azimuth is sampled at 0.1 deg in [|az|=20,30] (the SG/OA transition region where
# transmission rises by ~5 orders of magnitude) and 1 deg from there to ±150.
_EL_FIXED = np.linspace(-15.0, 15.0, 301)           # 0.1 deg
_AZ_SG_FIXED = np.linspace(-20.0, 20.0, 401)        # 0.1 deg
_AZ_OA_POS_FIXED = np.concatenate([
    np.linspace(20.0, 30.0, 101),                   # 0.1 deg in transition (101 pts)
    np.arange(31.0, 151.0, 1.0),                    # 1 deg in bulk (120 pts: 31..150)
])
_AZ_OA_NEG_FIXED = np.concatenate([
    np.arange(-150.0, -30.0, 1.0),                  # 1 deg in bulk (120 pts: -150..-31)
    np.linspace(-30.0, -20.0, 101),                 # 0.1 deg in transition (101 pts)
])
_N_SP_FIXED = 50


def passband_np(grid, is_sg, el_arr, sp_arr):
    """Bilinear interpolation of the passband grid. Returns shape (len(el_arr), len(sp_arr))."""
    gv = grid.values_sunglasses if is_sg else grid.values_open_aperture
    i = (el_arr[:, None] - grid.min_elevation) / grid.elevation_spacing
    j = (sp_arr[None, :] / grid.central_speed - grid.min_speed_ratio) / grid.speed_ratio_spacing
    mask = (i >= 0) & (i + 1 < gv.shape[0]) & (j >= 0) & (j + 1 < gv.shape[1])
    i0 = np.clip(i.astype(int), 0, gv.shape[0] - 2)
    j0 = np.clip(j.astype(int), 0, gv.shape[1] - 2)
    wi, wj = i - i0, j - j0
    return np.where(
        mask,
        (1 - wi) * ((1 - wj) * gv[i0, j0] + wj * gv[i0, j0 + 1])
        + wi * ((1 - wj) * gv[i0 + 1, j0] + wj * gv[i0 + 1, j0 + 1]),
        0.0,
    )


def transmission_np(grid, az_arr):
    az = (az_arr + 180) % 360 - 180
    i = np.abs(az) / grid.azimuthal_transmission_spacing
    i0 = np.clip(i.astype(int), 0, len(grid.azimuthal_transmission) - 1)
    i1 = np.clip(i0 + 1, 0, len(grid.azimuthal_transmission) - 1)
    return grid.azimuthal_transmission[i0] * (i1 - i) + grid.azimuthal_transmission[i1] * (i - i0)


def reference_integral_fixed_limits(grid, sw) -> float:
    """
    Ground-truth integral with fixed limits covering the full passband at high resolution.

    Fixed limits (no dynamic clipping):
      elevation:   -15 to 15 deg   at 0.1 deg  (301 pts)
      azimuth SG:  -20 to 20 deg   at 0.1 deg  (401 pts)
      azimuth OA: 0.1 deg in transition |az| ∈ [20, 30], 1 deg in bulk to ±150 (221 pts/side)
      speed: 50 samples from 0.9 to 1.1 × central_speed
    """
    sp = np.linspace(0.9 * grid.central_speed, 1.1 * grid.central_speed, _N_SP_FIXED)
    sin_be = math.sin(math.radians(sw.bulk_elevation))
    cos_be = math.cos(math.radians(sw.bulk_elevation))

    count_rate = 0.0
    for is_sg, az in [(True, _AZ_SG_FIXED), (False, _AZ_OA_NEG_FIXED), (False, _AZ_OA_POS_FIXED)]:
        passband_norm = float(passband_np(grid, is_sg, np.array([0.0]), np.array([grid.central_speed]))[0, 0])
        if passband_norm == 0.0:
            continue

        el = _EL_FIXED
        cos_el = np.cos(np.radians(el))
        sin_el = np.sin(np.radians(el))
        trans = transmission_np(grid, az)
        pb_sp3 = passband_np(grid, is_sg, el, sp) * sp[None, :] ** 3 / passband_norm
        cos_alpha = (
            sin_be * sin_el[:, None]
            + cos_be * cos_el[:, None] * np.cos(np.radians(az[None, :] - sw.bulk_azimuth))
        )  # (n_el, n_az)
        exp_vals = np.exp(
            -(sp[None, None, :] ** 2 + sw.bulk_speed ** 2
              - 2 * sp[None, None, :] * sw.bulk_speed * cos_alpha[:, :, None])
            / (2 * sw.thermal_speed ** 2)
        )  # (n_el, n_az, n_sp)

        integrand = cos_el[:, None, None] * trans[None, :, None] * pb_sp3[:, None, :] * exp_vals
        count_rate += (
            np.trapezoid(np.trapezoid(np.trapezoid(integrand, sp, axis=2), az, axis=1), el, axis=0)
            * grid.central_effective_area * sw.density
            * (np.sqrt(2 * np.pi) * sw.thermal_speed) ** -3
            * 1e5 * (math.pi / 180) ** 2
        )

    return float(count_rate)


def integral(grid, sw, n_el=_N_DEFAULT, n_az_sg=_N_DEFAULT, n_az_oa=_N_DEFAULT, n_sp=_N_DEFAULT):
    """Triple trapezoid integral matching the optimized calculate_integral logic but with arbitrary resolution."""
    sin_be = math.sin(math.radians(sw.bulk_elevation))
    cos_be = math.cos(math.radians(sw.bulk_elevation))

    maxw_lo = sw.bulk_speed - 5 * sw.thermal_speed
    maxw_hi = sw.bulk_speed + 5 * sw.thermal_speed

    count_rate = 0.0
    for region in (0, -1, 1):
        is_sg = region == 0
        min_el, max_el, min_az, max_az = _get_angular_limits(sw, region, grid)
        if max_el <= min_el or max_az <= min_az:
            continue

        passband_norm = float(passband_np(grid, is_sg, np.array([0.0]), np.array([grid.central_speed]))[0, 0])
        if passband_norm == 0.0:
            continue

        n_az = n_az_sg if is_sg else n_az_oa
        el = np.linspace(min_el, max_el, n_el)
        az = np.linspace(min_az, max_az, n_az)

        bnd_lo = grid.min_SG_boundary if is_sg else grid.min_OA_boundary
        bnd_hi = grid.max_SG_boundary if is_sg else grid.max_OA_boundary
        pb_lo = grid.central_speed * eval_boundary_min(bnd_lo, el)
        pb_hi = grid.central_speed * eval_boundary_max(bnd_hi, el)
        v_lo = np.maximum(pb_lo, maxw_lo)
        v_hi = np.minimum(pb_hi, maxw_hi)
        v_hi = np.where(v_hi < v_lo, v_lo, v_hi)
        # speeds[i, :] = linspace(v_lo[i], v_hi[i], n_sp)
        t = np.linspace(0.0, 1.0, n_sp)
        sp = v_lo[:, None] + (v_hi - v_lo)[:, None] * t[None, :]  # shape (n_el, n_sp)

        cos_el = np.cos(np.radians(el))
        sin_el = np.sin(np.radians(el))
        trans = transmission_np(grid, az)
        # passband_np expects 1D el, 1D sp; per-elevation sp means we evaluate row-by-row
        pb_sp3 = np.empty((n_el, n_sp))
        for i in range(n_el):
            pb_sp3[i] = passband_np(grid, is_sg, el[i:i + 1], sp[i])[0] * sp[i] ** 3 / passband_norm
        cos_alpha = (
            sin_be * sin_el[:, None]
            + cos_be * cos_el[:, None] * np.cos(np.radians(az[None, :] - sw.bulk_azimuth))
        )
        exp_vals = np.exp(
            -(sp[:, None, :] ** 2 + sw.bulk_speed ** 2
              - 2 * sp[:, None, :] * sw.bulk_speed * cos_alpha[:, :, None])
            / (2 * sw.thermal_speed ** 2)
        )

        integrand = cos_el[:, None, None] * trans[None, :, None] * pb_sp3[:, None, :] * exp_vals
        # Trapezoid in v with per-elevation x; then in az; then in el
        # speed integral per (el, az): integrate integrand[i, j, :] over sp[i]
        speed_integral = np.empty((n_el, n_az))
        for i in range(n_el):
            speed_integral[i] = np.trapezoid(integrand[i], sp[i], axis=-1)
        az_integral = np.trapezoid(speed_integral, az, axis=1)
        count_rate_region = (
            np.trapezoid(az_integral, el)
            * grid.central_effective_area * sw.density
            * (np.sqrt(2 * np.pi) * sw.thermal_speed) ** -3
            * 1e5 * (math.pi / 180) ** 2
        )
        count_rate += count_rate_region

    return count_rate
