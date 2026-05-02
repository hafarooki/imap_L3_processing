"""
Numba reference for the SWAPI proton solar wind integral with fixed limits.
Mirrors the math in docs/swapi/solar-wind-moments.md.

  * ``reference_integral_fixed_limits(grid, sw)`` — single sample.
  * ``reference_integrals_batch(grids, sws)``     — parallel across samples.

Fixed integration grid:
  elevation:  -15 to 15 deg @ 0.1° (301 pts)
  azimuth SG: -20 to 20 deg @ 0.1° (401 pts)
  azimuth OA: 0.1° in transition |az| ∈ [20, 30], 1° to ±150° (221 pts/side)
              — split at the |az| < 20° dead band so trapezoid doesn't
                bridge the gap.
  speed:      50 samples from 0.9 to 1.1 × central_speed
"""

import math

import numpy as np
from numba import njit, prange


_EL = np.linspace(-15.0, 15.0, 301)
_AZ_SG = np.linspace(-20.0, 20.0, 401)
_AZ_OA_NEG = np.concatenate(
    [np.arange(-150.0, -30.0, 1.0), np.linspace(-30.0, -20.0, 101)]
)
_AZ_OA_POS = np.concatenate([np.linspace(20.0, 30.0, 101), np.arange(31.0, 151.0, 1.0)])
_SP_RATIO = np.linspace(0.9, 1.1, 50)


def _trap_weights(x):
    """Per-point trapezoid weights on a 1-D grid (handles non-uniform spacing)."""
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


_EL_W = _trap_weights(_EL)
_AZ_SG_W = _trap_weights(_AZ_SG)
_AZ_OA_NEG_W = _trap_weights(_AZ_OA_NEG)
_AZ_OA_POS_W = _trap_weights(_AZ_OA_POS)
_SP_RATIO_W = _trap_weights(_SP_RATIO)


@njit(fastmath=True, inline="always")
def _passband(gv, min_el, el_sp, min_r, r_sp, vc, el, v):
    i = (el - min_el) / el_sp
    j = (v / vc - min_r) / r_sp
    if i < 0 or i + 1 >= gv.shape[0] or j < 0 or j + 1 >= gv.shape[1]:
        return 0.0
    i0, j0 = int(i), int(j)
    wi, wj = i - i0, j - j0
    return (1 - wi) * ((1 - wj) * gv[i0, j0] + wj * gv[i0, j0 + 1]) + wi * (
        (1 - wj) * gv[i0 + 1, j0] + wj * gv[i0 + 1, j0 + 1]
    )


@njit(fastmath=True, inline="always")
def _trans(t, t_sp, az):
    f = abs((az + 180.0) % 360.0 - 180.0) / t_sp
    n = t.shape[0]
    i0 = int(f)
    if i0 < 0:
        i0 = 0
    elif i0 >= n:
        i0 = n - 1
    i1 = i0 + 1 if i0 + 1 < n else n - 1
    return t[i0] * (i1 - f) + t[i1] * (f - i0)


@njit(fastmath=True)
def _region(
    gv,
    gv_norm,
    density,
    bulk_v,
    bulk_az,
    sin_be,
    cos_be,
    th,
    vc,
    ca,
    t,
    t_sp,
    min_el,
    el_sp,
    min_r,
    r_sp,
    el,
    ew,
    az,
    aw,
    sp,
    spw,
):
    if gv_norm <= 0.0:
        return 0.0
    inv_2th2 = 1.0 / (2.0 * th * th)
    total = 0.0
    for i in range(el.shape[0]):
        sin_el = math.sin(math.radians(el[i]))
        cos_el = math.cos(math.radians(el[i]))
        for j in range(az.shape[0]):
            cos_alpha = sin_be * sin_el + cos_be * cos_el * math.cos(
                math.radians(az[j] - bulk_az)
            )
            tr = _trans(t, t_sp, az[j])
            for k in range(sp.shape[0]):
                v = sp[k]
                pb = _passband(gv, min_el, el_sp, min_r, r_sp, vc, el[i], v)
                integrand = (
                    cos_el
                    * tr
                    * pb
                    * v
                    * v
                    * v
                    / gv_norm
                    * math.exp(
                        -(v * v + bulk_v * bulk_v - 2 * v * bulk_v * cos_alpha)
                        * inv_2th2
                    )
                )
                total += ew[i] * aw[j] * spw[k] * integrand
    pre = (
        ca
        * density
        * (math.sqrt(2.0 * math.pi) * th) ** -3
        * 1e5
        * (math.pi / 180.0) ** 2
    )
    return total * pre


@njit(parallel=True, fastmath=True)
def _batch(
    sg,
    oa,
    vc,
    ca,
    density,
    bulk_v,
    bulk_az,
    bulk_el,
    th,
    t,
    t_sp,
    min_el,
    el_sp,
    min_r,
    r_sp,
    el,
    ew,
    az_sg,
    az_sg_w,
    az_oa_neg,
    az_oa_neg_w,
    az_oa_pos,
    az_oa_pos_w,
    sp_ratio,
    sp_ratio_w,
):
    n = vc.shape[0]
    out = np.empty(n)
    for i in prange(n):
        sin_be = math.sin(math.radians(bulk_el[i]))
        cos_be = math.cos(math.radians(bulk_el[i]))
        sp = sp_ratio * vc[i]
        spw = sp_ratio_w * vc[i]
        sg_norm = _passband(sg[i], min_el, el_sp, min_r, r_sp, vc[i], 0.0, vc[i])
        oa_norm = _passband(oa[i], min_el, el_sp, min_r, r_sp, vc[i], 0.0, vc[i])
        args = (
            density[i],
            bulk_v[i],
            bulk_az[i],
            sin_be,
            cos_be,
            th[i],
            vc[i],
            ca[i],
            t,
            t_sp,
            min_el,
            el_sp,
            min_r,
            r_sp,
            el,
            ew,
        )
        out[i] = (
            _region(sg[i], sg_norm, *args, az_sg, az_sg_w, sp, spw)
            + _region(oa[i], oa_norm, *args, az_oa_neg, az_oa_neg_w, sp, spw)
            + _region(oa[i], oa_norm, *args, az_oa_pos, az_oa_pos_w, sp, spw)
        )
    return out


def reference_integrals_batch(
    grids,
    sws,
    central_speeds,
    central_effective_areas,
    azimuthal_transmission,
    transmission_spacing,
):
    """Compute N reference integrals in parallel via numba JIT.

    Species- and time-dependent quantities (`central_speeds`, `central_effective_areas`,
    `azimuthal_transmission`) are passed alongside the V-only `grids`, mirroring
    `calculate_integral`'s signature.
    """
    n = len(sws)
    if n == 0:
        return np.empty(0)
    nel, nsp = grids[0].values_sunglasses.shape
    sg = np.empty((n, nel, nsp))
    oa = np.empty((n, nel, nsp))
    vc = np.asarray(central_speeds, dtype=np.float64)
    ca = np.asarray(central_effective_areas, dtype=np.float64)
    density = np.empty(n)
    bv = np.empty(n)
    ba = np.empty(n)
    be = np.empty(n)
    th = np.empty(n)
    for i in range(n):
        sg[i] = grids[i].values_sunglasses
        oa[i] = grids[i].values_open_aperture
        density[i] = sws[i].density
        bv[i] = sws[i].bulk_speed
        ba[i] = sws[i].bulk_azimuth
        be[i] = sws[i].bulk_elevation
        th[i] = sws[i].thermal_speed
    g0 = grids[0]
    return _batch(
        sg,
        oa,
        vc,
        ca,
        density,
        bv,
        ba,
        be,
        th,
        np.ascontiguousarray(azimuthal_transmission, dtype=np.float64),
        float(transmission_spacing),
        float(g0.min_elevation),
        float(g0.elevation_spacing),
        float(g0.min_speed_ratio),
        float(g0.speed_ratio_spacing),
        _EL,
        _EL_W,
        _AZ_SG,
        _AZ_SG_W,
        _AZ_OA_NEG,
        _AZ_OA_NEG_W,
        _AZ_OA_POS,
        _AZ_OA_POS_W,
        _SP_RATIO,
        _SP_RATIO_W,
    )


def reference_integral_fixed_limits(
    grid,
    sw,
    central_speed,
    central_effective_area,
    azimuthal_transmission,
    transmission_spacing,
) -> float:
    """Single-sample ground truth (delegates to the parallel batch)."""
    return float(
        reference_integrals_batch(
            [grid],
            [sw],
            [central_speed],
            [central_effective_area],
            azimuthal_transmission,
            transmission_spacing,
        )[0]
    )
