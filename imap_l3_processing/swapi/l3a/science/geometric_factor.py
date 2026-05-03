"""Dynamic geometric factor G(V, q) computed from SWAPIResponse.

G^s(V) = (2 k* q |V|) · A_0(V) · I(V) · (π/180)²

where I(V) is the species-independent dimensionless integral
∫ r · cos(θ) · T(|φ|) · P(r, θ, φ, V) dr dθ dφ
over both aperture regions (SG + OA).

Result units: cm² · sr · eV.
"""

from __future__ import annotations

import numpy as np

from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import (
    SWAPIResponse,
    _TARGET_ELEVATIONS,
    _TARGET_SPEED_RATIOS,
)

_ELEVATION_SPACING = float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0])
_SPEED_RATIO_SPACING = float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0])
_COS_ELEVATIONS = np.cos(np.radians(_TARGET_ELEVATIONS))


def calculate_geometric_factor(
    swapi_response: SWAPIResponse,
    esa_voltage: float,
    charge_per_proton_charge: float = 1.0,
) -> float:
    """G(V, q) in cm² · sr · eV."""
    cache_key = (float(esa_voltage), float(charge_per_proton_charge))
    cached = swapi_response._gf_cache.get(cache_key)
    if cached is not None:
        return cached

    abs_v = abs(esa_voltage)
    prefactor = 2.0 * SWAPI_K_FACTOR * abs_v * charge_per_proton_charge
    a0 = swapi_response.get_central_effective_area(abs_v)
    grid = swapi_response.create_passband_grid(abs_v)

    az_trans = swapi_response.azimuthal_transmission
    az_spacing = swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG

    passband_norm_sg = _interp_passband(grid.values_sunglasses, grid, 0.0, 1.0)
    passband_norm_oa = _interp_passband(grid.values_open_aperture, grid, 0.0, 1.0)

    integral = 0.0

    for is_sg, az_lo, az_hi in [(True, -20.0, 20.0), (False, -150.0, 150.0)]:
        pb_values = grid.values_sunglasses if is_sg else grid.values_open_aperture
        pb_norm = passband_norm_sg if is_sg else passband_norm_oa
        if pb_norm <= 0.0:
            continue

        az_angles = np.arange(az_lo, az_hi + az_spacing, az_spacing)
        az_angles = az_angles[az_angles <= az_hi]
        trans_values = np.array(
            [_interp_transmission(az_trans, az_spacing, az) for az in az_angles]
        )
        az_integral = float(np.trapz(trans_values, az_angles))

        normalized_pb = pb_values / pb_norm
        integrand_2d = (
            _COS_ELEVATIONS[:, np.newaxis]
            * _TARGET_SPEED_RATIOS[np.newaxis, :]
            * normalized_pb
        )

        el_speed_integral = np.trapz(
            np.trapz(integrand_2d, _TARGET_SPEED_RATIOS, axis=1), _TARGET_ELEVATIONS
        )

        integral += az_integral * float(el_speed_integral)

    deg2_to_sr = (np.pi / 180.0) ** 2
    result = prefactor * a0 * integral * deg2_to_sr

    swapi_response._gf_cache[cache_key] = result
    return result


def _interp_transmission(az_trans, spacing, azimuth):
    azimuth = (azimuth + 180.0) % 360.0 - 180.0
    idx_f = abs(azimuth) / spacing
    i_lo = int(np.floor(idx_f))
    i_hi = i_lo + 1
    n = len(az_trans)
    i_lo = min(max(i_lo, 0), n - 1)
    i_hi = min(max(i_hi, 0), n - 1)
    w_lo = float(i_hi) - idx_f
    w_hi = idx_f - float(i_lo)
    return az_trans[i_lo] * w_lo + az_trans[i_hi] * w_hi


def _interp_passband(values, grid, elevation, speed_ratio):
    i_f = (elevation - grid.min_elevation) / grid.elevation_spacing
    if i_f < 0 or i_f + 1 >= values.shape[0]:
        return 0.0
    j_f = (speed_ratio - grid.min_speed_ratio) / grid.speed_ratio_spacing
    if j_f < 0 or j_f + 1 >= values.shape[1]:
        return 0.0
    i_lo = int(i_f)
    i_hi = i_lo + 1
    iw = i_f - i_lo
    j_lo = int(j_f)
    j_hi = j_lo + 1
    jw = j_f - j_lo
    return (1 - iw) * ((1 - jw) * values[i_lo, j_lo] + jw * values[i_lo, j_hi]) + iw * (
        (1 - jw) * values[i_hi, j_lo] + jw * values[i_hi, j_hi]
    )
