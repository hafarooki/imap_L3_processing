"""Alpha solar wind moments fitter (Stage 2 of the two-stage proton-frozen scheme).

Reuses the proton fitter's `_model_count_rates` (now species-parameterized) to model the
alpha contribution to the count rate. Stage 1 — the proton fit on the same chunk — is
performed by the caller and passed in as `proton_moments`. Stage 2 here is a 3-DOF
Levenberg–Marquardt over (n_α, T_α, Δv) where v_α = v_p* + Δv * B̂. The combined
observed model is `deadtime(proton_true + alpha_true)` so deadtime acts on the sum.

When magnetic data is unavailable, the fit uses the nominal Parker spiral direction
(45° from R toward −T in RTN) and flags the result with ALPHA_MAG_DATA_FALLBACK.

See `docs/swapi/solar-wind-moments.md` § "Alpha Particle Moments".
"""

from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np
import scipy.optimize
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    _model_count_rates,
    apply_deadtime_correction_array,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_K_FACTOR,
    get_alpha_peak_indices,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


@dataclass
class AlphaSolarWindMoments:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    delta_v: UFloat  # km/s, signed; +Δv ⇔ alpha drifts along +B̂ vs proton frame
    bad_fit_flag: int

    def bulk_velocity_rtn_nominal(self) -> ndarray:
        """Nominal RTN velocity vector (km/s); shape (3,)."""
        return np.array([v.nominal_value for v in self.bulk_velocity_rtn])

    def bulk_velocity_rtn_covariance(self) -> ndarray:
        """3×3 RTN velocity covariance (km²/s²); = Σ_vp + σ_Δv² B̂B̂ᵀ."""
        return np.array(covariance_matrix(self.bulk_velocity_rtn))


def _nan_alpha_moments(flag: int) -> AlphaSolarWindMoments:
    nan = ufloat(np.nan, np.nan)
    return AlphaSolarWindMoments(
        density=nan,
        temperature=nan,
        bulk_velocity_rtn=(nan, nan, nan),
        delta_v=nan,
        bad_fit_flag=int(flag),
    )


@numba.njit(nogil=True)
def _alpha_residuals_njit(
    x,
    proton_bulk,
    b_hat_rtn,
    proton_true_rate,
    count_rate,
    sigma,
    passband_grids,
    alpha_central_speeds,
    alpha_central_eff_areas,
    az_trans,
    az_trans_spacing,
    rotation_matrices,
):
    n_a = np.exp(x[0])
    T_a = np.exp(x[1])
    dv = x[2]
    v_a_rtn = proton_bulk + dv * b_hat_rtn
    alpha_true = _model_count_rates(
        n_a,
        T_a,
        v_a_rtn,
        passband_grids,
        alpha_central_speeds,
        alpha_central_eff_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
        ALPHA_PARTICLE_MASS_KG,
    )
    combined_obs = apply_deadtime_correction_array(proton_true_rate + alpha_true)
    return (combined_obs - count_rate) / sigma


def fit_solar_wind_alpha_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    measurement_time: ndarray,
    swapi_response: SWAPIResponse,
    proton_moments: ProtonSolarWindMoments,
    b_hat_rtn: ndarray,
    alpha_effective_area_scale: float,
    proton_effective_area_scale: float,
    rotation_matrices: Optional[ndarray] = None,
) -> AlphaSolarWindMoments:
    """Fit (n_α, T_α, Δv) given proton moments held fixed.

    ``count_rate`` / ``esa_voltage`` / ``measurement_time`` are flattened over (sweep, bin)
    with shape ``(n_sweeps × n_bins,)``. The plan recommends 5 sweeps × 62 coarse bins =
    310 residuals. Sweep ordering is the caller's responsibility (must match for all three).

    ``alpha_effective_area_scale = ε_α(t) / ε_p(t_lab)`` (note the proton-lab denominator
    even for alphas — see `solar-wind-moments.md` § "Alpha Particle Moments").

    ``b_hat_rtn`` is the unit MAG vector at the chunk center, rotated to RTN. If MAG data
    is unavailable (NaN or near-zero), the fit uses the nominal Parker spiral direction
    B̂ = (1/√2, −1/√2, 0) RTN and sets ``bad_fit_flag |= ALPHA_MAG_DATA_FALLBACK``.
    If the proton speed is also near-zero, returns ``bad_fit_flag = MAG_GAP`` with NaN
    moments.

    ``rotation_matrices`` may be precomputed and reused from the Stage 1 proton fit;
    if ``None``, computed internally from ``measurement_time``.
    """
    # Guard: stage 1 failed → don't trust v_p*.
    if int(proton_moments.bad_fit_flag) != int(SwapiL3Flags.NONE):
        return _nan_alpha_moments(SwapiL3Flags.STALE_PROTON)

    proton_bulk_rtn = proton_moments.bulk_velocity_rtn_nominal()
    proton_speed = np.linalg.norm(proton_bulk_rtn)
    mag_gap_fallback = False

    # If MAG data unavailable, assume alpha direction matches proton direction.
    # b_hat_rtn should be a unit vector; check for non-finite values, wrong magnitude,
    # or if _compute_b_hat_rtn already returned NaN due to data gaps or fill values.
    b_hat_check = np.asarray(b_hat_rtn, dtype=float)
    b_norm = np.linalg.norm(b_hat_check)
    # Unit vector should have norm ≈1; allow small tolerance for numerical error.
    is_valid_unit_vector = np.all(np.isfinite(b_hat_check)) and 0.99 < b_norm < 1.01
    if not is_valid_unit_vector:
        if proton_speed < 1e-12:
            return _nan_alpha_moments(SwapiL3Flags.MAG_GAP)
        b_hat_rtn = np.array([1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0])
        mag_gap_fallback = True
    else:
        b_hat_rtn = b_hat_check

    # SPICE shared with Stage 1 if provided; otherwise compute here.
    if rotation_matrices is None:
        from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

        rotation_matrices = get_swapi_geometry(measurement_time)

    # V-only passband grids cached by V; species/V/time-dependent scalars are per-measurement.
    passband_grids = numba.typed.List(
        [swapi_response.create_passband_grid(v) for v in esa_voltage]
    )
    central_effective_areas_lab = np.array(
        [swapi_response.get_central_effective_area(v) for v in esa_voltage]
    )
    az_trans = np.asarray(swapi_response.azimuthal_transmission, dtype=float)
    az_trans_spacing = float(swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    proton_central_speeds = np.array(
        [
            swapi_response.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E)
            for v in esa_voltage
        ]
    )
    alpha_central_speeds = np.array(
        [
            swapi_response.central_speed(v, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
            for v in esa_voltage
        ]
    )
    proton_central_eff_areas = central_effective_areas_lab * float(
        proton_effective_area_scale
    )
    alpha_central_eff_areas = central_effective_areas_lab * float(
        alpha_effective_area_scale
    )

    # Frozen pre-deadtime proton model rate. Deadtime acts on (proton + alpha) below.
    proton_true_rate = _model_count_rates(
        proton_moments.density.nominal_value,
        proton_moments.temperature.nominal_value,
        proton_bulk_rtn,
        passband_grids,
        proton_central_speeds,
        proton_central_eff_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
        PROTON_MASS_KG,
    )

    # Initial guess from the alpha bump after subtracting the deadtime-applied proton bg.
    initial_guess = _alpha_initial_guess(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        proton_true_rate=proton_true_rate,
        proton_temperature=proton_moments.temperature.nominal_value,
        passband_grids=passband_grids,
        alpha_central_speeds=alpha_central_speeds,
        alpha_central_eff_areas=alpha_central_eff_areas,
        az_trans=az_trans,
        az_trans_spacing=az_trans_spacing,
        rotation_matrices=rotation_matrices,
        proton_bulk_velocity_rtn=proton_bulk_rtn,
        b_hat_rtn=b_hat_rtn,
    )
    if initial_guess is None:
        return _nan_alpha_moments(SwapiL3Flags.HI_CHI_SQ)

    n0, T0, dv0, peak_bin_idx = initial_guess
    proton_bulk = proton_bulk_rtn

    # Subset all per-measurement arrays to only the alpha peak bins across
    # all sweeps, so LM fits the alpha bump rather than the proton-dominated
    # tails (which create an n↓/T↑ degeneracy).
    n_sweeps, n_bins = _infer_sweep_layout(esa_voltage)
    peak_flat_idx = np.concatenate([peak_bin_idx + s * n_bins for s in range(n_sweeps)])
    count_rate_peak = count_rate[peak_flat_idx]
    keep = count_rate_peak > 0
    if not np.all(keep):
        peak_flat_idx = peak_flat_idx[keep]
        count_rate_peak = count_rate_peak[keep]
    proton_true_rate_peak = proton_true_rate[peak_flat_idx]
    alpha_central_speeds_peak = alpha_central_speeds[peak_flat_idx]
    alpha_central_eff_areas_peak = alpha_central_eff_areas[peak_flat_idx]
    rotation_matrices_peak = rotation_matrices[peak_flat_idx]
    passband_grids_peak = numba.typed.List([passband_grids[i] for i in peak_flat_idx])

    sigma_peak = np.ones(len(count_rate_peak))

    def residuals(x):
        return _alpha_residuals_njit(
            x,
            proton_bulk,
            b_hat_rtn,
            proton_true_rate_peak,
            count_rate_peak,
            sigma_peak,
            passband_grids_peak,
            alpha_central_speeds_peak,
            alpha_central_eff_areas_peak,
            az_trans,
            az_trans_spacing,
            rotation_matrices_peak,
        )

    x0 = np.array([np.log(max(n0, 1e-3)), np.log(max(T0, 1e-3)), dv0])
    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)

    # Wrong-basin: signed-Δv flip (1-DOF basin ambiguity along B̂).
    chi2 = float(np.sum(result.fun**2))
    x_flipped = result.x.copy()
    x_flipped[2] = -x_flipped[2]
    chi2_flipped = float(np.sum(residuals(x_flipped) ** 2))
    if chi2_flipped < chi2:
        result = scipy.optimize.least_squares(
            residuals, x_flipped, method="lm", diff_step=1e-4
        )

    n_a_fit = float(np.exp(result.x[0]))
    T_a_fit = float(np.exp(result.x[1]))
    dv_fit = float(result.x[2])
    bulk_velocity_rtn = proton_bulk + dv_fit * b_hat_rtn
    bad_fit_flag = SwapiL3Flags.NONE if result.success else SwapiL3Flags.HI_CHI_SQ
    if mag_gap_fallback:
        bad_fit_flag |= SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK

    # Covariance in (log n, log T, Δv) space, scaled by reduced chi² (fitting error).
    n_data, n_params = len(result.fun), len(result.x)
    s_sq = float(np.sum(result.fun**2)) / max(n_data - n_params, 1)
    cov_x = s_sq * np.linalg.pinv(result.jac.T @ result.jac)
    density_sigma = float(n_a_fit * np.sqrt(max(cov_x[0, 0], 0.0)))
    temperature_sigma = float(T_a_fit * np.sqrt(max(cov_x[1, 1], 0.0)))
    delta_v_sigma = float(np.sqrt(max(cov_x[2, 2], 0.0)))

    # Σ_vα = Σ_vp + σ_Δv² B̂B̂ᵀ (additive Δv along the 1-DOF B̂ axis).
    sigma_dv2 = max(cov_x[2, 2], 0.0)
    velocity_covariance_rtn = (
        proton_moments.bulk_velocity_rtn_covariance()
        + sigma_dv2 * np.outer(b_hat_rtn, b_hat_rtn)
    )

    return AlphaSolarWindMoments(
        density=ufloat(n_a_fit, density_sigma),
        temperature=ufloat(T_a_fit, temperature_sigma),
        bulk_velocity_rtn=make_correlated_velocity(
            bulk_velocity_rtn, velocity_covariance_rtn
        ),
        delta_v=ufloat(dv_fit, delta_v_sigma),
        bad_fit_flag=int(bad_fit_flag),
    )


def _alpha_initial_guess(
    count_rate: ndarray,
    esa_voltage: ndarray,
    proton_true_rate: ndarray,
    proton_temperature: float,
    passband_grids: numba.typed.List,
    alpha_central_speeds: ndarray,
    alpha_central_eff_areas: ndarray,
    az_trans: ndarray,
    az_trans_spacing: float,
    rotation_matrices: ndarray,
    proton_bulk_velocity_rtn: ndarray,
    b_hat_rtn: ndarray,
) -> Optional[tuple]:
    """Return (n_α, T_α, Δv=0, peak_bin_indices) as a starting point for LM, or None if peak-finding fails."""

    n_meas = len(esa_voltage)
    if n_meas == 0:
        return None

    n_sweeps, n_bins = _infer_sweep_layout(esa_voltage)
    if n_sweeps is None:
        return None

    counts_per_sweep = count_rate.reshape(n_sweeps, n_bins)
    voltage_per_sweep = esa_voltage.reshape(n_sweeps, n_bins)[0]
    proton_obs_per_sweep = apply_deadtime_correction_array(
        proton_true_rate.reshape(n_sweeps, n_bins)
    )

    count_avg = counts_per_sweep.mean(axis=0)
    proton_bg_avg = proton_obs_per_sweep.mean(axis=0)
    energies_per_sweep = SWAPI_K_FACTOR * np.abs(voltage_per_sweep)
    proton_peak_index = np.argmax(proton_bg_avg)
    residual = np.maximum(0, count_avg - proton_bg_avg * 2)

    try:
        peak = get_alpha_peak_indices(residual, energies_per_sweep, proton_peak_index)
    except Exception:
        return None

    peak_idx = np.arange(peak.start, peak.stop)
    if len(peak_idx) < 3:
        return None
    residual_peak = np.maximum(residual[peak_idx], 0.0)
    if not np.any(residual_peak > 0):
        return None

    T_alpha = proton_temperature

    unit_alpha = _model_count_rates(
        1.0,
        T_alpha,
        proton_bulk_velocity_rtn,
        passband_grids,
        alpha_central_speeds,
        alpha_central_eff_areas,
        az_trans,
        az_trans_spacing,
        rotation_matrices,
        ALPHA_PARTICLE_MASS_KG,
    )
    unit_alpha_per_sweep = unit_alpha.reshape(n_sweeps, n_bins).mean(axis=0)
    denom = float(np.nanmean(unit_alpha_per_sweep[peak_idx]))
    if denom <= 0 or not np.isfinite(denom):
        return None
    n_alpha = float(np.nanmean(residual_peak)) / denom
    n_alpha = max(n_alpha, 1e-3)

    return (n_alpha, T_alpha, 0.0, peak_idx)


def _infer_sweep_layout(esa_voltage: ndarray) -> tuple:
    """Heuristic: detect n_sweeps from a periodic voltage axis."""
    n_meas = len(esa_voltage)
    for n_sweeps in (5, 1, 2, 3, 4, 6, 7, 8, 10):
        if n_meas % n_sweeps != 0:
            continue
        n_bins = n_meas // n_sweeps
        first = esa_voltage[:n_bins]
        if np.allclose(esa_voltage.reshape(n_sweeps, n_bins), first):
            return n_sweeps, n_bins
    return None, None
