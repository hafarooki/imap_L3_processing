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
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    apply_deadtime_correction_array,
    model_solar_wind_ideal_coincidence_rates,
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.proton_uncertainties import (
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_K_FACTOR,
    get_alpha_peak_indices,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
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
    magnetic_field_direction,
    proton_true_rate,
    alpha_ctx,
):
    n_a = np.exp(x[0])
    T_a = np.exp(x[1])
    dv = x[2]
    v_a_rtn = proton_bulk + dv * magnetic_field_direction
    alpha_true = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(n_a, v_a_rtn, T_a, alpha_ctx.mass_kg), alpha_ctx,
    )
    combined_obs = apply_deadtime_correction_array(proton_true_rate + alpha_true)
    return combined_obs - alpha_ctx.count_rate


def fit_solar_wind_alpha_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    measurement_time: ndarray,
    swapi_response: SwapiResponse,
    proton_moments: ProtonSolarWindFitResult,
    magnetic_field_direction: ndarray,
    alpha_effective_area_scale: float,
    proton_effective_area_scale: float,
    rotation_matrices: Optional[ndarray] = None,
) -> AlphaSolarWindMoments:
    # Guard: stage 1 failed → don't trust v_p*.
    if int(proton_moments.bad_fit_flag) != int(SwapiL3Flags.NONE):
        return _nan_alpha_moments(SwapiL3Flags.STALE_PROTON)

    proton_bulk_rtn = proton_moments.bulk_velocity_rtn_nominal()
    bad_fit_flag = SwapiL3Flags.NONE

    # MAG required: per-chunk gaps (NaN b_hat) → BAD_FIT.
    # No Parker-spiral substitution; MAG presence at file level is enforced
    # by the alpha-sw processor branch.
    magnetic_field_direction = np.asarray(magnetic_field_direction, dtype=float)
    if not np.all(np.isfinite(magnetic_field_direction)):
        return _nan_alpha_moments(SwapiL3Flags.BAD_FIT)

    # SPICE shared with Stage 1 if provided; otherwise compute here.
    if rotation_matrices is None:
        from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

        rotation_matrices = get_swapi_geometry(measurement_time)

    proton_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=swapi_response,
        central_effective_area_scale=proton_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=swapi_response,
        central_effective_area_scale=alpha_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )

    # Frozen pre-deadtime proton model rate. Deadtime acts on (proton + alpha) below.
    proton_true_rate = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(
            density=proton_moments.density.nominal_value,
            bulk_velocity_rtn=proton_bulk_rtn,
            temperature=proton_moments.temperature.nominal_value,
            mass_kg=proton_ctx.mass_kg,
        ),
        proton_ctx,
    )

    # Initial guess from the alpha bump after subtracting the deadtime-applied proton bg.
    initial_guess = _alpha_initial_guess(
        proton_true_rate=proton_true_rate,
        proton_temperature=proton_moments.temperature.nominal_value,
        alpha_ctx=alpha_ctx,
        proton_bulk_velocity_rtn=proton_bulk_rtn,
        magnetic_field_direction=magnetic_field_direction,
    )
    if initial_guess is None:
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.BAD_FIT)

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
    proton_true_rate_peak = proton_true_rate[peak_flat_idx]
    alpha_ctx_peak = alpha_ctx.subset(peak_flat_idx)

    def residuals(x):
        return _alpha_residuals_njit(
            x,
            proton_bulk,
            magnetic_field_direction,
            proton_true_rate_peak,
            alpha_ctx_peak,
        )

    x0 = np.array([np.log(max(n0, 1e-3)), np.log(max(T0, 1e-3)), dv0])
    result = scipy.optimize.least_squares(residuals, x0, method="lm", diff_step=1e-4)

    # Wrong-basin: signed-Δv flip (1-DOF basin ambiguity along B̂).
    mse = float(np.mean(result.fun**2))
    x_flipped = result.x.copy()
    x_flipped[2] = -x_flipped[2]
    mse_flipped = float(np.mean(residuals(x_flipped) ** 2))
    if mse_flipped < mse:
        result = scipy.optimize.least_squares(
            residuals, x_flipped, method="lm", diff_step=1e-4
        )

    n_a_fit = float(np.exp(result.x[0]))
    T_a_fit = float(np.exp(result.x[1]))
    dv_fit = float(result.x[2])
    bulk_velocity_rtn = proton_bulk + dv_fit * magnetic_field_direction
    if not result.success:
        bad_fit_flag |= SwapiL3Flags.BAD_FIT

    # Covariance in (log n, log T, Δv) space, scaled by residual variance s² = Σr²/(N−p).
    n_data, n_params = len(result.fun), len(result.x)
    residual_variance = float(np.sum(result.fun**2)) / max(n_data - n_params, 1)
    cov_x = residual_variance * np.linalg.pinv(result.jac.T @ result.jac)
    density_sigma = float(n_a_fit * np.sqrt(max(cov_x[0, 0], 0.0)))
    temperature_sigma = float(T_a_fit * np.sqrt(max(cov_x[1, 1], 0.0)))
    delta_v_sigma = float(np.sqrt(max(cov_x[2, 2], 0.0)))

    # Σ_vα = Σ_vp + σ_Δv² B̂B̂ᵀ (additive Δv along the 1-DOF B̂ axis).
    sigma_dv2 = max(cov_x[2, 2], 0.0)
    velocity_covariance_rtn = (
        proton_moments.bulk_velocity_rtn_covariance()
        + sigma_dv2 * np.outer(magnetic_field_direction, magnetic_field_direction)
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
    proton_true_rate: ndarray,
    proton_temperature: float,
    alpha_ctx: SolarWindFitContext,
    proton_bulk_velocity_rtn: ndarray,
    magnetic_field_direction: ndarray,
) -> Optional[tuple]:
    """Return (n_α, T_α, Δv=0, peak_bin_indices) as a starting point for LM, or None if peak-finding fails."""

    n_meas = len(alpha_ctx.esa_voltage)
    if n_meas == 0:
        return None

    n_sweeps, n_bins = _infer_sweep_layout(alpha_ctx.esa_voltage)
    if n_sweeps is None:
        return None

    counts_per_sweep = alpha_ctx.count_rate.reshape(n_sweeps, n_bins)
    voltage_per_sweep = alpha_ctx.esa_voltage.reshape(n_sweeps, n_bins)[0]
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

    unit_alpha = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, proton_bulk_velocity_rtn, T_alpha, alpha_ctx.mass_kg),
        alpha_ctx,
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
