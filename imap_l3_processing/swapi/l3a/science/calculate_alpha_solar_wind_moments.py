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

    ``b_hat_rtn`` is the unit MAG direction in RTN for the chunk. If MAG data is
    unavailable (NaN or near-zero), the fit uses the nominal Parker spiral direction
    B̂ = (1/√2, −1/√2, 0) RTN and sets ``bad_fit_flag |= ALPHA_MAG_DATA_FALLBACK``.
    If the reference proton velocity is non-finite or near-zero, returns NaN moments
    with ``bad_fit_flag |= BAD_FIT``.

    ``rotation_matrices`` may be precomputed and reused from the Stage 1 proton fit;
    if ``None``, computed internally from ``measurement_time``.
    """
    # Guard: stage 1 failed → don't trust v_p*.
    if int(proton_moments.bad_fit_flag) != int(SwapiL3Flags.NONE):
        return _nan_alpha_moments(SwapiL3Flags.STALE_PROTON)

    proton_bulk_rtn = proton_moments.bulk_velocity_rtn_nominal()
    bad_fit_flag = SwapiL3Flags.NONE

    # If MAG data is unavailable, use the nominal Parker spiral direction.
    # b_hat_rtn should be a unit vector; check for non-finite values, wrong magnitude,
    # or if compute_b_hat_rtn already returned NaN due to data gaps or fill values.
    b_hat_check = np.asarray(b_hat_rtn, dtype=float)
    b_norm = np.linalg.norm(b_hat_check)
    # Unit vector should have norm ≈1; allow small tolerance for numerical error.
    is_valid_unit_vector = np.all(np.isfinite(b_hat_check)) and 0.99 < b_norm < 1.01
    if not is_valid_unit_vector:
        b_hat_rtn = np.array([1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0])
        bad_fit_flag |= SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK
    else:
        b_hat_rtn = b_hat_check

    proton_speed = np.linalg.norm(proton_bulk_rtn)
    if not np.isfinite(proton_speed) or proton_speed < 1e-12:
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.BAD_FIT)

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
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.BAD_FIT)

    n0, T0, dv0, peak_bin_idx = initial_guess
    proton_bulk = proton_bulk_rtn

    n_sweeps, n_bins = _infer_sweep_layout(esa_voltage)
    proton_T = proton_moments.temperature.nominal_value

    def _run_lm(window_bin_idx: ndarray, x0_in: ndarray):
        """Subset all per-measurement arrays to ``window_bin_idx`` (a
        per-sweep bin index list, broadcast across all sweeps), run LM,
        apply the signed-Δv basin flip, and return (result, residuals_fn,
        n_data). LM fits the alpha bump on the chosen window only — the
        proton-dominated bins outside the window cause an n↓/T↑ degeneracy
        when included."""
        flat_idx = np.concatenate(
            [window_bin_idx + s * n_bins for s in range(n_sweeps)]
        )
        cr_pk = count_rate[flat_idx]
        keep = cr_pk > 0
        if not np.all(keep):
            flat_idx = flat_idx[keep]
            cr_pk = cr_pk[keep]
        pt_pk = proton_true_rate[flat_idx]
        a_cs_pk = alpha_central_speeds[flat_idx]
        a_ea_pk = alpha_central_eff_areas[flat_idx]
        rot_pk = rotation_matrices[flat_idx]
        pg_pk = numba.typed.List([passband_grids[i] for i in flat_idx])
        sigma = np.ones(len(cr_pk))

        def residuals(x):
            return _alpha_residuals_njit(
                x, proton_bulk, b_hat_rtn, pt_pk, cr_pk, sigma, pg_pk,
                a_cs_pk, a_ea_pk, az_trans, az_trans_spacing, rot_pk,
            )

        r = scipy.optimize.least_squares(residuals, x0_in, method="lm", diff_step=1e-4)
        c = float(np.sum(r.fun**2))
        x_flipped = r.x.copy()
        x_flipped[2] = -x_flipped[2]
        c_flipped = float(np.sum(residuals(x_flipped) ** 2))
        if c_flipped < c:
            r = scipy.optimize.least_squares(
                residuals, x_flipped, method="lm", diff_step=1e-4
            )
        return r, residuals, len(cr_pk)

    x0 = np.array([np.log(max(n0, 1e-3)), np.log(max(T0, 1e-3)), dv0])
    result, residuals, n_data = _run_lm(peak_bin_idx, x0)
    current_window = peak_bin_idx

    # Conditional multi-start (T₀ ∈ {2·T_p, 8·T_p}). When the default fit
    # lands in the runaway-T basin (T_α/T_p > 20), retry from cooler and
    # hotter T seeds and accept any alternative with strictly lower χ²
    # AND lower T_α than the default. The "lower T_α" gate keeps multi-
    # start from swapping a physical basin for a marginally lower-χ²
    # wrong one elsewhere in parameter space.
    T_a_default = float(np.exp(result.x[1]))
    if T_a_default > 20.0 * proton_T:
        chi2_default = float(np.sum(result.fun**2))
        for T_seed in (2.0 * proton_T, 8.0 * proton_T):
            x0_alt = np.array([np.log(max(n0, 1e-3)), np.log(T_seed), dv0])
            try:
                r_alt, res_alt, n_data_alt = _run_lm(current_window, x0_alt)
            except (RuntimeError, ValueError):
                continue
            T_a_alt = float(np.exp(r_alt.x[1]))
            chi2_alt = float(np.sum(r_alt.fun**2))
            if T_a_alt < T_a_default and chi2_alt < chi2_default:
                result = r_alt
                residuals = res_alt
                n_data = n_data_alt
                T_a_default = T_a_alt
                chi2_default = chi2_alt

    # Window-shrink refinement. Try LM on smaller windows obtained by
    # trimming one bin off each end of the current window. Accept the
    # trim only if it strictly improves per-bin SSE AND lowers T_α —
    # this dual gate keeps physical fits from being disturbed (a trim
    # that lowers per-bin SSE by going _hotter_ is a sign the fit was
    # already in the right basin). Per-bin SSE rather than total χ² is
    # the metric so windows of different sizes are comparable.
    #
    # Among the accepted candidates, candidates that drop T_α by more
    # than 30% are preferred over marginal improvers. This is what
    # rescues the cases where the current window has the LM stuck on a
    # T_α≈12·T_p collapsed-n basin and a single-bin trim from the right
    # side reveals the cool physical basin (T_α≈5–7·T_p). Without this
    # preference the greedy lowest-per-bin-SSE choice picks a tiny
    # T-drop trim that keeps the fit stuck.
    fresh_x0 = np.array([np.log(max(n0, 1e-3)), np.log(max(T0, 1e-3)), 0.0])
    # Anchor against the pre-shrink fit, so cumulative drift across many
    # iterations is bounded (each step's per-step n_ok ratio of 0.5 is
    # unreachable in practice but successive 0.93×-each-step trims still
    # cause big total collapse — cf. ci=859: T 8.4→4.5, n 0.06→0.014
    # without bound).
    n_initial = float(np.exp(result.x[0]))
    while len(current_window) > 5:
        T_now = float(np.exp(result.x[1]))
        n_now = float(np.exp(result.x[0]))
        chi2_now = float(np.sum(result.fun**2))
        per_bin_now = chi2_now / max(n_data, 1)
        # When the current fit is in the runaway-T basin (T_α > 20·T_p,
        # beyond the user-stated 19× upper limit) — relax the per-bin
        # SSE gate. Trims that escape the runaway basin can have *higher*
        # per-bin SSE than the runaway fit when the residual has
        # wider-than-Maxwellian tails. Cap the relaxation at 2× per-bin
        # SSE so we don't accept clearly worse candidates.
        unphysical_now = T_now > 20.0 * proton_T
        chi2_gate = (2.0 * per_bin_now) if unphysical_now else per_bin_now

        candidates = []
        # The (2,0)/(3,0) trim options exist for PUI He+ shelf escape:
        # cases like ci=313 have a wide-Maxwellian basin where neither
        # (1,0) nor (1,1) drop T enough to trigger the basin-transition
        # bypass below — the high-V PUI tail is 2+ bins deep, and one
        # bin trim leaves enough PUI bias to keep LM in the wide basin.
        # The bypass requires both T_t < 0.7·T_now and n_t ≥ n_now, so
        # multi-bin high-V trims are only accepted when they cross into
        # a clearly narrower-and-denser alpha-core basin.
        for trim_lo, trim_hi in (
            (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (4, 0)
        ):
            if len(current_window) - trim_lo - trim_hi < 5:
                continue
            new_window = current_window[trim_lo : len(current_window) - trim_hi]
            # Start the trim's LM from the fresh initial guess rather than
            # from the previous-window fit's parameters; if the previous
            # fit is in a wrong basin (T_α huge), restarting LM from there
            # tends to keep it stuck. The fresh T₀=4·T_p start lets each
            # candidate converge independently.
            try:
                r_t, res_t, n_t = _run_lm(new_window, fresh_x0)
            except (RuntimeError, ValueError):
                continue
            chi2_t = float(np.sum(r_t.fun**2))
            T_t = float(np.exp(r_t.x[1]))
            n_t_density = float(np.exp(r_t.x[0]))
            per_bin_t = chi2_t / max(n_t, 1)
            # In unphysical regime require T_t < 20·T_p (so the trim
            # actually escaped the runaway-T basin); otherwise just
            # require T_t < T_now. Per-bin gated as above.
            T_ok = (T_t < 20.0 * proton_T) if unphysical_now else (T_t < T_now)
            # Density-collapse guard: cumulative n_α must stay ≥50% of
            # the pre-shrink fit's n in the physical regime. Without this
            # the shrink walks both T and n downward together, since
            # smaller windows admit narrower-Gaussian smaller-n fits
            # whose per-bin SSE is genuinely lower. The unphysical
            # regime has n_now ≈ 0 so this ratio is meaningless — skip.
            n_ok = unphysical_now or n_t_density >= 0.5 * n_initial
            # Basin-transition bypass: when a trim drops T by >30% AND
            # keeps n at-or-above the current fit's n, the candidate has
            # crossed from the wide-Maxwellian basin (which fits the
            # PUI He+ shelf as alpha tail) into the narrow-alpha-core
            # basin. The new basin's per-bin SSE can be substantially
            # *higher* than the current fit's because the still-included
            # high-V bins are PUI shelf the narrow-core model can't
            # explain. The "n_t ≥ n_now" requirement is the safety gate:
            # genuinely-wrong basin transitions collapse n alongside T.
            big_drop_t = (T_t < 0.7 * T_now) and (n_t_density >= n_now)
            gate_ok = big_drop_t or (per_bin_t < chi2_gate)
            if gate_ok and T_ok and n_ok:
                candidates.append(
                    (per_bin_t, T_t, r_t, res_t, n_t, new_window)
                )
        if not candidates:
            break
        big_drop = [c for c in candidates if c[1] < 0.7 * T_now]
        pool = big_drop if big_drop else candidates
        pool.sort(key=lambda c: c[0])
        _, _, result, residuals, n_data, current_window = pool[0]

    n_a_fit = float(np.exp(result.x[0]))
    T_a_fit = float(np.exp(result.x[1]))
    dv_fit = float(result.x[2])
    bulk_velocity_rtn = proton_bulk + dv_fit * b_hat_rtn
    if not result.success:
        bad_fit_flag |= SwapiL3Flags.BAD_FIT

    # LM occasionally finds a degenerate basin where (n_α, T_α) blow up
    # (often n>>n_p with T>>10⁸ K) when the window is small and the residuals
    # cancel. Reject obviously unphysical results so downstream consumers see
    # a clean BAD_FIT NaN rather than a 10⁸-cm⁻³ density.
    if (
        not np.isfinite(n_a_fit)
        or not np.isfinite(T_a_fit)
        or n_a_fit > 100.0
        or T_a_fit > 1e8
    ):
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.BAD_FIT)

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

    # Initial alpha temperature: assume mass-ratio thermal equilibrium with protons
    # (T_α ≈ 4 T_p ⇒ v_th_α ≈ v_th_p), which sets the alpha bump width that we use
    # below to define the LM fit window. Using T_α = T_p as before would produce
    # an alpha model half as wide as reality and drive the window too narrow.
    T_alpha = 4.0 * proton_temperature

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

    # Refine the LM fit window: starting from the residual maximum inside
    # the [valley → 4×V_p] span, walk outward on each side while the
    # residual is monotonically decreasing, stopping at the first local
    # minimum. This isolates the alpha-core lobe and excludes any PUI
    # plateau that may sit above it on the high-V side — important because
    # at low n_p / hot solar wind the PUI shelf can rival the alpha bump
    # in count rate, and a wide LM window covering both produces the
    # runaway-T / collapsed-n degeneracy.
    refined_peak_idx = _walk_to_local_minima(
        residual=residual,
        original_peak_idx=peak_idx,
    )
    if len(refined_peak_idx) >= 3:
        peak_idx = refined_peak_idx

    return (n_alpha, T_alpha, 0.0, peak_idx)


def _walk_to_local_minima(
    residual: ndarray,
    original_peak_idx: ndarray,
    *,
    min_bins: int = 9,
) -> ndarray:
    """Pick the residual maximum inside `original_peak_idx`, then walk
    outward on each side while the residual is strictly decreasing. Stop at
    the first local minimum on each side. The result is the contiguous
    alpha-core lobe; any secondary bump (e.g., a PUI plateau on the high-V
    side) is left out by construction.

    Pads back up to ``min_bins`` symmetrically inside the original window
    if the walk produces a too-narrow lobe. The default 9-bin floor
    keeps the window from being trimmed past the small noise wiggles
    that can hide the alpha core's low-V slope; the downstream
    chi²-shrink loop will re-tighten when the data warrants it."""
    indices = np.sort(np.asarray(original_peak_idx))
    if len(indices) <= min_bins:
        return indices

    sub = residual[indices]
    p = int(np.argmax(sub))

    left = p
    while left > 0 and sub[left - 1] < sub[left]:
        left -= 1
    right = p
    while right < len(sub) - 1 and sub[right + 1] < sub[right]:
        right += 1

    while right - left + 1 < min_bins:
        grew = False
        if left > 0:
            left -= 1
            grew = True
        if right - left + 1 >= min_bins:
            break
        if right < len(sub) - 1:
            right += 1
            grew = True
        if not grew:
            break

    return indices[left : right + 1]


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
