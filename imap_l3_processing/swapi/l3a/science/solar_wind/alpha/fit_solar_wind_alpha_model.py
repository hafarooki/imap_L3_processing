from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.optimize
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    SolarWindParams,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
    r_squared,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor


@dataclass
class AlphaSolarWindFitResult:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    delta_v: UFloat  # km/s, signed; +Δv ⇔ alpha drifts along +B̂ vs proton frame
    quality_flag: int

    def velocity_rtn_nominal(self) -> ndarray:
        return np.array([v.nominal_value for v in self.velocity_rtn])

    def velocity_rtn_covariance(self) -> ndarray:
        return np.array(covariance_matrix(self.velocity_rtn))


def fit_solar_wind_alpha_model(
    proton_ctx: SolarWindFitContext,
    alpha_ctx: SolarWindFitContext,
    proton_moments: ProtonSolarWindFitResult,
    magnetic_field_direction: ndarray,
) -> AlphaSolarWindFitResult:
    if alpha_ctx.count_rate.ndim != 2:
        raise ValueError(
            f"alpha_ctx.count_rate must be 2D (n_sweeps, n_bins); got shape "
            f"{alpha_ctx.count_rate.shape}"
        )

    bad_fit_flag = int(proton_moments.quality_flag)
    proton_bulk_rtn = proton_moments.velocity_rtn_nominal()

    if not np.all(np.isfinite(proton_bulk_rtn)):
        return _nan_alpha_fit_result(bad_fit_flag)

    magnetic_field_direction = np.asarray(magnetic_field_direction, dtype=float)
    if not np.all(np.isfinite(magnetic_field_direction)):
        return _nan_alpha_fit_result(bad_fit_flag)

    proton_true_rate, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(
            density=proton_moments.density.nominal_value,
            velocity_rtn=proton_bulk_rtn,
            temperature=proton_moments.temperature.nominal_value,
            mass=proton_ctx.mass_kg,
        ),
        proton_ctx,
    )

    seed = calculate_initial_guess(
        alpha_ctx=alpha_ctx,
        proton_true_rate=proton_true_rate,
        proton_temperature=proton_moments.temperature.nominal_value,
        proton_velocity_rtn=proton_bulk_rtn,
    )
    if seed is None:
        return _nan_alpha_fit_result(bad_fit_flag | SwapiL3Flags.FIT_ERROR)

    alpha_density_seed, alpha_temperature_seed, delta_v_seed, peak_bin_idx = seed

    n_sweeps, n_bins = alpha_ctx.count_rate.shape
    peak_flat_idx = np.concatenate([peak_bin_idx + s * n_bins for s in range(n_sweeps)])
    proton_true_rate_peak = proton_true_rate[peak_flat_idx]
    alpha_ctx_peak = alpha_ctx.subset(peak_bin_idx)

    evaluator = _AlphaEvaluator(
        proton_bulk=proton_bulk_rtn,
        magnetic_field_direction=magnetic_field_direction,
        proton_true_rate=proton_true_rate_peak,
        alpha_ctx=alpha_ctx_peak,
    )

    x0 = np.array(
        [
            np.log(max(alpha_density_seed, 1e-3)),
            np.log(max(alpha_temperature_seed, 1e-3)),
            delta_v_seed,
        ]
    )
    result = scipy.optimize.least_squares(
        evaluator.residuals, x0, jac=evaluator.jacobian, method="lm"
    )

    return _construct_alpha_fit_result(
        result=result,
        alpha_ctx_peak=alpha_ctx_peak,
        n_peak_bins=peak_bin_idx.size,
        n_sweeps=n_sweeps,
        proton_moments=proton_moments,
        proton_bulk=proton_bulk_rtn,
        magnetic_field_direction=magnetic_field_direction,
        bad_fit_flag=bad_fit_flag,
    )


def _nan_alpha_fit_result(flag: int) -> AlphaSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return AlphaSolarWindFitResult(
        density=nan,
        temperature=nan,
        velocity_rtn=(nan, nan, nan),
        delta_v=nan,
        quality_flag=int(flag),
    )


def _construct_alpha_fit_result(
    result: scipy.optimize.OptimizeResult,
    alpha_ctx_peak: SolarWindFitContext,
    n_peak_bins: int,
    n_sweeps: int,
    proton_moments: ProtonSolarWindFitResult,
    proton_bulk: ndarray,
    magnetic_field_direction: ndarray,
    bad_fit_flag: int,
) -> AlphaSolarWindFitResult:
    if not result.success:
        return _nan_alpha_fit_result(bad_fit_flag | SwapiL3Flags.FIT_ERROR)

    alpha_density_fit = float(np.exp(result.x[0]))
    alpha_temperature_fit = float(np.exp(result.x[1]))
    delta_v_fit = float(result.x[2])
    velocity_rtn = proton_bulk + delta_v_fit * magnetic_field_direction

    fit_r_squared = _alpha_r_squared(
        residuals=result.fun,
        count_rate=alpha_ctx_peak.count_rate,
        n_sweeps=n_sweeps,
        n_peak_bins=n_peak_bins,
    )
    if fit_r_squared < 0.9:
        return _nan_alpha_fit_result(bad_fit_flag | SwapiL3Flags.BAD_FIT)

    cov_x = compute_hc3_parameter_covariance(result.jac, result.fun)
    density_sigma = float(alpha_density_fit * np.sqrt(max(cov_x[0, 0], 0.0)))
    temperature_sigma = float(alpha_temperature_fit * np.sqrt(max(cov_x[1, 1], 0.0)))
    delta_v_sigma = float(np.sqrt(max(cov_x[2, 2], 0.0)))

    sigma_dv2 = max(cov_x[2, 2], 0.0)
    velocity_covariance_rtn = (
        proton_moments.velocity_rtn_covariance()
        + sigma_dv2 * np.outer(magnetic_field_direction, magnetic_field_direction)
    )

    return AlphaSolarWindFitResult(
        density=ufloat(alpha_density_fit, density_sigma),
        temperature=ufloat(alpha_temperature_fit, temperature_sigma),
        velocity_rtn=make_correlated_velocity(
            velocity_rtn, velocity_covariance_rtn
        ),
        delta_v=ufloat(delta_v_fit, delta_v_sigma),
        quality_flag=int(bad_fit_flag),
    )


def _alpha_r_squared(
    residuals: ndarray, count_rate: ndarray, n_sweeps: int, n_peak_bins: int
) -> float:
    averaged_count_rate = np.nanmean(count_rate, axis=0)
    averaged_residual = np.nanmean(
        residuals.reshape(n_sweeps, n_peak_bins), axis=0
    )
    return r_squared(averaged_residual, averaged_count_rate)


class _AlphaEvaluator:
    def __init__(
        self,
        proton_bulk: ndarray,
        magnetic_field_direction: ndarray,
        proton_true_rate: ndarray,
        alpha_ctx: SolarWindFitContext,
    ):
        self.proton_bulk = proton_bulk
        self.magnetic_field_direction = magnetic_field_direction
        self.proton_true_rate = proton_true_rate
        self.alpha_ctx = alpha_ctx
        self._last_state: Optional[ndarray] = None
        self._last_residuals: Optional[ndarray] = None
        self._last_jacobian: Optional[ndarray] = None

    def _eval(self, x: ndarray) -> None:
        alpha_density = float(np.exp(x[0]))
        alpha_temperature = float(np.exp(x[1]))
        delta_v = float(x[2])
        alpha_velocity_rtn = self.proton_bulk + delta_v * self.magnetic_field_direction
        alpha_true, jacobian_alpha_5d = model_solar_wind_ideal_coincidence_rates(
            SolarWindParams(
                alpha_density, alpha_velocity_rtn, alpha_temperature, self.alpha_ctx.mass_kg
            ),
            self.alpha_ctx,
        )
        total_true = self.proton_true_rate + alpha_true
        deadtime = deadtime_factor(total_true)
        residuals = total_true * deadtime - self.alpha_ctx.count_rate.ravel()

        deadtime_squared = deadtime * deadtime
        jacobian = np.empty((alpha_true.size, 3))
        jacobian[:, 0] = deadtime_squared * jacobian_alpha_5d[:, LOG_DENSITY_IDX]
        jacobian[:, 1] = deadtime_squared * jacobian_alpha_5d[:, LOG_TEMPERATURE_IDX]
        jacobian[:, 2] = deadtime_squared * (
            jacobian_alpha_5d[:, VELOCITY_SLICE] @ self.magnetic_field_direction
        )

        self._last_state = x.copy()
        self._last_residuals = residuals
        self._last_jacobian = jacobian

    def _refresh(self, x: ndarray) -> None:
        if self._last_state is None or not np.array_equal(x, self._last_state):
            self._eval(x)

    def residuals(self, x: ndarray) -> ndarray:
        self._refresh(x)
        return self._last_residuals

    def jacobian(self, x: ndarray) -> ndarray:
        self._refresh(x)
        return self._last_jacobian
