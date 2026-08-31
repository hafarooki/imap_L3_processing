from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.escape_local_minimum import (
    escape_local_minimum,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.calculate_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.optimize_solar_wind_proton_params import (
    OptimizeSolarWindProtonParamsResult,
    optimize_solar_wind_proton_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
    r_squared,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


_R_SQUARED_PEAK_HALF_WIDTH = 4


@dataclass
class ProtonSolarWindFitResult:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    quality_flag: int

    def velocity_rtn_nominal(self) -> ndarray:
        return np.array([v.nominal_value for v in self.velocity_rtn])

    def velocity_rtn_covariance(self) -> ndarray:
        return np.array(covariance_matrix(self.velocity_rtn))


def fit_solar_wind_proton_model(ctx: SolarWindFitContext) -> ProtonSolarWindFitResult:
    initial_guess = calculate_initial_guess(ctx)
    first_result = optimize_solar_wind_proton_params(initial_guess, ctx)
    final_result = escape_local_minimum(first_result, ctx)
    return _construct_fit_result(final_result, ctx)


def _construct_fit_result(final_result, ctx):
    if not final_result.success:
        return _nan_proton_fit_result(int(SwapiL3Flags.FIT_ERROR))

    fit_r_squared = _coarse_peak_averaged_r_squared(
        final_result.residuals, ctx.count_rate
    )
    flag_bad = (
        fit_r_squared < 0.9
        or final_result.sw_params.temperature > 5.0e5
    )
    if flag_bad:
        return _nan_proton_fit_result(int(SwapiL3Flags.BAD_FIT))

    density_sigma, temperature_sigma, velocity_covariance = _derive_uncertainties(
        final_result, ctx
    )
    density = ufloat(final_result.sw_params.density, density_sigma)
    temperature = ufloat(final_result.sw_params.temperature, temperature_sigma)
    velocity_rtn = make_correlated_velocity(
        final_result.sw_params.velocity_rtn, velocity_covariance
    )
    return ProtonSolarWindFitResult(
        density=density,
        temperature=temperature,
        velocity_rtn=velocity_rtn,
        quality_flag=int(SwapiL3Flags.NONE),
    )


def _nan_proton_fit_result(bad_fit_flag: int) -> ProtonSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return ProtonSolarWindFitResult(
        density=nan,
        temperature=nan,
        velocity_rtn=(nan, nan, nan),
        quality_flag=bad_fit_flag,
    )


def _coarse_peak_averaged_r_squared(residuals: ndarray, count_rate: ndarray) -> float:
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    coarse_count_rate_per_sweep = count_rate[:, :n_coarse_bins]
    coarse_residual_per_sweep = residuals.reshape(count_rate.shape)[:, :n_coarse_bins]
    averaged_count_rate = np.nanmean(coarse_count_rate_per_sweep, axis=0)
    averaged_residual = np.nanmean(coarse_residual_per_sweep, axis=0)
    peak_index = int(np.nanargmax(averaged_count_rate))
    n_bins = averaged_count_rate.shape[0]
    clamped_peak_index = max(
        _R_SQUARED_PEAK_HALF_WIDTH,
        min(peak_index, n_bins - _R_SQUARED_PEAK_HALF_WIDTH - 1),
    )
    window = slice(
        clamped_peak_index - _R_SQUARED_PEAK_HALF_WIDTH,
        clamped_peak_index + _R_SQUARED_PEAK_HALF_WIDTH + 1,
    )
    return r_squared(averaged_residual[window], averaged_count_rate[window])


def _derive_uncertainties(
    result: OptimizeSolarWindProtonParamsResult,
    ctx: SolarWindFitContext,
) -> tuple[float, float, ndarray]:
    parameter_covariance = compute_hc3_parameter_covariance(
        result.jacobian, result.residuals
    )
    if not np.all(np.isfinite(parameter_covariance)):
        return np.nan, np.nan, np.full((3, 3), np.nan)

    log_density_variance = parameter_covariance[LOG_DENSITY_IDX, LOG_DENSITY_IDX]
    log_temperature_variance = parameter_covariance[
        LOG_TEMPERATURE_IDX, LOG_TEMPERATURE_IDX
    ]

    density_error = float(
        result.sw_params.density * np.sqrt(max(log_density_variance, 0.0))
    )
    temperature_error = float(
        result.sw_params.temperature * np.sqrt(max(log_temperature_variance, 0.0))
    )
    velocity_covariance = parameter_covariance[VELOCITY_SLICE, VELOCITY_SLICE]

    return (
        density_error,
        temperature_error,
        velocity_covariance,
    )
