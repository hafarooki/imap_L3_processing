from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import SolarWindFitContext
from imap_l3_processing.swapi.l3a.science.proton_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_optimizer import (
    optimize_solar_wind_params,
)
from imap_l3_processing.swapi.l3a.science.proton_basin_hopping import (
    escape_local_minimum,
)
from imap_l3_processing.swapi.l3a.science.proton_uncertainties import (
    uncertainties_from_residual_scaled_jacobian,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


@dataclass
class ProtonSolarWindFitResult:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    bad_fit_flag: int

    def bulk_velocity_rtn_nominal(self) -> ndarray:
        return np.array([v.nominal_value for v in self.bulk_velocity_rtn])

    def bulk_velocity_rtn_covariance(self) -> ndarray:
        return np.array(covariance_matrix(self.bulk_velocity_rtn))


def fit_solar_wind_proton_moments(ctx: SolarWindFitContext) -> ProtonSolarWindFitResult:
    initial_guess = calculate_initial_guess(ctx)
    first_result = optimize_solar_wind_params(initial_guess, ctx)
    final_result = escape_local_minimum(first_result, ctx)
    return _construct_fit_result(final_result)


def _construct_fit_result(final_result):
    density_sigma, temperature_sigma, velocity_covariance = (
        uncertainties_from_residual_scaled_jacobian(final_result)
    )
    density = ufloat(final_result.sw_params.density, density_sigma)
    temperature = ufloat(final_result.sw_params.temperature, temperature_sigma)
    bulk_velocity_rtn = make_correlated_velocity(
        final_result.sw_params.bulk_velocity_rtn, velocity_covariance
    )
    bad_fit_flag = SwapiL3Flags.NONE if final_result.success else SwapiL3Flags.BAD_FIT
    return ProtonSolarWindFitResult(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=bad_fit_flag,
    )
