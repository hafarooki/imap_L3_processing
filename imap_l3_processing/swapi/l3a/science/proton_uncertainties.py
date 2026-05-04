import numpy as np
from numpy import ndarray
import uncertainties.umath as umath
import uncertainties.unumpy as unp
from uncertainties import UFloat, correlated_values, covariance_matrix, ufloat

from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_optimizer import (
    OptimizeSolarWindParamsResult,
)


N_VELOCITY_ANGLE_MC_SAMPLES = 1000


def uncertainties_from_residual_scaled_jacobian(
    result: OptimizeSolarWindParamsResult,
) -> tuple[float, float, ndarray]:
    n_state_params = result.jacobian.shape[1]
    try:
        residual_variance = (
            float(np.sum(result.residuals ** 2))
            / max(len(result.residuals) - n_state_params, 1)
        )
        parameter_covariance = (
            residual_variance
            * np.linalg.pinv(result.jacobian.T @ result.jacobian)
        )
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.full((3, 3), np.nan)
    log_density_variance = parameter_covariance[LOG_DENSITY_IDX, LOG_DENSITY_IDX]
    log_temperature_variance = parameter_covariance[
        LOG_TEMPERATURE_IDX, LOG_TEMPERATURE_IDX
    ]
    return (
        float(result.sw_params.density * np.sqrt(log_density_variance)),
        float(result.sw_params.temperature * np.sqrt(log_temperature_variance)),
        parameter_covariance[VELOCITY_SLICE, VELOCITY_SLICE],
    )


def make_correlated_velocity(
    nominal: ndarray, covariance: ndarray
) -> tuple[UFloat, UFloat, UFloat]:
    is_finite = np.all(np.isfinite(covariance))
    is_positive_semidefinite = (
        is_finite and np.linalg.eigvalsh(covariance)[0] >= 0
    )
    if not is_positive_semidefinite:
        return tuple(ufloat(float(v), np.nan) for v in nominal)
    
    return tuple(correlated_values(nominal, covariance))
    


def derive_velocity_angles(
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat],
    epoch_tt2000_ns: float,
) -> tuple:
    from imap_l3_processing.swapi.l3a.utils import rotate_rtn_to_dps

    velocity_dps_unc = rotate_rtn_to_dps(
        np.array(bulk_velocity_rtn), epoch_tt2000_ns
    )
    velocity_dps = unp.nominal_values(velocity_dps_unc)
    velocity_dps_cov = np.array(covariance_matrix(velocity_dps_unc))

    speed_nominal = float(np.linalg.norm(velocity_dps))
    clock_nominal = float(np.degrees(np.arctan2(velocity_dps[1], velocity_dps[0])) % 360)
    deflection_nominal = float(np.degrees(np.arccos(-velocity_dps[2] / speed_nominal)))

    if not np.all(np.isfinite(velocity_dps_cov)):
        return (
            ufloat(speed_nominal, np.nan),
            ufloat(clock_nominal, np.nan),
            ufloat(deflection_nominal, np.nan),
        )

    speed = umath.sqrt(sum(x**2 for x in velocity_dps_unc))
    clock_sigma, deflection_sigma = _clock_and_deflection_sigmas_via_monte_carlo(
        velocity_dps, velocity_dps_cov, clock_nominal
    )

    return (
        speed,
        ufloat(clock_nominal, clock_sigma),
        ufloat(deflection_nominal, deflection_sigma),
    )


def _clock_and_deflection_sigmas_via_monte_carlo(
    velocity_mean: ndarray, velocity_cov: ndarray, clock_nominal: float
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(
        velocity_mean,
        velocity_cov,
        size=N_VELOCITY_ANGLE_MC_SAMPLES,
        check_valid="ignore",
    )

    sample_clocks = np.degrees(np.arctan2(samples[:, 1], samples[:, 0])) % 360.0
    clock_residuals_wrapped = ((sample_clocks - clock_nominal + 180.0) % 360.0) - 180.0
    clock_sigma = float(np.std(clock_residuals_wrapped, ddof=1))

    sample_speeds = np.linalg.norm(samples, axis=1)
    sample_deflections = np.degrees(
        np.arccos(np.clip(-samples[:, 2] / sample_speeds, -1.0, 1.0))
    )
    deflection_sigma = float(np.std(sample_deflections, ddof=1))

    return clock_sigma, deflection_sigma
