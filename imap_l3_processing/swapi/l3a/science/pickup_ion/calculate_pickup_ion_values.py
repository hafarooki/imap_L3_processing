from __future__ import annotations

from dataclasses import dataclass

import lmfit
import numdifftools as ndt
import numpy as np
from imap_processing.swapi.l2 import swapi_l2
from lmfit import Parameters
from numpy import ndarray
from scipy.linalg import inv
from uncertainties import ufloat

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.constants import SWAPI_L2_K_FACTOR
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    calculate_coincidence_rate,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
    build_chunk_collapsed_response,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse


_COARSE_SWEEP_LEN = 62
_HELIUM_MASS_PER_CHARGE_M_P_PER_E = 4.0


@dataclass
class PickupIonFitResult:
    fitting_params: FittingParameters
    chunk_response: ChunkCollapsedResponse
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution


def calculate_pickup_ion_values(
    swapi_response: SwapiResponse,
    voltages: np.ndarray,
    count_rates: np.ndarray,
    sw_velocity_rtn_kms: ndarray,
    bulk_sw_per_bin_swapi_kms: ndarray,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
    lower_energy_cutoff: float,
    upper_energy_cutoff: float,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    central_effective_area_scale: float = 1.0,
) -> PickupIonFitResult:
    voltages = np.asarray(voltages, dtype=float).reshape(-1, _COARSE_SWEEP_LEN)
    count_rates = np.asarray(count_rates, dtype=float).reshape(-1, _COARSE_SWEEP_LEN)
    bulk_sw_per_bin_swapi_kms = np.asarray(
        bulk_sw_per_bin_swapi_kms, dtype=float
    ).reshape(-1, _COARSE_SWEEP_LEN, 3)

    voltages_per_step = np.mean(voltages, axis=0)
    energies_per_step = voltages_per_step * SWAPI_L2_K_FACTOR
    bin_mask = (energies_per_step > lower_energy_cutoff) & (
        energies_per_step < upper_energy_cutoff
    )
    extracted_voltages = voltages_per_step[bin_mask]
    extracted_count_rates = count_rates[:, bin_mask]
    extracted_bulk_sw_per_bin_swapi_kms = bulk_sw_per_bin_swapi_kms[:, bin_mask]

    sw_velocity_kms = float(np.linalg.norm(sw_velocity_rtn_kms))

    chunk_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=extracted_voltages,
        bulk_sw_per_bin_kms=extracted_bulk_sw_per_bin_swapi_kms,
        mass_per_charge_m_p_per_e=_HELIUM_MASS_PER_CHARGE_M_P_PER_E,
        cutoff_speed_max_kms=sw_velocity_kms * 1.2,
        central_effective_area_scale=central_effective_area_scale,
    )

    fitting_params = _fit_pickup_ion_parameters(
        chunk_response=chunk_response,
        vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
        observed_count_rates=extracted_count_rates,
        sw_speed_kms=sw_velocity_kms,
    )
    return PickupIonFitResult(
        fitting_params=fitting_params,
        chunk_response=chunk_response,
        vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
    )


def _fit_pickup_ion_parameters(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    observed_count_rates: np.ndarray,
    sw_speed_kms: float,
) -> FittingParameters:
    """Run the Nelder-Mead PUI parameter fit.

    `observed_count_rates` is shape (n_sweeps, n_steps). `chunk_response` and
    `vasyliunas_siscoe_distribution` carry the precomputed geometry; the
    residual constructs a `FittingParameters` from each iteration's lmfit values.
    """
    params = Parameters()

    # free parameters
    params.add("ionization_rate", value=1e-7, min=0.6e-9, max=8.0e-7)
    params.add(
        "cutoff_speed",
        value=sw_speed_kms,
        min=sw_speed_kms * 0.8,
        max=sw_speed_kms * 1.2,
    )
    
    # held constant, not fit
    params.add("cooling_index", value=1.5, vary=False)
    params.add("background_count_rate", value=0.1, vary=False)


    minimizer = lmfit.Minimizer(
        _calculate_residuals,
        params,
        fcn_args=(observed_count_rates, chunk_response, vasyliunas_siscoe_distribution),
        scale_covar=False
    )
    result = minimizer.minimize(method="lm")

    nominal_values = result.params.valuesdict()

    flags = SwapiL3Flags.NONE
    hessian_fn = ndt.Hessian(minimizer.penalty)
    try:
        hessian_value = hessian_fn(result.x)
        cov_internal = inv(hessian_value)
        cov_external = minimizer._int2ext_cov_x(cov_internal, result.x)
        standard_errors = np.sqrt(np.diag(cov_external))  # NaN if not positive definite
    except Exception:
        standard_errors = np.full(len(result.var_names), np.nan)

    if not np.all(np.isfinite(standard_errors)):
        flags |= SwapiL3Flags.BAD_FIT

    best_fit_params = FittingParameters(
        cooling_index=nominal_values["cooling_index"],
        ionization_rate=nominal_values["ionization_rate"],
        cutoff_speed=nominal_values["cutoff_speed"],
        background_count_rate=nominal_values["background_count_rate"],
    )
    best_fit_rates = calculate_coincidence_rate(
        chunk_response, vasyliunas_siscoe_distribution, best_fit_params
    )
    
    # R^2 on the sweep-averaged spectrum.
    observed_sweep_average = np.nanmean(observed_count_rates, axis=0)
    best_fit_sweep_average = np.nanmean(best_fit_rates, axis=0)
    total_sum_of_squares = float(
        np.nansum((observed_sweep_average - np.nanmean(observed_sweep_average)) ** 2)
    )
    
    if total_sum_of_squares == 0:
        flags |= SwapiL3Flags.BAD_FIT
    else:
        residual_sum_of_squares = float(
            np.nansum((observed_sweep_average - best_fit_sweep_average) ** 2)
        )
        r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares
        if r_squared < 0.9:
            flags |= SwapiL3Flags.BAD_FIT

    if flags & SwapiL3Flags.BAD_FIT:
        nan_param = ufloat(np.nan, np.nan)
        return FittingParameters(
            nan_param, nan_param, nan_param, nan_param, flags,
        )

    fitted_standard_errors = dict(zip(result.var_names, standard_errors))
    # Claude: parameters held fixed contribute no column to the Hessian, so they
    # have no fitted uncertainty; report them at sigma = 0.
    param_vals = {
        name: ufloat(value, fitted_standard_errors.get(name, 0.0))
        for name, value in nominal_values.items()
    }

    return FittingParameters(
        param_vals["cooling_index"],
        param_vals["ionization_rate"],
        param_vals["cutoff_speed"],
        param_vals["background_count_rate"],
        flags,
    )


def _calculate_residuals(
    params: Parameters,
    observed_count_rates: np.ndarray,  # (n_sweeps, n_steps)
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
) -> float:
    parvals = params.valuesdict()
    fitting_params = FittingParameters(
        cooling_index=parvals["cooling_index"],
        ionization_rate=parvals["ionization_rate"],
        cutoff_speed=parvals["cutoff_speed"],
        background_count_rate=parvals["background_count_rate"],
    )

    modeled_rates = calculate_coincidence_rate(
        chunk_response, vasyliunas_siscoe_distribution, fitting_params
    )

    return modeled_rates.mean(axis=0) - observed_count_rates.mean(axis=0)
