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

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    ONE_AU_IN_KM,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
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
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    central_effective_area_scale: float = 1.0,
) -> PickupIonFitResult:
    voltages = np.asarray(voltages, dtype=float).reshape(-1, _COARSE_SWEEP_LEN)
    count_rates = np.asarray(count_rates, dtype=float).reshape(-1, _COARSE_SWEEP_LEN)
    bulk_sw_per_bin_swapi_kms = np.asarray(
        bulk_sw_per_bin_swapi_kms, dtype=float
    ).reshape(-1, _COARSE_SWEEP_LEN, 3)

    voltages_per_step = np.mean(voltages, axis=0)
    energies_per_step = np.abs(voltages_per_step) * SWAPI_L2_K_FACTOR

    sw_velocity_kms = float(np.linalg.norm(sw_velocity_rtn_kms))

    lower_energy_cutoff, upper_energy_cutoff = calculate_pickup_ion_fit_energy_range(
        sw_velocity_kms
    )

    bin_mask = (energies_per_step > lower_energy_cutoff) & (
        energies_per_step < upper_energy_cutoff
    )
    extracted_voltages = voltages_per_step[bin_mask]
    extracted_count_rates = count_rates[:, bin_mask]
    extracted_bulk_sw_per_bin_swapi_kms = bulk_sw_per_bin_swapi_kms[:, bin_mask]

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


def calculate_pickup_ion_fit_energy_range(
    solar_wind_bulk_speed_kms: float,
) -> tuple[float, float]:
    proton_energy_per_charge_ev = (
        0.5
        * PROTON_MASS_KG
        * (solar_wind_bulk_speed_kms * METERS_PER_KILOMETER) ** 2
        / PROTON_CHARGE_COULOMBS
    )

    # assumes alpha solar wind has the same bulk speed as proton solar wind
    nominal_alpha_peak = 2 * proton_energy_per_charge_ev

    # assumes that the PUI cutoff speed is 2x the solar wind speed (4x the energy) in the SC frame
    # accounts for the 4x mass per charge of He+ compared to protons
    # together, that's a factor of 2^2*4=4x4=16
    nominal_pui_he_cutoff = 16 * proton_energy_per_charge_ev

    # geometric mean (logarithmic midpoint) between estimated alpha peak and nominal PUI cutoff
    lower_edge = np.sqrt(nominal_alpha_peak * nominal_pui_he_cutoff)

    # use nominal PUI cutoff as the upper edge for the fitting range
    upper_edge = nominal_pui_he_cutoff

    return float(lower_edge), float(upper_edge)


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
    params.add("cooling_index", value=1.5, min=1.0, max=5.0)
    params.add("ionization_rate", value=1e-7, min=0.6e-9, max=8.0e-7)
    params.add(
        "cutoff_speed",
        value=sw_speed_kms,
        min=sw_speed_kms * 0.8,
        max=sw_speed_kms * 1.2,
    )
    params.add("background_count_rate", value=0.1, min=0, max=10.0)

    def map_to_internal(value, param):
        return np.arcsin(2 * (value - param.min) / (param.max - param.min) - 1)

    def simplex_vertex(cooling_index, ionization_rate, cutoff_speed, background):
        return [
            map_to_internal(cooling_index, params["cooling_index"]),
            map_to_internal(ionization_rate, params["ionization_rate"]),
            map_to_internal(cutoff_speed, params["cutoff_speed"]),
            map_to_internal(background, params["background_count_rate"]),
        ]

    initial_simplex = np.array(
        [
            simplex_vertex(1.5, 1e-7, sw_speed_kms, 0.1),
            simplex_vertex(5.0, 1e-7, sw_speed_kms, 0.1),
            simplex_vertex(1.5, 2.1e-7, sw_speed_kms, 0.1),
            simplex_vertex(1.5, 1e-7, sw_speed_kms * 1.2, 0.1),
            simplex_vertex(1.5, 1e-7, sw_speed_kms, 0.2),
        ]
    )

    minimizer = lmfit.Minimizer(
        _calculate_poisson_negative_log_likelihood,
        params,
        fcn_args=(observed_count_rates, chunk_response, vasyliunas_siscoe_distribution),
        scale_covar=False,
        options=dict(initial_simplex=initial_simplex),
    )
    result = minimizer.minimize(method="nelder")

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

    param_vals = {
        name: ufloat(nominal_values[name], std_err)
        for name, std_err in zip(result.var_names, standard_errors)
    }

    _set_background_to_fill_if_too_high(param_vals)

    return FittingParameters(
        param_vals["cooling_index"],
        param_vals["ionization_rate"],
        param_vals["cutoff_speed"],
        param_vals["background_count_rate"],
        flags,
    )


def _calculate_poisson_negative_log_likelihood(
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
    modeled_counts = modeled_rates * swapi_l2.SWAPI_LIVETIME
    observed_counts = observed_count_rates * swapi_l2.SWAPI_LIVETIME
    return float(np.sum(modeled_counts - observed_counts * np.log(modeled_counts)))


def _set_background_to_fill_if_too_high(param_vals):
    background = param_vals["background_count_rate"]
    if background.nominal_value > 1.0:
        param_vals["background_count_rate"] = ufloat(np.nan, np.nan)
