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
from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_L2_K_FACTOR,
)
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
from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import (
    MAX_CUTOFF_SPEED_RATIO,
    MAX_IONIZATION_RATE,
    MIN_CUTOFF_SPEED_RATIO,
    MIN_IONIZATION_RATE,
    is_good_fit,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species


_COARSE_SWEEP_LEN = 62
_PICKUP_ION_SPECIES = Species.HELIUM_PLUS


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
    time_as_tt2000: int,
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

    # fit and goodness-of-fit eval share the same lower cutoff, they differ only in upper cutoff
    modeled_esa_step_mask = energies_per_step > lower_energy_cutoff
    modeled_energies = energies_per_step[modeled_esa_step_mask]
    modeled_count_rates = count_rates[:, modeled_esa_step_mask]

    # collapsed response over the modeled ESA steps
    modeled_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=voltages_per_step[modeled_esa_step_mask],
        bulk_sw_per_bin_kms=bulk_sw_per_bin_swapi_kms[:, modeled_esa_step_mask, :],
        time_as_tt2000=time_as_tt2000,
        species=_PICKUP_ION_SPECIES,
        cutoff_speed_max_kms=sw_velocity_kms * MAX_CUTOFF_SPEED_RATIO,
    )

    # collapsed response over subset used for fitting
    fitting_esa_step_mask = modeled_energies < upper_energy_cutoff
    fit_window_response = ChunkCollapsedResponse(
        speed_in_sw_frame=modeled_response.speed_in_sw_frame,
        bin_weights=modeled_response.bin_weights[:, fitting_esa_step_mask],
    )

    fitting_params = _fit_pickup_ion_parameters(
        chunk_response=fit_window_response,
        vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
        observed_count_rates=modeled_count_rates[:, fitting_esa_step_mask],
        sw_speed_kms=sw_velocity_kms,
    )

    if not (int(fitting_params.flags) & int(SwapiL3Flags.BAD_FIT)):
        nominal_fitting_params = FittingParameters(
            ionization_rate=fitting_params.ionization_rate.nominal_value,
            cutoff_speed=fitting_params.cutoff_speed.nominal_value,
        )
        modeled_rates = calculate_coincidence_rate(
            modeled_response, vasyliunas_siscoe_distribution, nominal_fitting_params
        )
        if not is_good_fit(
            esa_energies=modeled_energies,
            model_rates=modeled_rates,
            observed_rates=modeled_count_rates,
            cutoff_speed_kms=nominal_fitting_params.cutoff_speed,
            sw_speed_kms=sw_velocity_kms,
            ionization_rate=nominal_fitting_params.ionization_rate,
            background_rate=SWAPI_BACKGROUND_RATE,
        ):
            nan_param = ufloat(np.nan, np.nan)
            fitting_params = FittingParameters(
                nan_param,
                nan_param,
                fitting_params.flags | SwapiL3Flags.BAD_FIT,
            )

    return PickupIonFitResult(
        fitting_params=fitting_params,
        chunk_response=fit_window_response,
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
    params.add(
        "ionization_rate",
        value=1e-7,
        min=MIN_IONIZATION_RATE,
        max=MAX_IONIZATION_RATE,
    )
    params.add(
        "cutoff_speed",
        value=sw_speed_kms,
        min=sw_speed_kms * MIN_CUTOFF_SPEED_RATIO,
        max=sw_speed_kms * MAX_CUTOFF_SPEED_RATIO,
    )

    def map_to_internal(value, param):
        return np.arcsin(2 * (value - param.min) / (param.max - param.min) - 1)

    def simplex_vertex(ionization_rate, cutoff_speed):
        return [
            map_to_internal(ionization_rate, params["ionization_rate"]),
            map_to_internal(cutoff_speed, params["cutoff_speed"]),
        ]

    initial_simplex = np.array(
        [
            simplex_vertex(1e-7, sw_speed_kms),
            simplex_vertex(2.1e-7, sw_speed_kms),
            simplex_vertex(1e-7, sw_speed_kms * MAX_CUTOFF_SPEED_RATIO),
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

    if flags & SwapiL3Flags.BAD_FIT:
        nan_param = ufloat(np.nan, np.nan)
        return FittingParameters(nan_param, nan_param, flags)

    param_vals = {
        name: ufloat(nominal_values[name], std_err)
        for name, std_err in zip(result.var_names, standard_errors)
    }

    return FittingParameters(
        param_vals["ionization_rate"],
        param_vals["cutoff_speed"],
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
        ionization_rate=parvals["ionization_rate"],
        cutoff_speed=parvals["cutoff_speed"],
    )

    modeled_rates = (
        calculate_coincidence_rate(
            chunk_response, vasyliunas_siscoe_distribution, fitting_params
        )
        + SWAPI_BACKGROUND_RATE
    )
    modeled_counts = modeled_rates * swapi_l2.SWAPI_LIVETIME
    observed_counts = observed_count_rates * swapi_l2.SWAPI_LIVETIME
    return float(np.sum(modeled_counts - observed_counts * np.log(modeled_counts)))

