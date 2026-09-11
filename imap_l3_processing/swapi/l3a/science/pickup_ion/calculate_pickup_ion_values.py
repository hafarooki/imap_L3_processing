from __future__ import annotations
from functools import partial

from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from uncertainties import UFloat, ufloat

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_COARSE_SWEEP_BINS,
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
    is_good_fit,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.moments import (
    calculate_helium_pui_density,
    calculate_helium_pui_temperature,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species

_PICKUP_ION_SPECIES = Species.HELIUM_PLUS
_INITIAL_IONIZATION_RATE_PER_S = 1e-7
_LOG_IONIZATION_RATE_INDEX = 0
_LOG_CUTOFF_SPEED_INDEX = 1
_SWEEPS_PER_CHUNK = 50
_COARSE_BIN_COUNT = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start


@dataclass(frozen=True)
class PickupIonFitInputData:
    time_as_tt2000: int
    """Center of the chunk [ns since J2000 TT]."""

    esa_energies: NDArray
    """Sweep-averaged coarse ESA energies, shape (62,) [eV/e]."""

    coincidence_count_rates: NDArray
    """Coarse-sweep coincidence count rates, shape (50, 62) [counts/s]."""

    solar_wind_velocity_rtn_sun: NDArray
    """Chunk-mean solar wind bulk velocity (Sun frame, RTN coords), shape (3,) [km/s]."""

    bulk_sw_per_bin_swapi_kms: NDArray
    """Rotated chunk-mean solar wind bulk velocity (SC frame, SWAPI coords),
    shape (50, 62, 3) [km/s]."""

    distance: float
    """IMAP's heliocentric distance [km]."""

    inflow_angle: float
    """IMAP's position angle to the neutral helium inflow vector [deg]."""

    def __post_init__(self):
        expected_shapes = {
            "esa_energies": (_COARSE_BIN_COUNT,),
            "coincidence_count_rates": (_SWEEPS_PER_CHUNK, _COARSE_BIN_COUNT),
            "solar_wind_velocity_rtn_sun": (3,),
            "bulk_sw_per_bin_swapi_kms": (_SWEEPS_PER_CHUNK, _COARSE_BIN_COUNT, 3),
        }
        for field_name, expected_shape in expected_shapes.items():
            actual_shape = np.shape(getattr(self, field_name))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{field_name} has shape {actual_shape}, expected {expected_shape}"
                )


@dataclass
class PickupIonFitResult:
    ionization_rate: UFloat
    """Fitted He+ pickup ion ionization rate [1/s]"""

    cutoff_speed: UFloat
    """Fitted He+ pickup ion cutoff speed [km/s]"""

    density: UFloat
    """Inferred He+ pickup ion density [cm^-3]"""

    temperature: UFloat
    """Inferred He+ pickup ion temperature [K]"""

    flags: SwapiL3Flags
    """Flags associated with the fit."""


def calculate_pickup_ion_values(
    fit_input: PickupIonFitInputData,
    *,
    swapi_response: SwapiResponse,
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
) -> PickupIonFitResult:
    """
    Fit the Vasyliunas-Siscoe helium PUI model to one 10-minute chunk.

    Parameters
    ----------
    fit_input : PickupIonFitInputData
        The chunk's L2 science data and the per-chunk geometry it is fit
        against.
    swapi_response : SwapiResponse
        The instrument response model.
    density_of_neutral_helium_lookup_table : DensityOfNeutralHeliumLookupTable
        The neutral helium density table for the model.
    """
    solar_wind_speed_inertial_frame = float(np.linalg.norm(fit_input.solar_wind_velocity_rtn_sun))
    distance = fit_input.distance
    inflow_angle = fit_input.inflow_angle

    # === calculate fitting energy range ===
    lower_energy_cutoff, upper_energy_cutoff = calculate_pickup_ion_fit_energy_range(
        solar_wind_speed_inertial_frame
    )
    # fit and goodness-of-fit eval share the same lower cutoff, they differ only in upper cutoff 
    
    # full range used for both
    shared_esa_step_mask = fit_input.esa_energies > lower_energy_cutoff
    shared_energies = fit_input.esa_energies[shared_esa_step_mask]
    shared_count_rates = fit_input.coincidence_count_rates[:, shared_esa_step_mask]


    # subset used for fitting
    fitting_esa_step_mask = shared_energies < upper_energy_cutoff
    fitting_count_rates = shared_count_rates[:, fitting_esa_step_mask]


    # === prepare forward model ===

    shared_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=shared_energies / SWAPI_L2_K_FACTOR,
        bulk_sw_per_bin_kms=fit_input.bulk_sw_per_bin_swapi_kms[:, shared_esa_step_mask, :],
        time_as_tt2000=fit_input.time_as_tt2000,
        species=_PICKUP_ION_SPECIES,
        cutoff_speed_max_kms=solar_wind_speed_inertial_frame * MAX_CUTOFF_SPEED_RATIO * 1.1,
    )
    fitting_response = ChunkCollapsedResponse(
        speed_grid=shared_response.speed_grid,
        bin_weights=shared_response.bin_weights[:, fitting_esa_step_mask],
    )

    model = lambda ionization_rate, cutoff_speed, response: calculate_coincidence_rate(
        response,
        ionization_rate=ionization_rate,
        cutoff_speed=cutoff_speed,
        distance=distance,
        inflow_angle=inflow_angle,
        solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
        density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
    ) 
    shared_window_model = partial(model, response=shared_response)
    fitting_window_model = partial(model, response=fitting_response)

    # === fit model ===

    initial_log_parameters = np.array(
        [
            np.log(_INITIAL_IONIZATION_RATE_PER_S),
            np.log(solar_wind_speed_inertial_frame),
        ]
    )

    def residuals(log_parameters: np.ndarray) -> np.ndarray:
        modeled_rates = (
            fitting_window_model(
                ionization_rate=float(np.exp(log_parameters[_LOG_IONIZATION_RATE_INDEX])),
                cutoff_speed=float(np.exp(log_parameters[_LOG_CUTOFF_SPEED_INDEX])),
            )
            + SWAPI_BACKGROUND_RATE
        )

        return modeled_rates.mean(axis=0) - fitting_count_rates.mean(axis=0)

    result: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
        residuals,
        initial_log_parameters,
        method="lm",
    )

    # === output formatting ===

    fitted_parameters = np.exp(result.x)
    log_parameter_covariance = compute_hc3_parameter_covariance(result.jac, result.fun)

    fitted_ionization_rate = float(fitted_parameters[_LOG_IONIZATION_RATE_INDEX])
    fitted_cutoff_speed = float(fitted_parameters[_LOG_CUTOFF_SPEED_INDEX])

    # handle cases where we set BAD_FIT flag and report fill values
    if (
        not result.success
        or not np.all(np.isfinite(log_parameter_covariance))
        or not is_good_fit(
            esa_energies=shared_energies,
            model_rates=shared_window_model(
                ionization_rate=fitted_ionization_rate,
                cutoff_speed=fitted_cutoff_speed,
            ),
            observed_rates=shared_count_rates,
            cutoff_speed_kms=fitted_cutoff_speed,
            sw_speed_kms=solar_wind_speed_inertial_frame,
            ionization_rate=fitted_ionization_rate,
        )
    ):
        nan_parameter = ufloat(np.nan, np.nan)
        return PickupIonFitResult(
            ionization_rate=nan_parameter,
            cutoff_speed=nan_parameter,
            density=nan_parameter,
            temperature=nan_parameter,
            flags=SwapiL3Flags.BAD_FIT,
        )

    # sigma(x) = x * sigma(ln x)
    parameter_sigmas = fitted_parameters * np.sqrt(np.diag(log_parameter_covariance))
    fitted_ionization_rate_error = float(parameter_sigmas[_LOG_IONIZATION_RATE_INDEX])
    fitted_cutoff_speed_error = float(parameter_sigmas[_LOG_CUTOFF_SPEED_INDEX])

    return PickupIonFitResult(
        ionization_rate=ufloat(fitted_ionization_rate, fitted_ionization_rate_error),
        cutoff_speed=ufloat(fitted_cutoff_speed, fitted_cutoff_speed_error),
        density=calculate_helium_pui_density(
            shared_response.speed_grid,
            ionization_rate=ufloat(fitted_ionization_rate, fitted_ionization_rate_error),
            cutoff_speed=ufloat(fitted_cutoff_speed, fitted_cutoff_speed_error),
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        ),
        temperature=calculate_helium_pui_temperature(
            shared_response.speed_grid,
            ionization_rate=ufloat(fitted_ionization_rate, fitted_ionization_rate_error),
            cutoff_speed=ufloat(fitted_cutoff_speed, fitted_cutoff_speed_error),
            distance=distance,
            inflow_angle=inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        ),
        flags=SwapiL3Flags.NONE,
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
