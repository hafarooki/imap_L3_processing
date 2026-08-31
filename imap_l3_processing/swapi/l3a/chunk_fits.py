import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import spiceypy
from spacepy import pycdf
from spiceypy.utils.exceptions import SpiceyError
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    FIVE_MINUTES_IN_NANOSECONDS,
    HE_PUI_PARTICLE_MASS_KG,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.moments import (
    calculate_helium_pui_density,
    calculate_helium_pui_temperature,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import (
    calculate_pui_energy_cutoff,
    calculate_ten_minute_velocities,
    rotate_rtn_velocity_to_swapi_per_bin,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    VasyliunasSiscoeDistribution,
    build_vasyliunas_siscoe_distribution,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    AlphaSolarWindFitResult,
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    ProtonSolarWindFitResult,
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.l3a.utils import (
    chunk_epoch,
    compute_direction_of_mean_magnetic_field_over_chunk,
    esa_voltage_to_proton_speed,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.predicted_ephemeris_tracker import PredictedEphemerisTracker

logger = logging.getLogger(__name__)

_shared: dict[str, Any] = {}


class ChunkFitter(ABC):
    @abstractmethod
    def precompute_geometry(self, chunks) -> list[tuple]: ...

    @abstractmethod
    def fit_chunk(self, chunk, *geometry) -> dict[str, Any]: ...


class ParallelChunkRunner:
    def __init__(self, swapi_response, efficiency_table):
        self._swapi_response = swapi_response
        self._efficiency_table = efficiency_table

    def run(self, chunks, fitter: ChunkFitter) -> dict[str, np.ndarray]:
        geometries = fitter.precompute_geometry(chunks)

        with multiprocessing.get_context("fork").Pool(
            processes=os.cpu_count(),
            initializer=_init_worker,
            initargs=(self._swapi_response, self._efficiency_table, fitter),
        ) as pool:
            results = pool.starmap(
                _run_one, [(chunk, geom) for chunk, geom in zip(chunks, geometries)]
            )

        return {k: np.array([r[k] for r in results]) for k in results[0].keys()}


def _init_worker(swapi_response, efficiency_table, fitter):
    _shared["swapi_response"] = swapi_response
    _shared["efficiency_table"] = efficiency_table
    _shared["fitter"] = fitter


def _run_one(chunk, geom):
    return _shared["fitter"].fit_chunk(chunk, *geom)


class ProtonChunkFitter(ChunkFitter):
    def precompute_geometry(self, chunks):
        geometries = []
        for chunk in chunks:
            epoch = chunk_epoch(chunk)
            rm = None
            sc_vel = None
            tracker = PredictedEphemerisTracker()
            try:
                rm = tracker.run(get_swapi_geometry, measurement_times(chunk, SWAPI_SCIENCE_BINS))
            except Exception:
                logger.warning(
                    "SPICE gap in rotation matrices.",
                    exc_info=True,
                )

            if rm is not None:
                try:
                    sc_vel = tracker.run(get_spacecraft_velocity_rtn, epoch)
                except Exception:
                    logger.warning(
                        "SPICE gap in spacecraft velocity.",
                        exc_info=True,
                    )
            flags = SwapiL3Flags.PREDICTIVE_EPHEMERIS if tracker.used_predict else SwapiL3Flags.NONE
            geometries.append((epoch, rm, sc_vel, flags))

        return geometries

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        rotation_matrices,
        sc_velocity_rtn,
        geometry_quality_flags: SwapiL3Flags,
    ):
        result = _fit_proton(data_chunk, epoch, rotation_matrices)
        moments = _proton_moments_from_fit(result, epoch, data_chunk, sc_velocity_rtn)
        moments["quality_flags"] |= geometry_quality_flags
        return moments

class AlphaChunkFitter(ChunkFitter):
    def __init__(self, mag_data):
        self.mag_data = mag_data

    def precompute_geometry(self, chunks):
        geometries = []
        for chunk in chunks:
            epoch = chunk_epoch(chunk)
            rm = None
            sc_vel = None
            tracker = PredictedEphemerisTracker()
            try:
                rm = tracker.run(get_swapi_geometry, measurement_times(chunk, SWAPI_SCIENCE_BINS))
            except Exception:
                logger.info(
                    f"Missing SPICE information at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}, continuing with fill value"
                )

            if rm is not None:
                try:
                    sc_vel = tracker.run(get_spacecraft_velocity_rtn, epoch)
                except Exception:
                    logger.warning(
                        "SPICE gap in spacecraft velocity.",
                        exc_info=True,
                    )

            b_hat = compute_direction_of_mean_magnetic_field_over_chunk(
                self.mag_data, int(epoch), int(THIRTY_SECONDS_IN_NANOSECONDS)
            )
            flags = SwapiL3Flags.PREDICTIVE_EPHEMERIS if tracker.used_predict else SwapiL3Flags.NONE
            geometries.append((epoch, rm, sc_vel, b_hat, flags))
        return geometries

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        rotation_matrices,
        sc_velocity_rtn,
        magnetic_field_direction,
        geometry_quality_flags: SwapiL3Flags,
    ):
        result = _fit_alpha(
            data_chunk, epoch, rotation_matrices, magnetic_field_direction
        )
        moments = _alpha_moments_from_fit(result, epoch, sc_velocity_rtn)
        moments["quality_flags"] |= geometry_quality_flags
        return moments


class PuiChunkFitter(ChunkFitter):
    def __init__(
        self,
        density_of_neutral_helium_lookup_table,
        hydrogen_inflow_vector,
        helium_inflow_vector,
        proton_results: dict,
    ):
        self.density_of_neutral_helium_lookup_table = (
            density_of_neutral_helium_lookup_table
        )
        self.hydrogen_inflow_vector = hydrogen_inflow_vector
        self.helium_inflow_vector = helium_inflow_vector
        self.proton_results = proton_results
        self.sw_velocity_rtn_by_chunk_epoch: dict[int, np.ndarray] = {}

    def precompute_geometry(self, chunks):
        """Average the 5-minute proton fits into the 10-minute cadence used by
        50-sweep PUI chunks, rotate each 10-minute RTN average into IMAP_SWAPI
        at every coarse-bin measurement time, and precompute the chunk SPICE
        state. SPICE gaps fall back to NaN/None fills so they propagate to fill
        values in the downstream fit."""
        ten_minute_sw_velocities_rtn, proton_sw_quality_flags = (
            calculate_ten_minute_velocities(
                self.proton_results["proton_sw_velocity_rtn"],
                list(self.proton_results["quality_flags"]),
            )
        )

        n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
        geometries = []
        for chunk, ten_minute_rtn, flag in zip(
            chunks, ten_minute_sw_velocities_rtn, proton_sw_quality_flags
        ):
            epoch = int(chunk.sci_start_time[0]) + FIVE_MINUTES_IN_NANOSECONDS
            self.sw_velocity_rtn_by_chunk_epoch[epoch] = ten_minute_rtn
            proton_sw_quality_flag = int(flag)
            n_sweeps = chunk.sci_start_time.shape[0]

            lower_energy_cutoff: float | None = None
            upper_energy_cutoff: float | None = None
            vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution | None = None

            tracker = PredictedEphemerisTracker()

            if np.any(np.isnan(ten_minute_rtn)):
                bulk_sw_per_bin_swapi = np.full((n_sweeps, n_coarse_bins, 3), np.nan)
            else:
                try:
                    bulk_sw_per_bin_swapi = tracker.run(rotate_rtn_velocity_to_swapi_per_bin,
                    chunk, ten_minute_rtn)
                except SpiceyError:
                    logger.info(
                        "SPICE gap when rotating proton SW velocity to IMAP_SWAPI "
                        f"in PUI chunk at epoch {epoch}; using fill value."
                    )
                    bulk_sw_per_bin_swapi = np.full((n_sweeps, n_coarse_bins, 3), np.nan)

                try:
                    chunk_ephemeris_time = spiceypy.unitim(
                        epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"
                    )
                    lower_energy_cutoff = 1.25 * tracker.run(calculate_pui_energy_cutoff,
                        PROTON_MASS_KG,
                        chunk_ephemeris_time,
                        ten_minute_rtn,
                        self.hydrogen_inflow_vector,
                    )
                    upper_energy_cutoff = 1.2 * tracker.run(calculate_pui_energy_cutoff,
                        HE_PUI_PARTICLE_MASS_KG,
                        chunk_ephemeris_time,
                        ten_minute_rtn,
                        self.helium_inflow_vector,
                    )
                    vasyliunas_siscoe_distribution = tracker.run(build_vasyliunas_siscoe_distribution,
                        chunk_ephemeris_time,
                        ten_minute_rtn,
                        self.density_of_neutral_helium_lookup_table,
                        self.helium_inflow_vector,
                    )
                except SpiceyError:
                    logger.info(
                        f"SPICE gap precomputing PUI chunk state at epoch {epoch};"
                        " using fill value."
                    )
                    lower_energy_cutoff = None
                    upper_energy_cutoff = None
                    vasyliunas_siscoe_distribution = None

            flag = SwapiL3Flags.PREDICTIVE_EPHEMERIS if tracker.used_predict else SwapiL3Flags.NONE

            geometries.append((
                epoch,
                ten_minute_rtn,
                bulk_sw_per_bin_swapi,
                proton_sw_quality_flag | flag,
                lower_energy_cutoff,
                upper_energy_cutoff,
                vasyliunas_siscoe_distribution,
            ))
        return geometries

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        sw_velocity_rtn,
        bulk_sw_per_bin_swapi,
        quality_flag,
        lower_energy_cutoff,
        upper_energy_cutoff,
        vasyliunas_siscoe_distribution,
    ):
        count_rates_window = data_chunk.coincidence_count_rate[
            :, SWAPI_COARSE_SWEEP_BINS
        ]
        if np.any(np.isnan(count_rates_window)):
            return _pui_fill_result(
                epoch, quality_flag, "gap in L2 coincidence count rate"
            )
        if np.any(np.isnan(sw_velocity_rtn)):
            return _pui_fill_result(
                epoch, quality_flag,
                "gap in 10-minute proton solar wind velocity (no usable proton fit in the window)",
            )
        if vasyliunas_siscoe_distribution is None:
            return _pui_fill_result(
                epoch, quality_flag,
                "no neutral-helium Vasyliunas-Siscoe distribution (SPICE gap building chunk state)",
            )

        try:
            voltages = (
                data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
            ).flatten()
            central_effective_area_scale = _shared[
                "efficiency_table"
            ].central_effective_area_scale_for(epoch, "helium")
            fit_result = calculate_pickup_ion_values(
                _shared["swapi_response"],
                voltages,
                count_rates_window.flatten(),
                sw_velocity_rtn,
                bulk_sw_per_bin_swapi,
                self.density_of_neutral_helium_lookup_table,
                lower_energy_cutoff,
                upper_energy_cutoff,
                vasyliunas_siscoe_distribution,
                central_effective_area_scale=central_effective_area_scale,
            )
        except Exception:
            logger.warning(
                f"PUI fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
                exc_info=True,
            )
            return _pui_fill_result(epoch, quality_flag)

        fit_params = fit_result.fitting_params
        density = temperature = ufloat(np.nan, np.nan)
        if not (int(fit_params.flags) & int(SwapiL3Flags.BAD_FIT)):
            density = calculate_helium_pui_density(
                fit_result.chunk_response,
                fit_result.vasyliunas_siscoe_distribution,
                fit_params,
            )
            temperature = calculate_helium_pui_temperature(
                fit_result.chunk_response,
                fit_result.vasyliunas_siscoe_distribution,
                fit_params,
            )

        return dict(
            epoch=epoch,
            cooling_index=fit_params.cooling_index,
            ionization_rate=fit_params.ionization_rate,
            cutoff_speed=fit_params.cutoff_speed,
            background_rate=fit_params.background_count_rate,
            density=density,
            temperature=temperature,
            quality_flags=int(quality_flag) | int(fit_params.flags),
        )


def _proton_moments_from_fit(result, epoch, data_chunk, sc_velocity_rtn):
    speed = sum(component**2 for component in result.velocity_rtn) ** 0.5
    speed_nom, speed_unc = speed.nominal_value, speed.std_dev
    velocity_rtn_sc = result.velocity_rtn_nominal()
    velocity_rtn_covariance = result.velocity_rtn_covariance()
    density_nom, density_unc = result.density.nominal_value, result.density.std_dev
    temp_nom, temp_unc = result.temperature.nominal_value, result.temperature.std_dev

    if sc_velocity_rtn is not None:
        velocity_rtn_sun = velocity_rtn_sc + sc_velocity_rtn
        sun_velocity_unc = [
            component + sc_component
            for component, sc_component in zip(
                result.velocity_rtn, sc_velocity_rtn
            )
        ]
        sun_speed = sum(component**2 for component in sun_velocity_unc) ** 0.5
        sun_speed_nom, sun_speed_unc = sun_speed.nominal_value, sun_speed.std_dev
    else:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing spacecraft velocity; sun-frame outputs are fill values"
        )
        velocity_rtn_sun = np.full(3, np.nan)
        sun_speed_nom = sun_speed_unc = np.nan

    if not np.isfinite(speed_nom):
        speed_nom = _peak_proton_speed_kms(data_chunk)

    return dict(
        epoch=epoch,
        proton_sw_speed=speed_nom,
        proton_sw_speed_uncert=speed_unc,
        proton_sw_speed_sun=sun_speed_nom,
        proton_sw_speed_sun_uncert=sun_speed_unc,
        proton_sw_temperature=temp_nom,
        proton_sw_temperature_uncert=temp_unc,
        proton_sw_density=density_nom,
        proton_sw_density_uncert=density_unc,
        proton_sw_velocity_rtn_sun=velocity_rtn_sun,
        proton_sw_velocity_rtn=velocity_rtn_sc,
        proton_sw_velocity_rtn_covariance=velocity_rtn_covariance,
        quality_flags=result.quality_flag,
    )


def _alpha_moments_from_fit(result, epoch, sc_velocity_rtn):
    alpha = result.alpha_moments
    speed = sum(component**2 for component in alpha.velocity_rtn) ** 0.5
    velocity_rtn_sc = alpha.velocity_rtn_nominal()
    velocity_rtn_covariance = alpha.velocity_rtn_covariance()

    if sc_velocity_rtn is not None:
        velocity_rtn_sun = velocity_rtn_sc + sc_velocity_rtn
        sun_velocity = [
            component + sc_component
            for component, sc_component in zip(alpha.velocity_rtn, sc_velocity_rtn)
        ]
        sun_speed = sum(component**2 for component in sun_velocity) ** 0.5
        sun_speed_nom, sun_speed_unc = sun_speed.nominal_value, sun_speed.std_dev
    else:
        logger.warning(
            f"Alpha fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing spacecraft velocity; sun-frame outputs are fill values"
        )
        velocity_rtn_sun = np.full(3, np.nan)
        sun_speed_nom = sun_speed_unc = np.nan

    return dict(
        epoch=epoch,
        alpha_sw_speed=speed.nominal_value,
        alpha_sw_speed_uncert=speed.std_dev,
        alpha_sw_speed_sun=sun_speed_nom,
        alpha_sw_speed_sun_uncert=sun_speed_unc,
        alpha_sw_density=alpha.density.nominal_value,
        alpha_sw_density_uncert=alpha.density.std_dev,
        alpha_sw_temperature=alpha.temperature.nominal_value,
        alpha_sw_temperature_uncert=alpha.temperature.std_dev,
        alpha_sw_velocity_rtn_sun=velocity_rtn_sun,
        alpha_sw_velocity_rtn=velocity_rtn_sc,
        alpha_sw_velocity_rtn_covariance=velocity_rtn_covariance,
        quality_flags=result.quality_flag,
    )


_PEAK_SPEED_HALF_WIDTH = 4


def _peak_proton_speed_kms(data_chunk) -> float:
    count_rate = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
    voltage = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
    mean_count_rate = np.nanmean(count_rate, axis=0)
    mean_voltage = np.nanmean(voltage, axis=0)
    if not np.any(np.isfinite(mean_count_rate)):
        return np.nan
    peak_index = int(np.nanargmax(mean_count_rate))
    n_bins = mean_count_rate.shape[0]
    clamped_peak_index = max(
        _PEAK_SPEED_HALF_WIDTH, min(peak_index, n_bins - _PEAK_SPEED_HALF_WIDTH - 1)
    )
    window = slice(
        clamped_peak_index - _PEAK_SPEED_HALF_WIDTH,
        clamped_peak_index + _PEAK_SPEED_HALF_WIDTH + 1,
    )
    rate_window = mean_count_rate[window]
    voltage_window = mean_voltage[window]
    valid = (
        np.isfinite(rate_window)
        & np.isfinite(voltage_window)
        & (voltage_window > 0.0)
        & (rate_window >= 0.0)
    )
    if not np.any(valid):
        return np.nan
    speed = esa_voltage_to_proton_speed(voltage_window[valid])
    weights = rate_window[valid]
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.nan
    return float(np.sum(weights * speed) / weight_sum)


def _coarse_subset_of_science_rotations(
    science_rotation_matrices: np.ndarray, n_sweeps: int
) -> np.ndarray:
    """Slice per-sweep coarse-bin rotations out of a flat science-bin rotation
    array. Coarse bins are the first 62 of the 71 science bins per sweep."""
    n_science_bins = SWAPI_SCIENCE_BINS.stop - SWAPI_SCIENCE_BINS.start
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    return (
        science_rotation_matrices.reshape(n_sweeps, n_science_bins, 3, 3)[
            :, :n_coarse_bins
        ]
        .reshape(-1, 3, 3)
    )


def _fit_proton(
    data_chunk, epoch, rotation_matrices
) -> ProtonSolarWindFitResult:
    if rotation_matrices is None:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing rotation matrices; using fill values"
        )
        return _nan_proton_result(SwapiL3Flags.NONE)
  
    if np.any(np.isnan(data_chunk.coincidence_count_rate[:, SWAPI_SCIENCE_BINS])):
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: NaN in input count rate; using fill values"
        )
        return _nan_proton_result(SwapiL3Flags.NONE)
  
    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    count_rates = data_chunk.coincidence_count_rate[:, SWAPI_SCIENCE_BINS]
    voltages = data_chunk.energy[:, SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR
    voltage_valid = (voltages > 0) & np.isfinite(voltages)
    keep_bins = np.all(voltage_valid, axis=0)
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    
    if not np.all(keep_bins[:n_coarse_bins]):
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: invalid voltage in coarse-sweep bin; using fill values"
        )
        return _nan_proton_result(SwapiL3Flags.FIT_ERROR)
    
    if not np.all(keep_bins):
        n_sweeps_local, n_bins_local = voltages.shape
        voltages = voltages[:, keep_bins]
        count_rates = count_rates[:, keep_bins]
        rotation_matrices = (
            rotation_matrices.reshape(n_sweeps_local, n_bins_local, 3, 3)[:, keep_bins]
            .reshape(-1, 3, 3)
        )
   
    ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=efficiency_table.central_effective_area_scale_for(epoch, "proton"),
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    try:
        result = fit_solar_wind_proton_model(ctx)
    except Exception:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
            exc_info=True,
        )
        return _nan_proton_result(SwapiL3Flags.FIT_ERROR)

    return result


def _pui_fill_result(epoch, proton_sw_quality_flag, reason=None) -> dict:
    if reason is not None:
        logger.warning(
            f"PUI fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: "
            f"{reason}; using fill values"
        )
    nan = ufloat(np.nan, np.nan)
    return dict(
        epoch=epoch,
        cooling_index=nan,
        ionization_rate=nan,
        cutoff_speed=nan,
        background_rate=nan,
        density=nan,
        temperature=nan,
        quality_flags=int(proton_sw_quality_flag),
    )


def _nan_proton_result(flag) -> ProtonSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return ProtonSolarWindFitResult(
        density=nan,
        temperature=nan,
        velocity_rtn=(nan, nan, nan),
        quality_flag=int(flag),
    )


@dataclass
class AlphaChunkFitResult:
    """Chunk-level alpha output: alpha moments + the proton moments they
    reference + the B̂ used as the field-aligned drift constraint, plus the
    consolidated `bad_fit_flag` rolled up from every stage that could fail
    (missing B̂, proton fit, alpha LM)."""

    alpha_moments: AlphaSolarWindFitResult
    proton_moments: ProtonSolarWindFitResult
    b_hat_rtn: np.ndarray
    quality_flag: int


def _nan_alpha_fit_result(flag) -> AlphaSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return AlphaSolarWindFitResult(
        density=nan,
        temperature=nan,
        velocity_rtn=(nan, nan, nan),
        delta_v=nan,
        quality_flag=int(flag),
    )


def _fit_alpha(
    data_chunk,
    epoch,
    rotation_matrices,
    magnetic_field_direction,
) -> AlphaChunkFitResult:
    nan_b_hat = np.full(3, np.nan)
    if (
        magnetic_field_direction is None
        or not np.all(np.isfinite(magnetic_field_direction))
    ):
        logger.warning(
            f"Alpha fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing or non-finite magnetic field direction; using fill values"
        )
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(SwapiL3Flags.NONE),
            proton_moments=_nan_proton_result(SwapiL3Flags.NONE),
            b_hat_rtn=nan_b_hat,
            quality_flag=int(SwapiL3Flags.NONE),
        )

    proton_moments = _fit_proton(data_chunk, epoch, rotation_matrices)
    if not np.all(
        np.isfinite(
            [component.nominal_value for component in proton_moments.velocity_rtn]
        )
    ):
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(proton_moments.quality_flag),
            proton_moments=proton_moments,
            b_hat_rtn=nan_b_hat,
            quality_flag=int(proton_moments.quality_flag),
        )

    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    try:
        count_rates = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
        voltages = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
        coarse_rotation_matrices = _coarse_subset_of_science_rotations(
            rotation_matrices, n_sweeps=data_chunk.sci_start_time.shape[0]
        )
        proton_ctx = build_solar_wind_fit_context(
            count_rate=count_rates,
            esa_voltage=voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=efficiency_table.central_effective_area_scale_for(epoch, "proton"),
            rotation_matrices=coarse_rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=count_rates,
            esa_voltage=voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=efficiency_table.central_effective_area_scale_for(epoch, "helium"),
            rotation_matrices=coarse_rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_moments = fit_solar_wind_alpha_model(
            proton_ctx=proton_ctx,
            alpha_ctx=alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=magnetic_field_direction,
        )
    except Exception:
        logger.warning(
            f"Alpha fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
            exc_info=True,
        )
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(SwapiL3Flags.FIT_ERROR),
            proton_moments=proton_moments,
            b_hat_rtn=nan_b_hat,
            quality_flag=int(SwapiL3Flags.FIT_ERROR),
        )

    return AlphaChunkFitResult(
        alpha_moments=alpha_moments,
        proton_moments=proton_moments,
        b_hat_rtn=magnetic_field_direction,
        quality_flag=int(alpha_moments.quality_flag),
    )
