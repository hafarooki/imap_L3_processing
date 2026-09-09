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
    ONE_SECOND_IN_NANOSECONDS,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    PickupIonFitInputData,
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import (
    rotate_rtn_velocity_to_swapi_per_bin,
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
from imap_l3_processing.swapi.species import Species
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.l3a.utils import (
    compute_direction_of_mean_magnetic_field_over_chunk,
    esa_voltage_to_proton_speed,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
    pickup_ion_chunk_epoch,
    solar_wind_chunk_epoch,
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
            epoch = solar_wind_chunk_epoch(chunk)
            rm = None
            sc_vel = None
            tracker = PredictedEphemerisTracker()
            try:
                science_bin_times = measurement_times(chunk.sci_start_time)[:, SWAPI_SCIENCE_BINS]
                rm = tracker.run(get_swapi_geometry, science_bin_times.ravel())
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
            epoch = solar_wind_chunk_epoch(chunk)
            rm = None
            sc_vel = None
            tracker = PredictedEphemerisTracker()
            try:
                science_bin_times = measurement_times(chunk.sci_start_time)[:, SWAPI_SCIENCE_BINS]
                rm = tracker.run(get_swapi_geometry, science_bin_times.ravel())
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
        proton_sw_results: dict,
    ):
        self.density_of_neutral_helium_lookup_table = (
            density_of_neutral_helium_lookup_table
        )
        self.hydrogen_inflow_vector = hydrogen_inflow_vector
        self.helium_inflow_vector = helium_inflow_vector
        self.proton_sw_results = proton_sw_results

    def _calculate_ten_minute_velocities(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average the 1-minute proton bulk SW velocity vectors over
            10-minute chunks and compute the bitwise-OR of the per-minute flags."""
        velocities_rtn = self.proton_sw_results["proton_sw_velocity_rtn"]
        velocities_rtn_sun = self.proton_sw_results["proton_sw_velocity_rtn_sun"]
        quality_flags = list(self.proton_sw_results["quality_flags"])

        ten_minute_velocities_rtn = []
        ten_minute_velocities_rtn_sun = []
        ten_minute_quality_flags = []
        for left_slice in range(0, len(velocities_rtn), 10):
            ten_min_slice = slice(left_slice, left_slice + 10)

            ten_minute_velocities_rtn.append(
                np.mean(velocities_rtn[ten_min_slice], axis=0)
            )
            ten_minute_velocities_rtn_sun.append(
                np.mean(velocities_rtn_sun[ten_min_slice], axis=0)
            )
            ten_minute_quality_flags.append(
                np.bitwise_or.reduce(quality_flags[ten_min_slice])
            )
        return (
            np.array(ten_minute_velocities_rtn),
            np.array(ten_minute_velocities_rtn_sun),
            np.array(ten_minute_quality_flags),
        )

    def precompute_geometry(self, chunks):
        (
            ten_minute_sw_velocities_rtn,
            ten_minute_sw_velocities_rtn_sun,
            proton_sw_quality_flags,
        ) = self._calculate_ten_minute_velocities()

        input_data_list = []
        for chunk, sw_velocity_rtn_sc, sw_velocity_rtn_sun, sw_flags in zip(
            chunks,
            ten_minute_sw_velocities_rtn,
            ten_minute_sw_velocities_rtn_sun,
            proton_sw_quality_flags,
        ):
            epoch = pickup_ion_chunk_epoch(chunk)
            proton_sw_quality_flag = int(sw_flags)

            if (
                np.any(np.isnan(sw_velocity_rtn_sc))
                or np.any(np.isnan(sw_velocity_rtn_sun))
            ):
                logger.info(f"solar wind velocity gap at {epoch}")
                input_data_list.append((None, proton_sw_quality_flag))
                continue

            tracker = PredictedEphemerisTracker()

            try:
                bulk_sw_per_bin_swapi = tracker.run(
                    rotate_rtn_velocity_to_swapi_per_bin,
                    chunk,
                    sw_velocity_rtn_sc
                )

                chunk_ephemeris_time = spiceypy.unitim(
                    epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"
                )

                def compute_chunk_position(ephemeris_time: float) -> tuple[float, float]:
                    imap_position_eclipj2000_frame = spiceypy.spkezr(
                        "IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN"
                    )[0][0:3]
                    distance_km, longitude, _latitude = spiceypy.reclat(
                        imap_position_eclipj2000_frame
                    )
                    inflow_angle = (
                        np.rad2deg(longitude)
                        - self.helium_inflow_vector.longitude_deg_eclipj2000
                    )
                    return distance_km, inflow_angle
                distance, inflow_angle = tracker.run(
                    compute_chunk_position, chunk_ephemeris_time
                )
            except SpiceyError:
                logger.info(f"SPICE gap at {epoch}")
                input_data_list.append((None, proton_sw_quality_flag))
                continue

            if tracker.used_predict:
                proton_sw_quality_flag |= SwapiL3Flags.PREDICTIVE_EPHEMERIS

            input_data_list.append((
                PickupIonFitInputData(
                    time_as_tt2000=epoch,
                    esa_energies=chunk.energy[:, SWAPI_COARSE_SWEEP_BINS].mean(axis=0),
                    coincidence_count_rates=chunk.coincidence_count_rate[
                        :, SWAPI_COARSE_SWEEP_BINS
                    ],
                    bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi[
                        :, SWAPI_COARSE_SWEEP_BINS, :
                    ],
                    solar_wind_velocity_rtn_sun=sw_velocity_rtn_sun,
                    distance=distance,
                    inflow_angle=inflow_angle,
                ),
                proton_sw_quality_flag,
            ))

        return input_data_list

    def fit_chunk(
        self,
        data_chunk,
        fit_input: PickupIonFitInputData | None,
        quality_flag: SwapiL3Flags,
    ):
        epoch = pickup_ion_chunk_epoch(data_chunk)

        count_rates_window = data_chunk.coincidence_count_rate[
            :, SWAPI_COARSE_SWEEP_BINS
        ]
        if np.any(np.isnan(count_rates_window)):
            return _pui_fill_result(
                epoch, quality_flag, "gap in L2 coincidence count rate"
            )
        if fit_input is None:
            return _pui_fill_result(
                epoch, quality_flag,
                "failed to load input data",
            )

        try:
            fit_result = calculate_pickup_ion_values(
                fit_input,
                swapi_response=_shared["swapi_response"],
                density_of_neutral_helium_lookup_table=(
                    self.density_of_neutral_helium_lookup_table
                ),
            )
        except Exception:
            logger.warning(
                f"PUI fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
                exc_info=True,
            )
            return _pui_fill_result(epoch, quality_flag)

        return dict(
            epoch=epoch,
            ionization_rate=fit_result.ionization_rate,
            cutoff_speed=fit_result.cutoff_speed,
            density=fit_result.density,
            temperature=fit_result.temperature,
            quality_flags=int(quality_flag) | int(fit_result.flags),
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
        time_as_tt2000=epoch,
        species=Species.PROTON,
        rotation_matrices=rotation_matrices,
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
        ionization_rate=nan,
        cutoff_speed=nan,
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
            time_as_tt2000=epoch,
            species=Species.PROTON,
            rotation_matrices=coarse_rotation_matrices,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=count_rates,
            esa_voltage=voltages,
            swapi_response=swapi_response,
            time_as_tt2000=epoch,
            species=Species.ALPHA,
            rotation_matrices=coarse_rotation_matrices,
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
