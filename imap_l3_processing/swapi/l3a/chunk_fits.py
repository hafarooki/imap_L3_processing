"""Per-chunk fitting strategies for SwapiProcessor's L3a parallel pipeline.

`ChunkFitter` is the strategy base class; `ProtonChunkFitter`,
`PuiProtonChunkFitter`, and `AlphaChunkFitter` each own one science pipeline.
`ParallelChunkRunner` precomputes geometry per chunk in the parent process and
submits each chunk to a fork-context Pool. When SPICE geometry is unavailable
`precompute_geometry` returns ``None`` for the rotation-matrix slot and
`fit_chunk` emits ``EPHEMERIS_GAP`` (not ``BAD_FIT``). For ``AlphaChunkFitter``
a second independent sentinel tracks MAG availability: a NaN ``b_hat_rtn``
while SPICE succeeded emits ``MAG_GAP``. ``BAD_FIT`` is reserved for chunks
where geometry was valid but the optimizer or peak-finder failed. Shared state
(SWAPIResponse cache, calibration table, the fitter itself) is passed once per
worker via `Pool` `initargs` and lives in the module-level `_shared` dict, so
per-chunk pickling stays bounded to chunk + geometry tuple.
"""

import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from uncertainties import ufloat, umath

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_moments import (
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    derive_velocity_angles,
    fit_solar_wind_proton_moments,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
    extract_coarse_sweep,
)
from imap_l3_processing.swapi.l3a.utils import (
    chunk_epoch,
    compute_b_hat_rtn,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

logger = logging.getLogger(__name__)

_shared: dict[str, Any] = {}


class ChunkFitter(ABC):
    """A pipeline that turns one chunk of L2 sweeps into a dict of L3a outputs.

    Implementations split the work into a parent-side `precompute_geometry`
    (SPICE calls, can carry heavy state) and a worker-side `fit_chunk` that
    consumes the precomputed tuple plus shared resources from `_shared`.
    """

    @abstractmethod
    def precompute_geometry(self, chunk) -> tuple:
        """Run SPICE/MAG queries for this chunk in the parent process."""

    @abstractmethod
    def fit_chunk(self, chunk, *geometry) -> dict[str, Any]:
        """Fit one chunk and return the L3a output dict."""


class ParallelChunkRunner:
    """Owns the fork pool and the shared resources that workers need.

    One runner is constructed per processor invocation and reused across
    descriptors; its `run(chunks, fitter)` precomputes geometry in the parent
    and dispatches `fitter.fit_chunk` across worker processes.
    """

    def __init__(self, swapi_response, efficiency_table):
        self._swapi_response = swapi_response
        self._efficiency_table = efficiency_table

    def run(self, chunks, fitter: ChunkFitter) -> dict[str, np.ndarray]:
        geometries = [fitter.precompute_geometry(chunk) for chunk in chunks]

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
    def precompute_geometry(self, chunk):
        epoch = chunk_epoch(chunk)
        try:
            rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
            sc_vel = get_spacecraft_velocity_rtn(epoch)
        except Exception:
            logger.warning(
                "SPICE gap in proton geometry, NaN-filling chunk", exc_info=True
            )
            rm = sc_vel = None
        return (epoch, rm, sc_vel)

    def fit_chunk(self, data_chunk, epoch, rotation_matrices, sc_velocity_rtn):
        speed_nom = speed_unc = clock_nom = clock_unc = defl_nom = defl_unc = np.nan
        sun_speed_nom = sun_speed_unc = np.nan
        density_nom = density_unc = temp_nom = temp_unc = np.nan
        bulk_velocity_rtn_sun = np.full(3, np.nan)
        bulk_velocity_rtn_sc = np.full(3, np.nan)
        velocity_covariance = np.full((3, 3), np.nan)
        quality_flag = SwapiL3Flags.NONE
        if rotation_matrices is None or sc_velocity_rtn is None:
            quality_flag |= SwapiL3Flags.EPHEMERIS_GAP
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
                proton_sw_clock_angle=clock_nom,
                proton_sw_clock_angle_uncert=clock_unc,
                proton_sw_deflection_angle=defl_nom,
                proton_sw_deflection_angle_uncert=defl_unc,
                proton_sw_bulk_velocity_rtn_sun=bulk_velocity_rtn_sun,
                proton_sw_bulk_velocity_rtn_sun_covariance=velocity_covariance,
                proton_sw_bulk_velocity_rtn_sc=bulk_velocity_rtn_sc,
                proton_sw_bulk_velocity_rtn_sc_covariance=velocity_covariance,
                quality_flags=quality_flag,
            )
        try:
            if np.any(
                np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))
            ):
                raise ValueError("Fill values in input data")
            result = _fit_proton(
                data_chunk, epoch, SWAPI_SCIENCE_BINS, rotation_matrices
            )
            quality_flag |= result.bad_fit_flag
            speed, clock_angle, deflection_angle = derive_velocity_angles(result, epoch)
            speed_nom, speed_unc = speed.nominal_value, speed.std_dev
            clock_nom, clock_unc = clock_angle.nominal_value, clock_angle.std_dev
            defl_nom, defl_unc = (
                deflection_angle.nominal_value,
                deflection_angle.std_dev,
            )
            bulk_velocity_rtn_sc = result.bulk_velocity_rtn_nominal()
            bulk_velocity_rtn_sun = bulk_velocity_rtn_sc + sc_velocity_rtn
            velocity_covariance = result.bulk_velocity_rtn_covariance()
            sun_velocity_unc = [
                component + sc_component
                for component, sc_component in zip(
                    result.bulk_velocity_rtn, sc_velocity_rtn
                )
            ]
            sun_speed = umath.sqrt(sum(component**2 for component in sun_velocity_unc))
            sun_speed_nom, sun_speed_unc = sun_speed.nominal_value, sun_speed.std_dev
            density_nom, density_unc = (
                result.density.nominal_value,
                result.density.std_dev,
            )
            temp_nom, temp_unc = (
                result.temperature.nominal_value,
                result.temperature.std_dev,
            )
        except Exception:
            logger.info(
                f"Exception occurred at epoch {epoch}, continuing with fill value",
                exc_info=True,
            )
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
            proton_sw_clock_angle=clock_nom,
            proton_sw_clock_angle_uncert=clock_unc,
            proton_sw_deflection_angle=defl_nom,
            proton_sw_deflection_angle_uncert=defl_unc,
            proton_sw_bulk_velocity_rtn_sun=bulk_velocity_rtn_sun,
            proton_sw_bulk_velocity_rtn_sun_covariance=velocity_covariance,
            proton_sw_bulk_velocity_rtn_sc=bulk_velocity_rtn_sc,
            proton_sw_bulk_velocity_rtn_sc_covariance=velocity_covariance,
            quality_flags=quality_flag,
        )


class AlphaChunkFitter(ChunkFitter):
    def __init__(self, mag_data, mag_data_level):
        self.mag_data = mag_data
        # "l2" or "l1d" — when "l1d", PRELIMINARY_MAG is OR'd into every chunk's flag.
        self.mag_data_level = mag_data_level

    def __getstate__(self):
        # mag_data is consumed parent-side in precompute_geometry to derive
        # b_hat per chunk; workers only ever read b_hat from the geometry tuple.
        # Drop the full MAG arrays from pickle so they don't ride into each worker.
        # mag_data_level is small and needed in the worker to set PRELIMINARY_MAG.
        return {**self.__dict__, "mag_data": None}

    def precompute_geometry(self, chunk):
        epoch = chunk_epoch(chunk)
        try:
            rm = get_swapi_geometry(measurement_times(chunk, SWAPI_COARSE_SWEEP_BINS))
        except Exception:
            logger.warning(
                "SPICE gap in alpha geometry, NaN-filling chunk", exc_info=True
            )
            rm = None
        # compute_b_hat_rtn returns NaN arrays on MAG gaps — does not raise.
        b_hat = compute_b_hat_rtn(
            self.mag_data, int(epoch), int(THIRTY_SECONDS_IN_NANOSECONDS)
        )
        return (epoch, rm, b_hat)

    def fit_chunk(self, data_chunk, epoch, rotation_matrices, b_hat_rtn):
        swapi_response = _shared["swapi_response"]
        efficiency_table = _shared["efficiency_table"]
        density_nom = density_unc = np.nan
        temp_nom = temp_unc = np.nan
        delta_v_nom = delta_v_unc = np.nan
        velocity_rtn = np.full(3, np.nan)
        velocity_cov = np.full((3, 3), np.nan)
        b_hat_out = np.full(3, np.nan)
        ref_density = ref_temperature = np.nan
        ref_velocity = np.full(3, np.nan)
        preliminary_mag_bit = (
            int(SwapiL3Flags.PRELIMINARY_MAG) if self.mag_data_level == "l1d" else 0
        )
        bad_fit_flag = int(SwapiL3Flags.BAD_FIT) | preliminary_mag_bit
        if rotation_matrices is None:
            bad_fit_flag = int(SwapiL3Flags.EPHEMERIS_GAP) | preliminary_mag_bit
            return _nan_alpha_record(epoch, bad_fit_flag)
        if b_hat_rtn is None or not np.all(np.isfinite(b_hat_rtn)):
            bad_fit_flag = int(SwapiL3Flags.MAG_GAP) | preliminary_mag_bit
            return _nan_alpha_record(epoch, bad_fit_flag)
        try:
            if np.any(
                np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))
            ):
                raise ValueError("Fill values in input data")
            proton_moments = _fit_proton(
                data_chunk, epoch, SWAPI_COARSE_SWEEP_BINS, rotation_matrices
            )
            count_rates = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
            voltages = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
            times = measurement_times(data_chunk, SWAPI_COARSE_SWEEP_BINS)
            mom = fit_solar_wind_alpha_moments(
                count_rates.flatten(),
                voltages.flatten(),
                times,
                swapi_response,
                proton_moments,
                b_hat_rtn,
                _eff_scale(efficiency_table, epoch, "alpha"),
                _eff_scale(efficiency_table, epoch, "proton"),
                rotation_matrices=rotation_matrices,
            )
            density_nom, density_unc = mom.density.nominal_value, mom.density.std_dev
            temp_nom, temp_unc = (
                mom.temperature.nominal_value,
                mom.temperature.std_dev,
            )
            delta_v_nom, delta_v_unc = mom.delta_v.nominal_value, mom.delta_v.std_dev
            velocity_rtn = mom.bulk_velocity_rtn_nominal()
            velocity_cov = mom.bulk_velocity_rtn_covariance()
            b_hat_out = b_hat_rtn
            ref_density = proton_moments.density.nominal_value
            ref_temperature = proton_moments.temperature.nominal_value
            ref_velocity = proton_moments.bulk_velocity_rtn_nominal()
            bad_fit_flag = int(mom.bad_fit_flag) | preliminary_mag_bit
        except Exception:
            logger.info(
                f"Alpha moments fit exception at epoch {epoch}; using NaN fill",
                exc_info=True,
            )
        return dict(
            epoch=epoch,
            alpha_sw_density=density_nom,
            alpha_sw_density_uncert=density_unc,
            alpha_sw_temperature=temp_nom,
            alpha_sw_temperature_uncert=temp_unc,
            alpha_sw_velocity_rtn=velocity_rtn,
            alpha_sw_velocity_covariance_rtn=velocity_cov,
            alpha_sw_delta_v=delta_v_nom,
            alpha_sw_delta_v_uncert=delta_v_unc,
            alpha_sw_b_hat_rtn=b_hat_out,
            alpha_sw_reference_proton_density=ref_density,
            alpha_sw_reference_proton_temperature=ref_temperature,
            alpha_sw_reference_proton_velocity_rtn=ref_velocity,
            bad_fit_flag=bad_fit_flag,
        )


class PuiProtonChunkFitter(ChunkFitter):
    def precompute_geometry(self, chunk):
        epoch = chunk_epoch(chunk)
        try:
            rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
        except Exception:
            logger.warning(
                "SPICE gap in pui geometry, NaN-filling chunk", exc_info=True
            )
            rm = None
        return (epoch, rm)

    def fit_chunk(self, data_chunk, epoch, rotation_matrices):
        speed = ufloat(np.nan, np.nan)
        clock_angle = ufloat(np.nan, np.nan)
        deflection_angle = ufloat(np.nan, np.nan)
        quality_flag = SwapiL3Flags.NONE
        if rotation_matrices is None:
            quality_flag |= SwapiL3Flags.EPHEMERIS_GAP
            return dict(
                proton_sw_speed=speed,
                proton_sw_clock_angle=clock_angle,
                proton_sw_deflection_angle=deflection_angle,
                quality_flags=quality_flag,
            )
        try:
            if np.any(
                np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))
            ):
                raise ValueError("Fill values in input data")
            result = _fit_proton(
                data_chunk, epoch, SWAPI_SCIENCE_BINS, rotation_matrices
            )
            quality_flag |= result.bad_fit_flag
            speed, clock_angle, deflection_angle = derive_velocity_angles(result, epoch)
        except Exception:
            logger.info(
                f"Exception occurred at epoch {epoch}, continuing with fill value",
                exc_info=True,
            )
        return dict(
            proton_sw_speed=speed,
            proton_sw_clock_angle=clock_angle,
            proton_sw_deflection_angle=deflection_angle,
            quality_flags=quality_flag,
        )


def _nan_alpha_record(epoch, bad_fit_flag):
    """Return an all-NaN alpha chunk dict with the given flag."""
    return dict(
        epoch=epoch,
        alpha_sw_density=np.nan,
        alpha_sw_density_uncert=np.nan,
        alpha_sw_temperature=np.nan,
        alpha_sw_temperature_uncert=np.nan,
        alpha_sw_velocity_rtn=np.full(3, np.nan),
        alpha_sw_velocity_covariance_rtn=np.full((3, 3), np.nan),
        alpha_sw_delta_v=np.nan,
        alpha_sw_delta_v_uncert=np.nan,
        alpha_sw_b_hat_rtn=np.full(3, np.nan),
        alpha_sw_reference_proton_density=np.nan,
        alpha_sw_reference_proton_temperature=np.nan,
        alpha_sw_reference_proton_velocity_rtn=np.full(3, np.nan),
        bad_fit_flag=bad_fit_flag,
    )


def _eff_scale(efficiency_table, epoch, kind):
    eps_lab = float(efficiency_table.eps_p_lab)
    if kind == "proton":
        return float(efficiency_table.get_proton_efficiency_for(epoch)) / eps_lab
    return float(efficiency_table.get_alpha_efficiency_for(epoch)) / eps_lab


def _fit_proton(
    data_chunk, epoch, bin_slice, rotation_matrices
) -> ProtonSolarWindMoments:
    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    count_rates = data_chunk.coincidence_count_rate[:, bin_slice].flatten()
    voltages = data_chunk.energy[:, bin_slice].flatten() / SWAPI_L2_K_FACTOR
    return fit_solar_wind_proton_moments(
        count_rates,
        voltages,
        swapi_response,
        central_effective_area_scale=_eff_scale(efficiency_table, epoch, "proton"),
        rotation_matrices=rotation_matrices,
    )
