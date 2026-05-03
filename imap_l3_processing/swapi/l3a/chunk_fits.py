"""Per-chunk fitting workers for SwapiProcessor's L3a parallel pipeline.

`run_parallel_chunks` is the orchestrator: it precomputes geometry per chunk
in the parent process and submits all chunks to a fork-context Pool.starmap.
Geometry functions return None-filled tuples on SPICE gaps; workers handle them
via their existing try/except. Shared state (SWAPIResponse cache, calibration
table) is accessed through the module-level `_shared` dict, populated by
`init_worker`.
"""

import logging
import multiprocessing
import os
from typing import Any

import numpy as np
from uncertainties import ufloat, umath

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
from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS

logger = logging.getLogger(__name__)

_shared = {}


def run_parallel_chunks(
    chunks, swapi_response, efficiency_table, geometry_fn, worker_fn
) -> dict[str, Any]:
    """Precompute geometry per chunk in the parent, then submit to a
    fork-context process pool. Geometry functions return None-filled tuples on
    SPICE gaps; workers handle them via their existing try/except.
    Returns a dict of stacked arrays, one entry per worker output key."""
    geometries = [geometry_fn(chunk) for chunk in chunks]

    with multiprocessing.get_context("fork").Pool(
        processes=os.cpu_count(),
        initializer=init_worker,
        initargs=(swapi_response, efficiency_table),
    ) as pool:
        results = pool.starmap(
            worker_fn, [(chunk, *geom) for chunk, geom in zip(chunks, geometries)]
        )

    return {k: np.array([r[k] for r in results]) for k in results[0].keys()}


def init_worker(swapi_response, efficiency_table):
    _shared["swapi_response"] = swapi_response
    _shared["efficiency_table"] = efficiency_table


def proton_geometry(chunk):
    epoch = chunk_epoch(chunk)
    try:
        rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
        sc_vel = get_spacecraft_velocity_rtn(epoch)
    except Exception:
        logger.warning("SPICE gap in proton geometry, NaN-filling chunk", exc_info=True)
        rm = sc_vel = None
    return (epoch, rm, sc_vel)


def pui_geometry(chunk):
    epoch = chunk_epoch(chunk)
    try:
        rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
    except Exception:
        logger.warning("SPICE gap in pui geometry, NaN-filling chunk", exc_info=True)
        rm = None
    return (epoch, rm)


def alpha_geometry(chunk, mag_l1d_data):
    epoch = chunk_epoch(chunk)
    try:
        rm = get_swapi_geometry(measurement_times(chunk, SWAPI_COARSE_SWEEP_BINS))
        b_hat = compute_b_hat_rtn(
            mag_l1d_data, int(epoch), int(THIRTY_SECONDS_IN_NANOSECONDS)
        )
    except Exception:
        logger.warning("Geometry gap in alpha fit, NaN-filling chunk", exc_info=True)
        rm = b_hat = None
    return (epoch, rm, b_hat)


def proton_chunk_worker(data_chunk, epoch, rotation_matrices, sc_velocity_rtn):
    speed_nom = speed_unc = clock_nom = clock_unc = defl_nom = defl_unc = np.nan
    sun_speed_nom = sun_speed_unc = np.nan
    density_nom = density_unc = temp_nom = temp_unc = np.nan
    bulk_velocity_rtn_sun = np.full(3, np.nan)
    bulk_velocity_rtn_sc = np.full(3, np.nan)
    velocity_covariance = np.full((3, 3), np.nan)
    quality_flag = SwapiL3Flags.NONE
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")
        result = _fit_proton(data_chunk, epoch, SWAPI_SCIENCE_BINS, rotation_matrices)
        quality_flag |= result.bad_fit_flag
        speed, clock_angle, deflection_angle = derive_velocity_angles(result, epoch)
        speed_nom, speed_unc = speed.nominal_value, speed.std_dev
        clock_nom, clock_unc = clock_angle.nominal_value, clock_angle.std_dev
        defl_nom, defl_unc = deflection_angle.nominal_value, deflection_angle.std_dev
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
        density_nom, density_unc = result.density.nominal_value, result.density.std_dev
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


def pui_proton_chunk_worker(data_chunk, epoch, rotation_matrices):
    speed = ufloat(np.nan, np.nan)
    clock_angle = ufloat(np.nan, np.nan)
    deflection_angle = ufloat(np.nan, np.nan)
    quality_flag = SwapiL3Flags.NONE
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")
        result = _fit_proton(data_chunk, epoch, SWAPI_SCIENCE_BINS, rotation_matrices)
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


def alpha_chunk_worker(data_chunk, epoch, rotation_matrices, b_hat_rtn):
    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
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
        return dict(
            epoch=epoch,
            alpha_sw_density=mom.density.nominal_value,
            alpha_sw_density_uncert=mom.density.std_dev,
            alpha_sw_temperature=mom.temperature.nominal_value,
            alpha_sw_temperature_uncert=mom.temperature.std_dev,
            alpha_sw_velocity_rtn=mom.bulk_velocity_rtn_nominal(),
            alpha_sw_velocity_covariance_rtn=mom.bulk_velocity_rtn_covariance(),
            alpha_sw_delta_v=mom.delta_v.nominal_value,
            alpha_sw_delta_v_uncert=mom.delta_v.std_dev,
            alpha_sw_b_hat_rtn=b_hat_rtn,
            alpha_sw_reference_proton_density=proton_moments.density.nominal_value,
            alpha_sw_reference_proton_temperature=proton_moments.temperature.nominal_value,
            alpha_sw_reference_proton_velocity_rtn=proton_moments.bulk_velocity_rtn_nominal(),
            bad_fit_flag=int(mom.bad_fit_flag),
        )
    except Exception:
        logger.info(
            f"Alpha moments fit exception at epoch {epoch}; using NaN fill",
            exc_info=True,
        )
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
            bad_fit_flag=int(SwapiL3Flags.BAD_FIT),
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
