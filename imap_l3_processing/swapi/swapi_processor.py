import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values

from imap_l3_processing.constants import (
    THIRTY_SECONDS_IN_NANOSECONDS,
    FIVE_MINUTES_IN_NANOSECONDS,
    ONE_SECOND_IN_NANOSECONDS,
)
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.models import (
    SwapiL3ProtonSolarWindData,
    SwapiL3AlphaSolarWindData,
    SwapiL3PickupIonData,
)
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import (
    calculate_ten_minute_velocities,
    calculate_pickup_ion_values,
    calculate_helium_pui_temperature,
    calculate_helium_pui_density,
)
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_moments import (
    AlphaSolarWindMoments,
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    fit_solar_wind_proton_moments,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    calculate_combined_sweeps,
    extract_coarse_sweep,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_SCIENCE_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import (
    chunk_l2_data,
    get_spacecraft_velocity_rtn,
    get_swapi_dsrf_to_rtn,
    get_swapi_geometry,
)
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import (
    calculate_combined_solar_wind_differential_flux,
)
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import (
    calculate_proton_solar_wind_vdf,
    calculate_alpha_solar_wind_vdf,
    calculate_pui_solar_wind_vdf,
    calculate_delta_minus_plus,
)
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.utils import save_data

logger = logging.getLogger(__name__)

# Shared state for multiprocessing workers, populated by _mp_init_worker.
_mp_shared = {}


def _mp_init_worker(swapi_response, efficiency_table, mag_l1d_data):
    _mp_shared["swapi_response"] = swapi_response
    _mp_shared["efficiency_table"] = efficiency_table
    _mp_shared["mag_l1d_data"] = mag_l1d_data


def _derive_proton_velocity_angles(
    fitting_result: ProtonSolarWindMoments,
    dsrf_to_rtn,
) -> tuple:
    """Return (speed, clock_angle, deflection_angle) as ufloats from proton moments."""
    R = dsrf_to_rtn.T
    u = R @ (fitting_result.bulk_velocity_rtn)

    speed = float(np.linalg.norm(u))
    speed2 = speed**2
    vxy2 = float(u[0] ** 2 + u[1] ** 2)
    vxy = float(np.sqrt(vxy2))

    cov_DPS = R @ fitting_result.velocity_covariance @ R.T

    g_speed = u / speed
    speed_sigma = float(np.sqrt(g_speed @ cov_DPS @ g_speed))

    if vxy2 > 0:
        g_clock = np.array([-u[1] / vxy2, u[0] / vxy2, 0.0])
        clock_sigma = float(np.degrees(np.sqrt(g_clock @ cov_DPS @ g_clock)))
        g_defl = np.array(
            [u[0] * u[2] / (speed2 * vxy), u[1] * u[2] / (speed2 * vxy), -vxy / speed2]
        )
        defl_sigma = float(np.degrees(np.sqrt(g_defl @ cov_DPS @ g_defl)))
    else:
        clock_sigma = np.nan
        defl_sigma = np.nan

    return (
        ufloat(speed, speed_sigma),
        ufloat(np.degrees(np.arctan2(u[1], u[0])) % 360, clock_sigma),
        ufloat(np.degrees(np.arccos(u[2] / speed)), defl_sigma),
    )


def _compute_b_hat_rtn(
    mag_l1d_data,
    chunk_epoch_center_tt2000_ns: int,
    chunk_epoch_delta_ns: int,
    dsrf_to_rtn,
) -> np.ndarray:
    """Return the unit MAG vector at the chunk center, expressed in RTN. Returns NaN
    when MAG data is missing, the rebin produces non-finite values, or |B| is too small
    to define a direction. The alpha fitter interprets a NaN return as MAG_GAP."""
    if mag_l1d_data is None:
        return np.full(3, np.nan)
    b_dsrf = mag_l1d_data.rebin_to(
        np.array([chunk_epoch_center_tt2000_ns]),
        np.array([chunk_epoch_delta_ns]),
    )[0]
    if not np.all(np.isfinite(b_dsrf)) or np.linalg.norm(b_dsrf) < 1e-12:
        return np.full(3, np.nan)
    R = dsrf_to_rtn
    b_rtn = R @ b_dsrf
    return b_rtn / np.linalg.norm(b_rtn)


def _measurement_times_for_chunk(data_chunk, bin_slice):
    passband_indices = np.arange(bin_slice.start, bin_slice.stop)
    return (
        data_chunk.sci_start_time[:, np.newaxis]
        + passband_indices * (12 / 72 * ONE_SECOND_IN_NANOSECONDS)
    ).flatten()


def _nan_proton_result(epoch):
    return (
        epoch,
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        SwapiL3Flags.NONE,
        np.full(3, np.nan),
        np.full(3, np.nan),
    )


def _nan_pui_proton_result():
    return (
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        ufloat(np.nan, np.nan),
        SwapiL3Flags.NONE,
    )


def _nan_alpha_result(epoch):
    mom = AlphaSolarWindMoments(
        density=np.nan,
        temperature=np.nan,
        bulk_velocity_rtn=np.full(3, np.nan),
        delta_v=np.nan,
        bad_fit_flag=int(SwapiL3Flags.HI_CHI_SQ),
    )
    mom._b_hat_rtn = np.full(3, np.nan)
    mom._ref_proton_density = np.nan
    mom._ref_proton_temperature = np.nan
    mom._ref_proton_velocity_rtn = np.full(3, np.nan)
    return epoch, mom


def _fit_proton_moments(
    data_chunk,
    swapi_response,
    efficiency_table,
    epoch,
    bin_slice,
    rotation_matrices,
) -> ProtonSolarWindMoments:
    count_rates = data_chunk.coincidence_count_rate[:, bin_slice].flatten()
    voltages = data_chunk.energy[:, bin_slice].flatten() / SWAPI_L2_K_FACTOR
    proton_eff_scale = float(efficiency_table.get_proton_efficiency_for(epoch)) / float(
        efficiency_table.eps_p_lab
    )
    return fit_solar_wind_proton_moments(
        count_rates,
        voltages,
        swapi_response,
        central_effective_area_scale=proton_eff_scale,
        rotation_matrices=rotation_matrices,
    )


def _proton_chunk_worker(
    data_chunk, epoch, rotation_matrices, dsrf_to_rtn, sc_velocity_rtn
):
    swapi_response = _mp_shared["swapi_response"]
    efficiency_table = _mp_shared["efficiency_table"]

    speed = ufloat(np.nan, np.nan)
    clock_angle = ufloat(np.nan, np.nan)
    deflection_angle = ufloat(np.nan, np.nan)
    density = ufloat(np.nan, np.nan)
    temperature = ufloat(np.nan, np.nan)
    bulk_velocity_rtn_sun = np.full(3, np.nan)
    bulk_velocity_rtn_sc = np.full(3, np.nan)
    quality_flag = SwapiL3Flags.NONE
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")
        fitting_result = _fit_proton_moments(
            data_chunk,
            swapi_response,
            efficiency_table,
            epoch,
            SWAPI_SCIENCE_BINS,
            rotation_matrices=rotation_matrices,
        )
        quality_flag |= fitting_result.bad_fit_flag
        speed, clock_angle, deflection_angle = _derive_proton_velocity_angles(
            fitting_result, dsrf_to_rtn
        )
        bulk_velocity_rtn_sc = fitting_result.bulk_velocity_rtn
        bulk_velocity_rtn_sun = fitting_result.bulk_velocity_rtn + sc_velocity_rtn
        density = ufloat(fitting_result.density, fitting_result.density_sigma)
        temperature = ufloat(
            fitting_result.temperature, fitting_result.temperature_sigma
        )
    except Exception:
        logger.info(
            f"Exception occurred at epoch {epoch}, continuing with fill value",
            exc_info=True,
        )
    return (
        epoch,
        speed,
        clock_angle,
        deflection_angle,
        density,
        temperature,
        quality_flag,
        bulk_velocity_rtn_sun,
        bulk_velocity_rtn_sc,
    )


def _pui_proton_chunk_worker(data_chunk, epoch, rotation_matrices, dsrf_to_rtn):
    swapi_response = _mp_shared["swapi_response"]
    efficiency_table = _mp_shared["efficiency_table"]

    speed = ufloat(np.nan, np.nan)
    clock_angle = ufloat(np.nan, np.nan)
    deflection_angle = ufloat(np.nan, np.nan)
    quality_flag = SwapiL3Flags.NONE
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")
        fitting_result = _fit_proton_moments(
            data_chunk,
            swapi_response,
            efficiency_table,
            epoch,
            SWAPI_SCIENCE_BINS,
            rotation_matrices=rotation_matrices,
        )
        quality_flag |= fitting_result.bad_fit_flag
        speed, clock_angle, deflection_angle = _derive_proton_velocity_angles(
            fitting_result, dsrf_to_rtn
        )
    except Exception:
        logger.info(
            f"Exception occurred at epoch {epoch}, continuing with fill value",
            exc_info=True,
        )
    return speed, clock_angle, deflection_angle, quality_flag


def _alpha_chunk_worker(
    data_chunk, epoch_center_of_chunk, rotation_matrices, b_hat_rtn
):
    swapi_response = _mp_shared["swapi_response"]
    efficiency_table = _mp_shared["efficiency_table"]

    nan_b_hat = np.full(3, np.nan)
    nan_ref_v = np.full(3, np.nan)

    def _annotate(mom, b_hat, n_ref, T_ref, v_ref):
        mom._b_hat_rtn = b_hat
        mom._ref_proton_density = n_ref
        mom._ref_proton_temperature = T_ref
        mom._ref_proton_velocity_rtn = v_ref
        return mom

    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")

        eps_p_lab = float(efficiency_table.eps_p_lab)
        proton_eff_scale = (
            float(efficiency_table.get_proton_efficiency_for(epoch_center_of_chunk))
            / eps_p_lab
        )
        alpha_eff_scale = (
            float(efficiency_table.get_alpha_efficiency_for(epoch_center_of_chunk))
            / eps_p_lab
        )

        proton_moments = _fit_proton_moments(
            data_chunk,
            swapi_response,
            efficiency_table,
            epoch_center_of_chunk,
            SWAPI_COARSE_SWEEP_BINS,
            rotation_matrices=rotation_matrices,
        )

        count_rates = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
        voltages = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
        measurement_times = _measurement_times_for_chunk(
            data_chunk, SWAPI_COARSE_SWEEP_BINS
        )

        mom = fit_solar_wind_alpha_moments(
            count_rates.flatten(),
            voltages.flatten(),
            measurement_times,
            swapi_response,
            proton_moments,
            b_hat_rtn,
            alpha_eff_scale,
            proton_eff_scale,
            rotation_matrices=rotation_matrices,
        )
        return epoch_center_of_chunk, _annotate(
            mom,
            b_hat_rtn,
            float(proton_moments.density),
            float(proton_moments.temperature),
            np.asarray(proton_moments.bulk_velocity_rtn, dtype=float),
        )
    except Exception:
        logger.info(
            f"Alpha moments fit exception at epoch {epoch_center_of_chunk}; using NaN fill",
            exc_info=True,
        )
        mom = AlphaSolarWindMoments(
            density=np.nan,
            temperature=np.nan,
            bulk_velocity_rtn=np.full(3, np.nan),
            delta_v=np.nan,
            bad_fit_flag=int(SwapiL3Flags.HI_CHI_SQ),
        )
        return epoch_center_of_chunk, _annotate(
            mom, nan_b_hat, np.nan, np.nan, nan_ref_v
        )


class SwapiProcessor(Processor):
    def __init__(
        self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata
    ):
        super().__init__(dependencies, input_metadata)

    @staticmethod
    def _fit_proton_moments_for_chunk(
        data_chunk,
        dependencies,
        epoch,
        bin_slice,
        rotation_matrices=None,
    ) -> ProtonSolarWindMoments:
        """Prepare arrays from *bin_slice* and fit proton solar wind moments.

        ``rotation_matrices``  may be passed when
        pre-computed by the caller (e.g. for the alpha stage-2 fit) to avoid a
        second SPICE call; if omitted ``fit_solar_wind_proton_moments`` resolves
        them internally.
        """
        return _fit_proton_moments(
            data_chunk,
            dependencies.swapi_response,
            dependencies.efficiency_calibration_table,
            epoch,
            bin_slice,
            rotation_matrices=rotation_matrices,
        )

    @staticmethod
    def _fit_alpha_moments_for_chunk(
        data_chunk,
        dependencies,
        epoch_center_of_chunk,
        rotation_matrices=None,
        b_hat_rtn=None,
    ) -> AlphaSolarWindMoments:
        """Run Stage 1 (proton fit on coarse-bin axis) and Stage 2 (alpha fit) for a
        5-sweep chunk, returning an `AlphaSolarWindMoments` augmented with three reference
        proton fields (`_ref_proton_*`) and the `_b_hat_rtn` actually used. NaN-filled
        when MAG is absent, the proton fit fails, or alpha peak-finding fails."""
        nan_b_hat = np.full(3, np.nan)
        nan_ref_v = np.full(3, np.nan)

        def _annotate(mom: AlphaSolarWindMoments, b_hat, n_ref, T_ref, v_ref):
            mom._b_hat_rtn = b_hat
            mom._ref_proton_density = n_ref
            mom._ref_proton_temperature = T_ref
            mom._ref_proton_velocity_rtn = v_ref
            return mom

        try:
            if np.any(
                np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))
            ):
                raise ValueError("Fill values in input data")

            if rotation_matrices is None:
                measurement_times = _measurement_times_for_chunk(
                    data_chunk, SWAPI_COARSE_SWEEP_BINS
                )
                rotation_matrices = get_swapi_geometry(measurement_times)

            eps_p_lab = float(dependencies.efficiency_calibration_table.eps_p_lab)
            proton_eff_scale = (
                float(
                    dependencies.efficiency_calibration_table.get_proton_efficiency_for(
                        epoch_center_of_chunk
                    )
                )
                / eps_p_lab
            )
            alpha_eff_scale = (
                float(
                    dependencies.efficiency_calibration_table.get_alpha_efficiency_for(
                        epoch_center_of_chunk
                    )
                )
                / eps_p_lab
            )

            proton_moments = SwapiProcessor._fit_proton_moments_for_chunk(
                data_chunk,
                dependencies,
                epoch_center_of_chunk,
                SWAPI_COARSE_SWEEP_BINS,
                rotation_matrices=rotation_matrices,
            )

            if b_hat_rtn is None:
                b_hat_rtn = _compute_b_hat_rtn(
                    getattr(dependencies, "mag_l1d_data", None),
                    int(epoch_center_of_chunk),
                    int(THIRTY_SECONDS_IN_NANOSECONDS),
                )

            count_rates = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
            voltages = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
            measurement_times = _measurement_times_for_chunk(
                data_chunk, SWAPI_COARSE_SWEEP_BINS
            )

            mom = fit_solar_wind_alpha_moments(
                count_rates.flatten(),
                voltages.flatten(),
                measurement_times,
                dependencies.swapi_response,
                proton_moments,
                b_hat_rtn,
                alpha_eff_scale,
                proton_eff_scale,
                rotation_matrices=rotation_matrices,
            )
            return _annotate(
                mom,
                b_hat_rtn,
                float(proton_moments.density),
                float(proton_moments.temperature),
                np.asarray(proton_moments.bulk_velocity_rtn, dtype=float),
            )
        except Exception:
            logger.info(
                f"Alpha moments fit exception at epoch {epoch_center_of_chunk}; using NaN fill",
                exc_info=True,
            )
            mom = AlphaSolarWindMoments(
                density=np.nan,
                temperature=np.nan,
                bulk_velocity_rtn=np.full(3, np.nan),
                delta_v=np.nan,
                bad_fit_flag=int(SwapiL3Flags.HI_CHI_SQ),
            )
            return _annotate(mom, nan_b_hat, np.nan, np.nan, nan_ref_v)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = SwapiL3ADependencies.fetch_dependencies(
                self.dependencies
            )

            if self.input_metadata.descriptor == "proton-sw":
                data = self.process_l3a_proton(l3a_dependencies.data, l3a_dependencies)
            elif self.input_metadata.descriptor == "alpha-sw":
                data = self.process_l3a_alpha_solar_wind(
                    l3a_dependencies.data, l3a_dependencies
                )
            elif self.input_metadata.descriptor == "pui-he":
                data = self.process_l3a_pui(l3a_dependencies.data, l3a_dependencies)
            else:
                raise NotImplementedError(
                    "unknown descriptor", self.input_metadata.descriptor
                )
            data.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(data)
            return [cdf_path]
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(
                self.dependencies
            )
            l3b_combined_vdf = self.process_l3b(l3b_dependencies.data, l3b_dependencies)
            l3b_combined_vdf.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(l3b_combined_vdf)
            return [cdf_path]

    def process_l3a_pui(
        self, data, dependencies: SwapiL3ADependencies
    ) -> SwapiL3PickupIonData:
        proton_solar_wind_speeds = []
        proton_solar_wind_clock_angles = []
        proton_solar_wind_deflection_angles = []
        proton_quality_flags = []

        chunks = list(chunk_l2_data(data, 5))

        # Precompute SPICE geometry for all chunks before entering multiprocessing.
        precomputed = []
        for data_chunk in chunks:
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                measurement_times = _measurement_times_for_chunk(
                    data_chunk, SWAPI_SCIENCE_BINS
                )
                rm = get_swapi_geometry(measurement_times)
                dsrf = get_swapi_dsrf_to_rtn(np.array([epoch]))[0]
                precomputed.append((epoch, rm, dsrf))
            except Exception:
                logger.warning(f"SPICE gap at epoch {epoch}, NaN-filling chunk")
                precomputed.append((epoch, None, None))

        submittable = [
            (i, chunks[i], precomputed[i])
            for i in range(len(chunks))
            if precomputed[i][1] is not None
        ]

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_mp_init_worker,
            initargs=(
                dependencies.swapi_response,
                dependencies.efficiency_calibration_table,
                None,
            ),
        ) as executor:
            future_to_idx = {
                executor.submit(_pui_proton_chunk_worker, chunk, epoch, rm, dsrf): i
                for i, chunk, (epoch, rm, dsrf) in submittable
            }
            pool_results = {idx: fut.result() for fut, idx in future_to_idx.items()}

        pui_proton_results = []
        for i in range(len(chunks)):
            if i in pool_results:
                pui_proton_results.append(pool_results[i])
            else:
                pui_proton_results.append(_nan_pui_proton_result())

        for speed, clock_angle, deflection_angle, quality_flag in pui_proton_results:
            proton_solar_wind_speeds.append(speed)
            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)
            proton_quality_flags.append(quality_flag)

        ten_minute_solar_wind_velocities, proton_sw_quality_flags = (
            calculate_ten_minute_velocities(
                nominal_values(proton_solar_wind_speeds),
                nominal_values(proton_solar_wind_deflection_angles),
                nominal_values(proton_solar_wind_clock_angles),
                proton_quality_flags,
            )
        )
        pui_epochs = []
        pui_cooling_index = []
        pui_ionization_rate = []
        pui_cutoff_speed = []
        pui_background_rate = []
        pui_density = []
        pui_temperature = []
        bad_fit_flags = []

        for data_chunk, sw_velocity in zip(
            chunk_l2_data(data, 50), ten_minute_solar_wind_velocities
        ):
            epoch = data_chunk.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
            cooling_index = ufloat(np.nan, np.nan)
            ionization_rate = ufloat(np.nan, np.nan)
            cutoff_speed = ufloat(np.nan, np.nan)
            background_count_rate = ufloat(np.nan, np.nan)
            density = ufloat(np.nan, np.nan)
            temperature = ufloat(np.nan, np.nan)
            bad_fit_flag = SwapiL3Flags.NONE
            try:
                if np.any(
                    np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))
                ) or np.any(np.isnan(sw_velocity)):
                    raise ValueError("Fill values in input data")
                fit_params = calculate_pickup_ion_values(
                    dependencies.swapi_response,
                    data_chunk.energy,
                    data_chunk.coincidence_count_rate,
                    epoch,
                    sw_velocity,
                    dependencies.density_of_neutral_helium_calibration_table,
                    dependencies.efficiency_calibration_table,
                    dependencies.hydrogen_inflow_vector,
                    dependencies.helium_inflow_vector,
                )
                cooling_index = fit_params.cooling_index
                ionization_rate = fit_params.ionization_rate
                cutoff_speed = fit_params.cutoff_speed
                background_count_rate = fit_params.background_count_rate
                bad_fit_flag |= fit_params.flags
                density = calculate_helium_pui_density(
                    epoch,
                    sw_velocity,
                    dependencies.density_of_neutral_helium_calibration_table,
                    fit_params,
                    dependencies.helium_inflow_vector,
                )
                temperature = calculate_helium_pui_temperature(
                    epoch,
                    sw_velocity,
                    dependencies.density_of_neutral_helium_calibration_table,
                    fit_params,
                    dependencies.helium_inflow_vector,
                )
            except Exception:
                logger.info(
                    f"Exception occurred at epoch {epoch}, continuing with fill value",
                    exc_info=True,
                )
            pui_epochs.append(epoch)
            pui_cooling_index.append(cooling_index)
            pui_ionization_rate.append(ionization_rate)
            pui_cutoff_speed.append(cutoff_speed)
            pui_background_rate.append(background_count_rate)
            pui_density.append(density)
            pui_temperature.append(temperature)
            bad_fit_flags.append(bad_fit_flag)
        pui_metadata = replace(self.input_metadata, descriptor="pui-he")
        pui_data = SwapiL3PickupIonData(
            pui_metadata,
            np.array(pui_epochs),
            np.array(pui_cooling_index),
            np.array(pui_ionization_rate),
            np.array(pui_cutoff_speed),
            np.array(pui_background_rate),
            np.array(pui_density),
            np.array(pui_temperature),
            np.bitwise_or(proton_sw_quality_flags, bad_fit_flags),
        )

        return pui_data

    def process_l3a_alpha_solar_wind(
        self, data, dependencies
    ) -> SwapiL3AlphaSolarWindData:
        epochs = []

        alpha_solar_wind_speeds = []
        alpha_solar_wind_densities = []
        alpha_solar_wind_temperatures = []
        alpha_solar_wind_bad_fit_flags = []
        alpha_solar_wind_pre_lut_densities = []
        alpha_solar_wind_pre_lut_temperatures = []

        moments_density = []
        moments_density_uncert = []
        moments_temperature = []
        moments_temperature_uncert = []
        moments_velocity_rtn = []
        moments_velocity_covariance_rtn = []
        moments_delta_v = []
        moments_delta_v_uncert = []
        moments_b_hat_rtn = []
        moments_ref_proton_density = []
        moments_ref_proton_temperature = []
        moments_ref_proton_velocity_rtn = []
        moments_bad_fit_flag = []

        chunks = list(chunk_l2_data(data, 5))

        # Precompute SPICE geometry for all chunks before entering multiprocessing.
        precomputed = []
        for data_chunk in chunks:
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                measurement_times = _measurement_times_for_chunk(
                    data_chunk, SWAPI_COARSE_SWEEP_BINS
                )
                rm = get_swapi_geometry(measurement_times)
                dsrf = get_swapi_dsrf_to_rtn(np.array([epoch]))[0]
                b_hat = _compute_b_hat_rtn(
                    getattr(dependencies, "mag_l1d_data", None),
                    int(epoch),
                    int(THIRTY_SECONDS_IN_NANOSECONDS),
                    dsrf_to_rtn=dsrf,
                )
                precomputed.append((epoch, rm, b_hat))
            except Exception:
                logger.warning(f"SPICE gap at epoch {epoch}, NaN-filling chunk")
                precomputed.append((epoch, None, None))

        submittable = [
            (i, chunks[i], precomputed[i])
            for i in range(len(chunks))
            if precomputed[i][1] is not None
        ]

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_mp_init_worker,
            initargs=(
                dependencies.swapi_response,
                dependencies.efficiency_calibration_table,
                getattr(dependencies, "mag_l1d_data", None),
            ),
        ) as executor:
            future_to_idx = {
                executor.submit(_alpha_chunk_worker, chunk, epoch, rm, b_hat): i
                for i, chunk, (epoch, rm, b_hat) in submittable
            }
            pool_results = {idx: fut.result() for fut, idx in future_to_idx.items()}

        alpha_results = []
        for i in range(len(chunks)):
            if i in pool_results:
                alpha_results.append(pool_results[i])
            else:
                alpha_results.append(_nan_alpha_result(precomputed[i][0]))

        for epoch, mom in alpha_results:
            alpha_solar_wind_speed = ufloat(np.nan, np.nan)
            alpha_density = ufloat(np.nan, np.nan)
            alpha_temperature = ufloat(np.nan, np.nan)
            alpha_pre_lut_density = ufloat(np.nan, np.nan)
            alpha_pre_lut_temperature = ufloat(np.nan, np.nan)
            bad_fit_flag = SwapiL3Flags.NONE
            epochs.append(epoch)
            alpha_solar_wind_speeds.append(alpha_solar_wind_speed)
            alpha_solar_wind_densities.append(alpha_density)
            alpha_solar_wind_temperatures.append(alpha_temperature)
            alpha_solar_wind_bad_fit_flags.append(bad_fit_flag)
            alpha_solar_wind_pre_lut_densities.append(alpha_pre_lut_density)
            alpha_solar_wind_pre_lut_temperatures.append(alpha_pre_lut_temperature)

            moments_density.append(mom.density)
            moments_density_uncert.append(mom.density_sigma)
            moments_temperature.append(mom.temperature)
            moments_temperature_uncert.append(mom.temperature_sigma)
            moments_velocity_rtn.append(mom.bulk_velocity_rtn)
            moments_velocity_covariance_rtn.append(mom.velocity_covariance_rtn)
            moments_delta_v.append(mom.delta_v)
            moments_delta_v_uncert.append(mom.delta_v_sigma)
            moments_b_hat_rtn.append(mom._b_hat_rtn)
            moments_ref_proton_density.append(mom._ref_proton_density)
            moments_ref_proton_temperature.append(mom._ref_proton_temperature)
            moments_ref_proton_velocity_rtn.append(mom._ref_proton_velocity_rtn)
            moments_bad_fit_flag.append(int(mom.bad_fit_flag))

        alpha_solar_wind_speed_metadata = replace(
            self.input_metadata, descriptor="alpha-sw"
        )
        alpha_solar_wind_l3_data = SwapiL3AlphaSolarWindData(
            alpha_solar_wind_speed_metadata,
            np.array(epochs),
            np.array(alpha_solar_wind_speeds),
            np.array(alpha_solar_wind_temperatures),
            np.array(alpha_solar_wind_densities),
            np.array(alpha_solar_wind_bad_fit_flags),
            np.array(alpha_solar_wind_pre_lut_temperatures),
            np.array(alpha_solar_wind_pre_lut_densities),
            alpha_sw_moments_density=np.array(moments_density),
            alpha_sw_moments_density_uncert=np.array(moments_density_uncert),
            alpha_sw_moments_temperature=np.array(moments_temperature),
            alpha_sw_moments_temperature_uncert=np.array(moments_temperature_uncert),
            alpha_sw_moments_velocity_rtn=np.array(moments_velocity_rtn),
            alpha_sw_moments_velocity_covariance_rtn=np.array(
                moments_velocity_covariance_rtn
            ),
            alpha_sw_moments_delta_v=np.array(moments_delta_v),
            alpha_sw_moments_delta_v_uncert=np.array(moments_delta_v_uncert),
            alpha_sw_moments_b_hat_rtn=np.array(moments_b_hat_rtn),
            alpha_sw_moments_reference_proton_density=np.array(
                moments_ref_proton_density
            ),
            alpha_sw_moments_reference_proton_temperature=np.array(
                moments_ref_proton_temperature
            ),
            alpha_sw_moments_reference_proton_velocity_rtn=np.array(
                moments_ref_proton_velocity_rtn
            ),
            alpha_sw_moments_bad_fit_flag=np.array(moments_bad_fit_flag),
        )
        return alpha_solar_wind_l3_data

    def process_l3a_proton(self, data, dependencies) -> SwapiL3ProtonSolarWindData:
        chunks = list(chunk_l2_data(data, 5))

        # Precompute SPICE geometry for all chunks before entering multiprocessing.
        # Chunks where SPICE fails (CK gaps) get None and are NaN-filled immediately.
        precomputed = []
        for data_chunk in chunks:
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                measurement_times = _measurement_times_for_chunk(
                    data_chunk, SWAPI_SCIENCE_BINS
                )
                rm = get_swapi_geometry(measurement_times)
                dsrf = get_swapi_dsrf_to_rtn(np.array([epoch]))[0]
                sc_vel = get_spacecraft_velocity_rtn(epoch)
                precomputed.append((epoch, rm, dsrf, sc_vel))
            except Exception:
                logger.warning(f"SPICE gap at epoch {epoch}, NaN-filling chunk")
                precomputed.append((epoch, None, None, None))

        results = []
        submittable = [
            (i, chunks[i], precomputed[i])
            for i in range(len(chunks))
            if precomputed[i][1] is not None
        ]

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_mp_init_worker,
            initargs=(
                dependencies.swapi_response,
                dependencies.efficiency_calibration_table,
                None,
            ),
        ) as executor:
            future_to_idx = {
                executor.submit(_proton_chunk_worker, chunk, epoch, rm, dsrf, sc_vel): i
                for i, chunk, (epoch, rm, dsrf, sc_vel) in submittable
            }
            pool_results = {idx: fut.result() for fut, idx in future_to_idx.items()}

        for i in range(len(chunks)):
            if i in pool_results:
                results.append(pool_results[i])
            else:
                results.append(_nan_proton_result(precomputed[i][0]))

        epochs = [r[0] for r in results]
        proton_solar_wind_speeds = [r[1] for r in results]
        proton_solar_wind_clock_angles = [r[2] for r in results]
        proton_solar_wind_deflection_angles = [r[3] for r in results]
        proton_solar_wind_density = [r[4] for r in results]
        proton_solar_wind_temperatures = [r[5] for r in results]
        quality_flags = [r[6] for r in results]
        bulk_velocities_rtn_sun = [r[7] for r in results]
        bulk_velocities_rtn_sc = [r[8] for r in results]

        proton_solar_wind_speed_metadata = replace(
            self.input_metadata, descriptor="proton-sw"
        )
        proton_solar_wind_l3_data = SwapiL3ProtonSolarWindData(
            proton_solar_wind_speed_metadata,
            np.array(epochs),
            np.array(proton_solar_wind_speeds),
            np.array(proton_solar_wind_temperatures),
            np.array(proton_solar_wind_density),
            np.array(proton_solar_wind_clock_angles),
            np.array(proton_solar_wind_deflection_angles),
            np.array(quality_flags),
            np.array(bulk_velocities_rtn_sun),
            np.array(bulk_velocities_rtn_sc),
        )

        return proton_solar_wind_l3_data

    def process_l3b(self, data, dependencies):
        epochs = []
        cdf_proton_velocities = []
        cdf_proton_probabilities = []
        cdf_alpha_velocities = []
        cdf_alpha_probabilities = []
        cdf_pui_velocities = []
        cdf_pui_probabilities = []
        combined_differential_fluxes = []
        combined_energies = []
        cdf_proton_deltas = []
        cdf_alpha_deltas = []
        cdf_pui_deltas = []
        combined_energy_deltas = []

        for data_chunk in chunk_l2_data(data, 50):
            center_of_epoch = data_chunk.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
            eps_p_lab = float(dependencies.efficiency_calibration_table.eps_p_lab)
            proton_eff_correction = (
                float(
                    dependencies.efficiency_calibration_table.get_proton_efficiency_for(
                        center_of_epoch
                    )
                )
                / eps_p_lab
            )
            alpha_eff_correction = (
                float(
                    dependencies.efficiency_calibration_table.get_alpha_efficiency_for(
                        center_of_epoch
                    )
                )
                / eps_p_lab
            )
            coincidence_count_rates_with_uncertainty = uarray(
                data_chunk.coincidence_count_rate,
                data_chunk.coincidence_count_rate_uncertainty,
            )
            average_coincident_count_rates, energies = calculate_combined_sweeps(
                coincidence_count_rates_with_uncertainty, data_chunk.energy
            )
            proton_velocities, proton_probabilities = calculate_proton_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                proton_eff_correction,
                dependencies.swapi_response,
            )
            alpha_velocities, alpha_probabilities = calculate_alpha_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                alpha_eff_correction,
                dependencies.swapi_response,
            )
            pui_velocities, pui_probabilities = calculate_pui_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                alpha_eff_correction,
                dependencies.swapi_response,
            )
            combined_differential_flux = (
                calculate_combined_solar_wind_differential_flux(
                    energies,
                    average_coincident_count_rates,
                    proton_eff_correction,
                    dependencies.swapi_response,
                )
            )
            epochs.append(center_of_epoch)
            cdf_proton_velocities.append(proton_velocities)
            cdf_proton_probabilities.append(proton_probabilities)
            cdf_proton_deltas.append(calculate_delta_minus_plus(proton_velocities))

            cdf_alpha_velocities.append(alpha_velocities)
            cdf_alpha_probabilities.append(alpha_probabilities)
            cdf_alpha_deltas.append(calculate_delta_minus_plus(alpha_velocities))

            cdf_pui_velocities.append(pui_velocities)
            cdf_pui_probabilities.append(pui_probabilities)
            cdf_pui_deltas.append(calculate_delta_minus_plus(pui_velocities))

            combined_differential_fluxes.append(combined_differential_flux)
            combined_energies.append(energies)
            combined_energy_deltas.append(calculate_delta_minus_plus(energies))

        l3b_combined_metadata = self.input_metadata
        l3b_combined_metadata.descriptor = "combined"
        l3b_combined_vdf = SwapiL3BCombinedVDF(
            input_metadata=l3b_combined_metadata,
            epoch=np.array(epochs),
            proton_sw_velocities=np.array(cdf_proton_velocities),
            proton_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_proton_deltas]
            ),
            proton_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_proton_deltas]
            ),
            proton_sw_combined_vdf=np.array(cdf_proton_probabilities),
            alpha_sw_velocities=np.array(cdf_alpha_velocities),
            alpha_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_alpha_deltas]
            ),
            alpha_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_alpha_deltas]
            ),
            alpha_sw_combined_vdf=np.array(cdf_alpha_probabilities),
            pui_sw_velocities=np.array(cdf_pui_velocities),
            pui_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_pui_deltas]
            ),
            pui_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_pui_deltas]
            ),
            pui_sw_combined_vdf=np.array(cdf_pui_probabilities),
            combined_energy=np.array(combined_energies),
            combined_energy_delta_minus=np.array(
                [delta.delta_minus for delta in combined_energy_deltas]
            ),
            combined_energy_delta_plus=np.array(
                [delta.delta_plus for delta in combined_energy_deltas]
            ),
            combined_differential_flux=np.array(combined_differential_fluxes),
        )
        return l3b_combined_vdf
