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
    derive_velocity_angles,
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
    compute_b_hat_rtn,
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
    return (epoch, mom, np.full(3, np.nan), np.nan, np.nan, np.full(3, np.nan))


def _efficiency_scales(efficiency_table, epoch):
    eps_p_lab = float(efficiency_table.eps_p_lab)
    return (
        float(efficiency_table.get_proton_efficiency_for(epoch)) / eps_p_lab,
        float(efficiency_table.get_alpha_efficiency_for(epoch)) / eps_p_lab,
    )


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
    proton_eff_scale, _ = _efficiency_scales(efficiency_table, epoch)
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
        speed, clock_angle, deflection_angle = derive_velocity_angles(
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
        speed, clock_angle, deflection_angle = derive_velocity_angles(
            fitting_result, dsrf_to_rtn
        )
    except Exception:
        logger.info(
            f"Exception occurred at epoch {epoch}, continuing with fill value",
            exc_info=True,
        )
    return speed, clock_angle, deflection_angle, quality_flag


def _fit_alpha_chunk(
    data_chunk,
    swapi_response,
    efficiency_table,
    epoch_center_of_chunk,
    rotation_matrices,
    b_hat_rtn,
):
    try:
        if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
            raise ValueError("Fill values in input data")

        proton_eff_scale, alpha_eff_scale = _efficiency_scales(
            efficiency_table, epoch_center_of_chunk
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
        return (
            epoch_center_of_chunk,
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
        return _nan_alpha_result(epoch_center_of_chunk)


def _alpha_chunk_worker(
    data_chunk, epoch_center_of_chunk, rotation_matrices, b_hat_rtn
):
    return _fit_alpha_chunk(
        data_chunk,
        _mp_shared["swapi_response"],
        _mp_shared["efficiency_table"],
        epoch_center_of_chunk,
        rotation_matrices,
        b_hat_rtn,
    )


class SwapiProcessor(Processor):
    def __init__(
        self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata
    ):
        super().__init__(dependencies, input_metadata)

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
                    dependencies.instrument_response_calibration_table,
                    dependencies.geometric_factor_calibration_table,
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
        chunks = list(chunk_l2_data(data, 5))

        precomputed = []
        for data_chunk in chunks:
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                measurement_times = _measurement_times_for_chunk(
                    data_chunk, SWAPI_COARSE_SWEEP_BINS
                )
                rm = get_swapi_geometry(measurement_times)
                dsrf = get_swapi_dsrf_to_rtn(np.array([epoch]))[0]
                b_hat = compute_b_hat_rtn(
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

        epochs, densities, density_uncerts = [], [], []
        temperatures, temperature_uncerts = [], []
        velocity_rtns, velocity_covariance_rtns = [], []
        delta_vs, delta_v_uncerts = [], []
        b_hat_rtns = []
        ref_proton_densities, ref_proton_temperatures, ref_proton_velocity_rtns = (
            [],
            [],
            [],
        )
        bad_fit_flags = []

        for epoch, mom, b_hat, ref_n, ref_T, ref_v in alpha_results:
            epochs.append(epoch)
            densities.append(mom.density)
            density_uncerts.append(mom.density_sigma)
            temperatures.append(mom.temperature)
            temperature_uncerts.append(mom.temperature_sigma)
            velocity_rtns.append(mom.bulk_velocity_rtn)
            velocity_covariance_rtns.append(mom.velocity_covariance_rtn)
            delta_vs.append(mom.delta_v)
            delta_v_uncerts.append(mom.delta_v_sigma)
            b_hat_rtns.append(b_hat)
            ref_proton_densities.append(ref_n)
            ref_proton_temperatures.append(ref_T)
            ref_proton_velocity_rtns.append(ref_v)
            bad_fit_flags.append(int(mom.bad_fit_flag))

        return SwapiL3AlphaSolarWindData(
            replace(self.input_metadata, descriptor="alpha-sw"),
            np.array(epochs),
            np.array(densities),
            np.array(density_uncerts),
            np.array(temperatures),
            np.array(temperature_uncerts),
            np.array(velocity_rtns),
            np.array(velocity_covariance_rtns),
            np.array(delta_vs),
            np.array(delta_v_uncerts),
            np.array(b_hat_rtns),
            np.array(ref_proton_densities),
            np.array(ref_proton_temperatures),
            np.array(ref_proton_velocity_rtns),
            np.array(bad_fit_flags),
        )

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
            instrument_efficiency = (
                dependencies.efficiency_calibration_table.get_proton_efficiency_for(
                    center_of_epoch
                )
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
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            alpha_velocities, alpha_probabilities = calculate_alpha_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            pui_velocities, pui_probabilities = calculate_pui_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            combined_differential_flux = (
                calculate_combined_solar_wind_differential_flux(
                    energies,
                    average_coincident_count_rates,
                    instrument_efficiency,
                    dependencies.geometric_factor_calibration_table,
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
