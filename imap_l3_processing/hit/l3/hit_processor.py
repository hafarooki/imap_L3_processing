import logging

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import UNSIGNED_INT1_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE
from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_l3_processing.hit.l3.models import HitDirectEventDataProduct
from imap_l3_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies
from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAEventReader, RawPHAEvent
from imap_l3_processing.hit.l3.pha.science.calculate_pha import process_pha_event
from imap_l3_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, transform_to_10_minute_chunks
from imap_l3_processing.hit.quality_flags import HitL3Flags
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase, \
    rotate_particle_vectors_from_hit_despun_to_imap_despun, rebin_by_pitch_angle_and_gyrophase
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data

logger = logging.getLogger(__name__)


class HitProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.descriptor == "macropixel":
            dependencies = HITL3SectoredDependencies.fetch_dependencies(self.dependencies)
            pitch_angle_data_product = self.process_pitch_angle_product(dependencies)
            pitch_angle_data_product.parent_file_names = self.get_parent_file_names()
            cdf_file_path = save_data(pitch_angle_data_product)
            return [cdf_file_path]
        elif self.input_metadata.descriptor == "direct-events":
            direct_event_dependencies = HitL3PhaDependencies.fetch_dependencies(self.dependencies)
            direct_event_data_product = self.process_direct_event_product(direct_event_dependencies)
            direct_event_data_product.parent_file_names = self.get_parent_file_names()
            cdf_file_path = save_data(direct_event_data_product)
            return [cdf_file_path]
        else:
            raise ValueError(
                f"Don't know how to generate '{self.input_metadata.descriptor}' /n Known HIT l3 data products: 'macropixel', 'direct-events'.")

    def process_direct_event_product(self,
                                     direct_event_dependencies: HitL3PhaDependencies) -> HitDirectEventDataProduct:

        epochs = []
        raw_pha_events: list[RawPHAEvent] = []
        for epoch, event_binary in zip(direct_event_dependencies.hit_l1_data.epoch,
                                       direct_event_dependencies.hit_l1_data.event_binary):
            try:
                event_raw_pha_events = PHAEventReader.read_all_pha_events(event_binary)
            except Exception as e:
                logger.warning(f"Could not read all pha events for epoch {epoch}", exc_info=True)
                continue

            epochs += [epoch] * len(event_raw_pha_events)
            raw_pha_events += event_raw_pha_events

        charge = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        energy = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        e_delta = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        e_prime = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        detected_range = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        particle_id = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        priority_buffer_number = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        latency = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        stim_tag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        long_event_flag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        haz_tag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        a_b_side = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        has_unread_adcs = np.full(shape=(len(raw_pha_events)), fill_value=False)
        culling_flag = np.full(shape=(len(raw_pha_events)), fill_value=False)

        pha_value = np.full(shape=(len(raw_pha_events), 64), fill_value=UNSIGNED_INT2_FILL_VALUE)
        energy_at_detector = np.full(shape=(len(raw_pha_events), 64), fill_value=np.nan)
        is_low_gain = np.full(shape=(len(raw_pha_events), 64), fill_value=False)

        detector_flags = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        deindex = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        epindex = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        stim_gain = np.full(shape=(len(raw_pha_events)), fill_value=False)
        a_l_stim = np.full(shape=(len(raw_pha_events)), fill_value=False)
        stim_step = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        dac_value = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)

        for i, raw_event in enumerate(raw_pha_events):
            event_output = process_pha_event(
                raw_event,
                direct_event_dependencies.cosine_correction_lookup,
                direct_event_dependencies.gain_lookup,
                direct_event_dependencies.range_fit_lookup,
                direct_event_dependencies.event_type_lookup
            )

            charge[i] = event_output.charge
            energy[i] = event_output.total_energy

            if event_output.e_delta is not None:
                e_delta[i] = event_output.e_delta
            if event_output.e_prime is not None:
                e_prime[i] = event_output.e_prime
            if event_output.detected_range is not None:
                detected_range[i] = event_output.detected_range.range.value
                a_b_side[i] = event_output.detected_range.side.value

            particle_id[i] = raw_event.particle_id
            priority_buffer_number[i] = raw_event.priority_buffer_num
            latency[i] = raw_event.time_tag
            stim_tag[i] = raw_event.stim_tag
            long_event_flag[i] = raw_event.long_event_flag
            haz_tag[i] = raw_event.haz_tag
            has_unread_adcs[i] = raw_event.has_unread_adcs
            culling_flag[i] = raw_event.culling_flag

            for event_energy_at_detector, word in zip(event_output.energies, raw_event.pha_words):
                pha_value[i, word.detector.address] = word.adc_value
                energy_at_detector[i, word.detector.address] = event_energy_at_detector
                is_low_gain[i, word.detector.address] = word.is_low_gain

            if raw_event.extended_header is not None:
                detector_flags[i] = raw_event.extended_header.detector_flags
                deindex[i] = raw_event.extended_header.delta_e_index
                epindex[i] = raw_event.extended_header.e_prime_index
            if raw_event.stim_block is not None:
                stim_gain[i] = raw_event.stim_block.stim_gain
                a_l_stim[i] = raw_event.stim_block.a_l_stim
                stim_step[i] = raw_event.stim_block.stim_step
            if raw_event.extended_stim_header is not None:
                dac_value[i] = raw_event.extended_stim_header.dac_value

        return HitDirectEventDataProduct(
            epoch=epochs,
            charge=charge,
            energy=energy,
            e_delta=e_delta,
            e_prime=e_prime,
            detected_range=detected_range,
            particle_id=particle_id,
            priority_buffer_number=priority_buffer_number,
            latency=latency,
            stim_tag=stim_tag,
            long_event_flag=long_event_flag,
            haz_tag=haz_tag,
            a_b_side=a_b_side,
            has_unread_adcs=has_unread_adcs,
            culling_flag=culling_flag,
            pha_value=pha_value,
            energy_at_detector=energy_at_detector,
            is_low_gain=is_low_gain,
            detector_flags=detector_flags,
            deindex=deindex,
            epindex=epindex,
            stim_gain=stim_gain,
            a_l_stim=a_l_stim,
            stim_step=stim_step,
            dac_value=dac_value,
            hit_flags=np.zeros_like(epochs, dtype=np.uint16),
            input_metadata=self.input_metadata,
        )

    def process_pitch_angle_product(self, dependencies: HITL3SectoredDependencies) -> HitPitchAngleDataProduct:
        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15

        mag_data = dependencies.mag_data

        hit_data = transform_to_10_minute_chunks(dependencies.data)

        input_intensity_data_by_species = {
            "hydrogen": (hit_data.h, hit_data.delta_plus_h, hit_data.delta_minus_h),
            "helium4": (hit_data.he4, hit_data.delta_plus_he4, hit_data.delta_minus_he4),
            "cno": (hit_data.cno, hit_data.delta_plus_cno, hit_data.delta_minus_cno),
            "NeMgSi": (hit_data.nemgsi, hit_data.delta_plus_nemgsi, hit_data.delta_minus_nemgsi),
            "iron": (hit_data.fe, hit_data.delta_plus_fe, hit_data.delta_minus_fe)}
        epoch_count = len(hit_data.epoch)
        cno_energy_count = hit_data.cno.shape[1]
        he4_energy_count = hit_data.he4.shape[1]
        h_energy_count = hit_data.h.shape[1]
        fe_energy_count = hit_data.fe.shape[1]
        nemgsi_energy_count = hit_data.nemgsi.shape[1]
        rebinned_pa_gyro_intensity_by_species = {"cno": self._create_nan_array(
            (epoch_count, cno_energy_count, number_of_pitch_angle_bins, number_of_gyrophase_bins)),
            "helium4": self._create_nan_array((epoch_count, he4_energy_count,
                                               number_of_pitch_angle_bins,
                                               number_of_gyrophase_bins)),
            "hydrogen": self._create_nan_array((epoch_count, h_energy_count,
                                                number_of_pitch_angle_bins,
                                                number_of_gyrophase_bins)),
            "iron": self._create_nan_array((epoch_count, fe_energy_count,
                                            number_of_pitch_angle_bins,
                                            number_of_gyrophase_bins)),
            "NeMgSi": self._create_nan_array((epoch_count, nemgsi_energy_count,
                                              number_of_pitch_angle_bins,
                                              number_of_gyrophase_bins))}

        rebinned_pa_only_intensity_by_species = {
            "cno": self._create_nan_array((epoch_count, cno_energy_count, number_of_pitch_angle_bins)),
            "helium4": self._create_nan_array((epoch_count, he4_energy_count, number_of_pitch_angle_bins)),
            "hydrogen": self._create_nan_array((epoch_count, h_energy_count, number_of_pitch_angle_bins)),
            "iron": self._create_nan_array((epoch_count, fe_energy_count, number_of_pitch_angle_bins)),
            "NeMgSi": self._create_nan_array((epoch_count, nemgsi_energy_count, number_of_pitch_angle_bins))}

        sector_unit_vectors = np.transpose(get_sector_unit_vectors(hit_data.zenith, hit_data.azimuth), (1, 0, 2))

        particle_unit_vectors = -sector_unit_vectors
        rotated_particle_unit_vectors = rotate_particle_vectors_from_hit_despun_to_imap_despun(particle_unit_vectors)

        pitch_angles, gyrophases, pitch_angle_deltas, gyrophase_deltas = get_hit_bin_polar_coordinates(
            number_of_pitch_angle_bins, number_of_gyrophase_bins)

        averaged_mag_data = mag_data.rebin_to(hit_data.epoch, hit_data.epoch_delta)
        hit_flags = np.full(len(hit_data.epoch), HitL3Flags.NONE)
        if dependencies.mag_is_preliminary:
            hit_flags |= HitL3Flags.PRELIMINARY_MAG
        measurement_pitch_angle = []
        measurement_gyrophase = []
        for time_index, average_mag_vector in enumerate(averaged_mag_data):
            mag_unit_vector = calculate_unit_vector(average_mag_vector)

            input_bin_pitch_angles = calculate_pitch_angle(rotated_particle_unit_vectors, mag_unit_vector)
            input_bin_gyrophases = calculate_gyrophase(rotated_particle_unit_vectors, mag_unit_vector)

            measurement_pitch_angle.append(input_bin_pitch_angles)
            measurement_gyrophase.append(input_bin_gyrophases)
            for species, intensity in input_intensity_data_by_species.items():
                rebinned_result = rebin_by_pitch_angle_and_gyrophase(intensity[0][time_index], intensity[1][time_index],
                                                                     intensity[2][time_index], input_bin_pitch_angles,
                                                                     input_bin_gyrophases, number_of_pitch_angle_bins,
                                                                     number_of_gyrophase_bins)

                intensity_by_pa_gyro, intensity_delta_plus_by_pa_gyro, intensity_delta_minus_by_pa_gyro = rebinned_result[
                                                                                                          0:3]
                intensity_by_pa, intensity_delta_plus_by_pa, intensity_delta_minus_by_pa = rebinned_result[3:6]

                rebinned_pa_gyro_intensity_by_species[species][0][time_index, ...] = intensity_by_pa_gyro
                rebinned_pa_gyro_intensity_by_species[species][1][time_index, ...] = intensity_delta_plus_by_pa_gyro
                rebinned_pa_gyro_intensity_by_species[species][2][time_index, ...] = intensity_delta_minus_by_pa_gyro
                rebinned_pa_only_intensity_by_species[species][0][time_index, ...] = intensity_by_pa
                rebinned_pa_only_intensity_by_species[species][1][time_index, ...] = intensity_delta_plus_by_pa
                rebinned_pa_only_intensity_by_species[species][2][time_index, ...] = intensity_delta_minus_by_pa

        return HitPitchAngleDataProduct(self.input_metadata, hit_data.epoch,
                                        hit_data.epoch_delta, pitch_angles, pitch_angle_deltas,
                                        gyrophases,
                                        gyrophase_deltas,
                                        rebinned_pa_gyro_intensity_by_species["hydrogen"][0],
                                        rebinned_pa_gyro_intensity_by_species["hydrogen"][1],
                                        rebinned_pa_gyro_intensity_by_species["hydrogen"][2],
                                        rebinned_pa_only_intensity_by_species["hydrogen"][0],
                                        rebinned_pa_only_intensity_by_species["hydrogen"][1],
                                        rebinned_pa_only_intensity_by_species["hydrogen"][2],
                                        hit_data.h_energy,
                                        hit_data.h_energy_delta_plus,
                                        hit_data.h_energy_delta_minus,
                                        rebinned_pa_gyro_intensity_by_species["helium4"][0],
                                        rebinned_pa_gyro_intensity_by_species["helium4"][1],
                                        rebinned_pa_gyro_intensity_by_species["helium4"][2],
                                        rebinned_pa_only_intensity_by_species["helium4"][0],
                                        rebinned_pa_only_intensity_by_species["helium4"][1],
                                        rebinned_pa_only_intensity_by_species["helium4"][2],
                                        hit_data.he4_energy,
                                        hit_data.he4_energy_delta_plus,
                                        hit_data.he4_energy_delta_minus,
                                        rebinned_pa_gyro_intensity_by_species["cno"][0],
                                        rebinned_pa_gyro_intensity_by_species["cno"][1],
                                        rebinned_pa_gyro_intensity_by_species["cno"][2],
                                        rebinned_pa_only_intensity_by_species["cno"][0],
                                        rebinned_pa_only_intensity_by_species["cno"][1],
                                        rebinned_pa_only_intensity_by_species["cno"][2],
                                        hit_data.cno_energy,
                                        hit_data.cno_energy_delta_plus,
                                        hit_data.cno_energy_delta_minus,
                                        rebinned_pa_gyro_intensity_by_species["NeMgSi"][0],
                                        rebinned_pa_gyro_intensity_by_species["NeMgSi"][1],
                                        rebinned_pa_gyro_intensity_by_species["NeMgSi"][2],
                                        rebinned_pa_only_intensity_by_species["NeMgSi"][0],
                                        rebinned_pa_only_intensity_by_species["NeMgSi"][1],
                                        rebinned_pa_only_intensity_by_species["NeMgSi"][2],
                                        hit_data.nemgsi_energy,
                                        hit_data.nemgsi_energy_delta_plus,
                                        hit_data.nemgsi_energy_delta_minus,
                                        rebinned_pa_gyro_intensity_by_species["iron"][0],
                                        rebinned_pa_gyro_intensity_by_species["iron"][1],
                                        rebinned_pa_gyro_intensity_by_species["iron"][2],
                                        rebinned_pa_only_intensity_by_species["iron"][0],
                                        rebinned_pa_only_intensity_by_species["iron"][1],
                                        rebinned_pa_only_intensity_by_species["iron"][2],
                                        hit_data.fe_energy,
                                        hit_data.fe_energy_delta_plus,
                                        hit_data.fe_energy_delta_minus,
                                        np.array(measurement_pitch_angle),
                                        np.array(measurement_gyrophase),
                                        azimuth=hit_data.azimuth,
                                        zenith=hit_data.zenith,
                                        hit_flags=hit_flags)

    @staticmethod
    def _create_nan_array(shape) -> tuple[np.array, np.array, np.array]:
        return np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
