from collections import namedtuple

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, \
    PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, \
    CodiceLoL3aDirectEventDataProduct, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct, \
    CodiceLoL3ChargeStateDistributionsDataProduct, CodiceLoL3a3dDistributionDataProduct
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_mass, calculate_mass_per_charge, convert_count_rate_to_intensity, \
    rebin_to_counts_by_species_elevation_and_spin_sector, combine_priorities_for_species_and_convert_to_rate, \
    calculate_normalization_factor, lookup_normalization_per_event, rebin_3d_distribution_azimuth_to_elevation
from imap_l3_processing.data_utils import safe_divide
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data

PriorityRate = namedtuple('PriorityRate', ('epoch', 'energy_table', 'priority_count'))


class CodiceLoProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.descriptor == "lo-partial-densities":
            dependencies = CodiceLoL3aPartialDensitiesDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_partial_densities(dependencies)
        elif self.input_metadata.descriptor == "lo-direct-events":
            dependencies = CodiceLoL3aDirectEventsDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_direct_event_data_product(dependencies)
        elif self.input_metadata.descriptor == "lo-sw-ratios":
            dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_ratios(dependencies)
        elif self.input_metadata.descriptor == "lo-sw-charge-state-distributions":
            dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_charge_state_distributions(dependencies)
        elif "3d-distribution" in self.input_metadata.descriptor:
            species = self.input_metadata.descriptor.split('-')[1]
            dependencies = CodiceLoL3a3dDistributionsDependencies.fetch_dependencies(self.dependencies, species)
            data_product = self.process_l3a_3d_distribution_product(dependencies)
        else:
            raise NotImplementedError(
                f"Unknown data level and descriptor for CoDICE: {self.input_metadata.data_level}, {self.input_metadata.descriptor}")

        data_product.parent_file_names = self.get_parent_file_names()
        return [save_data(data_product)]

    def process_l3a_ratios(self, dependencies: CodiceLoL3aRatiosDependencies) -> CodiceLoL3aRatiosDataProduct:
        input_data = dependencies.partial_density_data
        c4 = _average_over_block(input_data.cplus4_partial_density, 3)
        c5 = _average_over_block(input_data.cplus5_partial_density, 3)
        c6 = _average_over_block(input_data.cplus6_partial_density, 3)
        o5 = _average_over_block(input_data.oplus5_partial_density, 3)
        o6 = _average_over_block(input_data.oplus6_partial_density, 3)
        o7 = _average_over_block(input_data.oplus7_partial_density, 3)
        o8 = _average_over_block(input_data.oplus8_partial_density, 3)
        feloq = _average_over_block(input_data.fe_loq_partial_density, 3)
        fehiq = _average_over_block(input_data.fe_hiq_partial_density, 3)
        mg = _average_over_block(input_data.mg_partial_density, 3)

        return CodiceLoL3aRatiosDataProduct(
            input_metadata=self.input_metadata,
            epoch=_average_dates_over_block(input_data.epoch, 3),
            epoch_delta=_sum_over_block(input_data.epoch_delta, 3),
            c_to_o_ratio=safe_divide(c4 + c5 + c6, o5 + o6 + o7 + o8),
            mg_to_o_ratio=safe_divide(mg, o5 + o6 + o7 + o8),
            fe_to_o_ratio=safe_divide(feloq + fehiq, o5 + o6 + o7 + o8),
            c6_to_c5_ratio=safe_divide(c6, c5),
            c6_to_c4_ratio=safe_divide(c6, c4),
            o7_to_o6_ratio=safe_divide(o7, o6),
            felo_to_fehi_ratio=safe_divide(feloq, fehiq),
        )

    def process_l3a_charge_state_distributions(self,
                                               dependencies: CodiceLoL3aRatiosDependencies) -> CodiceLoL3ChargeStateDistributionsDataProduct:

        o5 = dependencies.partial_density_data.oplus5_partial_density
        o6 = dependencies.partial_density_data.oplus6_partial_density
        o7 = dependencies.partial_density_data.oplus7_partial_density
        o8 = dependencies.partial_density_data.oplus8_partial_density
        c4 = dependencies.partial_density_data.cplus4_partial_density
        c5 = dependencies.partial_density_data.cplus5_partial_density
        c6 = dependencies.partial_density_data.cplus6_partial_density

        o_densities = _average_over_block(np.column_stack((o5, o6, o7, o8)), block_size=3)
        o_distribution = safe_divide(o_densities, np.sum(o_densities, axis=1, keepdims=True))

        c_densities = _average_over_block(np.column_stack((c4, c5, c6)), block_size=3)
        c_distribution = safe_divide(c_densities, np.sum(c_densities, axis=1, keepdims=True))

        return CodiceLoL3ChargeStateDistributionsDataProduct(
            self.input_metadata,
            epoch=_average_dates_over_block(dependencies.partial_density_data.epoch, 3),
            epoch_delta=_sum_over_block(dependencies.partial_density_data.epoch_delta, 3),
            oxygen_charge_state_distribution=o_distribution,
            carbon_charge_state_distribution=c_distribution
        )

    def process_l3a_partial_densities(self, dependencies: CodiceLoL3aPartialDensitiesDependencies):
        codice_lo_l2_data = dependencies.codice_l2_lo_data
        mass_per_charge_lookup = dependencies.mass_per_charge_lookup
        h_plus_partial_density = calculate_partial_densities(
            codice_lo_l2_data.hplus,
            codice_lo_l2_data.energy_per_charge,
            mass_per_charge_lookup.hplus,
        )
        heplusplus_partial_density = calculate_partial_densities(codice_lo_l2_data.heplusplus,
                                                                 codice_lo_l2_data.energy_per_charge,
                                                                 mass_per_charge_lookup.heplusplus)
        cplus4_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus4,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.cplus4)
        cplus5_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus5,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.cplus5)
        cplus6_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus6,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.cplus6)
        oplus5_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus5,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.oplus5)
        oplus6_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus6,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.oplus6)
        oplus7_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus7,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.oplus7)
        oplus8_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus8,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.oplus8)
        ne_partial_density = calculate_partial_densities(codice_lo_l2_data.ne, codice_lo_l2_data.energy_per_charge,
                                                         mass_per_charge_lookup.ne)
        mg_partial_density = calculate_partial_densities(codice_lo_l2_data.mg, codice_lo_l2_data.energy_per_charge,
                                                         mass_per_charge_lookup.mg)
        si_partial_density = calculate_partial_densities(codice_lo_l2_data.si, codice_lo_l2_data.energy_per_charge,
                                                         mass_per_charge_lookup.si)
        fe_loq_partial_density = calculate_partial_densities(codice_lo_l2_data.fe_loq,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.fe_loq)
        fe_hiq_partial_density = calculate_partial_densities(codice_lo_l2_data.fe_hiq,
                                                             codice_lo_l2_data.energy_per_charge,
                                                             mass_per_charge_lookup.fe_hiq)

        return CodiceLoL3aPartialDensityDataProduct(
            input_metadata=self.input_metadata,
            data=CodiceLoPartialDensityData(
                epoch=codice_lo_l2_data.epoch,
                epoch_delta=codice_lo_l2_data.epoch_delta_plus,
                hplus_partial_density=h_plus_partial_density,
                heplusplus_partial_density=heplusplus_partial_density,
                cplus4_partial_density=cplus4_partial_density,
                cplus5_partial_density=cplus5_partial_density,
                cplus6_partial_density=cplus6_partial_density,
                oplus5_partial_density=oplus5_partial_density,
                oplus6_partial_density=oplus6_partial_density,
                oplus7_partial_density=oplus7_partial_density,
                oplus8_partial_density=oplus8_partial_density,
                ne_partial_density=ne_partial_density,
                mg_partial_density=mg_partial_density,
                si_partial_density=si_partial_density,
                fe_loq_partial_density=fe_loq_partial_density,
                fe_hiq_partial_density=fe_hiq_partial_density,
            )
        )

    def process_l3a_direct_event_data_product(
            self,
            dependencies: CodiceLoL3aDirectEventsDependencies
    ) -> CodiceLoL3aDirectEventDataProduct:
        codice_sw_priority_counts_l1a_data = dependencies.codice_lo_l1a_sw_priority_rates
        codice_nsw_priority_counts_l1a_data = dependencies.codice_lo_l1a_nsw_priority_rates
        codice_direct_events = dependencies.codice_l2_direct_events
        esa_energy_per_charge_lookup = dependencies.energy_lookup

        mass_coefficient_lookup = dependencies.mass_coefficient_lookup

        spin_angle_lut = SpinAngleLookup()

        mass_per_charge = calculate_mass_per_charge(codice_direct_events.energy_per_charge, codice_direct_events.tof)
        mass = calculate_mass(codice_direct_events.apd_energy, codice_direct_events.tof, mass_coefficient_lookup)

        direct_event_epochs = codice_direct_events.epoch

        sw_priority_counts = np.ma.stack(
            [
                codice_sw_priority_counts_l1a_data.p0_tcrs,
                codice_sw_priority_counts_l1a_data.p1_hplus,
                codice_sw_priority_counts_l1a_data.p2_heplusplus,
                codice_sw_priority_counts_l1a_data.p3_heavies,
                codice_sw_priority_counts_l1a_data.p4_dcrs,
            ],
            axis=1,
        )
        nsw_priority_counts = np.ma.stack(
            [
                codice_nsw_priority_counts_l1a_data.p5_heavies,
                codice_nsw_priority_counts_l1a_data.p6_hplus_heplusplus,
            ],
            axis=1,
        )
        sw_aligned, sw_missing = _align_priority_counts_variable_to_direct_event_epochs(
            sw_priority_counts, codice_sw_priority_counts_l1a_data.epoch, direct_event_epochs
        )
        nsw_aligned, nsw_missing = _align_priority_counts_variable_to_direct_event_epochs(
            nsw_priority_counts, codice_nsw_priority_counts_l1a_data.epoch, direct_event_epochs
        )
        acquisition_time_per_esa_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.acquisition_time_per_esa_step, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        rgfo_half_spin_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.rgfo_half_spin, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        rgfo_spin_sector_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.rgfo_spin_sector, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        rgfo_esa_step_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.rgfo_esa_step, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        half_spin_per_esa_step_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.half_spin_per_esa_step, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        nso_spin_sector_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.nso_spin_sector, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        nso_esa_step_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.nso_esa_step, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        nso_half_spin_aligned, _ = _align_priority_counts_variable_to_direct_event_epochs(
            codice_sw_priority_counts_l1a_data.nso_half_spin, codice_sw_priority_counts_l1a_data.epoch,
            direct_event_epochs
        )
        stacked_priorities = np.concatenate([
            np.ma.filled(sw_aligned, np.nan), np.ma.filled(nsw_aligned, np.nan)], axis=1)

        normalization = calculate_normalization_factor(stacked_priorities, codice_direct_events.num_events,
                                                       codice_direct_events.energy_step,
                                                       codice_direct_events.spin_sector)
        normalization_per_event = lookup_normalization_per_event(
            normalization,
            codice_direct_events.num_events,
            codice_direct_events.energy_step,
            codice_direct_events.spin_sector,
        )

        num_sw_priorities = sw_aligned.shape[1]
        num_nsw_priorities = nsw_aligned.shape[1]
        priority_missing_mask = np.zeros(
            (len(direct_event_epochs), num_sw_priorities + num_nsw_priorities), dtype=bool
        )
        priority_missing_mask[sw_missing, :num_sw_priorities] = True
        priority_missing_mask[nsw_missing, num_sw_priorities:] = True
        normalization_per_event[priority_missing_mask] = 0.0

        return CodiceLoL3aDirectEventDataProduct(
            input_metadata=self.input_metadata,
            epoch=codice_direct_events.epoch,
            epoch_delta=codice_direct_events.epoch_delta_plus,
            acquisition_time_per_esa_step=acquisition_time_per_esa_aligned,
            apd_energy=codice_direct_events.apd_energy,
            apd_id=codice_direct_events.apd_id,
            data_quality=codice_direct_events.data_quality,
            elevation=codice_direct_events.elevation_angle,
            energy_bin=np.flip(esa_energy_per_charge_lookup.bin_centers),
            energy_bin_delta_minus=np.flip(esa_energy_per_charge_lookup.delta_minus),
            energy_bin_delta_plus=np.flip(esa_energy_per_charge_lookup.delta_plus),
            energy_per_charge=codice_direct_events.energy_per_charge,
            energy_step=codice_direct_events.energy_step,
            esa_step=codice_sw_priority_counts_l1a_data.esa_step,
            gain=codice_direct_events.gain,
            half_spin_per_esa_step=half_spin_per_esa_step_aligned,
            multi_flag=codice_direct_events.multi_flag,
            nso_esa_step=nso_esa_step_aligned,
            nso_spin_sector=nso_spin_sector_aligned,
            nso_half_spin=nso_half_spin_aligned,
            num_events=codice_direct_events.num_events,
            position=codice_direct_events.position,
            mass_per_charge=mass_per_charge,
            mass=mass,
            normalization=np.flip(normalization, axis=2),
            normalization_per_event=normalization_per_event,
            rgfo_esa_step=rgfo_esa_step_aligned,
            rgfo_spin_sector=rgfo_spin_sector_aligned,
            rgfo_half_spin=rgfo_half_spin_aligned,
            spin_angle=codice_direct_events.spin_angle,
            spin_angle_bin=spin_angle_lut.bin_centers,
            spin_angle_bin_delta=spin_angle_lut.bin_deltas,
            spin_sector=codice_direct_events.spin_sector,
            tof=codice_direct_events.tof,
            type=codice_direct_events.type,
        )

    def process_l3a_3d_distribution_product(self, dependencies: CodiceLoL3a3dDistributionsDependencies):
        mass_species_bin_lookup = dependencies.mass_species_bin_lookup
        position_elevation_lut = PositionToElevationLookup()
        energy_lut = dependencies.energy_per_charge_lut
        geometric_factor_lut = dependencies.geometric_factors_lookup

        counts_3d_data = rebin_to_counts_by_species_elevation_and_spin_sector(
            direct_event_data=dependencies.l3a_direct_event_data, mass_species_bin_lookup=mass_species_bin_lookup)

        species_index = mass_species_bin_lookup.get_species_index(dependencies.species)
        counts_for_species = counts_3d_data[species_index]
        normalized_count_rates = combine_priorities_for_species_and_convert_to_rate(counts_for_species,
                                                                                    dependencies.l3a_direct_event_data.acquisition_time_per_esa_step)

        geometric_factors = geometric_factor_lut.get_geometric_factors(
            dependencies.l3a_direct_event_data.rgfo_half_spin, dependencies.l3a_direct_event_data.rgfo_spin_sector,
            dependencies.l3a_direct_event_data.rgfo_esa_step, dependencies.l3a_direct_event_data.half_spin_per_esa_step,
            self.input_metadata.start_date.date()
        )
        intensities = convert_count_rate_to_intensity(normalized_count_rates,
                                                      dependencies.energy_per_charge_lut,
                                                      dependencies.efficiency_factors_lut,
                                                      geometric_factors)

        intensity = rebin_3d_distribution_azimuth_to_elevation(intensities, np.arange(1, 25), position_elevation_lut,
                                                               dependencies.l3a_direct_event_data.half_spin_per_esa_step)

        return CodiceLoL3a3dDistributionDataProduct(
            input_metadata=self.input_metadata,
            epoch=dependencies.l3a_direct_event_data.epoch,
            epoch_delta=dependencies.l3a_direct_event_data.epoch_delta,
            elevation=position_elevation_lut.bin_centers,
            elevation_delta=position_elevation_lut.bin_deltas,
            spin_angle=dependencies.l3a_direct_event_data.spin_angle_bin,
            spin_angle_delta=dependencies.l3a_direct_event_data.spin_angle_bin_delta,
            energy=np.flip(energy_lut.bin_centers),
            energy_delta_plus=np.flip(energy_lut.delta_plus),
            energy_delta_minus=np.flip(energy_lut.delta_minus),
            species=dependencies.species,
            species_data=np.flip(intensity, axis=1),
            species_data_stat_uncert=np.flip(np.sqrt(counts_for_species), axis=1),
            rgfo_esa_step=dependencies.l3a_direct_event_data.rgfo_esa_step,
            rgfo_spin_sector=dependencies.l3a_direct_event_data.rgfo_spin_sector,
            rgfo_half_spin=dependencies.l3a_direct_event_data.rgfo_half_spin,
            half_spin_per_esa_step=np.flip(dependencies.l3a_direct_event_data.half_spin_per_esa_step),
        )


def _align_priority_counts_variable_to_direct_event_epochs(
        priority_variable: np.ndarray,
        priority_epochs: np.ndarray,
        direct_event_epochs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_direct_event_epochs = len(direct_event_epochs)
    aligned_shape = (num_direct_event_epochs,) + priority_variable.shape[1:]
    aligned = np.ma.masked_all(aligned_shape, priority_variable.dtype)
    missing_mask = np.ones(num_direct_event_epochs, dtype=bool)
    for i, direct_event_epoch in enumerate(direct_event_epochs):
        matching_indices = np.where(priority_epochs == direct_event_epoch)[0]
        if len(matching_indices) > 0:
            aligned[i] = priority_variable[matching_indices[0]]
            missing_mask[i] = False
    return aligned, missing_mask


def _average_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].mean(axis=0) for i in range(0, len(data_array), block_size)])


def _sum_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].sum() for i in range(0, len(data_array), block_size)])


def _average_dates_over_block(data_array: np.ndarray, block_size: int):
    dates_as_int = data_array.astype('datetime64[us]').astype('int64')
    return np.array([dates_as_int[i:i + block_size].mean() for i in range(0, len(dates_as_int), block_size)]).astype(
        "datetime64[us]").astype('O')
