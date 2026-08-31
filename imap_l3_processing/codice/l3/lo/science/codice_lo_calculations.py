from __future__ import annotations

from typing import TypeVar

import numpy as np

from imap_l3_processing.codice.l3.lo.constants import AZIMUTH_STEP_SIZE, ELEVATION_STEP_SIZE, ENERGY_STEP_SIZE, \
    ENERGY_LOST_IN_CARBON_FOIL, POST_ACCELERATION_VOLTAGE_IN_KV, CONVERSION_CONSTANT_K, CODICE_LO_NUM_AZIMUTH_BINS, \
    CONSTANT_C_FROM_INSTRUMENT_TEAM, CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_SPIN_SECTORS
from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, CodiceLoDirectEventData


def calculate_partial_densities(intensities: np.ndarray, esa_steps: np.ndarray, mass_per_charge: float):
    return np.nansum(
        CONSTANT_C_FROM_INSTRUMENT_TEAM * AZIMUTH_STEP_SIZE * ELEVATION_STEP_SIZE * ENERGY_STEP_SIZE * intensities * np.sqrt(
            esa_steps[np.newaxis, :, np.newaxis]) * np.sqrt(mass_per_charge), axis=(1, 2))


def calculate_total_number_of_events(priority_rate_variable: np.ndarray, acquisition_time: np.ndarray) -> np.ndarray[
    int]:
    acquisition_time_in_seconds = acquisition_time / 1_000_000
    counts = np.multiply(priority_rate_variable, acquisition_time_in_seconds[np.newaxis, :, np.newaxis])
    return np.sum(counts, axis=(1, 2), dtype=int)


def calculate_mass(apd_energy: np.ndarray, tof: np.ndarray, mass_coefficients: MassCoefficientLookup) -> np.ndarray:
    energy = np.log(apd_energy)
    tof = np.log(tof)

    mass_calculation = mass_coefficients[0] + (mass_coefficients[1] * energy) + (mass_coefficients[2] * tof) + (
            mass_coefficients[3] * energy * tof) + (mass_coefficients[4] * np.power(energy, 2)) + (
                               mass_coefficients[5] * np.power(tof, 3))
    return np.e ** mass_calculation


def calculate_mass_per_charge(energy_per_charge: np.ndarray, tof: np.ndarray) -> np.ndarray:
    return (energy_per_charge + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
            tof ** 2) * CONVERSION_CONSTANT_K


def rebin_direct_events_for_normalization(num_events: np.ndarray, spin_sector: np.ndarray, energy_step: np.ndarray,
                                          num_spin_sectors: int, num_energies: int) -> np.ndarray:
    base_counts = rebin_direct_events_by_energy_and_spin_sector(num_events, spin_sector, energy_step, num_spin_sectors,
                                                                num_energies)
    half_spin = num_spin_sectors // 2
    result = np.zeros_like(base_counts)
    result[:, :, :, 0:half_spin] = base_counts[:, :, :, 0:half_spin] + base_counts[:, :, :, half_spin:num_spin_sectors]
    result[:, :, :, half_spin:num_spin_sectors] = result[:, :, :, 0:half_spin]
    return result


def rebin_direct_events_by_energy_and_spin_sector(num_events: np.ndarray, spin_sector: np.ndarray,
                                                  energy_step: np.ndarray, num_spin_sectors: int, num_energies: int) -> np.ndarray:
    num_epochs = num_events.shape[0]
    num_priorities = num_events.shape[1]

    rebinned_output = np.zeros((num_epochs, num_priorities, num_energies, num_spin_sectors))

    for time_index in range(num_epochs):
        for priority_index in range(num_priorities):
            events_at_index = num_events[time_index, priority_index]
            if events_at_index is np.ma.masked:
                continue

            spin_sectors = spin_sector[time_index, priority_index, :events_at_index]
            energy_indices = energy_step[time_index, priority_index, :events_at_index]

            spin_sector_valid = ~np.ma.getmaskarray(spin_sectors)
            energy_index_valid = ~np.ma.getmaskarray(energy_indices)
            valid_events_mask = np.logical_and(spin_sector_valid, energy_index_valid)

            valid_spin_sectors = spin_sectors[valid_events_mask]
            valid_energy_indices = energy_indices[valid_events_mask]

            np.add.at(rebinned_output[time_index, priority_index], (valid_energy_indices, valid_spin_sectors), 1)
    return rebinned_output


def calculate_normalization_factor(priority_counts: np.ndarray, num_events: np.ndarray, energy_steps: np.ndarray,
                                   spin_sectors: np.ndarray) -> np.ndarray:
    numerator = np.concatenate((priority_counts, priority_counts), axis=3)
    num_energies = priority_counts.shape[2]
    num_spin_sectors = 2 * priority_counts.shape[3]
    denominator = rebin_direct_events_for_normalization(num_events, spin_sectors, energy_steps, num_spin_sectors,
                                                        num_energies)

    division_result = np.zeros(numerator.shape, dtype=float)
    np.divide(numerator, denominator, out=division_result, where=denominator != 0)

    output = np.zeros_like(division_result)
    output[denominator != 0] = np.maximum(1.0, division_result[denominator != 0])
    output[(denominator == 0) & (numerator == 0)] = 0.0
    output[(denominator == 0) & (numerator != 0)] = np.nan
    return output


def lookup_normalization_per_event(normalization: np.ndarray, num_events: np.ndarray, energy_steps: np.ndarray,
                                   spin_sectors: np.ndarray) -> np.ndarray:
    results = np.full(spin_sectors.shape, np.nan)
    for (epoch, priority), count in np.ma.ndenumerate(num_events, compressed=True):
        energy_step = energy_steps[epoch, priority, :count]
        spin_sector = spin_sectors[epoch, priority, :count]

        bad_spin_sectors = np.ma.getmaskarray(spin_sector)
        bad_energy_steps = np.ma.getmaskarray(energy_step)

        good_events = ~(bad_spin_sectors | bad_energy_steps)
        results[epoch, priority, :count][good_events] = normalization[
            epoch, priority, energy_step[good_events], spin_sector[good_events]]

    return results


def rebin_to_counts_by_species_elevation_and_spin_sector(direct_event_data: CodiceLoDirectEventData,
                                                         mass_species_bin_lookup: MassSpeciesBinLookup) -> np.ndarray:
    mass = direct_event_data.mass
    mass_per_charge = direct_event_data.mass_per_charge
    spin_sector = direct_event_data.spin_sector
    apd_id = direct_event_data.apd_id
    energy_step = direct_event_data.energy_step
    num_events = direct_event_data.num_events

    num_epochs = mass.shape[0]
    num_priorities = mass.shape[1]

    output = np.full((mass_species_bin_lookup.get_num_species(), num_epochs, num_priorities,
                      CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_SPIN_SECTORS, CODICE_LO_NUM_AZIMUTH_BINS), 0.0)

    for epoch_i in range(num_epochs):
        for priority_i in range(num_priorities):
            if num_events[epoch_i, priority_i] is np.ma.masked:
                continue

            for event_i in range(num_events[epoch_i, priority_i]):
                indices_of_event = epoch_i, priority_i, event_i
                masked_energy_step = energy_step[indices_of_event] is np.ma.masked
                masked_spin_sector = spin_sector[indices_of_event] is np.ma.masked
                masked_apd_id = apd_id[indices_of_event] is np.ma.masked
                if masked_energy_step or masked_spin_sector or masked_apd_id:
                    continue

                apd_id_of_event = int(apd_id[indices_of_event])
                if apd_id_of_event < 1 or apd_id_of_event > CODICE_LO_NUM_AZIMUTH_BINS:
                    continue

                species = mass_species_bin_lookup.get_species(mass[indices_of_event],
                                                              mass_per_charge[indices_of_event])
                if species is not None:
                    energy_i = energy_step[indices_of_event]
                    species_i = mass_species_bin_lookup.get_species_index(species)
                    spin_sector_i = spin_sector[indices_of_event]
                    apd_id_i = apd_id_of_event - 1
                    output[
                        species_i, epoch_i, priority_i, energy_i, spin_sector_i, apd_id_i] += \
                        direct_event_data.normalization_per_event[indices_of_event]

    return output


EPOCH = TypeVar("EPOCH")
PRIORITY = TypeVar("PRIORITY")
SPECIES = TypeVar("SPECIES")
POSITION = TypeVar("POSITION")
SPIN_ANGLE = TypeVar("SPIN_ANGLE")
ENERGY = TypeVar("ENERGY")


def combine_priorities_for_species_and_convert_to_rate(
        counts: np.ndarray[(EPOCH, PRIORITY, POSITION, SPIN_ANGLE, ENERGY)],
        acquisition_times: np.ndarray[(EPOCH, ENERGY,)]) -> np.ndarray:
    return np.sum(counts, axis=1) / (acquisition_times[:, :, np.newaxis, np.newaxis])


def rebin_3d_distribution_azimuth_to_elevation(intensity_data: np.ndarray,
                                               azimuths: np.ndarray,
                                               position_to_elevation_lut: PositionToElevationLookup,
                                               half_spin) -> np.ndarray:
    num_epochs = intensity_data.shape[0]
    num_elevations = len(position_to_elevation_lut.bin_centers)
    num_spin_angles = intensity_data.shape[2]
    num_energies = intensity_data.shape[1]
    rebinned = np.zeros((num_epochs, num_energies, num_spin_angles, num_elevations))

    elevation_indices = position_to_elevation_lut.apd_to_elevation_index(azimuths)
    for azimuth_index, elevation_index in zip(azimuths, elevation_indices):
        rebinned[:, :, :, elevation_index] += intensity_data[:, :, :, azimuth_index - 1]

    for epoch_i in range(len(half_spin)):
        ids_a = half_spin[epoch_i] % 2 == 0
        ids_b = half_spin[epoch_i] % 2 == 1
        rebinned[epoch_i, ids_a, 12:24, 0] = rebinned[epoch_i, ids_a, 0:12, 0]
        rebinned[epoch_i, ids_b, 0:12, 0] = rebinned[epoch_i, ids_b, 12:24, 0]

        rebinned[epoch_i, ids_a, 0:12, 12] = rebinned[epoch_i, ids_a, 12:24, 12]
        rebinned[epoch_i, ids_b, 12:24, 12] = rebinned[epoch_i, ids_b, 0:12, 12]

    return rebinned


def convert_count_rate_to_intensity(count_rates: np.ndarray,
                                    energy_per_charge: EnergyLookup,
                                    efficiency_lookup: EfficiencyLookup,
                                    geometric_factor: np.ndarray[(EPOCH, ENERGY, SPIN_ANGLE, POSITION)]) -> np.ndarray:
    reshaped_efficiency_data = efficiency_lookup.efficiency_data[np.newaxis, :, np.newaxis, :]
    denominator = geometric_factor * energy_per_charge.bin_centers[np.newaxis, :, np.newaxis,
                                     np.newaxis] * reshaped_efficiency_data
    intensities = count_rates / denominator
    return intensities


def compute_geometric_factors(num_epochs: int, num_energies: int):
    return np.ones((num_epochs, num_energies))
