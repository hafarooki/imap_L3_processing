import numpy as np
import uncertainties
from numpy.typing import ArrayLike
from uncertainties import umath, unumpy, wrap
from uncertainties.unumpy import nominal_values

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)

# Revised SWAPI ESA k-factor from high-resolution SIMION simulations (Q ≈ 1.89 eV/V at θ = 0).
# Used internally by L3 for passband normalization and central-speed conversions.
SWAPI_K_FACTOR = 1.89

# Pre-launch k-factor used by the L2 product to label its `esa_energy` field as
# `esa_energy = SWAPI_L2_K_FACTOR × |voltage|`. Different from SWAPI_K_FACTOR — divide L2's
# `esa_energy` by this to recover true ESA voltage before any L3 processing.
SWAPI_L2_K_FACTOR = 1.93

# SWAPI ESA sweep bin layout (72 bins total, indices 0–71):
#   Index 0       : always discarded (hardware artifact, never science data)
#   Indices 1–62  : coarse sweep passbands (62 bins, uniform energy steps)
#   Indices 63–71 : fine sweep passbands (9 bins, higher resolution near the proton peak)
SWAPI_DISCARDED_BIN = 0
SWAPI_COARSE_SWEEP_BINS = slice(1, 63)  # indices 1–62
SWAPI_FINE_SWEEP_BINS = slice(63, 72)  # indices 63–71
SWAPI_SCIENCE_BINS = slice(1, 72)  # indices 1–71, all usable bins (coarse + fine)


def esa_voltage_to_proton_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * PROTON_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def esa_voltage_to_alpha_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * ALPHA_PARTICLE_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / ALPHA_PARTICLE_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def find_peak_center_of_mass_index(
    peak_slice, count_rates, minimum_count_rate=0, minimum_bin_count=0
):
    count_rates = np.asarray(count_rates)
    indices = np.arange(len(count_rates))
    peak_indices = indices[peak_slice]
    peak_counts = count_rates[peak_slice]
    at_least_minimum = nominal_values(peak_counts) >= minimum_count_rate

    filtered_peak_indices = peak_indices[at_least_minimum]

    if len(filtered_peak_indices) < minimum_bin_count:
        raise Exception("Too few bins after removing low count rates")

    filtered_peak_counts = peak_counts[at_least_minimum]
    center_of_mass_index = np.sum(
        filtered_peak_indices * filtered_peak_counts
    ) / np.sum(filtered_peak_counts)
    return center_of_mass_index


def interpolate_energy(center_of_mass_index, energies):
    interpolate_lambda = lambda x: np.exp(
        np.interp(x, np.arange(len(energies)), np.log(energies))
    )
    interpolate = wrap(interpolate_lambda)
    return interpolate(center_of_mass_index)


def times_for_sweep(start_time):
    time_step_per_bin = 12 / 72
    return start_time + time_step_per_bin * np.arange(72)


def extract_coarse_sweep(data: np.ndarray):
    if data.ndim > 1:
        return data[:, 1:63]
    else:
        return data[1:63]


def calculate_combined_sweeps(coincidence_count_rates, energies):
    energies = np.mean(extract_coarse_sweep(energies), axis=0)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)
    average_coin_rates = np.sum(coincidence_count_rates, axis=0) / len(
        coincidence_count_rates
    )
    return average_coin_rates, energies


def get_alpha_peak_indices(residuals, energies, proton_peak_index) -> slice:
    energies = np.asarray(energies)
    assert np.all(energies[:-1] >= energies[1:]), "Energies must be decreasing"

    min_energy = 1.5 * energies[proton_peak_index]
    max_energy = 4.0 * energies[proton_peak_index]

    def find_start_of_alpha_particle_peak():
        start_bin = None
        for i in reversed(range(proton_peak_index)):
            if energies[i] >= min_energy:
                start_bin = i
                break
        if start_bin is None:
            return None
        for i in reversed(range(start_bin + 1)):
            if residuals[i] > residuals[i + 1] and residuals[i - 1] > residuals[i]:
                return i
        return None

    start_of_alpha_peak = find_start_of_alpha_particle_peak()
    if start_of_alpha_peak is None:
        raise Exception("Alpha peak not found")

    end_of_alpha_peak = np.searchsorted(-energies, -max_energy)
    return slice(int(end_of_alpha_peak), start_of_alpha_peak + 1)


def calculate_sw_speed(particle_mass, particle_charge, energy):
    """Energy-per-charge → speed for an ion of given mass/charge. Handles scalars,
    arrays, and uncertainties.UFloat values."""
    if np.size(energy) == 0:
        return np.array([])
    dimensions = np.asanyarray(energy).ndim
    if dimensions > 0:
        if isinstance(np.ravel(energy)[0], uncertainties.UFloat):
            return (
                unumpy.sqrt(2 * energy * particle_charge / particle_mass)
                / METERS_PER_KILOMETER
            )
        return (
            np.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER
        )
    else:
        return (
            umath.sqrt(2 * energy * particle_charge / particle_mass)
            / METERS_PER_KILOMETER
        )


def calculate_sw_speed_h_plus(energy):
    return calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, energy)
