import numpy as np
from numpy.typing import ArrayLike
from uncertainties import wrap
from uncertainties.unumpy import nominal_values

from imap_l3_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, METERS_PER_KILOMETER

SWAPI_K_FACTOR = 1.89  # eV/V, see docs/swapi/solar-wind-moments.md


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


def get_peak_indices(count_rates, width, mask=True) -> slice:
    max_indices = np.argwhere(
            (count_rates == np.max(count_rates, where=mask, initial=0)) & mask)
    left_min_index = np.min(max_indices)
    right_min_index = np.max(max_indices)

    if right_min_index - left_min_index > 1:
        raise Exception("Count rates contains multiple distinct peaks")

    return slice(max(0, left_min_index - width), right_min_index + width+1)

def find_peak_center_of_mass_index(peak_slice, count_rates, minimum_count_rate=0, minimum_bin_count=0):
    count_rates = np.asarray(count_rates)
    indices = np.arange(len(count_rates))
    peak_indices = indices[peak_slice]
    peak_counts = count_rates[peak_slice]
    at_least_minimum = nominal_values(peak_counts) >= minimum_count_rate

    filtered_peak_indices = peak_indices[at_least_minimum]

    if len(filtered_peak_indices) < minimum_bin_count:
        raise Exception("Too few bins after removing low count rates")

    filtered_peak_counts = peak_counts[at_least_minimum]
    center_of_mass_index = np.sum(filtered_peak_indices * filtered_peak_counts) / np.sum(filtered_peak_counts)
    return center_of_mass_index

def interpolate_energy(center_of_mass_index, energies):
    interpolate_lambda = lambda x: np.exp(np.interp(x, np.arange(len(energies)), np.log(energies)))
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