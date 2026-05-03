import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.geometric_factor import (
    calculate_geometric_factor,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse


def calculate_combined_solar_wind_differential_flux(
    energies: ndarray,
    average_count_rates: ndarray,
    eff_correction: float,
    swapi_response: SWAPIResponse,
    charge_per_proton_charge: float = 1.0,
):
    esa_voltages = np.asarray(energies, dtype=float) / (
        SWAPI_K_FACTOR * charge_per_proton_charge
    )
    geometric_factors = np.array(
        [
            calculate_geometric_factor(
                swapi_response, float(v), charge_per_proton_charge
            )
            for v in esa_voltages
        ]
    )

    denominator = geometric_factors * eff_correction
    return average_count_rates / denominator
