from unittest.mock import patch, Mock

import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import (
    FittingParameters,
    calculate_pickup_ion_values,
    ModelCountRateCalculator,
)
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from tests.test_helpers import get_test_data_path

density_of_neutral_helium_lut_path = get_test_data_path(
    "swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
)
density_of_neutral_helium_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(
    density_of_neutral_helium_lut_path
)


def compare_fitting_parameters():
    energy = np.array(
        [
            1.0000000e00,
            1.9098358e04,
            1.7541177e04,
            1.6113177e04,
            1.4798380e04,
            1.3591366e04,
            1.2485777e04,
            1.1467618e04,
            1.0532608e04,
            9.6755140e03,
            8.8850460e03,
            8.1653940e03,
            7.5017600e03,
            6.8884770e03,
            6.3279270e03,
            5.8114860e03,
            5.3388680e03,
            4.9013030e03,
            4.5042990e03,
            4.1383830e03,
        ]
    )
    sw_velocity_vector = np.array([0, 0, -500])
    model_count_rate_calculator = ModelCountRateCalculator(
        swapi_response=Mock(),
        solar_wind_vector=sw_velocity_vector,
        density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
        efficiency_table=Mock(),
        helium_inflow_vector=Mock(),
    )
    fit_params_1 = FittingParameters(1.5, 1e-7, 520, 0.1)
    fit_params_2 = FittingParameters(2.5, 1.0e-7, 480, 0.1)

    energy_labels = range(62, 46, -1)
    indices = list(zip(energy_labels, energy[1:17], strict=True))
    modeled_1 = model_count_rate_calculator.model_count_rate(
        indices, fit_params_1, 36789
    )
    modeled_2 = model_count_rate_calculator.model_count_rate(
        indices, fit_params_2, 36789
    )
    plt.errorbar(energy[1:17], modeled_1, yerr=modeled_1 / np.sqrt(modeled_1 * 50 / 6))
    plt.plot(energy[1:17], modeled_2)
    plt.show()


@patch("imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
def calculate_pickup_ions_with_minimize_on_random_synth_data(mock_spice):
    ephemeris_time_for_epoch = 100000
    mock_spice.unitim.return_value = ephemeris_time_for_epoch
    mock_light_time = 122.0
    mock_spice.spkezr.return_value = (np.array([0, 0, 0, 0, 0, 0]), mock_light_time)
    mock_spice.latrec.return_value = np.array([0, 2, 0])
    mock_spice.reclat.return_value = np.array(
        [0.99 * ONE_AU_IN_KM, np.deg2rad(255.7), 0.6]
    )

    def mock_sxform(from_frame, to_frame, et):
        if from_frame == "IMAP_SWAPI":
            return np.eye(6)
        return np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
            ]
        )

    mock_spice.sxform.side_effect = mock_sxform

    data_file_path = get_test_data_path(
        "swapi/imap_swapi_l2_50-sweeps_20100101_v002.cdf"
    )
    with CDF(str(data_file_path)) as cdf:
        energy = cdf["energy"][...]
        count_rate = cdf["swp_coin_rate"][...]

        background_count_rate_cutoff = 0.1
        epoch = 123_456_789_000_000_000
        sw_velocity_vector = np.array([0, 0, -500])
        energy_labels = range(62, 46, -1)
        indices = list(zip(energy_labels, energy[1:17], strict=True))
        model_count_rate_calculator = ModelCountRateCalculator(
            swapi_response=Mock(),
            solar_wind_vector=sw_velocity_vector,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
            efficiency_table=Mock(),
            helium_inflow_vector=Mock(),
        )
        rng = np.random.default_rng()

        # generated_data_params = FittingParameters(
        #     rng.uniform(1, 5),
        #     rng.uniform(0.6e-7, 2.1e-7),
        #     rng.uniform(400, 500),
        #     rng.uniform(0, 0.2),
        # )
        generated_data_params = FittingParameters(
            cooling_index=4.076926044739297,
            ionization_rate=1.7448617996003114e-07,
            cutoff_speed=474.25503847884136,
            background_count_rate=0.0333619285902611,
        )
        modeled_count_rates = model_count_rate_calculator.model_count_rate(
            indices, generated_data_params, 36789
        )
        count_rate[:, 1:17] = modeled_count_rates
        plt.errorbar(
            energy[1:17],
            modeled_count_rates,
            yerr=modeled_count_rates / np.sqrt(modeled_count_rates * 50 / 6),
        )
        actual_fitting_parameters = calculate_pickup_ion_values(
            swapi_response=Mock(),
            energy=energy,
            count_rates=count_rate,
            center_of_epoch=epoch,
            sw_velocity_vector=sw_velocity_vector,
            density_of_neutral_helium_lookup_table=density_of_neutral_helium_lookup_table,
            efficiency_table=Mock(),
            hydrogen_inflow_vector=Mock(),
            helium_inflow_vector=Mock(),
        )

        actual_fit_without_uncert = FittingParameters(
            actual_fitting_parameters.cooling_index.n,
            actual_fitting_parameters.ionization_rate.n,
            actual_fitting_parameters.cutoff_speed.n,
            actual_fitting_parameters.background_count_rate.n,
        )
        fitted_count_rate = model_count_rate_calculator.model_count_rate(
            indices, actual_fit_without_uncert, 36789
        )
        plt.plot(energy[1:17], fitted_count_rate)
        plt.show()

        print("Generated params:", generated_data_params)
        print("Fitted params:", actual_fitting_parameters)


calculate_pickup_ions_with_minimize_on_random_synth_data()
# compare_fitting_parameters()
