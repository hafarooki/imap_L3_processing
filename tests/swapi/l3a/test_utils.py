import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.models import MagL1dData
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import (
    chunk_l2_data,
    compute_b_hat_rtn,
    read_l1d_mag_data,
    read_l2_swapi_data,
)


class TestUtils(TestCase):
    def tearDown(self) -> None:
        if os.path.exists('temp_cdf.cdf'):
            os.remove('temp_cdf.cdf')

    def test_chunk_l2_data(self):
        epoch = np.array([0, 1, 2, 3])
        energy = np.array([[15000, 16000, 17000, 18000, 19000],
                           [25000, 26000, 27000, 28000, 29000],
                           [35000, 36000, 37000, 38000, 39000],
                           [45000, 46000, 47000, 48000, 49000], ]
                          )
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        data = SwapiL2Data(epoch, energy, coincidence_count_rate, coincidence_count_rate_uncertainty)
        chunks = list(chunk_l2_data(data, 2))

        expected_energy_chunk_1 = np.array([[15000, 16000, 17000, 18000, 19000],
                                            [25000, 26000, 27000, 28000, 29000]])
        expected_energy_chunk_2 = np.array([[35000, 36000, 37000, 38000, 39000],
                                            [45000, 46000, 47000, 48000, 49000]])

        expected_count_rate_chunk_1 = np.array([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        expected_count_rate_uncertainty_chunk_1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        first_chunk = chunks[0]

        np.testing.assert_array_equal(first_chunk.sci_start_time, np.array([0, 1]))
        np.testing.assert_array_equal(expected_energy_chunk_1, first_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_1, first_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_1,
                                      first_chunk.coincidence_count_rate_uncertainty)

        expected_count_rate_chunk_2 = np.array([[14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        expected_count_rate_uncertainty_chunk_2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        second_chunk = chunks[1]
        np.testing.assert_array_equal(np.array([2, 3]), second_chunk.sci_start_time)
        np.testing.assert_array_equal(expected_energy_chunk_2, second_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_2, second_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_2,
                                      second_chunk.coincidence_count_rate_uncertainty)

    def test_reading_l2_data_into_model(self):
        path = Path('temp_cdf.cdf')
        if path.exists():
            os.remove(path)

        temp_cdf = CDF('temp_cdf', '')
        temp_cdf["sci_start_time"] = np.array(['2010-01-01T00:00:46.000'])
        temp_cdf["esa_energy"] = np.array([1, -1e31, 3, 4], dtype=float)
        temp_cdf["swp_coin_rate"] = np.array([5, 6, 7, -1e31], dtype=float)
        temp_cdf["swp_coin_rate_stat_uncert_plus"] = np.array([2, 2, -1e31, 2, 2, 2, 2, 2], dtype=float)

        temp_cdf["sci_start_time"].attrs["FILLVAL"] = '0'
        temp_cdf["esa_energy"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate_stat_uncert_plus"].attrs["FILLVAL"] = -1e31

        temp_cdf.close()

        actual_swapi_l2_data = read_l2_swapi_data(CDF("temp_cdf.cdf"))

        epoch_as_tt2000 = 315576112184000000
        np.testing.assert_array_equal(np.array(epoch_as_tt2000), actual_swapi_l2_data.sci_start_time)
        np.testing.assert_array_equal(np.array([1, np.nan, 3, 4]), actual_swapi_l2_data.energy)
        np.testing.assert_array_equal(np.array([5, 6, 7, np.nan]), actual_swapi_l2_data.coincidence_count_rate)
        np.testing.assert_array_equal(np.array([2, 2, np.nan, 2, 2, 2, 2, 2]),
                                      actual_swapi_l2_data.coincidence_count_rate_uncertainty)

    def test_reading_l1d_mag_data_reads_b_rtn(self):
        path = Path('temp_cdf.cdf')
        if path.exists():
            os.remove(path)

        temp_cdf = CDF('temp_cdf', '')
        temp_cdf["epoch"] = [datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1)]
        temp_cdf["b_rtn"] = np.array([[1.0, 2.0, 3.0], [2.0e5, -2.0e5, 4.0]])
        temp_cdf["b_rtn"].attrs["FILLVAL"] = -1e31
        temp_cdf["b_rtn"].attrs["VALIDMIN"] = -1.0e5
        temp_cdf["b_rtn"].attrs["VALIDMAX"] = 1.0e5
        temp_cdf.close()

        actual = read_l1d_mag_data(path)

        np.testing.assert_allclose(
            actual.mag_data,
            np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, 4.0]]),
        )

    def test_compute_b_hat_rtn_averages_rtn_samples_directly(self):
        mag_data = MagL1dData(
            epoch=np.array([0, 6, 10, 20]),
            mag_data=np.array(
                [
                    [100.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 100.0],
                ]
            ),
        )

        actual = compute_b_hat_rtn(mag_data, 10, 5)

        np.testing.assert_allclose(actual, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
