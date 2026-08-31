import dataclasses
import unittest
from datetime import datetime
from unittest.mock import patch, Mock, sentinel

import numpy as np
from scipy.stats import linregress

from imap_l3_processing.maps.map_models import IntensityMapData, RectangularIntensityMapData, QUALITY_FLAGS_VAR_NAME
from imap_l3_processing.maps.mpfit import mpfit
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.maps.spectral_fit import power_law, fit_arrays_to_power_law, fit_spectral_index_map, \
    calculate_spectral_index_for_multiple_ranges, slice_energy_range_by_bin
from tests.test_helpers import get_test_data_path


class TestSpectralFit(unittest.TestCase):
    def test_power_law_function(self):
        params = (2, -2)
        x = np.array([1, 2, 3])
        y = np.array([4, 10, 22])
        err = np.array([2, 2, 2])
        keywords = {'xval': x, 'yval': y, 'errval': err}

        expected_residual = np.array([1, 1, 2])
        status, actual_residuals = power_law(params, **keywords)

        np.testing.assert_array_equal(actual_residuals, expected_residual)
        self.assertEqual(status, 0)

    def test_finds_best_fit(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result_A, result_A_error, result_gamma, result_gamma_error, result_chi_square = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_equal(result_gamma,
                                              np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_gamma_error,
                                                     np.array([0.060068]).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(result_A, np.array(true_A).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_A_error,
                                                     np.array([0.161515]).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal( result_chi_square ,np.zeros((1,1, *spacial_dimension_shape)))

    def test_finds_chi_squared_for_fit(self):
        energies = np.array([1, 10, 30, 100])
        flux_data = np.array([99, 10, 3, 1])
        errors = np.array([1,1, 0.1, 1e-3])

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result_A, result_A_error, result_gamma, result_gamma_error, result_chi_square = fit_arrays_to_power_law(flux, uncertainty, energies)
                fitted_flux = result_A * np.power(energies.reshape((-1,*spacial_dimension_shape)), -result_gamma)
                residual = fitted_flux - flux
                chisquared = np.sum(np.square(residual/uncertainty))
                reduced_chisquared = chisquared / (len(energies) - 2)
                np.testing.assert_array_almost_equal(result_chi_square, reduced_chisquared)

    def test_handles_small_fluxes(self):
        energies = np.array([5.65826829,  8.45453724, 12.70203527])
        flux_data = np.array([0.12815728, 0.00414464, 0.0383057])
        errors = np.array([0.12532869, 0.00414464, 0.02755597])

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result_a, result_a_error, result_gamma, result_gamma_error, result_chi_square = fit_arrays_to_power_law(flux, uncertainty, energies)
                fitted_flux = result_a * np.power(energies.reshape((-1,*spacial_dimension_shape)), -result_gamma)
                self.assertGreater(result_a, 0)

    def test_spectral_fit_map(self):
        epoch = np.array([datetime.now()])
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)
        latitude = np.array([0])
        longitude = np.array([0])
        full_shape = (len(epoch), len(energies), len(longitude), len(latitude))

        data = IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([]),
            energy=energies,
            energy_delta_plus=np.repeat(0.5, 23),
            energy_delta_minus=np.repeat(0.5, 23),
            energy_label=np.array([]),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.full(full_shape, 1.0),
            obs_date=np.ma.array(
                np.full(full_shape, datetime(year=2010, month=1, day=1))
            ),
            obs_date_range=np.full(full_shape, 100000),
            solid_angle=np.full(full_shape, 1.23),
            ena_intensity=np.array(flux_data).reshape(full_shape),
            ena_intensity_sys_err=np.array([]),
            ena_intensity_stat_uncert=np.array(errors).reshape(full_shape),
            quality_flags=np.full(full_shape, MapL3Flags.NONE),
        )

        spectral_intensity_map = fit_spectral_index_map(data)

        expected_energy_midpoint = np.sqrt(0.5 * 10.5)
        expected_energy_minus = expected_energy_midpoint - 0.5
        expected_energy_plus = 10.5 - expected_energy_midpoint
        expected_energy_label = f"0.5 - 10.5"
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index,
                                      np.array(true_gamma).reshape(1, 1, 1, 1))

        np.testing.assert_array_almost_equal(spectral_intensity_map.ena_spectral_index_stat_uncert,
                                             np.array([0.060068]).reshape(1, 1, 1, 1))

        np.testing.assert_array_almost_equal(spectral_intensity_map.ena_spectral_index_scalar_coefficient_stat_uncert,
                                             np.array([0.161515]).reshape(1, 1, 1, 1))

        np.testing.assert_array_almost_equal(spectral_intensity_map.energy, [expected_energy_midpoint])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_minus, [expected_energy_minus])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_plus, [expected_energy_plus])
        np.testing.assert_equal(spectral_intensity_map.energy_label, [expected_energy_label])

        np.testing.assert_array_almost_equal(spectral_intensity_map.longitude, data.longitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.latitude, data.latitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.solid_angle, data.solid_angle)

    def test_spectral_fit_fields_other_than_fit_fields(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])

        latitude = np.arange(-90, 90, 45)
        longitude = np.arange(0, 360, 45)

        input_shape = (1, 3, len(longitude), len(latitude))
        quality_flags = np.full(input_shape, MapL3Flags.NONE)
        quality_flags[0, 0, 1:3] = MapL3Flags.PREDICTIVE_EPHEMERIS
        quality_flags[0, 2, 4] = MapL3Flags.PERSISTED_LAST_POINT
        quality_flags[0, 1, 3] = MapL3Flags.PREDICTIVE_EPHEMERIS

        quality_flags[0, 2, 0:2] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO

        input_map = IntensityMapData(
            epoch=np.array([datetime.now()]),
            epoch_delta=np.array([1000000]),
            energy=input_energies,
            energy_delta_plus=input_deltas,
            energy_delta_minus=input_deltas,
            energy_label=np.array(["a", "b", "c"]),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.zeros(input_shape),
            obs_date=np.full(input_shape, datetime(2025, 1, 1)),
            obs_date_range=np.full(input_shape, 1),
            solid_angle=np.full((len(longitude), len(latitude)), 0.1),
            ena_intensity=np.full(input_shape, 1),
            ena_intensity_sys_err=np.full(input_shape, 1),
            ena_intensity_stat_uncert=np.full(input_shape, 1),
            quality_flags=quality_flags
        )

        input_map.obs_date[0, 0] = datetime(2025, 1, 1)
        input_map.obs_date[0, 1] = datetime(2025, 1, 1)
        input_map.obs_date[0, 2] = datetime(2027, 1, 1)

        input_map.exposure_factor[0, 0] = 1.0
        input_map.exposure_factor[0, 1] = 2.0
        input_map.exposure_factor[0, 2] = 3.0

        input_map.obs_date_range[0, 0] = 1
        input_map.obs_date_range[0, 1] = 1
        input_map.obs_date_range[0, 2] = 3

        output = fit_spectral_index_map(input_map)

        self.assertEqual(output.energy.shape[0], 1)
        self.assertEqual(output.energy[0], 10)
        np.testing.assert_allclose(output.energy_delta_minus, np.array([9]))
        np.testing.assert_allclose(output.energy_delta_plus, np.array([90]))
        self.assertEqual(output.energy_label.shape, (1,))
        self.assertEqual("1.0 - 100.0", output.energy_label[0])

        expected_ena_shape = np.array([1, 1, len(longitude), len(latitude)])
        np.testing.assert_array_equal(output.obs_date,
                                      np.full(expected_ena_shape, datetime(2026, 1, 1)))
        np.testing.assert_array_equal(output.obs_date_range, np.full(expected_ena_shape, 2))
        np.testing.assert_array_equal(output.exposure_factor, np.full(expected_ena_shape, 6))

        expected_quality_flags = np.full(expected_ena_shape, MapL3Flags.NONE)
        expected_quality_flags[0, 0, 0] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        expected_quality_flags[0, 0, 1] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PREDICTIVE_EPHEMERIS
        expected_quality_flags[0, 0, 2:4] = MapL3Flags.PREDICTIVE_EPHEMERIS
        expected_quality_flags[0, 0, 4] = MapL3Flags.PERSISTED_LAST_POINT

        np.testing.assert_array_equal(output.quality_flags, expected_quality_flags)

    def test_finds_best_fit_with_nan_in_flux(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        flux_data[len(flux_data) // 2] = np.nan
        flux_data[0] = np.nan
        flux_data[-1] = np.nan

        errors = 0.2 * np.abs(flux_data)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_allclose(scalar_coefficients,
                                           np.array(true_A).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(chisq, np.zeros((1, 1, *spacial_dimension_shape)))


    def test_spectral_fit_map_negative_gammas(self):
        epoch = np.array([datetime.now()])
        energies = np.geomspace(1, 10, 10)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)
        latitude = np.array([0])
        longitude = np.array([0])
        full_shape = (len(epoch), len(energies), len(longitude), len(latitude))

        data = IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([]),
            energy=energies,
            energy_delta_plus=np.repeat(0.5, 23),
            energy_delta_minus=np.repeat(0.5, 23),
            energy_label=np.array([]),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.full(full_shape, 1.0),
            obs_date=np.ma.array(
                np.full(full_shape, datetime(year=2010, month=1, day=1))
            ),
            obs_date_range=np.full(full_shape, 100000),
            solid_angle=np.full(full_shape, 1.23),
            ena_intensity=np.array(flux_data).reshape(full_shape),
            ena_intensity_sys_err=np.array([]),
            ena_intensity_stat_uncert=np.array(errors).reshape(full_shape),
            quality_flags=np.full(full_shape, MapL3Flags.NONE),
        )

        spectral_intensity_map = fit_spectral_index_map(data)

        expected_energy_midpoint = np.sqrt(0.5 * 10.5)
        expected_energy_minus = expected_energy_midpoint - 0.5
        expected_energy_plus = 10.5 - expected_energy_midpoint
        expected_energy_label = f"0.5 - 10.5"
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index_stat_uncert,
                                      np.full(shape=(1, 1, 1, 1), fill_value=np.nan))
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index_scalar_coefficient,
                                      np.full(shape=(1, 1, 1, 1), fill_value=np.nan))
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index,
                                      np.full(shape=(1, 1, 1, 1), fill_value=np.nan))
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index_chisq,
                                      np.full(shape=(1, 1, 1, 1), fill_value=np.nan))

        np.testing.assert_array_almost_equal(spectral_intensity_map.energy, [expected_energy_midpoint])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_minus, [expected_energy_minus])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_plus, [expected_energy_plus])
        np.testing.assert_equal(spectral_intensity_map.energy_label, [expected_energy_label])

        np.testing.assert_array_almost_equal(spectral_intensity_map.longitude, data.longitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.latitude, data.latitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.solid_angle, data.solid_angle)

    def test_finds_best_fit_with_nan_in_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)
        errors[len(errors) // 2] = np.nan
        errors[0] = np.nan
        errors[-1] = np.nan

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_allclose(scalar_coefficients,
                                           np.array(true_A).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(chisq, np.zeros((1, 1, *spacial_dimension_shape)))

    def test_finds_best_fit_with_zero_in_flux_and_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0
        errors[0:3] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(scalar_coefficients,
                                              np.array(true_A).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(chisq, np.zeros((1, 1, *spacial_dimension_shape)))

    def test_finds_best_fit_with_ibex_data(self):
        energies = np.array([0.71, 1.11, 1.74, 2.73, 4.29])
        flux_data = np.array([[[[46.710853, 61.45169],
                                [60.682266, 60.523616]],
                               [[34.896639, 11.987692],
                                [29.323209, 19.239248]],
                               [[18.899919, 17.140896],
                                [16.12169, 14.099767]],
                               [[14.709325, 12.165362],
                                [13.114869, 13.726924]],
                               [[5.815468, 5.071331],
                                [6.060111, 6.499908]]]])
        errors = np.array([[[[4.17239236e+02, 5.87577556e+02],
                             [4.59947913e+02, 7.02472212e+02]],
                            [[6.38942760e+01, 3.62103490e+01],
                             [2.87564900e+01, 5.06517210e+01]],
                            [[1.58932960e+01, 1.41284380e+01],
                             [1.34992710e+01, 9.66067900e+00]],
                            [[2.01988700e+00, 2.75456600e+00],
                             [1.83946100e+00, 2.25438200e+00]],
                            [[3.00438000e-01, 3.76740000e-01],
                             [2.15943000e-01, 3.76271000e-01]]]])

        scalar_coefficients, scalar_coefficient_errors, index, index_error, chisq = fit_arrays_to_power_law(flux_data, errors, energies)
        np.testing.assert_array_almost_equal(index, np.array([[[[1.811566, 1.489658],
                                                                 [1.480259, 1.317993]]]]))
        np.testing.assert_array_almost_equal(index_error, np.array([[[[0.279224, 0.409522],
                                                                       [0.260374, 0.318162]]]]))
        np.testing.assert_array_almost_equal(scalar_coefficients, np.array([[[[81.92187 , 44.822599],
                                                                            [52.494651, 44.647085]]]]))
        np.testing.assert_array_almost_equal(scalar_coefficient_errors, np.array([[[[32.245376, 26.075033],
         [19.584559, 20.080747]]]]))
        np.testing.assert_array_almost_equal(chisq, np.array([[[[0.44621 , 0.391638],
         [0.344915, 0.475525]]]]))

    def test_finds_best_fit_with_zeros_in_flux_and_not_uncertainty(self):
        energies = np.geomspace(1, 1e10, 23)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_almost_equal(result,
                                                     np.array([1.472697]).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(scalar_coefficients,
                                                     np.array([1.251819]).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(chisq, np.array([2.218472]).reshape(1, 1, *spacial_dimension_shape))

    def test_returns_nan_when_only_one_point_is_valid(self):
        energies = np.geomspace(1, 1e10, 5)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[1:] = 0
        errors[1:] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)
                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)

                np.testing.assert_array_almost_equal(result,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(scalar_coefficients,
                                              np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(scalar_coefficient_error,
                                              np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(chisq, np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))

    def test_returns_nan_when_there_are_less_than_2_positive_fluxes(self):
        energies = np.geomspace(1, 1e10, 5)

        flux_data = np.full_like(energies, 0)
        errors = np.full_like(flux_data, 2)

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                scalar_coefficients, scalar_coefficient_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_almost_equal(result,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(scalar_coefficients,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(scalar_coefficient_error,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_equal(chisq, np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))

    def test_spectral_fit_can_fit_multiple_energy_ranges(self):
        epoch = np.array([datetime.now()])
        input_energy_range_1 = np.geomspace(1, 100, 11)
        input_energy_range_2 = np.geomspace(101, 10000, 12)
        true_A_range_1, true_gamma_range_1 = 2.0, 1.5
        true_A_range_2, true_gamma_range_2 = 0.1, 3.5
        flux_data_range_1 = true_A_range_1 * np.power(input_energy_range_1, -true_gamma_range_1)
        flux_data_range_2 = true_A_range_2 * np.power(input_energy_range_2, -true_gamma_range_2)
        longitude = np.array([45])
        latitude = np.array([90])
        errors_range_1 = 0.1 * np.abs(flux_data_range_1)
        errors_range_2 = 0.0001 * np.abs(flux_data_range_2)
        energies = np.append(input_energy_range_1, input_energy_range_2)
        full_shape = (len(epoch), len(energies), len(longitude), len(latitude))
        variance = np.concat((errors_range_1, errors_range_2)).reshape(full_shape)
        flux = np.concat((flux_data_range_1, flux_data_range_2)).reshape(full_shape)

        quality_flags_range_1 = np.full((11,), MapL3Flags.NONE)
        quality_flags_range_1[0] = MapL3Flags.PREDICTIVE_EPHEMERIS

        quality_flags_range_2 = np.full((12,), MapL3Flags.NONE)
        quality_flags_range_2[1] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        quality_flags_range_2[2] = MapL3Flags.PREDICTIVE_EPHEMERIS
        quality_flags = np.concat((quality_flags_range_1, quality_flags_range_2)).reshape(full_shape)

        expected_quality_flags = np.array([
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PREDICTIVE_EPHEMERIS
        ]).reshape((1, 2, 1, 1))

        data = IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([10000]),
            energy=energies,
            energy_delta_plus=np.repeat(0.5, 23),
            energy_delta_minus=np.repeat(0.5, 23),
            energy_label=energies.astype(str),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.full(full_shape, 1.0),
            obs_date=np.ma.array(
                np.full(full_shape, datetime(year=2010, month=1, day=1))
            ),
            obs_date_range=np.full(full_shape, 100000),
            solid_angle=np.full(full_shape, 1.23),
            ena_intensity=flux,
            ena_intensity_sys_err=np.zeros_like(flux),
            ena_intensity_stat_uncert=np.array(variance).reshape(full_shape),
            quality_flags=quality_flags,
        )

        output_energies = np.array([[1, 100.5], [100.5, 10000.5]])
        spectral_index_map_data = calculate_spectral_index_for_multiple_ranges(data, output_energies)
        np.testing.assert_array_equal(spectral_index_map_data.ena_spectral_index[0, 0, 0, 0], true_gamma_range_1)
        np.testing.assert_array_equal(spectral_index_map_data.ena_spectral_index[0, 1, 0, 0], true_gamma_range_2)
        np.testing.assert_almost_equal(spectral_index_map_data.ena_spectral_index_scalar_coefficient[0, 0, 0, 0],
                                       true_A_range_1)
        np.testing.assert_almost_equal(spectral_index_map_data.ena_spectral_index_scalar_coefficient[0, 1, 0, 0],
                                       true_A_range_2)
        np.testing.assert_array_almost_equal(spectral_index_map_data.ena_spectral_index_stat_uncert[0, 0, 0, 0],
                                             0.020704)
        np.testing.assert_array_almost_equal(spectral_index_map_data.ena_spectral_index_stat_uncert[0, 1, 0, 0],
                                             2.00179e-05)

        np.testing.assert_almost_equal(spectral_index_map_data.energy,
                                       np.array([7.088723439378913, 1002.5219448969683]))
        np.testing.assert_almost_equal(spectral_index_map_data.energy_delta_plus, np.array([93.4112766, 8997.9780551]))
        np.testing.assert_almost_equal(spectral_index_map_data.energy_delta_minus, np.array([6.5887234, 902.0219449]))
        np.testing.assert_array_equal(spectral_index_map_data.epoch, epoch)
        np.testing.assert_array_equal(spectral_index_map_data.epoch_delta, data.epoch_delta)
        np.testing.assert_array_equal(spectral_index_map_data.latitude, latitude)
        np.testing.assert_array_equal(spectral_index_map_data.longitude, longitude)
        np.testing.assert_array_equal(spectral_index_map_data.solid_angle, data.solid_angle)

        np.testing.assert_array_equal(
            spectral_index_map_data.energy_label, ["0.5 - 100.5", "100.5 - 10000.5"]
        )

        np.testing.assert_array_equal(spectral_index_map_data.exposure_factor, np.reshape([11, 12], (1, 2, 1, 1)))
        np.testing.assert_array_equal(spectral_index_map_data.obs_date,
                                      np.full((1, 2, 1, 1), datetime(year=2010, month=1, day=1)))
        np.testing.assert_array_equal(spectral_index_map_data.obs_date_range, np.full((1, 2, 1, 1), 100000))

        self.assertEqual((1, 2, 1, 1), spectral_index_map_data.quality_flags.shape)
        np.testing.assert_array_equal(spectral_index_map_data.quality_flags, expected_quality_flags)

    @patch('imap_l3_processing.maps.spectral_fit.mpfit')
    def test_spectral_fit_returns_nan_if_fit_status_is_not_positive_or_equal_to_five(self, mock_mpfit):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        invalid_status_cases = [-1, 0, 5]

        for name, spacial_dimension_shape in cases:
            for status_code in invalid_status_cases:
                with self.subTest(f"Pixelization: {name}, status_code: {status_code}"):
                    mock_mpfit.return_value = Mock(status=status_code, params=(0, 0))
                    flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                    uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                    scalar, scalar_error, result, result_error, chisq = fit_arrays_to_power_law(flux, uncertainty, energies)
                    expected_nan_result = np.full((1, 1, *spacial_dimension_shape), np.nan)
                    np.testing.assert_array_equal(result, expected_nan_result)
                    np.testing.assert_array_equal(result_error, expected_nan_result)
                    np.testing.assert_array_equal(scalar, expected_nan_result)
                    np.testing.assert_array_equal(scalar_error, expected_nan_result)
                    np.testing.assert_array_equal(chisq, expected_nan_result)

    @patch('imap_l3_processing.maps.spectral_fit.mpfit', wraps=mpfit)
    @patch('imap_l3_processing.maps.spectral_fit.scipy.stats.linregress', wraps=linregress)
    def test_passes_initial_guess_to_mpfit_based_on_linear_fit_in_log_space(self, mock_linregress, mock_mpfit):
        energies = np.geomspace(1e1, 1e4, 6)
        flux_data = np.array([1e7, 1e5, 1e6, 1e4, 1e5, 1e3])

        errors = 0.2 * np.abs(flux_data)

        flux = np.array(flux_data).reshape(1, len(energies), 1)
        uncertainty = np.array(errors).reshape(1, len(energies), 1)

        mock_linregress.return_value = Mock(slope=3, intercept=5)

        _ = fit_arrays_to_power_law(flux, uncertainty, energies)
        input_energies, input_flux = mock_linregress.call_args[0]
        np.testing.assert_array_equal(input_energies, np.log10(energies))
        np.testing.assert_array_equal(input_flux, np.log10(flux_data))
        self.assertEqual((100000, -3), mock_mpfit.call_args.args[1])

    def test_spectral_fit_against_validation_data(self):
        test_cases = [
            (
                "hi45",
                RectangularIntensityMapData.read_from_path(
                    get_test_data_path("hi/fake_l2_maps/hi45-6months.cdf")
                ).intensity_map_data,
                "hi/validation/spectral_index/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv",
                "hi/validation/spectral_index/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv",
                "hi/validation/spectral_index/menlo-IMAP-Hi45_6months_4.0x4.0_fit_A0.csv",
                "hi/validation/spectral_index/menlo-IMAP-Hi45_6months_4.0x4.0_fit_A0_sig.csv",
                "hi/validation/spectral_index/menlo-IMAP-Hi45_6months_4.0x4.0_fit_chisq.csv",
            )
        ]

        for (
            name,
            input_data,
            expected_gamma_path,
            expected_sigma_path,
            expected_a_path,
            expected_a_sig_path,
            expected_chisq_path,
        ) in test_cases:
            with self.subTest(name):
                expected_gamma = np.loadtxt(
                    get_test_data_path(expected_gamma_path),
                    delimiter=",",
                    dtype=np.float64,
                ).T
                expected_a = np.loadtxt(get_test_data_path(expected_a_path), delimiter=",", dtype=np.float64).T
                expected_gamma_sigma = np.loadtxt(
                    get_test_data_path(expected_sigma_path),
                    delimiter=",",
                    dtype=np.float64,
                ).T
                expected_a_sigma = np.loadtxt(
                    get_test_data_path(expected_a_sig_path),
                    delimiter=",",
                    dtype=np.float64,
                ).T
                expected_chisq = np.loadtxt(get_test_data_path(expected_chisq_path), delimiter=",", dtype=np.float64).T

                output_data = calculate_spectral_index_for_multiple_ranges(input_data, [[0, np.inf]])

                np.testing.assert_allclose(output_data.ena_spectral_index[0, 0],
                                           expected_gamma, atol=1e-3)
                np.testing.assert_allclose(
                    output_data.ena_spectral_index_stat_uncert[0, 0],
                    expected_gamma_sigma,
                    atol=1e-3,
                )
                np.testing.assert_allclose(output_data.ena_spectral_index_scalar_coefficient[0, 0],
                                           expected_a, atol=1e-3)
                np.testing.assert_allclose(output_data.ena_spectral_index_scalar_coefficient_stat_uncert[0, 0],
                                           expected_a_sigma, atol=1e-3)
                np.testing.assert_allclose(output_data.ena_spectral_index_chisq[0, 0], expected_chisq)
                np.testing.assert_allclose(output_data.quality_flags, np.full((1,1,90,45), MapL3Flags.NONE))

    def test_slice_energy_range_by_bin(self):
        def build_array(*values):
            return np.array([
                [
                    [[a]] for a in values
                ]
            ])

        input_data = IntensityMapData(
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=np.array([1, 10, 100, 1000, 10000]),
            energy_delta_plus=np.array([0.3, 3, 30, 300, 3000]),
            energy_delta_minus=np.array([0.2, 2, 20, 200, 2000]),
            energy_label=np.array(["1", "10", "100", "1000", "10000"]),
            latitude=sentinel.latitude,
            longitude=sentinel.longitude,
            exposure_factor=build_array(1, 2, 3, 4, 5),
            obs_date=build_array(datetime(2026, 1, 1),
                                 datetime(2026, 1, 2),
                                 datetime(2026, 1, 3),
                                 datetime(2026, 1, 4),
                                 datetime(2026, 1, 5),
                                 ),
            obs_date_range=build_array(1000, 2000, 3000, 4000, 5000),
            solid_angle=sentinel.solid_angle,
            ena_intensity=build_array(100, 200, 300, 400, 500),
            ena_intensity_stat_uncert=build_array(11, 12, 13, 14, 15),
            ena_intensity_sys_err=build_array(21, 22, 23, 24, 25),
            quality_flags=build_array(
                MapL3Flags.NONE,
                MapL3Flags.NONE,
                MapL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.NONE
            ),
        )

        expected_data = IntensityMapData(
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=np.array([1, 10, 100]),
            energy_delta_plus=np.array([0.3, 3, 30]),
            energy_delta_minus=np.array([0.2, 2, 20]),
            energy_label=np.array(["1", "10", "100"]),
            latitude=sentinel.latitude,
            longitude=sentinel.longitude,
            exposure_factor=build_array(1, 2, 3),
            obs_date=build_array(datetime(2026, 1, 1),
                                 datetime(2026, 1, 2),
                                 datetime(2026, 1, 3),
                                 ),
            obs_date_range=build_array(1000, 2000, 3000),
            solid_angle=sentinel.solid_angle,
            ena_intensity=build_array(100, 200, 300),
            ena_intensity_stat_uncert=build_array(11, 12, 13),
            ena_intensity_sys_err=build_array(21, 22, 23),
            quality_flags=build_array(MapL3Flags.NONE, MapL3Flags.NONE, MapL3Flags.PREDICTIVE_EPHEMERIS),
        )

        actual = slice_energy_range_by_bin(input_data, 1, 3)
        self.assertIsInstance(actual, IntensityMapData)

        for field in dataclasses.fields(expected_data):
            np.testing.assert_equal(getattr(actual, field.name), getattr(expected_data, field.name))

    def test_slice_energy_range_by_bin_raises_error(self):
        def build_array(*values):
            return np.array([
                [
                    [[a]] for a in values
                ]
            ])

        input_data = IntensityMapData(
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=np.array([1, 10, 100, 1000, 10000]),
            energy_delta_plus=np.array([0.3, 3, 30, 300, 3000]),
            energy_delta_minus=np.array([0.2, 2, 20, 200, 2000]),
            energy_label=np.array(["1", "10", "100", "1000", "10000"]),
            latitude=sentinel.latitude,
            longitude=sentinel.longitude,
            exposure_factor=build_array(1, 2, 3, 4, 5),
            obs_date=build_array(datetime(2026, 1, 1),
                                 datetime(2026, 1, 2),
                                 datetime(2026, 1, 3),
                                 datetime(2026, 1, 4),
                                 datetime(2026, 1, 5),
                                 ),
            obs_date_range=build_array(1000, 2000, 3000, 4000, 5000),
            solid_angle=sentinel.solid_angle,
            ena_intensity=build_array(100, 200, 300, 400, 500),
            ena_intensity_stat_uncert=build_array(11, 12, 13, 14, 15),
            ena_intensity_sys_err=build_array(21, 22, 23, 24, 25),
            quality_flags=build_array(
                MapL3Flags.NONE,
                MapL3Flags.NONE,
                MapL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.NONE
            ),
        )
        cases = [
            (1, 10),
            (0, 3),
            (3, 0),
            (1, 1),
            (10, 3)
        ]
        for start_bin, end_bin in cases:
            with self.subTest(f"{start_bin}-{end_bin}"):
                with self.assertRaises(ValueError) as cm:
                    slice_energy_range_by_bin(input_data, start_bin, end_bin)

                self.assertEqual(str(cm.exception), f"Error slicing energy bins {start_bin},{end_bin}")
