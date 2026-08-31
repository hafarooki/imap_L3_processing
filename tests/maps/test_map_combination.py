import dataclasses
import unittest
from datetime import datetime

import numpy as np

from imap_l3_processing.maps.map_combination import UnweightedCombination, CombinationStrategy, \
    ExposureWeightedCombination, UncertaintyWeightedCombination
from imap_l3_processing.maps.map_models import IntensityMapData, RectangularIntensityMapData, RectangularCoords, \
    HealPixIntensityMapData, HealPixCoords
from imap_l3_processing.maps.quality_flags import MapL3Flags
from tests.maps.test_builders import construct_intensity_data_with_all_zero_fields, create_rectangular_intensity_map


class TestMapCombination(unittest.TestCase):
    def test_combine_maps_works_when_passed_all_nan_arrays(self):
        for combination_strategy in [UnweightedCombination, ExposureWeightedCombination,
                                     UncertaintyWeightedCombination]:
            with self.subTest(strategy=combination_strategy):
                combination = combination_strategy()
                self._combine_nan_maps(combination)

    def test_combine_maps_does_nothing_when_passed_a_single_map(self):
        for combination_strategy in [UnweightedCombination, ExposureWeightedCombination,
                                     UncertaintyWeightedCombination]:
            with self.subTest(strategy=combination_strategy):
                combination = combination_strategy()
                self._combine_single_map(combination)

    def test_combine_maps_works_for_maps_with_no_background_data(self):
        for combination_strategy in [UnweightedCombination, ExposureWeightedCombination,
                                     UncertaintyWeightedCombination]:
            with self.subTest(strategy=combination_strategy):
                combination = combination_strategy()
                self._combine_maps_with_no_background_data(combination)

    def test_combine_maps_works_for_maps_with_no_intensity_sys_errors(self):
        for combination_strategy in [UnweightedCombination, ExposureWeightedCombination,
                                     UncertaintyWeightedCombination]:
            with self.subTest(strategy=combination_strategy):
                combination = combination_strategy()
                self._combine_maps_with_no_intensity_sys_errors(combination)

    def test_check_maps_match(self):
        for combination_strategy in [UnweightedCombination, ExposureWeightedCombination,
                                     UncertaintyWeightedCombination]:
            with self.subTest(strategy=combination_strategy):
                combination = combination_strategy()
                self._check_maps_match(combination)

    def _combine_nan_maps(self, combination: CombinationStrategy):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.energy_delta_plus = np.array([np.nan])

        map_2 = construct_intensity_data_with_all_zero_fields()
        map_2.energy_delta_plus = np.array([np.nan])

        combine_with_nans = combination._combine([map_1, map_2])

        self.assertIsInstance(combine_with_nans, IntensityMapData)

    def _combine_single_map(self, combination: CombinationStrategy):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity_stat_uncert = np.array([1])

        combine_one = combination._combine([map_1])
        np.testing.assert_equal(dataclasses.asdict(combine_one), dataclasses.asdict(map_1))

    def _combine_maps_with_no_background_data(self, combination: CombinationStrategy):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity_stat_uncert = np.array([1])
        map_1.bg_intensity = None
        map_1.bg_intensity_sys_err = None
        map_1.bg_intensity_stat_uncert = None

        combine_one = combination._combine([map_1])
        np.testing.assert_equal(dataclasses.asdict(combine_one), dataclasses.asdict(map_1))

    def _combine_maps_with_no_intensity_sys_errors(self, combination: CombinationStrategy):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity_stat_uncert = np.array([1])
        map_1.ena_intensity_sys_err_plus = None
        map_1.ena_intensity_sys_err_minus = None

        combine_one = combination._combine([map_1])
        np.testing.assert_equal(dataclasses.asdict(combine_one), dataclasses.asdict(map_1))

    def _check_maps_match(self, combination: CombinationStrategy):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity_sys_err_plus = np.array([0] * 1),
        map_1.ena_intensity_sys_err_minus = np.array([0] * 2)

        fields_which_may_differ = {
            "epoch",
            "epoch_delta",
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "bg_intensity",
            "bg_intensity_stat_uncert",
            "bg_intensity_sys_err",
            "exposure_factor",
            "obs_date",
            "obs_date_range",
            "survival_probability",
            "ena_intensity_sys_err_minus",
            "ena_intensity_sys_err_plus",
            "quality_flags",
        }

        alternate_values_by_type = {datetime: datetime(2025, 5, 6), str: "label"}
        generic_value = 10

        for field in dataclasses.fields(map_1):
            replacement_value = alternate_values_by_type.get(type(getattr(map_1, field.name)[0]), generic_value)
            map_with_difference = dataclasses.replace(map_1, **{field.name: np.array([replacement_value])})
            if field.name not in fields_which_may_differ:
                with self.assertRaises(AssertionError, msg=field.name):
                    combination._check_maps_match([map_1, map_with_difference])
            else:
                try:
                    combination._check_maps_match([map_1, map_with_difference])
                except:
                    self.fail(f"Differences in other fields should be alright: {field.name}")

    def test_combine_maps_does_a_time_weighted_average_of_intensity(self):
        map_1 = construct_intensity_data_with_all_zero_fields()

        map_1.ena_intensity = np.array([1, np.nan, 3, 4, np.nan])
        map_1.ena_intensity_sys_err = np.array([1, np.nan, 10, 100, np.nan])
        map_1.ena_intensity_stat_uncert = np.array([10, np.nan, 10, 10, np.nan])

        map_1.bg_intensity = np.array([4, np.nan, 3, 1, np.nan])
        map_1.bg_intensity_sys_err = np.array([100, np.nan, 10, 1, np.nan])
        map_1.bg_intensity_stat_uncert = np.array([10, np.nan, 10, 10, np.nan])

        map_1.ena_intensity_sys_err_minus = np.array([10, np.nan, 10, 10, np.nan])
        map_1.ena_intensity_sys_err_plus = np.array([11, np.nan, 11, 11, np.nan])

        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])
        DATETIME_FILL = datetime(9999, 12, 31, 23, 59, 59, 999999)
        map_1.obs_date = np.ma.masked_equal(
            [
                datetime(2025, 5, 5),
                DATETIME_FILL,
                datetime(2025, 5, 7),
                datetime(2025, 5, 8),
                DATETIME_FILL,
            ],
            DATETIME_FILL,
        )

        map_1.survival_probability = np.array([1, np.nan, 1, 1, 1])

        map_2 = construct_intensity_data_with_all_zero_fields()
        map_2.ena_intensity = np.array([5, 6, 7, 8, np.nan])
        map_2.ena_intensity_sys_err = np.array([9, 4, 2, 0, np.nan])
        map_2.ena_intensity_stat_uncert = np.array([1, 2, 3, 4, np.nan])

        map_2.bg_intensity = np.array([8, 7, 6, 5, np.nan])
        map_2.bg_intensity_sys_err = np.array([0, 2, 4, 9, np.nan])
        map_2.bg_intensity_stat_uncert = np.array([4, 3, 2, 1, np.nan])

        map_2.ena_intensity_sys_err_minus = np.array([5, 6, 7, 8, np.nan])
        map_2.ena_intensity_sys_err_plus = np.array([8, 9, 10, 11, np.nan])

        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])
        map_2.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 9),
             datetime(2025, 5, 10),
             datetime(2025, 5, 11),
             datetime(2025, 5, 12),
             DATETIME_FILL],
            DATETIME_FILL)
        map_2.survival_probability = np.array([0, 0.5, np.nan, 1, 0])

        expected_combined_bg_stat_unc = [
            np.sqrt((1 * 100 + 9 * 16) / 16),
            np.sqrt((1 * 9) / 1),
            np.sqrt((25 * 100 + 25 * 4) / 100),
            np.sqrt((36 * 100 + 4 * 1) / 64),
            np.nan
        ]

        expected_combined_exposure = [4, 1, 10, 8, 0]
        expected_combined_intensity = [4, 6, 5, 5, np.nan]
        expected_sys_err = [
            (1 * 1 + 3 * 9) / (1 + 3),
            (1 * 4) / 1,
            (5 * 10 + 5 * 2) / (5 + 5),
            (6 * 100 + 2 * 0) / (6 + 2),
            np.nan
        ]
        expected_stat_unc = [
            np.sqrt((1 * 100 + 9 * 1) / 16),
            2,
            np.sqrt((25 * 100 + 25 * 9) / 100),
            np.sqrt((36 * 100 + 16 * 4) / 64),
            np.nan
        ]

        expected_combined_bg_intensity = [7, 7, 4.5, 2, np.nan]
        expected_combined_bg_intensity_sys_err = [
            (1 * 100 + 3 * 0) / (1 + 3),
            (0 * 10 + 1 * 2) / 1,
            (5 * 4 + 5 * 10) / (5 + 5),
            (6 * 1 + 2 * 9) / (6 + 2),
            np.nan,
        ]

        expected_combined_sys_err_minus = [
            (1 * 10 + 3 * 5) / (1 + 3),
            (1 * 6) / 1,
            (5 * 10 + 5 * 7) / (5 + 5),
            (6 * 10 + 2 * 8) / (6 + 2),
            np.nan
        ]

        expected_combined_sys_err_plus = [
            (1 * 11 + 3 * 8) / (1 + 3),
            (1 * 9) / 1,
            (5 * 11 + 5 * 10) / (5 + 5),
            (6 * 11 + 2 * 11) / (6 + 2),
            np.nan
        ]

        expected_obs_date = np.ma.array(
            [
                datetime(2025, 5, 8),
                datetime(2025, 5, 10),
                datetime(2025, 5, 9),
                datetime(2025, 5, 9),
                np.ma.masked,
            ]
        )

        expected_combined_survival_probability = [0.25, 0.5, 1, 1, np.nan]

        exposure_weighted_strategy = ExposureWeightedCombination()

        rectangular_map_1: RectangularIntensityMapData = (
            create_rectangular_intensity_map(map_1)
        )
        rectangular_map_2: RectangularIntensityMapData = (
            create_rectangular_intensity_map(map_2)
        )
        combine_two = exposure_weighted_strategy.combine_rectangular_intensity_map_data(
            [rectangular_map_1, rectangular_map_2]
        )

        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity, expected_combined_intensity
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity_sys_err, expected_sys_err
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity_stat_uncert, expected_stat_unc
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.exposure_factor, expected_combined_exposure
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.obs_date.mask, expected_obs_date.mask
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.obs_date, expected_obs_date
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.bg_intensity, expected_combined_bg_intensity
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.bg_intensity_sys_err,
            expected_combined_bg_intensity_sys_err,
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.bg_intensity_stat_uncert,
            expected_combined_bg_stat_unc,
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.survival_probability,
            expected_combined_survival_probability,
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity_sys_err_minus,
            expected_combined_sys_err_minus,
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity_sys_err_plus,
            expected_combined_sys_err_plus,
        )

    def test_combine_handles_maps_with_different_patterns_of_missing_intensity_and_bg_intensity(
            self,
    ):
        map_1 = construct_intensity_data_with_all_zero_fields(num_pixels=5)
        map_1.ena_intensity = np.array([1, np.nan, np.nan, np.nan, np.nan])
        map_1.bg_intensity = np.array([4, np.nan, 3, 1, np.nan])
        map_1.bg_intensity_sys_err = np.array([100, np.nan, 10, 1, np.nan])
        map_1.bg_intensity_stat_uncert = np.array([10, np.nan, 10, 10, np.nan])
        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])

        map_2 = construct_intensity_data_with_all_zero_fields(num_pixels=5)
        map_2.ena_intensity = np.array([5, 6, np.nan, np.nan, np.nan])
        map_2.bg_intensity = np.array([8, 7, 6, 5, np.nan])
        map_2.bg_intensity_sys_err = np.array([0, 2, 4, 9, np.nan])
        map_2.bg_intensity_stat_uncert = np.array([4, 3, 2, 1, np.nan])
        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])

        expected_combined_ena_intensity = np.array(
            [(1 * 1 + 5 * 3) / (1 + 3), (0 + 6 * 1) / (0 + 1), np.nan, np.nan, np.nan]
        )

        expected_combined_bg_intensity = np.array(
            [
                (4 * 1 + 8 * 3) / (1 + 3),
                (0 + 7 * 1) / (0 + 1),
                (3 * 5 + 6 * 5) / (5 + 5),
                (1 * 6 + 5 * 2) / (6 + 2),
                np.nan,
            ]
        )

        expected_combined_bg_stat_unc = [
            np.sqrt((1 * 100 + 9 * 16) / 16),
            np.sqrt((1 * 9) / 1),
            np.sqrt((25 * 100 + 25 * 4) / 100),
            np.sqrt((36 * 100 + 4 * 1) / 64),
            np.nan,
        ]

        expected_combined_bg_intensity_sys_err = [
            (1 * 100 + 3 * 0) / (1 + 3),
            (0 * 10 + 1 * 2) / 1,
            (5 * 4 + 5 * 10) / (5 + 5),
            (6 * 1 + 2 * 9) / (6 + 2),
            np.nan,
        ]

        combined_map = (
            ExposureWeightedCombination().combine_rectangular_intensity_map_data(
                [
                    create_rectangular_intensity_map(map_1),
                    create_rectangular_intensity_map(map_2),
                ]
            )
        )

        np.testing.assert_array_equal(
            combined_map.intensity_map_data.ena_intensity,
            expected_combined_ena_intensity,
        )
        np.testing.assert_array_equal(
            combined_map.intensity_map_data.bg_intensity, expected_combined_bg_intensity
        )
        np.testing.assert_array_equal(combined_map.intensity_map_data.bg_intensity_stat_uncert,
                                      expected_combined_bg_stat_unc)
        np.testing.assert_array_equal(
            combined_map.intensity_map_data.bg_intensity_sys_err,
            expected_combined_bg_intensity_sys_err,
        )

    def test_combine_maps_handles_integer_obs_date(self):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])
        DATETIME_FILL = -9223372036854775808
        map_1.obs_date = np.ma.masked_equal(
            [5, DATETIME_FILL, 7, 8, DATETIME_FILL], DATETIME_FILL
        )

        map_2 = construct_intensity_data_with_all_zero_fields()
        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])
        map_2.obs_date = np.ma.masked_equal(
            [9, 10, 11, 12, DATETIME_FILL], DATETIME_FILL
        )
        expected_obs_date = np.ma.array([8, 10, 9, 9, np.ma.masked])

        exposure_weighted_strategy = ExposureWeightedCombination()

        rectangular_map_1: RectangularIntensityMapData = (
            create_rectangular_intensity_map(map_1)
        )
        rectangular_map_2: RectangularIntensityMapData = (
            create_rectangular_intensity_map(map_2)
        )

        combine_two = exposure_weighted_strategy.combine_rectangular_intensity_map_data(
            [rectangular_map_1, rectangular_map_2])

        np.testing.assert_equal(combine_two.intensity_map_data.obs_date.mask, expected_obs_date.mask)
        np.testing.assert_equal(
            combine_two.intensity_map_data.obs_date, expected_obs_date
        )

    def test_combine_unweighted_combination_strategy(self):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity = np.array([1, np.nan, 3, 4, np.nan, np.nan, 0, 5, 13])
        map_1.exposure_factor = np.array([100, 0, 100, 100, 100, 100, 100, 100, 90])
        map_1.ena_intensity_sys_err = np.array(
            [3, 1, np.nan, 12, np.nan, 1, np.nan, np.nan, np.nan]
        )
        map_1.ena_intensity_stat_uncert = np.array(
            [6, 1, np.nan, 24, np.nan, 1, np.nan, np.nan, np.nan]
        )
        DATETIME_FILL = datetime(9999, 12, 31, 23, 59, 59, 999999)
        map_1.obs_date = np.ma.masked_equal(
            [
                datetime(2025, 5, 5),
                DATETIME_FILL,
                datetime(2025, 5, 7),
                datetime(2025, 5, 8),
                DATETIME_FILL,
                DATETIME_FILL,
                DATETIME_FILL,
                DATETIME_FILL,
                datetime(2025, 5, 9),
            ],
            DATETIME_FILL,
        )
        map_1.survival_probability = np.array([0.9, 1, 1, 1, np.nan, 1, 1, 1, 1])
        map_1.quality_flags = np.array([
            MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE,
            MapL3Flags.NONE,
            MapL3Flags.NONE,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE,
            MapL3Flags.NONE,
            MapL3Flags.NONE
        ])

        map_2 = construct_intensity_data_with_all_zero_fields()
        map_2.ena_intensity = np.array([5, 6, 7, 8, np.nan, 100, 9, 10, 11])
        map_2.exposure_factor = np.array([10, 0, 10, 10, 10, 100, 100, np.nan, 30])
        map_2.ena_intensity_sys_err = np.array([4, 1, 4, 5, np.nan, 1, np.nan, np.nan, np.nan])
        map_2.ena_intensity_stat_uncert = np.array([8, 1, 8, 10, np.nan, 1, np.nan, np.nan, np.nan])
        map_2.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 9),
             datetime(2025, 5, 10),
             datetime(2025, 5, 11),
             datetime(2025, 5, 12),
             DATETIME_FILL,
             DATETIME_FILL,
             DATETIME_FILL,
             DATETIME_FILL,
             datetime(2025, 5, 13),
             ],
            DATETIME_FILL)
        map_2.survival_probability = np.array([0.1, 1, 1, 1, 0.2, 0.2, 1, 0.8, 0])
        map_2.quality_flags = np.array([
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE,
            MapL3Flags.PERSISTED_LAST_POINT,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            MapL3Flags.PERSISTED_LAST_POINT,
            MapL3Flags.NONE,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE
        ])

        expected_combined_intensity = [3, np.nan, 5, 6, np.nan, 100, 4.5, 5, 12]
        expected_sys_err = [2.5, np.nan, np.nan, 6.5, np.nan, 1, np.nan, np.nan, np.nan]
        expected_stat_unc = [5, np.nan, np.nan, 13, np.nan, 1, np.nan, np.nan, np.nan]
        expected_obs_date = datetime(2025, 5, 11)
        expected_survival_probability = [0.5, np.nan, 1, 1, 0.2, 0.6, 1, 1, 0.5]
        expected_quality_flags = [
            MapL3Flags.PREDICTIVE_EPHEMERIS | MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.PERSISTED_LAST_POINT,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            MapL3Flags.PREDICTIVE_EPHEMERIS | MapL3Flags.PERSISTED_LAST_POINT,
            MapL3Flags.NONE,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE]

        exposure_unweighted_strategy = UnweightedCombination()

        rectangular_map_1: RectangularIntensityMapData = create_rectangular_intensity_map(map_1)
        rectangular_map_2: RectangularIntensityMapData = create_rectangular_intensity_map(map_2)

        combine_two = exposure_unweighted_strategy.combine_rectangular_intensity_map_data(
            [rectangular_map_1, rectangular_map_2])

        np.testing.assert_equal(combine_two.intensity_map_data.ena_intensity, expected_combined_intensity)
        np.testing.assert_equal(combine_two.intensity_map_data.ena_intensity_sys_err, expected_sys_err)
        np.testing.assert_equal(combine_two.intensity_map_data.ena_intensity_stat_uncert, expected_stat_unc)
        np.testing.assert_equal(combine_two.intensity_map_data.obs_date[-1], expected_obs_date)
        np.testing.assert_equal(combine_two.intensity_map_data.survival_probability, expected_survival_probability)
        np.testing.assert_equal(combine_two.intensity_map_data.quality_flags, expected_quality_flags)

    def test_combine_rectangular_intensity_map_data_errors_if_coords_not_matching(self):
        delta_array = np.array([1])
        label_array = np.array(["one"])

        def make_data(lat_delta=delta_array, lon_delta=delta_array, lat_label=label_array, lon_label=label_array):
            return RectangularIntensityMapData(
                intensity_map_data=construct_intensity_data_with_all_zero_fields(),
                coords=RectangularCoords(
                    latitude_delta=lat_delta,
                    longitude_delta=lon_delta,
                    latitude_label=lat_label,
                    longitude_label=lon_label,
                )
            )

        base_map = make_data()
        cases = [
            ("lat_delta", make_data(lat_delta=np.array([2]))),
            ("lon_delta", make_data(lon_delta=np.array([2]))),
            ("lat_label", make_data(lat_label=np.array(["two"]))),
            ("lon_label", make_data(lon_label=np.array(["two"]))),
        ]
        for name, not_matching_map in cases:
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    strategy = UnweightedCombination()
                    strategy.combine_rectangular_intensity_map_data([base_map, not_matching_map])

    def test_combine_healpix_intensity_map_data_errors_if_coords_not_matching(self):
        index_array = np.array([1])
        label_array = np.array(["one"])

        def make_data(pixel_index=index_array, pixel_index_label=label_array):
            return HealPixIntensityMapData(
                intensity_map_data=construct_intensity_data_with_all_zero_fields(),
                coords=HealPixCoords(
                    pixel_index=pixel_index,
                    pixel_index_label=pixel_index_label,
                )
            )

        base_map = make_data()
        cases = [
            ("pixel_index", make_data(pixel_index=np.array([2]), pixel_index_label=label_array)),
            ("pixel_index_label", make_data(pixel_index=index_array, pixel_index_label=np.array(["diff label"]))),
        ]
        for name, not_matching_map in cases:
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    strategy = UnweightedCombination()
                    strategy.combine_healpix_intensity_map_data([base_map, not_matching_map])

    def test_calculate_weighted_sys_err(self):
        uncertainties_1 = np.array([10, 20, 30, 40])
        uncertainties_2 = np.array([20, 40, 60, 80])

        exposures_1 = np.array([4, 3, 2, 1])
        exposures_2 = np.array([4, 3, 2, 1])

        expected = np.array([15, 30, 45, 60])
        actual = CombinationStrategy.calculate_weighted_sys_err(np.array([uncertainties_1, uncertainties_2]),
                                                                np.array([exposures_1, exposures_2]))
        np.testing.assert_array_equal(actual, expected)

    def test_uncertainty_weighted_combination(self):
        map_1 = construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity = np.array([1, np.nan, 3, 4, np.nan])
        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])
        map_1.ena_intensity_sys_err = np.array([1, np.nan, 10, 100, np.nan])
        map_1.ena_intensity_stat_uncert = np.array([10, np.nan, 10, 10, np.nan])
        DATETIME_FILL = datetime(9999, 12, 31, 23, 59, 59, 999999)
        map_1.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 5),
             DATETIME_FILL,
             datetime(2025, 5, 7),
             datetime(2025, 5, 8),
             DATETIME_FILL],
            DATETIME_FILL)
        map_1.survival_probability = np.array([1, 0.2, 0.8, 1, np.nan])

        map_2 = construct_intensity_data_with_all_zero_fields()
        map_2.ena_intensity = np.array([5, 6, 7, 8, np.nan])
        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])
        map_2.ena_intensity_sys_err = np.array([9, 4, np.inf, 0, np.nan])
        map_2.ena_intensity_stat_uncert = np.array([1, 2, np.inf, 4, np.nan])
        map_2.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 9),
             datetime(2025, 5, 10),
             datetime(2025, 5, 11),
             datetime(2025, 5, 12),
             DATETIME_FILL],
            DATETIME_FILL)
        map_2.survival_probability = np.array([1, 0.5, 0.2, 0.25, np.nan])

        expected_combined_exposure = [4, 1, 5, 8, 0]
        expected_combined_intensity = [
            (1 / (10 ** 2) + 5 / (1 ** 2)) / (1 / (10 ** 2) + 1 / (1 ** 2)),
            (6 / (2 ** 2)) / (1 / (2 ** 2)),
            (3 / (10 ** 2)) / (1 / (10 ** 2)),
            (4 / (10 ** 2) + 8 / (4 ** 2)) / (1 / (10 ** 2) + 1 / (4 ** 2)),
            np.nan
        ]
        expected_stat_unc = [
            np.sqrt(1 / (1 / (10 ** 2) + 1 / (1 ** 2))),
            np.sqrt(1 / (1 / (2 ** 2))),
            np.sqrt(1 / (1 / (10 ** 2))),
            np.sqrt(1 / (1 / (10 ** 2) + 1 / (4 ** 2))),
            np.nan
        ]
        expected_sys_err = [
            (1 * 1 + 3 * 9) / (1 + 3),
            (1 * 4) / 1,
            (5 * 10) / (5),
            (6 * 100 + 2 * 0) / (6 + 2),
            np.nan
        ]
        expected_obs_date = np.ma.array(
            [datetime(2025, 5, 8),
             datetime(2025, 5, 10),
             datetime(2025, 5, 7),
             datetime(2025, 5, 9), np.ma.masked])
        expected_survival_probability = np.array(
            [1, 0.5, 0.8, (1 / 10 ** 2 + 0.25 / 4 ** 2) / (1 / 10 ** 2 + 1 / 4 ** 2), np.nan])

        exposure_weighted_strategy = UncertaintyWeightedCombination()

        rectangular_map_1 = create_rectangular_intensity_map(map_1)
        rectangular_map_2 = create_rectangular_intensity_map(map_2)
        combine_two = exposure_weighted_strategy.combine_rectangular_intensity_map_data(
            [rectangular_map_1, rectangular_map_2])

        np.testing.assert_equal(combine_two.intensity_map_data.ena_intensity, expected_combined_intensity)
        np.testing.assert_equal(combine_two.intensity_map_data.ena_intensity_sys_err, expected_sys_err)
        np.testing.assert_equal(
            combine_two.intensity_map_data.ena_intensity_stat_uncert, expected_stat_unc
        )
        np.testing.assert_equal(combine_two.intensity_map_data.exposure_factor, expected_combined_exposure)
        np.testing.assert_equal(
            combine_two.intensity_map_data.obs_date.mask, expected_obs_date.mask
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.obs_date, expected_obs_date
        )
        np.testing.assert_equal(
            combine_two.intensity_map_data.survival_probability, expected_survival_probability
        )

    def test_combination_handles_missing_sp_data(self):
        test_cases = [
            ExposureWeightedCombination,
            UncertaintyWeightedCombination,
            UnweightedCombination
        ]

        for combination_strategy in test_cases:
            with self.subTest(combination_strategy.__name__):
                map_1 = construct_intensity_data_with_all_zero_fields()
                map_1.survival_probability = None

                map_2 = construct_intensity_data_with_all_zero_fields()
                map_2.survival_probability = None

                combined_map = combination_strategy().combine_rectangular_intensity_map_data([
                    create_rectangular_intensity_map(map_1),
                    create_rectangular_intensity_map(map_2)
                ])

                self.assertIsNone(combined_map.intensity_map_data.survival_probability)

    def test_combination_handles_quality_flag(self):
        test_cases = [
            ExposureWeightedCombination,
            UncertaintyWeightedCombination,
            UnweightedCombination
        ]

        for combination_strategy in test_cases:
            with self.subTest(combination_strategy.__name__):
                map_1 = construct_intensity_data_with_all_zero_fields()
                map_1.quality_flags = np.array([
                    MapL3Flags.NONE,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.NONE,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.NONE,
                    MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                ])

                map_2 = construct_intensity_data_with_all_zero_fields()
                map_2.quality_flags = np.array([
                    MapL3Flags.NONE,
                    MapL3Flags.NONE,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                ])

                combined_map = combination_strategy().combine_rectangular_intensity_map_data([
                    create_rectangular_intensity_map(map_1),
                    create_rectangular_intensity_map(map_2)
                ])

                np.testing.assert_equal(combined_map.intensity_map_data.quality_flags, [
                    MapL3Flags.NONE,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.PREDICTIVE_EPHEMERIS,
                    MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                    MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PREDICTIVE_EPHEMERIS,
                ])
