import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import sentinel, Mock

import numpy as np
from imap_processing.ena_maps.ena_maps import HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.spatial_utils import build_solid_angle_map
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF
from xarray import Dataset

from imap_l3_processing.cdf.cdf_utils import read_variable_and_mask_fill_values, read_numeric_variable
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS, SECONDS_PER_DAY, FIVE_MINUTES_IN_NANOSECONDS, \
    TT2000_EPOCH
from imap_l3_processing.maps import map_models
from imap_l3_processing.maps.map_models import (
    RectangularCoords,
    SpectralIndexMapData,
    RectangularSpectralIndexMapData,
    RectangularSpectralIndexDataProduct,
    RectangularIntensityMapData,
    IntensityMapData,
    RectangularIntensityDataProduct,
    HealPixIntensityMapData,
    HealPixCoords,
    convert_tt2000_time_to_datetime,
    _read_intensity_map_data_from_open_cdf,
    calculate_datetime_weighted_average,
    ISNRateData,
    ISNBackgroundSubtractedData,
    ISNBackgroundSubtractedDataProduct,
    ISNBackgroundSubtractedMapData,
    QUALITY_FLAGS_VAR_NAME,
)
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import (
    get_test_data_folder,
    get_integration_test_data_path,
    NumpyArrayMatcher,
)


class TestMapModels(unittest.TestCase):

    def test_rectangular_spectral_index_to_data_product_variables(self):
        input_metadata = Mock()

        spectral_index_data_product = RectangularSpectralIndexDataProduct(
            input_metadata=input_metadata,
            data=RectangularSpectralIndexMapData(
                spectral_index_map_data=SpectralIndexMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_spectral_index=sentinel.ena_spectral_index,
                    ena_spectral_index_stat_uncert=sentinel.ena_spectral_index_stat_uncert,
                    ena_spectral_index_scalar_coefficient=sentinel.ena_spectral_index_scalar_coefficient,
                    ena_spectral_index_scalar_coefficient_stat_uncert=sentinel.ena_spectral_index_scalar_coefficient_stat_uncert,
                    ena_spectral_index_chisq=sentinel.ena_spectral_index_chisq,
                    quality_flags=sentinel.quality_flags,
                ),
                coords=RectangularCoords(
                    latitude_delta=sentinel.latitude_delta,
                    latitude_label=sentinel.latitude_label,
                    longitude_delta=sentinel.longitude_delta,
                    longitude_label=sentinel.longitude_label,
                ),
            ),
            spice_frame_name=SpiceFrame.ECLIPJ2000,
        )

        actual_variables = spectral_index_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(
                map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus
            ),
            DataProductVariable(
                map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus
            ),
            DataProductVariable(
                map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label
            ),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(
                map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor
            ),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(
                map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range
            ),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(
                map_models.ENA_SPECTRAL_INDEX_VAR_NAME, sentinel.ena_spectral_index
            ),
            DataProductVariable(
                map_models.ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME,
                sentinel.ena_spectral_index_stat_uncert,
            ),
            DataProductVariable(
                map_models.ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_VAR_NAME,
                sentinel.ena_spectral_index_scalar_coefficient,
            ),
            DataProductVariable(
                map_models.ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_STAT_UNCERT_VAR_NAME,
                sentinel.ena_spectral_index_scalar_coefficient_stat_uncert,
            ),
            DataProductVariable(
                map_models.ENA_SPECTRAL_INDEX_CHISQ_VAR_NAME,
                sentinel.ena_spectral_index_chisq,
            ),
            DataProductVariable(
                map_models.QUALITY_FLAGS_VAR_NAME,
                sentinel.quality_flags
            ),
            DataProductVariable(
                map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta
            ),
            DataProductVariable(
                map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label
            ),
            DataProductVariable(
                map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta
            ),
            DataProductVariable(
                map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label
            ),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_rectangular_intensity_to_data_product_variables(self):
        test_cases = [
            ("no bg or sp", {}, []),
            (
                "bg",
                {
                    "bg_intensity": sentinel.bg_intensity,
                    "bg_intensity_stat_uncert": sentinel.bg_intensity_stat_uncert,
                    "bg_intensity_sys_err": sentinel.bg_intensity_sys_err,
                },
                [
                    DataProductVariable(
                        map_models.BG_INTENSITY_VAR_NAME, sentinel.bg_intensity
                    ),
                    DataProductVariable(
                        map_models.BG_INTENSITY_STAT_UNC_VAR_NAME,
                        sentinel.bg_intensity_stat_uncert,
                    ),
                    DataProductVariable(
                        map_models.BG_INTENSITY_SYS_ERR_VAR_NAME,
                        sentinel.bg_intensity_sys_err,
                    ),
                ],
            ),
            (
                "sp",
                {
                    "survival_probability": sentinel.survival_probability,
                },
                [
                    DataProductVariable(
                        map_models.SURVIVAL_PROBABILITY_VAR_NAME,
                        sentinel.survival_probability,
                    ),
                ],
            ),
            (
                "sys_err_plus and minus",
                {
                    "ena_intensity_sys_err_minus": sentinel.ena_intensity_sys_err_minus,
                    "ena_intensity_sys_err_plus": sentinel.ena_intensity_sys_err_plus,
                },
                [
                    DataProductVariable(
                        map_models.ENA_INTENSITY_SYS_ERR_MINUS_VAR_NAME,
                        sentinel.ena_intensity_sys_err_minus,
                    ),
                    DataProductVariable(
                        map_models.ENA_INTENSITY_SYS_ERR_PLUS_VAR_NAME,
                        sentinel.ena_intensity_sys_err_plus,
                    ),
                ],
            ),
        ]

        for (
                test_name,
                additional_inputs,
                expected_additional_data_products_vars,
        ) in test_cases:
            with self.subTest(test_name):
                input_metadata = sentinel.input_metadata

                data_product = RectangularIntensityDataProduct(
                    input_metadata=input_metadata,
                    data=RectangularIntensityMapData(
                        intensity_map_data=IntensityMapData(
                            epoch=sentinel.epoch,
                            epoch_delta=sentinel.epoch_delta,
                            energy=sentinel.energy,
                            energy_delta_plus=sentinel.energy_delta_plus,
                            energy_delta_minus=sentinel.energy_delta_minus,
                            energy_label=sentinel.energy_label,
                            latitude=sentinel.latitude,
                            longitude=sentinel.longitude,
                            exposure_factor=sentinel.exposure_factor,
                            obs_date=sentinel.obs_date,
                            obs_date_range=sentinel.obs_date_range,
                            solid_angle=sentinel.solid_angle,
                            ena_intensity=sentinel.ena_intensity,
                            ena_intensity_stat_uncert=sentinel.ena_intensity_stat_uncert,
                            ena_intensity_sys_err=sentinel.ena_intensity_sys_err,
                            quality_flags=sentinel.quality_flags,
                            **additional_inputs
                        ),
                        coords=RectangularCoords(
                            longitude_delta=sentinel.longitude_delta,
                            longitude_label=sentinel.longitude_label,
                            latitude_delta=sentinel.latitude_delta,
                            latitude_label=sentinel.latitude_label,

                        )
                    ),
                    spice_frame_name=SpiceFrame.ECLIPJ2000,
                )
                actual_variables = data_product.to_data_product_variables()

                expected_variables = [
                    DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
                    DataProductVariable(
                        map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta
                    ),
                    DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
                    DataProductVariable(
                        map_models.ENERGY_DELTA_PLUS_VAR_NAME,
                        sentinel.energy_delta_plus,
                    ),
                    DataProductVariable(
                        map_models.ENERGY_DELTA_MINUS_VAR_NAME,
                        sentinel.energy_delta_minus,
                    ),
                    DataProductVariable(
                        map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label
                    ),
                    DataProductVariable(
                        map_models.LATITUDE_VAR_NAME, sentinel.latitude
                    ),
                    DataProductVariable(
                        map_models.LONGITUDE_VAR_NAME, sentinel.longitude
                    ),
                    DataProductVariable(
                        map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor
                    ),
                    DataProductVariable(
                        map_models.OBS_DATE_VAR_NAME, sentinel.obs_date
                    ),
                    DataProductVariable(
                        map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range
                    ),
                    DataProductVariable(
                        map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle
                    ),
                    DataProductVariable(
                        map_models.ENA_INTENSITY_VAR_NAME, sentinel.ena_intensity
                    ),
                    DataProductVariable(
                        map_models.ENA_INTENSITY_STAT_UNCERT_VAR_NAME,
                        sentinel.ena_intensity_stat_uncert,
                    ),
                    DataProductVariable(
                        map_models.ENA_INTENSITY_SYS_ERR_VAR_NAME,
                        sentinel.ena_intensity_sys_err,
                    ),
                    DataProductVariable(
                        map_models.QUALITY_FLAGS_VAR_NAME,
                        sentinel.quality_flags,
                    ),
                    *expected_additional_data_products_vars,
                    DataProductVariable(
                        map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta
                    ),
                    DataProductVariable(
                        map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label
                    ),
                    DataProductVariable(
                        map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta
                    ),
                    DataProductVariable(
                        map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label
                    ),
                ]

                self.assertEqual(expected_variables, actual_variables)

    def test_isn_background_subtracted_to_data_product_variables(self):
        input_metadata = Mock()

        data_product = ISNBackgroundSubtractedDataProduct(
            input_metadata=input_metadata,
            data=ISNBackgroundSubtractedMapData(
                isn_rate_map_data=ISNBackgroundSubtractedData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    exposure_factor=sentinel.exposure_factor,
                    solid_angle=sentinel.solid_angle,
                    bg_rate=sentinel.bg_rate,
                    bg_rate_stat_uncert=sentinel.bg_rate_stat_uncert,
                    bg_rate_sys_err=sentinel.bg_rate_sys_err,
                    latitude=sentinel.latitude,
                    latitude_delta=sentinel.latitude_delta,
                    latitude_label=sentinel.latitude_label,
                    longitude=sentinel.longitude,
                    longitude_delta=sentinel.longitude_delta,
                    longitude_label=sentinel.longitude_label,
                    isn_bg_rate_subtracted=sentinel.isn_bg_rate_subtracted,
                    isn_bg_rate_subtracted_sys_err=sentinel.isn_bg_rate_subtracted_sys_err,
                    isn_bg_rate_subtracted_stat_uncert=sentinel.isn_bg_rate_subtracted_stat_uncert,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                )
            ),
            spice_frame_name=SpiceFrame.ECLIPJ2000,
        )

        actual_variables = data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(
                map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus
            ),
            DataProductVariable(
                map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus
            ),
            DataProductVariable(
                map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label
            ),
            DataProductVariable(
                map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor
            ),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.BG_RATE_VAR_NAME, sentinel.bg_rate),
            DataProductVariable(
                map_models.BG_RATE_STAT_UNCERT_VAR_NAME, sentinel.bg_rate_stat_uncert
            ),
            DataProductVariable(
                map_models.BG_RATE_SYS_ERR_VAR_NAME, sentinel.bg_rate_sys_err
            ),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(
                map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta
            ),
            DataProductVariable(
                map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label
            ),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(
                map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta
            ),
            DataProductVariable(
                map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label
            ),
            DataProductVariable(
                map_models.ISN_BG_RATE_SUBTRACTED_VAR_NAME,
                sentinel.isn_bg_rate_subtracted,
            ),
            DataProductVariable(
                map_models.ISN_BG_RATE_SUBTRACTED_VAR_SYS_ERR_NAME,
                sentinel.isn_bg_rate_subtracted_sys_err,
            ),
            DataProductVariable(
                map_models.ISN_BG_RATE_SUBTRACTED_STAT_UNCERT_VAR_NAME,
                sentinel.isn_bg_rate_subtracted_stat_uncert,
            ),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_healpix_map_nside_property(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l2_maps' / 'test_l2_map.cdf'
        actual = HealPixIntensityMapData.read_from_path(path_to_cdf)
        expected_nside = 2

        self.assertIsInstance(actual.coords.nside, int)
        self.assertEqual(expected_nside, actual.coords.nside)

    def test_ultra_l2_map_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-2deg-6mo_20250415_v102.cdf'

        map_data = RectangularIntensityMapData.read_from_path(path_to_cdf)
        actual_intensity_data = map_data.intensity_map_data
        actual_coords = map_data.coords

        with CDF(str(path_to_cdf)) as expected:
            date_range_var = read_variable_and_mask_fill_values(expected["obs_date_range"])
            obs_date = convert_tt2000_time_to_datetime(
                read_variable_and_mask_fill_values(expected["obs_date"]).filled(0))

            self.assertEqual(expected['epoch'][...], actual_intensity_data.epoch)
            np.testing.assert_array_equal(expected["epoch_delta"][...], actual_intensity_data.epoch_delta)
            np.testing.assert_array_equal(expected["energy"][...], actual_intensity_data.energy)
            np.testing.assert_array_equal(expected["energy_delta_plus"][...], actual_intensity_data.energy_delta_plus)
            np.testing.assert_array_equal(expected["energy_delta_minus"][...], actual_intensity_data.energy_delta_minus)
            np.testing.assert_array_equal(expected["energy_label"][...], actual_intensity_data.energy_label)
            np.testing.assert_array_equal(expected["latitude"][...], actual_intensity_data.latitude)
            np.testing.assert_array_equal(expected["latitude_delta"][...], actual_coords.latitude_delta)
            np.testing.assert_array_equal(expected["latitude_label"][...], actual_coords.latitude_label)
            np.testing.assert_array_equal(expected["longitude"][...], actual_intensity_data.longitude)
            np.testing.assert_array_equal(expected["longitude_delta"][...], actual_coords.longitude_delta)
            np.testing.assert_array_equal(expected["longitude_label"][...], actual_coords.longitude_label)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual_intensity_data.exposure_factor)
            np.testing.assert_array_equal(obs_date, actual_intensity_data.obs_date)
            np.testing.assert_array_equal(date_range_var, actual_intensity_data.obs_date_range)
            np.testing.assert_array_equal(expected["solid_angle"][...], actual_intensity_data.solid_angle)
            np.testing.assert_array_equal(expected["ena_intensity"][...], actual_intensity_data.ena_intensity)
            np.testing.assert_array_equal(expected["ena_intensity_stat_uncert"][...],
                                          actual_intensity_data.ena_intensity_stat_uncert)
            np.testing.assert_array_equal(expected["ena_intensity_sys_err"][...],
                                          actual_intensity_data.ena_intensity_sys_err)

    def test_ultra_healpix_intensity_read_from_xarray(self):
        full_shape = [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.HEALPIX_INDEX.value]
        input_xarray: Dataset = Dataset(
            data_vars={
                "latitude": ([CoordNames.HEALPIX_INDEX.value], np.full((12,), 4)),
                "longitude": ([CoordNames.HEALPIX_INDEX.value], np.full((12,), 5)),
                "solid_angle": ([CoordNames.HEALPIX_INDEX.value], np.full((12,), 6)),
                "obs_date_range": (full_shape, np.full((2, 15, 12), 7)),
                "obs_date": (full_shape, np.full((2, 15, 12), 8)),
                "exposure_factor": (full_shape, np.full((2, 15, 12), 9)),
                "ena_intensity": (full_shape, np.full((2, 15, 12), 10)),
                "ena_intensity_stat_uncert": (full_shape, np.full((2, 15, 12), 11)),
                "ena_intensity_sys_err": (full_shape, np.full((2, 15, 12), 12)),
                "epoch_delta": ([CoordNames.TIME.value], np.full((2,), 13)),
                "energy_delta_minus": (
                    [CoordNames.ENERGY_L2.value],
                    np.full((15,), 14),
                ),
                "energy_delta_plus": ([CoordNames.ENERGY_L2.value], np.full((15,), 15)),
                "energy_label": ([CoordNames.ENERGY_L2.value], np.full((15,), "123")),
                "pixel_index_label": (
                    [CoordNames.HEALPIX_INDEX.value],
                    np.full((12,), "Pixel"),
                ),
            },
            coords={
                CoordNames.TIME.value: np.full((2,), 1),
                CoordNames.ENERGY_L2.value: np.full((15,), 2),
                CoordNames.HEALPIX_INDEX.value: np.full((12,), 3),
            },
        )

        output: HealPixIntensityMapData = HealPixIntensityMapData.read_from_xarray(
            input_xarray
        )

        np.testing.assert_array_equal(
            input_xarray["latitude"], output.intensity_map_data.latitude
        )
        np.testing.assert_array_equal(
            input_xarray["longitude"], output.intensity_map_data.longitude
        )
        np.testing.assert_array_equal(
            input_xarray["solid_angle"], output.intensity_map_data.solid_angle
        )
        np.testing.assert_array_equal(
            input_xarray["obs_date"], output.intensity_map_data.obs_date
        )
        np.testing.assert_array_equal(
            input_xarray["obs_date_range"], output.intensity_map_data.obs_date_range
        )
        np.testing.assert_array_equal(
            input_xarray["exposure_factor"], output.intensity_map_data.exposure_factor
        )
        np.testing.assert_array_equal(
            input_xarray["ena_intensity"], output.intensity_map_data.ena_intensity
        )
        np.testing.assert_array_equal(
            input_xarray["ena_intensity_stat_uncert"],
            output.intensity_map_data.ena_intensity_stat_uncert,
        )
        np.testing.assert_array_equal(
            input_xarray["ena_intensity_sys_err"],
            output.intensity_map_data.ena_intensity_sys_err,
        )

        np.testing.assert_array_equal(
            input_xarray[CoordNames.TIME.value], output.intensity_map_data.epoch
        )
        np.testing.assert_array_equal(
            input_xarray["epoch_delta"], output.intensity_map_data.epoch_delta
        )
        np.testing.assert_array_equal(
            input_xarray[CoordNames.HEALPIX_INDEX.value], output.coords.pixel_index
        )
        np.testing.assert_array_equal(
            input_xarray["pixel_index_label"], output.coords.pixel_index_label
        )
        np.testing.assert_array_equal(
            input_xarray[CoordNames.ENERGY_L2.value], output.intensity_map_data.energy
        )
        np.testing.assert_array_equal(input_xarray["energy_delta_minus"], output.intensity_map_data.energy_delta_minus)
        np.testing.assert_array_equal(
            input_xarray["energy_delta_plus"],
            output.intensity_map_data.energy_delta_plus,
        )
        np.testing.assert_array_equal(
            input_xarray["energy_label"], output.intensity_map_data.energy_label
        )
        np.testing.assert_array_equal(output.intensity_map_data.quality_flags, np.full((2, 15, 12),MapL3Flags.NONE), strict=True)

    def test_read_intensity_map_with_rectangular_cords_data_from_cdf(self):

        rng = np.random.default_rng()

        map_data_shape = (1, 9, 90, 45)
        obs_date_datetime = np.full(map_data_shape, datetime.now())
        obs_date_fillval = -sys.maxsize - 1
        test_cases = [
            (
                "obs date is datetime",
                obs_date_datetime,
                obs_date_datetime,
                np.full(map_data_shape, False),
                False,
                True
            ),
            (
                "obs date is int",
                np.full(map_data_shape, 1e9, dtype=int),
                np.full(map_data_shape, TT2000_EPOCH) + timedelta(seconds=1),
                np.full(map_data_shape, False),
                True,
                False
            ),
            (
                "obs date is all fill",
                np.full(map_data_shape, obs_date_fillval, dtype=int),
                np.full(map_data_shape, TT2000_EPOCH),
                np.full(map_data_shape, True),
                True,
                True
            ),
        ]

        for test_name, obs_date_in_cdf, expected_obs_date, expected_obs_date_mask, include_bg, include_ena_sys_err_minus in test_cases:
            with tempfile.TemporaryDirectory() as temp_dir:
                pathname = os.path.join(temp_dir, "test_cdf")
                with CDF(pathname, '') as cdf:
                    cdf.col_major(True)

                    ena_intensity = rng.random(map_data_shape)
                    bg_intensity = ena_intensity * 0.01
                    energy = rng.random(9)
                    energy_delta_plus = rng.random(9)
                    energy_delta_minus = rng.random(9)
                    energy_label = energy.astype(str)
                    ena_intensity_stat_uncert = rng.random(map_data_shape)
                    ena_intensity_sys_err = rng.random(map_data_shape)

                    ena_intensity_sys_err_minus = rng.random(map_data_shape)
                    ena_intensity_sys_err_plus = rng.random(map_data_shape)

                    bg_intensity_stat_unc = rng.random(map_data_shape)
                    bg_intensity_sys_err = rng.random(map_data_shape)

                    survival_probability = rng.random(map_data_shape)

                    epoch = np.array([datetime.now()])
                    epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
                    exposure = np.full(map_data_shape[:-1], 1.0)
                    lat = np.arange(-88.0, 92.0, 4.0)
                    lat_delta = np.full(lat.shape, 2.0)
                    lat_label = [f"{x} deg" for x in lat]
                    lon = np.arange(0.0, 360.0, 4.0)
                    lon_delta = np.full(lon.shape, 2.0)
                    lon_label = [f"{x} deg" for x in lon]

                    obs_date = obs_date_in_cdf
                    obs_date_range = np.full(ena_intensity.shape, ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2)
                    solid_angle = build_solid_angle_map(4)
                    solid_angle = solid_angle[np.newaxis, ...]

                    cdf.new("epoch", epoch)
                    cdf.new("energy", energy, recVary=False)
                    cdf.new("latitude", lat, recVary=False)
                    cdf.new("latitude_delta", lat_delta, recVary=False)
                    cdf.new("latitude_label", lat_label, recVary=False)
                    cdf.new("longitude", lon, recVary=False)
                    cdf.new("longitude_delta", lon_delta, recVary=False)
                    cdf.new("longitude_label", lon_label, recVary=False)
                    cdf.new("ena_intensity", ena_intensity, recVary=True)
                    cdf.new("ena_intensity_stat_uncert", ena_intensity_stat_uncert, recVary=True)
                    cdf.new("ena_intensity_sys_err", ena_intensity_sys_err, recVary=True)
                    cdf.new("exposure_factor", exposure, recVary=True)
                    cdf.new("obs_date", obs_date, recVary=True)
                    cdf.new("obs_date_range", obs_date_range, recVary=True)
                    cdf.new("solid_angle", solid_angle, recVary=True)
                    cdf.new("epoch_delta", epoch_delta, recVary=True)
                    cdf.new("energy_delta_plus", energy_delta_plus, recVary=False)
                    cdf.new("energy_delta_minus", energy_delta_minus, recVary=False)
                    cdf.new("energy_label", energy_label, recVary=False)
                    cdf.new("survival_probability", survival_probability, recVary=False)
                    if include_bg:
                        cdf.new("bg_intensity", bg_intensity, recVary=True)
                        cdf.new("bg_intensity_stat_uncert", bg_intensity_stat_unc, recVary=True)
                        cdf.new("bg_intensity_sys_err", bg_intensity_sys_err, recVary=True)

                    if include_ena_sys_err_minus:
                        cdf.new("ena_intensity_sys_err_minus", ena_intensity_sys_err_minus, recVary=True)
                        cdf.new("ena_intensity_sys_err_plus", ena_intensity_sys_err_plus, recVary=True)

                    for var in cdf:
                        cdf[var].attrs['FILLVAL'] = 1000000

                    cdf["obs_date"].attrs["FILLVAL"] = obs_date_fillval

                for path in [pathname, Path(pathname)]:
                    with self.subTest(name=test_name, path=path):
                        result = RectangularIntensityMapData.read_from_path(path)
                        self.assertIsInstance(result, RectangularIntensityMapData)

                        rectangular_coords = result.coords
                        map_data = result.intensity_map_data

                        np.testing.assert_array_equal(map_data.epoch, epoch)
                        np.testing.assert_array_equal(map_data.epoch_delta, epoch_delta)
                        np.testing.assert_array_equal(map_data.energy, energy)
                        np.testing.assert_array_equal(map_data.energy_delta_plus, energy_delta_plus)
                        np.testing.assert_array_equal(
                            map_data.energy_delta_minus, energy_delta_minus
                        )
                        np.testing.assert_array_equal(
                            map_data.energy_label, energy_label
                        )
                        np.testing.assert_array_equal(map_data.latitude, lat)
                        np.testing.assert_array_equal(
                            rectangular_coords.latitude_delta, lat_delta
                        )
                        np.testing.assert_array_equal(rectangular_coords.latitude_label, lat_label)
                        np.testing.assert_array_equal(map_data.longitude, lon)
                        np.testing.assert_array_equal(rectangular_coords.longitude_delta, lon_delta)
                        np.testing.assert_array_equal(rectangular_coords.longitude_label, lon_label)
                        np.testing.assert_array_equal(
                            map_data.ena_intensity, ena_intensity
                        )
                        np.testing.assert_array_equal(
                            map_data.ena_intensity_stat_uncert,
                            ena_intensity_stat_uncert,
                        )
                        np.testing.assert_array_equal(map_data.ena_intensity_sys_err, ena_intensity_sys_err)
                        np.testing.assert_array_equal(map_data.exposure_factor, exposure)
                        np.testing.assert_array_equal(map_data.obs_date.data, expected_obs_date)
                        np.testing.assert_array_equal(map_data.obs_date.mask, expected_obs_date_mask)
                        np.testing.assert_array_equal(map_data.obs_date_range, obs_date_range)
                        np.testing.assert_array_equal(map_data.solid_angle, solid_angle)
                        np.testing.assert_array_equal(
                            map_data.survival_probability, survival_probability
                        )
                        if include_bg:
                            np.testing.assert_array_equal(
                                map_data.bg_intensity, bg_intensity
                            )
                            np.testing.assert_array_equal(
                                map_data.bg_intensity_sys_err, bg_intensity_sys_err
                            )
                            np.testing.assert_array_equal(
                                map_data.bg_intensity_stat_uncert, bg_intensity_stat_unc
                            )
                        else:
                            self.assertIsNone(map_data.bg_intensity)
                            self.assertIsNone(map_data.bg_intensity_sys_err)
                            self.assertIsNone(map_data.bg_intensity_stat_uncert)
                        if include_ena_sys_err_minus:
                            np.testing.assert_array_equal(
                                map_data.ena_intensity_sys_err_minus,
                                ena_intensity_sys_err_minus,
                            )
                            np.testing.assert_array_equal(
                                map_data.ena_intensity_sys_err_plus,
                                ena_intensity_sys_err_plus,
                            )
                        else:
                            self.assertIsNone(map_data.ena_intensity_sys_err_minus)
                            self.assertIsNone(map_data.ena_intensity_sys_err_plus)

    def test_read_intensity_map_with_rectangular_coords_data_from_cdf_missing_flags(
        self,
    ):

        rng = np.random.default_rng()

        map_data_shape = (1, 9, 90, 45)
        obs_date_fillval = -sys.maxsize - 1
        obs_date_in_cdf = (np.full(map_data_shape, 1e9, dtype=int),)
        expected_obs_date = (
            np.full(map_data_shape, TT2000_EPOCH) + timedelta(seconds=1),
        )
        expected_obs_date_mask = (np.full(map_data_shape, False),)

        input_quality_flags = np.full(map_data_shape, MapL3Flags.NONE)
        input_quality_flags[0, :, 83, :] = MapL3Flags.PREDICTIVE_EPHEMERIS
        input_quality_flags[0, :, 84, :] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        input_quality_flags[0, :, 85, :] = MapL3Flags.PERSISTED_LAST_POINT
        input_quality_flags[0, :, 86, :] = MapL3Flags.PREDICTIVE_EPHEMERIS | MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PERSISTED_LAST_POINT

        cases = (
            ("includes quality flags", input_quality_flags, input_quality_flags),
            ("does not include quality flags", None, np.full(map_data_shape, MapL3Flags.NONE)),
        )
        for (
            case_name,
            input_quality_flags,
            expected_quality_flags
        ) in cases:
            with tempfile.TemporaryDirectory() as temp_dir, self.subTest(case_name):
                pathname = os.path.join(temp_dir, "test_cdf")
                with CDF(pathname, "") as cdf:
                    cdf.col_major(True)

                    ena_intensity = rng.random(map_data_shape)
                    bg_intensity = ena_intensity * 0.01
                    energy = rng.random(9)
                    energy_delta_plus = rng.random(9)
                    energy_delta_minus = rng.random(9)
                    energy_label = energy.astype(str)
                    ena_intensity_stat_uncert = rng.random(map_data_shape)
                    ena_intensity_sys_err = rng.random(map_data_shape)

                    ena_intensity_sys_err_minus = rng.random(map_data_shape)
                    ena_intensity_sys_err_plus = rng.random(map_data_shape)

                    bg_intensity_stat_unc = rng.random(map_data_shape)
                    bg_intensity_sys_err = rng.random(map_data_shape)

                    survival_probability = rng.random(map_data_shape)

                    epoch = np.array([datetime.now()])
                    epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
                    exposure = np.full(map_data_shape[:-1], 1.0)
                    lat = np.arange(-88.0, 92.0, 4.0)
                    lat_delta = np.full(lat.shape, 2.0)
                    lat_label = [f"{x} deg" for x in lat]
                    lon = np.arange(0.0, 360.0, 4.0)
                    lon_delta = np.full(lon.shape, 2.0)
                    lon_label = [f"{x} deg" for x in lon]

                    obs_date = obs_date_in_cdf
                    obs_date_range = np.full(
                        ena_intensity.shape,
                        ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2,
                    )
                    solid_angle = build_solid_angle_map(4)
                    solid_angle = solid_angle[np.newaxis, ...]

                    cdf.new("epoch", epoch)
                    cdf.new("energy", energy, recVary=False)
                    cdf.new("latitude", lat, recVary=False)
                    cdf.new("latitude_delta", lat_delta, recVary=False)
                    cdf.new("latitude_label", lat_label, recVary=False)
                    cdf.new("longitude", lon, recVary=False)
                    cdf.new("longitude_delta", lon_delta, recVary=False)
                    cdf.new("longitude_label", lon_label, recVary=False)
                    cdf.new("ena_intensity", ena_intensity, recVary=True)
                    cdf.new(
                        "ena_intensity_stat_uncert",
                        ena_intensity_stat_uncert,
                        recVary=True,
                    )
                    cdf.new(
                        "ena_intensity_sys_err", ena_intensity_sys_err, recVary=True
                    )
                    cdf.new("exposure_factor", exposure, recVary=True)
                    cdf.new("obs_date", obs_date, recVary=True)
                    cdf.new("obs_date_range", obs_date_range, recVary=True)
                    cdf.new("solid_angle", solid_angle, recVary=True)
                    cdf.new("epoch_delta", epoch_delta, recVary=True)
                    cdf.new("energy_delta_plus", energy_delta_plus, recVary=False)
                    cdf.new("energy_delta_minus", energy_delta_minus, recVary=False)
                    cdf.new("energy_label", energy_label, recVary=False)
                    cdf.new("survival_probability", survival_probability, recVary=False)
                    cdf.new("bg_intensity", bg_intensity, recVary=True)
                    cdf.new(
                        "bg_intensity_stat_uncert", bg_intensity_stat_unc, recVary=True
                    )
                    cdf.new("bg_intensity_sys_err", bg_intensity_sys_err, recVary=True)
                    cdf.new(
                        "ena_intensity_sys_err_minus",
                        ena_intensity_sys_err_minus,
                        recVary=True,
                    )
                    cdf.new(
                        "ena_intensity_sys_err_plus",
                        ena_intensity_sys_err_plus,
                        recVary=True,
                    )
                    if input_quality_flags is not None:
                        cdf.new(QUALITY_FLAGS_VAR_NAME, input_quality_flags, recVary=True)

                    for var in cdf:
                        cdf[var].attrs["FILLVAL"] = 1000000

                    cdf["obs_date"].attrs["FILLVAL"] = obs_date_fillval

                for path in [pathname, Path(pathname)]:
                    result = RectangularIntensityMapData.read_from_path(path)
                    self.assertIsInstance(result, RectangularIntensityMapData)

                    rectangular_coords = result.coords
                    map_data = result.intensity_map_data

                    np.testing.assert_array_equal(map_data.epoch, epoch)
                    np.testing.assert_array_equal(map_data.epoch_delta, epoch_delta)
                    np.testing.assert_array_equal(map_data.energy, energy)
                    np.testing.assert_array_equal(
                        map_data.energy_delta_plus, energy_delta_plus
                    )
                    np.testing.assert_array_equal(
                        map_data.energy_delta_minus, energy_delta_minus
                    )
                    np.testing.assert_array_equal(map_data.energy_label, energy_label)
                    np.testing.assert_array_equal(map_data.latitude, lat)
                    np.testing.assert_array_equal(
                        rectangular_coords.latitude_delta, lat_delta
                    )
                    np.testing.assert_array_equal(
                        rectangular_coords.latitude_label, lat_label
                    )
                    np.testing.assert_array_equal(map_data.longitude, lon)
                    np.testing.assert_array_equal(
                        rectangular_coords.longitude_delta, lon_delta
                    )
                    np.testing.assert_array_equal(
                        rectangular_coords.longitude_label, lon_label
                    )
                    np.testing.assert_array_equal(map_data.ena_intensity, ena_intensity)
                    np.testing.assert_array_equal(
                        map_data.ena_intensity_stat_uncert, ena_intensity_stat_uncert
                    )
                    np.testing.assert_array_equal(
                        map_data.ena_intensity_sys_err, ena_intensity_sys_err
                    )
                    np.testing.assert_array_equal(map_data.exposure_factor, exposure)
                    np.testing.assert_array_equal(
                        map_data.obs_date.data, expected_obs_date
                    )
                    np.testing.assert_array_equal(
                        map_data.obs_date.mask, expected_obs_date_mask
                    )
                    np.testing.assert_array_equal(
                        map_data.obs_date_range, obs_date_range
                    )
                    np.testing.assert_array_equal(map_data.solid_angle, solid_angle)
                    np.testing.assert_array_equal(
                        map_data.survival_probability, survival_probability
                    )
                    np.testing.assert_array_equal(map_data.bg_intensity, bg_intensity)
                    np.testing.assert_array_equal(map_data.bg_intensity_sys_err, bg_intensity_sys_err)
                    np.testing.assert_array_equal(map_data.bg_intensity_stat_uncert, bg_intensity_stat_unc)
                    np.testing.assert_array_equal(map_data.ena_intensity_sys_err_minus,
                                                  ena_intensity_sys_err_minus)
                    np.testing.assert_array_equal(map_data.ena_intensity_sys_err_plus,
                                                  ena_intensity_sys_err_plus)
                    np.testing.assert_array_equal(map_data.quality_flags, expected_quality_flags, strict=True)

    def test_fill_values_in_read_rectangular_intensity_map_data_from_cdf(self):
        path = get_test_data_folder() / "hi" / "map_with_fill_values.cdf"
        result = RectangularIntensityMapData.read_from_path(path)
        map_data = result.intensity_map_data

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(
                map_data.epoch,
                cdf["epoch"],
            )

            self.assertTrue(np.all(map_data.epoch_delta.mask))
            self.assertTrue(np.all(map_data.obs_date.mask))
            self.assertTrue(np.all(map_data.obs_date_range.mask))

            np.testing.assert_array_equal(map_data.energy, np.full_like(cdf["energy"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_plus, np.full_like(cdf["energy_delta_plus"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_minus, np.full_like(cdf["energy_delta_minus"], np.nan))
            np.testing.assert_array_equal(map_data.latitude, np.full_like(cdf["latitude"], np.nan))
            np.testing.assert_array_equal(
                map_data.longitude, np.full_like(cdf["longitude"], np.nan)
            )
            np.testing.assert_array_equal(
                map_data.ena_intensity, np.full_like(cdf["ena_intensity"], np.nan)
            )
            np.testing.assert_array_equal(
                map_data.ena_intensity_stat_uncert,
                np.full_like(cdf["ena_intensity_stat_uncert"], np.nan),
            )
            np.testing.assert_array_equal(map_data.ena_intensity_sys_err,
                                          np.full_like(cdf["ena_intensity_sys_err"], np.nan))
            np.testing.assert_array_equal(map_data.exposure_factor, np.full_like(cdf["exposure_factor"], np.nan))
            np.testing.assert_array_equal(map_data.solid_angle, np.full_like(cdf["solid_angle"], np.nan))
            np.testing.assert_array_equal(map_data.bg_intensity, np.full_like(cdf["bg_intensity"], np.nan))
            np.testing.assert_array_equal(map_data.bg_intensity_sys_err,
                                          np.full_like(cdf["bg_intensity_sys_err"], np.nan))
            np.testing.assert_array_equal(
                map_data.bg_intensity_stat_uncert,
                np.full_like(cdf["bg_intensity_stat_uncert"], np.nan),
            )
            np.testing.assert_array_equal(
                map_data.survival_probability,
                np.full_like(cdf["survival_probability"], np.nan),
            )
            np.testing.assert_array_equal(
                map_data.ena_intensity_sys_err_plus, np.full_like(cdf["ena_intensity_sys_err_plus"], np.nan)
            )
            np.testing.assert_array_equal(
                map_data.ena_intensity_sys_err_minus, np.full_like(cdf["ena_intensity_sys_err_minus"], np.nan)
            )

    def test_healpix_intensity_map_data_to_skymap(self):
        expected_nside = 2

        num_epochs = 1
        num_energies = 5
        num_healpix_indices = 12 * (expected_nside ** 2)

        fake_data_per_energy_per_pixel = np.arange(num_epochs * num_energies * num_healpix_indices) \
            .reshape(num_epochs, num_energies, num_healpix_indices)

        fake_data_per_pixel = np.arange(num_healpix_indices)

        quality_flags = np.full_like(fake_data_per_energy_per_pixel, MapL3Flags.NONE)
        quality_flags[0, :, 43] = MapL3Flags.PREDICTIVE_EPHEMERIS
        quality_flags[0, :, 37] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        quality_flags[0, :, 18] = MapL3Flags.PERSISTED_LAST_POINT

        expected_pred_ephem = quality_flags & MapL3Flags.PREDICTIVE_EPHEMERIS
        expected_nominal_alpha_proton = quality_flags & MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        expected_persisted_last_point = quality_flags & MapL3Flags.PERSISTED_LAST_POINT

        intensity_map_data = IntensityMapData(
            epoch=np.array([np.datetime64('1970-01-01T00:00:00')]),
            epoch_delta=np.array([999]),
            energy=np.arange(num_energies) * 1.1,
            energy_delta_plus=np.arange(num_energies) * 1.2,
            energy_delta_minus=np.arange(num_energies) * 1.3,
            energy_label=np.arange(num_energies).astype(str),
            latitude=fake_data_per_pixel * 2.2,
            longitude=fake_data_per_pixel * 2.3,
            exposure_factor=fake_data_per_energy_per_pixel * 2.2,
            obs_date=np.datetime64('1970-01-01T00:00:00') + fake_data_per_energy_per_pixel * 10000,
            obs_date_range=fake_data_per_energy_per_pixel * 3.2,
            solid_angle=fake_data_per_pixel * 3.4,
            ena_intensity=fake_data_per_energy_per_pixel * 2.2,
            ena_intensity_stat_uncert=fake_data_per_energy_per_pixel * 2.3,
            ena_intensity_sys_err=fake_data_per_energy_per_pixel * 2.4,
            survival_probability=fake_data_per_energy_per_pixel * 0.01,
            quality_flags=quality_flags,
        )

        healpix_intensity_map_data = HealPixIntensityMapData(
            intensity_map_data=intensity_map_data,
            coords=HealPixCoords(
                pixel_index=np.arange(num_healpix_indices),
                pixel_index_label=np.arange(num_healpix_indices).astype(str),
            )
        )

        skymap = healpix_intensity_map_data.to_healpix_skymap()

        self.assertIsInstance(skymap, HealpixSkyMap)

        self.assertEqual(expected_nside, skymap.nside)
        actual_dataset = skymap.data_1d

        # @formatter:off
        np.testing.assert_array_equal(actual_dataset.coords[CoordNames.TIME.value].values, intensity_map_data.epoch)
        np.testing.assert_array_equal(actual_dataset.coords[CoordNames.ENERGY_L2.value].values, intensity_map_data.energy)
        np.testing.assert_array_equal(actual_dataset.coords[CoordNames.GENERIC_PIXEL.value].values, healpix_intensity_map_data.coords.pixel_index)

        np.testing.assert_array_equal(actual_dataset.data_vars["latitude"].values, intensity_map_data.latitude)
        np.testing.assert_array_equal(actual_dataset.data_vars["longitude"].values, intensity_map_data.longitude)
        np.testing.assert_array_equal(actual_dataset.data_vars["solid_angle"].values, intensity_map_data.solid_angle)

        np.testing.assert_array_equal(actual_dataset.data_vars["obs_date_range"].values, intensity_map_data.obs_date_range)
        np.testing.assert_array_equal(actual_dataset.data_vars["obs_date"].values, intensity_map_data.obs_date.astype(np.float64))

        np.testing.assert_array_equal(actual_dataset.data_vars["exposure_factor"].values, intensity_map_data.exposure_factor)
        np.testing.assert_array_equal(actual_dataset.data_vars["ena_intensity"].values, intensity_map_data.ena_intensity)
        np.testing.assert_array_equal(actual_dataset.data_vars["ena_intensity_stat_uncert"].values, intensity_map_data.ena_intensity_stat_uncert)
        np.testing.assert_array_equal(actual_dataset.data_vars["ena_intensity_sys_err"].values, intensity_map_data.ena_intensity_sys_err)

        np.testing.assert_array_equal(actual_dataset.data_vars["survival_probability"].values, intensity_map_data.survival_probability)
        np.testing.assert_array_equal(actual_dataset.data_vars["predicted_ephemeris_flag"].values, expected_pred_ephem)
        np.testing.assert_array_equal(actual_dataset.data_vars["nominal_alpha_proton_ratio_flag"].values, expected_nominal_alpha_proton)
        np.testing.assert_array_equal(actual_dataset.data_vars["persisted_last_point_flag"].values, expected_persisted_last_point)

        for key in [ "obs_date", "obs_date_range", "exposure_factor", "ena_intensity", "ena_intensity_stat_uncert", "ena_intensity_sys_err", "survival_probability", "predicted_ephemeris_flag", "nominal_alpha_proton_ratio_flag", "persisted_last_point_flag" ]:
            self.assertEqual((CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.GENERIC_PIXEL.value), actual_dataset.data_vars[key].dims)
        for key in [ "latitude", "longitude", "solid_angle" ]:
            self.assertEqual((CoordNames.GENERIC_PIXEL.value,), actual_dataset.data_vars[key].dims)
        # @formatter:on

    def test_healpix_intensity_map_data_to_skymap_with_no_sp(self):
        expected_nside = 2

        num_epochs = 1
        num_energies = 5
        num_healpix_indices = 12 * (expected_nside**2)

        fake_data_per_energy_per_pixel = np.arange(
            num_epochs * num_energies * num_healpix_indices
        ).reshape(num_epochs, num_energies, num_healpix_indices)

        fake_data_per_pixel = np.arange(num_healpix_indices)

        intensity_map_data = IntensityMapData(
            epoch=np.array([np.datetime64("1970-01-01T00:00:00")]),
            epoch_delta=np.array([999]),
            energy=np.arange(num_energies) * 1.1,
            energy_delta_plus=np.arange(num_energies) * 1.2,
            energy_delta_minus=np.arange(num_energies) * 1.3,
            energy_label=np.arange(num_energies).astype(str),
            latitude=fake_data_per_pixel * 2.2,
            longitude=fake_data_per_pixel * 2.3,
            exposure_factor=fake_data_per_energy_per_pixel * 2.2,
            obs_date=np.datetime64("1970-01-01T00:00:00")
            + fake_data_per_energy_per_pixel * 10000,
            obs_date_range=fake_data_per_energy_per_pixel * 3.2,
            solid_angle=fake_data_per_pixel * 3.4,
            ena_intensity=fake_data_per_energy_per_pixel * 2.2,
            ena_intensity_stat_uncert=fake_data_per_energy_per_pixel * 2.3,
            ena_intensity_sys_err=fake_data_per_energy_per_pixel * 2.4,
            quality_flags=np.full_like(fake_data_per_energy_per_pixel, MapL3Flags.NONE),
        )

        healpix_intensity_map_data = HealPixIntensityMapData(
            intensity_map_data=intensity_map_data,
            coords=HealPixCoords(
                pixel_index=np.arange(num_healpix_indices),
                pixel_index_label=np.arange(num_healpix_indices).astype(str),
            ),
        )

        skymap = healpix_intensity_map_data.to_healpix_skymap()

        self.assertNotIn("survival_probability", skymap.data_1d)

    def test_read_intensity_data_handles_missing_obs_date_and_missing_sp(self):
        cdf = CDF(
            str(
                get_integration_test_data_path(
                    "lo/multiple_arcs/imap_lo_l2_l090-ena-h-hf-nsp-ram-hae-6deg-1yr_20250415_v005.cdf"
                )
            )
        )

        intensity_map_data = _read_intensity_map_data_from_open_cdf(cdf)

        self.assertIsNotNone(intensity_map_data.obs_date)
        self.assertIsNone(intensity_map_data.survival_probability)

    def test_calculate_datetime_weighted_average(self):

        map_1_obs_date = np.ma.array(data=[
            datetime(2000, 1, 1),
            datetime(2000, 1, 1),
            datetime(9999, 12, 31)],
            mask=[False, False, True])
        map_2_obs_date = np.ma.array(data=[
            datetime(9999, 12, 31),
            datetime(2000, 1, 4),
            datetime(9999, 12, 31)],
            mask=[True, False, True])

        map_1_weights = [1, 1, 1]
        map_2_weights = [1, 2, 1]

        weights = np.array([map_1_weights, map_2_weights])
        original_weights = weights.copy()
        average_obs_date: np.ma.masked_array = calculate_datetime_weighted_average(
            np.ma.masked_array([map_1_obs_date, map_2_obs_date]),
            weights, axis=0)

        self.assertIsInstance(average_obs_date, np.ma.masked_array)

        self.assertEqual(datetime(2000, 1, 1), average_obs_date[0])
        self.assertFalse(average_obs_date.mask[0])

        self.assertEqual(datetime(2000, 1, 3), average_obs_date[1])
        self.assertFalse(average_obs_date.mask[1])

        self.assertEqual(TT2000_EPOCH, average_obs_date.data[2])
        self.assertTrue(average_obs_date.mask[2])

        np.testing.assert_equal(weights, original_weights, "should not mutate the input weights")


    def test_lo_isn_data_read_from_path(self):
        path_to_cdf = get_test_data_folder() / 'lo' / 'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr_20250415_v000'

        actual = ISNRateData.read_from_path(path_to_cdf)

        with CDF(str(path_to_cdf)) as expected:
            np.testing.assert_array_equal(actual.epoch, expected['epoch'][...])

            np.testing.assert_array_equal(actual.bg_rate, read_numeric_variable(expected['bg_rate']))
            np.testing.assert_array_equal(actual.bg_rate_stat_uncert,
                                          read_numeric_variable(expected['bg_rate_stat_uncert']))
            np.testing.assert_array_equal(actual.bg_rate_sys_err, read_numeric_variable(expected['bg_rate_sys_err']))
            np.testing.assert_array_equal(actual.ena_count_rate, read_numeric_variable(expected['ena_count_rate']))
            np.testing.assert_array_equal(actual.ena_count_rate_stat_uncert,
                                          read_numeric_variable(expected['ena_count_rate_stat_uncert']))
            np.testing.assert_array_equal(actual.ena_intensity, read_numeric_variable(expected['ena_intensity']))
            np.testing.assert_array_equal(actual.ena_intensity_stat_uncert,
                                          read_numeric_variable(expected['ena_intensity_stat_uncert']))
            np.testing.assert_array_equal(actual.ena_intensity_sys_err,
                                          read_numeric_variable(expected['ena_intensity_sys_err']))
            np.testing.assert_array_equal(actual.energy, expected['energy'][...])
            np.testing.assert_array_equal(actual.exposure_factor,
                                          read_numeric_variable(expected['exposure_factor']))
            np.testing.assert_array_equal(actual.latitude_delta, expected['latitude_delta'][...])
            np.testing.assert_array_equal(actual.latitude, expected['latitude'][...])
            np.testing.assert_array_equal(actual.longitude_delta, expected['longitude_delta'][...])
            np.testing.assert_array_equal(actual.longitude, expected['longitude'][...])
            np.testing.assert_array_equal(actual.solid_angle, read_numeric_variable(expected['solid_angle']))


if __name__ == '__main__':
    unittest.main()
