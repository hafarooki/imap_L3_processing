import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel, Mock, MagicMock, patch

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, ENERGY_VAR_NAME, \
    PROBABILITY_OF_SURVIVAL_VAR_NAME, HEALPIX_INDEX_VAR_NAME, ENERGY_LABEL_VAR_NAME, PIXEL_INDEX_LABEL_VAR_NAME, \
    SPIN_AXIS_LATITUDE_VAR_NAME, SPIN_AXIS_LONGITUDE_VAR_NAME, PROGRAM_VERSION_VAR_NAME, SPACECRAFT_RADIUS_VAR_NAME, \
    SPACECRAFT_LONGITUDE_VAR_NAME, SPACECRAFT_LATITUDE_VAR_NAME, SPACECRAFT_VELOCITY_X_VAR_NAME, \
    SPACECRAFT_VELOCITY_Y_VAR_NAME, SPACECRAFT_VELOCITY_Z_VAR_NAME, ELONGATION_EXCLUDED_VAR_NAME, \
    PIXEL_LATITUDE_VAR_NAME, PIXEL_LONGITUDE_VAR_NAME, GLOWS_FLAGS_VAR_NAME, ENERGY_DELTA_PLUS_VAR_NAME, \
    ENERGY_DELTA_MINUS_VAR_NAME
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import get_test_instrument_team_data_path, NumpyArrayMatcher


class TestL3eUltraModel(unittest.TestCase):
    def test_l3e_ultra_model_to_data_product_variables(self):
        test_cases = {
            'case 1': (
                [1.2234, 2, 4.51, 66.7666],
                [1, 41650233.0, 4.22, 9.5],
                ['1.22', '2.00', '4.51', '66.77'],
                ['1', '41650233', '4', '10']
            ),
            'case 2': (
                [4.536, 48.193, 4253.1],
                [13, 14.0, 34.2, 19.5],
                ['4.54', '48.19', '4253.10'],
                ['13', '14', '34', '20']
            )
        }

        for name, (energy_array, healpix_index_array, expected_energy_labels,
                   expected_healpix_labels) in test_cases.items():
            with self.subTest(name):
                l3e_ultra: GlowsL3EUltraData = GlowsL3EUltraData(
                    Mock(),
                    sentinel.epoch,
                    sentinel.epoch_delta,
                    energy_array,
                    sentinel.energy_delta_plus,
                    sentinel.energy_delta_minus,
                    healpix_index_array,
                    sentinel.probability_of_survival,
                    sentinel.spin_axis_latitude,
                    sentinel.spin_axis_longitude,
                    sentinel.program_version,
                    sentinel.spacecraft_radius,
                    sentinel.spacecraft_latitude,
                    sentinel.spacecraft_longitude,
                    sentinel.spacecraft_velocity_x,
                    sentinel.spacecraft_velocity_y,
                    sentinel.spacecraft_velocity_z,
                    sentinel.elongation_excluded,
                    sentinel.pixel_latitude,
                    sentinel.pixel_longitude,
                    sentinel.glows_flags,
                )
                data_products = l3e_ultra.to_data_product_variables()

                expected_data_products = [
                    DataProductVariable(EPOCH_CDF_VAR_NAME, sentinel.epoch),
                    DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, sentinel.epoch_delta),
                    DataProductVariable(ENERGY_VAR_NAME, energy_array),
                    DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
                    DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
                    DataProductVariable(HEALPIX_INDEX_VAR_NAME, healpix_index_array),
                    DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, sentinel.probability_of_survival),
                    DataProductVariable(ENERGY_LABEL_VAR_NAME, expected_energy_labels),
                    DataProductVariable(PIXEL_INDEX_LABEL_VAR_NAME, expected_healpix_labels),
                    DataProductVariable(SPIN_AXIS_LATITUDE_VAR_NAME, sentinel.spin_axis_latitude),
                    DataProductVariable(SPIN_AXIS_LONGITUDE_VAR_NAME, sentinel.spin_axis_longitude),
                    DataProductVariable(PROGRAM_VERSION_VAR_NAME, sentinel.program_version),
                    DataProductVariable(SPACECRAFT_RADIUS_VAR_NAME, sentinel.spacecraft_radius),
                    DataProductVariable(SPACECRAFT_LATITUDE_VAR_NAME, sentinel.spacecraft_latitude),
                    DataProductVariable(SPACECRAFT_LONGITUDE_VAR_NAME, sentinel.spacecraft_longitude),
                    DataProductVariable(SPACECRAFT_VELOCITY_X_VAR_NAME, sentinel.spacecraft_velocity_x),
                    DataProductVariable(SPACECRAFT_VELOCITY_Y_VAR_NAME, sentinel.spacecraft_velocity_y),
                    DataProductVariable(SPACECRAFT_VELOCITY_Z_VAR_NAME, sentinel.spacecraft_velocity_z),
                    DataProductVariable(ELONGATION_EXCLUDED_VAR_NAME, sentinel.elongation_excluded),
                    DataProductVariable(PIXEL_LATITUDE_VAR_NAME, sentinel.pixel_latitude),
                    DataProductVariable(PIXEL_LONGITUDE_VAR_NAME, sentinel.pixel_longitude),
                    DataProductVariable(GLOWS_FLAGS_VAR_NAME, sentinel.glows_flags),
                ]

                self.assertEqual(expected_data_products, data_products)

    @patch("imap_l3_processing.glows.l3e.glows_l3e_ultra_model.calculate_energy_deltas")
    def test_convert_dat_to_glows_l3e_ul_product(self, mock_calculate_energy_deltas):
        ul_file_path = get_test_instrument_team_data_path("glows/probSur.Imap.Ul_20250420_000000_2025.300.txt")
        expected_epoch = datetime(year=2009, month=1, day=1)
        epoch_delta = timedelta(hours=10)
        expected_epoch_delta_in_nanoseconds = 10*3600*1e9

        mock_calculate_energy_deltas.return_value = sentinel.energy_delta_plus, sentinel.energy_delta_minus
        expected_energy = np.array(
            [2.3751086, 3.0917682, 4.0246710, 5.2390656, 6.8198887, 8.8777057, 11.5564435, 15.0434572, 19.5826341,
             25.4914514, 33.1831812, 43.1957952, 56.2295915, 73.1961745, 95.2822137, 124.0324417, 161.4576950,
             210.1755551, 273.5934261, 356.1468544])

        row_0_probability_of_survival = np.array(
            [0.84827693E+00, 0.86517360E+00, 0.88086645E+00, 0.89551288E+00, 0.90925039E+00, 0.92211605E+00,
             0.93428862E+00, 0.94576000E+00, 0.95660359E+00, 0.96660933E+00, 0.97533752E+00, 0.98226689E+00,
             0.98714107E+00, 0.99025210E+00, 0.99218849E+00, 0.99346478E+00, 0.99439230E+00, 0.99512632E+00,
             0.99574163E+00, 0.99627086E+00])

        row_804_probability_of_survival = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan])

        row_3071_expected_probability_of_survival = np.array(
            [0.84498413E+00, 0.86231633E+00, 0.87840630E+00, 0.89344499E+00, 0.90753390E+00, 0.92079246E+00,
             0.93326641E+00, 0.94509283E+00, 0.95623858E+00, 0.96649540E+00, 0.97543873E+00, 0.98248159E+00,
             0.98738469E+00, 0.99048237E+00, 0.99239223E+00, 0.99364490E+00, 0.99455000E+00, 0.99526291E+00,
             0.99585178E+00, 0.99635733E+00])

        expected_survival_probability_shape = (1, 20, 3072)

        expected_heal_pix = np.arange(0, 3072)
        mock_metadata = Mock()

        spin_axis_lat = 45.0
        spin_axis_lon = 90.0

        args = MagicMock(spec=GlowsL3eCallArguments)
        expected_program_version = 'Ultra.v00.01'

        spacecraft_info = MagicMock(spec=GlowsL3eSpacecraftInfo)
        spacecraft_info.spin_axis_latitude = spin_axis_lat
        spacecraft_info.spin_axis_longitude = spin_axis_lon
        spacecraft_info.spacecraft_radius = .5
        spacecraft_info.spacecraft_longitude = 85.4
        spacecraft_info.spacecraft_latitude = 45.1

        spacecraft_info.spacecraft_velocity_x = 2.1
        spacecraft_info.spacecraft_velocity_y = 2.2
        spacecraft_info.spacecraft_velocity_z = 2.3

        args.spacecraft_info = spacecraft_info

        args.elongation = 30.0

        l3e_ul_product: GlowsL3EUltraData = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(mock_metadata,
                                                                                                  ul_file_path,
                                                                                                  expected_epoch,
                                                                                                  epoch_delta,
                                                                                                  args)

        mock_calculate_energy_deltas.assert_called_once_with(NumpyArrayMatcher(l3e_ul_product.energy),)

        np.testing.assert_equal([expected_epoch], l3e_ul_product.epoch, strict=True)
        np.testing.assert_equal([expected_epoch_delta_in_nanoseconds], l3e_ul_product.epoch_delta, strict=True)

        np.testing.assert_equal(l3e_ul_product.energy, expected_energy, strict=True)
        np.testing.assert_equal(l3e_ul_product.energy_delta_plus, sentinel.energy_delta_plus, strict=True)
        np.testing.assert_equal(l3e_ul_product.energy_delta_minus, sentinel.energy_delta_minus, strict=True)

        np.testing.assert_equal(expected_heal_pix, l3e_ul_product.healpix_index, strict=True)
        np.testing.assert_equal(expected_survival_probability_shape, l3e_ul_product.probability_of_survival.shape,
                                strict=True)
        np.testing.assert_equal(l3e_ul_product.probability_of_survival[0].T[0, :], row_0_probability_of_survival,
                                strict=True)
        np.testing.assert_equal(l3e_ul_product.probability_of_survival[0].T[804, :],
                                row_804_probability_of_survival, strict=True)
        np.testing.assert_equal(l3e_ul_product.probability_of_survival[0].T[3071, :],
                                row_3071_expected_probability_of_survival, strict=True)

        np.testing.assert_equal(np.array([spin_axis_lat]), l3e_ul_product.spin_axis_lat, strict=True)
        np.testing.assert_equal(np.array([spin_axis_lon]), l3e_ul_product.spin_axis_lon, strict=True)

        np.testing.assert_equal(np.array([expected_program_version]), l3e_ul_product.program_version, strict=True)

        np.testing.assert_equal(np.array([.5]), l3e_ul_product.spacecraft_radius, strict=True)
        np.testing.assert_equal(np.array([85.4]), l3e_ul_product.spacecraft_longitude, strict=True)
        np.testing.assert_equal(np.array([45.1]), l3e_ul_product.spacecraft_latitude, strict=True)

        np.testing.assert_equal(np.array([2.1]), l3e_ul_product.spacecraft_velocity_x, strict=True)
        np.testing.assert_equal(np.array([2.2]), l3e_ul_product.spacecraft_velocity_y, strict=True)
        np.testing.assert_equal(np.array([2.3]), l3e_ul_product.spacecraft_velocity_z, strict=True)

        np.testing.assert_equal(np.array([30.0]), l3e_ul_product.elongation_excluded, strict=True)

        np.testing.assert_equal((1, 3072), l3e_ul_product.pixel_latitude.shape, strict=True)
        np.testing.assert_equal((1, 3072), l3e_ul_product.pixel_longitude.shape, strict=True)
        np.testing.assert_equal(87.07582, l3e_ul_product.pixel_latitude[0, 0])
        np.testing.assert_equal(45.00000, l3e_ul_product.pixel_longitude[0, 0])

        np.testing.assert_equal(np.nan, l3e_ul_product.pixel_latitude[0, 804])
        np.testing.assert_equal(np.nan, l3e_ul_product.pixel_longitude[0, 804])

