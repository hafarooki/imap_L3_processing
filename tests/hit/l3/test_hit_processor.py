from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TypeVar
from unittest import TestCase
from unittest.mock import sentinel, patch, call, Mock

import numpy as np

from imap_l3_processing.constants import UNSIGNED_INT2_FILL_VALUE, UNSIGNED_INT1_FILL_VALUE
from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.hit.l3.models import HitL2Data, HitL1Data
from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAWord, Detector, RawPHAEvent, PHAExtendedHeader, StimBlock, \
    ExtendedStimHeader
from imap_l3_processing.hit.l3.pha.science.calculate_pha import EventOutput
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, DetectorSide, \
    DetectorRange
from imap_l3_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_l3_processing.models import MagData, InputMetadata
from imap_l3_processing.processor import Processor
from tests.test_helpers import NumpyArrayMatcher, create_dataclass_mock


class TestHitProcessor(TestCase):
    def test_is_a_processor(self):
        self.assertIsInstance(
            HitProcessor(Mock(), Mock()),
            Processor
        )

    @patch('imap_l3_processing.hit.l3.hit_processor.transform_to_10_minute_chunks')
    @patch("imap_l3_processing.utils.spiceypy")
    @patch('imap_l3_processing.hit.l3.hit_processor.save_data')
    @patch('imap_l3_processing.hit.l3.hit_processor.HITL3SectoredDependencies.fetch_dependencies')
    @patch('imap_l3_processing.hit.l3.hit_processor.calculate_unit_vector')
    @patch('imap_l3_processing.hit.l3.hit_processor.get_hit_bin_polar_coordinates')
    @patch('imap_l3_processing.hit.l3.hit_processor.get_sector_unit_vectors')
    @patch('imap_l3_processing.hit.l3.hit_processor.calculate_pitch_angle')
    @patch('imap_l3_processing.hit.l3.hit_processor.calculate_gyrophase')
    @patch('imap_l3_processing.hit.l3.hit_processor.rebin_by_pitch_angle_and_gyrophase')
    @patch('imap_l3_processing.hit.l3.hit_processor.rotate_particle_vectors_from_hit_despun_to_imap_despun')
    def test_process_pitch_angle_product(self, mock_rotate_particle_vectors_from_hit_despun_to_imap_despun,
                                         mock_hit_rebin, mock_calculate_gyrophase,
                                         mock_calculate_pitch_angle,
                                         mock_get_sector_unit_vectors, mock_get_hit_bin_polar_coordinates,
                                         mock_calculate_unit_vector,
                                         mock_fetch_dependencies, mock_save_data,
                                         mock_spiceypy, mock_transform_epochs):
        mock_spiceypy.ktotal.return_value = 0

        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="macropixel"
        )

        mock_processing_input_collection = Mock()
        mock_processing_input_collection.get_file_paths.return_value = [Path("test/parent_path")]

        epochs = np.array([123, 234])
        epoch_deltas = np.array([12, 13])
        averaged_mag_vectors = [sentinel.mag_vector1, sentinel.mag_vector2]

        mock_dependencies = Mock(spec=HITL3SectoredDependencies)
        mock_mag_data = create_dataclass_mock(MagData)
        mock_mag_data.rebin_to = Mock()
        mock_mag_data.rebin_to.return_value = averaged_mag_vectors
        mock_dependencies.mag_l1d_data = mock_mag_data
        mock_hit_data = create_dataclass_mock(HitL2Data)
        mock_hit_data.epoch = epochs
        mock_hit_data.epoch_delta = epoch_deltas

        CNO_time1 = np.full(2, 1)
        delta_plus_CNO_time1 = np.full(2, 11)
        delta_minus_CNO_time1 = np.full(2, 12)
        helium4_time1 = np.full(2, 2)
        delta_plus_helium4_time1 = np.full(2, 21)
        delta_minus_helium4_time1 = np.full(2, 22)
        hydrogen_time1 = np.full(2, 3)
        delta_plus_hydrogen_time1 = np.full(2, 31)
        delta_minus_hydrogen_time1 = np.full(2, 32)
        iron_time1 = np.full(2, 4)
        delta_plus_iron_time1 = np.full(2, 41)
        delta_minus_iron_time1 = np.full(2, 42)
        NeMgSi_time1 = np.full(2, 5)
        delta_plus_NeMgSi_time1 = np.full(2, 51)
        delta_minus_NeMgSi_time1 = np.full(2, 52)
        CNO_time2 = np.full(2, 6)
        delta_plus_CNO_time2 = np.full(2, 61)
        delta_minus_CNO_time2 = np.full(2, 62)
        helium4_time2 = np.full(2, 7)
        delta_plus_helium4_time2 = np.full(2, 71)
        delta_minus_helium4_time2 = np.full(2, 72)
        hydrogen_time2 = np.full(2, 8)
        delta_plus_hydrogen_time2 = np.full(2, 81)
        delta_minus_hydrogen_time2 = np.full(2, 82)
        iron_time2 = np.full(2, 9)
        delta_plus_iron_time2 = np.full(2, 91)
        delta_minus_iron_time2 = np.full(2, 92)
        NeMgSi_time2 = np.full(2, 10)
        delta_plus_NeMgSi_time2 = np.full(2, 101)
        delta_minus_NeMgSi_time2 = np.full(2, 102)

        mock_hit_data.cno = np.array([CNO_time1, CNO_time2])
        mock_hit_data.delta_plus_cno = np.array([delta_plus_CNO_time1, delta_plus_CNO_time2])
        mock_hit_data.delta_minus_cno = np.array([delta_minus_CNO_time1, delta_minus_CNO_time2])
        mock_hit_data.he4 = np.array([helium4_time1, helium4_time2])
        mock_hit_data.delta_plus_he4 = np.array([delta_plus_helium4_time1, delta_plus_helium4_time2])
        mock_hit_data.delta_minus_he4 = np.array([delta_minus_helium4_time1, delta_minus_helium4_time2])
        mock_hit_data.h = np.array([hydrogen_time1, hydrogen_time2])
        mock_hit_data.delta_plus_h = np.array([delta_plus_hydrogen_time1, delta_plus_hydrogen_time2])
        mock_hit_data.delta_minus_h = np.array([delta_minus_hydrogen_time1, delta_minus_hydrogen_time2])
        mock_hit_data.fe = np.array([iron_time1, iron_time2])
        mock_hit_data.delta_plus_fe = np.array([delta_plus_iron_time1, delta_plus_iron_time2])
        mock_hit_data.delta_minus_fe = np.array([delta_minus_iron_time1, delta_minus_iron_time2])
        mock_hit_data.nemgsi = np.array([NeMgSi_time1, NeMgSi_time2])
        mock_hit_data.delta_plus_nemgsi = np.array([delta_plus_NeMgSi_time1, delta_plus_NeMgSi_time2])
        mock_hit_data.delta_minus_nemgsi = np.array([delta_minus_NeMgSi_time1, delta_minus_NeMgSi_time2])

        mock_dependencies.data = sentinel.pre_transform_hit_data
        mock_fetch_dependencies.return_value = mock_dependencies

        mock_transform_epochs.return_value = mock_hit_data

        mock_calculate_unit_vector.side_effect = [sentinel.mag_unit_vector1, sentinel.mag_unit_vector2]
        mock_get_hit_bin_polar_coordinates.return_value = (
            sentinel.pitch_angles, sentinel.gyrophases, sentinel.pitch_angle_deltas, sentinel.gyrophase_deltas)

        sector_unit_vectors_zenith_first = np.full((8, 15, 3), np.sqrt(1 / 3))
        mock_get_sector_unit_vectors.return_value = sector_unit_vectors_zenith_first
        expected_sector_unit_vectors = np.full((15, 8, 3), np.sqrt(1 / 3))

        mock_rotate_particle_vectors_from_hit_despun_to_imap_despun.return_value = sentinel.rotated_particle_vector
        mock_calculate_pitch_angle.side_effect = [sentinel.pitch_angle1, sentinel.pitch_angle2]
        mock_calculate_gyrophase.side_effect = [sentinel.gyrophase1, sentinel.gyrophase2]

        rebinned_pa_gyro_CNO_time1 = np.full((2, 8, 15), 1)
        rebinned_pa_gyro_CNO_delta_plus_time1 = np.full((2, 8, 15), 100)
        rebinned_pa_gyro_CNO_delta_minus_time1 = np.full((2, 8, 15), 101)
        rebinned_pa_gyro_helium4_time1 = np.full((2, 8, 15), 3)
        rebinned_pa_gyro_helium4_delta_plus_time1 = np.full((2, 8, 15), 300)
        rebinned_pa_gyro_helium4_delta_minus_time1 = np.full((2, 8, 15), 301)
        rebinned_pa_gyro_hydrogen_time1 = np.full((2, 8, 15), 5)
        rebinned_pa_gyro_hydrogen_delta_plus_time1 = np.full((2, 8, 15), 500)
        rebinned_pa_gyro_hydrogen_delta_minus_time1 = np.full((2, 8, 15), 501)
        rebinned_pa_gyro_iron_time1 = np.full((2, 8, 15), 7)
        rebinned_pa_gyro_iron_delta_plus_time1 = np.full((2, 8, 15), 700)
        rebinned_pa_gyro_iron_delta_minus_time1 = np.full((2, 8, 15), 701)
        rebinned_pa_gyro_NeMgSi_time1 = np.full((2, 8, 15), 9)
        rebinned_pa_gyro_NeMgSi_delta_plus_time1 = np.full((2, 8, 15), 900)
        rebinned_pa_gyro_NeMgSi_delta_minus_time1 = np.full((2, 8, 15), 901)
        rebinned_pa_gyro_CNO_time2 = np.full((2, 8, 15), 2)
        rebinned_pa_gyro_CNO_delta_plus_time2 = np.full((2, 8, 15), 200)
        rebinned_pa_gyro_CNO_delta_minus_time2 = np.full((2, 8, 15), 201)
        rebinned_pa_gyro_helium4_time2 = np.full((2, 8, 15), 4)
        rebinned_pa_gyro_helium4_delta_plus_time2 = np.full((2, 8, 15), 400)
        rebinned_pa_gyro_helium4_delta_minus_time2 = np.full((2, 8, 15), 401)
        rebinned_pa_gyro_hydrogen_time2 = np.full((2, 8, 15), 6)
        rebinned_pa_gyro_hydrogen_delta_plus_time2 = np.full((2, 8, 15), 600)
        rebinned_pa_gyro_hydrogen_delta_minus_time2 = np.full((2, 8, 15), 601)
        rebinned_pa_gyro_iron_time2 = np.full((2, 8, 15), 8)
        rebinned_pa_gyro_iron_delta_plus_time2 = np.full((2, 8, 15), 800)
        rebinned_pa_gyro_iron_delta_minus_time2 = np.full((2, 8, 15), 801)
        rebinned_pa_gyro_NeMgSi_time2 = np.full((2, 8, 15), 10)
        rebinned_pa_gyro_NeMgSi_delta_plus_time2 = np.full((2, 8, 15), 1000)
        rebinned_pa_gyro_NeMgSi_delta_minus_time2 = np.full((2, 8, 15), 1001)

        rebinned_pa_CNO_time1 = np.full((2, 8), 11)
        rebinned_pa_CNO_delta_plus_time1 = np.full((2, 8), 1100)
        rebinned_pa_CNO_delta_minus_time1 = np.full((2, 8), 1101)
        rebinned_pa_helium4_time1 = np.full((2, 8), 13)
        rebinned_pa_helium4_delta_plus_time1 = np.full((2, 8), 1300)
        rebinned_pa_helium4_delta_minus_time1 = np.full((2, 8), 1301)
        rebinned_pa_hydrogen_time1 = np.full((2, 8), 15)
        rebinned_pa_hydrogen_delta_plus_time1 = np.full((2, 8), 1500)
        rebinned_pa_hydrogen_delta_minus_time1 = np.full((2, 8), 1501)
        rebinned_pa_iron_time1 = np.full((2, 8), 17)
        rebinned_pa_iron_delta_plus_time1 = np.full((2, 8), 1700)
        rebinned_pa_iron_delta_minus_time1 = np.full((2, 8), 1701)
        rebinned_pa_NeMgSi_time1 = np.full((2, 8), 19)
        rebinned_pa_NeMgSi_delta_plus_time1 = np.full((2, 8), 1900)
        rebinned_pa_NeMgSi_delta_minus_time1 = np.full((2, 8), 1901)
        rebinned_pa_CNO_time2 = np.full((2, 8), 12)
        rebinned_pa_CNO_delta_plus_time2 = np.full((2, 8), 1200)
        rebinned_pa_CNO_delta_minus_time2 = np.full((2, 8), 1201)
        rebinned_pa_helium4_time2 = np.full((2, 8), 14)
        rebinned_pa_helium4_delta_plus_time2 = np.full((2, 8), 1400)
        rebinned_pa_helium4_delta_minus_time2 = np.full((2, 8), 1401)
        rebinned_pa_hydrogen_time2 = np.full((2, 8), 16)
        rebinned_pa_hydrogen_delta_plus_time2 = np.full((2, 8), 1600)
        rebinned_pa_hydrogen_delta_minus_time2 = np.full((2, 8), 1601)
        rebinned_pa_iron_time2 = np.full((2, 8), 18)
        rebinned_pa_iron_delta_plus_time2 = np.full((2, 8), 1800)
        rebinned_pa_iron_delta_minus_time2 = np.full((2, 8), 1801)
        rebinned_pa_NeMgSi_time2 = np.full((2, 8), 20)
        rebinned_pa_NeMgSi_delta_plus_time2 = np.full((2, 8), 2000)
        rebinned_pa_NeMgSi_delta_minus_time2 = np.full((2, 8), 2001)

        mock_hit_rebin.side_effect = [
            (rebinned_pa_gyro_hydrogen_time1, rebinned_pa_gyro_hydrogen_delta_plus_time1,
             rebinned_pa_gyro_hydrogen_delta_minus_time1,
             rebinned_pa_hydrogen_time1, rebinned_pa_hydrogen_delta_plus_time1, rebinned_pa_hydrogen_delta_minus_time1),
            (rebinned_pa_gyro_helium4_time1, rebinned_pa_gyro_helium4_delta_plus_time1,
             rebinned_pa_gyro_helium4_delta_minus_time1,
             rebinned_pa_helium4_time1, rebinned_pa_helium4_delta_plus_time1, rebinned_pa_helium4_delta_minus_time1),
            (rebinned_pa_gyro_CNO_time1, rebinned_pa_gyro_CNO_delta_plus_time1, rebinned_pa_gyro_CNO_delta_minus_time1,
             rebinned_pa_CNO_time1, rebinned_pa_CNO_delta_plus_time1, rebinned_pa_CNO_delta_minus_time1),
            (rebinned_pa_gyro_NeMgSi_time1, rebinned_pa_gyro_NeMgSi_delta_plus_time1,
             rebinned_pa_gyro_NeMgSi_delta_minus_time1,
             rebinned_pa_NeMgSi_time1, rebinned_pa_NeMgSi_delta_plus_time1, rebinned_pa_NeMgSi_delta_minus_time1),
            (rebinned_pa_gyro_iron_time1, rebinned_pa_gyro_iron_delta_plus_time1,
             rebinned_pa_gyro_iron_delta_minus_time1,
             rebinned_pa_iron_time1, rebinned_pa_iron_delta_plus_time1, rebinned_pa_iron_delta_minus_time1),

            (rebinned_pa_gyro_hydrogen_time2, rebinned_pa_gyro_hydrogen_delta_plus_time2,
             rebinned_pa_gyro_hydrogen_delta_minus_time2,
             rebinned_pa_hydrogen_time2, rebinned_pa_hydrogen_delta_plus_time2, rebinned_pa_hydrogen_delta_minus_time2),
            (rebinned_pa_gyro_helium4_time2, rebinned_pa_gyro_helium4_delta_plus_time2,
             rebinned_pa_gyro_helium4_delta_minus_time2,
             rebinned_pa_helium4_time2, rebinned_pa_helium4_delta_plus_time2, rebinned_pa_helium4_delta_minus_time2),
            (rebinned_pa_gyro_CNO_time2, rebinned_pa_gyro_CNO_delta_plus_time2, rebinned_pa_gyro_CNO_delta_minus_time2,
             rebinned_pa_CNO_time2, rebinned_pa_CNO_delta_plus_time2, rebinned_pa_CNO_delta_minus_time2),
            (rebinned_pa_gyro_NeMgSi_time2, rebinned_pa_gyro_NeMgSi_delta_plus_time2,
             rebinned_pa_gyro_NeMgSi_delta_minus_time2,
             rebinned_pa_NeMgSi_time2, rebinned_pa_NeMgSi_delta_plus_time2, rebinned_pa_NeMgSi_delta_minus_time2),
            (rebinned_pa_gyro_iron_time2, rebinned_pa_gyro_iron_delta_plus_time2,
             rebinned_pa_gyro_iron_delta_minus_time2,
             rebinned_pa_iron_time2, rebinned_pa_iron_delta_plus_time2, rebinned_pa_iron_delta_minus_time2),
        ]

        processor = HitProcessor(mock_processing_input_collection, input_metadata)
        product = processor.process()
        mock_fetch_dependencies.assert_called_once_with(mock_processing_input_collection)

        mock_transform_epochs.assert_called_once_with(sentinel.pre_transform_hit_data)

        mock_mag_data.rebin_to.assert_called_once_with(mock_hit_data.epoch, mock_hit_data.epoch_delta)

        mock_calculate_unit_vector.assert_has_calls([
            call(sentinel.mag_vector1),
            call(sentinel.mag_vector2)
        ])
        mock_get_sector_unit_vectors.assert_called_once_with(mock_hit_data.zenith, mock_hit_data.azimuth)

        self.assertEqual(1, mock_rotate_particle_vectors_from_hit_despun_to_imap_despun.call_count)
        np.testing.assert_array_equal(
            mock_rotate_particle_vectors_from_hit_despun_to_imap_despun.call_args_list[0].args[0],
            -expected_sector_unit_vectors)

        self.assertEqual(2, mock_calculate_pitch_angle.call_count)
        np.testing.assert_array_equal(mock_calculate_pitch_angle.call_args_list[0].args[0],
                                      sentinel.rotated_particle_vector)
        self.assertEqual(mock_calculate_pitch_angle.call_args_list[0].args[1], sentinel.mag_unit_vector1)

        np.testing.assert_array_equal(mock_calculate_pitch_angle.call_args_list[1].args[0],
                                      sentinel.rotated_particle_vector)
        self.assertEqual(mock_calculate_pitch_angle.call_args_list[1].args[1], sentinel.mag_unit_vector2)

        self.assertEqual(2, mock_calculate_gyrophase.call_count)
        np.testing.assert_array_equal(mock_calculate_gyrophase.call_args_list[0].args[0],
                                      sentinel.rotated_particle_vector)
        self.assertEqual(mock_calculate_gyrophase.call_args_list[0].args[1], sentinel.mag_unit_vector1)

        np.testing.assert_array_equal(mock_calculate_gyrophase.call_args_list[1].args[0],
                                      sentinel.rotated_particle_vector)
        self.assertEqual(mock_calculate_gyrophase.call_args_list[1].args[1], sentinel.mag_unit_vector2)

        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15

        mock_hit_rebin.assert_has_calls([
            call(NumpyArrayMatcher(hydrogen_time1), NumpyArrayMatcher(delta_plus_hydrogen_time1),
                 NumpyArrayMatcher(delta_minus_hydrogen_time1), sentinel.pitch_angle1, sentinel.gyrophase1,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(helium4_time1), NumpyArrayMatcher(delta_plus_helium4_time1),
                 NumpyArrayMatcher(delta_minus_helium4_time1), sentinel.pitch_angle1, sentinel.gyrophase1,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(CNO_time1), NumpyArrayMatcher(delta_plus_CNO_time1),
                 NumpyArrayMatcher(delta_minus_CNO_time1), sentinel.pitch_angle1, sentinel.gyrophase1,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(NeMgSi_time1), NumpyArrayMatcher(delta_plus_NeMgSi_time1),
                 NumpyArrayMatcher(delta_minus_NeMgSi_time1), sentinel.pitch_angle1, sentinel.gyrophase1,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(iron_time1), NumpyArrayMatcher(delta_plus_iron_time1),
                 NumpyArrayMatcher(delta_minus_iron_time1), sentinel.pitch_angle1, sentinel.gyrophase1,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(hydrogen_time2), NumpyArrayMatcher(delta_plus_hydrogen_time2),
                 NumpyArrayMatcher(delta_minus_hydrogen_time2), sentinel.pitch_angle2, sentinel.gyrophase2,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(helium4_time2), NumpyArrayMatcher(delta_plus_helium4_time2),
                 NumpyArrayMatcher(delta_minus_helium4_time2), sentinel.pitch_angle2, sentinel.gyrophase2,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(CNO_time2), NumpyArrayMatcher(delta_plus_CNO_time2),
                 NumpyArrayMatcher(delta_minus_CNO_time2), sentinel.pitch_angle2, sentinel.gyrophase2,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(NeMgSi_time2), NumpyArrayMatcher(delta_plus_NeMgSi_time2),
                 NumpyArrayMatcher(delta_minus_NeMgSi_time2), sentinel.pitch_angle2, sentinel.gyrophase2,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(NumpyArrayMatcher(iron_time2), NumpyArrayMatcher(delta_plus_iron_time2),
                 NumpyArrayMatcher(delta_minus_iron_time2), sentinel.pitch_angle2, sentinel.gyrophase2,
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
        ])

        saved_data_product: HitPitchAngleDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(input_metadata, saved_data_product.input_metadata)
        self.assertEqual(["parent_path"], saved_data_product.parent_file_names)
        np.testing.assert_array_equal(saved_data_product.epochs, epochs)
        np.testing.assert_array_equal(saved_data_product.epoch_deltas, epoch_deltas)

        np.testing.assert_array_equal(saved_data_product.pitch_angles, sentinel.pitch_angles)
        np.testing.assert_array_equal(saved_data_product.pitch_angle_deltas, sentinel.pitch_angle_deltas)
        np.testing.assert_array_equal(saved_data_product.gyrophases, sentinel.gyrophases)
        np.testing.assert_array_equal(saved_data_product.gyrophase_deltas, sentinel.gyrophase_deltas)

        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity,
                                      [rebinned_pa_gyro_hydrogen_time1,
                                       rebinned_pa_gyro_hydrogen_time2])
        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity_delta_plus,
                                      [rebinned_pa_gyro_hydrogen_delta_plus_time1,
                                       rebinned_pa_gyro_hydrogen_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity_delta_minus,
                                      [rebinned_pa_gyro_hydrogen_delta_minus_time1,
                                       rebinned_pa_gyro_hydrogen_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity,
                                      [rebinned_pa_gyro_helium4_time1,
                                       rebinned_pa_gyro_helium4_time2])
        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity_delta_plus,
                                      [rebinned_pa_gyro_helium4_delta_plus_time1,
                                       rebinned_pa_gyro_helium4_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity_delta_minus,
                                      [rebinned_pa_gyro_helium4_delta_minus_time1,
                                       rebinned_pa_gyro_helium4_delta_minus_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity,
                                      [rebinned_pa_gyro_CNO_time1,
                                       rebinned_pa_gyro_CNO_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity_delta_plus,
                                      [rebinned_pa_gyro_CNO_delta_plus_time1,
                                       rebinned_pa_gyro_CNO_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity_delta_minus,
                                      [rebinned_pa_gyro_CNO_delta_minus_time1,
                                       rebinned_pa_gyro_CNO_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity,
                                      [rebinned_pa_gyro_NeMgSi_time1,
                                       rebinned_pa_gyro_NeMgSi_time2])
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity_delta_plus,
                                      [rebinned_pa_gyro_NeMgSi_delta_plus_time1,
                                       rebinned_pa_gyro_NeMgSi_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity_delta_minus,
                                      [rebinned_pa_gyro_NeMgSi_delta_minus_time1,
                                       rebinned_pa_gyro_NeMgSi_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity,
                                      [rebinned_pa_gyro_iron_time1,
                                       rebinned_pa_gyro_iron_time2])
        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity_delta_plus,
                                      [rebinned_pa_gyro_iron_delta_plus_time1,
                                       rebinned_pa_gyro_iron_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity_delta_minus,
                                      [rebinned_pa_gyro_iron_delta_minus_time1,
                                       rebinned_pa_gyro_iron_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity_pa,
                                      [rebinned_pa_hydrogen_time1, rebinned_pa_hydrogen_time2])
        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity_pa_delta_plus,
                                      [rebinned_pa_hydrogen_delta_plus_time1,
                                       rebinned_pa_hydrogen_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.h_macropixel_intensity_pa_delta_minus,
                                      [rebinned_pa_hydrogen_delta_minus_time1,
                                       rebinned_pa_hydrogen_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity_pa,
                                      [rebinned_pa_helium4_time1, rebinned_pa_helium4_time2])
        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity_pa_delta_plus,
                                      [rebinned_pa_helium4_delta_plus_time1, rebinned_pa_helium4_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.he4_macropixel_intensity_pa_delta_minus,
                                      [rebinned_pa_helium4_delta_minus_time1,
                                       rebinned_pa_helium4_delta_minus_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity_pa,
                                      [rebinned_pa_CNO_time1, rebinned_pa_CNO_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity_pa_delta_plus,
                                      [rebinned_pa_CNO_delta_plus_time1, rebinned_pa_CNO_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.cno_macropixel_intensity_pa_delta_minus,
                                      [rebinned_pa_CNO_delta_minus_time1, rebinned_pa_CNO_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity_pa,
                                      [rebinned_pa_NeMgSi_time1, rebinned_pa_NeMgSi_time2])
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity_pa_delta_plus,
                                      [rebinned_pa_NeMgSi_delta_plus_time1, rebinned_pa_NeMgSi_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_macropixel_intensity_pa_delta_minus,
                                      [rebinned_pa_NeMgSi_delta_minus_time1, rebinned_pa_NeMgSi_delta_minus_time2])

        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity_pa,
                                      [rebinned_pa_iron_time1, rebinned_pa_iron_time2])
        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity_pa_delta_plus,
                                      [rebinned_pa_iron_delta_plus_time1, rebinned_pa_iron_delta_plus_time2])
        np.testing.assert_array_equal(saved_data_product.iron_macropixel_intensity_pa_delta_minus,
                                      [rebinned_pa_iron_delta_minus_time1, rebinned_pa_iron_delta_minus_time2])

        self.assertIs(mock_hit_data.h_energy, saved_data_product.h_energies)
        self.assertIs(mock_hit_data.h_energy_delta_plus, saved_data_product.h_energy_delta_plus)
        self.assertIs(mock_hit_data.h_energy_delta_minus, saved_data_product.h_energy_delta_minus)
        self.assertIs(mock_hit_data.he4_energy, saved_data_product.he4_energies)
        self.assertIs(mock_hit_data.he4_energy_delta_plus, saved_data_product.he4_energy_delta_plus)
        self.assertIs(mock_hit_data.he4_energy_delta_minus, saved_data_product.he4_energy_delta_minus)
        self.assertIs(mock_hit_data.cno_energy, saved_data_product.cno_energies)
        self.assertIs(mock_hit_data.cno_energy_delta_plus, saved_data_product.cno_energy_delta_plus)
        self.assertIs(mock_hit_data.cno_energy_delta_minus, saved_data_product.cno_energy_delta_minus)
        self.assertIs(mock_hit_data.nemgsi_energy, saved_data_product.ne_mg_si_energies)
        self.assertIs(mock_hit_data.nemgsi_energy_delta_plus, saved_data_product.ne_mg_si_energy_delta_plus)
        self.assertIs(mock_hit_data.nemgsi_energy_delta_minus, saved_data_product.ne_mg_si_energy_delta_minus)
        self.assertIs(mock_hit_data.fe_energy, saved_data_product.iron_energies)
        self.assertIs(mock_hit_data.fe_energy_delta_plus, saved_data_product.iron_energy_delta_plus)
        self.assertIs(mock_hit_data.fe_energy_delta_minus, saved_data_product.iron_energy_delta_minus)
        self.assertIs(mock_hit_data.azimuth, saved_data_product.azimuth)
        self.assertIs(mock_hit_data.zenith, saved_data_product.zenith)

        np.testing.assert_array_equal(saved_data_product.measurement_pitch_angle,
                                      np.array([sentinel.pitch_angle1, sentinel.pitch_angle2]))
        np.testing.assert_array_equal(saved_data_product.measurement_gyrophase,
                                      np.array([sentinel.gyrophase1, sentinel.gyrophase2]))

        self.assertEqual([mock_save_data.return_value], product)

    @patch("imap_l3_processing.utils.spiceypy")
    @patch("imap_l3_processing.hit.l3.hit_processor.save_data")
    @patch("imap_l3_processing.hit.l3.hit_processor.process_pha_event", autospec=True)
    @patch("imap_l3_processing.hit.l3.hit_processor.HitL3PhaDependencies.fetch_dependencies")
    @patch("imap_l3_processing.hit.l3.hit_processor.PHAEventReader.read_all_pha_events")
    def test_process_direct_event_product(self, mock_read_all_events, mock_fetch_dependencies, mock_process_pha_event,
                                          mock_save_data, mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 0
        pha_word_1 = PHAWord(adc_overflow=False, adc_value=11,
                             detector=Detector(layer=1, side="A", segment="1A", address=2, group="L1A4"),
                             is_last_pha=True,
                             is_low_gain=True)

        raw_pha_event_1 = RawPHAEvent(particle_id=1, priority_buffer_num=2, stim_tag=False, haz_tag=False, time_tag=20,
                                      a_b_side_flag=False, has_unread_adcs=True, long_event_flag=False,
                                      culling_flag=True,
                                      spare=True, pha_words=[pha_word_1])

        raw_pha_event_2 = RawPHAEvent(particle_id=1, priority_buffer_num=2, stim_tag=False, haz_tag=False, time_tag=20,
                                      a_b_side_flag=False, has_unread_adcs=True, long_event_flag=False,
                                      culling_flag=True,
                                      spare=True, pha_words=[deepcopy(pha_word_1)])

        pha_word_2 = PHAWord(adc_overflow=True, adc_value=12,
                             detector=Detector(layer=2, side="B", segment="1B", address=5, group="L2B"),
                             is_last_pha=False,
                             is_low_gain=False)

        pha_word_3 = PHAWord(adc_overflow=False, adc_value=11,
                             detector=Detector(layer=1, side="A", segment="2A", address=8, group="L1A1"),
                             is_last_pha=True,
                             is_low_gain=True)

        raw_pha_event_3 = RawPHAEvent(particle_id=2, priority_buffer_num=3, stim_tag=True, haz_tag=True, time_tag=30,
                                      a_b_side_flag=True, has_unread_adcs=False, long_event_flag=True,
                                      culling_flag=False,
                                      spare=False, pha_words=[pha_word_2, pha_word_3],
                                      extended_header=PHAExtendedHeader(detector_flags=2, delta_e_index=2,
                                                                        e_prime_index=True),
                                      stim_block=StimBlock(stim_step=1, stim_gain=2, unused=3, a_l_stim=True),
                                      extended_stim_header=ExtendedStimHeader(dac_value=123, tbd=666))

        event_output_1 = EventOutput(original_event=raw_pha_event_1, charge=9.0, energies=[1], total_energy=99,
                                     detected_range=DetectedRange(DetectorRange.R2, DetectorSide.A), e_delta=103.7,
                                     e_prime=63.27)
        event_output_2 = EventOutput(original_event=raw_pha_event_2, charge=10.0, energies=[4], total_energy=100,
                                     detected_range=None, e_delta=None, e_prime=None)
        event_output_3 = EventOutput(original_event=raw_pha_event_3, charge=12.0, energies=[9, 8], total_energy=200,
                                     detected_range=DetectedRange(DetectorRange.R3, DetectorSide.A), e_delta=106.7,
                                     e_prime=69.27)

        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="direct-events"
        )

        mock_processing_input_collection = Mock()
        mock_processing_input_collection.get_file_paths.return_value = [Path("test/parent_path")]

        mock_hit_l3_pha_dependencies = mock_fetch_dependencies.return_value
        mock_hit_l3_pha_dependencies.hit_l1_data.epoch = [datetime(year=2020, month=2, day=1),
                                                          datetime(year=2020, month=2, day=1, hour=1)]
        mock_hit_l3_pha_dependencies.hit_l1_data.event_binary = [sentinel.binary_stream_1, sentinel.binary_stream_2]

        mock_process_pha_event.side_effect = [event_output_1, event_output_2, event_output_3]
        mock_read_all_events.side_effect = [[raw_pha_event_1, raw_pha_event_2], [raw_pha_event_3]]

        processor = HitProcessor(mock_processing_input_collection, input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(mock_processing_input_collection)
        mock_read_all_events.assert_has_calls([call(sentinel.binary_stream_1), call(sentinel.binary_stream_2)])

        mock_process_pha_event.assert_has_calls(
            [call(raw_pha_event_1, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup,
                  mock_hit_l3_pha_dependencies.event_type_lookup),
             call(raw_pha_event_2, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup,
                  mock_hit_l3_pha_dependencies.event_type_lookup),
             call(raw_pha_event_3, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup,
                  mock_hit_l3_pha_dependencies.event_type_lookup)],
            any_order=False)

        mock_save_data.assert_called_once()

        self.assertEqual([mock_save_data.return_value], product)

        direct_event_product = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(input_metadata, direct_event_product.input_metadata)
        self.assertEqual(["parent_path"], direct_event_product.parent_file_names)

        np.testing.assert_array_equal(direct_event_product.epoch, np.array(
            [datetime(year=2020, month=2, day=1), datetime(year=2020, month=2, day=1),
             datetime(year=2020, month=2, day=1, hour=1)]))
        np.testing.assert_array_equal(direct_event_product.charge, np.array([9.0, 10.0, 12.0]))
        np.testing.assert_array_equal(direct_event_product.energy, np.array([99, 100, 200]))
        np.testing.assert_array_equal(direct_event_product.particle_id, np.array([1, 1, 2]))

        np.testing.assert_array_equal(direct_event_product.e_delta, np.array([103.7, np.nan, 106.7]))
        np.testing.assert_array_equal(direct_event_product.e_prime, np.array([63.27, np.nan, 69.27]))

        np.testing.assert_array_equal(direct_event_product.detected_range, np.array([2, np.nan, 3]))

        np.testing.assert_array_equal(direct_event_product.priority_buffer_number, np.array([2, 2, 3]))
        np.testing.assert_array_equal(direct_event_product.latency, np.array([20, 20, 30]))
        np.testing.assert_array_equal(direct_event_product.stim_tag, np.array([False, False, True]))
        np.testing.assert_array_equal(direct_event_product.long_event_flag, np.array([False, False, True]))
        np.testing.assert_array_equal(direct_event_product.haz_tag, np.array([False, False, True]))
        np.testing.assert_array_equal(direct_event_product.a_b_side, np.array([0, np.nan, 0]))
        np.testing.assert_array_equal(direct_event_product.has_unread_adcs, np.array([True, True, False]))
        np.testing.assert_array_equal(direct_event_product.culling_flag, np.array([True, True, False]))

        self.assertEqual((3, 64), direct_event_product.pha_value.shape)
        expected_values = {(0, pha_word_1.detector.address): (
            pha_word_1.adc_value, event_output_1.energies[0], pha_word_1.is_low_gain),
            (1, pha_word_1.detector.address): (
                pha_word_1.adc_value, event_output_2.energies[0], pha_word_1.is_low_gain),
            (2, pha_word_2.detector.address): (
                pha_word_2.adc_value, event_output_3.energies[0], pha_word_2.is_low_gain),
            (2, pha_word_3.detector.address): (
                pha_word_3.adc_value, event_output_3.energies[1], pha_word_3.is_low_gain)}

        mask = np.full_like(direct_event_product.pha_value, True, dtype=bool)
        for idx, (adc_val, energy, is_low_gain) in expected_values.items():
            np.testing.assert_array_equal(direct_event_product.pha_value[idx], adc_val,
                                          err_msg=f"Did not match at index {idx}")
            np.testing.assert_array_equal(direct_event_product.energy_at_detector[idx], energy,
                                          err_msg=f"Did not match at index {idx}")
            np.testing.assert_array_equal(direct_event_product.is_low_gain[idx], is_low_gain,
                                          err_msg=f"Did not match at index {idx}")
            mask[idx] = False

        self.assertTrue(np.all(direct_event_product.pha_value[mask] == UNSIGNED_INT2_FILL_VALUE))
        self.assertTrue(np.all(np.isnan(direct_event_product.energy_at_detector[mask])))
        self.assertTrue(np.all(direct_event_product.is_low_gain[mask] == False))

        np.testing.assert_array_equal(direct_event_product.detector_flags,
                                      np.array([UNSIGNED_INT2_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE, 2]))
        np.testing.assert_array_equal(direct_event_product.deindex,
                                      np.array([UNSIGNED_INT2_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE, 2]))
        np.testing.assert_array_equal(direct_event_product.epindex,
                                      np.array([UNSIGNED_INT2_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE, True]))
        np.testing.assert_array_equal(direct_event_product.stim_gain, np.array([False, False, True]))
        np.testing.assert_array_equal(direct_event_product.a_l_stim, np.array([False, False, True]))
        np.testing.assert_array_equal(direct_event_product.stim_step,
                                      np.array([UNSIGNED_INT1_FILL_VALUE, UNSIGNED_INT1_FILL_VALUE, 1]))
        np.testing.assert_array_equal(direct_event_product.dac_value,
                                      np.array([UNSIGNED_INT2_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE, 123]))

    @patch("imap_l3_processing.hit.l3.hit_processor.PHAEventReader.read_all_pha_events")
    @patch("imap_l3_processing.hit.l3.hit_processor.process_pha_event", autospec=True)
    def test_direct_event_parsing_errors_result_in_skipped_event(self, mock_process_pha_event, mock_read_all_events):
        hit_l1_data = HitL1Data(epoch=np.array([datetime(year=2020, month=2, day=1),
                                                datetime(year=2020, month=2, day=1, hour=1)]),
                                event_binary=np.array([sentinel.binary_stream_1, sentinel.binary_stream_2]))
        mock_direct_event_dependencies = Mock()
        mock_direct_event_dependencies.hit_l1_data = hit_l1_data

        event = RawPHAEvent(
            particle_id=1,
            priority_buffer_num=1,
            time_tag=1,
            stim_tag=1,
            long_event_flag=False,
            haz_tag=False,
            a_b_side_flag=False,
            has_unread_adcs=False,
            culling_flag=False,
            spare=False,
            pha_words=[]
        )

        mock_read_all_events.side_effect = [Exception("error parsing"), [event]]

        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="direct-events"
        )

        processor = HitProcessor(Mock(), input_metadata)
        with self.assertLogs():
            actual_product = processor.process_direct_event_product(mock_direct_event_dependencies)

        self.assertEqual(1, len(actual_product.epoch))
        self.assertEqual(datetime(year=2020, month=2, day=1, hour=1), actual_product.epoch[0])

    def test_raise_error_if_descriptor_doesnt_match(self):
        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="spectral-index"
        )
        processor = HitProcessor(Mock(), input_metadata)

        with self.assertRaises(ValueError) as e:
            processor.process()

        self.assertEqual(e.exception.args[0],
                         f"Don't know how to generate 'spectral-index' /n Known HIT l3 data products: 'macropixel', 'direct-events'.")

    T = TypeVar("T")
