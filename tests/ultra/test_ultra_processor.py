import pickle
import unittest
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
import xarray as xr
from imap_data_access.processing_input import AncillaryInput, ScienceInput, ProcessingInputCollection
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.map_models import (
    HealPixIntensityMapData,
    IntensityMapData,
    HealPixCoords,
    SpectralIndexMapData,
    RectangularIntensityDataProduct,
    RectangularSpectralIndexDataProduct,
    RectangularSpectralIndexMapData,
    RectangularIntensityMapData,
)
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.ultra_l3_dependencies import (
    UltraL3Dependencies,
    UltraL3SpectralIndexDependencies,
    UltraL3CombinedDependencies,
)
from imap_l3_processing.ultra.ultra_processor import (
    UltraProcessor,
    correct_healpix_data_for_survival_probability,
)
from imap_l3_processing.utils import get_temp_cache_dir, clear_temp_cache
from tests.maps.test_builders import create_rectangular_intensity_map_data
from tests.test_helpers import get_test_data_path


class TestUltraProcessor(unittest.TestCase):
    def setUp(self):
        clear_temp_cache()

    def test_process_survival_probability_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_survival_probability(degree_spacing)

    def test_process_spectral_index_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_spectral_index(degree_spacing)

    def test_process_combined_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_combined_sensor(degree_spacing)

    def test_process_combined_survival_corrected_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_combined_sensor_survival_probability(degree_spacing)

    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbabilitySkyMap")
    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbability")
    @patch(
        "imap_l3_processing.ultra.ultra_processor.combine_glows_l3e_with_l1c_pointing"
    )
    def test_correct_healpix_data_for_survival_probability(
        self,
        mock_combine_glows_l3e_with_l1c_pointing,
        mock_survival_probability_pointing_set,
        mock_survival_skymap,
    ):
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)

        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()
        input_l2_healpix_map = _create_ultra_l2_healpix_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)
        input_l2_healpix_map.intensity_map_data.energy = sentinel.ultra_l2_energies
        input_l2_rectangular_map = create_rectangular_intensity_map_data()

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor_20250601_v000.cdf"
        input_l1c_pset_name = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_glows_l3e_name = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"

        dependencies = UltraL3Dependencies(
            ultra_l2_healpix_map=input_l2_healpix_map,
            ultra_l2_rectangular_map=input_l2_rectangular_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[
                Path(input_l2_map_name),
                Path(input_l1c_pset_name),
                Path(input_glows_l3e_name),
            ],
            energy_bin_group_sizes=sentinel.bin_groups,
        )

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [
            (sentinel.ultra_l1c_1, sentinel.glows_l3e_1),
            (sentinel.ultra_l1c_2, sentinel.glows_l3e_2),
            (sentinel.ultra_l1c_3, sentinel.glows_l3e_3),
        ]
        mock_survival_probability_pointing_set.side_effect = [
            sentinel.pset_1,
            sentinel.pset_2,
            sentinel.pset_3,
        ]
        computed_survival_probabilities = rng.random((1, 9, healpix_indices.shape[0]))
        expected_quality_flags = np.full((1, 9, healpix_indices.shape[0]), MapL3Flags.NONE)
        expected_quality_flags[rng.random((1, 9, healpix_indices.shape[0])) > 0.8] = MapL3Flags.PREDICTIVE_EPHEMERIS
        expected_quality_flags[rng.random((1, 9, healpix_indices.shape[0])) > 0.7] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO
        expected_quality_flags[rng.random((1, 9, healpix_indices.shape[0])) > 0.75] = MapL3Flags.PERSISTED_LAST_POINT

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset(
            {
                "exposure_weighted_survival_probabilities": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    computed_survival_probabilities,
                ),
                "quality_flags": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    expected_quality_flags,
                ),
            },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.HEALPIX_INDEX.value: healpix_indices,
            },
        )

        healpix_intensity_map_data = correct_healpix_data_for_survival_probability(dependencies, SpiceFrame.ECLIPJ2000)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_sp, sentinel.ultra_l1c_pset)
        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.ultra_l1c_1, sentinel.glows_l3e_1, bin_groups=sentinel.bin_groups),
            call(sentinel.ultra_l1c_2, sentinel.glows_l3e_2, bin_groups=sentinel.bin_groups),
            call(sentinel.ultra_l1c_3, sentinel.glows_l3e_3, bin_groups=sentinel.bin_groups)
        ])
        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     SpiceFrame.ECLIPJ2000, input_l2_healpix_map.coords.nside)
        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        actual_intensity_map_data = healpix_intensity_map_data.intensity_map_data
        intensity_data = input_l2_healpix_map.intensity_map_data

        np.testing.assert_array_equal(
            actual_intensity_map_data.ena_intensity,
            intensity_data.ena_intensity / computed_survival_probabilities,
        )
        np.testing.assert_array_equal(
            actual_intensity_map_data.ena_intensity_stat_uncert,
            intensity_data.ena_intensity_stat_uncert / computed_survival_probabilities,
        )
        np.testing.assert_array_equal(
            actual_intensity_map_data.ena_intensity_sys_err,
            intensity_data.ena_intensity_sys_err / computed_survival_probabilities,
        )

        np.testing.assert_array_equal(
            actual_intensity_map_data.survival_probability,
            computed_survival_probabilities,
        )

        np.testing.assert_array_equal(actual_intensity_map_data.epoch, intensity_data.epoch)
        np.testing.assert_array_equal(actual_intensity_map_data.epoch_delta, intensity_data.epoch_delta)
        np.testing.assert_array_equal(actual_intensity_map_data.energy, intensity_data.energy)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_delta_plus, intensity_data.energy_delta_plus)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_delta_minus,
                                      intensity_data.energy_delta_minus)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_label, intensity_data.energy_label)
        np.testing.assert_array_equal(actual_intensity_map_data.latitude, intensity_data.latitude)
        np.testing.assert_array_equal(actual_intensity_map_data.longitude, intensity_data.longitude)
        np.testing.assert_array_equal(actual_intensity_map_data.exposure_factor, intensity_data.exposure_factor)
        np.testing.assert_array_equal(actual_intensity_map_data.obs_date, intensity_data.obs_date)
        np.testing.assert_array_equal(actual_intensity_map_data.obs_date_range, intensity_data.obs_date_range)
        np.testing.assert_array_equal(actual_intensity_map_data.solid_angle, intensity_data.solid_angle)
        np.testing.assert_array_equal(actual_intensity_map_data.quality_flags, expected_quality_flags)
        coords = healpix_intensity_map_data.coords
        np.testing.assert_array_equal(
            coords.pixel_index, input_l2_healpix_map.coords.pixel_index
        )
        np.testing.assert_array_equal(
            coords.pixel_index_label, input_l2_healpix_map.coords.pixel_index_label
        )

    @patch("imap_l3_processing.ultra.ultra_processor.HealPixIntensityMapData")
    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbabilitySkyMap")
    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbability")
    @patch(
        "imap_l3_processing.ultra.ultra_processor.combine_glows_l3e_with_l1c_pointing"
    )
    def test_correct_healpix_data_for_survival_probability_caches_results(
        self,
        mock_combine_glows_l3e_with_l1c_pointing,
        mock_survival_probability_pointing_set,
        mock_survival_skymap,
        mock_healpix_intensity_map_data,
    ):
        mock_healpix_intensity_map_data.return_value = sentinel.healpix_intensity_map_data
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)

        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()
        input_l2_healpix_map = _create_ultra_l2_healpix_data(
            epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices
        )
        input_l2_healpix_map.intensity_map_data.energy = sentinel.ultra_l2_energies
        input_l2_rectangular_map = create_rectangular_intensity_map_data()

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor-2deg_20250601_v000.cdf"
        input_l1c_pset_name_1 = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_l1c_pset_name_2 = "imap_ultra_l1c_a-pset-descriptor_20250602_v000.cdf"
        input_glows_l3e_name_1 = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"
        input_glows_l3e_name_2 = "imap_glows_l3e_a-glows-descriptor_20250602_v000.cdf"

        dependencies = UltraL3Dependencies(
            ultra_l2_healpix_map=input_l2_healpix_map,
            ultra_l2_rectangular_map=input_l2_rectangular_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[
                Path("not a science file"),
                Path(input_l2_map_name),
                Path(input_l1c_pset_name_1),
                Path(input_l1c_pset_name_2),
                Path(input_glows_l3e_name_1),
                Path(input_glows_l3e_name_2),
            ],
            energy_bin_group_sizes=sentinel.bin_groups,
        )

        computed_survival_probabilities = rng.random((1, 9, healpix_indices.shape[0]))

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset(
            {
                "exposure_weighted_survival_probabilities": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    computed_survival_probabilities,
                ),
                "quality_flags": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    np.full_like(computed_survival_probabilities, MapL3Flags.NONE),
                ),
            },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.HEALPIX_INDEX.value: healpix_indices,
            },
        )

        healpix_intensity_map_data = correct_healpix_data_for_survival_probability(dependencies, SpiceFrame.ECLIPJ2000)

        cache_dir = get_temp_cache_dir()
        expected_cache_key = (f"['{input_l1c_pset_name_1}', '{input_l1c_pset_name_2}']"
        f"['{input_glows_l3e_name_1}', '{input_glows_l3e_name_2}']"
        "a-map-descriptor-nside32")
        expected_cache_path = cache_dir / sha256(expected_cache_key.encode("utf-8")).hexdigest()

        with open(expected_cache_path, "rb") as f:
            cached_data = pickle.load(f)

        self.assertEqual(healpix_intensity_map_data, cached_data)


    @patch("imap_l3_processing.ultra.ultra_processor.HealPixIntensityMapData")
    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbabilitySkyMap")
    @patch("imap_l3_processing.ultra.ultra_processor.UltraSurvivalProbability")
    @patch(
        "imap_l3_processing.ultra.ultra_processor.combine_glows_l3e_with_l1c_pointing"
    )
    def test_correct_healpix_data_for_survival_probability_uses_cached_result(
        self,
        mock_combine_glows_l3e_with_l1c_pointing,
        mock_survival_probability_pointing_set,
        mock_survival_skymap,
        mock_healpix_intensity_map_data,
    ):
        input_l2_map_name = "imap_ultra_l2_a-map-descriptor-2deg_20250601_v000.cdf"
        input_l1c_pset_name_1 = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_l1c_pset_name_2 = "imap_ultra_l1c_a-pset-descriptor_20250602_v000.cdf"
        input_glows_l3e_name_1 = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"
        input_glows_l3e_name_2 = "imap_glows_l3e_a-glows-descriptor_20250602_v000.cdf"

        cache_dir = get_temp_cache_dir()
        expected_cache_key = (f"['{input_l1c_pset_name_1}', '{input_l1c_pset_name_2}']"
        f"['{input_glows_l3e_name_1}', '{input_glows_l3e_name_2}']"
        "a-map-descriptor-nside32")
        expected_cache_path = cache_dir / sha256(expected_cache_key.encode("utf-8")).hexdigest()
        cached_result = sentinel.healpix_intensity_map_data

        with open(expected_cache_path, "wb") as f:
            pickle.dump(cached_result, f)

        rng = np.random.default_rng()
        healpix_indices = np.arange(12)

        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()
        input_l2_healpix_map = _create_ultra_l2_healpix_data(
            epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices
        )
        input_l2_healpix_map.intensity_map_data.energy = sentinel.ultra_l2_energies
        input_l2_rectangular_map = create_rectangular_intensity_map_data()

        dependencies = UltraL3Dependencies(
            ultra_l2_healpix_map=input_l2_healpix_map,
            ultra_l2_rectangular_map=input_l2_rectangular_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[
                Path("not a science file"),
                Path(input_l2_map_name),
                Path(input_l1c_pset_name_1),
                Path(input_l1c_pset_name_2),
                Path(input_glows_l3e_name_1),
                Path(input_glows_l3e_name_2),
            ],
            energy_bin_group_sizes=sentinel.bin_groups,
        )

        healpix_intensity_map_data = correct_healpix_data_for_survival_probability(dependencies, SpiceFrame.ECLIPJ2000)

        self.assertEqual(cached_result, healpix_intensity_map_data)
        mock_combine_glows_l3e_with_l1c_pointing.assert_not_called()
        mock_survival_probability_pointing_set.assert_not_called()
        mock_survival_skymap.assert_not_called()
        mock_healpix_intensity_map_data.assert_not_called()

    @patch('imap_l3_processing.ultra.ultra_processor.correct_healpix_data_for_survival_probability')
    @patch('imap_l3_processing.utils.spiceypy')
    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def _test_process_survival_probability(self, degree_spacing, mock_fetch_dependencies,
                                           mock_save_data, mock_spiceypy,
                                           mock_correct_healpix_data_for_survival_probability):
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        mock_spiceypy.ktotal.return_value = 1
        fake_spice = Path("path/to/fake/spice.tls")
        mock_spiceypy.kdata.return_value = [fake_spice]

        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()
        input_l2_healpix_map = _create_ultra_l2_healpix_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)
        input_l2_healpix_map.intensity_map_data.energy = sentinel.ultra_l2_energies
        input_l2_rectangular_map = create_rectangular_intensity_map_data()

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor_20250601_v000.cdf"
        input_l1c_pset_name = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_glows_l3e_name = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"
        input_deps = ProcessingInputCollection(ScienceInput(input_l2_map_name))
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"u90-ena-h-sf-sp-full-hae-{degree_spacing}deg-6mo")

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_healpix_map=input_l2_healpix_map,
            ultra_l2_rectangular_map=input_l2_rectangular_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[
                Path(input_l2_map_name),
                Path(input_l1c_pset_name),
                Path(input_glows_l3e_name),
            ],
            energy_bin_group_sizes=sentinel.bin_groups,
        )

        rectangular_predicted_ephemeris_data = np.array([0.73, 0.24, 0, 0.99, 1, 0])
        nominal_alpha_proton_ratio_data = np.full_like(rectangular_predicted_ephemeris_data, 0.0)
        persisted_last_point_data = np.full_like(rectangular_predicted_ephemeris_data, 0.0)

        expected_quality_flags = np.array([
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NONE
        ])

        mock_healpix_map_data = mock_correct_healpix_data_for_survival_probability.return_value
        mock_healpix_skymap = Mock()
        mock_healpix_map_data.to_healpix_skymap = Mock(return_value=mock_healpix_skymap)

        mock_rectangular_map_dataset = {
            "ena_intensity": Mock(values=sentinel.rectangular_ena_intensity),
            "ena_intensity_stat_uncert": Mock(
                values=sentinel.rectangular_ena_intensity_stat_uncert
            ),
            "ena_intensity_sys_err": Mock(
                values=sentinel.rectangular_ena_intensity_sys_err
            ),
            "survival_probability": Mock(
                values=sentinel.rectangular_survival_probability
            ),
            "predicted_ephemeris_flag": Mock(
                values=rectangular_predicted_ephemeris_data
            ),
            "nominal_alpha_proton_ratio_flag": Mock(values=nominal_alpha_proton_ratio_data),
            "persisted_last_point_flag": Mock(values=persisted_last_point_data)
        }

        mock_rectangular_sky_map = Mock(spec=RectangularSkyMap)
        mock_rectangular_sky_map.to_dataset.return_value = mock_rectangular_map_dataset
        mock_healpix_skymap.to_rectangular_skymap.return_value = mock_rectangular_sky_map, 0

        processor = UltraProcessor(input_deps, input_metadata)
        product = processor.process(SpiceFrame.IMAP_GCS)

        mock_fetch_dependencies.assert_called_once_with(input_deps)
        mock_correct_healpix_data_for_survival_probability.assert_called_once_with(
            mock_fetch_dependencies.return_value, SpiceFrame.IMAP_GCS
        )
        mock_healpix_map_data.to_healpix_skymap.assert_called_once()
        mock_healpix_skymap.to_rectangular_skymap.assert_called_once_with(degree_spacing, [
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "predicted_ephemeris_flag",
            "nominal_alpha_proton_ratio_flag",
            "persisted_last_point_flag",
            "survival_probability",
        ])

        mock_save_data.assert_called_once()
        actual_rectangular_data_product = mock_save_data.call_args_list[0].args[0]
        self.assertIsInstance(actual_rectangular_data_product, RectangularIntensityDataProduct)
        self.assertEqual(4, len(actual_rectangular_data_product.parent_file_names))
        self.assertEqual({input_l2_map_name, fake_spice.name, input_l1c_pset_name, input_glows_l3e_name},
                         set(actual_rectangular_data_product.parent_file_names))
        self.assertEqual(SpiceFrame.IMAP_GCS, actual_rectangular_data_product.spice_frame_name)
        actual_rectangular_data = actual_rectangular_data_product.data
        self.assertIsInstance(actual_rectangular_data.intensity_map_data, IntensityMapData)

        # @formatter:off
        self.assertEqual(sentinel.rectangular_ena_intensity, actual_rectangular_data.intensity_map_data.ena_intensity)
        self.assertEqual(sentinel.rectangular_ena_intensity_stat_uncert, actual_rectangular_data.intensity_map_data.ena_intensity_stat_uncert)
        self.assertEqual(sentinel.rectangular_ena_intensity_sys_err, actual_rectangular_data.intensity_map_data.ena_intensity_sys_err)
        self.assertEqual(sentinel.rectangular_survival_probability, actual_rectangular_data.intensity_map_data.survival_probability)
        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.quality_flags, expected_quality_flags, strict=True)

        expected_passthrough = input_l2_rectangular_map.intensity_map_data
        self.assertIs(expected_passthrough.epoch, actual_rectangular_data.intensity_map_data.epoch)
        self.assertIs(expected_passthrough.epoch_delta, actual_rectangular_data.intensity_map_data.epoch_delta)
        self.assertIs(expected_passthrough.energy, actual_rectangular_data.intensity_map_data.energy)
        self.assertIs(expected_passthrough.energy_delta_plus, actual_rectangular_data.intensity_map_data.energy_delta_plus)
        self.assertIs(expected_passthrough.energy_delta_minus, actual_rectangular_data.intensity_map_data.energy_delta_minus)
        self.assertIs(expected_passthrough.energy_label, actual_rectangular_data.intensity_map_data.energy_label)
        self.assertIs(expected_passthrough.latitude, actual_rectangular_data.intensity_map_data.latitude)
        self.assertIs(expected_passthrough.longitude, actual_rectangular_data.intensity_map_data.longitude)
        self.assertIs(expected_passthrough.exposure_factor, actual_rectangular_data.intensity_map_data.exposure_factor)
        self.assertIs(expected_passthrough.obs_date, actual_rectangular_data.intensity_map_data.obs_date)
        self.assertIs(expected_passthrough.obs_date_range, actual_rectangular_data.intensity_map_data.obs_date_range)
        self.assertIs(expected_passthrough.solid_angle, actual_rectangular_data.intensity_map_data.solid_angle)

        expected_coords = input_l2_rectangular_map.coords
        self.assertIs(expected_coords.latitude_delta, actual_rectangular_data.coords.latitude_delta)
        self.assertIs(expected_coords.latitude_label, actual_rectangular_data.coords.latitude_label)
        self.assertIs(expected_coords.longitude_delta, actual_rectangular_data.coords.longitude_delta)
        self.assertIs(expected_coords.longitude_label, actual_rectangular_data.coords.longitude_label)
        # @formatter:on

        self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.ultra.ultra_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.ultra.ultra_processor.ExposureWeightedCombination")
    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3CombinedDependencies.fetch_dependencies')
    def _test_process_combined_sensor(self, degree_spacing, mock_fetch_dependencies, mock_save_data,
                                      mock_exposure_weighted_combination, _):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"ulc-ena-h-sf-nsp-full-hae-{degree_spacing}deg-6mo",
                                       )

        mock_fetch_dependencies.return_value = UltraL3CombinedDependencies(
            u45_dependencies=UltraL3Dependencies(
                ultra_l2_healpix_map=sentinel.u45_l2_healpix_map,
                ultra_l2_rectangular_map=sentinel.u45_l2_rectangular_map,
                ultra_l1c_pset=[],
                glows_l3e_sp=[],
                energy_bin_group_sizes=None,
                dependency_file_paths=[
                    Path("folder/u45_map"),
                    Path("folder/u45_l1c"),
                ],
            ),
            u90_dependencies=UltraL3Dependencies(
                ultra_l2_healpix_map=sentinel.u90_l2_healpix_map,
                ultra_l2_rectangular_map=sentinel.u90_l2_rectangular_map,
                ultra_l1c_pset=[],
                glows_l3e_sp=[],
                energy_bin_group_sizes=None,
                dependency_file_paths=[
                    Path("folder/u90_map"),
                    Path("folder/u90_l1c"),
                ],
            ),
        )

        healpix_combination_return_value = mock_exposure_weighted_combination.return_value.combine_healpix_intensity_map_data.return_value
        healpix_combination_return_value.intensity_map_data.survival_probability = None
        rectangular_combination_return_value = mock_exposure_weighted_combination.return_value.combine_rectangular_intensity_map_data.return_value

        healpix_skymap = healpix_combination_return_value.to_healpix_skymap.return_value
        converted_rectangular_skymap = Mock()
        healpix_skymap.to_rectangular_skymap.return_value = (
            converted_rectangular_skymap,
            sentinel.who_even_knows_what_this_is,
        )

        predicted_ephemeris_rectangular_values = np.array([0.243, 0, 1, 0, 0.993, 0])
        nominal_alpha_proton_ratio_rectangular_values = np.array([0, 0.34, 0.8, 0, 0, 0])
        persisted_last_point_rectangular_values = np.array([0, 0, 0.4, 0, 0, 1])

        expected_quality_flags = np.array([
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            MapL3Flags.PREDICTIVE_EPHEMERIS | MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PERSISTED_LAST_POINT,
            MapL3Flags.NONE,
            MapL3Flags.PREDICTIVE_EPHEMERIS,
            MapL3Flags.PERSISTED_LAST_POINT
        ])
        converted_rectangular_skymap.to_dataset.return_value = {
            "ena_intensity": Mock(values=sentinel.rectangular_ena_intensity),
            "ena_intensity_stat_uncert": Mock(values=sentinel.rectangular_ena_intensity_stat_uncert),
            "ena_intensity_sys_err": Mock(values=sentinel.rectangular_ena_intensity_sys_err),
            "predicted_ephemeris_flag": Mock(values=predicted_ephemeris_rectangular_values),
            "nominal_alpha_proton_ratio_flag": Mock(values=nominal_alpha_proton_ratio_rectangular_values),
            "persisted_last_point_flag": Mock(values=persisted_last_point_rectangular_values),
        }

        processor = UltraProcessor(sentinel.dependencies, input_metadata)
        product = processor.process(spice_frame_name=sentinel.spice_frame)

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_exposure_weighted_combination.return_value.combine_healpix_intensity_map_data.assert_called_once_with(
            [sentinel.u45_l2_healpix_map, sentinel.u90_l2_healpix_map])

        mock_exposure_weighted_combination.return_value.combine_rectangular_intensity_map_data.assert_called_once_with(
            [sentinel.u45_l2_rectangular_map, sentinel.u90_l2_rectangular_map]
        )

        healpix_combination_return_value.to_healpix_skymap.assert_called_once()

        healpix_skymap.to_rectangular_skymap.assert_called_once_with(
            degree_spacing,
            [
                "ena_intensity",
                "ena_intensity_stat_uncert",
                "ena_intensity_sys_err",
                "predicted_ephemeris_flag",
                "nominal_alpha_proton_ratio_flag",
                "persisted_last_point_flag",
            ]
        )

        actual_data_product = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(
            sentinel.rectangular_ena_intensity, actual_data_product.data.intensity_map_data.ena_intensity
        )
        self.assertEqual(
            sentinel.rectangular_ena_intensity_stat_uncert,
            actual_data_product.data.intensity_map_data.ena_intensity_stat_uncert,
        )
        self.assertEqual(
            sentinel.rectangular_ena_intensity_sys_err,
            actual_data_product.data.intensity_map_data.ena_intensity_sys_err,
        )
        np.testing.assert_array_equal(actual_data_product.data.intensity_map_data.quality_flags, expected_quality_flags)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.obs_date, actual_data_product.data.intensity_map_data.obs_date)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.obs_date_range, actual_data_product.data.intensity_map_data.obs_date_range)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.solid_angle, actual_data_product.data.intensity_map_data.solid_angle)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.exposure_factor, actual_data_product.data.intensity_map_data.exposure_factor)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.longitude, actual_data_product.data.intensity_map_data.longitude)
        self.assertEqual(rectangular_combination_return_value.intensity_map_data.latitude, actual_data_product.data.intensity_map_data.latitude)

        self.assertEqual([mock_save_data.return_value], product)
        self.assertEqual(['u45_map', 'u45_l1c', 'u90_map', 'u90_l1c'], actual_data_product.parent_file_names)

    @patch('imap_l3_processing.ultra.ultra_processor.correct_healpix_data_for_survival_probability')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraProcessor._process_healpix_intensity_to_rectangular')
    @patch('imap_l3_processing.ultra.ultra_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.ultra.ultra_processor.UncertaintyWeightedCombination")
    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3CombinedDependencies.fetch_dependencies')
    def _test_process_combined_sensor_survival_probability(self, degree_spacing, mock_fetch_dependencies,
                                                           mock_save_data, mock_uncertainty_weighted_combination,
                                                           mock_get_parent_file_names, mock_healpix_to_rectangular,
                                                           mock_correct_healpix_data):
        mock_get_parent_file_names.return_value = ["ram_map", "antiram_map"]
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"ulc-ena-h-sf-sp-full-hae-{degree_spacing}deg-6mo",
                                       )
        mock_fetch_dependencies.return_value = UltraL3CombinedDependencies(
            u45_dependencies=UltraL3Dependencies(
                ultra_l2_healpix_map=sentinel.u45_l2_healpix_map,
                ultra_l2_rectangular_map=sentinel.u45_l2_rectangular_map,
                ultra_l1c_pset=[sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3],
                glows_l3e_sp=[sentinel.glows_pset_1, sentinel.glows_pset_2, sentinel.glows_pset_3],
                energy_bin_group_sizes=sentinel.energy_bin_sizes,
                dependency_file_paths=[
                    Path("folder/u45_map"),
                    Path("folder/u45_l1c"),
                ],
            ),
            u90_dependencies=UltraL3Dependencies(
                ultra_l2_healpix_map=sentinel.u90_l2_healpix_map,
                ultra_l2_rectangular_map=sentinel.u90_l2_rectangular_map,
                ultra_l1c_pset=[sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3],
                glows_l3e_sp=[sentinel.glows_pset_1, sentinel.glows_pset_2, sentinel.glows_pset_3],
                energy_bin_group_sizes=sentinel.energy_bin_sizes,
                dependency_file_paths=[
                    Path("folder/u90_map"),
                    Path("folder/u90_l1c"),
                ],
            ),
        )

        mock_combination_strategy = Mock()
        mock_uncertainty_weighted_combination.return_value = mock_combination_strategy

        mock_correct_healpix_data.side_effect = [sentinel.u45_l2_survival_corrected_map,
                                                 sentinel.u90_l2_survival_corrected_map]

        processor = UltraProcessor(sentinel.dependencies, input_metadata)
        product = processor.process(spice_frame_name=sentinel.spice_frame)

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_combination_strategy.combine_healpix_intensity_map_data.assert_called_once_with(
            [sentinel.u45_l2_survival_corrected_map, sentinel.u90_l2_survival_corrected_map])
        mock_combination_strategy.combine_rectangular_intensity_map_data.assert_called_once_with(
            [sentinel.u45_l2_rectangular_map, sentinel.u90_l2_rectangular_map])

        mock_healpix_to_rectangular.assert_called_once_with(
            mock_combination_strategy.combine_healpix_intensity_map_data.return_value,
            mock_combination_strategy.combine_rectangular_intensity_map_data.return_value,
            degree_spacing,
            spice_frame_name=sentinel.spice_frame)

        mock_save_data.assert_called_once_with(mock_healpix_to_rectangular.return_value)
        self.assertEqual([mock_save_data.return_value], product)

        mock_correct_healpix_data.assert_has_calls([
            call(mock_fetch_dependencies.return_value.u45_dependencies, sentinel.spice_frame),
            call(mock_fetch_dependencies.return_value.u90_dependencies, sentinel.spice_frame)
        ])
        mock_healpix_to_rectangular.return_value.add_paths_to_parents.assert_has_calls([
            call([
                Path("folder/u45_map"),
                Path("folder/u45_l1c"),
            ]),
            call([
                Path("folder/u90_map"),
                Path("folder/u90_l1c"),
            ]),
        ])

    @patch('imap_l3_processing.ultra.ultra_processor.correct_healpix_data_for_survival_probability')
    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def test_defaults_to_ECLIPJ2000_spice_frame(self, mock_fetch_dependencies,
                                                mock_save_data, mock_spiceypy,
                                                mock_correct_for_survival_probability):
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()

        mock_spiceypy.ktotal.return_value = 1

        fake_spice = Path("path/to/fake/spice.tls")
        mock_spiceypy.kdata.return_value = [fake_spice]

        input_l2_map = _create_ultra_l2_healpix_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)
        input_l2_rectangular_map = create_rectangular_intensity_map_data()

        input_l2_map.intensity_map_data.energy = sentinel.ultra_l2_energies

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor_20250601_v000.cdf"
        input_l1c_pset_name = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_glows_l3e_name = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"

        input_deps = ProcessingInputCollection(ScienceInput(input_l2_map_name))

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_healpix_map=input_l2_map,
            ultra_l2_rectangular_map=input_l2_rectangular_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[Path(input_l2_map_name), Path(input_l1c_pset_name), Path(input_glows_l3e_name)],
            energy_bin_group_sizes=None
        )
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"u90-ena-h-sf-sp-full-hae-2deg-6mo"
                                       )

        mock_healpix_skymap = Mock()
        mock_correct_for_survival_probability.return_value.to_healpix_skymap = Mock(return_value=mock_healpix_skymap)

        mock_rectangular_map_dataset = {
            "ena_intensity": Mock(values=sentinel.rectangular_ena_intensity),
            "ena_intensity_stat_uncert": Mock(values=sentinel.rectangular_ena_intensity_stat_unc),
            "ena_intensity_sys_err": Mock(values=sentinel.rectangular_ena_intensity_sys_err),
            "survival_probability": Mock(values=sentinel.rectangular_survival_probability),
            "predicted_ephemeris_flag": Mock(values=np.array([])),
            "nominal_alpha_proton_ratio_flag": Mock(values=np.array([])),
            "persisted_last_point_flag": Mock(values=np.array([])),
        }

        mock_rectangular_sky_map = Mock(spec=RectangularSkyMap)
        mock_rectangular_sky_map.to_dataset.return_value = mock_rectangular_map_dataset
        mock_healpix_skymap.to_rectangular_skymap.return_value = mock_rectangular_sky_map, 0

        processor = UltraProcessor(input_deps, input_metadata)
        processor.process()

        mock_correct_for_survival_probability.assert_called_once_with(
            mock_fetch_dependencies.return_value,
            SpiceFrame.ECLIPJ2000
        )

    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.calculate_spectral_index_for_multiple_ranges')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def _test_process_spectral_index(self, degree_spacing,
                                     mock_fetch_dependencies,
                                     mock_calculate_spectral_index, mock_save_data,
                                     mock_spiceypy):

        mock_spiceypy.ktotal.return_value = 0

        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        energy_range_file_name = 'imap_ultra_energy-range-descriptor_20250601_v000.dat'
        input_deps = ProcessingInputCollection(ScienceInput(map_file_name), AncillaryInput(energy_range_file_name))

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-{degree_spacing}deg-6mo")
        input_map_data = RectangularIntensityMapData(Mock(), Mock())
        dependencies = UltraL3SpectralIndexDependencies(input_map_data, sentinel.energy_ranges)
        mock_fetch_dependencies.return_value = dependencies

        mock_spectral_index_map_data = Mock(spec=SpectralIndexMapData)
        mock_calculate_spectral_index.return_value = mock_spectral_index_map_data

        expected_parent_file_names = [map_file_name, energy_range_file_name]

        processor = UltraProcessor(input_deps, input_metadata)
        product = processor.process()

        mock_save_data.assert_called_once()
        actual_rectangular_data_product = mock_save_data.call_args_list[0].args[0]
        self.assertIsInstance(actual_rectangular_data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(expected_parent_file_names, actual_rectangular_data_product.parent_file_names)
        self.assertEqual(processor.input_metadata, actual_rectangular_data_product.input_metadata)
        self.assertEqual(SpiceFrame.ECLIPJ2000, actual_rectangular_data_product.spice_frame_name)

        self.assertIsInstance(actual_rectangular_data_product, RectangularSpectralIndexDataProduct)

        actual_rectangular_data: RectangularSpectralIndexMapData = actual_rectangular_data_product.data
        self.assertIsInstance(actual_rectangular_data_product.data, RectangularSpectralIndexMapData)
        self.assertIs(mock_spectral_index_map_data, actual_rectangular_data.spectral_index_map_data)
        self.assertIs(input_map_data.coords, actual_rectangular_data.coords)

        mock_fetch_dependencies.assert_called_once_with(input_deps)
        mock_calculate_spectral_index.assert_called_once_with(dependencies.map_data.intensity_map_data,
                                                              sentinel.energy_ranges)
        self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.ultra_processor.calculate_spectral_index_for_multiple_ranges')
    @patch('imap_l3_processing.ultra.ultra_processor.fit_spectral_index_map')
    @patch('imap_l3_processing.ultra.ultra_processor.slice_energy_range_by_bin')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def test_process_spectral_index_with_custom_energy_bin_range(self,
                                     mock_fetch_dependencies, mock_slice_energy_range_by_bin, mock_fit_spectral_index_map,
                                     mock_calculate_spectral_index_for_multiple_ranges, mock_spiceypy, mock_save_data
                                     ):

        mock_spiceypy.ktotal.return_value = 0

        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        energy_range_file_name = 'imap_ultra_energy-range-descriptor_20250601_v000.dat'
        input_deps = ProcessingInputCollection(ScienceInput(map_file_name), AncillaryInput(energy_range_file_name))

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx0510-h-sf-sp-full-hae-4deg-6mo")
        dependencies = mock_fetch_dependencies.return_value

        processor = UltraProcessor(input_deps, input_metadata)
        result = processor.process()

        mock_slice_energy_range_by_bin.assert_called_once_with(dependencies.map_data.intensity_map_data, 5, 10)
        mock_fit_spectral_index_map.assert_called_once_with(mock_slice_energy_range_by_bin.return_value)

        mock_calculate_spectral_index_for_multiple_ranges.assert_not_called()
        mock_save_data.assert_called_once()
        [product] = mock_save_data.call_args.args
        self.assertIsInstance(product, RectangularSpectralIndexDataProduct)

        self.assertEqual(mock_fit_spectral_index_map.return_value, product.data.spectral_index_map_data)

    def test_process_raises_exception_when_generating_a_healpix_map(self):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-nside8-6mo")

        processor = UltraProcessor(ProcessingInputCollection(), input_metadata)

        with self.assertRaises(NotImplementedError):
            processor.process()

    @patch('imap_l3_processing.ultra.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def test_process_spectral_index_validating_output_values(self, mock_fetch_dependencies, mock_save_data):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-6deg-6mo")
        input_map_path = get_test_data_path('ultra/fake_ultra_map_data_with_breakpoint_at_15keV.cdf')
        fit_energy_ranges_path = get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
        dependencies = UltraL3SpectralIndexDependencies.from_file_paths(input_map_path, fit_energy_ranges_path)
        mock_fetch_dependencies.return_value = dependencies

        expected_ena_spectral_index = np.array([2] * (60 * 30) + [3.5] * (60 * 30)).reshape(1, 2, 60, 30)

        processing_input_collection = ProcessingInputCollection()
        processor = UltraProcessor(processing_input_collection, input_metadata)
        product = processor.process()

        actual_data_product: RectangularSpectralIndexDataProduct = mock_save_data.call_args[0][0]

        np.testing.assert_array_almost_equal(actual_data_product.data.spectral_index_map_data.ena_spectral_index,
                                             expected_ena_spectral_index)
        self.assertEqual([mock_save_data.return_value], product)

def _create_ultra_l2_healpix_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                                  intensity_stat_unc=None, healpix_indices=None) -> HealPixIntensityMapData:
    epoch = epoch if epoch is not None else np.array([datetime.now()])
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    healpix_indices = healpix_indices if healpix_indices is not None else np.arange(12)
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(healpix_indices)), fill_value=1)
    intensity_stat_uncert = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        flux.shape,
        fill_value=1)

    if isinstance(flux, np.ndarray):
        more_real_flux = flux
    else:
        more_real_flux = np.full((len(epoch), len(lon), len(lat), 9), fill_value=1)

    return HealPixIntensityMapData(
        IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([0]),
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(flux, 0),
            obs_date=np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1)),
            obs_date_range=np.full_like(more_real_flux, 0),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_intensity=flux,
            ena_intensity_stat_uncert=intensity_stat_uncert,
            ena_intensity_sys_err=np.full_like(flux, 0),
            quality_flags=np.full_like(flux, MapL3Flags.NONE)
        ),
        HealPixCoords(
            pixel_index=healpix_indices,
            pixel_index_label=np.full(healpix_indices.shape, "healpix index label")
        )
    )