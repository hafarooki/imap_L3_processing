import pickle
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
from imap_data_access import ScienceInput, AncillaryInput, ProcessingInputCollection

from imap_l3_processing.ultra.ultra_l3_dependencies import (
    UltraL3Dependencies,
    UltraL3SpectralIndexDependencies,
    UltraL3CombinedDependencies,
    load_or_create_healpix_l2,
)
from imap_l3_processing.utils import clear_temp_cache, get_temp_cache_dir
from tests.test_helpers import get_test_data_path

class TestUltraL3Dependencies(unittest.TestCase):
    def setUp(self):
        clear_temp_cache()
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ScienceInput')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ultra_l2')
    def test_fetch_dependencies(self, mock_ultra_l2, mock_science_input_constructor, mock_download, mock_read_xarray_healpix, mock_read_from_path_rectangular, mock_read_ultra_l1c,
                                mock_read_glows):
        l1c_input_paths = [Path("imap_ultra_l1c_45sensor-pset_20251010_v001.cdf"),
                           Path("imap_ultra_l1c_45sensor-pset_20251011_v001.cdf"),
                           Path("imap_ultra_l1c_45sensor-pset_20251012_v001.cdf")]

        glows_file_paths = [
            "imap_glows_l3e_survival-probability-ul-sf_20201001_v001.cdf",
            "imap_glows_l3e_survival-probability-ul-sf_20201002_v002.cdf",
            "imap_glows_l3e_survival-probability-ul-sf_20201003_v003.cdf"]

        l2_energy_bin_group_sizes = 'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
        l2_energy_bin_group_sizes_path = get_test_data_path(
            'ultra/imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv')

        input_collection = Mock()
        input_collection.get_file_paths.side_effect = [
            [sentinel.l2_server_path],
            l1c_input_paths,
            glows_file_paths,
            [l2_energy_bin_group_sizes]
        ]

        returned_download_paths = [sentinel.l2_map_path, l1c_input_paths[0], l1c_input_paths[1], l1c_input_paths[2],
                                   sentinel.glows_path_1, sentinel.glows_path_2, sentinel.glows_path_3,
                                   l2_energy_bin_group_sizes_path]

        mock_download.side_effect = returned_download_paths
        mock_science_input_constructor.return_value = Mock(descriptor="ultra-nsp-descriptor-6deg")

        l1c_data = [sentinel.ultra_l1c_data_1, sentinel.ultra_l1c_data_2,
                    sentinel.ultra_l1c_data_3]
        mock_read_ultra_l1c.side_effect = l1c_data

        glows_l3e_data = [sentinel.glows_data_1, sentinel.glows_data_2, sentinel.glows_data_3]
        mock_read_glows.side_effect = glows_l3e_data

        mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map_dataset]
        mock_read_xarray_healpix.return_value = sentinel.ultra_l2_healpix_data
        mock_read_from_path_rectangular.return_value = sentinel.ultra_l2_rectangular_data

        dependencies = UltraL3Dependencies.fetch_dependencies(input_collection)

        mock_science_input_constructor.assert_called_with(sentinel.l2_map_path.name)

        input_collection.get_file_paths.assert_has_calls([
            call("ultra", data_type="l2"),
            call("ultra", data_type="l1c"),
            call("glows"),
            call(data_type="ancillary", descriptor="l2-energy-bin-group-sizes")
        ])
        expected_data_dictionary = {"imap_ultra_l1c_45sensor-pset_20251010_v001": l1c_input_paths[0],
                                    "imap_ultra_l1c_45sensor-pset_20251011_v001": l1c_input_paths[1],
                                    "imap_ultra_l1c_45sensor-pset_20251012_v001": l1c_input_paths[2]}
        mock_ultra_l2.assert_has_calls([call(expected_data_dictionary, descriptor="ultra-nsp-descriptor-nside32")])


        expected_parent_file_paths = [sentinel.l2_server_path, *l1c_input_paths, *glows_file_paths,
                                      l2_energy_bin_group_sizes]
        mock_download.assert_has_calls([call(file_path) for file_path in
                                        expected_parent_file_paths])

        mock_read_xarray_healpix.assert_called_once_with(sentinel.ultra_l2_healpix_map_dataset)
        mock_read_from_path_rectangular.assert_called_once_with(sentinel.l2_map_path)

        self.assertEqual(l1c_data, dependencies.ultra_l1c_pset)
        self.assertEqual(glows_l3e_data, dependencies.glows_l3e_sp)
        self.assertEqual(sentinel.ultra_l2_healpix_data, dependencies.ultra_l2_healpix_map)
        self.assertEqual(sentinel.ultra_l2_rectangular_data, dependencies.ultra_l2_rectangular_map)
        self.assertEqual(returned_download_paths, dependencies.dependency_file_paths)
        np.testing.assert_array_equal(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46], dtype=np.uint8),
                                      dependencies.energy_bin_group_sizes, strict=True)

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.load_or_create_healpix_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    def test_fetch_dependencies_without_energy_ancillary_file(self,  mock_download,
                                                              mock_read_from_path, mock_read_ultra_l1c,
                                                              mock_read_glows,
                                                              mock_load_or_create_healpix_l2):
        l1c_input_paths = ["imap_ultra_l1c_pset_20251010_v001.cdf", "imap_ultra_l1c_pset_20251011_v001.cdf",
                           "imap_ultra_l1c_pset_20251012_v001.cdf"]

        glows_file_paths = [
            "imap_glows_l3e_survival-probability-ul-sf_20201001_v001.cdf",
            "imap_glows_l3e_survival-probability-ul-sf_20201002_v002.cdf",
            "imap_glows_l3e_survival-probability-ul-sf_20201003_v003.cdf"]

        input_collection = Mock()
        input_collection.get_file_paths.side_effect = [
            [sentinel.l2_server_path],
            l1c_input_paths,
            glows_file_paths,
            []
        ]
        mock_download.return_value = Path("downloaded_file")
        dependencies = UltraL3Dependencies.fetch_dependencies(input_collection)

        self.assertEqual(None, dependencies.energy_bin_group_sizes)

    def test_raise_error_for_more_than_one_input_files_paths(self):
        input_collection = Mock()
        input_collection.get_file_paths.return_value = [sentinel.imap_l2_map_path_1, sentinel.imap_l2_map_path_2]

        with self.assertRaises(AssertionError) as e:
            UltraL3Dependencies.fetch_dependencies(input_collection)

        self.assertEqual("Incorrect number of map dependencies: 2", str(e.exception))

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.load_or_create_healpix_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    def test_from_file_paths(
        self,
        mock_read_l1c: Mock,
        mock_read_glows: Mock,
        mock_read_l2_from_path: Mock,
        mock_load_or_create_healpix_l2: Mock,
    ):
        ultra_l2_input_path = Path("um_path")
        ultra_l1c_input_paths = [Path("imap/u_path_3"), Path("imap/u_path_1"), Path("imap/u_path_2")]
        glows_input_paths = [Path("g_path_1"), Path("g_path_2")]
        mock_read_l1c.side_effect = [sentinel.ultra_data3, sentinel.ultra_data1, sentinel.ultra_data2]
        mock_read_glows.side_effect = [sentinel.glows_data1, sentinel.glows_data2]
        mock_read_l2_from_path.return_value = sentinel.ultra_l2_rectangular_data
        mock_load_or_create_healpix_l2.return_value = sentinel.ultra_l2_healpix_data

        result = UltraL3Dependencies.from_file_paths(ultra_l2_input_path, ultra_l1c_input_paths, glows_input_paths,
                                                     None)

        mock_read_l2_from_path.assert_called_with(ultra_l2_input_path)
        mock_read_l1c.assert_has_calls(
            [call(file_path) for file_path in ultra_l1c_input_paths]
        )
        mock_read_glows.assert_has_calls([call(file_path) for file_path in glows_input_paths])
        mock_load_or_create_healpix_l2.assert_called_with(ultra_l2_input_path, ultra_l1c_input_paths)

        self.assertEqual(result.ultra_l1c_pset, [sentinel.ultra_data3, sentinel.ultra_data1, sentinel.ultra_data2])
        self.assertEqual(result.glows_l3e_sp, [sentinel.glows_data1, sentinel.glows_data2])
        self.assertEqual(result.ultra_l2_healpix_map, sentinel.ultra_l2_healpix_data)
        self.assertEqual(result.ultra_l2_rectangular_map, sentinel.ultra_l2_rectangular_data)

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ultra_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ScienceInput')
    def test_load_or_create_healpix_l2_saves_to_cache(
        self,
        mock_science_input_constructor,
        mock_ultra_l2,
        mock_read_l2_xarray: Mock,
    ):
        ultra_l2_input_path = Path("um_path")
        ultra_l1c_input_paths = [Path("imap/u_path_3"), Path("imap/u_path_1"), Path("imap/u_path_2")]
        mock_read_l2_xarray.return_value = sentinel.ultra_l2_healpix_data
        l2_ultra_descriptor = "u45-ena-h-hf-nsp-full-hae-6deg"
        mock_science_input_constructor.return_value = Mock(descriptor=l2_ultra_descriptor)
        mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map]

        result = load_or_create_healpix_l2(ultra_l2_input_path, ultra_l1c_input_paths)

        mock_read_l2_xarray.assert_called_with(sentinel.ultra_l2_healpix_map)
        mock_ultra_l2.assert_called_with(
            {
                "u_path_1": Path("imap/u_path_1"),
                "u_path_2": Path("imap/u_path_2"),
                "u_path_3": Path("imap/u_path_3"),
            },
            descriptor="u45-ena-h-hf-nsp-full-hae-nside32",
        )
        self.assertEqual(["u_path_1", "u_path_2", "u_path_3"], list(mock_ultra_l2.call_args.args[0].keys()))

        self.assertEqual(result, sentinel.ultra_l2_healpix_data)

        temp_cache_dir = get_temp_cache_dir()
        raw_cache_key = (
            "['u_path_1', 'u_path_2', 'u_path_3']u45-ena-h-hf-nsp-full-hae-nside32"
        )
        cached_healpix_filename = sha256(raw_cache_key.encode("utf-8")).hexdigest()
        with open(temp_cache_dir / cached_healpix_filename, "rb") as f:
            cached_healpix = pickle.load(f)
        self.assertEqual(sentinel.ultra_l2_healpix_data, cached_healpix)

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ultra_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.ScienceInput')
    def test_load_or_create_healpix_l2_loads_from_cache(
        self,
        mock_science_input_constructor,
        mock_ultra_l2,
        mock_read_l2_xarray: Mock,
    ):
        temp_cache_dir = get_temp_cache_dir()
        raw_cache_key = "['l1c_path_1.cdf', 'l1c_path_2.cdf']u45-ena-h-hf-nsp-full-hae-nside32"
        cached_healpix_filename = sha256(raw_cache_key.encode("utf-8")).hexdigest()
        with open(temp_cache_dir / cached_healpix_filename, "wb") as f:
            pickle.dump(sentinel.ultra_l2_healpix_data, f)
        ultra_l2_input_path = Path("um_path")
        ultra_l1c_input_paths = [Path("imap/l1c_path_1.cdf"), Path("imap/l1c_path_2.cdf")]
        l2_ultra_descriptor = "u45-ena-h-hf-nsp-full-hae-6deg"
        mock_science_input_constructor.return_value = Mock(descriptor=l2_ultra_descriptor)

        result = load_or_create_healpix_l2(ultra_l2_input_path, ultra_l1c_input_paths)

        mock_read_l2_xarray.assert_not_called()
        mock_ultra_l2.assert_not_called()

        self.assertEqual(result, sentinel.ultra_l2_healpix_data)


    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    def test_spectral_index_fetch_dependencies(self, mock_download, mock_rectangular_read_from_path):

        for input_data_type in ["l2", "l3"]:
            mock_rectangular_read_from_path.reset_mock()
            with self.subTest(input_data_type):
                map_file_name = f'imap_ultra_{input_data_type}_ultra-cool-descriptor_20250601_v000.cdf'
                ancillary_file_name = 'imap_ultra_spx-energy-ranges_20250601_v000.cdf'
                pointing_set_file_name = 'imap_ultra_l1c_u90spacecraftpset_20250601-repoint00001_v000.cdf'
                glows_file_name = 'imap_glows_l3e_survival-probability-ul-hf_20250601-repoint00001_v000.cdf'
                expected_energy_ranges = [[5, 15], [15, 50]]
                map_input = ScienceInput(map_file_name)
                ancillary_input = AncillaryInput(ancillary_file_name)
                extra_inputs_from_mapper = [ScienceInput(pointing_set_file_name), ScienceInput(glows_file_name)]
                processing_input_collection = ProcessingInputCollection(map_input, ancillary_input, *extra_inputs_from_mapper)

                mock_download.side_effect = [
                    "map_file",
                    get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
                ]

                ultra_l3_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(processing_input_collection)

                mock_download.assert_has_calls([
                    call(map_file_name),
                    call(ancillary_file_name)
                ])
                mock_rectangular_read_from_path.assert_called_once_with("map_file")
                self.assertEqual(ultra_l3_dependencies.map_data, mock_rectangular_read_from_path.return_value)
                np.testing.assert_array_equal(ultra_l3_dependencies.fit_energy_ranges, expected_energy_ranges)

    def test_spectral_index_fetch_dependencies_raises_exception_on_missing_map(self):
        ancillary_input = AncillaryInput('imap_ultra_spx-energy-ranges_20250601_v000.dat')
        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(ancillary_input))
        self.assertEqual("Expected 1 input map, got 0: []", str(context.exception))

    def test_spectral_index_fetch_dependencies_raises_exception_on_multiple_maps(self):
        ancillary_input = AncillaryInput('imap_ultra_spx-energy-ranges_20250601_v000.dat')
        science_input = ScienceInput('imap_ultra_l3_map_20250601_v000.cdf')

        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(ancillary_input, science_input, science_input))
        expected_msg = "Expected 1 input map, got 2: [imap_ultra_l3_map_20250601_v000.cdf, imap_ultra_l3_map_20250601_v000.cdf]"
        self.assertEqual(expected_msg, str(context.exception))

    def test_spectral_index_fetch_dependencies_raises_exception_on_missing_ancillary_file(self):
        science_input = ScienceInput('imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf')

        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(science_input))
        self.assertEqual("Missing fit energy ranges ancillary file", str(context.exception))

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    def test_spectral_index_from_file_paths(self, mock_read_from_path):
        map_file_path = Mock()
        ancillary_file_path = get_test_data_path('ultra') / 'imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat'
        expected_energy_range_values = np.loadtxt(ancillary_file_path)

        actual_dependencies = UltraL3SpectralIndexDependencies.from_file_paths(map_file_path, ancillary_file_path)
        self.assertEqual(mock_read_from_path.return_value, actual_dependencies.map_data)
        np.testing.assert_array_equal(actual_dependencies.fit_energy_ranges, expected_energy_range_values)

    def test_spectral_index_get_fit_energy_ranges(self):
        expected_energy_range_values = np.array([[5, 10], [15, 20]])

        dependencies = UltraL3SpectralIndexDependencies(map_data=Mock(),
                                                        fit_energy_ranges=expected_energy_range_values)
        actual_fit_energy_ranges = dependencies.get_fit_energy_ranges()

        np.testing.assert_array_equal(actual_fit_energy_ranges, expected_energy_range_values)

class TestUltraL3CombinedDependencies(unittest.TestCase):
    def setUp(self):
        clear_temp_cache()
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.load_or_create_healpix_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    def test_fetch_dependencies_no_survival_corrected(self, mock_download,
                                                      mock_l2_read_from_path, mock_l1c_read_from_path,
                                                      mock_load_or_create_healpix_l2,
                                                      ):
        u45_map_file_name = 'imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf'
        u90_map_file_name = 'imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf'
        u45_pset_file_name = ["imap_ultra_l1c_45sensor-spacecraftpset_20251010_v001.cdf", "imap_ultra_l1c_45sensor-spacecraftpset_20251011_v001.cdf",
                           "imap_ultra_l1c_45sensor-spacecraftpset_20251012_v001.cdf"]
        u90_pset_file_name = ["imap_ultra_l1c_90sensor-spacecraftpset_20251010_v001.cdf", "imap_ultra_l1c_90sensor-spacecraftpset_20251011_v001.cdf",
                           "imap_ultra_l1c_90sensor-spacecraftpset_20251012_v001.cdf"]
        l2_energy_bin_group_sizes_file_name = 'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
        l2_energy_bin_group_sizes_file_path = get_test_data_path("ultra/" + l2_energy_bin_group_sizes_file_name)

        u45_map_input = ScienceInput(u45_map_file_name)
        u90_map_input = ScienceInput(u90_map_file_name)
        u45_pset_inputs = [ScienceInput(pset) for pset in u45_pset_file_name]
        u90_pset_inputs = [ScienceInput(pset) for pset in u90_pset_file_name]
        ancillary_inputs = AncillaryInput(l2_energy_bin_group_sizes_file_name)

        processing_input_collection = ProcessingInputCollection(u45_map_input, u90_map_input, *u45_pset_inputs,
                                                                *u90_pset_inputs, ancillary_inputs)

        u45_l2_path = Path("imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf")
        u90_l2_path = Path("imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf")

        expected_file_paths = [
            Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf"),
            u45_l2_path,
            u90_l2_path,
            l2_energy_bin_group_sizes_file_path
        ]

        mock_download.side_effect = expected_file_paths

        mock_l1c_read_from_path.side_effect = [
            sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3,
            sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3,
        ]

        mock_l2_read_from_path.side_effect = [sentinel.u45_rectangular_l2, sentinel.u90_rectangular_l2]

        mock_load_or_create_healpix_l2.side_effect = [
            sentinel.u45_healpix_dataset,
            sentinel.u90_healpix_dataset
        ]

        combined_dependencies = UltraL3CombinedDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            *[call(filename) for filename in u45_pset_file_name],
            *[call(filename) for filename in u90_pset_file_name],
            call(u45_map_file_name),
            call(u90_map_file_name),
            call(l2_energy_bin_group_sizes_file_name)
        ])

        mock_l1c_read_from_path.assert_has_calls(
            [
                call(Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf")),
                call(Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf")),
                call(Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf")),
                call(Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf")),
                call(Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf")),
                call(Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf")),
            ]
        )

        mock_l2_read_from_path.assert_has_calls([call(u45_l2_path), call(u90_l2_path)])

        mock_load_or_create_healpix_l2.assert_has_calls(
            [
                call(u45_l2_path, expected_file_paths[:3]),
                call(u90_l2_path, expected_file_paths[3:6]),
            ]
        )
        self.assertIsInstance(combined_dependencies, UltraL3CombinedDependencies)
        u45_deps = combined_dependencies.u45_dependencies
        u90_deps = combined_dependencies.u90_dependencies
        self.assertIsInstance(u45_deps, UltraL3Dependencies)
        self.assertIsInstance(u90_deps, UltraL3Dependencies)

        self.assertEqual(u45_deps.ultra_l2_healpix_map, sentinel.u45_healpix_dataset)
        self.assertEqual(u90_deps.ultra_l2_healpix_map, sentinel.u90_healpix_dataset)
        self.assertEqual(u45_deps.ultra_l2_rectangular_map, sentinel.u45_rectangular_l2)
        self.assertEqual(u90_deps.ultra_l2_rectangular_map, sentinel.u90_rectangular_l2)
        self.assertEqual(u45_deps.ultra_l1c_pset, [sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3])
        self.assertEqual(u90_deps.ultra_l1c_pset, [sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3])
        self.assertEqual(u45_deps.glows_l3e_sp, [])
        self.assertEqual(u90_deps.glows_l3e_sp, [])

        np.testing.assert_array_equal(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46], dtype=np.uint8),
                                      u45_deps.energy_bin_group_sizes, strict=True)
        np.testing.assert_array_equal(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46], dtype=np.uint8),
                                      u90_deps.energy_bin_group_sizes, strict=True)

        expected_45_file_paths = [
            u45_l2_path,
            Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf"),
            l2_energy_bin_group_sizes_file_path
        ]
        expected_90_file_paths = [
            u90_l2_path,
            Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf"),
            l2_energy_bin_group_sizes_file_path
        ]
        self.assertEqual(u45_deps.dependency_file_paths, expected_45_file_paths)
        self.assertEqual(u90_deps.dependency_file_paths, expected_90_file_paths)

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.load_or_create_healpix_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    def test_fetch_dependencies_without_ancillary_file(self,  mock_download,
                                                       mock_l1c_read_from_path, mock_l2_read_from_path,
                                                       mock_load_or_create_healpix_l2):
        u45_map_file_name = 'imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf'
        u90_map_file_name = 'imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf'
        u45_pset_file_name = ["imap_ultra_l1c_45sensor-spacecraftpset_20251010_v001.cdf",
                              "imap_ultra_l1c_45sensor-spacecraftpset_20251011_v001.cdf",
                              "imap_ultra_l1c_45sensor-spacecraftpset_20251012_v001.cdf"]
        u90_pset_file_name = ["imap_ultra_l1c_90sensor-spacecraftpset_20251010_v001.cdf",
                              "imap_ultra_l1c_90sensor-spacecraftpset_20251011_v001.cdf",
                              "imap_ultra_l1c_90sensor-spacecraftpset_20251012_v001.cdf"]

        u45_map_input = ScienceInput(u45_map_file_name)
        u90_map_input = ScienceInput(u90_map_file_name)
        u45_pset_inputs = [ScienceInput(pset) for pset in u45_pset_file_name]
        u90_pset_inputs = [ScienceInput(pset) for pset in u90_pset_file_name]

        processing_input_collection = ProcessingInputCollection(u45_map_input, u90_map_input, *u45_pset_inputs,
                                                                *u90_pset_inputs)

        expected_file_paths = [
            Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf"),
            Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf"),
            Path("imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf"),
            Path("imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf"),
        ]

        mock_download.side_effect = expected_file_paths

        combined_dependencies = UltraL3CombinedDependencies.fetch_dependencies(processing_input_collection)

        self.assertIsNone(combined_dependencies.u45_dependencies.energy_bin_group_sizes)
        self.assertIsNone(combined_dependencies.u90_dependencies.energy_bin_group_sizes)

    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.load_or_create_healpix_l2')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.imap_data_access.download')
    @patch('imap_l3_processing.ultra.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    def test_fetch_dependencies_with_glows_files(self, mock_l2_read_from_paths, mock_download,
                                                 mock_l1c_read_from_path, mock_l3e_read_from_path,
                                                 mock_load_or_create_healpix_l2,
                                                 ):
        cases = [["sf", "spacecraftpset"], ["hf", "heliopset"]]
        for frame, pset in cases:
            with (self.subTest(frame)):
                u45_map_file_name = 'imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf'
                u90_map_file_name = 'imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf'
                u45_pset_file_names = [f"imap_ultra_l1c_45sensor-{pset}_20251010_v001.cdf",
                                       f"imap_ultra_l1c_45sensor-{pset}_20251011_v001.cdf",
                                       f"imap_ultra_l1c_45sensor-{pset}_20251012_v001.cdf"]
                u90_pset_file_names = [f"imap_ultra_l1c_90sensor-{pset}_20251010_v001.cdf",
                                       f"imap_ultra_l1c_90sensor-{pset}_20251011_v001.cdf",
                                       f"imap_ultra_l1c_90sensor-{pset}_20251012_v001.cdf"]
                glows_file_names = [f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf",
                                    f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf",
                                    f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf"]

                u45_map_input = ScienceInput(u45_map_file_name)
                u90_map_input = ScienceInput(u90_map_file_name)
                u45_pset_inputs = [ScienceInput(pset) for pset in u45_pset_file_names]
                u90_pset_inputs = [ScienceInput(pset) for pset in u90_pset_file_names]
                glows_inputs = [ScienceInput(pset) for pset in glows_file_names]

                processing_input_collection = ProcessingInputCollection(
                    u45_map_input,
                    u90_map_input,
                    *u45_pset_inputs,
                    *u90_pset_inputs,
                    *glows_inputs,
                )
                u45_l1c_paths = [
                    Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf"),
                    Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf"),
                    Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf"),
                ]
                u90_l1c_paths = [
                    Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf"),
                    Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf"),
                    Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf"),
                ]
                u45_l2_path = Path("imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf")
                u90_l2_path = Path("imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf")
                expected_file_paths = [
                    *u45_l1c_paths,
                    *u90_l1c_paths,
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf"),
                    u45_l2_path,
                    u90_l2_path,
                ]

                mock_download.side_effect = expected_file_paths

                mock_l1c_read_from_path.side_effect = [
                    sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3,
                    sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3,
                ]

                mock_load_or_create_healpix_l2.side_effect = [
                    sentinel.u45_healpix_dataset,
                    sentinel.u90_healpix_dataset
                ]
                mock_l2_read_from_paths.side_effect = [sentinel.u45_rectangular_l2, sentinel.u90_rectangular_l2]

                mock_l3e_read_from_path.side_effect = [
                    sentinel.glows_1, sentinel.glows_2, sentinel.glows_3,
                    sentinel.glows_1, sentinel.glows_2, sentinel.glows_3,
                ]

                combined_dependencies = UltraL3CombinedDependencies.fetch_dependencies(processing_input_collection)

                mock_download.assert_has_calls([
                    *[call(filename) for filename in u45_pset_file_names],
                    *[call(filename) for filename in u90_pset_file_names],
                    *[call(filename) for filename in glows_file_names],
                    call(u45_map_file_name),
                    call(u90_map_file_name)
                ])

                mock_l1c_read_from_path.assert_has_calls([
                    call(Path("imap_ultra_l1c_u45-pset_20251010_v001.cdf")),
                    call(Path("imap_ultra_l1c_u45-pset_20251011_v001.cdf")),
                    call(Path("imap_ultra_l1c_u45-pset_20251012_v001.cdf")),
                    call(Path("imap_ultra_l1c_u90-pset_20251010_v001.cdf")),
                    call(Path("imap_ultra_l1c_u90-pset_20251011_v001.cdf")),
                    call(Path("imap_ultra_l1c_u90-pset_20251012_v001.cdf")),
                ])

                mock_l3e_read_from_path.assert_has_calls([
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf")),
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf")),
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf")),
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf")),
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf")),
                    call(Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf")),
                ])

                mock_l2_read_from_paths.assert_has_calls([
                    call(u45_l2_path),
                    call(u90_l2_path),
                ])

                mock_load_or_create_healpix_l2.assert_has_calls([
                    call(u45_l2_path, u45_l1c_paths),
                    call(u90_l2_path, u90_l1c_paths),
                ])
                u45_deps = combined_dependencies.u45_dependencies
                u90_deps = combined_dependencies.u90_dependencies
                self.assertEqual(u45_deps.ultra_l2_healpix_map, sentinel.u45_healpix_dataset)
                self.assertEqual(u90_deps.ultra_l2_healpix_map, sentinel.u90_healpix_dataset)
                self.assertEqual(u45_deps.ultra_l2_rectangular_map, sentinel.u45_rectangular_l2)
                self.assertEqual(u90_deps.ultra_l2_rectangular_map, sentinel.u90_rectangular_l2)
                self.assertEqual(u45_deps.ultra_l1c_pset,
                                 [sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3])
                self.assertEqual(u90_deps.ultra_l1c_pset,
                                 [sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3])
                self.assertEqual(u45_deps.glows_l3e_sp,
                                 [sentinel.glows_1, sentinel.glows_2, sentinel.glows_3])
                self.assertEqual(u90_deps.glows_l3e_sp,
                                 [sentinel.glows_1, sentinel.glows_2, sentinel.glows_3])
                expected_u45_file_paths = [
                    u45_l2_path,
                    *u45_l1c_paths,
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf"),
                ]
                expected_u90_file_paths = [
                    u90_l2_path,
                    *u90_l1c_paths,
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251010_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251011_v001.cdf"),
                    Path(f"imap_glows_l3e_survival-probability-ul-{frame}_20251012_v001.cdf"),
                ]
                self.assertEqual(u45_deps.dependency_file_paths, expected_u45_file_paths)
                self.assertEqual(u90_deps.dependency_file_paths, expected_u90_file_paths)

