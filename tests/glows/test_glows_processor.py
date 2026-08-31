import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import patch, Mock, sentinel, call, MagicMock
from zipfile import ZIP_DEFLATED

import imap_data_access
import numpy as np
from imap_data_access import AncillaryFilePath
from imap_data_access.file_validation import Version
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows import l3d
from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, \
    GLOWS_L3E_ULTRA_HF_DESCRIPTOR, GLOWS_L3B_DESCRIPTOR, GLOWS_L3C_DESCRIPTOR, GLOWS_L3D_DESCRIPTOR, \
    GLOWS_L3E_HI_45_DESCRIPTOR, GLOWS_L3E_HI_90_DESCRIPTOR, GLOWS_L3E_LO_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor, process_l3d, process_l3e, process_l3bc, \
    process_l3e_ul_sf, process_l3e_hi, process_l3e_lo, process_l3e_ul_hf
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_initializer import GlowsL3BCInitializerData
from imap_l3_processing.glows.l3bc.models import ExternalDependencies
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializerOutput
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.glows.l3e.glows_l3e_utils import GlowsL3eVersionsForRepointings, LoPivotAngle
from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.utils import save_data
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path, get_test_data_folder, \
    assert_dataclass_fields

MODULE = 'imap_l3_processing.glows.glows_processor'

class TestGlowsProcessor(unittest.TestCase):

    def setUp(self):
        self.l3bc_initializer_patcher = patch(f"{MODULE}.GlowsL3BCInitializer.get_crs_to_process")
        self.mock_l3bc_initializer = self.l3bc_initializer_patcher.start()

        self.mock_external_deps = Mock()
        self.mock_l3bc_initializer.return_value = GlowsL3BCInitializerData(
            external_dependencies=self.mock_external_deps,
            l3bc_dependencies=[],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=Path("imap_2001_052_001.repoint.csv"),
        )

        self.l3d_initializer_patcher = patch(f"{MODULE}.GlowsL3DInitializer")
        self.mock_l3d_initializer = self.l3d_initializer_patcher.start()
        self.mock_l3d_initializer.should_process_l3d.return_value = None

        self.mock_l3e_initializer_patcher = patch(f"{MODULE}.GlowsL3EInitializer")
        self.mock_l3e_initializer = self.mock_l3e_initializer_patcher.start()
        self.mock_l3e_initializer.get_repointings_to_process.return_value = GlowsL3EInitializerOutput(
            dependencies=Mock(),
            repointings=GlowsL3eVersionsForRepointings(
                repointing_numbers=[],
                hi_90_repointings={},
                hi_45_repointings={},
                lo_repointings={},
                ultra_sf_repointings={},
                ultra_hf_repointings={},
            ),
            l3d_cdf_path=Path("path/to/l3d.cdf"),
            metakernel_with_predict_ephem=Mock(),
            metakernel_without_predict_ephem=Mock(),
        )

        self.fetch_reprocess_info_patcher = patch(f"{MODULE}.fetch_reprocess_info")
        self.mock_fetch_reprocess_info = self.fetch_reprocess_info_patcher.start()
        self.mock_reprocess_info = self.mock_fetch_reprocess_info.return_value

    def tearDown(self):
        self.l3bc_initializer_patcher.stop()
        self.l3d_initializer_patcher.stop()
        self.mock_l3e_initializer_patcher.stop()
        self.fetch_reprocess_info_patcher.stop()

        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3b'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3b')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3c'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3c')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')

    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.utils.spiceypy')
    def test_processor_handles_l3a(self, mock_spiceypy, mock_save_data, mock_l3a_data,
                                   mock_glows_dependencies_class, mock_glows_initializer):
        mock_spiceypy.ktotal.return_value = 0
        instrument = 'glows'
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        outgoing_data_level = "l3a"
        outgoing_version = 'v002'

        mock_fetched_dependencies = mock_glows_dependencies_class.fetch_dependencies.return_value
        mock_fetched_dependencies.ancillary_files = {"settings": get_test_instrument_team_data_path(
            "glows/imap_glows_pipeline-settings_20100101_v001.json")}
        l3a_json_path = get_test_data_folder() / "glows" / "imap_glows_l3a_20130908085214_orbX_modX_p_v00.json"
        with open(l3a_json_path) as f:
            mock_l3a_data.return_value.data = json.load(f)
        mock_cdf_path = mock_save_data.return_value

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version, repointing=5)

        mock_processing_input_collection = Mock()
        parent_file_path = Path("test/parent_path")
        expected_parent_file_names = ['parent_path']
        expected_global_metadata_attrs = {
            "flight_software_version": 131848,
            "ground_software_version": "0.3",
            "pkts_file_name": "data_l0/imap_l0_sci_glows_20241018_mockByGlowsTeam051_v00.pkts",
            "ancillary_data_files": [
                "data_ancillary/imap_glows_conversion_table_for_anc_data_v002.json",
                "data_ancillary/imap_glows_calibration_data_v002.dat",
                "data_ancillary/imap_glows_pipeline_settings_v002.json",
                "data_ancillary/imap_glows_map_of_uv_sources_v002.dat",
                "data_ancillary/imap_glows_map_of_excluded_regions_v002.dat",
                "data_ancillary/imap_glows_exclusions_by_instr_team_v002.dat",
                "data_ancillary/imap_glows_suspected_transients_v002.dat",
                "data_ancillary/imap_glows_map_of_extra_helio_bckgrd_v001.dat",
                "data_ancillary/imap_glows_time_dep_bckgrd_v001.dat",
                "data_ancillary/imap_l1_anc_sc_Merged_2010-2030_mockByGlowsTeam001.csv"
            ],
            "l2_input_file_name": "data_l2_histograms/imap_glows_l2_20130908085214_orbX_modX_p_v00.json"
        }
        mock_processing_input_collection.get_file_paths.return_value = [parent_file_path]

        processor = GlowsProcessor(dependencies=mock_processing_input_collection, input_metadata=input_metadata)
        products = processor.process()

        mock_glows_initializer.assert_not_called()
        mock_glows_dependencies_class.fetch_dependencies.assert_called_with(mock_processing_input_collection)
        expected_data_to_save = create_glows_l3a_from_dictionary(
            mock_l3a_data.return_value.data, replace(input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))

        expected_data_to_save.parent_file_names = expected_parent_file_names
        actual_data = mock_save_data.call_args.args[0]
        self.assertEqual(expected_parent_file_names, actual_data.parent_file_names)
        self.assertEqual(expected_global_metadata_attrs, actual_data.global_metadata_attrs)
        assert_dataclass_fields(expected_data_to_save, actual_data, omit=["global_metadata_attrs"])
        self.assertEqual([mock_cdf_path], products)

    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    def test_does_not_save_cdf_if_process_l3a_returns_none(self, mock_save_data, _):
        instrument = 'glows'
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        outgoing_data_level = "l3a"
        outgoing_version = 'v002'
        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version, repointing=5)

        mock_processing_input_collection = Mock()
        processor = GlowsProcessor(dependencies=mock_processing_input_collection, input_metadata=input_metadata)
        processor.process_l3a = Mock(return_value=None)
        products = processor.process()

        mock_save_data.assert_not_called()
        self.assertEqual([], products)

    @patch('imap_l3_processing.glows.glows_processor.create_glows_l3a_from_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    def test_process_l3a(self, l3a_data_constructor, create_glows_l3a_from_dictionary):
        input_metadata = InputMetadata('glows', "l3a", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)

        processor.add_spin_angle_delta = Mock()
        fetched_dependencies = Mock()

        mock_l3a_data = l3a_data_constructor.return_value
        mock_l3a_data.data = {
            "daily_lightcurve": {
                "exposure_times": np.array([1])
            }
        }
        result = processor.process_l3a(fetched_dependencies)

        self.assertIs(create_glows_l3a_from_dictionary.return_value, result)
        l3a_data_constructor.assert_called_once_with(fetched_dependencies.ancillary_files)
        mock_l3a_data.process_l2_data_file.assert_called_once_with(fetched_dependencies.data)
        mock_l3a_data.generate_l3a_data.assert_called_once_with(
            fetched_dependencies.ancillary_files)
        processor.add_spin_angle_delta.assert_called_with(mock_l3a_data.data,
                                                          fetched_dependencies.ancillary_files)
        create_glows_l3a_from_dictionary.assert_called_once_with(processor.add_spin_angle_delta.return_value,
                                                                 replace(input_metadata,
                                                                         descriptor=GLOWS_L3A_DESCRIPTOR))

    @patch('imap_l3_processing.glows.glows_processor.create_glows_l3a_from_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    def test_process_l3a_returns_none_when_no_input_histrogram_bins_are_valid(self, l3a_data_constructor,
                                                                              create_glows_l3a_from_dictionary):
        input_metadata = InputMetadata('glows', "l3a", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)

        mock_l3a_data = l3a_data_constructor.return_value
        mock_l3a_data.data = {
            "daily_lightcurve": {
                "exposure_times": np.array([])
            }
        }

        processor.add_spin_angle_delta = Mock()
        fetched_dependencies = Mock()
        result = processor.process_l3a(fetched_dependencies)

        self.assertIsNone(result)

        mock_l3a_data.process_l2_data_file.assert_called_once_with(fetched_dependencies.data)
        mock_l3a_data.generate_l3a_data.assert_called_once_with(
            fetched_dependencies.ancillary_files)
        processor.add_spin_angle_delta.assert_not_called()
        create_glows_l3a_from_dictionary.assert_not_called()

    def test_add_spin_angle_delta(self):
        cases = [
            (60, 3),
            (90, 2)
        ]

        for bins, expected_delta in cases:
            with self.subTest(bin=bins, expected_delta=expected_delta):
                ancillary_files = {}
                with open(get_test_data_path("glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")) as f:
                    example_data = json.load(f)

                with tempfile.TemporaryDirectory() as tempdir:
                    temp_file_path = Path(tempdir) / "settings.json"
                    example_settings = get_test_instrument_team_data_path(
                        "glows/imap_glows_pipeline-settings_20100101_v001.json")

                    with open(example_settings) as file:
                        loaded_file = json.load(file)

                    loaded_file['l3a_nominal_number_of_bins'] = bins

                    with open(temp_file_path, 'w') as file:
                        json.dump(loaded_file, file)
                    ancillary_files['settings'] = temp_file_path

                    result = GlowsProcessor.add_spin_angle_delta(deepcopy(example_data), ancillary_files)
                for k, v in example_data.items():
                    if k != "daily_lightcurve":
                        self.assertEqual(v, result[k])

                for k2, v2 in example_data["daily_lightcurve"].items():
                    self.assertEqual(v2, result["daily_lightcurve"][k2])
                spin_angle_delta = result["daily_lightcurve"]["spin_angle_delta"]
                self.assertEqual(len(example_data["daily_lightcurve"]["spin_angle"]), len(spin_angle_delta))
                self.assertTrue(np.all(spin_angle_delta == expected_delta))

    @patch("imap_l3_processing.utils.spiceypy")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_l3a_files')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer')
    def test_process_l3bc(self, mock_glows_l3bc_initializer, mock_archive_dependencies, mock_generate_l3bc,
                          mock_filter_bad_days,
                          mock_l3c_model_class, mock_l3b_model_class, mock_save_data,
                          mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 1
        mock_spiceypy.kdata.return_value = ['kernel_1', 'type', 'source', 'handle']

        first_cr_number_to_process = 2091
        second_cr_number_to_process = 2092

        external_deps = ExternalDependencies(f107_index_file_path=sentinel.f107_index_file_path,
                                             lyman_alpha_path=sentinel.lyman_alpha_path,
                                             omni2_data_path=sentinel.omni2_data_path)

        mock_archive_dependencies.side_effect = [Path("path1.zip"), Path("path2.zip")]

        l3a_data_1 = {"filename": "l3a_file_1"}
        first_dependency_version = Version(1, 1)
        first_dependency = GlowsL3BCDependencies(l3a_data=[l3a_data_1],
                                                 external_files=sentinel.external_files_1,
                                                 ancillary_files={
                                                     'bad_days_list': sentinel.bad_days_list_1,
                                                 },
                                                 carrington_rotation_number=first_cr_number_to_process,
                                                 version=first_dependency_version,
                                                 start_date=Mock(),
                                                 end_date=Mock(),
                                                 repointing_file_path=sentinel.repointing_file_path)

        l3a_data_2 = {"filename": "l3a_file_2"}
        second_dependency_version = Version(1, 2)
        second_dependency = GlowsL3BCDependencies(l3a_data=[l3a_data_2],
                                                  external_files=sentinel.external_files_2,
                                                  ancillary_files={
                                                      'bad_days_list': sentinel.bad_days_list_2,
                                                  },
                                                  carrington_rotation_number=second_cr_number_to_process,
                                                  version=second_dependency_version,
                                                  start_date=Mock(),
                                                  end_date=Mock(),
                                                  repointing_file_path=sentinel.repointing_file_path)

        mock_generate_l3bc.side_effect = [(sentinel.l3b_data_1, sentinel.l3c_data_1),
                                          (sentinel.l3b_data_2, sentinel.l3c_data_2)]
        mock_filter_bad_days.side_effect = [sentinel.filtered_days_1, sentinel.filtered_days_2]

        l3b_model_1 = Mock(parent_file_names=["file1"])
        l3b_model_2 = Mock(parent_file_names=["file2"])
        l3c_model_1 = Mock(parent_file_names=["file3"])
        l3c_model_2 = Mock(parent_file_names=["file4"])
        mock_l3b_model_class.from_instrument_team_dictionary.side_effect = [l3b_model_1, l3b_model_2]
        mock_l3c_model_class.from_instrument_team_dictionary.side_effect = [l3c_model_1, l3c_model_2]
        mock_save_data.side_effect = [Path("path/to/l3b_file_1.cdf"),
                                      Path("path/to/l3c_file_1.cdf"),
                                      Path("path/to/l3b_file_2.cdf"),
                                      Path("path/to/l3c_file_2.cdf")]
        input_major_version = 2
        input_version = VersionMap(
            {
                GLOWS_L3B_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3C_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3D_DESCRIPTOR: Version(5, 1),
                GLOWS_L3E_HI_45_DESCRIPTOR: Version(5, 1),
                GLOWS_L3E_HI_90_DESCRIPTOR: Version(5, 1),
                GLOWS_L3E_LO_DESCRIPTOR: Version(5, 1),
                GLOWS_L3E_ULTRA_SF_DESCRIPTOR: Version(5, 1),
                GLOWS_L3E_ULTRA_HF_DESCRIPTOR: Version(5, 1),
            }
        )
        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       version=input_version)

        mock_glows_l3bc_initializer.get_crs_to_process.return_value = GlowsL3BCInitializerData(
            external_dependencies=external_deps,
            l3bc_dependencies=[first_dependency, second_dependency],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        processor = GlowsProcessor(dependencies=sentinel.dependencies, input_metadata=input_metadata)
        products = processor.process()

        self.assertEqual([Path("path/to/l3b_file_1.cdf"),
                          Path("path/to/l3c_file_1.cdf"),
                          Path("path1.zip"),
                          Path("path/to/l3b_file_2.cdf"),
                          Path("path/to/l3c_file_2.cdf"),
                          Path("path2.zip")
                          ], products)

        dependencies_with_filtered_list_1 = replace(first_dependency, l3a_data=sentinel.filtered_days_1)
        dependencies_with_filtered_list_2 = replace(second_dependency, l3a_data=sentinel.filtered_days_2)

        mock_filter_bad_days.assert_has_calls(
            [call([l3a_data_1], sentinel.bad_days_list_1, first_cr_number_to_process),
             call([l3a_data_2], sentinel.bad_days_list_2, second_cr_number_to_process)])

        mock_generate_l3bc.assert_has_calls(
            [call(dependencies_with_filtered_list_1), call(dependencies_with_filtered_list_2)])

        mock_glows_l3bc_initializer.get_crs_to_process.assert_called_once_with(sentinel.dependencies,
                                                                               input_major_version)

        expected_l3b_metadata_1 = InputMetadata("glows", "l3b", first_dependency.start_date,
                                                first_dependency.end_date,
                                                VersionMap({GLOWS_L3B_DESCRIPTOR: first_dependency_version}),
                                                "ion-rate-profile")
        expected_l3b_metadata_2 = InputMetadata("glows", "l3b", second_dependency.start_date,
                                                second_dependency.end_date,
                                                VersionMap({GLOWS_L3B_DESCRIPTOR: second_dependency_version}),
                                                "ion-rate-profile")
        expected_l3c_metadata_1 = InputMetadata("glows", "l3c", first_dependency.start_date,
                                                first_dependency.end_date,
                                                VersionMap({GLOWS_L3C_DESCRIPTOR: first_dependency_version}),
                                                "sw-profile")
        expected_l3c_metadata_2 = InputMetadata("glows", "l3c", second_dependency.start_date,
                                                second_dependency.end_date,
                                                VersionMap({GLOWS_L3C_DESCRIPTOR: second_dependency_version}),
                                                "sw-profile")
        mock_l3b_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3b_data_1, expected_l3b_metadata_1),
             call(sentinel.l3b_data_2, expected_l3b_metadata_2)])

        mock_l3c_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3c_data_1, expected_l3c_metadata_1),
             call(sentinel.l3c_data_2, expected_l3c_metadata_2)])

        mock_save_data.assert_has_calls([
            call(l3b_model_1, cr_number=first_cr_number_to_process),
            call(l3c_model_1, cr_number=first_cr_number_to_process),
            call(l3b_model_2, cr_number=second_cr_number_to_process),
            call(l3c_model_2, cr_number=second_cr_number_to_process)
        ])

        mock_archive_dependencies.assert_has_calls([
            call(l3bc_deps=first_dependency, external_dependencies=external_deps),
            call(l3bc_deps=second_dependency, external_dependencies=external_deps),
        ])

        self.assertEqual(["file1", "l3a_file_1", "path1.zip", "kernel_1"], l3b_model_1.parent_file_names)
        self.assertEqual(["file2", "l3a_file_2", "path2.zip", "kernel_1"], l3b_model_2.parent_file_names)
        self.assertEqual(["file3", "path1.zip", "l3b_file_1.cdf", "kernel_1", ], l3c_model_1.parent_file_names)
        self.assertEqual(["file4", "path2.zip", "l3b_file_2.cdf", "kernel_1", ], l3c_model_2.parent_file_names)

    @patch('imap_l3_processing.glows.glows_processor.filter_l3a_files')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate.from_instrument_team_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind.from_instrument_team_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer")
    def test_process_l3bc_catches_no_data_error_and_continues(self, mock_glows_initializer_class, mock_save_data,
                                                              mock_archive_dependencies, mock_generate_l3bc,
                                                              mock_l3c_from_instrument_team_dictionary,
                                                              mock_l3b_from_instrument_team_dictionary, _):
        bc_dependencies_1 = GlowsL3BCDependencies(version=Version(1, 1),
                                                  carrington_rotation_number=2096,
                                                  start_date=datetime(year=2021, month=1, day=1),
                                                  end_date=datetime(year=2021, month=1, day=1),
                                                  l3a_data=[],
                                                  external_files=defaultdict(Mock),
                                                  ancillary_files=defaultdict(Mock),
                                                  repointing_file_path=sentinel.repointing_file_path
                                                  )
        bc_dependencies_2 = GlowsL3BCDependencies(version=Version(1, 1),
                                                  carrington_rotation_number=2096,
                                                  start_date=datetime(year=2021, month=1, day=1),
                                                  end_date=datetime(year=2021, month=1, day=1),
                                                  l3a_data=[],
                                                  external_files=defaultdict(Mock),
                                                  ancillary_files=defaultdict(Mock),
                                                  repointing_file_path=sentinel.repointing_file_path
                                                  )
        external_dependencies = ExternalDependencies(None, None, None)
        mock_glows_initializer_class.get_crs_to_process.return_value = GlowsL3BCInitializerData(
            external_dependencies=external_dependencies,
            l3bc_dependencies=[bc_dependencies_1, bc_dependencies_2],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        zip_file_1 = "imap_glows_archive_20210101_v001.zip"
        zip_file_2 = "imap_glows_archive_20210102_v001.zip"

        mock_archive_dependencies.side_effect = [zip_file_1, zip_file_2]

        l3b_path = Path("path_to/l3b")
        l3c_path = Path("path_to/l3c")
        mock_save_data.side_effect = [l3b_path, l3c_path]

        version_map = VersionMap({
            GLOWS_L3B_DESCRIPTOR: Version(1, 1),
            GLOWS_L3C_DESCRIPTOR: Version(1, 1),
            GLOWS_L3D_DESCRIPTOR: Version(5, 1),
            GLOWS_L3E_HI_45_DESCRIPTOR: Version(5, 1),
            GLOWS_L3E_HI_90_DESCRIPTOR: Version(5, 1),
            GLOWS_L3E_LO_DESCRIPTOR: Version(5, 1),
            GLOWS_L3E_ULTRA_SF_DESCRIPTOR: Version(5, 1),
            GLOWS_L3E_ULTRA_HF_DESCRIPTOR: Version(5, 1),
        })

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       version_map)

        processor = GlowsProcessor(Mock(), input_metadata)

        mock_generate_l3bc.side_effect = [
            CannotProcessCarringtonRotationError("All days for Carrington Rotation are in a bad season."),
            ({}, {})
        ]

        products = processor.process()

        mock_l3b_from_instrument_team_dictionary.assert_called_once()
        mock_l3c_from_instrument_team_dictionary.assert_called_once()

        mock_archive_dependencies.assert_has_calls([
            call(l3bc_deps=bc_dependencies_1, external_dependencies=external_dependencies),
            call(l3bc_deps=bc_dependencies_2, external_dependencies=external_dependencies)
        ])

        self.assertEqual(2, mock_save_data.call_count)
        mock_save_data.assert_has_calls([
            call(mock_l3b_from_instrument_team_dictionary.return_value, cr_number=2096),
            call(mock_l3c_from_instrument_team_dictionary.return_value, cr_number=2096),
        ])

        self.assertEqual(3, len(products))
        self.assertEqual([
            l3b_path,
            l3c_path,
            zip_file_2
        ], products)

    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    def test_process_l3bc_catches_exceptions_from_science_code_and_continues(self, mock_generate_l3bc, _):
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(
                l3a_data_folder_path / 'imap_glows_l3a_hist_20100201-repoint00032_v001.cdf')]

        external_dependencies = ExternalDependencies(
            f107_index_file_path=get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            omni2_data_path=get_test_instrument_team_data_path('glows/omni_2010.dat'),
            lyman_alpha_path=None
        )

        l3bc_deps = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            ancillary_files={
                'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
                'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
                'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
                'pipeline_settings': get_test_instrument_team_data_path(
                    'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
            },
            external_files={
                'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
                'omni_raw_data': get_test_instrument_team_data_path('glows/omni_2010.dat'),
            },
            carrington_rotation_number=1,
            start_date=Mock(), end_date=Mock(), version=Version(1, 1),
            repointing_file_path=sentinel.repointing_file_path
        )

        initializer_data = GlowsL3BCInitializerData(
            external_dependencies=external_dependencies,
            l3bc_dependencies=[l3bc_deps],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        mock_generate_l3bc.side_effect = Exception(
            'L3c not generated: observed cx rate is smaller than min of the cx_grid')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=Mock())
        processor_output = process_l3bc(processor, initializer_data)

        self.assertEqual([], processor_output.data_products)

    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    def test_l3bc_uses_all_l3a_file_names_for_l3b_parents(self, mock_save_data, mock_archive_deps):
        mock_archive_deps.return_value = Path("some/path/to/archive.zip")

        l3a_data_folder_path = get_test_data_path('glows/l3a_products')

        l3a_filenames = ['imap_glows_l3a_hist_20100104-repoint00004_v001.cdf',
                         'imap_glows_l3a_hist_20100119-repoint00019_v001.cdf']
        l3a_data = [create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / file) for file in l3a_filenames]

        l3bc_deps = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            ancillary_files={
                'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
                'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
                'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
                'pipeline_settings': get_test_instrument_team_data_path(
                    'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
            },
            external_files={
                'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
                'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat'),
            },
            carrington_rotation_number=2092,
            start_date=Mock(), end_date=Mock(), version=Version(1, 1),
            repointing_file_path=sentinel.repointing_file_path,
        )

        initializer_data = GlowsL3BCInitializerData(
            external_dependencies=sentinel.externel_deps,
            l3bc_dependencies=[l3bc_deps],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=Mock())
        _ = process_l3bc(processor, initializer_data)

        self.assertEqual(2, mock_save_data.call_count)
        [l3b_data_product] = mock_save_data.call_args_list[0].args

        expected_parents = [
            *l3a_filenames,
            'archive.zip',
            'f107_fluxtable.txt',
            'imap_glows_uv-anisotropy-1CR_20100101_v001.json',
            'imap_glows_WawHelioIonMP_20100101_v002.json',
            'imap_glows_bad-days-list_v001.dat',
            'imap_glows_pipeline-settings-L3bc_20250707_v002.json',
        ]

        self.assertEqual(set(expected_parents), set(l3b_data_product.parent_file_names))

    @patch("imap_l3_processing.glows.glows_processor.process_l3e")
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3b_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3c_json_file_from_cdf")
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch("imap_l3_processing.glows.glows_processor.os")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3DInitializer")
    @patch("imap_l3_processing.glows.glows_processor.read_pipeline_settings")
    def test_process_l3d(self, mock_read_pipeline_settings, mock_glows_l3d_initializer, mock_os, mock_shutil, mock_run,
                         mock_convert_json_to_l3d_data_product, mock_get_parent_file_names_from_l3d_json,
                         mock_rename_l3d, mock_save_data, mock_convert_l3c_to_json, mock_convert_l3b_to_json, _):

        cr_number = 2092
        mock_read_pipeline_settings.return_value = {'start_cr': cr_number}

        expected_end_cr = cr_number + 1
        glows_l3d_dependencies = GlowsL3DDependencies(
            external_files={
                "lya_raw_data": Path("path/to/lya"),
            },
            ancillary_files={
                "pipeline_settings": Path(
                    "glows/imap_glows_pipeline-settings-l3bcde_20250514_v004.json"
                ),
                "WawHelioIon": {
                    "speed": Path("path/to/speed"),
                    "p-dens": Path("path/to/p-dens"),
                    "uv-anis": Path("path/to/uv-anis"),
                    "phion": Path("path/to/phion"),
                    "lya": Path("path/to/lya"),
                    "e-dens": Path("path/to/e-dens"),
                },
            },
            l3b_file_paths=[sentinel.l3b_file_1, sentinel.l3b_file_2],
            l3c_file_paths=[sentinel.l3c_file_1, sentinel.l3c_file_2],
            end_cr=expected_end_cr,
        )

        old_l3d = Path('imap_glows_l3d_solar-hist_19470303-cr02090_v001.cdf')
        input_major_version = 12
        l3d_output_version = Version(input_major_version, 5)
        mock_glows_l3d_initializer.should_process_l3d.return_value = (
            l3d_output_version,
            glows_l3d_dependencies,
            old_l3d,
        )

        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {expected_end_cr}')

        mock_os.listdir.return_value = [
            f'{cr_number}_txt_file_1',
            f'{cr_number}_txt_file_2',
            f'{cr_number + 1}_txt_file_1',
            f'{cr_number + 1}_txt_file_2',
        ]

        mock_convert_json_to_l3d_data_product.return_value = sentinel.l3d_data_product

        mock_rename_l3d.return_value = [
            Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
            Path("imap_glows_lya_19470303_20100101_v000.dat"),
        ]

        mock_save_data.return_value = Path("l3d_cdf.cdf")

        input_version_map = VersionMap(
            {
                GLOWS_L3B_DESCRIPTOR: Version(2, 1),
                GLOWS_L3C_DESCRIPTOR: Version(2, 1),
                GLOWS_L3D_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3E_HI_45_DESCRIPTOR: Version(2, 1),
                GLOWS_L3E_HI_90_DESCRIPTOR: Version(2, 1),
                GLOWS_L3E_LO_DESCRIPTOR: Version(2, 1),
                GLOWS_L3E_ULTRA_SF_DESCRIPTOR: Version(2, 1),
                GLOWS_L3E_ULTRA_HF_DESCRIPTOR: Version(2, 1),
            }
        )
        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7), None, version=input_version_map)

        processing_input_collection = Mock()
        processor = GlowsProcessor(processing_input_collection, input_metadata)
        products = processor.process()

        mock_convert_l3b_to_json.assert_has_calls([call(sentinel.l3b_file_1), call(sentinel.l3b_file_2)])
        mock_convert_l3c_to_json.assert_has_calls([call(sentinel.l3c_file_1), call(sentinel.l3c_file_2)])
        mock_glows_l3d_initializer.should_process_l3d.assert_called_with(
            self.mock_external_deps, [], [], self.mock_reprocess_info, input_major_version)
        self.mock_fetch_reprocess_info.assert_called_with(processing_input_collection)
        self.assertEqual([
            Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
            Path("imap_glows_lya_19470303_20100101_v000.dat"),
            Path("l3d_cdf.cdf")
        ], products)

        self.assertEqual(2, mock_shutil.copy.call_count)
        mock_shutil.copy.assert_has_calls([
            call(Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
                 imap_data_access.config[
                     "DATA_DIR"] / "imap/ancillary/glows/imap_glows_e-dens_19470303_20100101_v000.dat"),
            call(Path("imap_glows_lya_19470303_20100101_v000.dat"),
                 imap_data_access.config["DATA_DIR"] / "imap/ancillary/glows/imap_glows_lya_19470303_20100101_v000.dat")
        ])

        self.assertEqual(2, mock_os.makedirs.call_count)
        mock_os.makedirs.assert_has_calls([
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d', exist_ok=True),
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt', exist_ok=True),
        ])

        expected_dependencies = json.dumps({
            'external_files': {
                'lya_raw_data': str(Path('path/to/lya')),
            },
            'ancillary_files': {
                'pipeline_settings':
                    str(Path('glows/imap_glows_pipeline-settings-l3bcde_20250514_v004.json')),
                'WawHelioIon': {
                    'speed': str(Path('path/to/speed')),
                    'p-dens': str(Path('path/to/p-dens')),
                    'uv-anis': str(Path('path/to/uv-anis')),
                    'phion': str(Path('path/to/phion')),
                    'lya': str(Path('path/to/lya')),
                    'e-dens': str(Path('path/to/e-dens'))
                }
            }
        })

        expected_working_directory = Path(l3d.__file__).parent / 'science'

        self.assertEqual(1, mock_run.call_count)
        mock_run.assert_has_calls([
            call([sys.executable, './generate_l3d.py', f'{expected_end_cr}', expected_dependencies],
                 cwd=str(expected_working_directory),
                 check=True,
                 capture_output=True, text=True),
        ])

        mock_os.listdir.assert_called_once_with(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')

        expected_l3d_txt_paths = [
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{expected_end_cr}_txt_file_1', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{expected_end_cr}_txt_file_2', ),
        ]
        mock_rename_l3d.assert_has_calls([call(expected_l3d_txt_paths, str(Version(None, l3d_output_version.minor)))])

        mock_get_parent_file_names_from_l3d_json.assert_called_once_with(expected_working_directory / 'data_l3d')

        expected_data_product_metadata = InputMetadata(instrument="glows", data_level="l3d", descriptor="solar-hist",
                                                       start_date=datetime(1947, 3, 3),
                                                       end_date=datetime(1947, 3, 3),
                                                       version=VersionMap({GLOWS_L3D_DESCRIPTOR: l3d_output_version}))

        mock_convert_json_to_l3d_data_product.assert_called_once_with(
            expected_working_directory / 'data_l3d' / f'imap_glows_l3d_solar-params-history_19470303-cr0{expected_end_cr}_v00.json',
            expected_data_product_metadata,
            mock_get_parent_file_names_from_l3d_json.return_value)

        mock_save_data.assert_called_once_with(sentinel.l3d_data_product, cr_number=expected_end_cr)

    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.PATH_TO_L3D_TOOLKIT', get_test_data_path('glows/science'))
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3d_adds_parent_file_names_to_output(self, mock_spicepy, mock_run, mock_rename_l3d_text_output,
                                                          mock_save_data):
        mock_spicepy.ktotal.return_value = 0
        l3b_path_1 = get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf')
        l3b_path_2 = get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100519_v011.cdf')
        l3c_path_1 = get_test_data_path('glows/imap_glows_l3c_sw-profile_20100422_v011.cdf')
        l3c_path_2 = get_test_data_path('glows/imap_glows_l3c_sw-profile_20100519_v011.cdf')

        pipeline_settings_path = get_test_data_path(
            'glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json')
        speed_path = get_test_data_path('glows/imap_glows_plasma-speed-Legendre-2010a_v001.dat')
        p_dens_path = get_test_data_path('glows/imap_glows_proton-density-Legendre-2010a_v001.dat')
        uv_anis_path = get_test_data_path('glows/imap_glows_uv-anisotropy-2010a_v001.dat')
        phion_path = get_test_data_path('glows/imap_glows_photoion-2010a_v001.dat')
        lya_path = get_test_data_path('glows/imap_glows_lya-2010a_v001.dat')
        e_dens_path = get_test_data_path('glows/imap_glows_electron-density-2010a_v001.dat')

        lyman_alpha_composite_path = get_test_data_path('glows/lyman_alpha_composite.nc')

        l3b_file_paths = [l3b_path_1, l3b_path_2]
        l3c_file_paths = [l3c_path_1, l3c_path_2]
        ancillary_inputs = {
            'pipeline_settings': pipeline_settings_path,
            'WawHelioIon': {
                'speed': speed_path,
                'p-dens': p_dens_path,
                'uv-anis': uv_anis_path,
                'phion': phion_path,
                'lya': lya_path,
                'e-dens': e_dens_path
            }
        }
        external_inputs = {
            'lya_raw_data': lyman_alpha_composite_path
        }

        cr_number = 2095
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number}')

        l3d_dependencies = GlowsL3DDependencies(l3b_file_paths=l3b_file_paths, l3c_file_paths=l3c_file_paths,
                                                ancillary_files=ancillary_inputs, external_files=external_inputs,
                                                end_cr=cr_number)

        _ = process_l3d(l3d_dependencies, Version(1, 1))

        [save_data_call_args] = mock_save_data.call_args_list
        actual_data_product = save_data_call_args.args[0]

        expected_parent_file_names = [
            "imap_glows_plasma-speed-2010a_v003.dat",
            "imap_glows_proton-density-2010a_v003.dat",
            "imap_glows_uv-anisotropy-2010a_v003.dat",
            "imap_glows_photoion-2010a_v003.dat",
            "imap_glows_lya-2010a_v003.dat",
            "imap_glows_electron-density-2010a_v003.dat",
            "lyman_alpha_composite.nc",
            "imap_glows_l3b_ion-rate-profile_20100422_v011.cdf",
            "imap_glows_l3b_ion-rate-profile_20100519_v011.cdf",
            "imap_glows_l3c_sw-profile_20100422_v011.cdf",
            "imap_glows_l3c_sw-profile_20100519_v011.cdf"
        ]

        expected_l3d_txt_file_paths = [
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_e-dens_19470303-cr02095_v00.dat",
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_uv-anis_19470303-cr02095_v00.dat",
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_phion_19470303-cr02095_v00.dat",
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_p-dens_19470303-cr02095_v00.dat",
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_lya_19470303-cr02095_v00.dat",
            get_test_data_path("glows/science")
            / "data_l3d_txt"
            / "imap_glows_l3d_speed_19470303-cr02095_v00.dat",
        ]

        self.assertCountEqual(
            expected_parent_file_names, actual_data_product.parent_file_names
        )
        self.assertCountEqual(
            expected_l3d_txt_file_paths, mock_rename_l3d_text_output.call_args[0][0]
        )
        self.assertEqual("v001", mock_rename_l3d_text_output.call_args[0][1])

    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.read_pipeline_settings')
    def test_process_l3d_handles_unexpected_exception_from_science(self, mock_read_pipeline_settings,
                                                                   mock_convert_json_to_l3d,
                                                                   mock_run, mock_os, _, __):
        ancillary_files = {
            'pipeline_settings': get_test_data_path(
                "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
            'WawHelioIon': {
                'speed': Path('path/to/speed'),
                'p-dens': Path('path/to/p-dens'),
                'uv-anis': Path('path/to/uv-anis'),
                'phion': Path('path/to/phion'),
                'lya': Path('path/to/lya'),
                'e-dens': Path('path/to/e-dens')
            }
        }
        external_files = {
            'lya_raw_data': Path('path/to/lya'),
        }
        mock_read_pipeline_settings.return_value = {'start_cr': 2091}
        l3b_file_paths = []
        l3c_file_paths = []
        expected_cr = 2096
        l3d_dependencies = GlowsL3DDependencies(ancillary_files=ancillary_files,
                                                external_files=external_files,
                                                l3b_file_paths=l3b_file_paths,
                                                l3c_file_paths=l3c_file_paths,
                                                end_cr=expected_cr)

        mock_os.listdir.return_value = ['2096_txt_file_1',
                                        '2096_txt_file_2',
                                        '2096_txt_file_3',
                                        '2096_txt_file_4',
                                        '2096_txt_file_5',
                                        '2096_txt_file_6'
                                        ]

        unexpected_exception = r"""Traceback (most recent call last):
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\generate_l3d.py", line 46, in <module>
            solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES,data_l3b,data_l3c)
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\l3d_SolarParamHistory.py", line 554, in update_solar_params_hist
            self._update_l3bc_data(data_l3b,data_l3c,CR)
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\l3d_SolarParamHistory.py", line 516, in _update_l3bc_data
            anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read = self._generate_cr_solar_params(CR, data_l3b, data_l3c)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\funcs.py", line 71, in calculate_mean_date
            l3a=read_json(f)
                ^^^^^^^^^^^^
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\funcs.py", line 451, in read_json
            fp = open(fn, 'r')
                 ^^^^^^^^^^^^^
        FileNotFoundError: [Errno 2] No such file or directory: 'imap_glows_l3a_hist_20100511-repoint00131_v011.cdf'"""

        mock_run.side_effect = [CalledProcessError(cmd="", returncode=1, stderr=unexpected_exception)]

        with self.assertRaises(Exception) as context:
            process_l3d(l3d_dependencies, 1)
        self.assertEqual(unexpected_exception, context.exception.stderr)

        mock_convert_json_to_l3d.assert_not_called()

    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    def test_process_glows_l3d_drift(self, mock_save_data, _):
        l3d_drift_test_data = get_test_data_path("glows/l3d_drift_test")
        lyman_alpha_composite = l3d_drift_test_data / "lyman_alpha_composite.nc"
        expected_cr = 2098

        l3b_1 = l3d_drift_test_data / "imap_glows_l3b_ion-rate-profile_20100422_v013.cdf"
        l3b_2 = l3d_drift_test_data / "imap_glows_l3b_ion-rate-profile_20100519_v013.cdf"
        l3c_1 = l3d_drift_test_data / "imap_glows_l3c_sw-profile_20100422_v012.cdf"
        l3c_2 = l3d_drift_test_data / "imap_glows_l3c_sw-profile_20100519_v012.cdf"

        plasma_speed_legendre = l3d_drift_test_data / "imap_glows_plasma-speed-2010a_20100101_v003.dat"
        proton_density_legendre = l3d_drift_test_data / "imap_glows_proton-density-2010a_20100101_v003.dat"
        uv_anisotropy = l3d_drift_test_data / "imap_glows_uv-anisotropy-2010a_20100101_v003.dat"
        photoion = l3d_drift_test_data / "imap_glows_photoion-2010a_20100101_v003.dat"
        lya_2010a = l3d_drift_test_data / "imap_glows_lya-2010a_20100101_v003.dat"
        electron_density = l3d_drift_test_data / "imap_glows_electron-density-2010a_20100101_v003.dat"
        pipeline_settings = l3d_drift_test_data / "imap_glows_pipeline-settings-l3bcde_20100101_v006.json"

        l3d_dependencies = GlowsL3DDependencies(
            external_files={
                "lya_raw_data": lyman_alpha_composite,
            },
            ancillary_files={
                "pipeline_settings": pipeline_settings,
                "WawHelioIon": {
                    "speed": plasma_speed_legendre,
                    "p-dens": proton_density_legendre,
                    "uv-anis": uv_anisotropy,
                    "phion": photoion,
                    "lya": lya_2010a,
                    "e-dens": electron_density,
                },
            },
            l3b_file_paths=[l3b_1, l3b_2],
            l3c_file_paths=[l3c_1, l3c_2],
            end_cr=expected_cr,
        )

        glows_l3d_output = process_l3d(l3d_dependencies, Version(1, 4))

        expected_txt_filenames = ["imap_glows_e-dens_19470303_20100629_v004.dat",
                                  "imap_glows_lya_19470303_20100629_v004.dat",
                                  "imap_glows_p-dens_19470303_20100629_v004.dat",
                                  "imap_glows_phion_19470303_20100629_v004.dat",
                                  "imap_glows_speed_19470303_20100629_v004.dat",
                                  "imap_glows_uv-anis_19470303_20100629_v004.dat"]
        for file in expected_txt_filenames:
            self.assertIn(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / file, glows_l3d_output.l3d_text_file_paths)

        test_cases = [
            (
                expected_txt_filenames[0],
                "electron_density",
                849,
                [1947.167990000000, -1, 1250.5],
                [2010.492523, 6.01955, 2098.5],
            ),
            (
                expected_txt_filenames[1],
                "lyman_alpha",
                849,
                [1947.167990000000, 6.199757637899209e+11, 1250.5],
                [2010.49252, 391688271318.51849, 2098.50000, ],
            ),
            (
                expected_txt_filenames[2],
                "proton_density",
                849,
                [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1.25050000e+03, -1.00000000e+00],
                [2010.49252260274, 3.533487319946289, 3.50542426109314, 3.409455299377441, 3.878353357315063,
                 4.971815586090088, 6.224776268005371, 7.651707649230957, 8.510316848754883, 8.865212440490723,
                 8.813090324401855, 9.078702926635742, 9.078702926635742, 7.368694305419922, 5.78929615020752,
                 4.63243579864502, 3.519419431686401, 3.034561634063721, 3.000047445297241, 3.000047445297241,
                 2098.5, 88888.0],
            ),
            (
                expected_txt_filenames[3],
                "phion",
                849,
                [1947.167990000000, 1.830438692701064e-07, 1250.5],
                [2010.49252260274, 1.064334895772845e-07, 2098.5],
            ),
            (
                expected_txt_filenames[4],
                "plasma_speed",
                849,
                [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1.25050000e+03, -1],
                [2010.49252260274, 397.0, 399.0, 406.0, 374.0, 316.0, 268.0, 228.0, 209.0, 202.0, 203.0, 198.0,
                 198.0, 235.0, 283.0, 332.0, 398.0, 436.0, 439.0, 439.0, 2098.5, 88888.0],
            ),
            (
                expected_txt_filenames[5],
                "uv_anisotropy",
                849,
                [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1.25050000e+03, -1],
                [2010.49252260274, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2098.5, 88888.0],
            ),
        ]

        [save_data_call_args] = mock_save_data.call_args_list
        l3d_data_product = save_data_call_args.args[0]

        expected_cdf_filename = "imap_glows_l3d_solar-hist_19470303-cr02098_v001.0004.cdf"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            save_data(l3d_data_product, cr_number=expected_cr, folder_path=tmp_dir)

            with CDF(str(tmp_dir / expected_cdf_filename)) as actual_cdf:
                for filename, cdf_var_name, length_of_data, first_line, last_line in test_cases:
                    with self.subTest(msg=filename):
                        actual = np.loadtxt(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / filename)

                        self.assertEqual(length_of_data, len(actual))
                        self.assertEqual(length_of_data, len(actual_cdf[cdf_var_name][...]))

                        np.testing.assert_allclose(actual[0], first_line)
                        np.testing.assert_allclose(actual[-1], last_line)

    @patch('imap_l3_processing.glows.glows_processor.compute_glows_flags_for_repoint')
    @patch('imap_l3_processing.glows.glows_processor.get_lo_pivot_angles')
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.determine_spacecraft_info_using_predict_if_needed')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_hf')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_sf')
    def test_process_l3e(self, mock_process_ultra, mock_process_ultra_hf, mock_process_lo, mock_process_hi,
                         mock_determine_spacecraft_info,
                         mock_get_pointing_date_range,
                         mock_get_lo_pivot_angles,
                         mock_compute_glows_flags_for_repoint,
                         ):
        mock_process_hi.side_effect = [
            [Path('path/to/first_hi_l3e')],
            [Path('path/to/second_hi_l3e')],
        ]
        mock_process_lo.return_value = [Path('path/to/lo_l3e')]
        mock_process_ultra.return_value = [Path('path/to/ultra_l3e')]
        mock_process_ultra_hf.return_value = [Path('path/to/ultra_l3e_hf')]
        mock_get_lo_pivot_angles.return_value = {25: LoPivotAngle("l1b_nhk.cdf", 75)}
        mock_compute_glows_flags_for_repoint.return_value = 4
        mock_determine_spacecraft_info.return_value = sentinel.spacecraft_info, GlowsL3Flags.PREDICTIVE_EPHEMERIS, ["spice kernel"]

        expected_l3e_products = [
            Path('path/to/lo_l3e'),
            Path('path/to/first_hi_l3e'), Path('path/to/second_hi_l3e'),
            Path('path/to/ultra_l3e'),
            Path('path/to/ultra_l3e_hf')
        ]

        start_epoch = datetime(2020, 1, 1)
        end_epoch = datetime(2020, 1, 2)
        epoch_delta = timedelta(hours=12)
        repointing_midpoint = datetime(2020, 1, 1, 12)
        mock_get_pointing_date_range.return_value = (start_epoch, end_epoch)

        mock_dependencies = Mock()

        mock_dependencies.get_hi_parents.return_value = ["hi_ancillary.dat"]
        mock_dependencies.get_lo_parents.return_value = ["lo_ancillary.dat"]
        mock_dependencies.get_ul_parents.return_value = ["ul_ancillary.dat"]

        l3d_cdf_path = Path("path/to/l3d.cdf")
        initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_dependencies,
            repointings=GlowsL3eVersionsForRepointings(
                repointing_numbers=[25],
                hi_90_repointings={25: Version(None, 1)},
                hi_45_repointings={25: Version(None, 2)},
                lo_repointings={25: Version(None, 3)},
                ultra_sf_repointings={25: Version(None, 4)},
                ultra_hf_repointings={25: Version(None, 4)},
            ),
            l3d_cdf_path=l3d_cdf_path,
            metakernel_with_predict_ephem=Mock(),
            metakernel_without_predict_ephem=Mock(),
        )

        actual_l3e_products = process_l3e(initializer_data)
        mock_get_pointing_date_range.assert_called_once_with(25)
        mock_compute_glows_flags_for_repoint.assert_called_once_with(l3d_cdf_path, repointing_midpoint)
        mock_determine_spacecraft_info.assert_called_once_with(
            datetime(2020, 1, 1, 12),
            initializer_data.metakernel_with_predict_ephem,
            initializer_data.metakernel_without_predict_ephem,
        )
        expected_flags = 4 | 2**15
        mock_process_hi.assert_has_calls(
            [
                call(
                    ["hi_ancillary.dat", "spice kernel"],
                    25,
                    start_epoch,
                    epoch_delta,
                    90,
                    Version(None, 1),
                    expected_flags,
                    sentinel.spacecraft_info,
                ),
                call(
                    ["hi_ancillary.dat", "spice kernel"],
                    25,
                    start_epoch,
                    epoch_delta,
                    135,
                    Version(None, 2),
                    expected_flags,
                    sentinel.spacecraft_info,
                ),
            ]
        )
        mock_process_lo.assert_called_once_with(["lo_ancillary.dat", "spice kernel", "l1b_nhk.cdf"], 25, start_epoch, epoch_delta, 75,
                                                Version(None, 3), expected_flags, sentinel.spacecraft_info)
        mock_process_ultra.assert_called_once_with(["ul_ancillary.dat", "spice kernel"], 25, start_epoch, epoch_delta, Version(None, 4), expected_flags, sentinel.spacecraft_info)
        mock_process_ultra_hf.assert_called_once_with(["ul_ancillary.dat", "spice kernel"], 25, start_epoch, epoch_delta, Version(None, 4), expected_flags, sentinel.spacecraft_info)

        self.assertEqual(expected_l3e_products, actual_l3e_products)
        mock_get_lo_pivot_angles.assert_called_once_with([25])

    @patch('imap_l3_processing.glows.glows_processor.determine_spacecraft_info_using_predict_if_needed')
    @patch('imap_l3_processing.glows.glows_processor.get_lo_pivot_angles')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_hf')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_sf')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.compute_glows_flags_for_repoint')
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.process_l3d')
    def test_process_l3e_invoked_from_top_level_process(self, mock_process_l3d, mock_get_pointing_date_range, mock_compute_flags,
                                         mock_process_lo, mock_process_hi, mock_process_ul_sf, mock_process_ul_hf,
                                         mock_get_lo_pivot_angles, mock_determine_spacecraft_info):
        mock_get_lo_pivot_angles.return_value = {
            2902: LoPivotAngle("l1b_2902", 90),
            2905: LoPivotAngle("l1b_2905", 90),
        }
        mock_compute_flags.return_value = 0
        self.mock_l3d_initializer.should_process_l3d.return_value = (sentinel.l3d_version, sentinel.glows_l3d_dependency, sentinel.old_l3d)
        process_l3d_result = GlowsL3DProcessorOutput(sentinel.l3d_cdf_file_path, [sentinel.l3d_text_file_paths],
                                         sentinel.last_processed_cr)
        mock_process_l3d.return_value = process_l3d_result
        mock_determine_spacecraft_info.return_value = sentinel.spacecraft_info, GlowsL3Flags.PREDICTIVE_EPHEMERIS, ["spice kernel"]


        input_major_version = 5

        input_version_map = VersionMap(
            {
                GLOWS_L3B_DESCRIPTOR: Version(2, 1),
                GLOWS_L3C_DESCRIPTOR: Version(2, 1),
                GLOWS_L3D_DESCRIPTOR: Version(2, 1),
                GLOWS_L3E_HI_45_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3E_HI_90_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3E_LO_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3E_ULTRA_SF_DESCRIPTOR: Version(input_major_version, 1),
                GLOWS_L3E_ULTRA_HF_DESCRIPTOR: Version(input_major_version, 1),
            }
        )
        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7), None, version=input_version_map)

        mock_get_pointing_date_range.return_value = datetime(2025, 1, 1), datetime(2025, 1, 2)

        l3e_initialzer_repointings = GlowsL3eVersionsForRepointings(
            [2902, 2903, 2904, 2905, 2906, 2907],
            {2902: Version(None, 1), 2903: Version(1,11)},
            {2902: Version(None, 1), 2904: Version(2,12)},
            {2902: Version(None, 1), 2905: Version(3,13)},
            {2902: Version(None, 1), 2906: Version(4,14)},
            {2902: Version(None, 1), 2907: Version(5,15)},
        )

        l3e_dependencies = GlowsL3EDependencies(
            Path("2025/05/03/imap_glows_energy_grid_lo"),
            Path("2025/05/03/imap_glows_energy_grid_hi"),
            Path("2025/05/03/imap_glows_energy_grid_ultra"),
            Path("2025/05/03/imap_glows_tess_xyz_8"),
            Path("2025/05/03/imap_glows_tess_ang16"),
            Path("2025/05/03/imap_glows_lya_series"),
            Path("2025/05/03/imap_glows_solar_uv_anisotropy"),
            Path("2025/05/03/imap_glows_speed_3d_sw"),
            Path("2025/05/03/imap_glows_density_3d_sw"),
            Path("2025/05/03/imap_glows_phion_hydrogen"),
            Path("2025/05/03/imap_glows_sw_eqtr_electrons"),
            {
                "executable_dependency_paths": {
                    "energy-grid-lo": "EnGridLo.dat",
                    "energy-grid-hi": "EnGridHi.dat",
                    "energy-grid-ultra": "EnGridUltra.dat",
                    "tess-xyz-8": "tessXYZ8.dat",
                    "tess-ang-16": "tessAng16.dat",
                    "ionization-files": "ionization.files.dat",
                }
            },
            Path("path/to/some/pipeline_settings_file.csv"),
            Path("repoint.csv"),
        )
        self.mock_l3e_initializer.get_repointings_to_process.return_value = GlowsL3EInitializerOutput(
            l3e_dependencies,
            l3e_initialzer_repointings,
            sentinel.l3d_cdf_file_path,
            metakernel_with_predict_ephem=Mock(),
            metakernel_without_predict_ephem=Mock(),
        )

        expected_lo_files = [
            [f'imap_glows_l3e_survival-probability-lo_20250101-repoint02902_v001.cdf', sentinel.lo_dat_1],
            [f'imap_glows_l3e_survival-probability-lo_20250101-repoint02905_v003.0013.cdf', sentinel.lo_dat_2],
        ]
        mock_process_lo.side_effect = expected_lo_files

        expected_hi90_files = [
            [
                f"imap_glows_l3e_survival-probability-hi-90_20250101-repoint02902_v001.cdf",
                sentinel.hi90_dat_1,
            ],
            [
                f"imap_glows_l3e_survival-probability-hi-90_20250101-repoint02903_v001.0011.cdf",
                sentinel.hi90_dat_2,
            ],
        ]
        expected_hi45_files = [
            [
                f"imap_glows_l3e_survival-probability-hi-45_20250101-repoint02902_v001.cdf",
                sentinel.hi45_dat_1,
            ],
            [
                f"imap_glows_l3e_survival-probability-hi-45_20250101-repoint02904_v002.0012.cdf",
                sentinel.hi45_dat_2,
            ],
        ]
        mock_process_hi.side_effect = [
            expected_hi90_files[0],
            expected_hi45_files[0],
            expected_hi90_files[1],
            expected_hi45_files[1],
        ]

        expected_ul_sf_files = [
            [
                f"imap_glows_l3e_survival-probability-ul-sf_20250101-repoint02902_v001.cdf",
                sentinel.ul_sf_dat_1,
            ],
            [
                f"imap_glows_l3e_survival-probability-ul-sf_20250101-repoint02906_v004.0014.cdf",
                sentinel.ul_sf_dat_2,
            ],
        ]
        mock_process_ul_sf.side_effect = expected_ul_sf_files

        expected_ul_hf_files = [
            [
                f"imap_glows_l3e_survival-probability-ul-hf_20250101-repoint02902_v001.cdf",
                sentinel.ul_hf_dat_1,
            ],
            [
                f"imap_glows_l3e_survival-probability-ul-hf_20250101-repoint02907_v005.0015.cdf",
                sentinel.ul_hf_dat_2,
            ],
        ]
        mock_process_ul_hf.side_effect = expected_ul_hf_files

        expected_products = [
            sentinel.l3d_text_file_paths,
            sentinel.l3d_cdf_file_path,
            f"imap_glows_l3e_survival-probability-lo_20250101-repoint02902_v001.cdf",
            sentinel.lo_dat_1,
            f"imap_glows_l3e_survival-probability-hi-90_20250101-repoint02902_v001.cdf",
            sentinel.hi90_dat_1,
            f"imap_glows_l3e_survival-probability-hi-45_20250101-repoint02902_v001.cdf",
            sentinel.hi45_dat_1,
            f"imap_glows_l3e_survival-probability-ul-sf_20250101-repoint02902_v001.cdf",
            sentinel.ul_sf_dat_1,
            f"imap_glows_l3e_survival-probability-ul-hf_20250101-repoint02902_v001.cdf",
            sentinel.ul_hf_dat_1,
            f"imap_glows_l3e_survival-probability-hi-90_20250101-repoint02903_v001.0011.cdf",
            sentinel.hi90_dat_2,
            f"imap_glows_l3e_survival-probability-hi-45_20250101-repoint02904_v002.0012.cdf",
            sentinel.hi45_dat_2,
            f"imap_glows_l3e_survival-probability-lo_20250101-repoint02905_v003.0013.cdf",
            sentinel.lo_dat_2,
            f"imap_glows_l3e_survival-probability-ul-sf_20250101-repoint02906_v004.0014.cdf",
            sentinel.ul_sf_dat_2,
            f"imap_glows_l3e_survival-probability-ul-hf_20250101-repoint02907_v005.0015.cdf",
            sentinel.ul_hf_dat_2,
        ]

        processing_input_collection = Mock()
        processor = GlowsProcessor(processing_input_collection, input_metadata)
        products = processor.process()


        self.mock_l3e_initializer.get_repointings_to_process.assert_called_once_with(
            process_l3d_result,
            sentinel.old_l3d,
            self.mock_l3bc_initializer.return_value.repoint_file_path,
            input_version_map,
            self.mock_reprocess_info,
        )
        self.assertEqual(expected_products, products)
        self.mock_fetch_reprocess_info.assert_called_once_with(processing_input_collection)

    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_ultra_sf(self, mock_shutil, mock_determine_call_args, mock_run,
                                  mock_convert_dat_to_glows_l3e_ul_product,
                                  mock_save_data, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]

        epoch_start_date = datetime(year=2024, month=10, day=7)
        epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
        epoch_delta = (epoch_end_date - epoch_start_date) / 2
        repointing = 20
        version = Version(1, 12)

        expected_input_metadata = InputMetadata('glows', "l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                                                version=VersionMap({GLOWS_L3E_ULTRA_SF_DESCRIPTOR:version}), descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR,
                                                repointing=repointing)

        ultra_args = ["20241007_000000", "date.001", "vx", "vy", "vz", "30.000"]

        call_args_object = MagicMock(spec=GlowsL3eCallArguments)
        call_args_object.to_argument_list.return_value = ultra_args
        mock_determine_call_args.return_value = call_args_object

        mock_convert_dat_to_glows_l3e_ul_product.return_value = sentinel.ultra_data_1

        mock_save_data.return_value = "imap_glows_l3e_survival-probability-ul-sf_20241007-repoint00020_v001.0012.cdf"

        parent_file_names = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        glows_flags = 4
        spacecraft_info = Mock()
        products = process_l3e_ul_sf(parent_file_names, repointing, epoch_start_date, epoch_delta, version, glows_flags, spacecraft_info)

        expected_repointing_midpoint = epoch_start_date + epoch_delta
        mock_determine_call_args.assert_called_once_with(epoch_start_date, expected_repointing_midpoint, 30, spacecraft_info=spacecraft_info)

        mock_run.assert_called_once_with(["./survProbUltra"] + ultra_args)

        output_data_path = Path("probSur.Imap.Ul_20241007_000000_date.001.dat")

        mock_convert_dat_to_glows_l3e_ul_product.assert_called_once_with(
            expected_input_metadata, output_data_path, expected_repointing_midpoint, epoch_delta, call_args_object)

        expected_first_data_path = AncillaryFilePath(
            "imap_glows_survival-probability-ul-sf-raw_20241007_v012.dat").construct_path()

        mock_shutil.move.assert_called_once_with(output_data_path, expected_first_data_path)

        mock_save_data.assert_called_once_with(sentinel.ultra_data_1)
        survival_data_product: GlowsL3EUltraData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)
        np.testing.assert_array_equal(survival_data_product.glows_flags, np.array([glows_flags], dtype=np.uint16))

        self.assertEqual(products, ["imap_glows_l3e_survival-probability-ul-sf_20241007-repoint00020_v001.0012.cdf",
                                    expected_first_data_path])

    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_ultra_hf(self, mock_shutil, mock_determine_call_args, mock_run,
                                  mock_convert_dat_to_glows_l3e_ul_product,
                                  mock_save_data, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]

        epoch_start_date = datetime(year=2024, month=10, day=7)
        epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
        epoch_delta = (epoch_end_date - epoch_start_date) / 2
        repointing = 20
        version = Version(1, 12)

        input_metadata = InputMetadata('glows', "l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                                       version=VersionMap({GLOWS_L3E_ULTRA_HF_DESCRIPTOR: version}), descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR,
                                       repointing=repointing)

        spacecraft_frame_spacecraft_info = GlowsL3eSpacecraftInfo(
            spacecraft_radius=500,
            spacecraft_longitude=200,
            spacecraft_latitude=65,
            spacecraft_velocity_x=120,
            spacecraft_velocity_y=240,
            spacecraft_velocity_z=360,
            spin_axis_longitude=240,
            spin_axis_latitude=3,
        )

        mock_determine_call_args.return_value = GlowsL3eCallArguments(
            formatted_date="20241007_000000",
            decimal_date="date.001",
            spacecraft_info=spacecraft_frame_spacecraft_info,
            elongation=30
        )

        expected_rest_frame_args = GlowsL3eCallArguments(
            formatted_date="20241007_000000",
            decimal_date="date.001",
            spacecraft_info=GlowsL3eSpacecraftInfo(
                spacecraft_radius=500,
                spacecraft_longitude=200,
                spacecraft_latitude=65,
                spacecraft_velocity_x=0,
                spacecraft_velocity_y=0,
                spacecraft_velocity_z=0,
                spin_axis_longitude=240,
                spin_axis_latitude=3,
            ),
            elongation=30
        )

        mock_convert_dat_to_glows_l3e_ul_product.return_value = sentinel.ultra_data_hf

        mock_save_data.return_value = "imap_glows_l3e_survival-probability-ul-hf_20241007-repoint00020_v001.0012.cdf"

        parent_file_names = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        glows_flags = 8
        products = process_l3e_ul_hf(parent_file_names, repointing, epoch_start_date, epoch_delta, version, glows_flags, spacecraft_info=spacecraft_frame_spacecraft_info)

        expected_repointing_midpoint = epoch_start_date + epoch_delta
        mock_determine_call_args.assert_called_once_with(epoch_start_date, expected_repointing_midpoint, 30, spacecraft_info=spacecraft_frame_spacecraft_info)

        mock_run.assert_called_once_with(["./survProbUltra"] + expected_rest_frame_args.to_argument_list())

        output_data_path = Path("probSur.Imap.Ul.V0_20241007_000000_date.001.dat")

        mock_convert_dat_to_glows_l3e_ul_product.assert_called_once_with(
            input_metadata, output_data_path, expected_repointing_midpoint, epoch_delta, expected_rest_frame_args)

        expected_first_data_path = AncillaryFilePath(
            "imap_glows_survival-probability-ul-hf-raw_20241007_v012.dat").construct_path()

        mock_shutil.move.assert_called_once_with(output_data_path, expected_first_data_path)

        mock_save_data.assert_called_once_with(sentinel.ultra_data_hf)
        survival_data_product: GlowsL3EUltraData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)
        np.testing.assert_array_equal(survival_data_product.glows_flags, np.array([glows_flags], dtype=np.uint16))

        self.assertEqual(products, ["imap_glows_l3e_survival-probability-ul-hf_20241007-repoint00020_v001.0012.cdf",
                                    expected_first_data_path])

    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_hi(self, mock_shutil, mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_hi_product, mock_save_data):

        test_cases = [
            ("90", 90, GLOWS_L3E_HI_90_DESCRIPTOR),
            ("45", 135, GLOWS_L3E_HI_45_DESCRIPTOR)
        ]

        for descriptor_elongation, elongation, descriptor in test_cases:
            with (self.subTest(elongation=elongation)):
                mock_determine_call_args.reset_mock()
                mock_run.reset_mock()
                mock_convert_dat_to_glows_l3e_hi_product.reset_mock()
                mock_save_data.reset_mock()
                mock_shutil.reset_mock()

                epoch_start_date = datetime(year=2024, month=10, day=7)
                epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
                epoch_delta = (epoch_end_date - epoch_start_date) / 2
                repointing = 20
                version = Version(1, 12)

                expected_input_metadata = InputMetadata(
                    instrument='glows', data_level="l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                    version=VersionMap({descriptor: version}), descriptor=f'survival-probability-hi-{descriptor_elongation}',
                    repointing=repointing)

                l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
                l3e_dependencies.pipeline_settings = {'start_cr': 2090}
                l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")

                hi_args = ["20241007_000000", "date.001", "vx", "vy", "vz", descriptor_elongation]

                mock_call_args_object = MagicMock(spec=GlowsL3eCallArguments)
                mock_call_args_object.to_argument_list.return_value = hi_args
                mock_determine_call_args.return_value = mock_call_args_object
                mock_convert_dat_to_glows_l3e_hi_product.return_value = sentinel.hi_data_1

                saved_cdf_path = Path(
                    f"imap_glows_l3e_survival-probability-hi-{descriptor_elongation}_20241007-repoint00020_v001.0012.cdf")

                mock_save_data.return_value = saved_cdf_path

                parent_file_names = ["some_l3e_hi_parent.dat", "some_repointing_file.repoint.csv"]
                glows_flags = 16
                spacecraft_info = Mock()
                products = process_l3e_hi(parent_file_names, repointing, epoch_start_date, epoch_delta, elongation,
                                          version, glows_flags, spacecraft_info=spacecraft_info)

                expected_repointing_midpoint = epoch_start_date + epoch_delta
                mock_determine_call_args.assert_called_once_with(epoch_start_date, expected_repointing_midpoint,
                                                                 float(elongation), spacecraft_info=spacecraft_info)

                mock_run.assert_called_once_with(["./survProbHi"] + hi_args)

                first_output_data_path = Path(
                    f"probSur.Imap.Hi_20241007_000000_date.001_{descriptor_elongation[:5]}.dat")

                mock_convert_dat_to_glows_l3e_hi_product.assert_called_once_with(
                    expected_input_metadata,
                    first_output_data_path,
                    expected_repointing_midpoint,
                    epoch_delta,
                    mock_call_args_object
                )

                mock_save_data.assert_called_once_with(sentinel.hi_data_1)

                survival_data_product: GlowsL3EHiData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(parent_file_names, survival_data_product.parent_file_names)
                np.testing.assert_array_equal(survival_data_product.glows_flags,
                                              np.array([glows_flags], dtype=np.uint16))

                expected_output_data_path = AncillaryFilePath(
                    f"imap_glows_survival-probability-hi-{descriptor_elongation}-raw_20241007_v012.dat"
                ).construct_path()

                mock_shutil.move.assert_called_once_with(first_output_data_path, expected_output_data_path)

                expected_products = [
                    saved_cdf_path,
                    expected_output_data_path
                ]
                self.assertEqual(expected_products, products)

    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_lo(self, mock_shutil,
                            mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_lo_product, mock_save_data):

        test_cases = [(75, "75.00"), (105, "105.0")]
        for elongation, elongation_filename in test_cases:
            with (self.subTest(elongation)):
                epoch_start_date = datetime(year=2024, month=10, day=7)
                epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
                epoch_delta = (epoch_end_date - epoch_start_date) / 2
                repointing = 20
                version = Version(1, 12)

                expected_input_metadata = InputMetadata(
                    instrument='glows', data_level="l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                    version=VersionMap({GLOWS_L3E_LO_DESCRIPTOR: version}), descriptor=GLOWS_L3E_LO_DESCRIPTOR, repointing=repointing)

                lo_call_args = ["20241007_000000", "date.100", "vx", "vy", "vz", f"{elongation:.3f}"]

                spacecraft_info = GlowsL3eSpacecraftInfo(
                    spacecraft_radius=np.float32(100.0),
                    spacecraft_longitude=np.float32(100.0),
                    spacecraft_latitude=np.float32(100.0),
                    spacecraft_velocity_x=np.float32(100.0),
                    spacecraft_velocity_y=np.float32(100.0),
                    spacecraft_velocity_z=np.float32(100.0),
                    spin_axis_longitude=np.float32(100.0),
                    spin_axis_latitude=np.float32(100.0),
                )

                l3e_args = GlowsL3eCallArguments(
                    formatted_date="20241007_000000",
                    decimal_date="date.100",
                    spacecraft_info=spacecraft_info,
                    elongation=elongation,
                )
                l3e_args.to_argument_list = Mock(return_value=lo_call_args)
                mock_determine_call_args.return_value = l3e_args

                lo_data_1 = Mock()
                mock_convert_dat_to_glows_l3e_lo_product.return_value = lo_data_1

                output_cdf_path = Path("imap_glows_l3e_survival-probability-lo_20241007-repoint00020_v001.0012.cdf")
                mock_save_data.return_value = output_cdf_path

                parent_file_names = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]

                glows_flags = 32

                products = process_l3e_lo(parent_file_names, repointing, epoch_start_date, epoch_delta, elongation,
                                          version, glows_flags, spacecraft_info=spacecraft_info)

                expected_repointing_midpoint = epoch_start_date + epoch_delta
                mock_determine_call_args.assert_called_once_with(epoch_start_date, expected_repointing_midpoint,
                                                                 elongation, spacecraft_info=spacecraft_info)

                mock_run.assert_called_once_with(["./survProbLo"] + lo_call_args)

                first_output_file_path = Path(f"probSur.Imap.Lo_20241007_000000_date.100_{elongation_filename}.dat")

                mock_convert_dat_to_glows_l3e_lo_product.assert_called_once_with(expected_input_metadata,
                                                                                 first_output_file_path,
                                                                                 expected_repointing_midpoint,
                                                                                 epoch_delta,
                                                                                 elongation, l3e_args)

                expected_first_output_file_path = AncillaryFilePath(
                    "imap_glows_survival-probability-lo-raw_20241007_v012.dat"
                ).construct_path()

                mock_shutil.move.assert_called_once_with(first_output_file_path, expected_first_output_file_path)

                mock_save_data.assert_called_once_with(lo_data_1)
                survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(parent_file_names,
                                 survival_data_product.parent_file_names)
                np.testing.assert_array_equal(survival_data_product.glows_flags,
                                              np.array([glows_flags], dtype=np.uint16))

                self.assertEqual([output_cdf_path,
                                  expected_first_output_file_path], products)

                mock_shutil.reset_mock()
                mock_determine_call_args.reset_mock()
                mock_run.reset_mock()
                mock_convert_dat_to_glows_l3e_lo_product.reset_mock()
                mock_save_data.reset_mock()

    @patch('imap_l3_processing.glows.glows_processor.compute_glows_flags_for_repoint')
    @patch('imap_l3_processing.glows.glows_processor.get_lo_pivot_angles')
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_hf')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul_sf')
    @patch('imap_l3_processing.glows.glows_processor.determine_spacecraft_info_using_predict_if_needed')
    def test_process_l3e_skips_repointing_on_exception(self, mock_determine_spacecraft_info, mock_process_ultra_sf, mock_process_ultra_hf,
                                                       mock_process_lo,
                                                       mock_process_hi, mock_get_pointing_date_range,
                                                       mock_get_lo_pivot_angles,
                                                       mock_compute_glows_flags_for_repoint):
        mock_compute_glows_flags_for_repoint.return_value = 0

        mock_process_hi.side_effect = [
            ValueError("Failed to generate hi"), [Path('path/to/first_hi-45_l3e')],
            [Path('path/to/second_hi-90_l3e')], [Path('path/to/second_hi-45_l3e')],
            [Path('path/to/third_hi-90_l3e')], [Path('path/to/third_hi-45_l3e')],
            [Path('path/to/fourth_hi-90_l3e')], [Path('path/to/fourth_hi-45_l3e')],
        ]

        mock_process_lo.side_effect = [
            [Path('path/to/first_lo_l3e')],
            ValueError("Failed to generate lo"),
            [Path('path/to/third_lo_l3e')],
            [Path('path/to/fourth_lo_l3e')],
        ]

        mock_process_ultra_sf.side_effect = [
            [Path('path/to/first_ultra_l3e')],
            [Path('path/to/second_ultra_l3e')],
            ValueError("Failed to generate ultra!"),
            [Path('path/to/fourth_ultra_l3e')]
        ]

        mock_process_ultra_hf.side_effect = [
            [Path('path/to/first_ultra_l3e_hf')],
            [Path('path/to/second_ultra_l3e_hf')],
            [Path('path/to/third_ultra_l3e_hf')],
            ValueError("Failed to generate ultra!")
        ]
        mock_get_lo_pivot_angles.return_value = {
            24: LoPivotAngle(parent_filename="l1b_nhk_24", pivot_angle=124),
            25: LoPivotAngle(parent_filename="l1b_nhk_25", pivot_angle=125),
            26: LoPivotAngle(parent_filename="l1b_nhk_26", pivot_angle=126),
            27: LoPivotAngle(parent_filename=None, pivot_angle=90),
        }

        expected_l3e_products = [
            Path('path/to/first_lo_l3e'),
            Path('path/to/first_hi-45_l3e'),
            Path('path/to/first_ultra_l3e'),
            Path('path/to/first_ultra_l3e_hf'),
            Path('path/to/second_hi-90_l3e'),
            Path('path/to/second_hi-45_l3e'),
            Path('path/to/second_ultra_l3e'),
            Path('path/to/second_ultra_l3e_hf'),
            Path('path/to/third_lo_l3e'),
            Path('path/to/third_hi-90_l3e'),
            Path('path/to/third_hi-45_l3e'),
            Path('path/to/third_ultra_l3e_hf'),
            Path('path/to/fourth_lo_l3e'),
            Path('path/to/fourth_hi-90_l3e'),
            Path('path/to/fourth_hi-45_l3e'),
            Path('path/to/fourth_ultra_l3e'),
        ]

        start_epoch_1 = datetime(2020, 1, 1)
        end_epoch_1 = datetime(2020, 1, 2)
        epoch_delta_1 = timedelta(hours=12)

        start_epoch_2 = datetime(2020, 1, 2)
        end_epoch_2 = datetime(2020, 1, 4)
        epoch_delta_2 = timedelta(hours=24)

        start_epoch_3 = datetime(2020, 1, 4)
        end_epoch_3 = datetime(2020, 1, 5)
        epoch_delta_3 = timedelta(hours=12)

        start_epoch_4 = datetime(2020, 1, 5)
        end_epoch_4 = datetime(2020, 1, 6)
        epoch_delta_4 = timedelta(hours=12)

        mock_get_pointing_date_range.side_effect = [
            (start_epoch_1, end_epoch_1),
            (start_epoch_2, end_epoch_2),
            (start_epoch_3, end_epoch_3),
            (start_epoch_4, end_epoch_4),
        ]

        mock_determine_spacecraft_info.side_effect = [
            (sentinel.spacecraft_info24, GlowsL3Flags.NONE, []),
            (sentinel.spacecraft_info25, GlowsL3Flags.NONE, []),
            (sentinel.spacecraft_info26, GlowsL3Flags.NONE, []),
            (sentinel.spacecraft_info27, GlowsL3Flags.PREDICTIVE_EPHEMERIS, ["predict"]),
        ]

        mock_dependencies = Mock()

        hi_parents = ["imap_glows_hi-ancillary_20100101_v001.dat"]
        mock_dependencies.get_hi_parents.return_value = hi_parents
        lo_parents = ["imap_glows_lo-ancillary_20100101_v001.dat"]
        mock_dependencies.get_lo_parents.return_value = lo_parents
        ultra_parents = ["imap_glows_ul-ancillary_20100101_v001.dat"]
        mock_dependencies.get_ul_parents.return_value = ultra_parents

        metakernel_with_predict_ephem, metakernel_without_predict_ephem = Mock(), Mock()

        initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_dependencies,
            repointings=GlowsL3eVersionsForRepointings(
                repointing_numbers=[24, 25, 26, 27],
                hi_90_repointings={24: Version(None, 1), 25: Version(None, 1), 26: Version(None, 1), 27: Version(None, 1)},
                hi_45_repointings={24: Version(None, 2), 25: Version(None, 2), 26: Version(None, 2), 27: Version(None, 2)},
                lo_repointings={24: Version(None, 3), 25: Version(None, 3), 26: Version(None, 3), 27: Version(None, 3)},
                ultra_hf_repointings={24: Version(None, 4), 25: Version(None, 4), 26: Version(None, 4), 27: Version(None, 4)},
                ultra_sf_repointings={24: Version(None, 4), 25: Version(None, 4), 26: Version(None, 4), 27: Version(None, 4)},
            ),
            l3d_cdf_path=Path("path/to/l3d.cdf"),
            metakernel_with_predict_ephem=metakernel_with_predict_ephem,
            metakernel_without_predict_ephem=metakernel_without_predict_ephem,
        )

        actual_l3e_products = process_l3e(initializer_data)

        mock_get_pointing_date_range.assert_has_calls([call(24), call(25), call(26), call(27)])

        mock_process_hi.assert_has_calls([
            call(hi_parents, 24, start_epoch_1, epoch_delta_1, 90, Version(None, 1), 0, sentinel.spacecraft_info24),
            call(hi_parents, 24, start_epoch_1, epoch_delta_1, 135, Version(None, 2), 0, sentinel.spacecraft_info24),
            call(hi_parents, 25, start_epoch_2, epoch_delta_2, 90, Version(None, 1), 0, sentinel.spacecraft_info25),
            call(hi_parents, 25, start_epoch_2, epoch_delta_2, 135, Version(None, 2), 0, sentinel.spacecraft_info25),
            call(hi_parents, 26, start_epoch_3, epoch_delta_3, 90, Version(None, 1), 0, sentinel.spacecraft_info26),
            call(hi_parents, 26, start_epoch_3, epoch_delta_3, 135, Version(None, 2), 0, sentinel.spacecraft_info26),
            call(hi_parents + ["predict"], 27, start_epoch_4, epoch_delta_4, 90, Version(None, 1), GlowsL3Flags.PREDICTIVE_EPHEMERIS, sentinel.spacecraft_info27),
            call(hi_parents + ["predict"], 27, start_epoch_4, epoch_delta_4, 135, Version(None, 2), GlowsL3Flags.PREDICTIVE_EPHEMERIS, sentinel.spacecraft_info27),
        ])
        mock_process_lo.assert_has_calls([
            call(lo_parents + ["l1b_nhk_24"], 24, start_epoch_1, epoch_delta_1, 124, Version(None, 3), 0, sentinel.spacecraft_info24),
            call(lo_parents + ["l1b_nhk_25"], 25, start_epoch_2, epoch_delta_2, 125, Version(None, 3), 0, sentinel.spacecraft_info25),
            call(lo_parents + ["l1b_nhk_26"], 26, start_epoch_3, epoch_delta_3, 126, Version(None, 3), 0, sentinel.spacecraft_info26),
            call(lo_parents + ["predict"], 27, start_epoch_4, epoch_delta_4, 90, Version(None, 3), GlowsL3Flags.PREDICTIVE_EPHEMERIS, sentinel.spacecraft_info27),
        ])

        mock_process_ultra_sf.assert_has_calls([
            call(ultra_parents, 24, start_epoch_1, epoch_delta_1, Version(None, 4), 0, sentinel.spacecraft_info24),
            call(ultra_parents, 25, start_epoch_2, epoch_delta_2, Version(None, 4), 0, sentinel.spacecraft_info25),
            call(ultra_parents, 26, start_epoch_3, epoch_delta_3, Version(None, 4), 0, sentinel.spacecraft_info26),
            call(ultra_parents + ["predict"], 27, start_epoch_4, epoch_delta_4, Version(None, 4), GlowsL3Flags.PREDICTIVE_EPHEMERIS, sentinel.spacecraft_info27),
        ])

        mock_process_ultra_hf.assert_has_calls([
            call(ultra_parents, 24, start_epoch_1, epoch_delta_1, Version(None, 4), 0, sentinel.spacecraft_info24),
            call(ultra_parents, 25, start_epoch_2, epoch_delta_2, Version(None, 4), 0, sentinel.spacecraft_info25),
            call(ultra_parents, 26, start_epoch_3, epoch_delta_3, Version(None, 4), 0, sentinel.spacecraft_info26),
            call(ultra_parents + ["predict"], 27, start_epoch_4, epoch_delta_4, Version(None, 4), GlowsL3Flags.PREDICTIVE_EPHEMERIS, sentinel.spacecraft_info27),
        ])

        self.assertEqual(expected_l3e_products, actual_l3e_products)
        mock_get_lo_pivot_angles.assert_called_once_with([24, 25, 26, 27])

    @patch("imap_l3_processing.glows.glows_processor.json")
    @patch("imap_l3_processing.glows.glows_processor.ZipFile")
    def test_archive_dependencies(self, mock_zip, mock_json):
        version_number = Version(1, 1)
        expected_filepath = TEMP_CDF_FOLDER_PATH / f"imap_glows_l3b-archive_20250314_v001.zip"
        expected_json_filename = "cr_to_process.json"

        l3bc_dependencies = GlowsL3BCDependencies(
            version=version_number,
            carrington_rotation_number=2095,
            start_date=datetime.fromisoformat("2025-03-14 12:34:56.789"),
            end_date=datetime.fromisoformat("2025-03-24 12:34:56.789"),
            l3a_data=[{"filename": "imap_glows_l3a_hist_20250314-repoint000001_v001.cdf"},
                      {"filename": "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"}],
            external_files={},
            ancillary_files={
                'uv_anisotropy': Path("uv_anisotropy.dat"),
                'WawHelioIonMP_parameters': Path("waw_helio_ion.dat"),
                'bad_days_list': Path("bad_days_list.dat"),
                'pipeline_settings': Path("pipeline_settings.json"),
            },
            repointing_file_path=Path("repointing.csv")
        )

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        expected_json_to_serialize = {"cr_rotation_number": 2095,
                                      "l3a_paths": ["imap_glows_l3a_hist_20250314-repoint000001_v001.cdf",
                                                    "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"],
                                      "cr_start_date": "2025-03-14 12:34:56.789000",
                                      "cr_end_date": "2025-03-24 12:34:56.789000",
                                      "bad_days_list": "bad_days_list.dat",
                                      "pipeline_settings": "pipeline_settings.json",
                                      "waw_helioion_mp": "waw_helio_ion.dat",
                                      "uv_anisotropy": "uv_anisotropy.dat",
                                      "repointing_file": "repointing.csv",
                                      }

        mock_zip_file = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_file

        actual_zip_file_name = GlowsProcessor.archive_dependencies(l3bc_dependencies, external_dependencies)

        self.assertEqual(expected_filepath, actual_zip_file_name)

        mock_zip.assert_called_with(expected_filepath, "w", ZIP_DEFLATED)

        mock_json.dumps.assert_called_once_with(expected_json_to_serialize)

        mock_zip_file.write.assert_has_calls([
            call(Path("lyman_alpha_path"), "lyman_alpha_composite.nc"),
            call(Path("omni2_data_path"), "omni2_all_years.dat"),
            call(Path("f107_index_file_path"), "f107_fluxtable.txt"),
        ])
        mock_zip_file.writestr.assert_called_once_with(expected_json_filename, mock_json.dumps.return_value)



if __name__ == '__main__':
    unittest.main()
