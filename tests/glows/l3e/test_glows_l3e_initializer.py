import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call, sentinel, Mock

from imap_data_access import RepointInput
from imap_data_access.file_validation import Version

from imap_l3_processing.glows.descriptors import GLOWS_L3BCDE_DESCRIPTORS
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer, GlowsL3EInitializerOutput
from imap_l3_processing.glows.l3e.glows_l3e_utils import GlowsL3eVersionsForRepointings
from imap_l3_processing.models import VersionMap
from tests.test_helpers import create_mock_query_results

MODULE = 'imap_l3_processing.glows.l3e.glows_l3e_initializer'

class TestGlowsL3EInitializer(unittest.TestCase):
    @patch(f'{MODULE}.get_pointing_date_range')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.imap_data_access.query')
    @patch(f'{MODULE}.GlowsL3EDependencies.collect_spice_dependencies')
    def test_get_repointings_to_process(self, mock_collect_spice_dependencies, mock_query, mock_get_most_recently_uploaded_ancillary,
                                        mock_find_first_updated_cr, mock_identify_versions_for_l3e_output_files,
                                        mock_fetch_dependencies, mock_get_pointing_date_range):
        mock_query.side_effect = create_mock_query_results([
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ])

        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]

        mock_spice_with_predict, mock_spice_without_predict = Mock(), Mock()

        mock_collect_spice_dependencies.return_value = (mock_spice_with_predict, mock_spice_without_predict)

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf')
        updated_l3d_text_file_path = Path("imap_glows_e-dens_19470303_20100101_v000.dat")
        glows_l3d_processor_output = GlowsL3DProcessorOutput(updated_l3d, [updated_l3d_text_file_path], sentinel.last_processed_cr)
        previous_l3d = 'imap_glows_l3d_solar-hist_19470303-cr02090_v000.cdf'

        mock_find_first_updated_cr.return_value = 2091

        mock_l3e_dependencies = mock_fetch_dependencies.return_value
        mock_l3e_dependencies.pipeline_settings = {"start_cr": sentinel.start_of_mission_cr}
        mock_l3e_dependencies.repointing_file = Path('path/to/repointing_file')

        expected_hi_45 = {1234: Version(None, 1), 2468: Version(None, 1)}
        expected_hi_90 = {1234: Version(None, 2), 2468: Version(None, 2)}
        expected_lo = {1234: Version(None, 3), 2468: Version(None, 3)}
        expected_ultra = {1234: Version(None, 4), 2468: Version(None, 4)}

        expected_repointings = GlowsL3eVersionsForRepointings(
            repointing_numbers=[2468, 1234],
            hi_90_repointings=expected_hi_90,
            hi_45_repointings=expected_hi_45,
            lo_repointings=expected_lo,
            ultra_hf_repointings=expected_ultra,
            ultra_sf_repointings=expected_ultra,
        )

        expected_initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_l3e_dependencies,
            repointings=expected_repointings,
            l3d_cdf_path=updated_l3d,
            metakernel_without_predict_ephem=mock_spice_without_predict,
            metakernel_with_predict_ephem=mock_spice_with_predict,
        )

        mock_identify_versions_for_l3e_output_files.return_value = expected_repointings

        mock_get_pointing_date_range.side_effect = [
            (datetime(2010, 1, 1), datetime(2010, 1, 2)),
            (datetime(2011, 2, 1), datetime(2011, 2, 2)),
        ]

        input_major_version = VersionMap(
            {descriptor: Version(2, 1) for descriptor in GLOWS_L3BCDE_DESCRIPTORS}
        )

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output,
            previous_l3d,
            repointing_file_path,
            input_major_version,
            sentinel.reprocess_info,
        )

        mock_find_first_updated_cr.assert_called_once_with(updated_l3d, previous_l3d)

        mock_identify_versions_for_l3e_output_files.assert_called_once_with(sentinel.start_of_mission_cr, sentinel.last_processed_cr, 2090, repointing_file_path,
                                                                            input_major_version, sentinel.reprocess_info)

        mock_query.assert_has_calls([
            call(table="ancillary", instrument='glows', descriptor='pipeline-settings-l3bcde'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-lo'),
            call(table="ancillary", instrument='glows', descriptor='tess-xyz-8'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-hi'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-ultra'),
            call(table="ancillary", instrument='glows', descriptor='tess-ang-16'),
        ])

        mock_l3e_dependencies.copy_dependencies.assert_called_once()

        [fetch_dependencies_call] = mock_fetch_dependencies.call_args_list

        [actual_l3e_inputs] = fetch_dependencies_call.args

        pipeline_l3e_input_paths = actual_l3e_inputs.get_file_paths(source="glows")
        pipeline_l3e_input_filenames = [p.name for p in pipeline_l3e_input_paths]

        self.assertEqual([
            updated_l3d.name,
            "imap_glows_e-dens_19470303_20100101_v000.dat",
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ], pipeline_l3e_input_filenames)

        [repoint_input] = actual_l3e_inputs.get_file_paths(data_type=RepointInput.data_type)
        self.assertEqual("imap_2026_105_01.repoint.csv", repoint_input.name)

        self.assertEqual(actual_initializer_output, expected_initializer_data)

        mock_get_pointing_date_range.assert_has_calls([
            call(1234),
            call(2468)
        ])

        mock_collect_spice_dependencies.assert_called_once_with(
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2011, 2, 2),
        )

    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.imap_data_access')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    def test_get_repointings_to_process_identical_l3d_files(
        self,
        mock_find_first_updated_cr,
        mock_identify_versions_for_l3e_output_files,
        mock_fetch_dependencies,
        mock_imap_data_access,
        mock_get_most_recently_uploaded_ancillary,
    ):
        updated_l3d = Path(
            "path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf"
        )
        updated_l3d_text_file_path = Path(
            "imap_glows_e-dens_19470303_20100101_v000.dat"
        )
        expected_last_cr = 2091
        glows_l3d_processor_output = GlowsL3DProcessorOutput(
            updated_l3d, [updated_l3d_text_file_path], expected_last_cr
        )
        previous_l3d = "imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf"

        mock_l3e_dependencies = mock_fetch_dependencies.return_value
        mock_l3e_dependencies.pipeline_settings = {
            "start_cr": sentinel.start_of_mission_cr
        }
        mock_find_first_updated_cr.return_value = None
        mock_identify_versions_for_l3e_output_files.return_value = GlowsL3eVersionsForRepointings(
            repointing_numbers=[],
            hi_90_repointings={},
            hi_45_repointings={},
            lo_repointings={},
            ultra_sf_repointings={},
            ultra_hf_repointings={},
        )
        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output,
            previous_l3d,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )
        mock_find_first_updated_cr.assert_called_once_with(glows_l3d_processor_output.l3d_cdf_file_path, previous_l3d)
        mock_identify_versions_for_l3e_output_files.assert_called_once_with(
            sentinel.start_of_mission_cr,
            expected_last_cr,
            None,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )
        self.assertIsNone(actual_initializer_output)

    @patch(f'{MODULE}.imap_data_access.query')
    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    def test_get_repointings_to_process_uses_mission_start_cr_when_no_previous_l3d(self, mock_find_first_updated_cr,
                                                                                   mock_identify_versions_for_l3e_output_files,
                                                                                   mock_fetch_dependencies,
                                                                                   mock_get_most_recently_uploaded_ancillary,
                                                                                   _, ):
        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]
        mock_identify_versions_for_l3e_output_files.return_value = GlowsL3eVersionsForRepointings(
            repointing_numbers=[],
            ultra_sf_repointings={},
            ultra_hf_repointings={},
            lo_repointings={},
            hi_45_repointings={},
            hi_90_repointings={}
        )

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v012.0001.cdf')
        glows_l3d_processor_output = GlowsL3DProcessorOutput(updated_l3d, [], sentinel.last_processed_cr)
        previous_l3d = None

        mock_fetch_dependencies.return_value.pipeline_settings = {"start_cr": sentinel.start_of_mission_cr}

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        _ = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output, previous_l3d,
            repointing_file_path, sentinel.version_map,
            sentinel.reprocess_info,
        )

        mock_find_first_updated_cr.assert_not_called()
        mock_identify_versions_for_l3e_output_files.assert_called_once_with(
            sentinel.start_of_mission_cr,
            sentinel.last_processed_cr,
            sentinel.start_of_mission_cr,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )
