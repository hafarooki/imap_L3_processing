import unittest
from itertools import product
from math import prod
from unittest.mock import patch, call, Mock, sentinel

from imap_data_access.file_validation import Version

from imap_l3_processing.glows.descriptors import GLOWS_L3E_ULTRA_HF_DESCRIPTOR, GLOWS_L3E_HI_45_DESCRIPTOR, \
    GLOWS_L3E_HI_90_DESCRIPTOR, GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3B_DESCRIPTOR
from imap_l3_processing.glows.l3d.glows_l3d_initializer import GlowsL3DInitializer
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo, ReprocessTargets
from tests.test_helpers import create_mock_query_results


class TestGlowsL3DInitializer(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer(self, mock_fetch_l3d_dependencies, mock_download, mock_query, mock_read_pipeline_settings,
                             mock_read_cdf_parents):
        l3bs = [
            'imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
        ]
        l3cs = [
            'imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
        ]

        external_dependencies = Mock()

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
            'imap_glows_plasma-speed-2026d_19470301_v001.dat',
            'imap_glows_proton-density-2026d_19470301_v000.dat',
            'imap_glows_lya-2026d_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
            'imap_glows_photoion-2026d_19470301_v000.dat',
            'imap_glows_electron-density-2026d_19470301_v000.dat'
            'lyman-alpha-composite.nc'
        }

        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        test_cases = {
            'matching versions with legacy': (None, None),
            'matching versions with new version format': (1, 1),
            'upgraded major version': (1, 2),
        }
        for name, (existing_major_version, requested_major_version) in test_cases.items():
            with self.subTest(name=name):
                most_recent_l3d = f'imap_glows_l3d_solar-hist_19470301-cr00002_{Version(existing_major_version, 1)}.cdf'
                mock_query.side_effect = self._create_ancillary_query_results([
                    'imap_glows_l3d_solar-hist_19470301-cr00001_v000.cdf',
                    'imap_glows_l3d_solar-hist_19470301-cr00001_v001.cdf',
                    'imap_glows_l3d_solar-hist_19470301-cr00001_v002.cdf',
                    'imap_glows_l3d_solar-hist_19470301-cr00002_v000.cdf',
                    most_recent_l3d,
                ])
                mock_read_cdf_parents.reset_mock()
                mock_download.reset_mock()
                mock_read_pipeline_settings.reset_mock()
                mock_fetch_l3d_dependencies.reset_mock()
                actual_version, actual_l3d_deps, actual_old_l3d = GlowsL3DInitializer.should_process_l3d(external_dependencies,
                                                                                                         l3bs, l3cs, ReprocessInfo([]), requested_major_version)

                mock_read_cdf_parents.assert_called_once_with(most_recent_l3d)

                mock_query.assert_has_calls([
                    call(instrument='glows', data_level="l3d", descriptor="solar-hist"),
                    call(table='ancillary', instrument='glows', descriptor='plasma-speed-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='proton-density-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='uv-anisotropy-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='photoion-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='lya-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='electron-density-2026d', version='latest'),
                    call(table='ancillary', instrument='glows', descriptor='pipeline-settings-l3bcde', version='latest'),
                ])

                mock_download.assert_called_once_with("imap_glows_pipeline-settings-l3bcde_19470301_v000.json")
                mock_read_pipeline_settings.assert_called_once_with(mock_download.return_value)

                [fetch_dependencies_call] = mock_fetch_l3d_dependencies.call_args_list

                [actual_l3d_inputs, actual_external_deps] = fetch_dependencies_call.args

                pipeline_l3d_input_paths = actual_l3d_inputs.get_file_paths(source="glows")
                pipeline_l3d_input_filenames = [p.name for p in pipeline_l3d_input_paths]
                self.assertEqual([
                    'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
                    'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v001.cdf',
                    'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
                    'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
                    'imap_glows_l3c_sw-profile_20200201-cr00002_v001.cdf',
                    'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
                    'imap_glows_plasma-speed-2026d_19470301_v001.dat',
                    'imap_glows_proton-density-2026d_19470301_v000.dat',
                    'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
                    'imap_glows_photoion-2026d_19470301_v000.dat',
                    'imap_glows_lya-2026d_19470301_v000.dat',
                    'imap_glows_electron-density-2026d_19470301_v000.dat',
                    'imap_glows_pipeline-settings-l3bcde_19470301_v000.json'
                ], pipeline_l3d_input_filenames)

                self.assertEqual(external_dependencies, actual_external_deps)

                self.assertEqual(Version(requested_major_version, 2), actual_version)
                self.assertEqual(mock_fetch_l3d_dependencies.return_value, actual_l3d_deps)
                self.assertEqual(most_recent_l3d, actual_old_l3d)


    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_should_process_l3d_when_major_version_changes(self, mock_fetch_l3d_dependencies, mock_query, _, mock_read_pipeline_settings, mock_read_cdf_parents):
        most_recent_l3d = 'imap_glows_l3d_solar-hist_19470301-cr00002_v001.cdf'
        external_dependencies = Mock()
        l3bs = [
            'imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
        ]
        l3cs = [
            'imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
        ]

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
            'lyman-alpha-composite.nc',
            'imap_glows_plasma-speed-2026d_19470301_v001.dat',
            'imap_glows_proton-density-2026d_19470301_v000.dat',
            'imap_glows_lya-2026d_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
            'imap_glows_photoion-2026d_19470301_v000.dat',
            'imap_glows_electron-density-2026d_19470301_v000.dat'

        }

        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        mock_query.side_effect = self._create_ancillary_query_results([
            most_recent_l3d
        ])
        new_major_version = 1
        actual_version, actual_l3d_deps, actual_old_l3d = GlowsL3DInitializer.should_process_l3d(external_dependencies,
                                                                                                 l3bs, l3cs,
                                                                                                 ReprocessInfo([]),
                                                                                                 new_major_version)

        self.assertEqual(Version(new_major_version, 2), actual_version)
        self.assertEqual(mock_fetch_l3d_dependencies.return_value, actual_l3d_deps)
        self.assertEqual(most_recent_l3d, actual_old_l3d)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings', return_value={"start_cr": 0})
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_returns_no_old_cdf_if_none_found(self, mock_fetch_l3d_deps, mock_query,
                                                              mock_read_cdf_parents, _, __):
        l3bs = ['imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf']
        l3cs = ['imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf']

        external_deps = Mock()

        mock_query.side_effect = self._create_ancillary_query_results([])

        _, __, old_l3d = GlowsL3DInitializer.should_process_l3d(external_deps, l3bs, l3cs, ReprocessInfo([]),None)
        mock_read_cdf_parents.assert_not_called()
        mock_fetch_l3d_deps.assert_called_once()
        self.assertIsNone(old_l3d)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_should_not_process_if_same_major_version(self, mock_fetch_l3d_deps, mock_query,
                                                mock_read_cdf_parents, mock_read_pipeline_settings, _):
        l3bs = [
            'imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf'
        ]
        l3cs = [
            'imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf'
        ]

        external_dependencies = Mock()

        mock_query.side_effect = self._create_ancillary_query_results(
            ['imap_glows_l3d_solar-hist_19470301-cr00002_v001.cdf'])

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_plasma-speed-2026d_19470301_v001.dat',
            'imap_glows_proton-density-2026d_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
            'imap_glows_photoion-2026d_19470301_v000.dat',
            'imap_glows_lya-2026d_19470301_v000.dat',
            'imap_glows_electron-density-2026d_19470301_v000.dat',
            'lyman-alpha-composite.nc'
        }

        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        actual_l3d_deps = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs, ReprocessInfo({}),None)
        mock_fetch_l3d_deps.assert_not_called()
        self.assertIsNone(actual_l3d_deps)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_should_not_process_when_no_l3bs(self, mock_fetch_l3d_deps):
        l3bs = []
        l3cs = []

        external_dependencies = Mock()

        actual_l3d_deps = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs, ReprocessInfo([]),None)
        mock_fetch_l3d_deps.assert_not_called()
        self.assertIsNone(actual_l3d_deps)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_does_not_process_when_no_l3bs_after_start_cr(self, mock_fetch_l3d_deps,
                                                                          mock_read_pipeline_settings, mock_query, __):
        l3bs = ['imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf', ]
        l3cs = ['imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf', ]

        external_dependencies = Mock()

        mock_query.side_effect = self._create_ancillary_query_results([])

        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        actual_l3d_deps = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs, ReprocessInfo([]),None)
        mock_fetch_l3d_deps.assert_not_called()
        self.assertIsNone(actual_l3d_deps)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_forces_reprocess_if_reprocess_ancillary_specifies_l3e(self, mock_fetch_dependencies, mock_read_pipeline_settings, mock_query, mock_read_cdf_parents, _):
        most_recent_l3d = 'imap_glows_l3d_solar-hist_19470301-cr00002_v001.0002.cdf'

        mock_fetch_dependencies.return_value = sentinel.l3d_deps
        mock_query.side_effect = self._create_ancillary_query_results([most_recent_l3d]) * 6
        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        l3bs = [
            'imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00002_v000.cdf',
        ]
        l3cs = [
            'imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00002_v000.cdf',
        ]

        external_dependencies = Mock()

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200101-cr00002_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00000_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00002_v000.cdf',
            'imap_glows_plasma-speed-2026d_19470301_v001.dat',
            'imap_glows_proton-density-2026d_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
            'imap_glows_photoion-2026d_19470301_v000.dat',
            'imap_glows_lya-2026d_19470301_v000.dat',
            'imap_glows_electron-density-2026d_19470301_v000.dat'
        }

        cases = [
            (GLOWS_L3E_HI_45_DESCRIPTOR, {GLOWS_L3E_HI_45_DESCRIPTOR: ReprocessTargets([], [])}),
            (GLOWS_L3E_HI_90_DESCRIPTOR, {GLOWS_L3E_HI_90_DESCRIPTOR: ReprocessTargets([], [])}),
            (GLOWS_L3E_LO_DESCRIPTOR, {GLOWS_L3E_LO_DESCRIPTOR: ReprocessTargets([], [])}),
            (GLOWS_L3E_ULTRA_SF_DESCRIPTOR, {GLOWS_L3E_ULTRA_SF_DESCRIPTOR: ReprocessTargets([], [])}),
            (GLOWS_L3E_ULTRA_HF_DESCRIPTOR, {GLOWS_L3E_ULTRA_HF_DESCRIPTOR: ReprocessTargets([], [])}),
        ]

        for case, products_to_reprocess in cases:
            with self.subTest(products_to_reprocess):
                mock_fetch_dependencies.reset_mock()

                reprocess_info = ReprocessInfo(products_to_reprocess)

                version, l3d_deps, old_l3d = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs, reprocess_info, 1)

                self.assertEqual(Version(1, 3), version)
                self.assertIs(sentinel.l3d_deps, l3d_deps)
                self.assertEqual(most_recent_l3d, old_l3d)

                [fetch_dependencies_call] = mock_fetch_dependencies.call_args_list

                [actual_l3d_inputs, actual_external_deps] = fetch_dependencies_call.args

                pipeline_l3d_input_paths = actual_l3d_inputs.get_file_paths(source="glows")
                pipeline_l3d_input_filenames = [p.name for p in pipeline_l3d_input_paths]
                self.assertEqual([
                    'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
                    'imap_glows_l3b_ion-rate-profile_20200101-cr00002_v000.cdf',
                    'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
                    'imap_glows_l3c_sw-profile_20200101-cr00002_v000.cdf',
                    'imap_glows_plasma-speed-2026d_19470301_v001.dat',
                    'imap_glows_proton-density-2026d_19470301_v000.dat',
                    'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
                    'imap_glows_photoion-2026d_19470301_v000.dat',
                    'imap_glows_lya-2026d_19470301_v000.dat',
                    'imap_glows_electron-density-2026d_19470301_v000.dat',
                    'imap_glows_pipeline-settings-l3bcde_19470301_v000.json'
                ], pipeline_l3d_input_filenames)

                self.assertEqual(external_dependencies, actual_external_deps)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_pipeline_settings')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_does_not_force_reprocess_if_no_l3e_products(self, mock_fetch_dependencies, mock_read_pipeline_settings, mock_query, mock_read_cdf_parents, _):
        most_recent_l3d = 'imap_glows_l3d_solar-hist_19470301-cr00002_v001.0002.cdf'

        mock_fetch_dependencies.return_value = sentinel.l3d_deps
        mock_query.side_effect = self._create_ancillary_query_results([most_recent_l3d]) * 2
        mock_read_pipeline_settings.return_value = {"start_cr": 1}

        l3bs = ['imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf']
        l3cs = ['imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',]

        external_dependencies = Mock()

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_plasma-speed-2026d_19470301_v001.dat',
            'imap_glows_proton-density-2026d_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2026d_19470301_v000.dat',
            'imap_glows_photoion-2026d_19470301_v000.dat',
            'imap_glows_lya-2026d_19470301_v000.dat',
            'imap_glows_electron-density-2026d_19470301_v000.dat'
        }

        cases = [
            ("no l3e products to reprocess", {GLOWS_L3B_DESCRIPTOR: ReprocessTargets([], [])}),
            ("no products", {})
        ]

        for case, products_to_reprocess in cases:
            reprocess_info = ReprocessInfo(products_to_reprocess)

            self.assertIsNone(GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs,
                                                                            reprocess_info, 1))



    def _create_ancillary_query_results(self, l3d_query_result):
        return [
            create_mock_query_results(l3d_query_result),
            create_mock_query_results(['imap_glows_plasma-speed-2026d_19470301_v001.dat']),
            create_mock_query_results(['imap_glows_proton-density-2026d_19470301_v000.dat']),
            create_mock_query_results(['imap_glows_uv-anisotropy-2026d_19470301_v000.dat']),
            create_mock_query_results(['imap_glows_photoion-2026d_19470301_v000.dat']),
            create_mock_query_results(['imap_glows_lya-2026d_19470301_v000.dat']),
            create_mock_query_results(['imap_glows_electron-density-2026d_19470301_v000.dat']),
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_19470301_v000.json']),
        ]
