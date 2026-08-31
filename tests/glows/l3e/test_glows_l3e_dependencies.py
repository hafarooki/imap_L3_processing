from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, Mock, call, sentinel, mock_open

from imap_data_access import RepointInput

from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.utils import SpiceKernelTypes, FurnishMetakernelOutput
from tests.test_helpers import get_test_data_path


class TestGlowsL3EDependencies(TestCase):
    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.imap_data_access.download')
    def test_fetch_dependencies(self, mock_download):
        mock_processing_input_collection = Mock()

        mock_energy_grid_lo = Path('energy_grid_lo_path')
        mock_energy_grid_hi = Path('energy_grid_hi_sdc_path')
        mock_energy_grid_ultra = Path('energy_grid_ultra_sdc_path')
        mock_tess_xyz_8 = Path('tess_xyz_8_sdc_path')
        mock_lya_series = Path('lya_series_sdc_path')
        mock_solar_uv_anisotropy = Path('solar_uv_anisotropy_sdc_path')
        mock_speed_3d_sw = Path('speed_3d_sw_sdc_path')
        mock_density_3d_sw = Path('density_3d_sw_sdc_path')
        mock_phion_hydrogen = Path('phion_hydrogen_sdc_path')
        mock_sw_eqtr_electrons = Path('sw_eqtr_electrons_sdc_path')
        mock_pipeline_settings = Path('l3bcde_pipeline_settings.json')
        mock_tess_ang_16 = Path('tess_ang_16_path')
        mock_repoint_file = Path('repoint.csv')

        mock_processing_input_collection.get_file_paths.side_effect = [
            [mock_lya_series],
            [mock_solar_uv_anisotropy],
            [mock_speed_3d_sw],
            [mock_density_3d_sw],
            [mock_phion_hydrogen],
            [mock_sw_eqtr_electrons],
            [mock_pipeline_settings],
            [mock_tess_xyz_8],
            [mock_tess_ang_16],
            [mock_energy_grid_lo],
            [mock_energy_grid_hi],
            [mock_energy_grid_ultra],
            [mock_repoint_file]
        ]

        mock_lya_series_path = Mock()
        mock_solar_uv_anisotropy_path = Mock()
        mock_speed_3d_sw_path = Mock()
        mock_density_3d_sw_path = Mock()
        mock_phion_hydrogen_path = Mock()
        mock_sw_eqtr_electrons_path = Mock()
        pipeline_settings_path = get_test_data_path("glows/imap_glows_pipeline-settings-l3bcde_20251113_v004.json")
        mock_energy_grid_lo_path = Mock()
        mock_tess_xyz_8_path = Mock()
        mock_energy_grid_hi_path = Mock()
        mock_energy_grid_ultra_path = Mock()
        mock_tess_ang_16_path = Mock()
        mock_downloaded_repoint_file = Mock()

        mock_download.side_effect = [
            mock_lya_series_path,
            mock_solar_uv_anisotropy_path,
            mock_speed_3d_sw_path,
            mock_density_3d_sw_path,
            mock_phion_hydrogen_path,
            mock_sw_eqtr_electrons_path,
            pipeline_settings_path,
            mock_tess_xyz_8_path,
            mock_tess_ang_16_path,
            mock_energy_grid_lo_path,
            mock_energy_grid_hi_path,
            mock_energy_grid_ultra_path,
            mock_downloaded_repoint_file,
        ]

        actual_dependencies = GlowsL3EDependencies.fetch_dependencies(mock_processing_input_collection)

        self.assertEqual(13, mock_processing_input_collection.get_file_paths.call_count)

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source="glows", descriptor="lya"),
            call(source="glows", descriptor="uv-anis"),
            call(source="glows", descriptor="speed"),
            call(source="glows", descriptor="p-dens"),
            call(source="glows", descriptor="phion"),
            call(source="glows", descriptor="e-dens"),
            call(source="glows", descriptor="pipeline-settings-l3bcde"),
            call(source="glows", descriptor="tess-xyz-8"),
            call(source="glows", descriptor="tess-ang-16"),
            call(source="glows", descriptor="energy-grid-lo"),
            call(source="glows", descriptor="energy-grid-hi"),
            call(source="glows", descriptor="energy-grid-ultra"),
            call(data_type=RepointInput.data_type),
        ])

        mock_download.assert_has_calls([
            call(mock_lya_series),
            call(mock_solar_uv_anisotropy),
            call(mock_speed_3d_sw),
            call(mock_density_3d_sw),
            call(mock_phion_hydrogen),
            call(mock_sw_eqtr_electrons),
            call(mock_pipeline_settings),
            call(mock_tess_xyz_8),
            call(mock_tess_ang_16),
            call(mock_energy_grid_lo),
            call(mock_energy_grid_hi),
            call(mock_energy_grid_ultra),
            call(mock_repoint_file),
        ], any_order=False)

        expected_pipeline_settings = {"executable_dependency_paths": {
            "energy-grid-lo": "EnGridLo.dat",
            "energy-grid-hi": "EnGridHi.dat",
            "energy-grid-ultra": "EnGridUltra.dat",
            "tess-xyz-8": "tessXYZ8.dat",
            "tess-ang-16": "tessAng16.dat",
            "ionization-files": "ionization.files.dat",
        }}

        self.assertEqual(mock_lya_series_path, actual_dependencies.lya_series)
        self.assertEqual(mock_solar_uv_anisotropy_path, actual_dependencies.solar_uv_anisotropy)
        self.assertEqual(mock_speed_3d_sw_path, actual_dependencies.speed_3d_sw)
        self.assertEqual(mock_density_3d_sw_path, actual_dependencies.density_3d_sw)
        self.assertEqual(mock_phion_hydrogen_path, actual_dependencies.phion_hydrogen)
        self.assertEqual(mock_sw_eqtr_electrons_path, actual_dependencies.sw_eqtr_electrons)
        self.assertEqual(expected_pipeline_settings['executable_dependency_paths'],
                         actual_dependencies.pipeline_settings['executable_dependency_paths'])
        self.assertEqual(mock_energy_grid_lo_path, actual_dependencies.energy_grid_lo)
        self.assertEqual(mock_energy_grid_hi_path, actual_dependencies.energy_grid_hi)
        self.assertEqual(mock_energy_grid_ultra_path, actual_dependencies.energy_grid_ultra)
        self.assertEqual(mock_tess_xyz_8_path, actual_dependencies.tess_xyz_8)
        self.assertEqual(mock_tess_ang_16_path, actual_dependencies.tess_ang16)
        self.assertEqual(
            pipeline_settings_path, actual_dependencies.pipeline_settings_file
        )

        self.assertEqual(
            mock_downloaded_repoint_file, actual_dependencies.repointing_file
        )

    @patch("imap_l3_processing.glows.l3e.glows_l3e_dependencies.furnish_spice_metakernel")
    def test_collect_spice_dependencies(self, mock_furnish_spice_metakernel):
        expected_kernel_types_with_predicted = [
            SpiceKernelTypes.ScienceFrames,
            SpiceKernelTypes.EphemerisPredicted,
            SpiceKernelTypes.EphemerisReconstructed,
            SpiceKernelTypes.AttitudeHistory,
            SpiceKernelTypes.PointingAttitude,
            SpiceKernelTypes.PlanetaryEphemeris,
            SpiceKernelTypes.Leapseconds,
            SpiceKernelTypes.SpacecraftClock
        ]
        expected_kernel_types_without_predicted = [
            x for x in expected_kernel_types_with_predicted if x != SpiceKernelTypes.EphemerisPredicted
        ]

        metakernel_with_predict_path = Path("metakernel_with_predict_ephem.txt")
        kernels_with_predict = [Path("some_kernel_with_imap_data"), Path("predicted_ephem"), Path("some_kernel_with_planet_data")]
        metakernel_without_predict_path = Path("metakernel_without_predict_ephem.txt")
        kernels_without_predict = [Path("some_kernel_with_imap_data"), Path("some_kernel_with_planet_data")]
        metakernel_output_with_predict = FurnishMetakernelOutput(metakernel_path=metakernel_with_predict_path,
                                         spice_kernel_paths=kernels_with_predict, )
        metakernel_output_without_predict = FurnishMetakernelOutput(metakernel_path=metakernel_without_predict_path,
                                         spice_kernel_paths=kernels_without_predict)
        mock_furnish_spice_metakernel.side_effect = [
            metakernel_output_with_predict,
            metakernel_output_without_predict
        ]

        actual_metakernel_with_predict, actual_metakernel_without_predict = GlowsL3EDependencies.collect_spice_dependencies(sentinel.start_date, sentinel.end_date)

        mock_furnish_spice_metakernel.assert_has_calls([
            call(
                start_date=sentinel.start_date,
                end_date=sentinel.end_date,
                kernel_types=expected_kernel_types_with_predicted,
                metakernel_file_name="metakernel_with_predict_ephem.txt",
            ),
            call(
                start_date=sentinel.start_date,
                end_date=sentinel.end_date,
                kernel_types=expected_kernel_types_without_predicted,
                metakernel_file_name="metakernel_without_predict_ephem.txt",

            ),
        ])

        self.assertEqual(metakernel_output_with_predict, actual_metakernel_with_predict)
        self.assertEqual(metakernel_output_without_predict, actual_metakernel_without_predict)



    def test_get_hi_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_hi.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_hi_parents())

    def test_get_lo_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_lo.dat",
            "tess_xyz_8.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_lo_parents())

    def test_get_ul_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_ultra.dat",
            "tess_ang16.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_ul_parents())

    @patch('builtins.open', new_callable=mock_open)
    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.shutil')
    def test_copy_dependencies(self, mock_shutil, mock_file):
        glows_l3e_dependencies = GlowsL3EDependencies(
            Path('2025/05/03/imap_glows_energy_grid_lo'),
            Path('2025/05/03/imap_glows_energy_grid_hi'),
            Path('2025/05/03/imap_glows_energy_grid_ultra'),
            Path('2025/05/03/imap_glows_tess_xyz_8'),
            Path('2025/05/03/imap_glows_tess_ang16'),
            Path('2025/05/03/imap_glows_lya_series'),
            Path('2025/05/03/imap_glows_solar_uv_anisotropy'),
            Path('2025/05/03/imap_glows_speed_3d_sw'),
            Path('2025/05/03/imap_glows_density_3d_sw'),
            Path('2025/05/03/imap_glows_phion_hydrogen'),
            Path('2025/05/03/imap_glows_sw_eqtr_electrons'),
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
            Path("repoint.csv")
        )

        expected_energy_grid_lo = 'EnGridLo.dat'
        expected_energy_grid_hi = 'EnGridHi.dat'
        expected_energy_grid_ultra = 'EnGridUltra.dat'
        expected_tess_xyz_8 = 'tessXYZ8.dat'
        expected_tess_ang16 = 'tessAng16.dat'

        glows_l3e_dependencies.copy_dependencies()

        mock_shutil.copy.assert_has_calls([
            call(glows_l3e_dependencies.energy_grid_lo, expected_energy_grid_lo),
            call(glows_l3e_dependencies.energy_grid_hi, expected_energy_grid_hi),
            call(glows_l3e_dependencies.energy_grid_ultra, expected_energy_grid_ultra),
            call(glows_l3e_dependencies.tess_xyz_8, expected_tess_xyz_8),
            call(glows_l3e_dependencies.tess_ang16, expected_tess_ang16),
            call(glows_l3e_dependencies.lya_series, glows_l3e_dependencies.lya_series.name),
            call(glows_l3e_dependencies.solar_uv_anisotropy, glows_l3e_dependencies.solar_uv_anisotropy.name),
            call(glows_l3e_dependencies.speed_3d_sw, glows_l3e_dependencies.speed_3d_sw.name),
            call(glows_l3e_dependencies.density_3d_sw, glows_l3e_dependencies.density_3d_sw.name),
            call(glows_l3e_dependencies.phion_hydrogen, glows_l3e_dependencies.phion_hydrogen.name),
            call(glows_l3e_dependencies.sw_eqtr_electrons, glows_l3e_dependencies.sw_eqtr_electrons.name),
        ])

        mock_file.assert_called_once_with("ionization.files.dat", "w")
        mock_file.return_value.write.assert_called_once_with((
            f"{glows_l3e_dependencies.lya_series.name}\n"
            f"{glows_l3e_dependencies.solar_uv_anisotropy.name}\n"
            f"{glows_l3e_dependencies.speed_3d_sw.name}\n"
            f"{glows_l3e_dependencies.density_3d_sw.name}\n"
            f"{glows_l3e_dependencies.phion_hydrogen.name}\n"
            f"{glows_l3e_dependencies.sw_eqtr_electrons.name}\n"
        ))

    def _create_l3e_dependencies(self) -> GlowsL3EDependencies:
        return GlowsL3EDependencies(
            energy_grid_lo=Path("some/folder/energy_grid_lo.dat"),
            energy_grid_hi=Path("some/folder/energy_grid_hi.dat"),
            energy_grid_ultra=Path("some/folder/energy_grid_ultra.dat"),
            tess_xyz_8=Path("some/folder/tess_xyz_8.dat"),
            tess_ang16=Path("some/folder/tess_ang16.dat"),
            lya_series=Path("some/folder/lya_series.dat"),
            solar_uv_anisotropy=Path("some/folder/solar_uv_anisotropy.dat"),
            speed_3d_sw=Path("some/folder/speed_3d_sw.dat"),
            density_3d_sw=Path("some/folder/density_3d_sw.dat"),
            phion_hydrogen=Path("some/folder/phion_hydrogen.dat"),
            sw_eqtr_electrons=Path("some/folder/sw_eqtr_electrons.dat"),
            pipeline_settings_file=Path("some/folder/pipeline_settings.json"),
            pipeline_settings={},
            repointing_file=Path("some/folder/repointing_file.csv"),
        )
