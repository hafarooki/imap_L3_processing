import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import sentinel, patch, call

import numpy as np
from spacepy import pycdf

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH, CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.models import GlowsL3BIonizationRate, GlowsL3CSolarWind, CRToProcess, \
    ExternalDependencies, F107_FLUX_TABLE_URL, LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from tests.swapi.cdf_model_test_case import CdfModelTestCase
from tests.test_helpers import get_test_instrument_team_data_path


class TestModels(CdfModelTestCase):

    @patch("imap_l3_processing.glows.l3bc.models.ExternalDependencies._comment_headers")
    @patch("imap_l3_processing.glows.l3bc.models.download_external_dependency")
    def test_external_dependencies_fetch_dependencies(self, mock_download_external_dependencies, mock_comment_header):
        mock_download_external_dependencies.side_effect = [
            Path("external/file/f107_path"),
            Path("external/file/lya_path"),
            Path("external/file/omni_path"),
        ]

        external_dependencies = ExternalDependencies.fetch_dependencies()

        mock_download_external_dependencies.assert_has_calls([
            call(F107_FLUX_TABLE_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt'),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, TEMP_CDF_FOLDER_PATH / 'lyman_alpha.txt'),
            call(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'omni2.txt')
        ])

        mock_comment_header.assert_called_once_with(Path("external/file/f107_path"))

        expected_external_deps = ExternalDependencies(
            f107_index_file_path=Path("external/file/f107_path"),
            lyman_alpha_path=Path("external/file/lya_path"),
            omni2_data_path=Path("external/file/omni_path"),
        )
        self.assertEqual(expected_external_deps, external_dependencies)

    @patch("imap_l3_processing.glows.l3bc.models.download_external_dependency")
    def test_external_dependencies_files_not_found(self, mock_download_external_dependencies):
        mock_download_external_dependencies.return_value = None

        external_dependencies = ExternalDependencies.fetch_dependencies()

        self.assertEqual(ExternalDependencies(None, None, None), external_dependencies)

    @patch("imap_l3_processing.glows.l3bc.models.download_external_dependency")
    def test_external_dependencies_comment_headers(self, mock_download_external_dependencies):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(tmpdir) / "flux_table.txt"
            with open(file_name, "w") as fp:
                fp.writelines([
                    "fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi\n",
                    "----------  ----------  ------------  --------------  -----------  -----------  ----------\n",
                    "20041028    170000      02453307.229  002022.605      000132.7     000130.9     000117.8\n",
                    "20041028    200000      02453307.354  002022.610      000135.8     000134.0     000120.6\n"])

            mock_download_external_dependencies.side_effect = [
                file_name,
                None,
                None
            ]

            ExternalDependencies.fetch_dependencies()

            with open(file_name, "r") as fp:
                lines = fp.readlines()
                self.assertEqual("#", lines[0][0])
                self.assertEqual("#", lines[1][0])

    @patch("imap_l3_processing.glows.l3bc.models.datetime")
    @patch("imap_l3_processing.glows.l3bc.models.imap_data_access.download")
    def test_cr_to_process_buffer_time_has_elapsed_since_cr(self, mock_download, mock_datetime):
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            fake_pipeline_settings = temp_dir / "fake_pipeline_settings.json"

            pipeline_settings = {"initializer_time_delta_days": 28.2}
            fake_pipeline_settings.write_text(json.dumps(pipeline_settings))

            mock_download.return_value = fake_pipeline_settings

            cr_to_process = CRToProcess(
                l3a_file_names=set(),
                uv_anisotropy_file_name="uv_anistropy.dat",
                waw_helio_ion_mp_file_name="waw_helion.dat",
                bad_days_list_file_name="bad_days.dat",
                pipeline_settings_file_name="pipeline_settings.json",
                cr_start_date=datetime(2010, 1, 1),
                cr_end_date=datetime(2010, 2, 1),
                cr_rotation_number=2091,
            )

            test_cases = [
                (datetime(2010, 2, 28), False),
                (datetime(2010, 3, 10), True)
            ]

            for current_time, expected_buffer_time_has_elapsed_since_cr in test_cases:
                with self.subTest(current_time):
                    mock_datetime.now.reset_mock()
                    mock_datetime.now.return_value = current_time

                    actual = cr_to_process.buffer_time_has_elapsed_since_cr()
                    self.assertEqual(expected_buffer_time_has_elapsed_since_cr, actual)

                    mock_download.assert_called_with("pipeline_settings.json")

                    mock_datetime.now.assert_called_once()

    @patch("imap_l3_processing.glows.l3bc.models.validate_dependencies")
    @patch("imap_l3_processing.glows.l3bc.models.imap_data_access.download")
    def test_cr_to_process_has_valid_external_dependencies(self, mock_download, mock_validate_dependencies):
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            fake_pipeline_settings = temp_dir / "fake_pipeline_settings.json"

            pipeline_settings = {"initializer_time_delta_days": 30.1}
            fake_pipeline_settings.write_text(json.dumps(pipeline_settings))

            mock_download.return_value = fake_pipeline_settings

            cr_to_process = CRToProcess(
                l3a_file_names=set(),
                uv_anisotropy_file_name="uv_anistropy.dat",
                waw_helio_ion_mp_file_name="waw_helion.dat",
                bad_days_list_file_name="bad_days.dat",
                pipeline_settings_file_name="pipeline_settings.json",
                cr_start_date=datetime(2010, 1, 1),
                cr_end_date=datetime(2010, 2, 1),
                cr_rotation_number=2091
            )

            external_dependencies = ExternalDependencies(
                f107_index_file_path=Path("f107_index_file_path"),
                lyman_alpha_path=Path("lyman_alpha_path"),
                omni2_data_path=Path("omni2_data_path")
            )

            actual_has_valid_ext_deps = cr_to_process.has_valid_external_dependencies(external_dependencies)

            mock_validate_dependencies.assert_called_once_with(
                datetime(2010, 1, 1),
                datetime(2010, 2, 1),
                timedelta(days=30.1),
                Path("omni2_data_path"),
                Path("f107_index_file_path"),
                Path("lyman_alpha_path"),
            )
            self.assertEqual(mock_validate_dependencies.return_value, actual_has_valid_ext_deps)

    def test_l3b_to_data_product_variables(self):
        data = GlowsL3BIonizationRate(input_metadata=sentinel.input_metadata,
                                      epoch=sentinel.epoch,
                                      epoch_delta=sentinel.epoch_delta,
                                      mean_time=sentinel.mean_time,
                                      cr=sentinel.cr,
                                      uv_anisotropy_factor=sentinel.uv_anisotropy_factor,
                                      lat_grid=sentinel.lat_grid,
                                      sum_rate=sentinel.sum_rate,
                                      ph_rate=sentinel.ph_rate,
                                      cx_rate=sentinel.cx_rate,
                                      sum_uncert=sentinel.sum_uncert,
                                      ph_uncert=sentinel.ph_uncert,
                                      cx_uncert=sentinel.cx_uncert,
                                      lat_grid_label=sentinel.lat_grid_label,
                                      uv_anisotropy_flag=sentinel.uv_anisotropy_flag,
                                      used_l3a=sentinel.used_l3a,
                                      glows_flags=sentinel.glows_flags,
                                      )

        variables = data.to_data_product_variables()

        variables = iter(variables)
        self.assert_variable_attributes(next(variables), sentinel.epoch, "epoch")
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, "epoch_delta")
        self.assert_variable_attributes(next(variables), sentinel.mean_time, "mean_time")
        self.assert_variable_attributes(next(variables), sentinel.cr, "cr")
        self.assert_variable_attributes(next(variables), sentinel.uv_anisotropy_factor, "uv_anisotropy_factor")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid, "lat_grid")
        self.assert_variable_attributes(next(variables), sentinel.sum_rate, "sum_rate")
        self.assert_variable_attributes(next(variables), sentinel.ph_rate, "ph_rate")
        self.assert_variable_attributes(next(variables), sentinel.cx_rate, "cx_rate")
        self.assert_variable_attributes(next(variables), sentinel.sum_uncert, "sum_uncert")
        self.assert_variable_attributes(next(variables), sentinel.ph_uncert, "ph_uncert")
        self.assert_variable_attributes(next(variables), sentinel.cx_uncert, "cx_uncert")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_label, "lat_grid_label")
        self.assert_variable_attributes(next(variables), sentinel.uv_anisotropy_flag, "uv_anisotropy_flag")
        self.assert_variable_attributes(next(variables), sentinel.used_l3a, "used_l3a")
        self.assert_variable_attributes(next(variables), sentinel.glows_flags, "glows_flags")

        self.assertIsNone(next(variables, None))

    def test_l3b_from_instrument_team_dictionary(self):
        glows_instrument_team_data_path = get_test_instrument_team_data_path('glows')
        with open(glows_instrument_team_data_path / 'imap_glows_l3b_cr_2091_v00.json') as f:
            instrument_team_l3b_dict = json.load(f)

        instrument_team_l3b_dict['header']['ancillary_data_files'][
            'pipeline_settings'] = glows_instrument_team_data_path / 'imap_glows_pipeline-settings-l3bcde_20250514_v004.json'

        result = GlowsL3BIonizationRate.from_instrument_team_dictionary(instrument_team_l3b_dict,
                                                                        sentinel.input_metadata)

        latitude_grid = np.array(instrument_team_l3b_dict["ion_rate_profile"]["lat_grid"])
        self.assertIsInstance(result, GlowsL3BIonizationRate)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)

        np.testing.assert_array_equal([datetime.fromisoformat("2009-12-20 20:14:51.360")], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)
        np.testing.assert_equal([datetime.fromisoformat("2010-01-02 03:43:28.667")], result.mean_time)
        self.assertEqual([10000], result.uv_anisotropy_flag)
        self.assertEqual([2091], result.cr)
        np.testing.assert_equal([instrument_team_l3b_dict["uv_anisotropy_factor"]], result.uv_anisotropy_factor)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["sum_rate"]], result.sum_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["ph_rate"]], result.ph_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["cx_rate"]], result.cx_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["sum_uncert"]], result.sum_uncert)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["ph_uncert"]], result.ph_uncert)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["cx_uncert"]], result.cx_uncert)
        np.testing.assert_equal(
            ['-90°', '-80°', '-70°', '-60°', '-50°', '-40°', '-30°', '-20°', '-10°', '0°', '10°', '20°', '30°', '40°',
             '50°', '60°', '70°', '80°', '90°'], result.lat_grid_label)
        self.assertEqual([
            'imap_glows_WawHelioIonMP_v002.json',
            'imap_glows_bad-days-list_v001.dat',
            'imap_glows_pipeline-settings-l3bcde_20250514_v004.json',
            'imap_glows_uv-anisotropy-1CR_v002.json',
            'imap_glows_plasma-speed-2010a_v003.dat',
            'imap_glows_proton-density-2010a_v003.dat',
            'imap_glows_uv-anisotropy-2010a_v003.dat',
            'imap_glows_photoion-2010a_v003.dat',
            'imap_glows_lya-2010a_v003.dat',
            'imap_glows_electron-density-2010a_v003.dat',
            'f107_fluxtable.txt'
        ], result.parent_file_names)
        np.testing.assert_equal(result.used_l3a[0, :3], np.array([
            "imap_glows_l3a_20100101000000_orbX_modX_p_v00.json",
            "imap_glows_l3a_20100102000000_orbX_modX_p_v00.json",
            "imap_glows_l3a_20100103000000_orbX_modX_p_v00.json"]))
        self.assertEqual((1, 50), result.used_l3a.shape)
        np.testing.assert_array_equal(result.used_l3a[0, 3:], "")
        np.testing.assert_equal([instrument_team_l3b_dict["glows_flags"]], result.glows_flags)

    def test_l3c_to_data_product_variables(self):
        data = GlowsL3CSolarWind(input_metadata=sentinel.input_metadata,
                                 epoch=sentinel.epoch,
                                 epoch_delta=sentinel.epoch_delta,
                                 mean_time=sentinel.mean_time,
                                 cr=sentinel.cr,
                                 lat_grid=sentinel.lat_grid,
                                 lat_grid_label=sentinel.lat_grid_label,
                                 plasma_speed_ecliptic=sentinel.plasma_speed_ecliptic,
                                 proton_density_ecliptic=sentinel.proton_density_ecliptic,
                                 alpha_abundance_ecliptic=sentinel.alpha_abundance_ecliptic,
                                 plasma_speed_profile=sentinel.plasma_speed_profile,
                                 proton_density_profile=sentinel.proton_density_profile,
                                 glows_flags=sentinel.glows_flags,
                                 )

        variables = data.to_data_product_variables()

        variables = iter(variables)
        self.assert_variable_attributes(next(variables), sentinel.epoch, "epoch",
                                        expected_data_type=pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, "epoch_delta",
                                        expected_data_type=pycdf.const.CDF_INT8)
        self.assert_variable_attributes(next(variables), sentinel.mean_time, "mean_time",
                                        expected_data_type=pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(next(variables), sentinel.cr, "cr", expected_data_type=pycdf.const.CDF_INT2)
        self.assert_variable_attributes(next(variables), sentinel.lat_grid, "lat_grid",
                                        expected_data_type=pycdf.const.CDF_FLOAT, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_label, "lat_grid_label",
                                        expected_data_type=pycdf.const.CDF_CHAR, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), sentinel.plasma_speed_ecliptic, "plasma_speed_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.proton_density_ecliptic, "proton_density_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.alpha_abundance_ecliptic, "alpha_abundance_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.plasma_speed_profile, "plasma_speed_profile",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.proton_density_profile, "proton_density_profile",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.glows_flags, "glows_flags")

        self.assertIsNone(next(variables, None))

    def test_l3c_from_instrument_team_dictionary(self):
        glows_instrument_team_data_path = get_test_instrument_team_data_path('glows')
        with open(glows_instrument_team_data_path / 'imap_glows_l3c_cr_2091_v00.json') as f:
            model = json.load(f)
        model['header']['ancillary_data_files'][
            'pipeline_settings'] = glows_instrument_team_data_path / 'imap_glows_pipeline-settings-L3bc_20250707_v002.json'
        latitude_grid = model["solar_wind_profile"]["lat_grid"]
        result = GlowsL3CSolarWind.from_instrument_team_dictionary(
            model, sentinel.input_metadata)

        self.assertIsInstance(result, GlowsL3CSolarWind)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)
        np.testing.assert_array_equal([datetime.fromisoformat("2009-12-20 20:14:51.360")], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)
        np.testing.assert_array_equal([datetime.fromisoformat("2010-01-02 03:43:28.667")], result.mean_time)

        self.assertEqual([2091], result.cr)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        self.assertEqual(
            ['-90°', '-80°', '-70°', '-60°', '-50°', '-40°', '-30°', '-20°', '-10°', '0°', '10°', '20°', '30°', '40°',
             '50°', '60°', '70°', '80°', '90°'], result.lat_grid_label)
        np.testing.assert_equal([model["solar_wind_profile"]["plasma_speed"]], result.plasma_speed_profile)
        np.testing.assert_equal([model["solar_wind_profile"]["proton_density"]], result.proton_density_profile)

        self.assertEqual([model["solar_wind_ecliptic"]["plasma_speed"]], result.plasma_speed_ecliptic)
        self.assertEqual([model["solar_wind_ecliptic"]["proton_density"]], result.proton_density_ecliptic)
        self.assertEqual([model["solar_wind_ecliptic"]["alpha_abundance"]], result.alpha_abundance_ecliptic)
        np.testing.assert_equal([model["glows_flags"]], result.glows_flags)

        self.assertEqual([
            "imap_glows_WawHelioIonMP_v002.json",
            "imap_glows_bad-days-list_v001.dat",
            "imap_glows_pipeline-settings-L3bc_20250707_v002.json",
            "imap_glows_uv-anisotropy-1CR_v002.json",
            "imap_glows_plasma-speed-2010a_v003.dat",
            "imap_glows_proton-density-2010a_v003.dat",
            "imap_glows_uv-anisotropy-2010a_v003.dat",
            "imap_glows_photoion-2010a_v003.dat",
            "imap_glows_lya-2010a_v003.dat",
            "imap_glows_electron-density-2010a_v003.dat",
            "omni2_all_years.dat"
        ], result.parent_file_names)


if __name__ == '__main__':
    unittest.main()
