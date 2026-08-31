import dataclasses
import json
import logging
import os
import shutil
import subprocess
import unittest
from datetime import timedelta, datetime
from functools import wraps
from pathlib import Path
from typing import Callable
from unittest import skipIf, skip
from unittest.mock import patch, Mock

import imap_data_access
import numpy as np
from imap_data_access import ProcessingInputCollection, RepointInput
from imap_data_access import ScienceInput, AncillaryInput
from imap_data_access.file_validation import ScienceFilePath, AncillaryFilePath, Version
from spacepy.pycdf import CDF

import imap_l3_data_processor
import tests
from imap_l3_processing.glows.descriptors import GLOWS_L3BCDE_DESCRIPTORS
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.utils import save_data
from tests.integration.integration_test_helpers import mock_imap_data_access, run_istp_compliance_check
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, \
    with_tempdir, get_run_local_data_path, run_periodically, create_mock_version_map

GLOWS_INTEGRATION_DATA_DIR = get_run_local_data_path(
    "glows_l3bcde_integration_data_dir"
)
INTEGRATION_TEST_DATA = Path(__file__).parent / "test_data"
GLOWS_TEST_DATA = get_test_data_path("glows")


def generate_test_function_import_path(fn: Callable) -> str:
    path_to_imap_processing_dir = Path(tests.__file__).parent.parent
    path_to_file = Path(fn.__code__.co_filename).relative_to(path_to_imap_processing_dir)
    path_in_import_style = str(path_to_file).replace('.py', '.').replace(os.path.sep, '.')
    return path_in_import_style + fn.__qualname__


def run_test_in_docker(test_to_run: Callable):
    @wraps(test_to_run)
    def decorated(self):
        if os.getenv("IN_GLOWS_INTEGRATION_DOCKER"):
            test_to_run(self)
        else:
            l3_processing_dir = Path(tests.__file__).parent.parent

            docker_build = subprocess.run(["docker", "build", "--platform", "linux/amd64", "-q", "-f", "Dockerfile_glows_integration", "."],
                                          cwd=l3_processing_dir, capture_output=True)
            image_hash = docker_build.stdout.strip().decode('utf-8')

            print(f"Built docker container: {image_hash}")

            args = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                "--mount", f'type=bind,src={l3_processing_dir}/temp_cdf_data,dst=/temp_cdf_data',
                "--mount", f'type=bind,src={l3_processing_dir}/run_local_input_data,dst=/run_local_input_data',
            ]

            if imap_api_key := os.getenv("IMAP_API_KEY"):
                args += ["-e", f"IMAP_API_KEY={imap_api_key}"]

            if imap_data_access_url := os.getenv("IMAP_DATA_ACCESS_URL"):
                args += ["-e", f"IMAP_DATA_ACCESS_URL={imap_data_access_url}"]

            args += [image_hash, generate_test_function_import_path(test_to_run)]

            subprocess.run(args, cwd=l3_processing_dir, check=True)

    return decorated

class TestGlowsProcessorIntegration(unittest.TestCase):
    @with_tempdir
    def test_glows_l3a(self, tmp_dir):
        input_l2_cdf_path = self._fill_official_l2_cdf_with_json_values(tmp_dir)

        l2_science_file_path = ScienceFilePath(input_l2_cdf_path)

        date_in_path = l2_science_file_path.start_date
        start_date = datetime.strptime(date_in_path, "%Y%m%d")
        end_date = start_date + timedelta(days=1)
        l3a_descriptor = 'hist'
        input_metadata = InputMetadata(
            instrument='glows',
            data_level='l3a',
            descriptor=l3a_descriptor,
            start_date=start_date,
            end_date=end_date,
            version=create_mock_version_map(descriptor=l3a_descriptor,minor_version=1),
            repointing=l2_science_file_path.repointing
        )

        expected_json_path = get_test_instrument_team_data_path(
            "glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")
        with open(expected_json_path) as f:
            instrument_team_dict = json.load(f)
        expected_output = create_glows_l3a_from_dictionary(instrument_team_dict, input_metadata)

        with CDF(str(input_l2_cdf_path)) as cdf_data:
            l2_glows_data = read_l2_glows_data(cdf_data)

        dependencies = GlowsL3ADependencies(l2_glows_data, {
            "calibration_data": get_test_instrument_team_data_path(
                "glows/imap_glows_calibration-data_20100101_v002.dat"),
            "settings": get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings_20100101_v001.json"),
            "time_dependent_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_time-dep-bckgrd_20100101_v001.dat"),
            "extra_heliospheric_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_map-of-extra-helio-bckgrd_20100101_v002.dat"),
        })

        processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)
        l3a_data = processor.process_l3a(dependencies)

        output_cdf_path = save_data(l3a_data, delete_if_present=True)
        print(output_cdf_path)

        expected_dict = dataclasses.asdict(expected_output)
        actual_dict = dataclasses.asdict(l3a_data)

        self.assertEqual(input_metadata.repointing, l3a_data.identifier)

        np.testing.assert_allclose(actual_dict['photon_flux'], expected_dict['photon_flux'], rtol=1e-3)
        np.testing.assert_allclose(actual_dict['photon_flux_uncertainty'], expected_dict['photon_flux_uncertainty'],
                                   rtol=1e-3)
        np.testing.assert_allclose(actual_dict['raw_histogram'], expected_dict['raw_histogram'])
        np.testing.assert_allclose(actual_dict['exposure_times'], expected_dict['exposure_times'], rtol=1e-3)
        np.testing.assert_allclose(actual_dict['number_of_bins'], expected_dict['number_of_bins'])
        self.assertEqual(actual_dict['epoch'], expected_dict['epoch'])
        np.testing.assert_allclose(actual_dict['epoch_delta'], expected_dict['epoch_delta'])
        np.testing.assert_allclose(actual_dict['spin_angle'], expected_dict['spin_angle'])
        np.testing.assert_allclose(actual_dict['spin_angle_delta'], expected_dict['spin_angle_delta'])
        np.testing.assert_allclose(actual_dict['latitude'], expected_dict['latitude'], atol=1e-3)
        np.testing.assert_allclose(actual_dict['longitude'], expected_dict['longitude'], atol=1e-3)
        np.testing.assert_allclose(actual_dict['filter_temperature_average'],
                                   expected_dict['filter_temperature_average'])
        np.testing.assert_allclose(actual_dict['filter_temperature_std_dev'],
                                   expected_dict['filter_temperature_std_dev'])
        np.testing.assert_allclose(actual_dict['hv_voltage_average'], expected_dict['hv_voltage_average'])
        np.testing.assert_allclose(actual_dict['hv_voltage_std_dev'], expected_dict['hv_voltage_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_period_average'], expected_dict['spin_period_average'])
        np.testing.assert_allclose(actual_dict['spin_period_std_dev'], expected_dict['spin_period_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_period_ground_average'],
                                   expected_dict['spin_period_ground_average'])
        np.testing.assert_allclose(actual_dict['spin_period_ground_std_dev'],
                                   expected_dict['spin_period_ground_std_dev'])
        np.testing.assert_allclose(actual_dict['pulse_length_average'], expected_dict['pulse_length_average'])
        np.testing.assert_allclose(actual_dict['pulse_length_std_dev'], expected_dict['pulse_length_std_dev'])
        np.testing.assert_allclose(actual_dict['position_angle_offset_average'],
                                   expected_dict['position_angle_offset_average'])
        np.testing.assert_allclose(actual_dict['position_angle_offset_std_dev'],
                                   expected_dict['position_angle_offset_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_axis_orientation_average'],
                                   expected_dict['spin_axis_orientation_average'])
        np.testing.assert_allclose(actual_dict['spin_axis_orientation_std_dev'],
                                   expected_dict['spin_axis_orientation_std_dev'])
        np.testing.assert_allclose(actual_dict['spacecraft_location_average'],
                                   expected_dict['spacecraft_location_average'])
        np.testing.assert_allclose(actual_dict['spacecraft_location_std_dev'],
                                   expected_dict['spacecraft_location_std_dev'])

        self.assertEqual(actual_dict['input_metadata'], expected_dict['input_metadata'])

        istp_compliance_message = run_istp_compliance_check(output_cdf_path)
        print("ISTP Compliance:\n", istp_compliance_message)
        self.assertIn("PASSED variable checks", istp_compliance_message)

    def test_l3a_handles_l2_input_that_is_all_flagged(self):
        bad_l2_cdf_path = get_test_data_path("glows/imap_glows_l2_hist_20251113-repoint00047_v003.cdf")

        l2_science_file_path = ScienceFilePath(bad_l2_cdf_path)

        date_in_path = l2_science_file_path.start_date
        start_date = datetime.strptime(date_in_path, "%Y%m%d")
        end_date = start_date + timedelta(days=1)
        input_metadata = InputMetadata(
            instrument="glows",
            data_level="l3a",
            descriptor="hist",
            start_date=start_date,
            end_date=end_date,
            version=VersionMap({}, Version(None, 1)),
            repointing=l2_science_file_path.repointing,
        )

        with CDF(str(bad_l2_cdf_path)) as cdf_data:
            l2_glows_data = read_l2_glows_data(cdf_data)

        dependencies = GlowsL3ADependencies(
            l2_glows_data,
            {
                "calibration_data": get_test_instrument_team_data_path(
                    "glows/imap_glows_calibration-data_20100101_v002.dat"
                ),
                "settings": get_test_instrument_team_data_path(
                    "glows/imap_glows_pipeline-settings_20100101_v001.json"
                ),
                "time_dependent_bckgrd": get_test_instrument_team_data_path(
                    "glows/imap_glows_time-dep-bckgrd_20100101_v001.dat"
                ),
                "extra_heliospheric_bckgrd": get_test_instrument_team_data_path(
                    "glows/imap_glows_map-of-extra-helio-bckgrd_20100101_v002.dat"
                ),
            },
        )

        processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)
        l3a_output = processor.process_l3a(dependencies)

        self.assertIsNone(l3a_output)

    @run_periodically(timedelta(days=14))
    @run_test_in_docker
    def test_l3bcde_first_time_processing(self):
        input_files = [
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251113-repoint00047_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251129-repoint00063_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251201-repoint00065_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251226-repoint00090_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-1CR_20251113_v002.json",
            GLOWS_TEST_DATA / "imap_glows_WawHelioIonMP_20251113_v007.json",
            GLOWS_TEST_DATA / "imap_glows_bad-days-list_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_pipeline-settings-l3bcde_20251113_v006.json",
            GLOWS_TEST_DATA / "imap_glows_plasma-speed-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_proton-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_photoion-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_lya-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_electron-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-ang-16_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-xyz-8_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-lo_20251113_v002.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-hi_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-ultra_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_ionization-files_20251113_v002.dat",
            GLOWS_TEST_DATA / "imap_lo_l1b_nhk_20251103-repoint00037_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_force-reprocessing-config_20250101_v000.csv",
            INTEGRATION_TEST_DATA / "spice" / "imap_2026_090_01.repoint",
            INTEGRATION_TEST_DATA / "spice" / "imap_2025_105_2026_105_01.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA / "spice" / "imap_recon_20250415_20260415_v01.bsp",
        ]
        with mock_imap_data_access(GLOWS_INTEGRATION_DATA_DIR, input_files):

            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            for folder in ["data_l3b", "data_l3c", "data_l3d", "data_l3d_txt"]:
                path = PATH_TO_L3D_TOOLKIT / folder
                if path.exists():
                    shutil.rmtree(path)

            processing_input = ProcessingInputCollection(RepointInput("imap_2026_090_01.repoint"),
                                                         AncillaryInput("imap_glows_force-reprocessing-config_20250101_v000.csv"))

            version_map = VersionMap({descriptor: Version(i, 2) for i, descriptor in enumerate(GLOWS_L3BCDE_DESCRIPTORS)})
            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version=version_map, start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))

            processor = GlowsProcessor(processing_input, input_metadata)
            processor.process()

            expected_files = [
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20251102-cr02304_v000.0001.cdf'),
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20251130-cr02305_v000.0001.cdf'),

                ScienceFilePath('imap_glows_l3c_sw-profile_20251102-cr02304_v000.0001.cdf'),
                ScienceFilePath('imap_glows_l3c_sw-profile_20251130-cr02305_v000.0001.cdf'),

                ScienceFilePath('imap_glows_l3d_solar-hist_19470303-cr02306_v002.0001.cdf'),
                AncillaryFilePath('imap_glows_uv-anis_19470303_20260110_v001.dat'),
                AncillaryFilePath('imap_glows_lya_19470303_20260110_v001.dat'),
                AncillaryFilePath('imap_glows_e-dens_19470303_20260110_v001.dat'),
                AncillaryFilePath('imap_glows_p-dens_19470303_20260110_v001.dat'),
                AncillaryFilePath('imap_glows_speed_19470303_20260110_v001.dat'),
                AncillaryFilePath('imap_glows_phion_19470303_20260110_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-ul-sf_20251102-repoint00036_v006.0001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-ul-sf_20251115-repoint00049_v006.0001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-ul-sf-raw_20251102_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-ul-sf-raw_20251115_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-ul-hf_20251102-repoint00036_v007.0001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-ul-hf_20251115-repoint00049_v007.0001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-ul-hf-raw_20251102_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-ul-hf-raw_20251115_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20251102-repoint00036_v003.0001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20251115-repoint00049_v003.0001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20251102_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20251115_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20251102-repoint00036_v004.0001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20251115-repoint00049_v004.0001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20251102_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20251115_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20251102-repoint00036_v005.0001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20251115-repoint00049_v005.0001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20251102_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20251115_v001.dat'),
            ]

            for file_path in expected_files:
                self.assertTrue(file_path.construct_path().exists(), msg=str(file_path.construct_path()))

    @run_periodically(timedelta(days=14))
    @run_test_in_docker
    def test_l3bcde_automatic_reprocessing(self):
        new_l3a_file = GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251118-repoint00052_v001.cdf"
        l3bcde_input_files = [
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251113-repoint00047_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251129-repoint00063_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251201-repoint00065_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20251226-repoint00090_v001.cdf",
            new_l3a_file,
            GLOWS_TEST_DATA
            / "imap_glows_l3b_ion-rate-profile_20251102-cr02304_v001.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3b_ion-rate-profile_20251130-cr02305_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3c_sw-profile_20251102-cr02304_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3c_sw-profile_20251130-cr02305_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3d_solar-hist_19470303-cr02304_v001.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3e_survival-probability-hi-45_20251102-repoint00036_v001.0000.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3e_survival-probability-hi-90_20251102-repoint00036_v001.0000.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3e_survival-probability-lo_20251102-repoint00036_v001.0000.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3e_survival-probability-ul-hf_20251102-repoint00036_v001.0000.cdf",
            GLOWS_TEST_DATA
            / "imap_glows_l3e_survival-probability-ul-sf_20251102-repoint00036_v001.0000.cdf",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-1CR_20251113_v002.json",
            GLOWS_TEST_DATA / "imap_glows_WawHelioIonMP_20251113_v007.json",
            GLOWS_TEST_DATA / "imap_glows_bad-days-list_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_pipeline-settings-l3bcde_20251113_v006.json",
            GLOWS_TEST_DATA / "imap_glows_plasma-speed-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_proton-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_photoion-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_lya-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_electron-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-ang-16_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-xyz-8_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-lo_20251113_v002.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-hi_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-ultra_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_ionization-files_20251113_v002.dat",
            GLOWS_TEST_DATA / "imap_glows_force-reprocessing-config_20250101_v000.csv",
            INTEGRATION_TEST_DATA / "spice" / "imap_2026_090_01.repoint",
            INTEGRATION_TEST_DATA / "spice" / "imap_2025_105_2026_105_01.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA / "spice" / "imap_recon_20250415_20260415_v01.bsp",
        ]

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        with mock_imap_data_access(get_run_local_data_path("glows_reprocessing"), l3bcde_input_files):
            processing_input = ProcessingInputCollection(RepointInput("imap_2026_090_01.repoint"),
                                                         AncillaryInput("imap_glows_force-reprocessing-config_20250101_v000.csv"))
            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version=VersionMap({}, Version(1, 0)), start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))

            processor = GlowsProcessor(processing_input, input_metadata)
            processor.process()

            expected_files = [
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20251102-cr02304_v001.0002.cdf'),

                ScienceFilePath('imap_glows_l3c_sw-profile_20251102-cr02304_v001.0002.cdf'),

                ScienceFilePath('imap_glows_l3d_solar-hist_19470303-cr02306_v001.0002.cdf'),
                AncillaryFilePath('imap_glows_uv-anis_19470303_20260110_v002.dat'),
                AncillaryFilePath('imap_glows_lya_19470303_20260110_v002.dat'),
                AncillaryFilePath('imap_glows_e-dens_19470303_20260110_v002.dat'),
                AncillaryFilePath('imap_glows_p-dens_19470303_20260110_v002.dat'),
                AncillaryFilePath('imap_glows_speed_19470303_20260110_v002.dat'),
                AncillaryFilePath('imap_glows_phion_19470303_20260110_v002.dat'),

                ScienceFilePath("imap_glows_l3e_survival-probability-hi-45_20251102-repoint00036_v001.0001.cdf"),
                ScienceFilePath("imap_glows_l3e_survival-probability-hi-90_20251102-repoint00036_v001.0001.cdf"),
                ScienceFilePath("imap_glows_l3e_survival-probability-lo_20251102-repoint00036_v001.0001.cdf"),
                ScienceFilePath("imap_glows_l3e_survival-probability-ul-hf_20251102-repoint00036_v001.0001.cdf"),
                ScienceFilePath("imap_glows_l3e_survival-probability-ul-sf_20251102-repoint00036_v001.0001.cdf"),
            ]

            for file_path in expected_files:
                self.assertTrue(file_path.construct_path().exists(), msg=str(file_path.construct_path()))

    @run_periodically(timedelta(days=14))
    @run_test_in_docker
    def test_glows_l3abcde_from_l2(self):
        ancillary_file_paths = [
            GLOWS_TEST_DATA / "imap_glows_l2-calibration_20251112_v003.dat",
            GLOWS_TEST_DATA
            / "imap_glows_l3a-map-of-extra-helio-bckgrd_20251112_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_pipeline-settings_20251112_v002.json",
            GLOWS_TEST_DATA / "imap_glows_l3a-time-dep-bckgrd_20251112_v001.dat",
        ]

        prod_data_folder = get_run_local_data_path("glows_prod_data")

        l2_paths = list(prod_data_folder.rglob("*.cdf"))
        input_files = l2_paths + ancillary_file_paths

        l3a_integration_data_dir = get_run_local_data_path("glows_l3a_with_prod_l2")
        with (mock_imap_data_access(l3a_integration_data_dir, input_files)):
            for i, l2_path in enumerate(l2_paths):
                l2_science_file_path = ScienceFilePath(l2_path)
                l2_science_input = ScienceInput(l2_path.name)

                start_date, end_date = l2_science_input.get_time_range()

                input_metadata = InputMetadata(
                    instrument="glows",
                    data_level="l3a",
                    start_date=start_date,
                    end_date=end_date,
                    version=VersionMap({}, Version(None, 1)),
                    descriptor="hist",
                    repointing=l2_science_file_path.repointing,
                )

                ancillary_inputs = [AncillaryInput(ancillary.name) for ancillary in ancillary_file_paths]
                processing_inputs = [l2_science_input, *ancillary_inputs]
                processor = GlowsProcessor(ProcessingInputCollection(*processing_inputs), input_metadata)

                try:
                    _ = processor.process()
                except Exception as e:
                    print(f"Processing L3a day {start_date} failed! Reason: {e}")

        l3bcde_ancillary_inputs = [
            GLOWS_TEST_DATA / "imap_glows_WawHelioIonMP_20251113_v007.json",
            GLOWS_TEST_DATA / "imap_glows_bad-days-list_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_electron-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-hi_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-lo_20251113_v002.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-ultra_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_lya-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_photoion-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_pipeline-settings-l3bcde_20251113_v006.json",
            GLOWS_TEST_DATA / "imap_glows_plasma-speed-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_proton-density-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-ang-16_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-xyz-8_20251113_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-1CR_20251113_v002.json",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-2026d_20251113_v003.dat",
            GLOWS_TEST_DATA / "imap_glows_force-reprocessing-config_20250101_v000.csv",
            INTEGRATION_TEST_DATA / "spice" / "imap_2026_090_01.repoint",
            INTEGRATION_TEST_DATA / "spice" / "imap_2025_105_2026_105_01.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_science_120.tf",
            INTEGRATION_TEST_DATA / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA / "spice" / "imap_recon_20250925_20260520_v01.bsp",
        ]

        lo_l1b_folder = Path("/Users/harrison/Development/imap_L3_processing/data/imap/lo/l1b")
        lo_l1b_inputs = list(lo_l1b_folder.rglob("*.cdf"))

        l3a_inputs = list((l3a_integration_data_dir / "imap/glows/l3a").rglob("*.cdf"))

        repoint_file_path = [INTEGRATION_TEST_DATA / "spice" / "imap_2026_139_01.repoint"]

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        input_paths = l3bcde_ancillary_inputs + l3a_inputs + repoint_file_path + lo_l1b_inputs
        with mock_imap_data_access(get_run_local_data_path("glows_l3bcde_with_prod_l2"), input_paths):
            processing_input = ProcessingInputCollection(RepointInput("imap_2026_139_01.repoint"),
                                                         AncillaryInput("imap_glows_force-reprocessing-config_20250101_v000.csv"))
            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version=VersionMap({}, Version(None, 1)), start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))

            processor = GlowsProcessor(processing_input, input_metadata)
            processor.process()

    @skipIf(os.getenv("IMAP_API_KEY") is None, "Only runs with prod IMAP_API_KEY")
    @run_test_in_docker
    def test_run_glows_l3be_against_prod(self):
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        imap_data_access.config["DATA_DIR"] = get_run_local_data_path("glows_l3bcde_prod")

        test_reprocessing_file = get_test_data_path("glows/imap_glows_force-reprocessing-config_20250101_v000.csv")
        shutil.copy(test_reprocessing_file, AncillaryFilePath(test_reprocessing_file.name).construct_path())

        repoint_file_name = "imap_2026_227_01.repoint"
        processing_input = ProcessingInputCollection(RepointInput(repoint_file_name), AncillaryInput(test_reprocessing_file.name))

        descriptors_to_produce = [
            "ion-rate-profile",
            "sw-profile",
            "solar-hist",
            "survival-probability-hi-45",
            "survival-probability-hi-90",
            "survival-probability-lo",
            "survival-probability-ul-sf",
            "survival-probability-ul-hf",
        ]

        dependency = {
            "dependency": json.loads(processing_input.serialize()),
            "version": {
                desc: {
                    "major_version": 1,
                    "minor_version": 1,
                } for desc in descriptors_to_produce
            }
        }

        with patch("imap_l3_data_processor._parse_cli_arguments") as mock_parse_cli_arguments:
            mock_arguments = Mock()
            mock_arguments.instrument = "glows"
            mock_arguments.data_level = "l3b"
            mock_arguments.descriptor = "ion-rate-profile"
            mock_arguments.start_date = "20260101"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = json.dumps(dependency)
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments
            imap_l3_data_processor.imap_l3_processor()

    @staticmethod
    def _fill_official_l2_cdf_with_json_values(output_folder: Path) -> Path:
        official_l2_path = get_test_data_path("glows/imap_glows_l2_hist_20251224-repoint00088_v001.cdf")
        json_file_path = get_test_instrument_team_data_path("glows/imap_glows_l2_20130908085214_orbX_modX_p_v00.json")

        new_file_path = output_folder / official_l2_path.name
        new_file_path.unlink(missing_ok=True)

        with CDF(str(new_file_path), masterpath=str(official_l2_path)) as cdf:
            with open(json_file_path) as f:
                instrument_data = json.load(f)

                start_of_epoch_window = datetime.fromisoformat(instrument_data["start_time"])
                end_of_epoch_window = datetime.fromisoformat(instrument_data["end_time"])
                epoch_window = end_of_epoch_window - start_of_epoch_window
                epoch = start_of_epoch_window + epoch_window / 2

                cdf["epoch"][0] = epoch
                cdf['start_time'][0] = instrument_data["start_time"]
                cdf['end_time'][0] = instrument_data["end_time"]

                lightcurve_vars = [
                    "spin_angle",
                    "photon_flux",
                    "exposure_times",
                    "flux_uncertainties",
                    "ecliptic_lon",
                    "ecliptic_lat",
                ]
                for var in lightcurve_vars:
                    cdf[var] = np.array(instrument_data["daily_lightcurve"][var])[np.newaxis, :]

                cdf["raw_histograms"] = np.array(instrument_data["daily_lightcurve"]["raw_histogram"])[np.newaxis, :]
                cdf["histogram_flag_array"] = np.array(
                    [int(f, 16) for f in instrument_data["daily_lightcurve"]["histogram_flag_array"]])[np.newaxis, :]

                single_value_vars = [
                    "filter_temperature_average",
                    "filter_temperature_std_dev",
                    "hv_voltage_average",
                    "hv_voltage_std_dev",
                    "spin_period_average",
                    "spin_period_std_dev",
                    "pulse_length_average",
                    "pulse_length_std_dev",
                    "spin_period_ground_average",
                    "spin_period_ground_std_dev",
                    "position_angle_offset_average",
                    "position_angle_offset_std_dev",
                    "identifier"
                ]
                for var in single_value_vars:
                    cdf[var][0] = instrument_data[var]

                vector_vars = [
                    "spacecraft_location_average",
                    "spacecraft_location_std_dev",
                    "spacecraft_velocity_average",
                    "spacecraft_velocity_std_dev",
                    "spin_axis_orientation_average",
                    "spin_axis_orientation_std_dev"
                ]
                for var in vector_vars:
                    cdf[var][0] = np.array(list(instrument_data[var].values()))

                cdf["bad_time_flag_occurrences"][0] = list(instrument_data["bad_time_flag_occurences"].values())
                cdf["number_of_good_l1b_inputs"][0] = instrument_data["header"]["number_of_l1b_files_used"]
                cdf["total_l1b_inputs"][0] = instrument_data["header"]["number_of_all_l1b_files"]

        return new_file_path


if __name__ == '__main__':
    unittest.main()
