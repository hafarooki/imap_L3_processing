import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, \
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, \
    HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, HELIUM_INFLOW_VECTOR_DESCRIPTOR, \
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR, CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, \
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies


class TestSwapiL3ADependencies(unittest.TestCase):

    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_file_paths):
        input_collection = ProcessingInputCollection()

        start_date = '20100105'
        mission = 'imap'
        instrument = 'swapi'
        data_level = 'l2'
        version = 'v010'

        mock_download.side_effect = [
            sentinel.swapi_l2_data,
            sentinel.efficiency_file,
            sentinel.neutral_helium_table,
            sentinel.hydrogen_vector,
            sentinel.helium_vector,
            sentinel.azimuthal_transmission,
            sentinel.central_effective_area,
            sentinel.passband_fit_coefficients,
        ]

        swapi_science_file_download_path = f"{mission}_{instrument}_{data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_efficiency_file_name = f"{mission}_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_density_of_neutral_helium_lookup = f"{mission}_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_hydrogen_inflow_filename = f"{mission}_{instrument}_{HYDROGEN_INFLOW_VECTOR_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_helium_inflow_filename = f"{mission}_{instrument}_{HELIUM_INFLOW_VECTOR_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_azimuthal_transmission_filename = f"{mission}_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{start_date}_{version}.csv"
        swapi_central_effective_area_filename = f"{mission}_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{start_date}_{version}.csv"
        swapi_passband_fit_coefficients_filename = f"{mission}_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{start_date}_{version}.csv"

        input_collection.add([
            ScienceInput(swapi_science_file_download_path),
            AncillaryInput(swapi_efficiency_file_name),
            AncillaryInput(swapi_density_of_neutral_helium_lookup),
            AncillaryInput(swapi_hydrogen_inflow_filename),
            AncillaryInput(swapi_helium_inflow_filename),
            AncillaryInput(swapi_azimuthal_transmission_filename),
            AncillaryInput(swapi_central_effective_area_filename),
            AncillaryInput(swapi_passband_fit_coefficients_filename),
        ])

        actual_swapi_l3_dependencies = SwapiL3ADependencies.fetch_dependencies(input_collection)

        science_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'swapi' / 'l2' / '2010' / '01'
        ancillary_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'ancillary' / 'swapi'

        mock_download.assert_has_calls([
            call(science_data_dir / swapi_science_file_download_path),
            call(ancillary_data_dir / swapi_efficiency_file_name),
            call(ancillary_data_dir / swapi_density_of_neutral_helium_lookup),
            call(ancillary_data_dir / swapi_hydrogen_inflow_filename),
            call(ancillary_data_dir / swapi_helium_inflow_filename),
            call(ancillary_data_dir / swapi_azimuthal_transmission_filename),
            call(ancillary_data_dir / swapi_central_effective_area_filename),
            call(ancillary_data_dir / swapi_passband_fit_coefficients_filename),
        ])

        mock_from_file_paths.assert_called_with(
            sentinel.swapi_l2_data,
            sentinel.efficiency_file,
            sentinel.neutral_helium_table,
            sentinel.hydrogen_vector,
            sentinel.helium_vector,
            sentinel.azimuthal_transmission,
            sentinel.central_effective_area,
            sentinel.passband_fit_coefficients,
            None,  # mag_l1d path is optional; absent when no MAG dep is provided
        )

        self.assertEqual(mock_from_file_paths.return_value, actual_swapi_l3_dependencies)

    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.InflowVector.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.DensityOfNeutralHeliumLookupTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SWAPIResponse.from_files')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_l2_swapi_data')
    def test_from_file_paths(self, mock_read_l2_swapi, mock_swapi_response_from_files,
                             mock_neutral_helium_from_file,
                             mock_efficiency_lookup_class,
                             mock_inflow_vector_from_file, mock_CDF):
        l2 = Path("imap_swapi_l2_sci_20100105_v010.cdf")
        efficiency = Path("imap_swapi_efficiency-lut_20100105_v010.cdf")
        helium = Path("imap_swapi_density-of-neutral-helium-lut_20100105_v010.cdf")
        h_vec = Path("imap_swapi_hydrogen-inflow-vector_20100105_v010.dat")
        he_vec = Path("imap_swapi_helium-inflow-vector_20100105_v010.dat")
        az = Path("imap_swapi_azimuthal-transmission_20100105_v010.csv")
        ea = Path("imap_swapi_central-effective-area_20100105_v010.csv")
        pb = Path("imap_swapi_passband-fit-coefficients_20100105_v010.csv")

        mock_read_l2_swapi.return_value = sentinel.swapi_l2_data
        mock_efficiency_lookup_class.return_value = sentinel.efficiency_lookup
        mock_neutral_helium_from_file.return_value = sentinel.neutral_helium_data
        mock_inflow_vector_from_file.side_effect = [sentinel.hydrogen_vector, sentinel.helium_vector]
        mock_swapi_response_from_files.return_value = sentinel.swapi_response

        expected = SwapiL3ADependencies(
            data=sentinel.swapi_l2_data,
            efficiency_calibration_table=sentinel.efficiency_lookup,
            density_of_neutral_helium_calibration_table=sentinel.neutral_helium_data,
            hydrogen_inflow_vector=sentinel.hydrogen_vector,
            helium_inflow_vector=sentinel.helium_vector,
            swapi_response=sentinel.swapi_response,
        )

        actual = SwapiL3ADependencies.from_file_paths(l2, efficiency, helium, h_vec, he_vec,
                                                     az, ea, pb)

        mock_CDF.assert_called_once_with(str(l2))
        mock_read_l2_swapi.assert_called_once_with(mock_CDF.return_value)
        mock_efficiency_lookup_class.assert_called_once_with(efficiency)
        mock_neutral_helium_from_file.assert_called_once_with(helium)
        mock_inflow_vector_from_file.assert_has_calls([call(h_vec), call(he_vec)])
        mock_swapi_response_from_files.assert_called_once_with(az, ea, pb)

        self.assertEqual(expected, actual)
