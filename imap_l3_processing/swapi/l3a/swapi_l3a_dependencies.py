from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, \
    PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, \
    ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR, \
    GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR, INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, \
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, \
    HELIUM_INFLOW_VECTOR_DESCRIPTOR, PROTON_SW_AZIMUTHAL_TRANSMISSION_DESCRIPTOR, \
    PROTON_SW_CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, PROTON_SW_PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    ProtonTemperatureAndDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import \
    InstrumentResponseLookupTableCollection


@dataclass
class SwapiL3ADependencies:
    data: SwapiL2Data
    proton_temperature_density_calibration_table: ProtonTemperatureAndDensityCalibrationTable
    alpha_temperature_density_calibration_table: AlphaTemperatureDensityCalibrationTable
    clock_angle_and_flow_deflection_calibration_table: ClockAngleCalibrationTable
    efficiency_calibration_table: EfficiencyCalibrationTable
    geometric_factor_calibration_table: GeometricFactorCalibrationTable
    instrument_response_calibration_table: InstrumentResponseLookupTableCollection
    density_of_neutral_helium_calibration_table: DensityOfNeutralHeliumLookupTable
    hydrogen_inflow_vector: InflowVector
    helium_inflow_vector: InflowVector
    swapi_response: SWAPIResponse

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        # @formatter:off
        science_dependency_file = dependencies.get_file_paths(source='swapi', descriptor=SWAPI_L2_DESCRIPTOR)
        proton_density_and_temperature_calibration_file = dependencies.get_file_paths(source='swapi', descriptor=PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
        alpha_density_and_temperature_calibration_file = dependencies.get_file_paths(source='swapi', descriptor=ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
        clock_and_deflection_file = dependencies.get_file_paths(source='swapi', descriptor=CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR)
        efficiency_calibration_table = dependencies.get_file_paths(source='swapi', descriptor=EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR)
        geometric_factor_calibration_table = dependencies.get_file_paths(source='swapi', descriptor=GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR)
        instrument_response_table = dependencies.get_file_paths(source='swapi', descriptor=INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR)
        neutral_helium_table = dependencies.get_file_paths(source='swapi', descriptor=DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR)
        hydrogen_vector_paths = dependencies.get_file_paths(source='swapi', descriptor=HYDROGEN_INFLOW_VECTOR_DESCRIPTOR)
        helium_vector_paths = dependencies.get_file_paths(source='swapi', descriptor=HELIUM_INFLOW_VECTOR_DESCRIPTOR)
        azimuthal_transmission_paths = dependencies.get_file_paths(source='swapi', descriptor=PROTON_SW_AZIMUTHAL_TRANSMISSION_DESCRIPTOR)
        central_effective_area_paths = dependencies.get_file_paths(source='swapi', descriptor=PROTON_SW_CENTRAL_EFFECTIVE_AREA_DESCRIPTOR)
        passband_fit_coefficients_paths = dependencies.get_file_paths(source='swapi', descriptor=PROTON_SW_PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR)
        # @formatter:on

        science_download_path = download(science_dependency_file[0])
        proton_density_and_temperature_calibration_file_path = download(
            proton_density_and_temperature_calibration_file[0])
        alpha_density_and_temperature_calibration_file_path = download(
            alpha_density_and_temperature_calibration_file[0])
        clock_and_deflection_file_path = download(clock_and_deflection_file[0])
        efficiency_calibration_table_path = download(efficiency_calibration_table[0])
        geometric_factor_calibration_table_path = download(geometric_factor_calibration_table[0])
        instrument_response_table_path = download(instrument_response_table[0])
        neutral_helium_table_path = download(neutral_helium_table[0])
        hydrogen_vector_path = download(hydrogen_vector_paths[0])
        helium_vector_path = download(helium_vector_paths[0])
        azimuthal_transmission_path = download(azimuthal_transmission_paths[0])
        central_effective_area_path = download(central_effective_area_paths[0])
        passband_fit_coefficients_path = download(passband_fit_coefficients_paths[0])

        return cls.from_file_paths(
            science_download_path,
            proton_density_and_temperature_calibration_file_path,
            alpha_density_and_temperature_calibration_file_path,
            clock_and_deflection_file_path,
            efficiency_calibration_table_path,
            geometric_factor_calibration_table_path,
            instrument_response_table_path,
            neutral_helium_table_path,
            hydrogen_vector_path,
            helium_vector_path,
            azimuthal_transmission_path,
            central_effective_area_path,
            passband_fit_coefficients_path,
        )

    @classmethod
    def from_file_paths(cls, science_dependency_path: Path, proton_density_and_temperature_calibration_path: Path,
                        alpha_density_and_temperature_calibration_path: Path, clock_and_deflection_file_path: Path,
                        efficiency_calibration_path: Path, geometric_factor_calibration_path: Path,
                        instrument_response_path: Path, neutral_helium_path: Path, hydrogen_inflow_vector_path: Path,
                        helium_inflow_vector_path: Path, azimuthal_transmission_path: Path,
                        central_effective_area_path: Path, passband_fit_coefficients_path: Path):
        swapi_l2_data = read_l2_swapi_data(CDF(str(science_dependency_path)))
        proton_density_temp_lookup = ProtonTemperatureAndDensityCalibrationTable.from_file(
            proton_density_and_temperature_calibration_path)
        alpha_density_temp_lookup = AlphaTemperatureDensityCalibrationTable.from_file(
            alpha_density_and_temperature_calibration_path)
        clock_deflection_lookup = ClockAngleCalibrationTable.from_file(clock_and_deflection_file_path)
        efficiency_lookup = EfficiencyCalibrationTable(efficiency_calibration_path)
        geometric_factor_calibration_lookup = GeometricFactorCalibrationTable.from_file(
            geometric_factor_calibration_path)
        instrument_response_lookup = InstrumentResponseLookupTableCollection.from_file(instrument_response_path)
        neutral_helium_lookup = DensityOfNeutralHeliumLookupTable.from_file(neutral_helium_path)

        hydrogen_inflow_vector = InflowVector.from_file(hydrogen_inflow_vector_path)
        helium_inflow_vector = InflowVector.from_file(helium_inflow_vector_path)
        swapi_response = SWAPIResponse.from_files(
            azimuthal_transmission_path, central_effective_area_path, passband_fit_coefficients_path)

        return cls(swapi_l2_data,
                   proton_density_temp_lookup,
                   alpha_density_temp_lookup,
                   clock_deflection_lookup,
                   efficiency_lookup,
                   geometric_factor_calibration_lookup,
                   instrument_response_lookup,
                   neutral_helium_lookup,
                   hydrogen_inflow_vector,
                   helium_inflow_vector,
                   swapi_response)
