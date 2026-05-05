from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, \
    GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR, INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, \
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, \
    HELIUM_INFLOW_VECTOR_DESCRIPTOR, AZIMUTHAL_TRANSMISSION_DESCRIPTOR, \
    CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR, \
    MAG_RTN_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data, read_mag_rtn_data
from imap_l3_processing.utils import select_mag_path
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import \
    InstrumentResponseLookupTableCollection


@dataclass
class SwapiL3ADependencies:
    data: SwapiL2Data
    efficiency_calibration_table: EfficiencyCalibrationTable
    geometric_factor_calibration_table: GeometricFactorCalibrationTable
    instrument_response_calibration_table: InstrumentResponseLookupTableCollection
    density_of_neutral_helium_calibration_table: DensityOfNeutralHeliumLookupTable
    hydrogen_inflow_vector: InflowVector
    helium_inflow_vector: InflowVector
    swapi_response: SwapiResponse
    mag_data: Optional[MagData] = None
    mag_is_preliminary: bool = False

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        # @formatter:off
        science_dependency_file = dependencies.get_file_paths(source='swapi', descriptor=SWAPI_L2_DESCRIPTOR)
        efficiency_calibration_table = dependencies.get_file_paths(source='swapi', descriptor=EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR)
        geometric_factor_calibration_table = dependencies.get_file_paths(source='swapi', descriptor=GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR)
        instrument_response_table = dependencies.get_file_paths(source='swapi', descriptor=INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR)
        neutral_helium_table = dependencies.get_file_paths(source='swapi', descriptor=DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR)
        hydrogen_vector_paths = dependencies.get_file_paths(source='swapi', descriptor=HYDROGEN_INFLOW_VECTOR_DESCRIPTOR)
        helium_vector_paths = dependencies.get_file_paths(source='swapi', descriptor=HELIUM_INFLOW_VECTOR_DESCRIPTOR)
        azimuthal_transmission_paths = dependencies.get_file_paths(source='swapi', descriptor=AZIMUTHAL_TRANSMISSION_DESCRIPTOR)
        central_effective_area_paths = dependencies.get_file_paths(source='swapi', descriptor=CENTRAL_EFFECTIVE_AREA_DESCRIPTOR)
        passband_fit_coefficients_paths = dependencies.get_file_paths(source='swapi', descriptor=PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR)
        # @formatter:on

        mag_path, mag_level = select_mag_path(dependencies, MAG_RTN_DESCRIPTOR)
        mag_is_preliminary = mag_level == "l1d"

        return cls.from_file_paths(
            download(science_dependency_file[0]),
            download(efficiency_calibration_table[0]),
            download(geometric_factor_calibration_table[0]),
            download(instrument_response_table[0]),
            download(neutral_helium_table[0]),
            download(hydrogen_vector_paths[0]),
            download(helium_vector_paths[0]),
            download(azimuthal_transmission_paths[0]),
            download(central_effective_area_paths[0]),
            download(passband_fit_coefficients_paths[0]),
            mag_path,
            mag_is_preliminary,
        )

    @classmethod
    def from_file_paths(cls, science_dependency_path: Path,
                        efficiency_calibration_path: Path, geometric_factor_calibration_path: Path,
                        instrument_response_path: Path, neutral_helium_path: Path, hydrogen_inflow_vector_path: Path,
                        helium_inflow_vector_path: Path, azimuthal_transmission_path: Path,
                        central_effective_area_path: Path, passband_fit_coefficients_path: Path,
                        mag_path: Optional[Path] = None,
                        mag_is_preliminary: bool = False):
        return cls(
            data=read_l2_swapi_data(CDF(str(science_dependency_path))),
            efficiency_calibration_table=EfficiencyCalibrationTable(efficiency_calibration_path),
            geometric_factor_calibration_table=GeometricFactorCalibrationTable.from_file(
                geometric_factor_calibration_path),
            instrument_response_calibration_table=InstrumentResponseLookupTableCollection.from_file(
                instrument_response_path),
            density_of_neutral_helium_calibration_table=DensityOfNeutralHeliumLookupTable.from_file(
                neutral_helium_path),
            hydrogen_inflow_vector=InflowVector.from_file(hydrogen_inflow_vector_path),
            helium_inflow_vector=InflowVector.from_file(helium_inflow_vector_path),
            swapi_response=SwapiResponse.from_files(
                azimuthal_transmission_path, central_effective_area_path, passband_fit_coefficients_path),
            mag_data=read_mag_rtn_data(mag_path) if mag_path is not None else None,
            mag_is_preliminary=mag_is_preliminary,
        )
