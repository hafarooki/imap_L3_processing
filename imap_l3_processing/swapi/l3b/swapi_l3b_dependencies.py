from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.swapi.descriptors import (
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR,
    SWAPI_L2_DESCRIPTOR,
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR,
    CENTRAL_EFFECTIVE_AREA_DESCRIPTOR,
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)


@dataclass
class SwapiL3BDependencies:
    data: SwapiL2Data
    swapi_response: SWAPIResponse
    efficiency_calibration_table: EfficiencyCalibrationTable

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        science_dependency_file = dependencies.get_file_paths(
            source="swapi", descriptor=SWAPI_L2_DESCRIPTOR
        )
        efficiency_table_lookup_file = dependencies.get_file_paths(
            source="swapi", descriptor=EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR
        )
        azimuthal_transmission_paths = dependencies.get_file_paths(
            source="swapi", descriptor=AZIMUTHAL_TRANSMISSION_DESCRIPTOR
        )
        central_effective_area_paths = dependencies.get_file_paths(
            source="swapi", descriptor=CENTRAL_EFFECTIVE_AREA_DESCRIPTOR
        )
        passband_fit_coefficients_paths = dependencies.get_file_paths(
            source="swapi", descriptor=PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
        )

        download(science_dependency_file[0])
        download(efficiency_table_lookup_file[0])
        download(azimuthal_transmission_paths[0])
        download(central_effective_area_paths[0])
        download(passband_fit_coefficients_paths[0])

        return cls.from_file_paths(
            science_dependency_file[0],
            efficiency_table_lookup_file[0],
            azimuthal_transmission_paths[0],
            central_effective_area_paths[0],
            passband_fit_coefficients_paths[0],
        )

    @classmethod
    def from_file_paths(
        cls,
        science_dependency_path: Path,
        efficiency_calibration_table_path: Path,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
    ):
        swapi_l2_data = read_l2_swapi_data(CDF(str(science_dependency_path)))
        efficiency_calibration_table = EfficiencyCalibrationTable(
            efficiency_calibration_table_path
        )
        swapi_response = SWAPIResponse.from_files(
            azimuthal_transmission_path,
            central_effective_area_path,
            passband_fit_coefficients_path,
        )

        return cls(swapi_l2_data, swapi_response, efficiency_calibration_table)
