from dataclasses import dataclass

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.models import CodiceHiL2SectoredIntensitiesData
from imap_l3_processing.models import MagData
from imap_l3_processing.utils import read_mag_data


@dataclass
class CodicePitchAngleDependencies:
    mag_l1d_data: MagData
    codice_sectored_intensities_data: CodiceHiL2SectoredIntensitiesData

    @classmethod
    def fetch_dependencies(cls, input_collection: ProcessingInputCollection):
        codice_file_paths = input_collection.get_file_paths("codice", "hi-sectored")
        mag_file_paths = input_collection.get_file_paths("mag", "norm-dsrf")

        for download_location_file_path in [*codice_file_paths, *mag_file_paths]:
            download(download_location_file_path)

        return cls.from_file_paths(mag_file_paths[0], codice_file_paths[0])

    @classmethod
    def from_file_paths(cls, mag_file_path, codice_l2_sectored_intensities_path):
        mag_data = read_mag_data(mag_file_path)
        sectored_intensities = CodiceHiL2SectoredIntensitiesData.read_from_cdf(codice_l2_sectored_intensities_path)

        return cls(mag_data, sectored_intensities)
