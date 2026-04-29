from __future__ import annotations

from dataclasses import dataclass

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.hit.l3.models import HitL2Data
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from imap_l3_processing.models import MagData
from imap_l3_processing.utils import read_mag_data

HIT_L2_DESCRIPTOR = "macropixel-intensity"
MAG_L1D_DESCRIPTOR = "norm-dsrf"


@dataclass
class HITL3SectoredDependencies:
    data: HitL2Data
    mag_l1d_data: MagData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HITL3SectoredDependencies:
        hit_data_dependency = dependencies.get_file_paths(source="hit", descriptor=HIT_L2_DESCRIPTOR)
        mag_dependency = dependencies.get_file_paths(source="mag", descriptor=MAG_L1D_DESCRIPTOR)

        hit_data_path = download(hit_data_dependency[0])
        mag_data_path = download(mag_dependency[0])

        mag_data = read_mag_data(mag_data_path)
        hit_data = read_l2_hit_data(hit_data_path)

        return HITL3SectoredDependencies(data=hit_data, mag_l1d_data=mag_data)
