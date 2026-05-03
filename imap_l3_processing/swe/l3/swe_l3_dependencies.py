from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.models import MagData
from imap_l3_processing.swe.l3.models import SweL2Data, SweConfiguration, SwapiL3aProtonData, SweL1bData
from imap_l3_processing.swe.l3.utils import read_l2_swe_data, read_l3a_swapi_proton_data, read_swe_config, \
    read_l1b_swe_data
from imap_l3_processing.utils import read_mag_data

MAG_DESPUN_L1D_DESCRIPTOR = "norm-dsrf"
SWAPI_L3A_PROTON_DESCRIPTOR = "proton-sw"
SWE_CONFIG_DESCRIPTOR = "config"


@dataclass
class SweL3Dependencies:
    swe_l2_data: SweL2Data
    swe_l1b_data: SweL1bData
    mag_data: MagData
    swapi_l3a_proton_data: SwapiL3aProtonData
    configuration: SweConfiguration

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> SweL3Dependencies:
        science_files = dependencies.processing_input
        swe_config_dependency = dependencies.get_file_paths(source='swe', descriptor=SWE_CONFIG_DESCRIPTOR)[0]

        try:
            swe_l2_dependency = next(
                d.imap_file_paths[0] for d in science_files if d.source == "swe" and d.data_type == "l2")
        except StopIteration:
            raise ValueError(f"Missing SWE l2 dependency.")
        try:
            swe_l1b_dependency = next(
                d.imap_file_paths[0] for d in science_files if d.source == "swe" and d.data_type == "l1b")
        except StopIteration:
            raise ValueError(f"Missing SWE l1b dependency.")
        mag_dependency = next(
            (d.imap_file_paths[0] for d in science_files if d.source == "mag"
            and d.descriptor == MAG_DESPUN_L1D_DESCRIPTOR
            and d.data_type == "l2"), None)
        if mag_dependency is None:
            try:
                mag_dependency = next(
                    d.imap_file_paths[0] for d in science_files if d.source == "mag"
                    and d.descriptor == MAG_DESPUN_L1D_DESCRIPTOR
                    and d.data_type == "l1d"
                )
            except StopIteration:
                raise ValueError(f"Missing MAG {MAG_DESPUN_L1D_DESCRIPTOR} dependency.")
        try:
            swapi_dependency = next(
                d.imap_file_paths[0] for d in science_files if d.source == "swapi"
                and d.descriptor == SWAPI_L3A_PROTON_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing SWAPI {SWAPI_L3A_PROTON_DESCRIPTOR} dependency.")

        swe_l2_file = download(swe_l2_dependency.construct_path())
        swe_l1b_file = download(swe_l1b_dependency.construct_path())
        mag_file = download(mag_dependency.construct_path())
        swapi_file = download(swapi_dependency.construct_path())
        swe_config = download(swe_config_dependency)

        return cls.from_file_paths(swe_l2_file, swe_l1b_file, mag_file, swapi_file, swe_config)

    @classmethod
    def from_file_paths(cls, swe_l2_file_path: Path, swe_l1b_file_path: Path, mag_file_path: Path,
                        swapi_file_path: Path,
                        configuration_file_path: Path) -> SweL3Dependencies:
        mag_l1d_data = read_mag_data(mag_file_path)
        swe_l1b_data = read_l1b_swe_data(swe_l1b_file_path)
        swe_l2_data = read_l2_swe_data(swe_l2_file_path)
        swapi_l3a_proton_data = read_l3a_swapi_proton_data(swapi_file_path)
        configuration = read_swe_config(configuration_file_path)

        return cls(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, configuration)
