import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imap_l3_processing.constants import HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS
from imap_l3_processing.swapi.l3a.science.speed_calculation import calculate_sw_speed


@dataclass
class InstrumentResponseLookupTable:
    energy: np.ndarray
    elevation: np.ndarray
    azimuth: np.ndarray
    d_energy: np.ndarray
    d_elevation: np.ndarray
    d_azimuth: np.ndarray
    response: np.ndarray
    integral_factor: np.ndarray = None

    def __post_init__(self):
        elevation_radians = np.deg2rad(self.elevation)

        denominator = (self.d_energy * np.cos(
            elevation_radians) * self.d_elevation * self.d_azimuth).sum()

        speed = calculate_sw_speed(HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS,
                                   self.energy)
        self.integral_factor = self.response * \
                               speed ** 4 * \
                               self.d_energy * np.cos(np.deg2rad(self.elevation)) * \
                               self.d_azimuth * self.d_elevation / denominator


class InstrumentResponseLookupTableCollection:
    def __init__(self, lookup_tables: dict[int, InstrumentResponseLookupTable]):
        self.lookup_tables = lookup_tables

    def get_table_for_energy_bin(self, index: int) -> InstrumentResponseLookupTable:
        return self.lookup_tables[index]

    @classmethod
    def from_file(cls, zip_path: Path):
        files = {}
        with zipfile.ZipFile(zip_path) as zip_file:
            for file_name in zip_file.namelist():
                if match := re.search(r'^response_ESA(\d*).dat$', Path(file_name).name):
                    number = int(match[1])
                    transposed = np.loadtxt(zip_file.open(file_name)).T
                    files[int(number)] = InstrumentResponseLookupTable(*transposed)
        return cls(files)
