import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_utils import calculate_energy_deltas
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy_grid"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "surv_prob"
HEALPIX_INDEX_VAR_NAME = "healpix_index"
ENERGY_LABEL_VAR_NAME = "energy_label"
PIXEL_INDEX_LABEL_VAR_NAME = "healpix_index_label"
SPIN_AXIS_LATITUDE_VAR_NAME = "spin_axis_latitude"
SPIN_AXIS_LONGITUDE_VAR_NAME = "spin_axis_longitude"
PROGRAM_VERSION_VAR_NAME = "program_version"
SPACECRAFT_RADIUS_VAR_NAME = "spacecraft_radius"
SPACECRAFT_LONGITUDE_VAR_NAME = "spacecraft_longitude"
SPACECRAFT_LATITUDE_VAR_NAME = "spacecraft_latitude"
SPACECRAFT_VELOCITY_X_VAR_NAME = "spacecraft_velocity_x"
SPACECRAFT_VELOCITY_Y_VAR_NAME = "spacecraft_velocity_y"
SPACECRAFT_VELOCITY_Z_VAR_NAME = "spacecraft_velocity_z"
ELONGATION_EXCLUDED_VAR_NAME = "elongation_excluded"
PIXEL_LATITUDE_VAR_NAME = "pixel_latitude"
PIXEL_LONGITUDE_VAR_NAME = "pixel_longitude"
GLOWS_FLAGS_VAR_NAME = "glows_flags"


@dataclass
class GlowsL3EUltraData(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    healpix_index: np.ndarray
    probability_of_survival: np.ndarray
    spin_axis_lat: np.ndarray
    spin_axis_lon: np.ndarray
    program_version: np.ndarray
    spacecraft_radius: np.ndarray
    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    spacecraft_velocity_x: np.ndarray
    spacecraft_velocity_y: np.ndarray
    spacecraft_velocity_z: np.ndarray
    elongation_excluded: np.ndarray
    pixel_latitude: np.ndarray
    pixel_longitude: np.ndarray
    glows_flags: np.ndarray

    @classmethod
    def convert_dat_to_glows_l3e_ul_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: datetime,
                                            epoch_delta: timedelta,
                                            args: GlowsL3eCallArguments):
        with open(file_path) as input_data:
            lines = input_data.readlines()

            energy_line = [line for line in lines if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

            code_version_line = [line for line in lines if line.startswith("# code version")]
            code_version = code_version_line[0].split(',')[0][14:].strip()

        data_table = np.loadtxt(file_path, skiprows=200, dtype=np.float64)

        healpix_indexes = np.arange(0, 3072)

        existing_healpix = data_table[:, 0]
        probability_of_survival = data_table[:, 3:]
        pixel_latitude = data_table[:, 1]
        pixel_longitude = data_table[:, 2]


        probability_of_survival_to_return = np.full((len(energies), len(healpix_indexes)), np.nan, dtype=float)
        pixel_latitude_to_return = np.full(len(healpix_indexes), np.nan, dtype=float)
        pixel_longitude_to_return = np.full(len(healpix_indexes), np.nan, dtype=float)

        for healpix, prob_sur, pixel_lat, pixel_lon in zip(existing_healpix, probability_of_survival, pixel_latitude, pixel_longitude):
            probability_of_survival_to_return[:, int(healpix)] = prob_sur
            pixel_latitude_to_return[int(healpix)] = pixel_lat
            pixel_longitude_to_return[int(healpix)] = pixel_lon

        transposed_prob_sur = np.array([probability_of_survival_to_return])

        energy_delta_plus, energy_delta_minus = calculate_energy_deltas(energies)

        return cls(
            input_metadata,
            epoch=np.array([epoch]),
            epoch_delta=np.array([epoch_delta.total_seconds() * ONE_SECOND_IN_NANOSECONDS]),
            energy=energies,
            energy_delta_plus=energy_delta_plus,
            energy_delta_minus=energy_delta_minus,
            healpix_index=healpix_indexes,
            probability_of_survival=transposed_prob_sur,
            spin_axis_lat=np.array([args.spacecraft_info.spin_axis_latitude]),
            spin_axis_lon=np.array([args.spacecraft_info.spin_axis_longitude]),
            program_version=np.array([code_version]),
            spacecraft_radius=np.array([args.spacecraft_info.spacecraft_radius]),
            spacecraft_longitude=np.array([args.spacecraft_info.spacecraft_longitude]),
            spacecraft_latitude=np.array([args.spacecraft_info.spacecraft_latitude]),
            spacecraft_velocity_x=np.array([args.spacecraft_info.spacecraft_velocity_x]),
            spacecraft_velocity_y=np.array([args.spacecraft_info.spacecraft_velocity_y]),
            spacecraft_velocity_z=np.array([args.spacecraft_info.spacecraft_velocity_z]),
            elongation_excluded=np.array([args.elongation]),
            pixel_latitude=np.array([pixel_latitude_to_return]),
            pixel_longitude=np.array([pixel_longitude_to_return]),
            glows_flags=np.array([0], dtype=np.uint16),
        )

    def to_data_product_variables(self) -> list[DataProductVariable]:
        energy_labels = [f"{i:.2f}" for i in self.energy]
        pixel_labels = [f"{i:.0f}" for i in self.healpix_index]
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(HEALPIX_INDEX_VAR_NAME, self.healpix_index),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, energy_labels),
            DataProductVariable(PIXEL_INDEX_LABEL_VAR_NAME, pixel_labels),
            DataProductVariable(SPIN_AXIS_LATITUDE_VAR_NAME, self.spin_axis_lat),
            DataProductVariable(SPIN_AXIS_LONGITUDE_VAR_NAME, self.spin_axis_lon),
            DataProductVariable(PROGRAM_VERSION_VAR_NAME, self.program_version),
            DataProductVariable(SPACECRAFT_RADIUS_VAR_NAME, self.spacecraft_radius),
            DataProductVariable(SPACECRAFT_LATITUDE_VAR_NAME, self.spacecraft_latitude),
            DataProductVariable(SPACECRAFT_LONGITUDE_VAR_NAME, self.spacecraft_longitude),
            DataProductVariable(SPACECRAFT_VELOCITY_X_VAR_NAME, self.spacecraft_velocity_x),
            DataProductVariable(SPACECRAFT_VELOCITY_Y_VAR_NAME, self.spacecraft_velocity_y),
            DataProductVariable(SPACECRAFT_VELOCITY_Z_VAR_NAME, self.spacecraft_velocity_z),
            DataProductVariable(ELONGATION_EXCLUDED_VAR_NAME, self.elongation_excluded),
            DataProductVariable(PIXEL_LATITUDE_VAR_NAME, self.pixel_latitude),
            DataProductVariable(PIXEL_LONGITUDE_VAR_NAME, self.pixel_longitude),
            DataProductVariable(GLOWS_FLAGS_VAR_NAME, self.glows_flags),
        ]
