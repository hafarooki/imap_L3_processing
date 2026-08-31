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
SPIN_ANGLE_VAR_NAME = "spin_angle"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "surv_prob"
ENERGY_LABEL_VAR_NAME = "energy_label"
SPIN_ANGLE_LABEL_VAR_NAME = "spin_angle_label"
ELONGATION_VAR_NAME = "elongation"
SPIN_AXIS_LATITUDE_VAR_NAME = "spin_axis_latitude"
SPIN_AXIS_LONGITUDE_VAR_NAME = "spin_axis_longitude"
PROGRAM_VERSION_VAR_NAME = "program_version"
SPACECRAFT_RADIUS_VAR_NAME = "spacecraft_radius"
SPACECRAFT_LONGITUDE_VAR_NAME = "spacecraft_longitude"
SPACECRAFT_LATITUDE_VAR_NAME = "spacecraft_latitude"
SPACECRAFT_VELOCITY_X_VAR_NAME = "spacecraft_velocity_x"
SPACECRAFT_VELOCITY_Y_VAR_NAME = "spacecraft_velocity_y"
SPACECRAFT_VELOCITY_Z_VAR_NAME = "spacecraft_velocity_z"
GLOWS_FLAGS_VAR_NAME = "glows_flags"


@dataclass
class GlowsL3ELoData(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray
    elongation: np.ndarray
    spin_axis_lat: np.ndarray
    spin_axis_lon: np.ndarray
    program_version: np.ndarray
    spacecraft_radius: np.ndarray
    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    spacecraft_velocity_x: np.ndarray
    spacecraft_velocity_y: np.ndarray
    spacecraft_velocity_z: np.ndarray
    glows_flags: np.ndarray

    @classmethod
    def convert_dat_to_glows_l3e_lo_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: datetime, epoch_delta: timedelta, elongation: int,
                                            args: GlowsL3eCallArguments):
        with open(file_path) as input_data:
            lines = input_data.readlines()

            energy_line = [line for line in lines if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

            code_version_line = [line for line in lines if line.startswith("# code version")]
            code_version = code_version_line[0].split(',')[0][14:].strip()

        spin_angle_and_survival_probabilities = np.loadtxt(file_path, skiprows=200, dtype=np.float64)
        spin_angles = spin_angle_and_survival_probabilities[:, 0]
        survival_probabilities = np.array([spin_angle_and_survival_probabilities[:, 1:].T])

        energy_delta_plus, energy_delta_minus = calculate_energy_deltas(energies)

        return cls(
            input_metadata=input_metadata,
            epoch=np.array([epoch]),
            epoch_delta=np.array([epoch_delta.total_seconds() * ONE_SECOND_IN_NANOSECONDS]),
            energy=energies,
            energy_delta_plus=energy_delta_plus,
            energy_delta_minus=energy_delta_minus,
            spin_angle=spin_angles,
            probability_of_survival=survival_probabilities,
            elongation=np.array([elongation]),
            spin_axis_lat=np.array([args.spacecraft_info.spin_axis_latitude]),
            spin_axis_lon=np.array([args.spacecraft_info.spin_axis_longitude]),
            program_version=np.array([code_version]),
            spacecraft_radius=np.array([args.spacecraft_info.spacecraft_radius]),
            spacecraft_longitude=np.array([args.spacecraft_info.spacecraft_longitude]),
            spacecraft_latitude=np.array([args.spacecraft_info.spacecraft_latitude]),
            spacecraft_velocity_x=np.array([args.spacecraft_info.spacecraft_velocity_x]),
            spacecraft_velocity_y=np.array([args.spacecraft_info.spacecraft_velocity_y]),
            spacecraft_velocity_z=np.array([args.spacecraft_info.spacecraft_velocity_z]),
            glows_flags=np.array([0], dtype=np.uint16),
       )

    def to_data_product_variables(self) -> list[DataProductVariable]:
        spin_angle_labels = [f"{i:.0f}" for i in self.spin_angle]
        energy_labels = [f"{i:.2f}" for i in self.energy]

        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, energy_labels),
            DataProductVariable(SPIN_ANGLE_LABEL_VAR_NAME, spin_angle_labels),
            DataProductVariable(ELONGATION_VAR_NAME, self.elongation),
            DataProductVariable(SPIN_AXIS_LATITUDE_VAR_NAME, self.spin_axis_lat),
            DataProductVariable(SPIN_AXIS_LONGITUDE_VAR_NAME, self.spin_axis_lon),
            DataProductVariable(PROGRAM_VERSION_VAR_NAME, self.program_version),
            DataProductVariable(SPACECRAFT_RADIUS_VAR_NAME, self.spacecraft_radius),
            DataProductVariable(SPACECRAFT_LATITUDE_VAR_NAME, self.spacecraft_latitude),
            DataProductVariable(SPACECRAFT_LONGITUDE_VAR_NAME, self.spacecraft_longitude),
            DataProductVariable(SPACECRAFT_VELOCITY_X_VAR_NAME, self.spacecraft_velocity_x),
            DataProductVariable(SPACECRAFT_VELOCITY_Y_VAR_NAME, self.spacecraft_velocity_y),
            DataProductVariable(SPACECRAFT_VELOCITY_Z_VAR_NAME, self.spacecraft_velocity_z),
            DataProductVariable(GLOWS_FLAGS_VAR_NAME, self.glows_flags),
        ]
