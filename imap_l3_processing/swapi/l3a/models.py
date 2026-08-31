from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
from spacepy.pycdf import CDF
from uncertainties.unumpy import nominal_values, std_devs

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import (
    THIRTY_SECONDS_IN_NANOSECONDS,
    FIVE_MINUTES_IN_NANOSECONDS,
)
from imap_l3_processing.models import DataProduct, DataProductVariable
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_uncert"
PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME = "proton_sw_speed_sun"
PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_sun_uncert"
PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "proton_sw_temperature"
PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_temperature_uncert"
PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "proton_sw_density"
PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_density_uncert"

PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME = "proton_sw_velocity_rtn_sun"
PROTON_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME = "proton_sw_velocity_rtn"
PROTON_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME = (
    "proton_sw_velocity_rtn_covariance"
)
PROTON_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME = (
    "proton_sw_velocity_rtn_uncert"
)

ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME = "alpha_sw_speed"
ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_uncert"
ALPHA_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME = "alpha_sw_speed_sun"
ALPHA_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_sun_uncert"
ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "alpha_sw_density"
ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_density_uncert"
ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_temperature"
ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_temperature_uncert"
ALPHA_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME = "alpha_sw_velocity_rtn_sun"
ALPHA_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME = "alpha_sw_velocity_rtn"
ALPHA_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME = (
    "alpha_sw_velocity_rtn_covariance"
)
ALPHA_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME = (
    "alpha_sw_velocity_rtn_uncert"
)

PUI_COOLING_INDEX_CDF_VAR_NAME = "pui_cooling_index"
PUI_IONIZATION_RATE_CDF_VAR_NAME = "pui_ionization_rate"
PUI_CUTOFF_SPEED_CDF_VAR_NAME = "pui_cutoff_speed"
PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME = "pui_background_count_rate"
PUI_DENSITY_CDF_VAR_NAME = "pui_density"
PUI_TEMPERATURE_CDF_VAR_NAME = "pui_temperature"
PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME = "pui_cooling_index_uncert"
PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME = "pui_ionization_rate_uncert"
PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME = "pui_cutoff_speed_uncert"
PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME = "pui_background_count_rate_uncert"
PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "pui_density_uncert"
PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "pui_temperature_uncert"

SWAPI_QUALITY_FLAGS_CDF_VAR_NAME = "swp_flags"

VELOCITY_RTN_LABEL_CDF_VAR_NAME = "velocity_rtn_label"
ALPHA_VELOCITY_RTN_LABEL_CDF_VAR_NAME = "alpha_sw_velocity_rtn_label"
ALPHA_VELOCITY_RTN_SUN_LABEL_CDF_VAR_NAME = "alpha_sw_velocity_rtn_sun_label"

PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_LABEL_CDF_VAR_NAME = "proton_sw_velocity_rtn_sun_label"
PROTON_SOLAR_WIND_VELOCITY_RTN_LABEL_CDF_VAR_NAME = "proton_sw_velocity_rtn_label"


@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray
    proton_sw_speed: np.ndarray
    proton_sw_speed_uncert: np.ndarray
    proton_sw_speed_sun: np.ndarray
    proton_sw_speed_sun_uncert: np.ndarray
    proton_sw_temperature: np.ndarray
    proton_sw_temperature_uncert: np.ndarray
    proton_sw_density: np.ndarray
    proton_sw_density_uncert: np.ndarray
    proton_sw_velocity_rtn_sun: np.ndarray  # shape (N, 3), km/s, inertial RTN
    proton_sw_velocity_rtn: np.ndarray  # shape (N, 3), km/s, RTN in SC rest frame
    proton_sw_velocity_rtn_covariance: np.ndarray  # shape (N, 3, 3), km²/s²
    quality_flags: np.ndarray[SwapiL3Flags]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(
                PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, self.proton_sw_speed
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME,
                self.proton_sw_speed_uncert,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME,
                self.proton_sw_speed_sun,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME,
                self.proton_sw_speed_sun_uncert,
            ),
            DataProductVariable(
                EPOCH_DELTA_CDF_VAR_NAME,
                np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS),
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, self.proton_sw_temperature
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                self.proton_sw_temperature_uncert,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME, self.proton_sw_density
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
                self.proton_sw_density_uncert,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME,
                self.proton_sw_velocity_rtn_sun,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME,
                self.proton_sw_velocity_rtn,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME,
                self.proton_sw_velocity_rtn_covariance,
            ),
            DataProductVariable(
                PROTON_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME,
                np.sqrt(
                    np.diagonal(
                        self.proton_sw_velocity_rtn_covariance, axis1=1, axis2=2
                    )
                ),
            ),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags),
            DataProductVariable(VELOCITY_RTN_LABEL_CDF_VAR_NAME, value=["R", "T", "N"]),
            DataProductVariable(PROTON_SOLAR_WIND_VELOCITY_RTN_LABEL_CDF_VAR_NAME, value=["Vp SC R", "Vp SC T", "Vp SC N"]),
            DataProductVariable(PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_LABEL_CDF_VAR_NAME, value=["Vp Sun R", "Vp Sun T", "Vp Sun N"]),
        ]


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    epoch: np.ndarray[datetime]
    alpha_sw_speed: np.ndarray
    alpha_sw_speed_uncert: np.ndarray
    alpha_sw_speed_sun: np.ndarray
    alpha_sw_speed_sun_uncert: np.ndarray
    alpha_sw_density: np.ndarray
    alpha_sw_density_uncert: np.ndarray
    alpha_sw_temperature: np.ndarray
    alpha_sw_temperature_uncert: np.ndarray
    alpha_sw_velocity_rtn_sun: np.ndarray  # shape (N, 3), km/s, inertial RTN
    alpha_sw_velocity_rtn: np.ndarray  # shape (N, 3), km/s, RTN in SC rest frame
    alpha_sw_velocity_rtn_covariance: np.ndarray  # shape (N, 3, 3), km²/s²
    quality_flags: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(
                ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, self.alpha_sw_speed
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME,
                self.alpha_sw_speed_uncert,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME,
                self.alpha_sw_speed_sun,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME,
                self.alpha_sw_speed_sun_uncert,
            ),
            DataProductVariable(
                EPOCH_DELTA_CDF_VAR_NAME,
                np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS),
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME, self.alpha_sw_density
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
                self.alpha_sw_density_uncert,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, self.alpha_sw_temperature
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                self.alpha_sw_temperature_uncert,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME,
                self.alpha_sw_velocity_rtn_sun,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME,
                self.alpha_sw_velocity_rtn,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME,
                self.alpha_sw_velocity_rtn_covariance,
            ),
            DataProductVariable(
                ALPHA_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME,
                np.sqrt(
                    np.diagonal(
                        self.alpha_sw_velocity_rtn_covariance, axis1=1, axis2=2
                    )
                ),
            ),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags),
            DataProductVariable(VELOCITY_RTN_LABEL_CDF_VAR_NAME, value=["R", "T", "N"]),
            DataProductVariable(ALPHA_VELOCITY_RTN_LABEL_CDF_VAR_NAME, value=["Va SC R", "Va SC T", "Va SC N"]),
            DataProductVariable(ALPHA_VELOCITY_RTN_SUN_LABEL_CDF_VAR_NAME, value=["Va Sun R", "Va Sun T", "Va Sun N"]),
        ]


@dataclass
class SwapiL3PickupIonData(DataProduct):
    epoch: np.ndarray[float]
    cooling_index: np.ndarray[float]
    ionization_rate: np.ndarray[float]
    cutoff_speed: np.ndarray[float]
    background_rate: np.ndarray[float]
    density: np.ndarray[float]
    temperature: np.ndarray[float]
    quality_flags: np.ndarray[SwapiL3Flags]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(
                EPOCH_DELTA_CDF_VAR_NAME,
                np.full_like(self.epoch, FIVE_MINUTES_IN_NANOSECONDS),
            ),
            DataProductVariable(
                PUI_COOLING_INDEX_CDF_VAR_NAME, nominal_values(self.cooling_index)
            ),
            DataProductVariable(
                PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cooling_index)
            ),
            DataProductVariable(
                PUI_IONIZATION_RATE_CDF_VAR_NAME, nominal_values(self.ionization_rate)
            ),
            DataProductVariable(
                PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME,
                std_devs(self.ionization_rate),
            ),
            DataProductVariable(
                PUI_CUTOFF_SPEED_CDF_VAR_NAME, nominal_values(self.cutoff_speed)
            ),
            DataProductVariable(
                PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cutoff_speed)
            ),
            DataProductVariable(
                PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME,
                nominal_values(self.background_rate),
            ),
            DataProductVariable(
                PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME,
                std_devs(self.background_rate),
            ),
            DataProductVariable(PUI_DENSITY_CDF_VAR_NAME, nominal_values(self.density)),
            DataProductVariable(
                PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.density)
            ),
            DataProductVariable(
                PUI_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.temperature)
            ),
            DataProductVariable(
                PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.temperature)
            ),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags),
        ]


@dataclass
class SwapiL2Data:
    sci_start_time: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    coincidence_count_rate_uncertainty: np.ndarray[float]


@dataclass
class SwapiL3aProtonDataFromCDF:
    l2_parent_file_name: str
    velocity_rtn: np.ndarray
    velocity_rtn_covariance: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    quality_flags: np.ndarray

    @classmethod
    def from_file(cls, data_file:Path)-> Self:
        with CDF(str(data_file)) as cdf:
            l2_parent_file_name = next(parent for parent in cdf.attrs["Parents"] if parent.startswith("imap_swapi_l2_"))
            velocity_rtn = read_numeric_variable(cdf[PROTON_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME])
            velocity_rtn_covariance = read_numeric_variable(cdf[PROTON_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME])
            density = read_numeric_variable(cdf[PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME])
            temperature = read_numeric_variable(cdf[PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME])
            quality_flags = cdf[SWAPI_QUALITY_FLAGS_CDF_VAR_NAME][...]
        return cls(
            l2_parent_file_name,
            velocity_rtn,
            velocity_rtn_covariance,
            density,
            temperature,
            quality_flags
        )