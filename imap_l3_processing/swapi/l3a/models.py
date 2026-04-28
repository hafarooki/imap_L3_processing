from dataclasses import dataclass
from datetime import datetime

import numpy as np
from uncertainties.unumpy import nominal_values, std_devs

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_uncert"
PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "proton_sw_temperature"
PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_temperature_uncert"
PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "proton_sw_density"
PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_density_uncert"

PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME = "proton_sw_clock_angle"
PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_clock_angle_uncert"

PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME = "proton_sw_deflection_angle"
PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_deflection_angle_uncert"

ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME = "alpha_sw_speed"
ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_uncert"
ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "alpha_sw_density"
ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_density_uncert"
ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_temperature"
ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_temperature_uncert"
ALPHA_SOLAR_WIND_PRE_LUT_DENSITY_CDF_VAR_NAME = "alpha_sw_pre_lut_density"
ALPHA_SOLAR_WIND_PRE_LUT_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_pre_lut_temperature"

# 3-DOF moments fit (n_α, T_α, Δv) outputs — see docs/swapi/solar-wind-moments.md.
ALPHA_SOLAR_WIND_MOMENTS_DENSITY_CDF_VAR_NAME = "alpha_sw_moments_density"
ALPHA_SOLAR_WIND_MOMENTS_DENSITY_UNCERT_CDF_VAR_NAME = "alpha_sw_moments_density_uncert"
ALPHA_SOLAR_WIND_MOMENTS_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_moments_temperature"
ALPHA_SOLAR_WIND_MOMENTS_TEMPERATURE_UNCERT_CDF_VAR_NAME = "alpha_sw_moments_temperature_uncert"
ALPHA_SOLAR_WIND_MOMENTS_VELOCITY_RTN_CDF_VAR_NAME = "alpha_sw_moments_velocity_rtn"
ALPHA_SOLAR_WIND_MOMENTS_VELOCITY_COVARIANCE_RTN_CDF_VAR_NAME = "alpha_sw_moments_velocity_covariance_rtn"
ALPHA_SOLAR_WIND_MOMENTS_DELTA_V_CDF_VAR_NAME = "alpha_sw_moments_delta_v"
ALPHA_SOLAR_WIND_MOMENTS_DELTA_V_UNCERT_CDF_VAR_NAME = "alpha_sw_moments_delta_v_uncert"
ALPHA_SOLAR_WIND_MOMENTS_B_HAT_RTN_CDF_VAR_NAME = "alpha_sw_moments_b_hat_rtn"
ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_DENSITY_CDF_VAR_NAME = "alpha_sw_moments_reference_proton_density"
ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_moments_reference_proton_temperature"
ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_VELOCITY_RTN_CDF_VAR_NAME = "alpha_sw_moments_reference_proton_velocity_rtn"
ALPHA_SOLAR_WIND_MOMENTS_BAD_FIT_FLAG_CDF_VAR_NAME = "alpha_sw_moments_bad_fit_flag"

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


@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]
    proton_sw_temperature: np.ndarray[float]
    proton_sw_density: np.ndarray[float]
    proton_sw_clock_angle: np.ndarray[float]
    proton_sw_deflection_angle: np.ndarray[float]
    quality_flags: np.ndarray[SwapiL3Flags]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.proton_sw_speed)),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.proton_sw_speed)),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS)),
            DataProductVariable(PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.proton_sw_temperature)),
            DataProductVariable(PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_temperature)),
            DataProductVariable(PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME, nominal_values(self.proton_sw_density)),
            DataProductVariable(PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_density)),
            DataProductVariable(PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME, nominal_values(self.proton_sw_clock_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_clock_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME,
                                nominal_values(self.proton_sw_deflection_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_deflection_angle)),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags)
        ]


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    epoch: np.ndarray[datetime]
    alpha_sw_speed: np.ndarray
    alpha_sw_temperature: np.ndarray
    alpha_sw_density: np.ndarray
    bad_fit_flag: np.ndarray
    alpha_sw_pre_lut_temperature: np.ndarray
    alpha_sw_pre_lut_density: np.ndarray
    # Moments fit (n_α, T_α, Δv) outputs — populated by Stage 2 of the alpha pipeline.
    # NaN-filled when the moments fit fails or wasn't run; the LUT outputs above are
    # independent and remain populated whenever the LUT pipeline succeeds.
    alpha_sw_moments_density: np.ndarray = None  # (N,)
    alpha_sw_moments_density_uncert: np.ndarray = None  # (N,)
    alpha_sw_moments_temperature: np.ndarray = None  # (N,)
    alpha_sw_moments_temperature_uncert: np.ndarray = None  # (N,)
    alpha_sw_moments_velocity_rtn: np.ndarray = None  # (N, 3)
    alpha_sw_moments_velocity_covariance_rtn: np.ndarray = None  # (N, 3, 3)
    alpha_sw_moments_delta_v: np.ndarray = None  # (N,) signed km/s
    alpha_sw_moments_delta_v_uncert: np.ndarray = None  # (N,)
    alpha_sw_moments_b_hat_rtn: np.ndarray = None  # (N, 3)
    alpha_sw_moments_reference_proton_density: np.ndarray = None  # (N,)
    alpha_sw_moments_reference_proton_temperature: np.ndarray = None  # (N,)
    alpha_sw_moments_reference_proton_velocity_rtn: np.ndarray = None  # (N, 3)
    alpha_sw_moments_bad_fit_flag: np.ndarray = None  # (N,) — separate from LUT bad_fit_flag

    def to_data_product_variables(self) -> list[DataProductVariable]:
        variables = [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS)),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.alpha_sw_speed)),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.alpha_sw_speed)),
            DataProductVariable(ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.alpha_sw_temperature)),
            DataProductVariable(ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.alpha_sw_temperature)),
            DataProductVariable(ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME, nominal_values(self.alpha_sw_density)),
            DataProductVariable(ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.alpha_sw_density)),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.bad_fit_flag),
            DataProductVariable(ALPHA_SOLAR_WIND_PRE_LUT_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.alpha_sw_pre_lut_temperature)),
            DataProductVariable(ALPHA_SOLAR_WIND_PRE_LUT_DENSITY_CDF_VAR_NAME, nominal_values(self.alpha_sw_pre_lut_density)),
        ]
        if self.alpha_sw_moments_density is not None:
            variables.extend([
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_DENSITY_CDF_VAR_NAME, self.alpha_sw_moments_density),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_DENSITY_UNCERT_CDF_VAR_NAME, self.alpha_sw_moments_density_uncert),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_TEMPERATURE_CDF_VAR_NAME, self.alpha_sw_moments_temperature),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_TEMPERATURE_UNCERT_CDF_VAR_NAME, self.alpha_sw_moments_temperature_uncert),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_VELOCITY_RTN_CDF_VAR_NAME, self.alpha_sw_moments_velocity_rtn),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_VELOCITY_COVARIANCE_RTN_CDF_VAR_NAME, self.alpha_sw_moments_velocity_covariance_rtn),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_DELTA_V_CDF_VAR_NAME, self.alpha_sw_moments_delta_v),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_DELTA_V_UNCERT_CDF_VAR_NAME, self.alpha_sw_moments_delta_v_uncert),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_B_HAT_RTN_CDF_VAR_NAME, self.alpha_sw_moments_b_hat_rtn),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_DENSITY_CDF_VAR_NAME, self.alpha_sw_moments_reference_proton_density),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_TEMPERATURE_CDF_VAR_NAME, self.alpha_sw_moments_reference_proton_temperature),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_REF_PROTON_VELOCITY_RTN_CDF_VAR_NAME, self.alpha_sw_moments_reference_proton_velocity_rtn),
                DataProductVariable(ALPHA_SOLAR_WIND_MOMENTS_BAD_FIT_FLAG_CDF_VAR_NAME, self.alpha_sw_moments_bad_fit_flag),
            ])
        return variables


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
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, FIVE_MINUTES_IN_NANOSECONDS)),
            DataProductVariable(PUI_COOLING_INDEX_CDF_VAR_NAME, nominal_values(self.cooling_index)),
            DataProductVariable(PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cooling_index)),
            DataProductVariable(PUI_IONIZATION_RATE_CDF_VAR_NAME, nominal_values(self.ionization_rate)),
            DataProductVariable(PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.ionization_rate)),
            DataProductVariable(PUI_CUTOFF_SPEED_CDF_VAR_NAME, nominal_values(self.cutoff_speed)),
            DataProductVariable(PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cutoff_speed)),
            DataProductVariable(PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME, nominal_values(self.background_rate)),
            DataProductVariable(PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.background_rate)),
            DataProductVariable(PUI_DENSITY_CDF_VAR_NAME, nominal_values(self.density)),
            DataProductVariable(PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.density)),
            DataProductVariable(PUI_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.temperature)),
            DataProductVariable(PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.temperature)),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags),
        ]


@dataclass
class SwapiL2Data:
    sci_start_time: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    coincidence_count_rate_uncertainty: np.ndarray[float]
