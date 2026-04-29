from unittest.mock import Mock

import numpy as np
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import (
    THIRTY_SECONDS_IN_NANOSECONDS,
    FIVE_MINUTES_IN_NANOSECONDS,
)
from imap_l3_processing.swapi.l3a.models import (
    SwapiL3ProtonSolarWindData,
    EPOCH_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME,
    EPOCH_DELTA_CDF_VAR_NAME,
    SwapiL3AlphaSolarWindData,
    ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME,
    ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
    SwapiL3PickupIonData,
    PUI_COOLING_INDEX_CDF_VAR_NAME,
    PUI_IONIZATION_RATE_CDF_VAR_NAME,
    PUI_CUTOFF_SPEED_CDF_VAR_NAME,
    PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME,
    PUI_DENSITY_CDF_VAR_NAME,
    PUI_TEMPERATURE_CDF_VAR_NAME,
    PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME,
    PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME,
    PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME,
    PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME,
    PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
    PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
    SWAPI_QUALITY_FLAGS_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SUN_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_CDF_VAR_NAME,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_getting_proton_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        expected_nominal_values = np.arange(10, step=1.0)
        expected_std = np.arange(5, step=0.5)
        proton_speed = uarray(expected_nominal_values, expected_std)
        expected_temperature_nominal_values = np.arange(1000, 2000, step=100.0)
        expected_temperature_std = np.arange(50, step=5.0)
        temperature_data = uarray(
            expected_temperature_nominal_values, expected_temperature_std
        )
        expected_density_nominal_values = np.arange(3, 13, step=1.0)
        expected_density_std = np.arange(1, step=0.1)
        density_data = uarray(expected_density_nominal_values, expected_density_std)
        expected_clock_angle = np.arange(10, step=1.0)
        expected_clock_angle_std = np.arange(2, step=0.2)
        clock_angle_data = uarray(expected_clock_angle, expected_clock_angle_std)
        expected_flow_deflection = np.arange(100, step=10.0)
        expected_flow_deflection_std = np.arange(1, step=0.1)
        flow_deflection_data = uarray(
            expected_flow_deflection, expected_flow_deflection_std
        )
        quality_flags = np.full(20, SwapiL3Flags.NONE)
        quality_flags[3:5] |= SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED
        bulk_velocity_rtn_sun = np.random.rand(10, 3)
        bulk_velocity_rtn_sc = np.random.rand(10, 3)
        data = SwapiL3ProtonSolarWindData(
            Mock(),
            epoch_data,
            proton_speed,
            temperature_data,
            density_data,
            clock_angle_data,
            flow_deflection_data,
            quality_flags,
            bulk_velocity_rtn_sun,
            bulk_velocity_rtn_sc,
        )

        variables = data.to_data_product_variables()

        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(
            variables[1], expected_nominal_values, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[2], expected_std, PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[3], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[4],
            expected_temperature_nominal_values,
            PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[5],
            expected_temperature_std,
            PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[6],
            expected_density_nominal_values,
            PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[7],
            expected_density_std,
            PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[8],
            expected_clock_angle,
            PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[9],
            expected_clock_angle_std,
            PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[10],
            expected_flow_deflection,
            PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[11],
            expected_flow_deflection_std,
            PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[12],
            bulk_velocity_rtn_sun,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SUN_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[13],
            bulk_velocity_rtn_sc,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[14], quality_flags, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME
        )

    def test_getting_alpha_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        expected_speed_nominal_values = np.arange(10, step=1.0)
        expected_speed_std = np.arange(5, step=0.5)
        alpha_speed = uarray(expected_speed_nominal_values, expected_speed_std)
        expected_temperature_nominal_values = np.arange(300000, step=30000.0)
        expected_temperature_std_devs = np.arange(50000, step=5000.0)
        alpha_temperature = uarray(
            expected_temperature_nominal_values, expected_temperature_std_devs
        )
        expected_alpha_density_nominal_values = np.arange(2, step=0.2)
        expected_alpha_density_std_devs = np.arange(1, step=0.1)
        expected_flag_values = np.full_like(epoch_data, SwapiL3Flags.NONE)
        expected_flag_values[: len(epoch_data) // 2] = SwapiL3Flags.HI_CHI_SQ

        alpha_density = uarray(
            expected_alpha_density_nominal_values, expected_alpha_density_std_devs
        )

        expected_pre_lut_temperature_nominal_values = np.arange(400000, step=40000.0)
        expected_pre_lut_temperature_std_devs = np.arange(20000, step=2000.0)
        alpha_pre_lut_temperature = uarray(
            expected_pre_lut_temperature_nominal_values,
            expected_pre_lut_temperature_std_devs,
        )
        expected_pre_lut_alpha_density_nominal_values = np.arange(2.4, step=0.24)
        expected_pre_lut_alpha_density_std_devs = np.arange(1.5, step=0.15)
        alpha_pre_lut_density = uarray(
            expected_pre_lut_alpha_density_nominal_values,
            expected_pre_lut_alpha_density_std_devs,
        )

        data = SwapiL3AlphaSolarWindData(
            Mock(),
            epoch_data,
            alpha_speed,
            alpha_temperature,
            alpha_density,
            expected_flag_values,
            alpha_pre_lut_temperature,
            alpha_pre_lut_density,
        )
        variables = data.to_data_product_variables()

        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(
            variables[1], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[2],
            expected_speed_nominal_values,
            ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[3],
            expected_speed_std,
            ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[4], expected_temperature_nominal_values, "alpha_sw_temperature"
        )
        self.assert_variable_attributes(
            variables[5], expected_temperature_std_devs, "alpha_sw_temperature_uncert"
        )
        self.assert_variable_attributes(
            variables[6], expected_alpha_density_nominal_values, "alpha_sw_density"
        )
        self.assert_variable_attributes(
            variables[7], expected_alpha_density_std_devs, "alpha_sw_density_uncert"
        )

        self.assert_variable_attributes(variables[8], expected_flag_values, "swp_flags")

        self.assert_variable_attributes(
            variables[9],
            expected_pre_lut_temperature_nominal_values,
            "alpha_sw_pre_lut_temperature",
        )
        self.assert_variable_attributes(
            variables[10],
            expected_pre_lut_alpha_density_nominal_values,
            "alpha_sw_pre_lut_density",
        )

    def test_getting_pui_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        expected_epoch_delta = np.full(10, FIVE_MINUTES_IN_NANOSECONDS)
        expected_cooling_index_nominal = np.arange(10, step=1.0)
        expected_cooling_index_std_dev = np.arange(0.1, step=0.01)
        expected_cooling_index = uarray(
            expected_cooling_index_nominal, expected_cooling_index_std_dev
        )

        expected_ionization_rate_nominal = np.arange(300000, step=30000.0)
        expected_ionization_rate_std_dev = np.arange(10, step=1.0)
        expected_ionization_rate = uarray(
            expected_ionization_rate_nominal, expected_ionization_rate_std_dev
        )

        expected_cutoff_speed_nominal = np.arange(50000, step=5000.0)
        expected_cutoff_speed_std_dev = np.arange(5, step=0.5)
        expected_cutoff_speed = uarray(
            expected_cutoff_speed_nominal, expected_cutoff_speed_std_dev
        )

        expected_background_nominal = np.arange(1, step=0.1)
        expected_background_std_dev = np.arange(0.01, step=0.001)
        expected_background = uarray(
            expected_background_nominal, expected_background_std_dev
        )

        expected_density_nominal = np.arange(0.001, step=0.0001)
        expected_density_std_dev = np.arange(0.0001, step=0.00001)
        expected_density = uarray(expected_density_nominal, expected_density_std_dev)

        expected_temperature_nominal = np.arange(1e5, step=10000.0)
        expected_temperature_std_dev = np.arange(100, step=10.0)
        expected_temperature = uarray(
            expected_temperature_nominal, expected_temperature_std_dev
        )

        expected_quality_flags = np.full(20, 0)

        data = SwapiL3PickupIonData(
            Mock(),
            epoch_data,
            expected_cooling_index,
            expected_ionization_rate,
            expected_cutoff_speed,
            expected_background,
            expected_density,
            expected_temperature,
            expected_quality_flags,
        )
        variables = data.to_data_product_variables()

        self.assertEqual(15, len(variables))
        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(
            variables[1], expected_epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[2], expected_cooling_index_nominal, PUI_COOLING_INDEX_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[3],
            expected_cooling_index_std_dev,
            PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[4],
            expected_ionization_rate_nominal,
            PUI_IONIZATION_RATE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[5],
            expected_ionization_rate_std_dev,
            PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[6], expected_cutoff_speed_nominal, PUI_CUTOFF_SPEED_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[7],
            expected_cutoff_speed_std_dev,
            PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[8],
            expected_background_nominal,
            PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[9],
            expected_background_std_dev,
            PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[10], expected_density_nominal, PUI_DENSITY_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[11],
            expected_density_std_dev,
            PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[12], expected_temperature_nominal, PUI_TEMPERATURE_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[13],
            expected_temperature_std_dev,
            PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[14], expected_quality_flags, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME
        )
