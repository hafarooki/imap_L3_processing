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
    PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME,
    EPOCH_DELTA_CDF_VAR_NAME,
    SwapiL3AlphaSolarWindData,
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
    PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SUN_COVARIANCE_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_CDF_VAR_NAME,
    PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_COVARIANCE_CDF_VAR_NAME,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_getting_proton_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        expected_nominal_values = np.arange(10, step=1.0)
        expected_std = np.arange(5, step=0.5)
        expected_sun_nominal_values = np.arange(20, 30, step=1.0)
        expected_sun_std = np.arange(10, step=1.0)
        expected_temperature_nominal_values = np.arange(1000, 2000, step=100.0)
        expected_temperature_std = np.arange(50, step=5.0)
        expected_density_nominal_values = np.arange(3, 13, step=1.0)
        expected_density_std = np.arange(1, step=0.1)
        expected_clock_angle = np.arange(10, step=1.0)
        expected_clock_angle_std = np.arange(2, step=0.2)
        expected_flow_deflection = np.arange(100, step=10.0)
        expected_flow_deflection_std = np.arange(1, step=0.1)
        quality_flags = np.full(20, SwapiL3Flags.NONE)
        quality_flags[3:5] |= SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED
        bulk_velocity_rtn_sun = np.random.rand(10, 3)
        bulk_velocity_rtn_sun_covariance = np.random.rand(10, 3, 3)
        bulk_velocity_rtn_sc = np.random.rand(10, 3)
        bulk_velocity_rtn_sc_covariance = np.random.rand(10, 3, 3)
        data = SwapiL3ProtonSolarWindData(
            Mock(),
            epoch_data,
            expected_nominal_values,
            expected_std,
            expected_sun_nominal_values,
            expected_sun_std,
            expected_temperature_nominal_values,
            expected_temperature_std,
            expected_density_nominal_values,
            expected_density_std,
            expected_clock_angle,
            expected_clock_angle_std,
            expected_flow_deflection,
            expected_flow_deflection_std,
            bulk_velocity_rtn_sun,
            bulk_velocity_rtn_sun_covariance,
            bulk_velocity_rtn_sc,
            bulk_velocity_rtn_sc_covariance,
            quality_flags,
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
            variables[3],
            expected_sun_nominal_values,
            PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[4],
            expected_sun_std,
            PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[5], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[6],
            expected_temperature_nominal_values,
            PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[7],
            expected_temperature_std,
            PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[8],
            expected_density_nominal_values,
            PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[9],
            expected_density_std,
            PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[10],
            expected_clock_angle,
            PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[11],
            expected_clock_angle_std,
            PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[12],
            expected_flow_deflection,
            PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[13],
            expected_flow_deflection_std,
            PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[14],
            bulk_velocity_rtn_sun,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SUN_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[15],
            bulk_velocity_rtn_sun_covariance,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SUN_COVARIANCE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[16],
            bulk_velocity_rtn_sc,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[17],
            bulk_velocity_rtn_sc_covariance,
            PROTON_SOLAR_WIND_BULK_VELOCITY_RTN_SC_COVARIANCE_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[18], quality_flags, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME
        )

    def test_getting_alpha_sw_data_product_variables(self):
        n = 10
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        density = np.arange(n, dtype=float)
        density_uncert = np.arange(n, dtype=float) * 0.1
        temperature = np.arange(n, dtype=float) * 1e4
        temperature_uncert = np.arange(n, dtype=float) * 1e3
        velocity_rtn = np.random.randn(n, 3)
        velocity_covariance_rtn = np.random.randn(n, 3, 3)
        delta_v = np.arange(n, dtype=float) * 10
        delta_v_uncert = np.arange(n, dtype=float)
        b_hat_rtn = np.random.randn(n, 3)
        ref_proton_density = np.arange(n, dtype=float) * 5
        ref_proton_temperature = np.arange(n, dtype=float) * 1e5
        ref_proton_velocity_rtn = np.random.randn(n, 3)
        bad_fit_flag = np.full(n, SwapiL3Flags.NONE)
        bad_fit_flag[: n // 2] = SwapiL3Flags.BAD_FIT

        data = SwapiL3AlphaSolarWindData(
            Mock(),
            epoch_data,
            density,
            density_uncert,
            temperature,
            temperature_uncert,
            velocity_rtn,
            velocity_covariance_rtn,
            delta_v,
            delta_v_uncert,
            b_hat_rtn,
            ref_proton_density,
            ref_proton_temperature,
            ref_proton_velocity_rtn,
            bad_fit_flag,
        )
        variables = data.to_data_product_variables()

        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(
            variables[1], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )
        self.assert_variable_attributes(variables[2], density, "alpha_sw_density")
        self.assert_variable_attributes(
            variables[3], density_uncert, "alpha_sw_density_uncert"
        )
        self.assert_variable_attributes(
            variables[4], temperature, "alpha_sw_temperature"
        )
        self.assert_variable_attributes(
            variables[5], temperature_uncert, "alpha_sw_temperature_uncert"
        )
        self.assert_variable_attributes(
            variables[6], velocity_rtn, "alpha_sw_velocity_rtn"
        )
        self.assert_variable_attributes(
            variables[7],
            velocity_covariance_rtn,
            "alpha_sw_velocity_covariance_rtn",
        )
        self.assert_variable_attributes(variables[8], delta_v, "alpha_sw_delta_v")
        self.assert_variable_attributes(
            variables[9], delta_v_uncert, "alpha_sw_delta_v_uncert"
        )
        self.assert_variable_attributes(variables[10], b_hat_rtn, "alpha_sw_b_hat_rtn")
        self.assert_variable_attributes(
            variables[11], ref_proton_density, "alpha_sw_reference_proton_density"
        )
        self.assert_variable_attributes(
            variables[12],
            ref_proton_temperature,
            "alpha_sw_reference_proton_temperature",
        )
        self.assert_variable_attributes(
            variables[13],
            ref_proton_velocity_rtn,
            "alpha_sw_reference_proton_velocity_rtn",
        )
        self.assert_variable_attributes(variables[14], bad_fit_flag, "swp_flags")

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
