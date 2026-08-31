import unittest
from unittest.mock import Mock, sentinel

import numpy as np

from imap_l3_processing.swe.l3.models import SweL3Data, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    ENERGY_CDF_VAR_NAME, \
    ENERGY_DELTA_PLUS_CDF_VAR_NAME, ENERGY_DELTA_MINUS_CDF_VAR_NAME, PITCH_ANGLE_CDF_VAR_NAME, \
    PITCH_ANGLE_DELTA_CDF_VAR_NAME, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, \
    PHASE_SPACE_DENSITY_1D_CDF_VAR_NAME, PHASE_SPACE_DENSITY_OUTWARD_CDF_VAR_NAME, \
    PHASE_SPACE_DENSITY_INWARD_CDF_VAR_NAME, \
    SPACECRAFT_POTENTIAL_CDF_VAR_NAME, CORE_HALO_BREAKPOINT_CDF_VAR_NAME, CORE_FIT_NUM_POINTS_CDF_VAR_NAME, \
    CHISQ_C_CDF_VAR_NAME, CHISQ_H_CDF_VAR_NAME, CORE_DENSITY_FIT_CDF_VAR_NAME, HALO_DENSITY_FIT_CDF_VAR_NAME, \
    CORE_T_PARALLEL_FIT_CDF_VAR_NAME, HALO_T_PARALLEL_FIT_CDF_VAR_NAME, CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME, \
    HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME, CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, \
    HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, CORE_SPEED_FIT_CDF_VAR_NAME, HALO_SPEED_FIT_CDF_VAR_NAME, \
    CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, \
    INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME, \
    CORE_DENSITY_INTEGRATED_CDF_VAR_NAME, HALO_DENSITY_INTEGRATED_CDF_VAR_NAME, TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME, \
    CORE_SPEED_INTEGRATED_CDF_VAR_NAME, HALO_SPEED_INTEGRATED_CDF_VAR_NAME, TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME, \
    CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, \
    CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, \
    HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, \
    HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, \
    CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, SweL3MomentData, CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, CORE_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME, \
    HALO_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME, TOTAL_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME, \
    PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME, GYROPHASE_DELTA_CDF_VAR_NAME, \
    GYROPHASE_BINS_CDF_VAR_NAME, INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME, \
    INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_CDF_VAR_NAME, ENERGY_LABEL, PITCH_ANGLE_LABEL, GYROPHASE_LABEL, RTN_LABEL, \
    CORE_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME, \
    HALO_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME, TOTAL_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME, \
    CORE_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, HALO_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, \
    TOTAL_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, TENSOR_ID, SWE_FLAGS_VAR_NAME, \
    CORE_VELOCITY_VECTOR_RTN_FIT_LABEL, HALO_VELOCITY_VECTOR_RTN_FIT_LABEL, CORE_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL, \
    HALO_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL, TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL, \
    CORE_TEMPERATURE_TENSOR_INTEGRATED_LABEL, HALO_TEMPERATURE_TENSOR_INTEGRATED_LABEL, \
    TOTAL_TEMPERATURE_TENSOR_INTEGRATED_LABEL
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_data_to_product_variables(self):
        energy = np.array([10, 20, 30])
        pitch_angle = np.array([11, 12, 13, 14, 15])
        gyrophase_bins = np.array([20, 40, 60, 80])
        core_t_perpendicular_integrated = np.array([[1, 2], [3, 4]])
        halo_t_perpendicular_integrated = np.array([[5, 6], [7, 8]])
        total_t_perpendicular_integrated = np.array([[9, 10], [11, 12]])
        core_temperature_perpendicular_to_mag = np.array([[11, 12], [13, 14]])
        halo_temperature_perpendicular_to_mag = np.array([[15, 16], [17, 18]])
        total_temperature_perpendicular_to_mag = np.array([[19, 20], [21, 22]])

        data = SweL3Data(
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=energy,
            energy_delta_plus=sentinel.energy_delta_plus,
            energy_delta_minus=sentinel.energy_delta_minus,
            pitch_angle=pitch_angle,
            pitch_angle_delta=sentinel.pitch_angle_delta,
            gyrophase_bins=gyrophase_bins,
            gyrophase_delta=sentinel.gyrophase_delta,
            phase_space_density_by_pitch_angle=sentinel.psd_by_pitch_angle,
            phase_space_density_by_pitch_angle_and_gyrophase=sentinel.psd_by_pitch_angle_and_gyrophase,
            phase_space_density_1d=sentinel.energy_spectrum,
            phase_space_density_inward=sentinel.energy_spectrum_inbound,
            phase_space_density_outward=sentinel.energy_spectrum_outbound,
            intensity_by_pitch_angle=sentinel.intensity_by_pitch_angle,
            intensity_by_pitch_angle_and_gyrophase=sentinel.intensity_by_pitch_angle_and_gyrophase,
            intensity_uncertainty_by_pitch_angle=sentinel.intensity_uncertainty_by_pitch_angle,
            intensity_uncertainty_by_pitch_angle_and_gyrophase=sentinel.intensity_uncertainty_by_pitch_angle_and_gyrophase,
            spacecraft_potential=sentinel.spacecraft_potential,
            core_halo_breakpoint=sentinel.core_halo_breakpoint,
            moment_data=SweL3MomentData(
                core_fit_num_points=sentinel.core_fit_num_points,
                core_chisq=sentinel.chisq_c,
                halo_chisq=sentinel.chisq_h,
                core_density_fit=sentinel.core_density_fit,
                halo_density_fit=sentinel.halo_density_fit,
                core_t_parallel_fit=sentinel.core_t_parallel_fit,
                halo_t_parallel_fit=sentinel.halo_t_parallel_fit,
                core_t_perpendicular_fit=sentinel.core_t_perpendicular_fit,
                halo_t_perpendicular_fit=sentinel.halo_t_perpendicular_fit,
                core_temperature_phi_rtn_fit=sentinel.core_temperature_phi_rtn_fit,
                halo_temperature_phi_rtn_fit=sentinel.halo_temperature_phi_rtn_fit,
                core_temperature_theta_rtn_fit=sentinel.core_temperature_theta_rtn_fit,
                halo_temperature_theta_rtn_fit=sentinel.halo_temperature_theta_rtn_fit,
                core_speed_fit=sentinel.core_speed_fit,
                halo_speed_fit=sentinel.halo_speed_fit,
                core_velocity_vector_rtn_fit=sentinel.core_velocity_vector_rtn_fit,
                halo_velocity_vector_rtn_fit=sentinel.halo_velocity_vector_rtn_fit,
                core_density_integrated=sentinel.core_density_integrated,
                halo_density_integrated=sentinel.halo_density_integrated,
                total_density_integrated=sentinel.total_density_integrated,
                core_speed_integrated=sentinel.core_speed_integrated,
                halo_speed_integrated=sentinel.halo_speed_integrated,
                total_speed_integrated=sentinel.total_speed_integrated,
                core_velocity_vector_rtn_integrated=sentinel.core_velocity_vector_rtn_integrated,
                halo_velocity_vector_rtn_integrated=sentinel.halo_velocity_vector_rtn_integrated,
                total_velocity_vector_rtn_integrated=sentinel.total_velocity_vector_rtn_integrated,
                core_heat_flux_magnitude_integrated=sentinel.core_heat_flux_magnitude_integrated,
                core_heat_flux_theta_integrated=sentinel.core_heat_flux_theta_integrated,
                core_heat_flux_phi_integrated=sentinel.core_heat_flux_phi_integrated,
                halo_heat_flux_magnitude_integrated=sentinel.halo_heat_flux_magnitude_integrated,
                halo_heat_flux_theta_integrated=sentinel.halo_heat_flux_theta_integrated,
                halo_heat_flux_phi_integrated=sentinel.halo_heat_flux_phi_integrated,
                total_heat_flux_magnitude_integrated=sentinel.total_heat_flux_magnitude_integrated,
                total_heat_flux_theta_integrated=sentinel.total_heat_flux_theta_integrated,
                total_heat_flux_phi_integrated=sentinel.total_heat_flux_phi_integrated,
                core_t_parallel_integrated=sentinel.core_t_parallel_integrated,
                core_t_perpendicular_integrated=core_t_perpendicular_integrated,
                halo_t_parallel_integrated=sentinel.halo_t_parallel_integrated,
                halo_t_perpendicular_integrated=halo_t_perpendicular_integrated,
                total_t_parallel_integrated=sentinel.total_t_parallel_integrated,
                total_t_perpendicular_integrated=total_t_perpendicular_integrated,
                core_temperature_theta_rtn_integrated=sentinel.core_temperature_theta_rtn_integrated,
                core_temperature_phi_rtn_integrated=sentinel.core_temperature_phi_rtn_integrated,
                halo_temperature_theta_rtn_integrated=sentinel.halo_temperature_theta_rtn_integrated,
                halo_temperature_phi_rtn_integrated=sentinel.halo_temperature_phi_rtn_integrated,
                total_temperature_theta_rtn_integrated=sentinel.total_temperature_theta_rtn_integrated,
                total_temperature_phi_rtn_integrated=sentinel.total_temperature_phi_rtn_integrated,
                core_temperature_parallel_to_mag=sentinel.core_temperature_parallel_to_mag,
                core_temperature_perpendicular_to_mag=core_temperature_perpendicular_to_mag,
                halo_temperature_parallel_to_mag=sentinel.halo_temperature_parallel_to_mag,
                halo_temperature_perpendicular_to_mag=halo_temperature_perpendicular_to_mag,
                total_temperature_parallel_to_mag=sentinel.total_temperature_parallel_to_mag,
                total_temperature_perpendicular_to_mag=total_temperature_perpendicular_to_mag,
                core_temperature_tensor_integrated=sentinel.core_temperature_tensor_integrated,
                halo_temperature_tensor_integrated=sentinel.halo_temperature_tensor_integrated,
                total_temperature_tensor_integrated=sentinel.total_temperature_tensor_integrated,
                quality_flags=sentinel.moment_quality_flags,
            ),
            swe_flags=sentinel.swp_flags,
            inst_el=sentinel.inst_el,
            inst_el_label=sentinel.inst_el_label,
            inst_az=sentinel.inst_az,
            inst_az_label=sentinel.inst_az_label,
            raw_1d_psd_rebinned=sentinel.raw_1d_psd_rebinned,
            raw_psd_by_theta_rebinned=sentinel.raw_psd_by_theta_rebinned,
            raw_psd_by_phi_rebinned=sentinel.raw_psd_by_phi_rebinned,
            input_metadata=Mock(),
        )

        variables = data.to_data_product_variables()

        variables = iter(variables)
        # @formatter:off
        self.assert_variable_attributes(next(variables), sentinel.epoch, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), energy, ENERGY_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.energy_delta_plus, ENERGY_DELTA_PLUS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.energy_delta_minus, ENERGY_DELTA_MINUS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), pitch_angle, PITCH_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.pitch_angle_delta, PITCH_ANGLE_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), gyrophase_bins, GYROPHASE_BINS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.gyrophase_delta, GYROPHASE_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.psd_by_pitch_angle, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.psd_by_pitch_angle_and_gyrophase, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.energy_spectrum, PHASE_SPACE_DENSITY_1D_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.energy_spectrum_inbound, PHASE_SPACE_DENSITY_INWARD_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.energy_spectrum_outbound, PHASE_SPACE_DENSITY_OUTWARD_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.intensity_by_pitch_angle, INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.intensity_by_pitch_angle_and_gyrophase, INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.intensity_uncertainty_by_pitch_angle, INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.intensity_uncertainty_by_pitch_angle_and_gyrophase, INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.spacecraft_potential, SPACECRAFT_POTENTIAL_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_halo_breakpoint, CORE_HALO_BREAKPOINT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_fit_num_points, CORE_FIT_NUM_POINTS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.chisq_c, CHISQ_C_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.chisq_h, CHISQ_H_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_density_fit, CORE_DENSITY_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_density_fit, HALO_DENSITY_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_t_parallel_fit, CORE_T_PARALLEL_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_t_parallel_fit, HALO_T_PARALLEL_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_t_perpendicular_fit, CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_t_perpendicular_fit, HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_phi_rtn_fit, CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_phi_rtn_fit, HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_theta_rtn_fit, CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_theta_rtn_fit, HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_speed_fit, CORE_SPEED_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_speed_fit, HALO_SPEED_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_velocity_vector_rtn_fit, CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_velocity_vector_rtn_fit, HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_density_integrated, CORE_DENSITY_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_density_integrated, HALO_DENSITY_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_density_integrated, TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_speed_integrated, CORE_SPEED_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_speed_integrated, HALO_SPEED_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_speed_integrated, TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_velocity_vector_rtn_integrated, CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_velocity_vector_rtn_integrated, HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_velocity_vector_rtn_integrated, TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_heat_flux_magnitude_integrated, CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_heat_flux_theta_integrated, CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_heat_flux_phi_integrated, CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_heat_flux_magnitude_integrated, HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_heat_flux_theta_integrated, HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_heat_flux_phi_integrated, HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_heat_flux_magnitude_integrated, TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_heat_flux_theta_integrated, TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_heat_flux_phi_integrated, TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_t_parallel_integrated, CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [1,3], CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [2,4], CORE_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_t_parallel_integrated, HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [5,7], HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [6,8], HALO_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_t_parallel_integrated, TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [9,11], TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [10,12], TOTAL_T_PERPENDICULAR_RATIO_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_theta_rtn_integrated, CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_phi_rtn_integrated, CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_theta_rtn_integrated, HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_phi_rtn_integrated, HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_temperature_theta_rtn_integrated, TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_temperature_phi_rtn_integrated, TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_parallel_to_mag, CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [11,13], CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [12,14], CORE_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_parallel_to_mag, HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [15,17], HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [16,18], HALO_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_temperature_parallel_to_mag, TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [19,21], TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [20,22], TOTAL_TEMPERATURE_RATIO_PERPENDICULAR_TO_MAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.core_temperature_tensor_integrated, CORE_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.halo_temperature_tensor_integrated, HALO_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.total_temperature_tensor_integrated, TOTAL_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.swp_flags, SWE_FLAGS_VAR_NAME)
        self.assert_variable_attributes(next(variables), sentinel.raw_1d_psd_rebinned, 'raw_1d_psd_rebinned')
        self.assert_variable_attributes(next(variables), sentinel.raw_psd_by_theta_rebinned, 'raw_psd_by_theta_rebinned')
        self.assert_variable_attributes(next(variables), sentinel.raw_psd_by_phi_rebinned, 'raw_psd_by_phi_rebinned')
        self.assert_variable_attributes(next(variables), sentinel.inst_az, 'inst_az')
        self.assert_variable_attributes(next(variables), sentinel.inst_el, 'inst_el')
        self.assert_variable_attributes(next(variables), ["10.0 eV", "20.0 eV", "30.0 eV"], ENERGY_LABEL)
        self.assert_variable_attributes(next(variables), ["PA 11", "PA 12", "PA 13", "PA 14", "PA 15"], PITCH_ANGLE_LABEL)
        self.assert_variable_attributes(next(variables), ["Gyrophase 20", "Gyrophase 40", "Gyrophase 60", "Gyrophase 80"], GYROPHASE_LABEL)
        self.assert_variable_attributes(next(variables), ["R", "T", "N"], RTN_LABEL)
        self.assert_variable_attributes(next(variables), ["Core v fit R", "Core v fit T", "Core v fit N"], CORE_VELOCITY_VECTOR_RTN_FIT_LABEL)
        self.assert_variable_attributes(next(variables), ["Halo v fit R", "Halo v fit T", "Halo v fit N"], HALO_VELOCITY_VECTOR_RTN_FIT_LABEL)
        self.assert_variable_attributes(next(variables), ["Core v int. R", "Core v int. T", "Core v int. N"], CORE_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), ["Halo v int. R", "Halo v int. T", "Halo v int. N"], HALO_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), ["Total v int. R", "Total v int. T", "Total v int. N"], TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), ["Core T_XX", "Core T_XY", "Core T_YY", "Core T_XZ", "Core T_YZ", "Core T_ZZ"], CORE_TEMPERATURE_TENSOR_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), ["Halo T_XX", "Halo T_XY", "Halo T_YY", "Halo T_XZ", "Halo T_YZ", "Halo T_ZZ"], HALO_TEMPERATURE_TENSOR_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), ["Total T_XX", "Total T_XY", "Total T_YY", "Total T_XZ", "Total T_YZ", "Total T_ZZ"], TOTAL_TEMPERATURE_TENSOR_INTEGRATED_LABEL)
        self.assert_variable_attributes(next(variables), sentinel.inst_az_label, 'inst_az_label')
        self.assert_variable_attributes(next(variables), sentinel.inst_el_label, 'inst_el_label')
        self.assert_variable_attributes(next(variables),[1,2,3,4,5, 6], TENSOR_ID)

        self.assertEqual([], list(variables), "unexpected variable found")


if __name__ == '__main__':
    unittest.main()
