from unittest.mock import patch, sentinel

import numpy as np

from imap_l3_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, HE_PUI_PARTICLE_MASS_KG
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import (
    calculate_ten_minute_velocities,
    convert_velocity_relative_to_imap,
    convert_velocity_to_reference_frame,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.spice_test_case import SpiceTestCase

_UTILS_MODULE = "imap_l3_processing.swapi.l3a.science.pickup_ion.utils"

_FAKE_SXFORM_ROTATION = np.array(
    [
        [-0.803319036, -0.595067395, -0.023944118, 0.0, 0.0, 0.0],
        [0.594803234, -0.803675802, 0.017728995, 0.0, 0.0, 0.0],
        [-0.029793255, 0.0, 0.999556082, 0.0, 0.0, 0.0],
        [-1.16314295e-06, 1.56750981e-06, 6.68593934e-08, -0.803319036, -0.595067395, -0.023944118],
        [-1.56457525e-06, -1.16063465e-06, -1.21809529e-07, 0.594803234, -0.803675802, 0.017728995],
        [1.26218156e-07, 5.29395592e-23, 3.76211978e-09, -0.029793255, 0.0, 0.999556082],
    ]
)


class ConvertVelocityRelativeToImapTest(SpiceTestCase):
    @patch(f"{_UTILS_MODULE}.spiceypy")
    def test_adds_imap_inertial_velocity_to_rotated_input(self, mock_spice):
        mock_spice.sxform.return_value = _FAKE_SXFORM_ROTATION
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 98, 77, 66]), 12.459)

        input_velocity = np.array([[12, 34, 45], [67, 89, 45]])
        ephemeris_time = 2000

        output_velocity = convert_velocity_relative_to_imap(
            input_velocity, ephemeris_time, "INPUT_FRAME", "OUTPUT_FRAME"
        )

        expected = np.array(
            [
                [67.05039482, 57.6104663, 110.62250463],
                [-9.860859, 46.122475, 108.983876],
            ]
        )
        np.testing.assert_array_almost_equal(output_velocity, expected)
        mock_spice.sxform.assert_called_with("INPUT_FRAME", "OUTPUT_FRAME", ephemeris_time)
        mock_spice.spkezr.assert_called_with(
            "IMAP", ephemeris_time, "OUTPUT_FRAME", "NONE", "SUN"
        )


class ConvertVelocityToReferenceFrameTest(SpiceTestCase):
    @patch(f"{_UTILS_MODULE}.spiceypy")
    def test_applies_state_block_of_sxform_to_velocity(self, mock_spice):
        mock_spice.sxform.return_value = _FAKE_SXFORM_ROTATION
        ephemeris_time = 10_000_000

        input_2d = np.array([[1, 2, 3], [1, 2, 3]])
        result_2d = convert_velocity_to_reference_frame(
            input_2d, ephemeris_time, "FROM", "TO"
        )
        expected_row = (
            1 * _FAKE_SXFORM_ROTATION[3:6, 3]
            + 2 * _FAKE_SXFORM_ROTATION[3:6, 4]
            + 3 * _FAKE_SXFORM_ROTATION[3:6, 5]
        )
        np.testing.assert_array_almost_equal([expected_row, expected_row], result_2d)
        mock_spice.sxform.assert_called_with("FROM", "TO", ephemeris_time)

        result_1d = convert_velocity_to_reference_frame(
            input_2d[0], ephemeris_time, "FROM", "TO"
        )
        np.testing.assert_array_almost_equal(expected_row, result_1d)


class CalculateTenMinuteVelocitiesTest(SpiceTestCase):
    def _velocities(self):
        x = np.arange(1, 22)
        y = np.arange(10, 211, 10)
        z = np.arange(10, 211, 10)
        return np.transpose([x, y, z]).astype(float)

    def test_averages_per_minute_velocities_in_ten_minute_windows(self):
        velocities = self._velocities()
        quality_flags = np.repeat(SwapiL3Flags.NONE, 21)

        averaged, ten_minute_flags = calculate_ten_minute_velocities(
            velocities, list(quality_flags)
        )

        expected_velocities = np.array(
            [[5.5, 55.0, 55.0], [15.5, 155.0, 155.0], [21.0, 210.0, 210.0]]
        )
        np.testing.assert_array_equal(averaged, expected_velocities)
        np.testing.assert_array_equal(
            ten_minute_flags, np.repeat(SwapiL3Flags.NONE, 3)
        )

    def test_ors_per_minute_quality_flags_within_window(self):
        velocities = self._velocities()
        quality_flags = np.repeat(SwapiL3Flags.NONE, 21)
        quality_flags[13] = SwapiL3Flags.FIT_ERROR

        _, ten_minute_flags = calculate_ten_minute_velocities(
            velocities, list(quality_flags)
        )

        np.testing.assert_array_equal(
            ten_minute_flags,
            np.array(
                [SwapiL3Flags.NONE, SwapiL3Flags.FIT_ERROR, SwapiL3Flags.NONE]
            ),
        )

    def test_combines_multiple_per_minute_quality_flags_within_window(self):
        velocities = self._velocities()
        quality_flags = np.repeat(SwapiL3Flags.NONE, 21)
        other_flag = 1 << 3
        quality_flags[13] = SwapiL3Flags.FIT_ERROR
        quality_flags[14] = other_flag

        _, ten_minute_flags = calculate_ten_minute_velocities(
            velocities, list(quality_flags)
        )

        np.testing.assert_array_equal(
            ten_minute_flags,
            np.array(
                [
                    SwapiL3Flags.NONE,
                    SwapiL3Flags.FIT_ERROR | other_flag,
                    SwapiL3Flags.NONE,
                ]
            ),
        )
