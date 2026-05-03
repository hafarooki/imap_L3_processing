from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from uncertainties import ufloat

from imap_l3_processing.swapi.l3a.chunk_fits import proton_chunk_worker
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


class TestChunkFits(TestCase):
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.derive_velocity_angles")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    def test_proton_chunk_worker_outputs_sun_frame_speed(
        self, mock_fit_proton, mock_derive_velocity_angles
    ):
        data_chunk = Mock()
        data_chunk.coincidence_count_rate = np.ones((5, 72))
        bulk_velocity_rtn_sc = np.array([400.0, 10.0, 5.0])
        sc_velocity_rtn = np.array([10.0, -3.0, 2.0])
        velocity_covariance = np.diag([4.0, 9.0, 16.0])

        fit_result = Mock()
        fit_result.bad_fit_flag = SwapiL3Flags.NONE
        fit_result.bulk_velocity_rtn_nominal.return_value = bulk_velocity_rtn_sc
        fit_result.bulk_velocity_rtn_covariance.return_value = velocity_covariance
        fit_result.bulk_velocity_rtn = tuple(
            ufloat(component, sigma)
            for component, sigma in zip(
                bulk_velocity_rtn_sc, np.sqrt(np.diag(velocity_covariance))
            )
        )
        fit_result.density = ufloat(5.0, 0.5)
        fit_result.temperature = ufloat(12000.0, 100.0)
        mock_fit_proton.return_value = fit_result
        mock_derive_velocity_angles.return_value = (
            ufloat(np.linalg.norm(bulk_velocity_rtn_sc), 1.0),
            ufloat(30.0, 2.0),
            ufloat(4.0, 0.2),
        )

        result = proton_chunk_worker(
            data_chunk,
            123,
            np.tile(np.eye(3), (5, 1, 1)),
            sc_velocity_rtn,
        )

        expected_sun_velocity = bulk_velocity_rtn_sc + sc_velocity_rtn
        expected_sun_speed = np.linalg.norm(expected_sun_velocity)
        expected_sun_speed_uncert = (
            np.sqrt(expected_sun_velocity @ velocity_covariance @ expected_sun_velocity)
            / expected_sun_speed
        )

        np.testing.assert_allclose(
            result["proton_sw_bulk_velocity_rtn_sc"], bulk_velocity_rtn_sc
        )
        np.testing.assert_allclose(
            result["proton_sw_bulk_velocity_rtn_sun"], expected_sun_velocity
        )
        self.assertAlmostEqual(result["proton_sw_speed_sun"], expected_sun_speed)
        self.assertAlmostEqual(
            result["proton_sw_speed_sun_uncert"], expected_sun_speed_uncert
        )
