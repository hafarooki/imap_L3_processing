from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from uncertainties import ufloat

from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitter,
    ProtonChunkFitter,
    _shared,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


class TestChunkFits(TestCase):
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.derive_velocity_angles")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    def test_proton_fitter_outputs_sun_frame_speed(
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

        result = ProtonChunkFitter().fit_chunk(
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

    def _call_alpha_fit_chunk(self, mag_data_level, rotation_matrices, b_hat_rtn):
        """Call AlphaChunkFitter.fit_chunk with the given geometry arguments.
        _shared is normally populated by the worker initializer; stub it here so
        the early access doesn't KeyError."""
        _shared["swapi_response"] = Mock()
        _shared["efficiency_table"] = Mock()
        try:
            data_chunk = Mock()
            data_chunk.coincidence_count_rate = np.full((5, 72), np.nan)
            fitter = AlphaChunkFitter(mag_data=None, mag_data_level=mag_data_level)
            return fitter.fit_chunk(data_chunk, 0, rotation_matrices, b_hat_rtn)
        finally:
            _shared.clear()

    def test_alpha_fitter_ephemeris_gap_when_rotation_matrices_none(self):
        """SPICE unavailable → EPHEMERIS_GAP, not BAD_FIT."""
        result = self._call_alpha_fit_chunk("l2", None, np.full(3, np.nan))
        self.assertTrue(result["bad_fit_flag"] & int(SwapiL3Flags.EPHEMERIS_GAP))
        self.assertFalse(result["bad_fit_flag"] & int(SwapiL3Flags.BAD_FIT))
        self.assertFalse(result["bad_fit_flag"] & int(SwapiL3Flags.MAG_GAP))

    def test_alpha_fitter_mag_gap_when_b_hat_nan_but_geometry_valid(self):
        """MAG data missing but SPICE OK → MAG_GAP, not BAD_FIT."""
        rm = np.tile(np.eye(3), (5, 1, 1))
        result = self._call_alpha_fit_chunk("l2", rm, np.full(3, np.nan))
        self.assertTrue(result["bad_fit_flag"] & int(SwapiL3Flags.MAG_GAP))
        self.assertFalse(result["bad_fit_flag"] & int(SwapiL3Flags.BAD_FIT))
        self.assertFalse(result["bad_fit_flag"] & int(SwapiL3Flags.EPHEMERIS_GAP))

    def test_alpha_fitter_sets_preliminary_mag_on_ephemeris_gap_when_source_is_l1d(
        self,
    ):
        result = self._call_alpha_fit_chunk("l1d", None, np.full(3, np.nan))
        self.assertTrue(result["bad_fit_flag"] & int(SwapiL3Flags.PRELIMINARY_MAG))

    def test_alpha_fitter_sets_preliminary_mag_on_mag_gap_when_source_is_l1d(self):
        rm = np.tile(np.eye(3), (5, 1, 1))
        result = self._call_alpha_fit_chunk("l1d", rm, np.full(3, np.nan))
        self.assertTrue(result["bad_fit_flag"] & int(SwapiL3Flags.PRELIMINARY_MAG))

    def test_alpha_fitter_omits_preliminary_mag_when_source_is_l2(self):
        result = self._call_alpha_fit_chunk("l2", None, np.full(3, np.nan))
        self.assertFalse(result["bad_fit_flag"] & int(SwapiL3Flags.PRELIMINARY_MAG))
