import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import spiceypy
from imap_processing.spice.geometry import SpiceFrame, imap_state
from spiceypy import KernelPool, SpiceyError

from imap_l3_processing.predicted_ephemeris_tracker import PredictedEphemerisTracker
from tests.test_helpers import get_integration_test_spice_data_path, get_spice_data_path


class TestPredictedEphemerisTracker(unittest.TestCase):
    def test_predicted_ephemeris_tracker_determines_if_function_needs_predicted_ephemeris(
        self,
    ):
        spice_file_names = [
            "imap_recon_20250925_20260511_v01.bsp",
            "naif0012.tls",
            "imap_sclk_0171.tsc",
            "imap_science_120.tf",
            "imap_pred_od039_20260810_20260921_v01.bsp",
            "de440.bsp",
        ]
        spice_test_paths = [
            str(get_integration_test_spice_data_path(file_name))
            for file_name in spice_file_names
        ]

        with KernelPool(spice_test_paths):
            et_no_predict = spiceypy.datetime2et(datetime(2025, 10, 12))
            et_no_coverage = spiceypy.datetime2et(datetime(2024, 5, 12))
            et_requiring_predict = spiceypy.datetime2et(datetime(2026, 9, 12))

            tracker = PredictedEphemerisTracker()
            state1 = tracker.run(imap_state, et_no_predict, SpiceFrame.ECLIPJ2000)
            self.assertFalse(tracker.used_predict)

            with self.assertRaises(SpiceyError):
                tracker.run(imap_state, et_no_coverage, SpiceFrame.ECLIPJ2000)
            self.assertFalse(tracker.used_predict)

            state2 = tracker.run(
                imap_state, et_requiring_predict, SpiceFrame.ECLIPJ2000
            )
            self.assertTrue(tracker.used_predict)

            expected_state_no_predict = [
                1.40439489e08,
                4.72855667e07,
                8.22993201e04,
                -1.02614632e01,
                2.79858385e01,
                6.54239949e-03,
            ]
            expected_state_requiring_predict = [
                1.46205788e08,
                -2.88643896e07,
                9.66721867e04,
                5.22302215e00,
                2.88895439e01,
                1.19768410e-02,
            ]

            np.testing.assert_allclose(expected_state_no_predict, state1)
            np.testing.assert_allclose(expected_state_requiring_predict, state2)

    def test_predicted_ephemeris_tracker_skips_check_if_predict_already_used(self):
        predict_kernel = get_integration_test_spice_data_path(
            "imap_pred_od039_20260810_20260921_v01.bsp"
        )
        with KernelPool([str(predict_kernel)]):

            def requires_predict():
                if spiceypy.ktotal("ALL") == 0:
                    raise SpiceyError()
                return "ok"

            mock_spice_function_1 = Mock(side_effect=requires_predict)
            mock_spice_function_2 = Mock(side_effect=requires_predict)
            tracker = PredictedEphemerisTracker()
            result_1 = tracker.run(mock_spice_function_1)
            self.assertTrue(tracker.used_predict)
            self.assertEqual("ok", result_1)
            result_2 = tracker.run(mock_spice_function_2)
            self.assertEqual("ok", result_2)

            self.assertEqual(2, mock_spice_function_1.call_count)
            self.assertEqual(1, mock_spice_function_2.call_count)


    def test_predicted_ephemeris_tracker_skips_retry_if_no_predict_kernel(self):
        other_kernel = get_integration_test_spice_data_path(
            "imap_2025_105_2026_105_01.ah.bc"
        )
        with KernelPool([str(other_kernel)]):
            mock_spice_function_1 = Mock(side_effect=SpiceyError())
            tracker = PredictedEphemerisTracker()
            with self.assertRaises(SpiceyError):
                tracker.run(mock_spice_function_1)
            self.assertEqual(1, mock_spice_function_1.call_count)

    def test_predicted_ephemeris_tracker_allows_unknown_kernel_names(self):
        kernels = [str(x) for x in get_spice_data_path("").iterdir()]

        with KernelPool(kernels):
            tracker = PredictedEphemerisTracker()
            tracker.run(spiceypy.spkezr, "IMAP", spiceypy.datetime2et(datetime(2025,10, 10)), "ECLIPJ2000", "NONE", "SUN")