import unittest
from datetime import datetime

from scipy.constants import milli
from spacepy import pycdf

from imap_l3_processing.swapi.response.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.species import Species
from tests.test_helpers import get_test_data_path


class TestEfficiencyCalibrationTable(unittest.TestCase):
    def test_relative_efficiency_for_protons(self):
        """Each queried time returns the proton efficiency from the most recent row that precedes it, unscaled."""
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1)), Species.PROTON), 0.1)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2013, month=10, day=1)), Species.PROTON),
            0.1)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2014, month=10, day=3)), Species.PROTON),
            0.09)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=1)), Species.PROTON),
            0.09)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=3)), Species.PROTON),
            0.0882)

    def test_relative_efficiency_for_helium(self):
        """A helium species reads the helium column by the same most-recent-preceding-row rule as protons."""
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1)), Species.ALPHA), 0.9)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2013, month=10, day=1)), Species.ALPHA),
            0.9)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2014, month=10, day=3)), Species.ALPHA),
            0.95)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=1)), Species.ALPHA),
            0.95)
        self.assertEqual(
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=3)), Species.ALPHA),
            0.99)

    def test_relative_efficiency_handles_float_input(self):
        """A TT2000 passed as a float rather than an int resolves to the same row."""
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        time_as_float = float(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1)))
        self.assertEqual(efficiency_table.relative_efficiency(time_as_float, Species.ALPHA), 0.9)
        self.assertEqual(efficiency_table.relative_efficiency(time_as_float, Species.PROTON), 0.1)

    def test_raises_exception_if_ask_for_time_before_the_table_starts(self):
        """A time earlier than every row raises rather than falling back to the first row."""
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        with self.assertRaises(ValueError) as content_manager:
            time = datetime(year=1999, month=1, day=4)
            efficiency_table.relative_efficiency(pycdf.lib.datetime_to_tt2000(time), Species.PROTON)

        self.assertEqual((f"No efficiency data for {time}",), content_manager.exception.args)

    def test_loads_calibration_table_with_single_row(self):
        """A one-row table still parses and serves both species for any later time."""
        calibration_table_path = get_test_data_path(
            "swapi/imap_swapi_efficiency-lut-single-row-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        # file has one row, with 0.1 for H, 0.9 for He.

        time_after_row = pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1))
        self.assertEqual(efficiency_table.relative_efficiency(time_after_row, Species.PROTON), 0.1)
        self.assertEqual(efficiency_table.relative_efficiency(time_after_row, Species.ALPHA), 0.9)
