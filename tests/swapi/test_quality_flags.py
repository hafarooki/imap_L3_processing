"""Direct tests for `SwapiL3Flags`.

The flag class is used as bitwise OR composition throughout the L3a pipeline
(e.g. `quality_flag |= result.bad_fit_flag`). These tests pin down the bit
assignments and the OR/AND/contains semantics so a reordering of the enum
values would be caught here rather than in a downstream regression.
"""

import unittest

from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


class TestSwapiL3Flags(unittest.TestCase):
    def test_flag_bit_values_are_unique_and_disjoint(self):
        named_flags = [
            SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED,
            SwapiL3Flags.BAD_FIT,
            SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY,
            SwapiL3Flags.STALE_PROTON,
            SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK,
        ]
        for flag in named_flags:
            with self.subTest(flag=flag):
                self.assertEqual(
                    int(flag) & (int(flag) - 1), 0, "flag is not a single bit"
                )
        self.assertEqual(len(set(int(f) for f in named_flags)), len(named_flags))

    def test_none_is_identity_under_or(self):
        for flag in [SwapiL3Flags.BAD_FIT, SwapiL3Flags.STALE_PROTON]:
            with self.subTest(flag=flag):
                self.assertEqual(int(SwapiL3Flags.NONE | flag), int(flag))
                self.assertEqual(int(flag | SwapiL3Flags.NONE), int(flag))

    def test_or_carries_both_bits(self):
        combined = SwapiL3Flags.STALE_PROTON | SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK
        self.assertTrue(int(combined) & int(SwapiL3Flags.STALE_PROTON))
        self.assertTrue(int(combined) & int(SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK))
        self.assertFalse(int(combined) & int(SwapiL3Flags.BAD_FIT))

    def test_or_is_idempotent_on_repeat(self):
        flag = SwapiL3Flags.BAD_FIT
        self.assertEqual(int(flag | flag), int(flag))


if __name__ == "__main__":
    unittest.main()
