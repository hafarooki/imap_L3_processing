"""Pin SWAPI descriptor strings.

Descriptors are used as `source/descriptor` keys against the SDC processing
input collection — typos there silently cause `get_file_paths` to return
empty lists, which downstream surfaces as `IndexError` in `fetch_dependencies`.
These tests catch accidental rename/typo regressions cheaply.
"""

import unittest

from imap_l3_processing.swapi import descriptors


_PUBLIC_DESCRIPTORS = [
    descriptors.SWAPI_L2_DESCRIPTOR,
    descriptors.ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR,
    descriptors.GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR,
    descriptors.GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
    descriptors.EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR,
    descriptors.INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR,
    descriptors.DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR,
    descriptors.HYDROGEN_INFLOW_VECTOR_DESCRIPTOR,
    descriptors.HELIUM_INFLOW_VECTOR_DESCRIPTOR,
    descriptors.AZIMUTHAL_TRANSMISSION_DESCRIPTOR,
    descriptors.CENTRAL_EFFECTIVE_AREA_DESCRIPTOR,
    descriptors.PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR,
    descriptors.SWAPI_L3A_ALPHA_SW_DESCRIPTOR,
    descriptors.MAG_RTN_L1D_DESCRIPTOR,
]


class TestDescriptors(unittest.TestCase):
    def test_descriptors_are_nonempty_strings_without_whitespace(self):
        for d in _PUBLIC_DESCRIPTORS:
            with self.subTest(descriptor=d):
                self.assertIsInstance(d, str)
                self.assertGreater(len(d), 0)
                self.assertEqual(d, d.strip())
                self.assertNotIn(" ", d)

    def test_descriptors_are_unique(self):
        self.assertEqual(len(_PUBLIC_DESCRIPTORS), len(set(_PUBLIC_DESCRIPTORS)))

    def test_known_descriptor_strings(self):
        # Pin the wire-format strings: changing these breaks SDC dependency lookup.
        self.assertEqual(descriptors.SWAPI_L2_DESCRIPTOR, "sci")
        self.assertEqual(descriptors.MAG_RTN_L1D_DESCRIPTOR, "norm-rtn")
        self.assertEqual(descriptors.SWAPI_L3A_ALPHA_SW_DESCRIPTOR, "alpha-sw")
        self.assertEqual(
            descriptors.GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
            "energy-gf-pui-lut",
        )
        self.assertEqual(
            descriptors.GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR, "energy-gf-sw-lut"
        )
        self.assertEqual(
            descriptors.EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, "efficiency-lut"
        )


if __name__ == "__main__":
    unittest.main()
