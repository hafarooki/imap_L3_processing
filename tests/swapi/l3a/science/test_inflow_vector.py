"""Tests for `InflowVector`."""

import tempfile
import unittest
from pathlib import Path

from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from tests.test_helpers import get_test_data_path


class TestInflowVectorConstruction(unittest.TestCase):
    def test_direct_construction_assigns_fields(self):
        v = InflowVector(
            speed_km_per_s=22.0,
            longitude_deg_eclipj2000=180.0,
            latitude_deg_eclipj2000=5.0,
        )
        self.assertEqual(v.speed_km_per_s, 22.0)
        self.assertEqual(v.longitude_deg_eclipj2000, 180.0)
        self.assertEqual(v.latitude_deg_eclipj2000, 5.0)

    def test_dataclass_equality(self):
        a = InflowVector(22.0, 180.0, 5.0)
        b = InflowVector(22.0, 180.0, 5.0)
        c = InflowVector(22.0, 180.0, 5.1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestInflowVectorFromFile(unittest.TestCase):
    def test_from_hydrogen_file(self):
        v = InflowVector.from_file(
            get_test_data_path(
                "swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
            )
        )
        self.assertEqual(v.speed_km_per_s, 22.0)
        self.assertEqual(v.longitude_deg_eclipj2000, 252.2)
        self.assertEqual(v.latitude_deg_eclipj2000, 9.0)

    def test_from_helium_file(self):
        v = InflowVector.from_file(
            get_test_data_path(
                "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
            )
        )
        self.assertEqual(v.speed_km_per_s, 25.4)
        self.assertEqual(v.longitude_deg_eclipj2000, 255.7)
        self.assertEqual(v.latitude_deg_eclipj2000, 5.1)

    def test_raises_on_wrong_number_of_columns(self):
        # The neutral-helium LUT file has many rows of (angle, distance, density)
        # — three columns per row but many rows, so np.loadtxt returns shape (N,3).
        # The squeeze leaves it (N,3), which fails the (3,) assertion.
        with self.assertRaises(AssertionError) as ctx:
            InflowVector.from_file(
                get_test_data_path(
                    "swapi/imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"
                )
            )
        self.assertIn("Failed to parse Inflow Vector", str(ctx.exception))

    def test_handles_extra_whitespace_and_comment_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as tmp:
            tmp.write("# header line ignored by np.loadtxt\n")
            tmp.write("   30.5   100.0   -10.0   \n")
            tmp_path = Path(tmp.name)
        try:
            v = InflowVector.from_file(tmp_path)
            self.assertEqual(v.speed_km_per_s, 30.5)
            self.assertEqual(v.longitude_deg_eclipj2000, 100.0)
            self.assertEqual(v.latitude_deg_eclipj2000, -10.0)
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
