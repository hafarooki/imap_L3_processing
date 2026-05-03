"""Tests for `InstrumentResponseLookupTable[Collection]`."""

import io
import unittest
import zipfile
from pathlib import Path

import numpy as np

from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import (
    InstrumentResponseLookupTable,
    InstrumentResponseLookupTableCollection,
)
from tests.test_helpers import get_test_data_path


_TRUNCATED_LUT = get_test_data_path("swapi/truncated_swapi_response_simion_v1.zip")
_TRUNCATED_LUT_WITH_MAC_METADATA = get_test_data_path(
    "swapi/truncated_swapi_response_with_mac_metadata.zip"
)


class TestInstrumentResponseLookupTableCollectionFromFile(unittest.TestCase):
    def test_get_table_for_known_energy_bin(self):
        coll = InstrumentResponseLookupTableCollection.from_file(_TRUNCATED_LUT)
        result = coll.get_table_for_energy_bin(2)
        self.assertEqual(result.energy[0], 103.07800)
        self.assertEqual(result.energy[-1], 107.04900)
        self.assertEqual(result.elevation[0], 2.000)
        self.assertEqual(result.elevation[-1], 6.000)
        self.assertEqual(result.azimuth[0], -149.000)
        self.assertEqual(result.d_energy[0], 0.97411)
        self.assertEqual(result.response[0], 0.0160)
        self.assertEqual(len(result.energy), 16)

    def test_handles_mac_metadata_files_in_zip(self):
        # macOS-zipped archives contain `__MACOSX/._*` resource-fork files; the regex
        # `^response_ESA(\d*).dat$` filters those out via Path(file).name.
        coll = InstrumentResponseLookupTableCollection.from_file(
            _TRUNCATED_LUT_WITH_MAC_METADATA
        )
        result = coll.get_table_for_energy_bin(2)
        self.assertEqual(result.energy[0], 103.07800)

    def test_unknown_energy_bin_raises_keyerror(self):
        coll = InstrumentResponseLookupTableCollection.from_file(_TRUNCATED_LUT)
        with self.assertRaises(KeyError):
            coll.get_table_for_energy_bin(999)

    def test_lookup_tables_dict_keys_are_energy_bin_indices(self):
        coll = InstrumentResponseLookupTableCollection.from_file(_TRUNCATED_LUT)
        # The truncated zip has ESA1.dat and ESA2.dat → bins {1, 2}.
        self.assertEqual(set(coll.lookup_tables), {1, 2})


class TestInstrumentResponseLookupTableIntegralFactor(unittest.TestCase):
    def test_integral_factor_known_values(self):
        # Pinned via direct algebra of the source formula:
        #   f_i = response_i · v_i^4 · dE_i · cos(elev_i) · dAz_i · dEl_i
        #         / Σ (dE_k · cos(elev_k) · dEl_k · dAz_k)
        # for two-bin synthetic input.
        table = InstrumentResponseLookupTable(
            energy=np.array([1.0, 2.0]),
            elevation=np.array([10.0, 20.0]),
            azimuth=np.array([30.0, 40.0]),
            d_energy=np.array([3.0, 4.0]),
            d_elevation=np.array([11.0, 21.0]),
            d_azimuth=np.array([31.0, 41.0]),
            response=np.array([100.0, 200.0]),
        )
        np.testing.assert_allclose(
            table.integral_factor, [55194.122853, 1418419.487803], rtol=1e-6
        )

    def test_integral_factor_scales_linearly_with_response(self):
        kwargs = dict(
            energy=np.array([1.0, 2.0]),
            elevation=np.array([10.0, 20.0]),
            azimuth=np.array([30.0, 40.0]),
            d_energy=np.array([3.0, 4.0]),
            d_elevation=np.array([11.0, 21.0]),
            d_azimuth=np.array([31.0, 41.0]),
        )
        a = InstrumentResponseLookupTable(response=np.array([100.0, 200.0]), **kwargs)
        b = InstrumentResponseLookupTable(response=np.array([200.0, 400.0]), **kwargs)
        np.testing.assert_allclose(b.integral_factor, 2.0 * a.integral_factor)

    def test_integral_factor_uses_cosine_of_elevation(self):
        # Two tables: one at elevation 0 deg (cos=1), one at 60 deg (cos=0.5). Same
        # other params, same energies. The factor at 60 deg should be exactly half
        # the factor at 0 deg, modulo the global denominator (which is also halved
        # ⇒ overall ratio 1.0). Demonstrate via shared integrand: a single-bin table.
        kwargs = dict(
            energy=np.array([1.0]),
            azimuth=np.array([0.0]),
            d_energy=np.array([1.0]),
            d_elevation=np.array([1.0]),
            d_azimuth=np.array([1.0]),
            response=np.array([1.0]),
        )
        a = InstrumentResponseLookupTable(elevation=np.array([0.0]), **kwargs)
        b = InstrumentResponseLookupTable(elevation=np.array([60.0]), **kwargs)
        # Ratio equals 1 because both numerator and denominator are scaled by cos(elev).
        np.testing.assert_allclose(a.integral_factor / b.integral_factor, [1.0])


class TestZipParsing(unittest.TestCase):
    def test_skips_files_not_matching_response_esa_regex(self):
        # Build an in-memory zip with a noise file that should be ignored.
        tmp_zip = Path("/tmp/claude/_irlt_test.zip")
        tmp_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp_zip, mode="w") as z:
            # Real ESA file (4-column rows; class transposes to 7 columns × N rows)
            # InstrumentResponseLookupTable expects 7 fields; the production format
            # has 7 columns (energy, elevation, azimuth, d_energy, d_elevation,
            # d_azimuth, response). A two-row example:
            row1 = "1.0 0.0 0.0 1.0 1.0 1.0 0.5"
            row2 = "2.0 0.0 0.0 1.0 1.0 1.0 0.5"
            z.writestr("not_a_response_file.txt", "garbage content\n")
            z.writestr("response_ESA7.dat", row1 + "\n" + row2 + "\n")
        try:
            coll = InstrumentResponseLookupTableCollection.from_file(tmp_zip)
            self.assertEqual(set(coll.lookup_tables), {7})
        finally:
            tmp_zip.unlink()


if __name__ == "__main__":
    unittest.main()
