import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
    OA_PLATEAU_AZIMUTH_MAX_DEG,
    OA_PLATEAU_AZIMUTH_MIN_DEG,
    OA_PLATEAU_TRANSMISSION,
    SG_PLATEAU_AZIMUTH_MAX_DEG,
    SG_PLATEAU_TRANSMISSION,
    interpolate_azimuthal_transmission,
    validate_azimuthal_transmission_values,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path

_TRANSMISSION_CSV = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_SPACING_DEG = SwapiResponse.AZIMUTHAL_TRANSMISSION_SPACING_DEG


def _load_real_grid() -> AzimuthalTransmissionGrid:
    # Build the grid the same way `SwapiResponse.from_files` does.
    df = pd.read_csv(_TRANSMISSION_CSV)
    values = df["transmission"].fillna(0).values.astype(float)
    return AzimuthalTransmissionGrid(values=values, spacing=_SPACING_DEG)


class TestInterpolateAzimuthalTransmission(unittest.TestCase):
    """
    `interpolate_azimuthal_transmission` linearly interpolates the azimuthal
    transmission curve `T(|az|)` with two cases of special handling:

    (a) the input azimuth is wrapped into (-180°, 180°] before lookup, and
    (b) the lookup uses `|az|`, so the curve is implicitly mirrored at zero.
    """

    @classmethod
    def setUpClass(cls):
        cls.grid = _load_real_grid()
        cls.values = cls.grid.values
        cls.n = len(cls.values)
        cls.last_az = _SPACING_DEG * (cls.n - 1)

    def test_at_grid_point_returns_csv_value(self):
        """Exact-index lookups: interpolation weights collapse to 1, so the
        result must equal the CSV cell at that index. Sample indices span
        the SG attenuated region, the SG/OA shoulder, the OA plateau, and
        the zero-padded tail."""
        for idx in [0, 10, 199, 299, 500, 1000, self.n - 1]:
            with self.subTest(idx=idx):
                az = idx * _SPACING_DEG
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, float(self.values[idx]))

    def test_linear_interpolation_halfway_between_grid_points(self):
        """Halfway between two adjacent CSV rows must equal their average.
        Pick a pair on the SG→OA rising shoulder so the two endpoints
        differ meaningfully (rather than both sitting on the SG floor)."""
        idx = 250  # az ≈ 25°, mid-shoulder
        az_lower = idx * _SPACING_DEG
        midpoint_az = az_lower + 0.5 * _SPACING_DEG
        expected = 0.5 * (self.values[idx] + self.values[idx + 1])
        got = interpolate_azimuthal_transmission(self.grid, midpoint_az)
        self.assertAlmostEqual(got, float(expected))

    def test_linear_interpolation_off_center_between_grid_points(self):
        """Off-center fractions exercise the interpolation weights, not just
        the symmetric midpoint case: at fraction f between idx and idx+1, the
        result must be `(1-f)·v[idx] + f·v[idx+1]`. Pick the same SG→OA
        shoulder pair so the endpoints differ meaningfully."""
        idx = 250  # az ≈ 25°, mid-shoulder
        az_lower = idx * _SPACING_DEG
        for fraction in [0.25, 0.7, 0.9]:
            with self.subTest(fraction=fraction):
                az = az_lower + fraction * _SPACING_DEG
                expected = (1 - fraction) * self.values[idx] + fraction * self.values[idx + 1]
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, float(expected))

    def test_symmetric_about_zero(self):
        """The lookup uses |az|, so T(-x) must equal T(+x) for any x."""
        for az in [0.37, 12.5, 29.95, 88.123]:
            with self.subTest(az=az):
                positive = interpolate_azimuthal_transmission(self.grid, az)
                negative = interpolate_azimuthal_transmission(self.grid, -az)
                self.assertAlmostEqual(positive, negative)

    def test_clamps_at_and_past_the_last_index(self):
        """At and past the last stored index, both bracketing neighbors are
        clamped to the array end so the result equals the last table value.
        Covers three sub-cases that share the same expected outcome:
          - exactly on the last index (i_lower = n-1, i_upper = n)
          - just past the last index (both indices need clamping)
          - near the upper edge of the wrap interval
        """
        last_value = float(self.values[-1])
        for label, az in [
            ("exactly on last index", self.last_az),
            ("just past last index", self.last_az + 0.07),
            ("near upper wrap edge", 179.95),
        ]:
            with self.subTest(case=label, az=az):
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertEqual(got, last_value)

    def test_wraps_arguments_into_canonical_180_interval(self):
        """The wrap formula `(az + 180) % 360 - 180` maps any real azimuth into
        (-180°, 180°]. Some particular cases:
          - 359° wraps to -1° (|az| = 1°)
          - exact ±180° both wrap to 180°
          - -181° wraps to +179° (|az| = 179°)
          - multi-period inputs (±541°) wrap into the canonical interval
            (541 mod 360 = 181 → wraps to -179° → |az| = 179°)
        """
        idx_at_1deg = int(round(1.0 / _SPACING_DEG))
        expected_at_1deg = float(self.values[idx_at_1deg])
        last_value = float(self.values[-1])
        idx_at_179deg = int(round(179.0 / _SPACING_DEG))
        expected_at_179deg = float(self.values[idx_at_179deg])

        for label, az, expected in [
            ("almost full revolution", 359.0, expected_at_1deg),
            ("exact +180", 180.0, last_value),
            ("exact -180", -180.0, last_value),
            ("just past -180", -181.0, expected_at_179deg),
            ("multi-period positive", 541.0, expected_at_179deg),
            ("multi-period negative", -541.0, expected_at_179deg),
        ]:
            with self.subTest(case=label, az=az):
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, expected)


class TestInterpolateAzimuthalTransmissionWithShorterGrid(unittest.TestCase):
    """`interpolate_azimuthal_transmission` against a synthetic grid that does
    not span the full 0-180 deg range, used to exercise the `i_lower >= n`
    clamp branch that the production-sized grid cannot reach."""

    def test_clamps_lower_index_when_past_grid_extent(self):
        """When the floor of `|az|/spacing` lands past the last array index,
        both bracketing indices clamp to `n-1`. With a tiny synthetic grid
        whose extent (3°) is well below the canonical 180° wrap interval,
        a wrapped azimuth of 10° drives `i_lower` past the end; the lookup
        completes without raising and reads only the final stored value."""
        grid = AzimuthalTransmissionGrid(
            values=np.array([0.1, 0.2, 0.3, 0.4]),
            spacing=1.0,
        )
        # |az| = 10° / spacing 1° = 10, well past last index 3 (n=4).
        got = interpolate_azimuthal_transmission(grid, 10.0)
        self.assertTrue(np.isfinite(got))


class TestRealGridShape(unittest.TestCase):
    """Sanity-check the loaded CSV layout — the interpolator tests above
    assume 0.1° spacing and a fixed table length."""

    def test_csv_layout_matches_production_spacing(self):
        grid = _load_real_grid()
        self.assertEqual(grid.spacing, 0.1)
        # CSV spans 0° to 180° inclusive at 0.1° → 1801 rows.
        self.assertEqual(len(grid.values), 1801)
        # The SG region is attenuated by ≈1/1000 at az=0.
        self.assertAlmostEqual(float(grid.values[0]), 1e-3)


def _build_grid_satisfying_invariants(spacing: float = 0.1, n: int = 1801) -> np.ndarray:
    """Build a synthetic values array that respects the SG-floor and OA-plateau
    invariants the interpolator's short-circuit branches assume. Cells outside
    those two ranges are set to a neutral 0.5 so a test can corrupt a single
    cell in a chosen range without contaminating the other."""
    azimuths = np.arange(n) * spacing
    values = np.full(n, 0.5)
    values[azimuths <= SG_PLATEAU_AZIMUTH_MAX_DEG] = SG_PLATEAU_TRANSMISSION
    in_plateau = (azimuths >= OA_PLATEAU_AZIMUTH_MIN_DEG) & (azimuths <= OA_PLATEAU_AZIMUTH_MAX_DEG)
    values[in_plateau] = OA_PLATEAU_TRANSMISSION
    return values


class TestValidateAzimuthalTransmissionValues(unittest.TestCase):
    """`validate_azimuthal_transmission_values` enforces the constant-value
    invariants the interpolator's short-circuit branches rely on:
      - SG plateau: T(|az|) = SG_PLATEAU_TRANSMISSION
        for all 0 ≤ |az| ≤ SG_PLATEAU_AZIMUTH_MAX_DEG.
      - OA plateau: T(|az|) = OA_PLATEAU_TRANSMISSION for all
        OA_PLATEAU_AZIMUTH_MIN_DEG ≤ |az| ≤ OA_PLATEAU_AZIMUTH_MAX_DEG.
    If the loaded CSV deviates in either range, the interpolator would return
    short-circuit values that contradict the data, so loading must fail loudly.
    """

    def test_production_csv_satisfies_invariants(self):
        grid = _load_real_grid()
        validate_azimuthal_transmission_values(grid.values, grid.spacing)

    def test_synthetic_grid_satisfying_invariants_passes(self):
        values = _build_grid_satisfying_invariants()
        validate_azimuthal_transmission_values(values, 0.1)

    def test_raises_when_sg_plateau_value_is_wrong(self):
        values = _build_grid_satisfying_invariants()
        # Index 50 corresponds to |az| = 5°, well inside the SG plateau range.
        values[50] = 0.5
        with self.assertRaises(ValueError):
            validate_azimuthal_transmission_values(values, 0.1)

    def test_raises_when_oa_plateau_value_is_wrong(self):
        values = _build_grid_satisfying_invariants()
        # Index 500 corresponds to |az| = 50°, inside the OA plateau range.
        values[500] = 0.5
        with self.assertRaises(ValueError):
            validate_azimuthal_transmission_values(values, 0.1)

    def test_raises_when_sg_plateau_endpoint_is_wrong(self):
        # |az| = 9.0° (= SG_PLATEAU_AZIMUTH_MAX_DEG) is inclusive.
        values = _build_grid_satisfying_invariants()
        endpoint_index = int(round(SG_PLATEAU_AZIMUTH_MAX_DEG / 0.1))
        values[endpoint_index] = 0.5
        with self.assertRaises(ValueError):
            validate_azimuthal_transmission_values(values, 0.1)

    def test_raises_when_oa_plateau_endpoint_is_wrong(self):
        # The 115° endpoint is inclusive — corrupt it and validation must trip.
        values = _build_grid_satisfying_invariants()
        endpoint_index = int(round(OA_PLATEAU_AZIMUTH_MAX_DEG / 0.1))
        values[endpoint_index] = 0.5
        with self.assertRaises(ValueError):
            validate_azimuthal_transmission_values(values, 0.1)


class TestSwapiResponseRejectsInvalidTransmissionCsv(unittest.TestCase):
    """`SwapiResponse.from_files` must run `validate_azimuthal_transmission_values`
    so that a bad transmission CSV is rejected at load time rather than
    silently driving the short-circuit branches in the interpolator to lie."""

    def test_from_files_raises_when_csv_violates_invariants(self):
        df = pd.read_csv(get_test_instrument_team_data_path(
            "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
        ))
        # Corrupt a single cell inside the OA plateau range.
        df.loc[500, "transmission"] = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv_path = Path(tmpdir) / "bad_azimuthal_transmission.csv"
            df.to_csv(bad_csv_path, index=False)

            with self.assertRaises(ValueError):
                SwapiResponse.from_files(
                    azimuthal_transmission_path=bad_csv_path,
                    central_effective_area_path=get_test_instrument_team_data_path(
                        "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
                    ),
                    passband_fit_coefficients_path=get_test_instrument_team_data_path(
                        "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
                    ),
                    efficiency_table_path=get_test_data_path(
                        "swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
