import unittest

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    REGION_OPEN_APERTURE_NEG,
    REGION_OPEN_APERTURE_POS,
    REGION_SUNGLASSES,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.integration_limits import (
    SPEED_HALF_WIDTH_VTH,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
    bulk_speed,
    thermal_speed,
    thermal_speed_to_temperature,
)
from imap_l3_processing.swapi.response.passband_grid import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import (
    NOMINAL_SWAPI_TO_RTN_ROTATION,
    NOMINAL_TEST_EPOCH_TT2000,
    load_swapi_response,
    proton_params,
)

# --- module-level fixtures: real instrument-team CSVs and one warmed grid ----

# A 1 keV proton ESA voltage gives a comfortably typical central speed
# (~437 km/s) and a passband elevation range that contains 0.
_ESA_VOLTAGE = 1000.0 / SWAPI_K_FACTOR

# Region azimuth bands fixed by the doc.
_SG_AZIMUTH_LO_DEG = -20.0
_SG_AZIMUTH_HI_DEG = +20.0
_OA_AZIMUTH_INNER_DEG = 20.0
_OA_AZIMUTH_OUTER_DEG = 150.0

_BULK_SPEED_KM_S = 450.0


class TestSpeedWindowMissesPassband(unittest.TestCase):
    """Tests for `speed_window_misses_passband` — the cheap pre-check that returns True only when `bulk_speed ± k·vth` is fully disjoint from the widest possible passband speed range (`central_speed · [0.9, 1.1]`)."""

    @classmethod
    def setUpClass(cls):
        cls.response = load_swapi_response(warm_cache_voltages=[_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(
            NOMINAL_TEST_EPOCH_TT2000, _ESA_VOLTAGE, Species.PROTON
        )

    def test_returns_false_when_bulk_speed_sits_in_passband(self):
        """A 450 km/s bulk at the 1 keV step sits well inside the passband envelope, so no miss is reported."""
        sw = proton_params()
        self.assertFalse(speed_window_misses_passband(sw, self.response_grid))

    def test_returns_true_when_bulk_speed_above_passband(self):
        """A 1500 km/s bulk puts the entire `bulk ± k·σ` window above the upper passband edge, so a miss is reported."""
        sw = proton_params(velocity_rtn=(0.0, -1500.0, 0.0))
        self.assertTrue(speed_window_misses_passband(sw, self.response_grid))

    def test_returns_true_when_bulk_speed_below_passband(self):
        """A 100 km/s bulk puts the entire `bulk ± k·σ` window below the lower passband edge, so a miss is reported."""
        sw = proton_params(velocity_rtn=(0.0, -100.0, 0.0))
        self.assertTrue(speed_window_misses_passband(sw, self.response_grid))

    def test_window_overlaps_passband_when_k_sigma_exceeds_offset_above_edge(self):
        """With bulk 30 km/s above the passband envelope edge and `k·σ = 50 km/s`, the window still reaches into the passband and no miss is reported — pinning `SPEED_HALF_WIDTH_VTH` as the scaling constant when paired with the detach test below."""
        v_passband_hi = self.response_grid.central_speed * 1.1
        bulk = v_passband_hi + 30.0
        sigma_overlap = 50.0 / SPEED_HALF_WIDTH_VTH
        T_overlap = thermal_speed_to_temperature(sigma_overlap, PROTON_MASS_KG)
        sw_overlap = proton_params(
            velocity_rtn=(0.0, -bulk, 0.0), temperature=T_overlap
        )
        self.assertFalse(speed_window_misses_passband(sw_overlap, self.response_grid))

    def test_window_detaches_from_passband_when_k_sigma_falls_below_offset(self):
        """With the same 30 km/s offset above the edge but `k·σ = 5 km/s`, the window stays clear of the passband and a miss is reported — the companion case to the overlap test."""
        v_passband_hi = self.response_grid.central_speed * 1.1
        bulk = v_passband_hi + 30.0
        sigma_detached = 5.0 / SPEED_HALF_WIDTH_VTH
        T_detached = thermal_speed_to_temperature(sigma_detached, PROTON_MASS_KG)
        sw_detached = proton_params(
            velocity_rtn=(0.0, -bulk, 0.0), temperature=T_detached
        )
        self.assertTrue(speed_window_misses_passband(sw_detached, self.response_grid))


class TestGetAngularQuadratureSunglassesRegion(unittest.TestCase):
    """Tests for `get_angular_quadrature` on the SG region — the 21×21 GL rectangle clamped to the SG elevation range and `[-20°, +20°]` azimuth band."""

    @classmethod
    def setUpClass(cls):
        cls.response = load_swapi_response(warm_cache_voltages=[_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(
            NOMINAL_TEST_EPOCH_TT2000, _ESA_VOLTAGE, Species.PROTON
        )
        cls.identity = NOMINAL_SWAPI_TO_RTN_ROTATION

    def test_sg_region_is_active_for_sun_pointed_bulk(self):
        """A boresight-aligned bulk with a narrow Maxwellian falls inside the SG band, so the region is active and a quadrature is returned."""
        sw = proton_params()
        skip, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_sg_azimuth_window_is_inside_minus_20_to_plus_20_degrees(self):
        """For a boresight-aligned bulk, every azimuth GL node lies inside the SG `[-20°, +20°]` band."""
        sw = proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertGreaterEqual(quadrature.azimuth_points.min(), _SG_AZIMUTH_LO_DEG)
        self.assertLessEqual(quadrature.azimuth_points.max(), _SG_AZIMUTH_HI_DEG)

    def test_sg_azimuth_window_clamps_to_plus_minus_20_for_broad_distribution(self):
        """A very hot plasma's 180° angular extent forces the azimuth window to clamp exactly to the `[-20°, +20°]` SG band, with the outermost GL nodes hugging both edges."""
        sw = proton_params(temperature=1e10)
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        # GL nodes lie strictly inside the band.
        np.testing.assert_array_less(
            quadrature.azimuth_points, _SG_AZIMUTH_HI_DEG + 1e-9
        )
        np.testing.assert_array_less(
            _SG_AZIMUTH_LO_DEG - 1e-9, quadrature.azimuth_points
        )
        # The outermost 21-node GL node sits ≈0.125° inside the edge of a 40°
        # window (analytically `20°·(1 - x_GL[-1])` ≈ 0.125°). 1.5° is a loose
        # cushion above that — only meant to catch the case where the window
        # collapses or shifts away from the edge entirely.
        self.assertGreater(quadrature.azimuth_points.max(), _SG_AZIMUTH_HI_DEG - 1.5)
        self.assertLess(quadrature.azimuth_points.min(), _SG_AZIMUTH_LO_DEG + 1.5)

    def test_sg_elevation_window_clamps_into_passband_elevation_range(self):
        """A hot plasma's saturated angular extent is clipped by the SG passband elevation range, so every elevation node lies inside that range."""
        sw = proton_params(temperature=1e10)
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        sg_el_lo, sg_el_hi = self.response_grid.sg_passband.elevation_range
        self.assertGreaterEqual(quadrature.elevation_points.min(), sg_el_lo)
        self.assertLessEqual(quadrature.elevation_points.max(), sg_el_hi)

    def test_sg_quadrature_has_21_nodes_in_each_angular_dimension(self):
        """The SG angular quadrature is built with the doc-prescribed (Nθ, Nφ) = (21, 21) Gauss-Legendre node counts."""
        sw = proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertEqual(quadrature.elevation_points.shape, (21,))
        self.assertEqual(quadrature.azimuth_points.shape, (21,))

    def test_sin_cos_caches_match_their_angle_arrays(self):
        """The precomputed sin/cos arrays on the quadrature exactly match `sin`/`cos` of the corresponding angle (in radians) at every node."""
        sw = proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        np.testing.assert_allclose(
            quadrature.sin_elevation, np.sin(np.radians(quadrature.elevation_points))
        )
        np.testing.assert_allclose(
            quadrature.cos_elevation, np.cos(np.radians(quadrature.elevation_points))
        )
        np.testing.assert_allclose(
            quadrature.sin_azimuth, np.sin(np.radians(quadrature.azimuth_points))
        )
        np.testing.assert_allclose(
            quadrature.cos_azimuth, np.cos(np.radians(quadrature.azimuth_points))
        )

    def test_sg_region_skipped_when_bulk_far_from_sg_band(self):
        """A bulk pointed at azimuth ≈ 90° is 70° away from the SG band and a narrow Maxwellian cannot bridge the gap, so the SG region is skipped."""
        sw = proton_params(velocity_rtn=(-_BULK_SPEED_KM_S, 0.0, 0.0))
        skip, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetAngularQuadratureOpenAperturePositive(unittest.TestCase):
    """Tests for `get_angular_quadrature` on the OA+ region (az ∈ [+20°, +150°]) with a bulk pointed at az ≈ +90°."""

    @classmethod
    def setUpClass(cls):
        cls.response = load_swapi_response(warm_cache_voltages=[_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(
            NOMINAL_TEST_EPOCH_TT2000, _ESA_VOLTAGE, Species.PROTON
        )
        cls.identity = np.eye(3)
        # Bulk pointed at az=+90°: identity rotation makes v_inst = (-v, 0, 0),
        # giving azimuth = atan2(v, 0) = 90° and elevation = 0°.
        cls.sw_bulk_at_90 = proton_params(
            velocity_rtn=(-_BULK_SPEED_KM_S, 0.0, 0.0)
        )

    def test_oa_positive_region_is_active_for_bulk_at_90_deg_azimuth(self):
        """A bulk at az ≈ +90° lies inside the OA+ band, so the region is active and a quadrature is returned."""
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_oa_positive_azimuth_window_is_inside_20_to_150_degrees(self):
        """For a bulk at az ≈ +90°, every OA+ azimuth node lies strictly inside the `[+20°, +150°]` OA+ band."""
        _, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        self.assertGreaterEqual(
            quadrature.azimuth_points.min(), _OA_AZIMUTH_INNER_DEG
        )
        self.assertLessEqual(
            quadrature.azimuth_points.max(), _OA_AZIMUTH_OUTER_DEG
        )

    def test_oa_negative_region_is_skipped_for_bulk_at_plus_90_deg_azimuth(self):
        """A bulk at az ≈ +90° is too far from the OA- band for a narrow Maxwellian to reach, so the OA- region is skipped."""
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetAngularQuadratureOpenApertureNegative(unittest.TestCase):
    """Tests for `get_angular_quadrature` on the OA- region (az ∈ [-150°, -20°]) with a bulk pointed at az ≈ -90°."""

    @classmethod
    def setUpClass(cls):
        cls.response = load_swapi_response(warm_cache_voltages=[_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(
            NOMINAL_TEST_EPOCH_TT2000, _ESA_VOLTAGE, Species.PROTON
        )
        cls.identity = np.eye(3)
        # Bulk at az=-90°: v_inst = (+v, 0, 0) ⇒ az = atan2(-v, 0) = -90°.
        cls.sw_bulk_at_minus_90 = proton_params(
            velocity_rtn=(+_BULK_SPEED_KM_S, 0.0, 0.0)
        )

    def test_oa_negative_region_is_active_for_bulk_at_minus_90_deg_azimuth(self):
        """A bulk at az ≈ -90° lies inside the OA- band, so the region is active and a quadrature is returned."""
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_oa_negative_azimuth_window_is_inside_minus_150_to_minus_20(self):
        """For a bulk at az ≈ -90°, every OA- azimuth node lies strictly inside the `[-150°, -20°]` OA- band."""
        _, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        self.assertGreaterEqual(
            quadrature.azimuth_points.min(), -_OA_AZIMUTH_OUTER_DEG
        )
        self.assertLessEqual(
            quadrature.azimuth_points.max(), -_OA_AZIMUTH_INNER_DEG
        )

    def test_oa_positive_region_is_skipped_for_bulk_at_minus_90_deg_azimuth(self):
        """A bulk at az ≈ -90° is too far from the OA+ band for a narrow Maxwellian to reach, so the OA+ region is skipped."""
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetSpeedQuadrature(unittest.TestCase):
    """Tests for `get_speed_quadrature` — the 15-node Gauss-Legendre quadrature over the intersection of `bulk_speed ± k·vth` and the per-elevation passband speed range."""

    @classmethod
    def setUpClass(cls):
        cls.response = load_swapi_response(warm_cache_voltages=[_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(
            NOMINAL_TEST_EPOCH_TT2000, _ESA_VOLTAGE, Species.PROTON
        )

    def test_speed_window_lies_inside_per_elevation_passband(self):
        """For warm plasma the passband binds the window, so every GL node lies inside `[r_min(θ)·v0, r_max(θ)·v0]` at the given elevation."""
        sw = proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        ratio_lo, ratio_hi = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        passband_lo = self.response_grid.central_speed * ratio_lo
        passband_hi = self.response_grid.central_speed * ratio_hi
        self.assertGreaterEqual(speed_quadrature.points.min(), passband_lo)
        self.assertLessEqual(speed_quadrature.points.max(), passband_hi)

    def test_cold_plasma_window_lies_inside_bulk_plus_minus_k_vth(self):
        """For cold plasma (T = 480 K) at the passband center, the `bulk ± k·σ` interval is the binding constraint and every GL node sits inside it."""
        sw = proton_params(
            velocity_rtn=(0.0, -self.response_grid.central_speed, 0.0),
            temperature=480.0,
        )
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
        window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma
        self.assertGreaterEqual(speed_quadrature.points.min(), window_lo)
        self.assertLessEqual(speed_quadrature.points.max(), window_hi)

    def test_cold_plasma_outer_gl_nodes_sit_close_to_window_edges(self):
        """For the same cold-plasma case, the outermost GL nodes sit within ~2 km/s of `bulk ± k·σ`, pinning the window endpoints to `bulk ± k·σ` rather than some narrower subset."""
        sw = proton_params(
            velocity_rtn=(0.0, -self.response_grid.central_speed, 0.0),
            temperature=480.0,
        )
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
        window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma
        self.assertLess(window_hi - speed_quadrature.points.max(), 2.0)
        self.assertLess(speed_quadrature.points.min() - window_lo, 2.0)

    def test_speed_quadrature_arrays_are_all_15_long(self):
        """Each array on the returned `SpeedQuadrature` (points, weights, `speed_cubed_times_passband`) has the doc-prescribed Nv = 15 nodes."""
        sw = proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        self.assertEqual(speed_quadrature.points.shape, (15,))
        self.assertEqual(speed_quadrature.weights.shape, (15,))
        self.assertEqual(speed_quadrature.speed_cubed_times_passband.shape, (15,))

    def test_returns_skip_when_bulk_speed_window_does_not_overlap_passband(self):
        """A bulk so far above the passband that `bulk - k·σ` exceeds the upper edge yields skip=True and a `None` quadrature."""
        sw = proton_params(
            velocity_rtn=(0.0, -1500.0, 0.0), temperature=10_000.0
        )
        skip, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        self.assertTrue(skip)
        self.assertIsNone(speed_quadrature)

    def test_uses_region_specific_passband(self):
        """With a hot plasma so the passband binds the window in both regions, the SG and OA quadratures land on their own region's passband edge and the two windows are distinct."""
        sw = proton_params(temperature=1_000_000.0)
        _, sg_quad = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        _, oa_quad = get_speed_quadrature(
            sw, self.response_grid, REGION_OPEN_APERTURE_POS, 0.0
        )
        sg_ratio_lo, _ = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        oa_ratio_lo, _ = speed_ratio_range_at_elevation(
            self.response_grid.oa_passband, 0.0
        )
        sg_lo = self.response_grid.central_speed * sg_ratio_lo
        oa_lo = self.response_grid.central_speed * oa_ratio_lo
        # Each quadrature's lower edge sits at its own region's passband edge.
        self.assertGreaterEqual(sg_quad.points.min(), sg_lo)
        self.assertGreaterEqual(oa_quad.points.min(), oa_lo)
        # And the two windows are actually distinct — the SG and OA passbands
        # have different speed-ratio bounds, so a region mix-up would be
        # silently caught here.
        self.assertNotEqual(sg_quad.points.min(), oa_quad.points.min())

    def test_speed_cubed_times_passband_is_v3_times_normalized_passband_at_each_node(self):
        """The `speed_cubed_times_passband` array equals `v³ · P(θ, v/v0) / P(0, 1)` reconstructed from `interpolate_passband` at every GL node."""
        sw = proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        on_axis_peak = interpolate_passband(
            self.response_grid.sg_passband, 0.0, 1.0
        )
        expected = np.array(
            [
                v**3
                * interpolate_passband(
                    self.response_grid.sg_passband,
                    0.0,
                    v / self.response_grid.central_speed,
                )
                / on_axis_peak
                for v in speed_quadrature.points
            ]
        )
        np.testing.assert_allclose(
            speed_quadrature.speed_cubed_times_passband, expected
        )


if __name__ == "__main__":
    unittest.main()
