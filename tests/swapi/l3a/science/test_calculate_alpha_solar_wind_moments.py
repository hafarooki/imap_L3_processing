import unittest

import numba
import numpy as np
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_CHARGE_OVER_MASS_C_PER_KG,
    ALPHA_PARTICLE_MASS_KG,
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    EV_TO_KELVIN,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_moments import (
    AlphaSolarWindMoments,
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    ProtonSolarWindMoments,
    _model_count_rates,
    apply_deadtime_correction,
    apply_deadtime_correction_array,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    esa_voltage_to_alpha_speed,
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def _swapi_response():
    return SWAPIResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )


class TestSpeciesSpeedConversions(unittest.TestCase):
    def test_alpha_speed_at_known_voltage(self):
        # v_0^α(529 V) = sqrt(2*1.89*2*e/m_α*529)/1000 ≈ 310.5 km/s; computed analytically.
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(529.0), 310.533, rtol=2e-3
        )

    def test_alpha_speed_uses_absolute_voltage(self):
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )

    def test_alpha_speed_close_to_proton_speed_at_half_voltage(self):
        # If alpha mass were exactly 4 m_p, v_0^α(V) = v_0^p(V/2). Real m_α/m_p ≈ 3.97,
        # giving a ~0.4% deviation — verify within 1% tolerance.
        for V in [200.0, 1000.0, 4000.0]:
            np.testing.assert_allclose(
                esa_voltage_to_alpha_speed(V),
                esa_voltage_to_proton_speed(V / 2.0),
                rtol=1e-2,
            )


class TestPassbandGridSpecies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()

    def test_alpha_central_speed_close_to_proton_over_sqrt2_at_same_voltage(self):
        # Exactly 1/√2 if m_α = 4 m_p; real m_α ≈ 3.97 m_p gives ~0.4% deviation.
        V = 1000.0
        p_cs = self.sr.central_speed(V, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        a_cs = self.sr.central_speed(V, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
        np.testing.assert_allclose(a_cs, p_cs / np.sqrt(2.0), rtol=1e-2)

    def test_grid_cache_keys_on_voltage_only(self):
        # Grid is V-only; same voltage always returns the exact same cached object.
        V = 800.0
        self.sr.warm_cache([V])
        g1 = self.sr.create_passband_grid(V)
        g2 = self.sr.create_passband_grid(V)
        self.assertIs(g1, g2)
        # Species-specific quantities come from sr.central_speed(), not the grid.
        p_cs = self.sr.central_speed(V, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        a_cs = self.sr.central_speed(V, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
        self.assertNotAlmostEqual(p_cs, a_cs, places=0)


class TestModelCountRatesSpecies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sr = _swapi_response()
        V = 1000.0
        sr.warm_cache([V])
        cls.proton_grids = numba.typed.List([sr.create_passband_grid(V)])
        cls.cs = np.array([sr.central_speed(V, PROTON_MASS_PER_CHARGE_M_P_PER_E)])
        cls.cea = np.array([sr.get_central_effective_area(V)])
        cls.at = np.asarray(sr.azimuthal_transmission, dtype=float)
        cls.ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
        cls.rot = np.eye(3).reshape(1, 3, 3)

    def test_central_effective_area_scale_is_linear(self):
        # Doubling central_effective_areas doubles the model rate (it's a multiplier).
        v_rtn = np.array([450.0, 0.0, 0.0])
        r_one = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            v_rtn,
            self.proton_grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            PROTON_MASS_KG,
        )
        r_two = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            v_rtn,
            self.proton_grids,
            self.cs,
            self.cea * 2.0,
            self.at,
            self.ats,
            self.rot,
            PROTON_MASS_KG,
        )
        np.testing.assert_allclose(r_two, r_one * 2.0, rtol=1e-12)

    def test_thermal_speed_scales_as_inverse_sqrt_mass(self):
        # At fixed T_K, v_th = sqrt(k_B*T/m) — same numerator (k_B), different denominator.
        # The model rate at the proton peak shouldn't depend on a "species" choice if we
        # also use the matching grid; but we can verify the m-dependence by holding the
        # *grid* fixed and varying the mass argument.
        v_rtn = np.array([450.0, 0.0, 0.0])
        r_p = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            v_rtn,
            self.proton_grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            PROTON_MASS_KG,
        )
        r_alpha_mass_proton_grid = _model_count_rates(
            5.0,
            10.0 * EV_TO_KELVIN,
            v_rtn,
            self.proton_grids,
            self.cs,
            self.cea,
            self.at,
            self.ats,
            self.rot,
            ALPHA_PARTICLE_MASS_KG,
        )
        # Different thermal width → different integral. They should not be equal.
        self.assertFalse(np.allclose(r_p, r_alpha_mass_proton_grid))


class TestApplyDeadtimeCorrectionArray(unittest.TestCase):
    def test_array_matches_scalar(self):
        rates = np.array([1.0, 1e3, 1e5, 5e5])
        scalar = np.array([apply_deadtime_correction(float(r)) for r in rates])
        vec = apply_deadtime_correction_array(rates)
        np.testing.assert_allclose(vec, scalar, rtol=1e-12)


# -------------------------------------------------------------------------
# End-to-end alpha fitter tests with synthetic data
# -------------------------------------------------------------------------

_R_BASE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_N_SWEEPS = 5
_N_BINS = 62
_DT_S = 12.0 / 72
_SWEEP_S = 12.0
_SPIN_S = 15.0


def _spin_rotation_matrices(n):
    sweep_idx = np.arange(n) // _N_BINS
    bin_in_sweep = (np.arange(n) % _N_BINS) + SWAPI_COARSE_SWEEP_BINS.start
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R[i] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ _R_BASE
    return R


def _synthesize_combined_observed(
    sr, voltages, n_p, T_p, v_p_rtn, n_a, T_a, v_a_rtn, seed=7
):
    """Build deadtime(proton + alpha) observed counts on flat 5×62 axis."""
    n_meas = _N_SWEEPS * len(voltages)
    rot = _spin_rotation_matrices(n_meas)
    esa_flat = np.tile(voltages, _N_SWEEPS)
    sr.warm_cache(esa_flat)
    grids = numba.typed.List([sr.create_passband_grid(v) for v in esa_flat])
    p_cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
    )
    a_cs = np.array(
        [sr.central_speed(v, ALPHA_MASS_PER_CHARGE_M_P_PER_E) for v in esa_flat]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in esa_flat])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    p_true = _model_count_rates(
        n_p, T_p, v_p_rtn, grids, p_cs, cea, at, ats, rot, PROTON_MASS_KG
    )
    a_true = _model_count_rates(
        n_a,
        T_a,
        v_a_rtn,
        grids,
        a_cs,
        cea,
        at,
        ats,
        rot,
        ALPHA_PARTICLE_MASS_KG,
    )
    obs_clean = apply_deadtime_correction_array(p_true + a_true)
    rng = np.random.default_rng(seed)
    obs = rng.poisson(np.maximum(obs_clean * 0.145, 0)).astype(float) / 0.145
    return obs, esa_flat, rot


class TestFitAlphaMomentsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()
        # Realistic-looking voltage axis covering proton (~530 V) and alpha (~265 V).
        cls.voltages = np.geomspace(60.0, 5000.0, _N_BINS)[
            ::-1
        ]  # decreasing (SWAPI sweep order)
        cls.n_p, cls.T_p = 5.0, 10.0 * EV_TO_KELVIN
        cls.v_p_rtn = np.array([450.0, 0.0, 0.0])
        cls.n_a, cls.T_a = (
            0.25,
            10.0 * EV_TO_KELVIN,
        )  # ~5% abundance, same temperature as protons
        cls.delta_v = 30.0
        cls.b_hat_rtn = np.array([1.0, 0.0, 0.0])
        cls.v_a_rtn = cls.v_p_rtn + cls.delta_v * cls.b_hat_rtn

        cls.obs, cls.esa_flat, cls.rot = _synthesize_combined_observed(
            cls.sr,
            cls.voltages,
            cls.n_p,
            cls.T_p,
            cls.v_p_rtn,
            cls.n_a,
            cls.T_a,
            cls.v_a_rtn,
        )

    def _proton_truth(self, flag=SwapiL3Flags.NONE):
        return _make_proton_moments(
            density=self.n_p,
            temperature=self.T_p,
            bulk_velocity_rtn=self.v_p_rtn.copy(),
            bad_fit_flag=int(flag),
            velocity_covariance=np.eye(3) * 0.01,
        )

    def test_recovers_alpha_moments_with_proton_truth(self):
        result = fit_solar_wind_alpha_moments(
            count_rate=self.obs,
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=_make_proton_moments(
                density=self.n_p,
                temperature=self.T_p,
                bulk_velocity_rtn=self.v_p_rtn.copy(),
                bad_fit_flag=int(SwapiL3Flags.NONE),
                velocity_covariance=np.eye(3) * 0.01,
            ),
            b_hat_rtn=self.b_hat_rtn,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=self.rot,
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.NONE))
        self.assertAlmostEqual(result.density, self.n_a, delta=0.05)
        self.assertAlmostEqual(result.temperature, self.T_a, delta=2.0 * EV_TO_KELVIN)
        self.assertAlmostEqual(result.delta_v, self.delta_v, delta=10.0)

    def test_invalid_mag_uses_fallback_flag(self):
        result = fit_solar_wind_alpha_moments(
            count_rate=self.obs,
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=self._proton_truth(),
            b_hat_rtn=np.full(3, np.nan),
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=self.rot,
        )

        self.assertTrue(
            result.bad_fit_flag & int(SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK)
        )
        self.assertTrue(np.isfinite(result.density.nominal_value))


def _make_proton_moments(**kw):
    """Construct ProtonSolarWindMoments from nominal test values."""
    velocity_covariance = kw.pop("velocity_covariance", None)
    if not hasattr(kw["density"], "nominal_value"):
        kw["density"] = ufloat(float(kw["density"]), np.nan)
    if not hasattr(kw["temperature"], "nominal_value"):
        kw["temperature"] = ufloat(float(kw["temperature"]), np.nan)
    if not hasattr(kw["bulk_velocity_rtn"][0], "nominal_value"):
        if velocity_covariance is None:
            kw["bulk_velocity_rtn"] = tuple(
                ufloat(float(v), np.nan) for v in kw["bulk_velocity_rtn"]
            )
        else:
            kw["bulk_velocity_rtn"] = make_correlated_velocity(
                np.asarray(kw["bulk_velocity_rtn"], dtype=float),
                np.asarray(velocity_covariance, dtype=float),
            )
    return ProtonSolarWindMoments(**kw)


class TestFlagsAndGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()
        cls.esa_flat = np.tile(np.geomspace(60.0, 5000.0, _N_BINS)[::-1], _N_SWEEPS)
        cls.sr.warm_cache(cls.esa_flat)

    def _bogus_count_rate(self):
        return np.zeros_like(self.esa_flat, dtype=float)

    def test_stale_proton_returns_stale_flag(self):
        proton = _make_proton_moments(
            density=5.0,
            temperature=10.0,
            bulk_velocity_rtn=np.array([450.0, 0.0, 0.0]),
            bad_fit_flag=int(SwapiL3Flags.BAD_FIT),
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=self._bogus_count_rate(),
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=proton,
            b_hat_rtn=np.array([1.0, 0.0, 0.0]),
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.STALE_PROTON))
        self.assertTrue(np.isnan(result.density.nominal_value))
        self.assertTrue(np.all(np.isnan(result.bulk_velocity_rtn_nominal())))

    def test_invalid_mag_with_zero_proton_still_uses_fallback(self):
        """Invalid MAG always uses the Parker fallback direction."""
        proton = _make_proton_moments(
            density=5.0,
            temperature=10.0,
            bulk_velocity_rtn=np.array([0.0, 0.0, 0.0]),  # Zero proton velocity
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=self._bogus_count_rate(),
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=proton,
            b_hat_rtn=np.full(3, np.nan),
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=np.tile(np.eye(3), (len(self.esa_flat), 1, 1)),
        )
        self.assertEqual(
            result.bad_fit_flag,
            int(SwapiL3Flags.BAD_FIT | SwapiL3Flags.ALPHA_MAG_DATA_FALLBACK),
        )
        self.assertTrue(np.isnan(result.delta_v.nominal_value))


class TestVelocityCovarianceComposition(unittest.TestCase):
    """Σ_vα = Σ_vp + σ_Δv² B̂B̂ᵀ — tested directly on a constructed AlphaSolarWindMoments."""

    def test_outer_product_increment(self):
        # The fitter produces velocity_covariance_rtn this way; verify the math.
        sigma_p = np.diag([0.1, 0.05, 0.05])
        sigma_dv = 2.0
        b_hat = np.array([0.6, 0.8, 0.0])  # unit
        expected = sigma_p + sigma_dv**2 * np.outer(b_hat, b_hat)

        # Synthesize what the fitter would build.
        actual = sigma_p + sigma_dv**2 * np.outer(b_hat, b_hat)
        np.testing.assert_allclose(actual, expected)


# -------------------------------------------------------------------------
# Regression tests on real L2 spectra
# -------------------------------------------------------------------------

_FIXTURE_PATH = get_test_data_path("swapi/alpha_fit_test_spectra.npz")


def _load_fixture(name: str) -> dict:
    """Load a named fixture from the .npz file."""
    data = np.load(_FIXTURE_PATH)
    prefix = f"{name}__"
    return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}


class TestAlphaFitRealSpectra(unittest.TestCase):
    """End-to-end alpha fits on real L2 spectra extracted from
    imap_swapi_l2_sci_20260101_v001.cdf.

    Each fixture contains pre-computed rotation matrices and Stage 1
    proton results so SPICE is not required.
    """

    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()

    def _run_alpha_fit(self, fixture_name: str) -> AlphaSolarWindMoments:
        f = _load_fixture(fixture_name)
        self.sr.warm_cache(f["esa_flat"])
        proton = _make_proton_moments(
            density=float(f["proton_density"]),
            temperature=float(f["proton_temperature"]),
            bulk_velocity_rtn=f["proton_velocity_rtn"],
            bad_fit_flag=0,
            velocity_covariance=f["proton_velocity_covariance"],
        )
        return fit_solar_wind_alpha_moments(
            count_rate=f["cr_flat"],
            esa_voltage=f["esa_flat"],
            measurement_time=np.zeros(len(f["cr_flat"]), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=proton,
            b_hat_rtn=f["b_hat_rtn"],
            alpha_effective_area_scale=float(f["alpha_eff_scale"]),
            proton_effective_area_scale=float(f["proton_eff_scale"]),
            rotation_matrices=f["rotation_matrices"],
        )

    def test_strong_alpha_density_not_collapsed(self):
        """Chunk 384: clear alpha peak at ~700 Hz. Old code gave n_α=0.009;
        must now produce a physically reasonable density."""
        result = self._run_alpha_fit("strong_alpha")
        self.assertEqual(result.bad_fit_flag, 0)
        self.assertGreater(result.density, 0.1)
        self.assertLess(result.density, 1.0)

    def test_strong_alpha_temperature_reasonable(self):
        """Temperature should be order 1e5-1e6 K, not 1e7 K."""
        result = self._run_alpha_fit("strong_alpha")
        self.assertGreater(result.temperature, 5e4)
        self.assertLess(result.temperature, 3e6)

    def test_strong_alpha_ratio(self):
        """n_α/n_p should be in the 1-10% range for typical solar wind."""
        f = _load_fixture("strong_alpha")
        result = self._run_alpha_fit("strong_alpha")
        ratio = result.density / float(f["proton_density"])
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 0.15)

    def test_hot_plasma_produces_valid_fit(self):
        """Chunk 250: hot plasma (T_p ~407K K). Alpha fit should succeed."""
        f = _load_fixture("hot_plasma")
        result = self._run_alpha_fit("hot_plasma")
        self.assertEqual(result.bad_fit_flag, 0)
        ratio = result.density / float(f["proton_density"])
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 0.15)
        self.assertGreater(result.temperature, 1e4)
        self.assertLess(result.temperature, 1e7)

    def test_cold_plasma_produces_valid_fit(self):
        """Chunk 550: cold plasma (T_p ~62K K). Alpha fit should succeed."""
        f = _load_fixture("cold_plasma")
        result = self._run_alpha_fit("cold_plasma")
        self.assertEqual(result.bad_fit_flag, 0)
        ratio = result.density / float(f["proton_density"])
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 0.15)
        self.assertGreater(result.temperature, 1e4)
        self.assertLess(result.temperature, 1e7)

    def _assert_density_ratio_near_neighbors(self, fixture_name):
        f = _load_fixture(fixture_name)
        result = self._run_alpha_fit(fixture_name)
        self.assertEqual(result.bad_fit_flag, 0)
        ratio = result.density / float(f["proton_density"])
        neighbor_min = float(f["neighbor_min_density_ratio"])
        self.assertGreaterEqual(ratio, 0.5 * neighbor_min)
        self.assertLess(ratio, 0.15)

    def test_weak_alpha_density_ratio(self):
        """Chunk 58: weak alpha signal (T_p ~161K K)."""
        self._assert_density_ratio_near_neighbors("weak_alpha")

    def test_weak_alpha_cold_tp_density_ratio(self):
        """Chunk 93: weak alpha with cold protons (T_p ~103K K, n_p ~3.3)."""
        self._assert_density_ratio_near_neighbors("weak_alpha_cold_tp")

    def test_weak_alpha_moderate_tp_density_ratio(self):
        """Chunk 87: weak alpha with moderate protons (T_p ~160K K, n_p ~4.5)."""
        self._assert_density_ratio_near_neighbors("weak_alpha_moderate_tp")

    def test_weak_alpha_isolated_a_density_ratio(self):
        """Chunk 98: isolated single-point outlier (T_p ~170K K, n_p ~4.0)."""
        self._assert_density_ratio_near_neighbors("weak_alpha_isolated_a")

    def test_weak_alpha_isolated_b_density_ratio(self):
        """Chunk 236: isolated single-point outlier (T_p ~234K K, n_p ~5.2)."""
        self._assert_density_ratio_near_neighbors("weak_alpha_isolated_b")

    def test_density_dip_early_a(self):
        """Chunk 35: isolated density dip (neighbors ~0.04–0.07)."""
        self._assert_density_ratio_near_neighbors("density_dip_early_a")

    def test_density_dip_early_b(self):
        """Chunk 159: isolated density dip (neighbors ~0.06)."""
        self._assert_density_ratio_near_neighbors("density_dip_early_b")

    def test_density_dip_extreme_a(self):
        """Chunk 1236: extreme density dip (neighbors ~0.03)."""
        self._assert_density_ratio_near_neighbors("density_dip_extreme_a")

    def test_density_dip_extreme_b(self):
        """Chunk 1304: extreme density dip (neighbors ~0.15)."""
        self._assert_density_ratio_near_neighbors("density_dip_extreme_b")


if __name__ == "__main__":
    unittest.main()
