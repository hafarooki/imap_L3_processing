"""Regression test for the cold/slow/dense wrong-basin pathology.

Drawn from `docs/swapi/figure_src/plot_fit_accuracy.py`, sample index 7244 of
the WIND/SWE 2025 dataset (seed=7). Truth is a slow (v_R=297 km/s), cold
(T=16,290 K), dense (n=20.2 cm⁻³) stream. With the default initial guess
(v_T=-30, v_N=0, T=60000·(v/400)² = 31,877 K), LM walks downhill into a deep
narrow alternate basin with v_T≈-104 km/s and n≈121 cm⁻³ (≈300× higher MSE
than the truth basin). The spin-axis-flip basin-hop heuristic does not recover
truth: the wrong-basin solution is not related to truth by a 180° spin-axis
rotation.

This test fails on the current implementation and will pass when either the
initial guess or the basin-hopping logic recovers the cold/slow/dense regime.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    fit_solar_wind_proton_moments,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind_forward_model import (
    SolarWindParams,
    apply_deadtime_correction_array,
    model_solar_wind_coincidence_rates,
)
from imap_l3_processing.swapi.response.swapi_response import SWAPIResponse

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"


# Mean SWAPI L2 coarse-sweep voltages (V), descending. Indices 1..62 of the
# 72-bin sweep; fine-sweep bins (clustered near the proton peak) are excluded.
# Matches plot_fit_accuracy.py.
_COARSE_VOLTAGES = np.array([
     9895.52,  9088.69,  8348.80,  7667.55,  7042.16,  6469.31,  5941.77,  5457.31,
     5013.22,  4603.65,  4230.77,  3886.92,  3569.16,  3278.72,  3011.13,  2766.25,
     2539.54,  2333.83,  2144.24,  1969.31,  1808.74,  1660.86,  1525.75,  1401.82,
     1287.58,  1182.24,  1085.15,   995.55,   914.31,   839.94,   771.70,   709.46,
      651.59,   598.47,   549.91,   505.12,   463.89,   425.92,   391.18,   359.35,
      329.94,   303.02,   278.25,   255.55,   234.77,   215.61,   197.95,   181.82,
      167.04,   153.46,   140.91,   129.50,   118.91,   109.20,   100.30,    92.11,
       84.61,    77.73,    71.40,    65.59,    60.23,    55.34,
])
_N_BINS = 62
_N_SWEEPS = 5

# RTN -> SWAPI rotation matrices at sweep midpoints. SPICE-derived, captures
# ~4° spin-axis tilt off -R_RTN and the ~15.13 s spin period. Used as
# anchors for per-bin rotation that captures within-sweep spin (~285° of
# rotation across the 12 s sweep). Matches plot_fit_accuracy.py.
_SWEEP_MIDPOINT_MATRICES = np.array([
    [[+0.0705, +0.9157, +0.3955],
     [-0.9968, +0.0792, -0.0057],
     [-0.0365, -0.3939, +0.9184]],
    [[-0.0141, -0.1350, +0.9907],
     [-0.9972, +0.0743, -0.0041],
     [-0.0731, -0.9881, -0.1357]],
    [[-0.0721, -0.9884, +0.1340],
     [-0.9974, +0.0716, -0.0084],
     [-0.0013, -0.1342, -0.9909]],
    [[-0.0183, -0.3937, -0.9191],
     [-0.9971, +0.0750, -0.0122],
     [+0.0737, +0.9162, -0.3939]],
    [[+0.0683, +0.7775, -0.6251],
     [-0.9968, +0.0795, -0.0100],
     [+0.0420, +0.6238, +0.7805]],
])
_BINS_PER_SWEEP = 72
_SWEEP_DURATION_S = 12.0
_DT_S = _SWEEP_DURATION_S / _BINS_PER_SWEEP
_BIN_INDICES_IN_SWEEP = np.arange(1, 63)  # 62 coarse bins, 1-indexed


def _build_per_bin_rotation_matrices() -> np.ndarray:
    """Spin midpoint matrices about the spin axis to per-bin times, so each
    coincidence-rate bin samples a different spacecraft phase."""
    spin_axis = _SWEEP_MIDPOINT_MATRICES[:, 1, :].mean(axis=0)
    spin_axis = spin_axis / np.linalg.norm(spin_axis)
    x_axis = _SWEEP_MIDPOINT_MATRICES[:, 0, :]
    x_perp = x_axis - (x_axis @ spin_axis)[:, None] * spin_axis
    e1 = x_perp[0] / np.linalg.norm(x_perp[0])
    e2 = np.cross(spin_axis, e1)
    phase = np.unwrap(np.arctan2(x_perp @ e2, x_perp @ e1))
    midpoint_times_s = (np.arange(_N_SWEEPS) + 0.5) * _SWEEP_DURATION_S
    omega, _ = np.polyfit(midpoint_times_s, phase, 1)

    sweep_index = np.repeat(np.arange(_N_SWEEPS), _N_BINS)
    bin_index = np.tile(_BIN_INDICES_IN_SWEEP, _N_SWEEPS)
    sample_times_s = sweep_index * _SWEEP_DURATION_S + bin_index * _DT_S
    midpoint_times_for_samples = (sweep_index + 0.5) * _SWEEP_DURATION_S
    delta_phi = omega * (sample_times_s - midpoint_times_for_samples)

    cos_dp, sin_dp = np.cos(delta_phi), np.sin(delta_phi)
    one_minus_cos = 1.0 - cos_dp
    ax, ay, az = spin_axis
    K = np.array([
        [cos_dp + ax*ax*one_minus_cos, ax*ay*one_minus_cos - az*sin_dp, ax*az*one_minus_cos + ay*sin_dp],
        [ay*ax*one_minus_cos + az*sin_dp, cos_dp + ay*ay*one_minus_cos, ay*az*one_minus_cos - ax*sin_dp],
        [az*ax*one_minus_cos - ay*sin_dp, az*ay*one_minus_cos + ax*sin_dp, cos_dp + az*az*one_minus_cos],
    ]).transpose(2, 0, 1)
    base = _SWEEP_MIDPOINT_MATRICES[sweep_index]
    return np.einsum("nij,njk->nik", base, K)


_PER_BIN_ROTATION_MATRICES = _build_per_bin_rotation_matrices()


def _load_swapi_response() -> SWAPIResponse:
    return SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )


class TestColdSlowDenseWrongBasin(unittest.TestCase):
    """Cold/slow/dense plasma must not converge into the wrong v_T/v_N basin."""

    # WIND/SWE-derived truth, sample 7244 of seed=7.
    TRUE_DENSITY = 20.2090
    TRUE_TEMPERATURE_K = 16290.0
    TRUE_VELOCITY_RTN = np.array([297.0, 29.4, -19.6])
    POISSON_SEED = 7244

    @classmethod
    def setUpClass(cls):
        swapi_response = _load_swapi_response()
        all_voltages = np.tile(_COARSE_VOLTAGES, _N_SWEEPS)
        swapi_response.warm_cache(all_voltages)
        per_bin_rotation_matrices = _PER_BIN_ROTATION_MATRICES

        truth_params = SolarWindParams(
            density=cls.TRUE_DENSITY,
            bulk_velocity_rtn=cls.TRUE_VELOCITY_RTN.copy(),
            temperature=cls.TRUE_TEMPERATURE_K,
            mass_kg=PROTON_MASS_KG,
        )
        base_ctx = build_solar_wind_fit_context(
            count_rate=np.ones_like(all_voltages),
            esa_voltage=all_voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=1.0,
            rotation_matrices=per_bin_rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )
        rates = apply_deadtime_correction_array(
            model_solar_wind_coincidence_rates(truth_params, base_ctx)
        )
        rng = np.random.default_rng(cls.POISSON_SEED)
        count_rate = (
            rng.poisson(np.maximum(rates * 0.145, 0.0)).astype(float) / 0.145
        )

        cls.ctx = build_solar_wind_fit_context(
            count_rate=count_rate,
            esa_voltage=all_voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=1.0,
            rotation_matrices=per_bin_rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )
        cls.result = fit_solar_wind_proton_moments(cls.ctx)

    def test_radial_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[0].nominal_value,
            self.TRUE_VELOCITY_RTN[0],
            atol=2.0,
        )

    def test_tangential_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[1].nominal_value,
            self.TRUE_VELOCITY_RTN[1],
            atol=2.0,
        )

    def test_normal_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[2].nominal_value,
            self.TRUE_VELOCITY_RTN[2],
            atol=2.0,
        )

    def test_density_recovered(self):
        np.testing.assert_allclose(
            self.result.density.nominal_value, self.TRUE_DENSITY, rtol=0.05,
        )

    def test_temperature_recovered(self):
        np.testing.assert_allclose(
            self.result.temperature.nominal_value, self.TRUE_TEMPERATURE_K, rtol=0.05,
        )


def _setup_synthetic_fit(
    density: float, temperature_k: float, velocity_rtn: np.ndarray, seed: int,
):
    swapi_response = _load_swapi_response()
    all_voltages = np.tile(_COARSE_VOLTAGES, _N_SWEEPS)
    swapi_response.warm_cache(all_voltages)
    per_bin_rotation_matrices = _PER_BIN_ROTATION_MATRICES

    truth_params = SolarWindParams(
        density=density,
        bulk_velocity_rtn=velocity_rtn.copy(),
        temperature=temperature_k,
        mass_kg=PROTON_MASS_KG,
    )
    base_ctx = build_solar_wind_fit_context(
        count_rate=np.ones_like(all_voltages),
        esa_voltage=all_voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=per_bin_rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    rates = apply_deadtime_correction_array(
        model_solar_wind_coincidence_rates(truth_params, base_ctx)
    )
    rng = np.random.default_rng(seed)
    count_rate = rng.poisson(np.maximum(rates * 0.145, 0.0)).astype(float) / 0.145

    ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=all_voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=per_bin_rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    return fit_solar_wind_proton_moments(ctx)


class _RecoverySuite(unittest.TestCase):
    """Common assertions for a synthetic-truth recovery test."""

    __test__ = False  # only the subclasses run

    TRUE_DENSITY: float
    TRUE_TEMPERATURE_K: float
    TRUE_VELOCITY_RTN: np.ndarray
    POISSON_SEED: int

    @classmethod
    def setUpClass(cls):
        cls.result = _setup_synthetic_fit(
            cls.TRUE_DENSITY, cls.TRUE_TEMPERATURE_K,
            cls.TRUE_VELOCITY_RTN, cls.POISSON_SEED,
        )

    def test_radial_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[0].nominal_value,
            self.TRUE_VELOCITY_RTN[0], atol=2.0,
        )

    def test_tangential_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[1].nominal_value,
            self.TRUE_VELOCITY_RTN[1], atol=2.0,
        )

    def test_normal_speed_recovered(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn[2].nominal_value,
            self.TRUE_VELOCITY_RTN[2], atol=2.0,
        )

    def test_density_recovered(self):
        np.testing.assert_allclose(
            self.result.density.nominal_value, self.TRUE_DENSITY, rtol=0.05,
        )

    def test_temperature_recovered(self):
        np.testing.assert_allclose(
            self.result.temperature.nominal_value,
            self.TRUE_TEMPERATURE_K, rtol=0.05,
        )


class TestSlowWarmHighVtWrongBasin(_RecoverySuite):
    """Sample 1302 of WIND/SWE 2025: slow (v_R=312), warm (T≈22 kK), low-density
    (n=5.2) with a large positive v_T (66.7 km/s). The fit lands at v_T≈-111
    km/s (sign-flipped, magnitude inflated), n≈3.9, T≈25.9 kK. Distinct
    failure mode from the cold/slow/dense case 7244 — same regime as case 1305.
    """
    __test__ = True
    TRUE_DENSITY = 5.218
    TRUE_TEMPERATURE_K = 21870.0
    TRUE_VELOCITY_RTN = np.array([312.0, 66.7, 3.4])
    POISSON_SEED = 1302


class TestColdSlowDenseAdjacentWrongBasin(_RecoverySuite):
    """Sample 7246 of WIND/SWE 2025: same plasma regime as case 7244
    (cold/slow/dense; v_R=292, v_T=34.4, v_N=-20.2, T=14550 K, n=20.0). Case
    7244 was fixed by the Gaussian-T initial guess + iso-|v| arc basin hop;
    7246 still flips to v_T≈-105, n≈97. Likely needs further tuning of the
    same code path or a different basin-hopping seed."""
    __test__ = True
    TRUE_DENSITY = 20.035
    TRUE_TEMPERATURE_K = 14550.0
    TRUE_VELOCITY_RTN = np.array([292.0, 34.4, -20.2])
    POISSON_SEED = 7246


class TestSlowWarmPositiveVtNegativeVN(_RecoverySuite):
    """Sample 1307 of WIND/SWE 2025: v_R=315, T≈22.6 kK, n=6.78, v_T=+79.6,
    v_N=-4.4. With per-bin rotation and spin-axis-aligned IG (vT≈-25), LM still
    walks into the v_T≈-129 antipodal basin."""
    __test__ = True
    TRUE_DENSITY = 6.784
    TRUE_TEMPERATURE_K = 22560.0
    TRUE_VELOCITY_RTN = np.array([315.1, 79.6, -4.4])
    POISSON_SEED = 1307


class TestFastLowDensityNoiseDominated(_RecoverySuite):
    """Sample 6763 of WIND/SWE 2025: v_R=531, T≈166 kK, very low n=0.14, v_T=-18.3,
    v_N=+10.7. Low SNR — the fit pulls v_N to ~0 instead of +10.7. Borderline
    between basin-flip and noise-limited miss."""
    __test__ = True
    TRUE_DENSITY = 0.143
    TRUE_TEMPERATURE_K = 165700.0
    TRUE_VELOCITY_RTN = np.array([531.3, -18.3, 10.7])
    POISSON_SEED = 6763


class TestFastVeryHotLargeNegativeVt(_RecoverySuite):
    """Sample 4093 of WIND/SWE 2025: v_R=577, very hot (T≈843 kK), n=7.3,
    v_T=-118.8, v_N=+28.2. Recovered correctly by single-shot LM on dual-basin;
    flipped by 3-stage method."""
    __test__ = True
    TRUE_DENSITY = 7.306
    TRUE_TEMPERATURE_K = 843400.0
    TRUE_VELOCITY_RTN = np.array([577.1, -118.8, 28.2])
    POISSON_SEED = 4093


class TestFastHotLargeNegativeVtNegativeVN(_RecoverySuite):
    """Sample 7285 of WIND/SWE 2025: v_R=519, hot (T≈667 kK), n=11.4,
    v_T=-95.9, v_N=-46.6. Recovered correctly by single-shot LM on dual-basin;
    flipped by 3-stage method."""
    __test__ = True
    TRUE_DENSITY = 11.395
    TRUE_TEMPERATURE_K = 666600.0
    TRUE_VELOCITY_RTN = np.array([518.8, -95.9, -46.6])
    POISSON_SEED = 7285


if __name__ == "__main__":
    unittest.main()
