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

# RTN -> SWAPI rotation matrices, one per sweep at sweep midpoints. SPICE-
# derived, captures ~4° spin-axis tilt off -R_RTN and the ~15.13 s spin
# period. Matches plot_fit_accuracy.py.
_ROTATION_MATRICES = np.array([
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
        per_bin_rotation_matrices = np.repeat(_ROTATION_MATRICES, _N_BINS, axis=0)

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


if __name__ == "__main__":
    unittest.main()
