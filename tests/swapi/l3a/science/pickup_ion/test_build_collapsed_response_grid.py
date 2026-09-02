import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from imap_l3_processing.constants import HE_PUI_PARTICLE_MASS_PER_CHARGE_M_P_PER_E
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    build_collapsed_response_grid,
)
from scripts.swapi.generate_collapsed_response_grid_reference import shell_integral_h
from tests.swapi._helpers import build_default_v_prime_grid_kms, load_swapi_response

_REFERENCE_CSV_PATH = (
    Path(__file__).resolve().parents[4]
    / "test_data"
    / "swapi"
    / "collapsed_response_grid_reference.csv"
)

# Must match the fixed condition baked into the reference CSV (see
# scripts/swapi/generate_collapsed_response_grid_reference.py).
_ESA_VOLTAGE = 5000.0
_MASS_PER_CHARGE = HE_PUI_PARTICLE_MASS_PER_CHARGE_M_P_PER_E
_BULK_SPEED = 450.0
_BULK_AZIMUTH = 5.0
_BULK_ELEVATION = -10.0


class BuildCollapsedResponseGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        swapi_response = load_swapi_response(
            warm_cache_voltages=np.array([_ESA_VOLTAGE])
        )
        cls.response_grid = swapi_response.get_response_grid(
            esa_voltage=_ESA_VOLTAGE,
            mass_per_charge_m_p_per_e=_MASS_PER_CHARGE,
        )
        reference = pd.read_csv(_REFERENCE_CSV_PATH, comment="#")
        cls.reference_v_prime = reference["v_prime_kms"].to_numpy()
        cls.reference_h = reference["h_truth_km3_per_s"].to_numpy()

    def test_grid_matches_precomputed_shell_integral_reference(self):
        speed_in_sw_frame = build_default_v_prime_grid_kms(
            self.response_grid, _BULK_SPEED
        )
        collapsed = build_collapsed_response_grid(
            self.response_grid,
            bulk_speed=_BULK_SPEED,
            bulk_azimuth=_BULK_AZIMUTH,
            bulk_elevation=_BULK_ELEVATION,
            speed_in_sw_frame=speed_in_sw_frame,
        )

        self.assertAlmostEqual(
            float(collapsed.speed_in_sw_frame[0]),
            float(self.reference_v_prime[0]),
            places=6,
            msg="production v' axis lower endpoint must match reference",
        )
        self.assertAlmostEqual(
            float(collapsed.speed_in_sw_frame[-1]),
            float(self.reference_v_prime[-1]),
            places=6,
            msg="production v' axis upper endpoint must match reference",
        )

        h_truth = np.interp(
            collapsed.speed_in_sw_frame, self.reference_v_prime, self.reference_h
        )

        # Compare only where the reference is meaningfully above the noise
        # floor. Near v' extremes both implementations differ in their
        # handling of the shell/passband transition and dblquad itself emits
        # "integral probably divergent" warnings there — those points carry
        # no useful signal.
        peak = float(h_truth.max())
        self.assertGreater(peak, 0.0, "reference must have positive H somewhere")
        trusted = h_truth > 0.05 * peak
        self.assertGreater(trusted.sum(), 16, "trusted region too small")

        rel_err = np.abs(collapsed.values[trusted] - h_truth[trusted]) / h_truth[trusted]
        worst = int(np.argmax(rel_err))
        worst_index = int(np.flatnonzero(trusted)[worst])
        self.assertLess(
            float(rel_err.max()),
            0.015,
            f"worst point: idx={worst_index} "
            f"v'={collapsed.speed_in_sw_frame[worst_index]:.3f} "
            f"H_fn={collapsed.values[worst_index]:.6e} "
            f"H_truth={h_truth[worst_index]:.6e} "
            f"rel_err={rel_err.max():.3%}",
        )

    def test_raises_when_speed_in_sw_frame_is_not_one_dimensional(self):
        speed_in_sw_frame_2d = np.zeros((4, 8))
        with self.assertRaisesRegex(ValueError, "speed_in_sw_frame"):
            build_collapsed_response_grid(
                self.response_grid,
                bulk_speed=_BULK_SPEED,
                bulk_azimuth=_BULK_AZIMUTH,
                bulk_elevation=_BULK_ELEVATION,
                speed_in_sw_frame=speed_in_sw_frame_2d,
            )

    def test_matches_shell_integral_when_grid_cell_aligned_with_singularity(self):
        # The closed-form φ-inversion integrand has a 1/sin(Δφ) factor that
        # diverges at the shell boundary cos(Δφ) = ±1. If the discrete
        # (speed_ratio, elevation) grid lands a cell arbitrarily close to that
        # boundary, that one cell would produce a spurious spike in H(v').
        # Constructed setup: choose v' so the (speed_ratio index 21, elevation
        # index 16) cell from the kernel's internal 32×32 grid lands at
        # cos_delta_azimuth = 1 - 5e-7, deep inside the |cos| > 1 - 1e-5 guard
        # zone. ESA voltage 10000 V widens v'_max enough that the contrived
        # singular v' lands at ~12 % into the v' range — far enough from the
        # boundary that the 32×32 grid is accurate to ~5 % against the
        # dblquad truth — while sr_idx 21 is still inside the passband-filter
        # cutoff (passband > 1e-3) so the cos guard genuinely activates.
        esa_voltage = 10000.0
        swapi_response = load_swapi_response(
            warm_cache_voltages=np.array([esa_voltage])
        )
        response_grid = swapi_response.get_response_grid(
            esa_voltage=esa_voltage,
            mass_per_charge_m_p_per_e=_MASS_PER_CHARGE,
        )
        bulk_speed = _BULK_SPEED
        central_speed = response_grid.central_speed

        speed_ratio = 0.9 + 21 * 0.2 / 31
        speed = central_speed * speed_ratio
        elevation_deg = -15 + 16 * 30 / 31
        cos_elevation = np.cos(np.deg2rad(elevation_deg))
        cos_delta_azimuth_target = 1.0 - 5e-7
        cos_angle_target = cos_delta_azimuth_target * cos_elevation
        v_singular_kms = float(np.sqrt(
            speed ** 2 + bulk_speed ** 2
            - 2 * cos_angle_target * speed * bulk_speed
        ))

        collapsed = build_collapsed_response_grid(
            response_grid,
            bulk_speed=bulk_speed,
            bulk_azimuth=0.0,
            bulk_elevation=0.0,
            speed_in_sw_frame=np.array([v_singular_kms]),
        )
        h_production = float(collapsed.values[0])

        h_truth = shell_integral_h(
            response_grid,
            v_prime=v_singular_kms,
            bulk_speed=bulk_speed,
            bulk_azimuth_deg=0.0,
            bulk_elevation_deg=0.0,
        )

        # With the guard in place, rel_err ≈ 5 %. Without it, the one
        # near-singular cell drives rel_err to ~18 %. A 10 % tolerance
        # distinguishes the two with a ~2× margin.
        rel_err = abs(h_production - h_truth) / h_truth
        self.assertLess(
            rel_err,
            0.10,
            f"v'={v_singular_kms:.4f}: production H={h_production:.3e}, "
            f"dblquad truth H={h_truth:.3e}, rel_err={rel_err:.2%}",
        )


if __name__ == "__main__":
    unittest.main()
