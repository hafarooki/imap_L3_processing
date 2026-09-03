#!/usr/bin/env python3
"""Generate a dense ground-truth shell-integral reference for the collapsed
response grid (H(v', V)).

Output: tests/test_data/swapi/collapsed_response_grid_reference.csv
Usage:  uv run python scripts/swapi/generate_collapsed_response_grid_reference.py
"""

import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.integrate import dblquad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imap_l3_processing.constants import (
    CENTIMETERS_PER_METER,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    solar_wind_frame_speed_range,
)
from imap_l3_processing.swapi.l3a.utils import (
    velocity_components_to_angles_in_instrument_frame,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_l3_processing.swapi.response.passband_grid import interpolate_passband
from tests.swapi._helpers import load_swapi_response

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PATH = (
    _REPO_ROOT / "tests" / "test_data" / "swapi" / "collapsed_response_grid_reference.csv"
)

_N_POINTS = 256
_ESA_VOLTAGE = 5000.0
_MASS_PER_CHARGE = 4.0
_BULK_SPEED = 450.0
_BULK_AZIMUTH_DEG = 5.0
_BULK_ELEVATION_DEG = -10.0

_EFFECTIVE_AREA_CM2_TO_KM2 = 1.0 / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 2
_SPEED_RATIO_MIN = 0.9
_SPEED_RATIO_MAX = 1.1


def shell_integral_h(
    response_grid,
    v_prime: float,
    bulk_speed: float,
    bulk_azimuth_deg: float,
    bulk_elevation_deg: float,
) -> float:
    """H(v', V) by `dblquad` over the shell ‖v − v_sw‖ = v', split by passband
    region (sunglasses, open-aperture ±φ) for smooth adaptive convergence."""
    bulk_elevation = math.radians(bulk_elevation_deg)
    bulk_azimuth = math.radians(bulk_azimuth_deg)
    central_speed = float(response_grid.central_speed)

    # SWAPI convention: hat{v}(θ,φ) = (-cosθ sinφ, -cosθ cosφ, -sinθ).
    bulk_direction = np.array([
        -math.cos(bulk_elevation) * math.sin(bulk_azimuth),
        -math.cos(bulk_elevation) * math.cos(bulk_azimuth),
        -math.sin(bulk_elevation),
    ])
    velocity = bulk_speed * bulk_direction

    # cos-α band from |v|² = v_sw² + v'² + 2 v_sw v' cos α (law of cosines).
    cos_alpha_bounds = (
        (np.array([_SPEED_RATIO_MIN, _SPEED_RATIO_MAX]) * central_speed) ** 2
        - bulk_speed**2 - v_prime**2
    ) / (2.0 * bulk_speed * v_prime)
    if cos_alpha_bounds[0] >= 1.0 or cos_alpha_bounds[1] <= -1.0:
        return 0.0
    alpha_max, alpha_min = np.arccos(np.clip(cos_alpha_bounds, -1.0, 1.0))

    # Right-handed orthonormal frame with bulk_direction as polar axis.
    e_in_xy = np.array([-math.cos(bulk_azimuth), math.sin(bulk_azimuth), 0.0])
    rotation = np.column_stack([e_in_xy, np.cross(bulk_direction, e_in_xy), bulk_direction])

    def integrate(passband, azimuth_min_deg, azimuth_max_deg):
        return dblquad(
            lambda alpha, beta: _shell_integrand(
                alpha, beta, velocity, rotation, v_prime, response_grid,
                passband, azimuth_min_deg, azimuth_max_deg,
            ),
            0.0, 2.0 * math.pi, alpha_min, alpha_max,
            epsabs=1e-13, epsrel=1e-3,
        )[0]

    return (
        integrate(response_grid.sg_passband, -20.0, 20.0)
        + integrate(response_grid.oa_passband, 20.0, 180.0)
        + integrate(response_grid.oa_passband, -180.0, -20.0)
    )


@numba.njit
def _shell_integrand(
    alpha, beta, velocity, rotation, v_prime, response_grid,
    passband, azimuth_min_deg, azimuth_max_deg,
):
    # `alpha`: polar angle on the shell from bulk-velocity direction;
    #   caller restricts to the α-band where the shell crosses
    #   |v|/v_0 ∈ [SPEED_RATIO_MIN, SPEED_RATIO_MAX].
    # `beta`: azimuth around bulk-velocity direction, ∈ [0, 2π].
    # `passband`/azimuth window: caller picks one region so the integrand is
    #   smooth (returns 0 outside the window).
    sin_alpha = math.sin(alpha)
    local_x = sin_alpha * math.cos(beta)
    local_y = sin_alpha * math.sin(beta)
    local_z = math.cos(alpha)
    shell_x = rotation[0, 0] * local_x + rotation[0, 1] * local_y + rotation[0, 2] * local_z
    shell_y = rotation[1, 0] * local_x + rotation[1, 1] * local_y + rotation[1, 2] * local_z
    shell_z = rotation[2, 0] * local_x + rotation[2, 1] * local_y + rotation[2, 2] * local_z
    vx = velocity[0] + v_prime * shell_x
    vy = velocity[1] + v_prime * shell_y
    vz = velocity[2] + v_prime * shell_z
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    azimuth_deg, elevation_deg = velocity_components_to_angles_in_instrument_frame(vx, vy, vz)
    if not (azimuth_min_deg <= azimuth_deg <= azimuth_max_deg):
        return 0.0
    effective_area = (
        response_grid.central_effective_area
        * _EFFECTIVE_AREA_CM2_TO_KM2
        * interpolate_passband(passband, elevation_deg, speed / response_grid.central_speed)
        * interpolate_azimuthal_transmission(response_grid.azimuthal_transmission, azimuth_deg)
    )
    # H is an angular integral (docs convention); dα dβ comes from dblquad, so
    # only the sin α solid-angle factor is multiplied in here.
    d3v = sin_alpha
    return speed * effective_area * d3v


_WORKER_RESPONSE_GRID = None


def _worker_init(response_grid):
    global _WORKER_RESPONSE_GRID
    _WORKER_RESPONSE_GRID = response_grid


def _worker_compute(v_prime):
    return shell_integral_h(
        _WORKER_RESPONSE_GRID,
        v_prime=float(v_prime),
        bulk_speed=_BULK_SPEED,
        bulk_azimuth_deg=_BULK_AZIMUTH_DEG,
        bulk_elevation_deg=_BULK_ELEVATION_DEG,
    )


def main():
    print(f"Loading SWAPI response and warming cache at V={_ESA_VOLTAGE}...", flush=True)
    swapi_response = load_swapi_response(
        warm_cache_voltages=np.array([_ESA_VOLTAGE])
    )
    response_grid = swapi_response.get_response_grid(
        esa_voltage=_ESA_VOLTAGE,
        mass_per_charge_m_p_per_e=_MASS_PER_CHARGE,
    )

    v_prime_min, v_prime_max = solar_wind_frame_speed_range(
        response_grid.central_speed, _BULK_SPEED
    )
    v_prime_grid = np.linspace(v_prime_min, v_prime_max, _N_POINTS)

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(
        f"Computing {_N_POINTS}-point shell integral (v' ∈ "
        f"[{v_prime_min:.3f}, {v_prime_max:.3f}] km/s) at "
        f"bulk=({_BULK_SPEED}, {_BULK_AZIMUTH_DEG}°, {_BULK_ELEVATION_DEG}°) "
        f"across {n_workers} workers...",
        flush=True,
    )
    h_values = np.empty(_N_POINTS, dtype=float)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(response_grid,),
    ) as pool:
        for i, h in enumerate(pool.imap(_worker_compute, v_prime_grid, chunksize=1)):
            h_values[i] = h
            if (i + 1) % 16 == 0 or i == _N_POINTS - 1:
                print(
                    f"  {i + 1:>3}/{_N_POINTS}  v'={v_prime_grid[i]:.3f}  H={h:.6e}",
                    flush=True,
                )

    df = pd.DataFrame({"v_prime_kms": v_prime_grid, "h_truth_km3_per_s": h_values})

    header = (
        f"# Collapsed-response-grid shell integral H(v', V), dblquad reference.\n"
        f"# esa_voltage={_ESA_VOLTAGE} V, mass_per_charge={_MASS_PER_CHARGE} m_p/e\n"
        f"# bulk_speed={_BULK_SPEED} km/s, "
        f"bulk_azimuth={_BULK_AZIMUTH_DEG} deg, "
        f"bulk_elevation={_BULK_ELEVATION_DEG} deg\n"
        f"# N_POINTS={_N_POINTS}\n"
    )
    with open(_OUTPUT_PATH, "w") as f:
        f.write(header)
        df.to_csv(f, index=False)
    print(f"Saved {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
