"""Shared helpers for the SWAPI documentation figures."""

import io
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from PIL import Image
from spacepy import pycdf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse

FIGURES_DIR = REPO_ROOT / "docs" / "swapi" / "figures"


def require_imap_api_key() -> None:
    if not os.environ.get("IMAP_API_KEY"):
        raise SystemExit("IMAP_API_KEY is not set")


DEFAULT_PNG_DPI = 100
DEFAULT_PNG_PALETTE_COLORS = 256


def save_figure(
    figure,
    output_path: Path,
    *,
    dpi: int = DEFAULT_PNG_DPI,
    palette_colors: int | None = DEFAULT_PNG_PALETTE_COLORS,
    **savefig_kwargs,
) -> Path:
    """Write `figure` to `output_path` as a PNG, byte-identical across runs.

    The figure is rasterized at `dpi` and, unless `palette_colors` is None,
    reduced to that many indexed colours: the figures are flat-colour line art,
    which loses nothing visible and stores about three times smaller.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("bbox_inches", "tight")

    if output_path.suffix != ".png":
        raise ValueError(f"Unsupported figure format: {output_path.suffix!r}")

    # Claude: rasterize into memory so the bytes on disk are always written by
    # Pillow, which keeps no timestamp or renderer-version chunks.
    savefig_kwargs["dpi"] = dpi
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", **savefig_kwargs)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        if palette_colors is None:
            rendered.save(output_path, optimize=True)
        else:
            # Claude: quantize() needs RGB, and the flattened figure is opaque.
            opaque = rendered.convert("RGB")
            opaque.quantize(colors=palette_colors).save(output_path, optimize=True)
    return output_path


_INSTRUMENT_DATA_DIR = REPO_ROOT / "instrument_team_data" / "swapi"
_EFFICIENCY_TABLE_PATH = (
    REPO_ROOT
    / "tests"
    / "test_data"
    / "swapi"
    / "imap_swapi_efficiency-lut-test_20241020_v001.dat"
)


def load_swapi_response() -> SwapiResponse:
    return SwapiResponse.from_files(
        azimuthal_transmission_path=_INSTRUMENT_DATA_DIR
        / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        central_effective_area_path=_INSTRUMENT_DATA_DIR
        / "imap_swapi_central-effective-area_20260425_v001.csv",
        passband_fit_coefficients_path=_INSTRUMENT_DATA_DIR
        / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
        efficiency_table_path=_EFFICIENCY_TABLE_PATH,
    )


def find_sweep_start_index(cdf_path: Path, sweep_start_time_utc: str) -> int:
    """Return the index of the sweep whose `sci_start_time` is `sweep_start_time_utc`.

    Pinning a figure's input block by timestamp instead of by sweep index keeps
    the figure reproducible across reprocessing: the plot scripts query the L2
    with `version="latest"`, and a reprocessed file can gain or lose sweeps at
    either end, which shifts every positional index.
    """
    with pycdf.CDF(str(cdf_path)) as cdf:
        sweep_start_times = [str(t) for t in cdf["sci_start_time"][...]]
    # Claude: sci_start_time is the sweep start; `epoch` is the sweep centre.
    matches = [i for i, t in enumerate(sweep_start_times) if t == sweep_start_time_utc]
    if not matches:
        raise SystemExit(
            f"No sweep starting at {sweep_start_time_utc} in {cdf_path.name}. "
            f"File covers {sweep_start_times[0]} .. {sweep_start_times[-1]}."
        )
    return matches[0]


SWEEP_DURATION_S = 12.0
BINS_PER_SWEEP = 72
BIN_PERIOD_S = SWEEP_DURATION_S / BINS_PER_SWEEP
SPIN_PERIOD_S = 15.13

COARSE_BIN_INDICES_IN_SWEEP = np.arange(1, 63)

COARSE_SWEEP_VOLTAGES_MEAN_V = np.array(
    [
        9895.52,
        9088.69,
        8348.80,
        7667.55,
        7042.16,
        6469.31,
        5941.77,
        5457.31,
        5013.22,
        4603.65,
        4230.77,
        3886.92,
        3569.16,
        3278.72,
        3011.13,
        2766.25,
        2539.54,
        2333.83,
        2144.24,
        1969.31,
        1808.74,
        1660.86,
        1525.75,
        1401.82,
        1287.58,
        1182.24,
        1085.15,
        995.55,
        914.31,
        839.94,
        771.70,
        709.46,
        651.59,
        598.47,
        549.91,
        505.12,
        463.89,
        425.92,
        391.18,
        359.35,
        329.94,
        303.02,
        278.25,
        255.55,
        234.77,
        215.61,
        197.95,
        181.82,
        167.04,
        153.46,
        140.91,
        129.50,
        118.91,
        109.20,
        100.30,
        92.11,
        84.61,
        77.73,
        71.40,
        65.59,
        60.23,
        55.34,
    ]
)

# SWAPI→RTN at the first-sweep midpoint of a real SPICE attitude near 2026-01-01;
# spin axis (+Y column) sits ~4° off −R̂_RTN. Per-bin matrices are built by
# spinning this about its own +Y at the nominal IMAP spin period.
ANCHOR_ROTATION_SWAPI_TO_RTN = np.array(
    [
        [+0.0705, +0.9157, +0.3955],
        [-0.9968, +0.0792, -0.0057],
        [-0.0365, -0.3939, +0.9184],
    ]
).T
ANCHOR_TIME_S = 0.5 * SWEEP_DURATION_S
# Sign reproduces independent SPICE-derived midpoints across a 5-sweep cycle.
SPIN_OMEGA_RAD_S = -2.0 * np.pi / SPIN_PERIOD_S


def compute_per_bin_rotation_matrices(
    n_sweeps: int,
    bin_indices_in_sweep: np.ndarray = COARSE_BIN_INDICES_IN_SWEEP,
) -> np.ndarray:
    """Synthetic per-bin SWAPI→RTN matrices: anchor spun about its spin axis.

    Returns shape (n_sweeps · n_bins, 3, 3) in sweep-major, bin-minor order.
    """
    sweep_index = np.repeat(np.arange(n_sweeps), len(bin_indices_in_sweep))
    bin_index = np.tile(bin_indices_in_sweep, n_sweeps)
    sample_times_s = sweep_index * SWEEP_DURATION_S + bin_index * BIN_PERIOD_S

    spin_axis = ANCHOR_ROTATION_SWAPI_TO_RTN[:, 1] / np.linalg.norm(
        ANCHOR_ROTATION_SWAPI_TO_RTN[:, 1]
    )
    delta_phi = SPIN_OMEGA_RAD_S * (sample_times_s - ANCHOR_TIME_S)

    ax, ay, az = spin_axis
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return rot @ ANCHOR_ROTATION_SWAPI_TO_RTN


def peak_esa_voltage_for_proton_bulk_speed(bulk_speed_km_s: float) -> float:
    return float(
        PROTON_MASS_KG
        * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


_T = TypeVar("_T")


def run_parallel_map(
    process_one: Callable[[int], _T],
    n_items: int,
    *,
    desc: str,
    chunksize: int = 10,
) -> list[_T]:
    """Run `process_one(i)` for i in [0, n_items) across forked workers.

    Module-level `_worker_state` is inherited by children via fork — callers
    should populate it in the parent before invoking this helper.
    """
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
    # Claude: docs/update_figures.py sets FIGURE_WORKER_PROCESSES so that nested
    # pools share the machine when several figure scripts run side by side.
    n_workers = int(os.environ.get("FIGURE_WORKER_PROCESSES") or 0) or (
        os.cpu_count() or 1
    )
    print(f"Running {n_items} {desc} across {n_workers} processes...")
    start = time.perf_counter()
    results: list[_T | None] = [None] * n_items
    report_every = max(1, n_items // 20)
    with multiprocessing.get_context("fork").Pool(processes=n_workers) as pool:
        completed = 0
        for index, value in pool.imap_unordered(
            _IndexedCall(process_one), range(n_items), chunksize=chunksize
        ):
            results[index] = value
            completed += 1
            if completed % report_every == 0 or completed == n_items:
                elapsed = time.perf_counter() - start
                print(f"  {desc}: {completed}/{n_items} ({elapsed:.1f}s)")
    print(f"  {desc} done in {time.perf_counter() - start:.1f}s.")
    return results  # type: ignore[return-value]


class _IndexedCall:
    """Picklable wrapper that returns (index, func(index)) for ordered reassembly."""

    def __init__(self, func: Callable[[int], _T]):
        self._func = func

    def __call__(self, index: int):
        return index, self._func(index)


def velocity_rtn_from_swapi_angles(
    bulk_speed_km_s: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    """Inverse of `bulk_angles_in_instrument_frame` for R = identity (RTN ≡ SWAPI)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    horizontal = bulk_speed_km_s * np.cos(el)
    return np.array(
        [
            -horizontal * np.sin(az),
            -horizontal * np.cos(az),
            -bulk_speed_km_s * np.sin(el),
        ]
    )


NOMINAL_EPOCH_TT2000 = pycdf.lib.datetime_to_tt2000(datetime(2026, 1, 1))
