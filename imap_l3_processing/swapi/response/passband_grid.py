from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR

_TARGET_ELEVATIONS = np.arange(-15, 15 + 0.5, 0.5)
_TARGET_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)
# Fraction of the global grid max at which the integration window terminates
# (per-elevation speed-ratio bounds in `_passband_boundaries`, elevation range
# in `_elevation_range`). The polynomial response is strictly positive
# everywhere it is defined, so without a relative cutoff the boundary would be
# V-independent (set only by CSV coverage). With this cutoff the integration
# tightens around the physically significant region and varies with V.
_PASSBAND_BOUNDARY_THRESHOLD = 1e-2


class PassbandGrid(NamedTuple):
    """V-only passband geometry. Species- and time-dependent quantities (central speed,
    central effective area, azimuthal transmission) are tracked separately on
    `SWAPIResponse` and passed alongside this grid into `calculate_integral`."""

    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values_sunglasses: NDArray
    values_open_aperture: NDArray
    min_OA_boundary: (
        NDArray  # shape (2, n): row 0 = elevations, row 1 = min speed ratios
    )
    max_OA_boundary: (
        NDArray  # shape (2, n): row 0 = elevations, row 1 = max speed ratios
    )
    min_SG_boundary: NDArray
    max_SG_boundary: NDArray
    # V-dependent elevation range per region (min, max). Bound at the
    # linear-interp threshold-crossing elevation between the outermost active row
    # (max passband > 1% of grid max) and its inactive neighbor.
    oa_elevation_range: tuple
    sg_elevation_range: tuple


def eval_boundary_min(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the min (left) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — guaranteed to bracket the
    active passband speed range."""
    idx = np.clip(
        np.searchsorted(boundary[0], elevations, side="right") - 1,
        0,
        boundary.shape[1] - 1,
    )
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.minimum(boundary[1][idx], boundary[1][idx_next])


def eval_boundary_max(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the max (right) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — guaranteed to bracket the
    active passband speed range."""
    idx = np.clip(
        np.searchsorted(boundary[0], elevations, side="right") - 1,
        0,
        boundary.shape[1] - 1,
    )
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.maximum(boundary[1][idx], boundary[1][idx_next])


def build_passband_grid(
    oa_values: pd.DataFrame, sg_values: pd.DataFrame
) -> PassbandGrid:
    """Build a PassbandGrid from per-region passband-value DataFrames (as returned by
    `SWAPIResponse.get_passband_values`)."""
    oa_grid = _build_passband_array(oa_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
    sg_grid = _build_passband_array(sg_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)

    min_OA_boundary, max_OA_boundary = _passband_boundaries(
        oa_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
    )
    min_SG_boundary, max_SG_boundary = _passband_boundaries(
        sg_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
    )
    oa_elevation_range = _elevation_range(oa_grid, _TARGET_ELEVATIONS)
    sg_elevation_range = _elevation_range(sg_grid, _TARGET_ELEVATIONS)

    return PassbandGrid(
        min_elevation=float(_TARGET_ELEVATIONS[0]),
        elevation_spacing=float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
        min_speed_ratio=float(_TARGET_SPEED_RATIOS[0]),
        speed_ratio_spacing=float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]),
        values_open_aperture=oa_grid,
        values_sunglasses=sg_grid,
        min_OA_boundary=min_OA_boundary,
        max_OA_boundary=max_OA_boundary,
        min_SG_boundary=min_SG_boundary,
        max_SG_boundary=max_SG_boundary,
        oa_elevation_range=oa_elevation_range,
        sg_elevation_range=sg_elevation_range,
    )


def _build_passband_array(
    values_df: pd.DataFrame, target_elevations: NDArray, target_speed_ratios: NDArray
) -> NDArray:
    pivot = values_df.reset_index()
    pivot["speed_ratio"] = np.sqrt(pivot["energy_ratio"] / SWAPI_K_FACTOR)
    pivot = (
        pivot.drop(columns="energy_ratio")
        .set_index(["elevation", "speed_ratio"])["value"]
        .unstack("speed_ratio")
    )
    pivot = pivot.fillna(0.0)

    src_speed_ratios = pivot.columns.values
    result = np.zeros((len(target_elevations), len(target_speed_ratios)))
    for i, elev in enumerate(target_elevations):
        if elev in pivot.index:
            result[i] = np.interp(
                target_speed_ratios,
                src_speed_ratios,
                pivot.loc[elev].values,
                left=0.0,
                right=0.0,
            )
    return result.copy(order="C")


def _passband_boundaries(
    grid_values: NDArray, target_elevations: NDArray, target_speed_ratios: NDArray
) -> tuple[NDArray, NDArray]:
    """Return (min_boundary, max_boundary) each shape (2, n_active):
    row 0 = elevation values, row 1 = the first speed-ratio pixel at or below
    `_PASSBAND_BOUNDARY_THRESHOLD * max(grid)` outside the active row. Falls back
    to the edge cell's speed ratio when the active region touches the grid edge."""
    cutoff = _PASSBAND_BOUNDARY_THRESHOLD * float(grid_values.max())
    n_speed = grid_values.shape[1]
    active_elevations, min_ratios, max_ratios = [], [], []
    for elev, row in zip(target_elevations, grid_values):
        above_idx = np.where(row > cutoff)[0]
        if len(above_idx) == 0:
            continue
        i_lo = int(above_idx[0])
        i_hi = int(above_idx[-1])
        active_elevations.append(float(elev))
        if i_lo > 0:
            min_ratios.append(float(target_speed_ratios[i_lo - 1]))
        else:
            min_ratios.append(float(target_speed_ratios[i_lo]))
        if i_hi < n_speed - 1:
            max_ratios.append(float(target_speed_ratios[i_hi + 1]))
        else:
            max_ratios.append(float(target_speed_ratios[i_hi]))
    elevs = np.array(active_elevations)
    return np.vstack([elevs, np.array(min_ratios)]), np.vstack(
        [elevs, np.array(max_ratios)]
    )


def _elevation_range(grid_values: NDArray, target_elevations: NDArray) -> tuple:
    """Elevation range bracketing the rows whose max passband exceeds
    `_PASSBAND_BOUNDARY_THRESHOLD * max(grid)`, with the outer bounds at the linear-
    interp threshold-crossing elevation between the outermost active row (with
    row-max `M_a > cutoff`) and its inactive neighbor (`M_b ≤ cutoff`):
    `θ_threshold = θ_a ± spacing · (M_a - cutoff) / (M_a - M_b)`. Approximates the
    max-over-speed of the bilinear interp by the linear interp of per-row maxes,
    which is exact when the max-cell index aligns between the two rows (true for
    the unimodal SWAPI passband). Falls back to the edge row's elevation when the
    active region touches the grid edge."""
    cutoff = _PASSBAND_BOUNDARY_THRESHOLD * float(grid_values.max())
    row_max = grid_values.max(axis=1)
    active_idx = np.where(row_max > cutoff)[0]
    if len(active_idx) == 0:
        return (float(target_elevations[0]), float(target_elevations[0]))
    target_spacing = float(target_elevations[1] - target_elevations[0])
    n_el = len(target_elevations)
    i_lo = int(active_idx[0])
    i_hi = int(active_idx[-1])
    M_lo = float(row_max[i_lo])
    M_hi = float(row_max[i_hi])
    if i_lo > 0:
        M_lo_minus = float(row_max[i_lo - 1])
        lo = float(target_elevations[i_lo]) - target_spacing * (M_lo - cutoff) / (
            M_lo - M_lo_minus
        )
    else:
        lo = float(target_elevations[i_lo])
    if i_hi < n_el - 1:
        M_hi_plus = float(row_max[i_hi + 1])
        hi = float(target_elevations[i_hi]) + target_spacing * (M_hi - cutoff) / (
            M_hi - M_hi_plus
        )
    else:
        hi = float(target_elevations[i_hi])
    return (lo, hi)
