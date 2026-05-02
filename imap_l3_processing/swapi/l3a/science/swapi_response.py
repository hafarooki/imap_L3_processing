from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR

_TARGET_ELEVATIONS = np.arange(-15, 15 + 0.5, 0.5)
_TARGET_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)
# Fraction of the global grid max below which the passband is treated as zero
# for integration-boundary purposes. The polynomial response is strictly
# positive everywhere it is defined, so without a threshold the boundary would
# be V-independent (set only by CSV coverage). With this threshold the
# boundary tightens around the physically significant region and varies with V.
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
    # Voltage-independent bilinear-nonzero elevation range per region (min, max).
    # Reflects (CSV-stored elevation extent) ± half target spacing, clamped to target grid.
    oa_active_el_range: tuple
    sg_active_el_range: tuple


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


def eval_boundary_min(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the min (left) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — always at or outside the passband."""
    idx = np.clip(
        np.searchsorted(boundary[0], elevations, side="right") - 1,
        0,
        boundary.shape[1] - 1,
    )
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.minimum(boundary[1][idx], boundary[1][idx_next])


def eval_boundary_max(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the max (right) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — always at or outside the passband."""
    idx = np.clip(
        np.searchsorted(boundary[0], elevations, side="right") - 1,
        0,
        boundary.shape[1] - 1,
    )
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.maximum(boundary[1][idx], boundary[1][idx_next])


def _passband_boundaries(
    grid_values: NDArray, target_elevations: NDArray, target_speed_ratios: NDArray
) -> tuple[NDArray, NDArray]:
    """Return (min_boundary, max_boundary) each shape (2, n_active):
    row 0 = elevation values, row 1 = boundary speed ratio (one grid step outside the
    region where the passband exceeds `_PASSBAND_BOUNDARY_THRESHOLD * max(grid)`)."""
    spacing = float(target_speed_ratios[1] - target_speed_ratios[0])
    cutoff = _PASSBAND_BOUNDARY_THRESHOLD * float(grid_values.max())
    active_elevations, min_ratios, max_ratios = [], [], []
    for elev, row in zip(target_elevations, grid_values):
        above = target_speed_ratios[row > cutoff]
        if len(above) > 0:
            active_elevations.append(float(elev))
            min_ratios.append(float(above[0]) - spacing)
            max_ratios.append(float(above[-1]) + spacing)
    elevs = np.array(active_elevations)
    return np.vstack([elevs, np.array(min_ratios)]), np.vstack(
        [elevs, np.array(max_ratios)]
    )


def _active_el_range(min_boundary: NDArray, target_elevations: NDArray) -> tuple:
    """Bilinear-nonzero elevation range = active row range ± half target-spacing,
    clamped to target grid extent. V-dependent: an elevation row is "active" only if
    at least one of its passband values exceeds the boundary threshold (see
    `_passband_boundaries`)."""
    if min_boundary.shape[1] == 0:
        return (float(target_elevations[0]), float(target_elevations[0]))
    target_spacing = float(target_elevations[1] - target_elevations[0])
    min_active = float(min_boundary[0, 0])
    max_active = float(min_boundary[0, -1])
    lo = max(float(target_elevations[0]), min_active - target_spacing)
    hi = min(float(target_elevations[-1]), max_active + target_spacing)
    return (lo, hi)


@dataclass
class SWAPIResponse:
    azimuthal_transmission: (
        NDArray  # shape (N,), evenly spaced at 0.1 deg intervals from 0
    )
    central_effective_area_voltage: NDArray  # shape (M,), ESA voltages in V
    central_effective_area: NDArray  # shape (M,), effective area in cm^2
    passband_fit_coefficients: (
        pd.DataFrame
    )  # index: (region, energy_ratio, elevation), columns: [2, 1, 0]
    passband_esa_voltage_limits: dict  # {region: (min_esa_voltage, max_esa_voltage)}
    _grid_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def get_central_effective_area(self, esa_voltage: float) -> float:
        # np.interp clamps out-of-range inputs to the endpoint values.
        return float(
            np.interp(
                np.abs(esa_voltage),
                self.central_effective_area_voltage,
                self.central_effective_area,
            )
        )

    def get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self.passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        coeffs = self.passband_fit_coefficients.xs(region, level="region")
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))
        return pd.DataFrame(values, index=coeffs.index, columns=["value"])

    # Azimuthal transmission table is sampled at 0.1 deg spacing; constant for all V/species.
    AZIMUTHAL_TRANSMISSION_SPACING_DEG = 0.1

    def central_speed(
        self, esa_voltage: float, mass_per_charge_m_p_per_e: float
    ) -> float:
        """Central proton-frame speed (km/s) at the given ESA voltage for a species
        whose mass-per-charge is `mass_per_charge_m_p_per_e` (1 for proton, ≈2 for alpha):
            v_0 = sqrt(2 k* |V| (e/m_p) / mass_per_charge_m_p_per_e).
        """
        from imap_l3_processing.constants import (
            METERS_PER_KILOMETER,
            PROTON_CHARGE_OVER_MASS_C_PER_KG,
        )

        return float(
            np.sqrt(
                2.0
                * SWAPI_K_FACTOR
                * abs(esa_voltage)
                * PROTON_CHARGE_OVER_MASS_C_PER_KG
                / float(mass_per_charge_m_p_per_e)
            )
            / METERS_PER_KILOMETER
        )

    def warm_cache(self, esa_voltages) -> None:
        """Build and cache PassbandGrids for every unique finite voltage in `esa_voltages`.

        This is the only path that builds grids. Call this in the parent process before
        forking workers so the ~1.8 ms pandas pivot is paid once per voltage and forked
        workers inherit the populated cache via COW rather than rebuilding independently.
        Calling with a voltage already in the cache is a no-op.
        """
        for v in np.unique(np.asarray(esa_voltages, dtype=float).ravel()):
            key = float(v)
            if np.isfinite(v) and key not in self._grid_cache:
                self._grid_cache[key] = self._build_passband_grid(key)

    def create_passband_grid(self, esa_voltage: float) -> PassbandGrid:
        """Return the cached PassbandGrid for `esa_voltage`.

        Raises KeyError if `warm_cache` was not called for this voltage. The cache must
        be populated before any call to this method — call `warm_cache(voltages)` first.
        """
        cache_key = float(esa_voltage)
        try:
            return self._grid_cache[cache_key]
        except KeyError:
            raise KeyError(
                f"No PassbandGrid cached for ESA voltage {esa_voltage} V. "
                f"Call warm_cache([{esa_voltage}]) before create_passband_grid."
            ) from None

    def _build_passband_grid(self, esa_voltage: float) -> PassbandGrid:
        """Build a PassbandGrid from scratch for the given voltage (no caching)."""
        oa_values = self.get_passband_values(esa_voltage, "OA")
        sg_values = self.get_passband_values(esa_voltage, "SG")

        oa_grid = _build_passband_array(
            oa_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
        )
        sg_grid = _build_passband_array(
            sg_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
        )

        # Mask cells below the threshold to zero so production and reference
        # integrals see the same passband (the polynomial-fit "tail" below
        # threshold is treated as instrument-team-flagged artifact).
        oa_grid = np.where(
            oa_grid >= _PASSBAND_BOUNDARY_THRESHOLD * float(oa_grid.max()),
            oa_grid,
            0.0,
        )
        sg_grid = np.where(
            sg_grid >= _PASSBAND_BOUNDARY_THRESHOLD * float(sg_grid.max()),
            sg_grid,
            0.0,
        )

        min_OA_boundary, max_OA_boundary = _passband_boundaries(
            oa_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
        )
        min_SG_boundary, max_SG_boundary = _passband_boundaries(
            sg_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
        )
        oa_active_el_range = _active_el_range(min_OA_boundary, _TARGET_ELEVATIONS)
        sg_active_el_range = _active_el_range(min_SG_boundary, _TARGET_ELEVATIONS)

        return PassbandGrid(
            min_elevation=float(_TARGET_ELEVATIONS[0]),
            elevation_spacing=float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
            min_speed_ratio=float(_TARGET_SPEED_RATIOS[0]),
            speed_ratio_spacing=float(
                _TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]
            ),
            values_open_aperture=oa_grid,
            values_sunglasses=sg_grid,
            min_OA_boundary=min_OA_boundary,
            max_OA_boundary=max_OA_boundary,
            min_SG_boundary=min_SG_boundary,
            max_SG_boundary=max_SG_boundary,
            oa_active_el_range=oa_active_el_range,
            sg_active_el_range=sg_active_el_range,
        )

    @classmethod
    def from_files(
        cls,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
    ) -> "SWAPIResponse":
        transmission_df = pd.read_csv(azimuthal_transmission_path)
        area_df = pd.read_csv(central_effective_area_path)
        coeffs_df = pd.read_csv(
            passband_fit_coefficients_path,
            index_col=["region", "energy_ratio", "elevation"],
        )
        limit_cols = ["min_esa_voltage", "max_esa_voltage"]
        limits_df = coeffs_df[limit_cols]
        coeffs_df = coeffs_df.drop(columns=limit_cols)
        esa_limits = {
            region: (
                float(limits_df.xs(region, level="region")["min_esa_voltage"].iloc[0]),
                float(limits_df.xs(region, level="region")["max_esa_voltage"].iloc[0]),
            )
            for region in limits_df.index.get_level_values("region").unique()
        }
        coeffs_df.columns = coeffs_df.columns.astype(int)

        return cls(
            azimuthal_transmission=transmission_df["transmission"].fillna(0).values,
            central_effective_area_voltage=area_df["esa_voltage"].values,
            central_effective_area=area_df["effective_area"].values,
            passband_fit_coefficients=coeffs_df,
            passband_esa_voltage_limits=esa_limits,
        )
