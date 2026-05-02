from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR

_TARGET_ELEVATIONS = np.arange(-15, 15 + 0.5, 0.5)
_TARGET_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)


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
    min_OA_boundary: NDArray  # shape (2, n): row 0 = elevations, row 1 = min speed ratios
    max_OA_boundary: NDArray  # shape (2, n): row 0 = elevations, row 1 = max speed ratios
    min_SG_boundary: NDArray
    max_SG_boundary: NDArray
    # Voltage-independent bilinear-nonzero elevation range per region (min, max).
    # Reflects (CSV-stored elevation extent) ± half target spacing, clamped to target grid.
    oa_active_el_range: tuple
    sg_active_el_range: tuple


def _build_passband_array(values_df: pd.DataFrame, target_elevations: NDArray,
                          target_speed_ratios: NDArray) -> NDArray:
    pivot = values_df.reset_index()
    pivot['speed_ratio'] = np.sqrt(pivot['energy_ratio'] / SWAPI_K_FACTOR)
    pivot = pivot.drop(columns='energy_ratio').set_index(['elevation', 'speed_ratio'])['value'].unstack('speed_ratio')
    pivot = pivot.fillna(0.0)

    src_speed_ratios = pivot.columns.values
    result = np.zeros((len(target_elevations), len(target_speed_ratios)))
    for i, elev in enumerate(target_elevations):
        if elev in pivot.index:
            result[i] = np.interp(target_speed_ratios, src_speed_ratios, pivot.loc[elev].values, left=0.0, right=0.0)
    return result.copy(order='C')


def eval_boundary_min(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the min (left) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — always at or outside the passband."""
    idx = np.clip(np.searchsorted(boundary[0], elevations, side='right') - 1, 0, boundary.shape[1] - 1)
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.minimum(boundary[1][idx], boundary[1][idx_next])


def eval_boundary_max(boundary: NDArray, elevations: NDArray) -> NDArray:
    """Evaluate the max (right) speed-ratio boundary at each elevation using the most
    expansive of the two nearest stored grid points — always at or outside the passband."""
    idx = np.clip(np.searchsorted(boundary[0], elevations, side='right') - 1, 0, boundary.shape[1] - 1)
    idx_next = np.clip(idx + 1, 0, boundary.shape[1] - 1)
    return np.maximum(boundary[1][idx], boundary[1][idx_next])


def _passband_boundaries(grid_values: NDArray, target_elevations: NDArray,
                         target_speed_ratios: NDArray) -> tuple[NDArray, NDArray]:
    """Return (min_boundary, max_boundary) each shape (2, n_active):
    row 0 = elevation values, row 1 = boundary speed ratio (one grid step outside nonzero region)."""
    spacing = float(target_speed_ratios[1] - target_speed_ratios[0])
    active_elevations, min_ratios, max_ratios = [], [], []
    for elev, row in zip(target_elevations, grid_values):
        above = target_speed_ratios[row > 0]
        if len(above) > 0:
            active_elevations.append(float(elev))
            min_ratios.append(float(above[0]) - spacing)
            max_ratios.append(float(above[-1]) + spacing)
    elevs = np.array(active_elevations)
    return np.vstack([elevs, np.array(min_ratios)]), np.vstack([elevs, np.array(max_ratios)])


def _active_el_range(min_boundary: NDArray, target_elevations: NDArray) -> tuple:
    """Bilinear-nonzero elevation range = active row range ± half target-spacing,
    clamped to target grid extent. Voltage-independent (depends only on which CSV
    elevations have entries)."""
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
    azimuthal_transmission: NDArray  # shape (N,), evenly spaced at 0.1 deg intervals from 0
    central_effective_area_voltage: NDArray  # shape (M,), ESA voltages in V
    central_effective_area: NDArray  # shape (M,), effective area in cm^2
    passband_fit_coefficients: pd.DataFrame  # index: (region, energy_ratio, elevation), columns: [2, 1, 0]
    passband_esa_voltage_limits: dict  # {region: (min_esa_voltage, max_esa_voltage)}
    # Voltage-independent passband geometry, precomputed once at load time and shared
    # across every PassbandGrid (set by from_files; defaults are for tests that build
    # a SWAPIResponse directly without going through from_files).
    _min_OA_boundary: NDArray = field(default_factory=lambda: np.empty((2, 0)))
    _max_OA_boundary: NDArray = field(default_factory=lambda: np.empty((2, 0)))
    _min_SG_boundary: NDArray = field(default_factory=lambda: np.empty((2, 0)))
    _max_SG_boundary: NDArray = field(default_factory=lambda: np.empty((2, 0)))
    _oa_active_el_range: tuple = (-12.0, 10.5)
    _sg_active_el_range: tuple = (-10.5, 7.0)
    _grid_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def get_central_effective_area(self, esa_voltage: float) -> float:
        return float(np.interp(np.abs(esa_voltage), self.central_effective_area_voltage, self.central_effective_area))

    def get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self.passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        coeffs = self.passband_fit_coefficients.xs(region, level='region')
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))
        return pd.DataFrame(values, index=coeffs.index, columns=['value'])

    # Azimuthal transmission table is sampled at 0.1 deg spacing; constant for all V/species.
    AZIMUTHAL_TRANSMISSION_SPACING_DEG = 0.1

    def central_speed(self, esa_voltage: float, mass_per_charge_m_p_per_e: float) -> float:
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
                2.0 * SWAPI_K_FACTOR * abs(esa_voltage)
                * PROTON_CHARGE_OVER_MASS_C_PER_KG / float(mass_per_charge_m_p_per_e)
            ) / METERS_PER_KILOMETER
        )

    def create_passband_grid(self, esa_voltage: float) -> PassbandGrid:
        """Build a V-only PassbandGrid (passband-shape arrays + boundaries + active-el ranges).

        Species- and time-dependent quantities — central speed, central effective area,
        azimuthal transmission — are tracked separately on this `SWAPIResponse` and must be
        passed alongside the grid when calling `calculate_integral`. Cache key is V only.

        Cached because pandas pivot/unstack inside `_build_passband_array` dominates this
        function (~1.8 ms per call), and fits typically reuse the same 72 ESA voltages
        across many sweeps within one processor run.
        """
        cache_key = float(esa_voltage)
        cached = self._grid_cache.get(cache_key)
        if cached is not None:
            return cached

        oa_values = self.get_passband_values(esa_voltage, 'OA')
        sg_values = self.get_passband_values(esa_voltage, 'SG')

        oa_grid = _build_passband_array(oa_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        sg_grid = _build_passband_array(sg_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)

        grid = PassbandGrid(
            min_elevation=float(_TARGET_ELEVATIONS[0]),
            elevation_spacing=float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
            min_speed_ratio=float(_TARGET_SPEED_RATIOS[0]),
            speed_ratio_spacing=float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]),
            values_open_aperture=oa_grid,
            values_sunglasses=sg_grid,
            min_OA_boundary=self._min_OA_boundary,
            max_OA_boundary=self._max_OA_boundary,
            min_SG_boundary=self._min_SG_boundary,
            max_SG_boundary=self._max_SG_boundary,
            oa_active_el_range=self._oa_active_el_range,
            sg_active_el_range=self._sg_active_el_range,
        )
        self._grid_cache[cache_key] = grid
        return grid

    @classmethod
    def from_files(cls, azimuthal_transmission_path: Path, central_effective_area_path: Path,
                   passband_fit_coefficients_path: Path) -> 'SWAPIResponse':
        transmission_df = pd.read_csv(azimuthal_transmission_path)
        area_df = pd.read_csv(central_effective_area_path)
        coeffs_df = pd.read_csv(passband_fit_coefficients_path, index_col=['region', 'energy_ratio', 'elevation'])
        limit_cols = ['min_esa_voltage', 'max_esa_voltage']
        limits_df = coeffs_df[limit_cols]
        coeffs_df = coeffs_df.drop(columns=limit_cols)
        esa_limits = {
            region: (
                float(limits_df.xs(region, level='region')['min_esa_voltage'].iloc[0]),
                float(limits_df.xs(region, level='region')['max_esa_voltage'].iloc[0]),
            )
            for region in limits_df.index.get_level_values('region').unique()
        }
        coeffs_df.columns = coeffs_df.columns.astype(int)

        response = cls(
            azimuthal_transmission=transmission_df['transmission'].fillna(0).values,
            central_effective_area_voltage=area_df['esa_voltage'].values,
            central_effective_area=area_df['effective_area'].values,
            passband_fit_coefficients=coeffs_df,
            passband_esa_voltage_limits=esa_limits,
        )

        # Precompute voltage-independent passband geometry (boundaries + active el ranges).
        # Any in-range voltage gives the same result because np.exp(np.polyval(...)) is
        # always strictly positive, so a row's "active" speed_ratios are determined solely
        # by which (energy_ratio, elevation) pairs exist in the CSV.
        rep_voltage = float(np.median(list(esa_limits['OA'])))
        oa_grid = _build_passband_array(
            response.get_passband_values(rep_voltage, 'OA'),
            _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        sg_grid = _build_passband_array(
            response.get_passband_values(rep_voltage, 'SG'),
            _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        response._min_OA_boundary, response._max_OA_boundary = _passband_boundaries(
            oa_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        response._min_SG_boundary, response._max_SG_boundary = _passband_boundaries(
            sg_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        response._oa_active_el_range = _active_el_range(response._min_OA_boundary, _TARGET_ELEVATIONS)
        response._sg_active_el_range = _active_el_range(response._min_SG_boundary, _TARGET_ELEVATIONS)
        return response
