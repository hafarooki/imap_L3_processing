from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SWAPI_K_FACTOR = 1.89  # eV/V

_PASSBAND_LIMIT_POLY_DEG = 5
_TARGET_ELEVATIONS = np.arange(-12, 11, 1.0)
_TARGET_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)


class PassbandGrid(NamedTuple):
    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    central_effective_area: float
    central_speed: float
    values_sunglasses: NDArray
    values_open_aperture: NDArray
    azimuthal_transmission: NDArray
    azimuthal_transmission_spacing: float
    min_OA_poly: NDArray
    max_OA_poly: NDArray
    min_SG_poly: NDArray
    max_SG_poly: NDArray
    min_SG_elevation: float
    max_SG_elevation: float
    min_OA_elevation: float
    max_OA_elevation: float


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


def _fit_passband_limits(grid_values: NDArray, target_elevations: NDArray,
                         target_speed_ratios: NDArray) -> tuple[NDArray, NDArray]:
    spacing = float(target_speed_ratios[1] - target_speed_ratios[0])
    fallback = float(target_speed_ratios[len(target_speed_ratios) // 2])
    min_ratios = []
    max_ratios = []
    for row in grid_values:
        above = target_speed_ratios[row > 0]
        if len(above) > 0:
            min_ratios.append(float(above[0]) - spacing)
            max_ratios.append(float(above[-1]) + spacing)
        else:
            min_ratios.append(fallback)
            max_ratios.append(fallback)
    min_poly = np.polyfit(target_elevations, min_ratios, _PASSBAND_LIMIT_POLY_DEG)
    max_poly = np.polyfit(target_elevations, max_ratios, _PASSBAND_LIMIT_POLY_DEG)
    return min_poly, max_poly


def _elevation_fov_limits(grid_values: NDArray, target_elevations: NDArray) -> tuple[float, float]:
    spacing = float(target_elevations[1] - target_elevations[0])
    nonzero_mask = grid_values.max(axis=1) > 0
    nonzero_elevations = target_elevations[nonzero_mask]
    if len(nonzero_elevations) == 0:
        center = float(target_elevations[len(target_elevations) // 2])
        return center, center
    return float(nonzero_elevations[0] - spacing), float(nonzero_elevations[-1] + spacing)


@dataclass
class SWAPIResponse:
    azimuthal_transmission: NDArray  # shape (N,), evenly spaced at 0.1 deg intervals from 0
    central_effective_area_voltage: NDArray  # shape (M,), ESA voltages in V
    central_effective_area: NDArray  # shape (M,), effective area in cm^2
    passband_fit_coefficients: pd.DataFrame  # index: (region, energy_ratio, elevation), columns: [2, 1, 0]
    passband_esa_voltage_limits: dict  # {region: (min_esa_voltage, max_esa_voltage)}

    def get_central_effective_area(self, esa_voltage: float) -> float:
        return float(np.interp(np.abs(esa_voltage), self.central_effective_area_voltage, self.central_effective_area))

    def get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self.passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        coeffs = self.passband_fit_coefficients.xs(region, level='region')
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))
        return pd.DataFrame(values, index=coeffs.index, columns=['value'])

    def create_passband_grid(self, esa_voltage: float) -> PassbandGrid:
        from imap_l3_processing.swapi.l3a.science.speed_calculation import esa_voltage_to_proton_speed

        central_speed = float(esa_voltage_to_proton_speed(esa_voltage))
        central_effective_area = self.get_central_effective_area(esa_voltage)

        oa_values = self.get_passband_values(esa_voltage, 'OA')
        sg_values = self.get_passband_values(esa_voltage, 'SG')

        oa_grid = _build_passband_array(oa_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        sg_grid = _build_passband_array(sg_values, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)

        min_oa_poly, max_oa_poly = _fit_passband_limits(oa_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        min_sg_poly, max_sg_poly = _fit_passband_limits(sg_grid, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS)
        min_oa_elev, max_oa_elev = _elevation_fov_limits(oa_grid, _TARGET_ELEVATIONS)
        min_sg_elev, max_sg_elev = _elevation_fov_limits(sg_grid, _TARGET_ELEVATIONS)

        return PassbandGrid(
            min_elevation=float(_TARGET_ELEVATIONS[0]),
            elevation_spacing=float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
            min_speed_ratio=float(_TARGET_SPEED_RATIOS[0]),
            speed_ratio_spacing=float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]),
            central_effective_area=central_effective_area,
            central_speed=central_speed,
            values_open_aperture=oa_grid,
            values_sunglasses=sg_grid,
            azimuthal_transmission=self.azimuthal_transmission,
            azimuthal_transmission_spacing=0.1,
            min_OA_poly=min_oa_poly,
            max_OA_poly=max_oa_poly,
            min_SG_poly=min_sg_poly,
            max_SG_poly=max_sg_poly,
            min_SG_elevation=min_sg_elev,
            max_SG_elevation=max_sg_elev,
            min_OA_elevation=min_oa_elev,
            max_OA_elevation=max_oa_elev,
        )

    @classmethod
    def from_files(cls, azimuthal_transmission_path: Path, central_effective_area_path: Path,
                   passband_fit_coefficients_path: Path) -> 'SWAPIResponse':
        transmission_df = pd.read_csv(azimuthal_transmission_path)
        area_df = pd.read_csv(central_effective_area_path)
        coeffs_df = pd.read_csv(passband_fit_coefficients_path, index_col=['region', 'energy_ratio', 'elevation'])
        limit_cols = ['min_esa_voltage', 'max_esa_voltage']
        if all(c in coeffs_df.columns for c in limit_cols):
            limits_df = coeffs_df[limit_cols]
            coeffs_df = coeffs_df.drop(columns=limit_cols)
            esa_limits = {
                region: (
                    float(limits_df.xs(region, level='region')['min_esa_voltage'].iloc[0]),
                    float(limits_df.xs(region, level='region')['max_esa_voltage'].iloc[0]),
                )
                for region in limits_df.index.get_level_values('region').unique()
            }
        else:
            esa_limits = {}
        coeffs_df.columns = coeffs_df.columns.astype(int)

        return cls(
            azimuthal_transmission=transmission_df['transmission'].fillna(0).values,
            central_effective_area_voltage=area_df['esa_voltage'].values,
            central_effective_area=area_df['effective_area'].values,
            passband_fit_coefficients=coeffs_df,
            passband_esa_voltage_limits=esa_limits,
        )
