from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
    validate_azimuthal_transmission_values,
)
from imap_l3_processing.swapi.response.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.response.passband_grid import (
    PassbandGrid,
    build_passband_grid,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.species import Species


class ResponseGrid(NamedTuple):
    sg_passband: PassbandGrid
    oa_passband: PassbandGrid
    central_speed: float
    central_effective_area: float
    azimuthal_transmission: AzimuthalTransmissionGrid


class SwapiResponse:
    AZIMUTHAL_TRANSMISSION_SPACING_DEG = 0.1

    def __init__(
        self,
        azimuthal_transmission: NDArray,
        central_effective_area_at_voltage: Callable,
        passband_fit_coefficients: pd.DataFrame,
        passband_esa_voltage_limits: dict,
        efficiency_table: EfficiencyCalibrationTable,
    ):
        self._azimuthal_transmission_grid = AzimuthalTransmissionGrid(
                values=np.asarray(azimuthal_transmission, dtype=float),
                spacing=float(self.AZIMUTHAL_TRANSMISSION_SPACING_DEG),
            )
        self._central_effective_area_at_voltage = central_effective_area_at_voltage
        self._passband_fit_coefficients = passband_fit_coefficients
        self._passband_esa_voltage_limits = passband_esa_voltage_limits
        self._passband_grid_cache: dict = {}
        self._response_grid_cache: dict = {}
        self._efficiency_table = efficiency_table

    @classmethod
    def from_files(
        cls,
        *,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
        efficiency_table_path: Path,
    ) -> "SwapiResponse":
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

        central_effective_area_values = area_df["effective_area"].values
        azimuthal_transmission_values = transmission_df["transmission"].fillna(0).values
        validate_azimuthal_transmission_values(
            azimuthal_transmission_values, cls.AZIMUTHAL_TRANSMISSION_SPACING_DEG
        )
        return cls(
            azimuthal_transmission=azimuthal_transmission_values,
            central_effective_area_at_voltage=interp1d(
                area_df["esa_voltage"].values,
                central_effective_area_values,
                kind="linear",
                bounds_error=False,
                fill_value=(
                    float(central_effective_area_values[0]),
                    float(central_effective_area_values[-1]),
                ),
                assume_sorted=True,
            ),
            passband_fit_coefficients=coeffs_df,
            passband_esa_voltage_limits=esa_limits,
            efficiency_table=EfficiencyCalibrationTable(efficiency_table_path),
        )

    def _get_central_effective_area(self, esa_voltage: float) -> float:
        return float(self._central_effective_area_at_voltage(np.abs(esa_voltage)))

    def _central_speed(
        self, esa_voltage: float, mass_per_charge_m_p_per_e: float
    ) -> float:
        # formula: v_0 = sqrt(2 k* |V| (e/m_p) / mass_per_charge_m_p_per_e).

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
        for v in np.unique(np.asarray(esa_voltages, dtype=float).ravel()):
            key = self._cache_key(v)
            if np.isfinite(v) and key not in self._passband_grid_cache:
                self._passband_grid_cache[key] = dict()
                for region in ("SG", "OA"):
                    passband = build_passband_grid(self._get_passband_values(v, region))
                    self._passband_grid_cache[key][region] = passband

    def get_response_grid(
        self,
        time_as_tt2000: int,
        esa_voltage: float,
        species: Species,
    ) -> ResponseGrid:
        relative_efficiency = self._efficiency_table.relative_efficiency(time_as_tt2000, species)

        cache_key = (
            self._cache_key(float(esa_voltage)),
            species,
            self._cache_key(float(relative_efficiency)),
        )

        cached = self._response_grid_cache.get(cache_key)
        if cached is not None:
            return cached

        passband_cache_key = self._cache_key(esa_voltage)
        if passband_cache_key not in self._passband_grid_cache:
            raise KeyError(
                f"No PassbandGrid cached for ESA voltage {esa_voltage} V. "
                f"Call warm_cache([{esa_voltage}]) first."
            )
        passband_pair = self._passband_grid_cache[passband_cache_key]

        response_grid = ResponseGrid(
            sg_passband=passband_pair["SG"],
            oa_passband=passband_pair["OA"],
            central_speed=self._central_speed(esa_voltage, species.mass_per_charge_m_p_per_e),
            central_effective_area=self._get_central_effective_area(esa_voltage) * relative_efficiency,
            azimuthal_transmission=self._azimuthal_transmission_grid,
        )

        self._response_grid_cache[cache_key] = response_grid
        return response_grid

    def _get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self._passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        coeffs = self._passband_fit_coefficients.xs(region, level="region")

        # we use energy-angle passbands for modeling the coincidence rate,
        # but fits are to voltage-angle passbands as a function of natural log of beam energy
        # effectively equivalent as long as the x-axis set to is E/|V| and not E or |V|
        # but need to select the one via for this voltage via the k factor
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))

        return pd.DataFrame(values, index=coeffs.index, columns=["value"])

    def _cache_key(self, x) -> float:
        return round(float(x), 3)
