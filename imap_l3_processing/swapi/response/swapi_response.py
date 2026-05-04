from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.passband_grid import (
    PassbandGrid,
    build_passband_grid,
)


@dataclass
class SWAPIResponse:
    azimuthal_transmission: NDArray  # shape (N,), evenly spaced at 0.1 deg intervals from 0
    central_effective_area_voltage: NDArray  # shape (M,), ESA voltages in V
    central_effective_area: NDArray  # shape (M,), effective area in cm^2
    passband_fit_coefficients: pd.DataFrame  # index: (region, energy_ratio, elevation), columns: [2, 1, 0]
    passband_esa_voltage_limits: dict  # {region: (min_esa_voltage, max_esa_voltage)}
    _grid_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # Azimuthal transmission table is sampled at 0.1 deg spacing; constant for all V/species.
    AZIMUTHAL_TRANSMISSION_SPACING_DEG = 0.1

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

    def get_central_effective_area(self, esa_voltage: float) -> float:
        # np.interp clamps out-of-range inputs to the endpoint values.
        return float(
            np.interp(
                np.abs(esa_voltage),
                self.central_effective_area_voltage,
                self.central_effective_area,
            )
        )

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

    def create_response_grid(
        self,
        esa_voltage: float,
        mass_per_charge_m_p_per_e: float,
        central_effective_area_scale: float = 1.0,
    ) -> "ResponseGrid":
        from imap_l3_processing.swapi.response.response_grid import ResponseGrid

        return ResponseGrid(
            passband_grid=self.create_passband_grid(esa_voltage),
            central_speed=self.central_speed(esa_voltage, mass_per_charge_m_p_per_e),
            central_effective_area=(
                self.get_central_effective_area(esa_voltage)
                * float(central_effective_area_scale)
            ),
            azimuthal_transmission=np.asarray(self.azimuthal_transmission, dtype=float),
            azimuthal_transmission_spacing=float(self.AZIMUTHAL_TRANSMISSION_SPACING_DEG),
        )

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
        return build_passband_grid(
            oa_values=self._get_passband_values(esa_voltage, "OA"),
            sg_values=self._get_passband_values(esa_voltage, "SG"),
        )

    def _get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self.passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        coeffs = self.passband_fit_coefficients.xs(region, level="region")
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))
        return pd.DataFrame(values, index=coeffs.index, columns=["value"])
