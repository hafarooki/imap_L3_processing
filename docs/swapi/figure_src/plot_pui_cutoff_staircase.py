#!/usr/bin/env python3
"""
Show why the forward model imposes the f_PUI cutoff as a partial cell rather
than a hard per-node Heaviside.

At a single ESA voltage, sweep the PUI cutoff speed v_b across several v' grid
edges and plot the predicted coincidence rate three ways:

  * without correction — the cutoff node switches on/off as a whole cell, so
    the rate is a discontinuous staircase in v_b (one jump per grid edge);
  * partial cell (production) — the cell holding the cutoff is scaled by the
    fraction of its width below v_b and everything above is zeroed, so the rate
    is continuous in v_b and tracks the converged integral;
  * a fine-grid reference (N=6000) — the converged continuous integral.

The residual is a sum of per-bin rates, so the staircase's whole-cell flips
become discontinuities in it: they pin the optimum onto a grid node and leave
the parameter covariance non-positive-definite, which the fit reports as
BAD_FIT. The partial cell removes both.

Output: docs/swapi/figures/pui_cutoff_staircase.png
Usage:  uv run python docs/swapi/figure_src/plot_pui_cutoff_staircase.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import (
    FIGURES_DIR,
    NOMINAL_EPOCH_TT2000,
    load_swapi_response,
    save_figure,
)

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    calculate_coincidence_rate,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    build_chunk_collapsed_response,
    build_collapsed_response_grid,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    _filled_shell_vdf_without_cutoff,
    vasyliunas_siscoe_vdf,
)
from imap_l3_processing.swapi.l3a.utils import (
    velocity_components_to_angles_in_instrument_frame,
)
from imap_l3_processing.swapi.species import Species

_SW_SPEED_KMS = 450.0
_SW_AZIMUTH_DEG = 0.0
_SW_ELEVATION_DEG = -10.0
_IONIZATION_RATE_HZ = 2e-7
_SW_SPEED_INERTIAL_KMS = 450.0
_INFLOW_PSI_DEG = 75.0

# Grid top is the largest fit-allowed cutoff (1.2 * upstream speed), so the
# sweep below stays interior to the grid where the correction is active.
_GRID_MAX_KMS = 1.2 * _SW_SPEED_KMS
_CUTOFF_SWEEP_KMS = np.linspace(445.0, 461.0, 100)
_FINE_GRID_POINTS = 6000

# Voltage step probing the PUI shell near the cutoff, where the boundary cell is
# a large fraction of the bin's total weight (clearest staircase).
_ESA_VOLTAGE_V = 7300.8

_DENSITY_LUT_PATH = (
    REPO_ROOT / "instrument_team_data" / "swapi" / "density-of-neutral-helium-lut.dat"
)


class _FineSpeedGrid:
    """A `UniformSpeedGrid`-shaped grid with an arbitrary cell count.

    The production grid is fixed at 512 cells, so the converged reference below
    needs its own, much finer grid. This exposes the attributes
    `vasyliunas_siscoe_vdf` reads, so the reference runs through the same
    distribution code as production.
    """

    def __init__(self, min_speed_kms: float, max_speed_kms: float, size: int) -> None:
        self.centers = np.linspace(min_speed_kms, max_speed_kms, size)
        self.spacing = float(self.centers[1] - self.centers[0])
        self.size = size
        self.spherical_shell_integration_weights = self.centers**2 * self.spacing


def main():
    voltages = (
        pd.read_csv(
            REPO_ROOT / "tests" / "test_data" / "swapi" / "pui_count_rate_reference.csv"
        )
        .iloc[:, 0]
        .to_numpy()
    )
    voltage_index = int(np.argmin(np.abs(voltages - _ESA_VOLTAGE_V)))

    swapi_response = load_swapi_response()
    swapi_response.warm_cache(voltages)
    density_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(
        _DENSITY_LUT_PATH
    )

    def distribution_kwargs(cutoff_speed_kms: float) -> dict:
        return dict(
            ionization_rate=_IONIZATION_RATE_HZ,
            cutoff_speed=float(cutoff_speed_kms),
            distance=ONE_AU_IN_KM,
            inflow_angle=_INFLOW_PSI_DEG,
            solar_wind_speed_inertial_frame=_SW_SPEED_INERTIAL_KMS,
            density_of_neutral_helium_lookup_table=density_lookup_table,
        )

    azimuth_rad = np.radians(_SW_AZIMUTH_DEG)
    elevation_rad = np.radians(_SW_ELEVATION_DEG)
    bulk_vec = _SW_SPEED_KMS * np.array(
        [
            -np.cos(elevation_rad) * np.sin(azimuth_rad),
            -np.cos(elevation_rad) * np.cos(azimuth_rad),
            -np.sin(elevation_rad),
        ]
    )
    bulk_per_bin = np.broadcast_to(bulk_vec, (1, voltages.size, 3)).copy()

    chunk_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=voltages,
        bulk_sw_per_bin_kms=bulk_per_bin,
        time_as_tt2000=NOMINAL_EPOCH_TT2000,
        species=Species.HELIUM_PLUS,
        cutoff_speed_max_kms=_GRID_MAX_KMS,
    )
    coarse_grid = chunk_response.speed_grid.centers
    delta_v_prime = chunk_response.speed_grid.spacing
    bin_weights = chunk_response.bin_weights[0, voltage_index, :]

    # Fine-grid reference response for the same voltage step.
    bulk_azimuth_deg, bulk_elevation_deg = (
        velocity_components_to_angles_in_instrument_frame(
            bulk_vec[0], bulk_vec[1], bulk_vec[2]
        )
    )
    fine_speed_grid = _FineSpeedGrid(1.0, _GRID_MAX_KMS, _FINE_GRID_POINTS)
    fine_collapsed = build_collapsed_response_grid(
        swapi_response.get_response_grid(
            NOMINAL_EPOCH_TT2000, float(voltages[voltage_index]), Species.HELIUM_PLUS
        ),
        float(_SW_SPEED_KMS),
        bulk_azimuth_deg,
        bulk_elevation_deg,
        speed_in_sw_frame=fine_speed_grid.centers,
    )
    fine_weights = (
        fine_collapsed.values * fine_speed_grid.spherical_shell_integration_weights
    )

    # Below the neutral-helium LUT support the distribution is physically zero;
    # the LUT extrapolation returns non-finite values there, so clamp them.
    def finite(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        values[~np.isfinite(values)] = 0.0
        return values

    rate_without = np.empty_like(_CUTOFF_SWEEP_KMS)
    rate_with = np.empty_like(_CUTOFF_SWEEP_KMS)
    rate_reference = np.empty_like(_CUTOFF_SWEEP_KMS)
    with np.errstate(all="ignore"):
        for i, cutoff in enumerate(_CUTOFF_SWEEP_KMS):
            kwargs = distribution_kwargs(cutoff)

            # No correction: hard per-node Heaviside, whole cells flip on/off.
            # Claude: the uncorrected variant is not reachable through the
            # Claude: production entry point, so the cutoff-free distribution is
            # Claude: imported directly and the whole-cell Heaviside applied here.
            f_coarse = finite(_filled_shell_vdf_without_cutoff(coarse_grid, **kwargs))
            f_coarse[coarse_grid > cutoff] = 0.0
            rate_without[i] = bin_weights @ f_coarse
            # Production: trapezoidal partial cell, via the real forward model.
            rate_with[i] = calculate_coincidence_rate(chunk_response, **kwargs)[
                0, voltage_index
            ]
            # Converged reference: same partial-cell correction, fine grid.
            rate_reference[i] = fine_weights @ finite(
                vasyliunas_siscoe_vdf(fine_speed_grid, **kwargs)
            )

    assert np.all(np.isfinite([rate_without, rate_with, rate_reference]))

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    grid_edges = coarse_grid[
        (coarse_grid >= _CUTOFF_SWEEP_KMS[0]) & (coarse_grid <= _CUTOFF_SWEEP_KMS[-1])
    ]
    for edge in grid_edges:
        axis.axvline(edge, color="0.85", lw=0.8, zorder=0)
    axis.axvline(grid_edges[0], color="0.85", lw=0.8, zorder=0, label="$v'$ grid edges")

    axis.plot(
        _CUTOFF_SWEEP_KMS,
        rate_reference,
        "-",
        color="0.4",
        lw=1,
        label="fine-grid reference",
    )
    axis.scatter(
        _CUTOFF_SWEEP_KMS,
        rate_without,
        color="tab:red",
        s=10,
        label="no correction",
    )
    axis.scatter(
        _CUTOFF_SWEEP_KMS,
        rate_with,
        color="tab:blue",
        s=10,
        marker="x",
        label="with correction",
    )

    axis.set_xlabel(r"Cutoff speed $v_b$ [km/s]")
    axis.set_ylabel(f"Coincidence rate at {voltages[voltage_index]:.0f} V [Hz]")
    axis.set_xlim(_CUTOFF_SWEEP_KMS[0], _CUTOFF_SWEEP_KMS[-1])
    axis.annotate(
        rf"grid $\Delta v' = {delta_v_prime:.2f}$ km/s",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="0.3",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "pui_cutoff_staircase.png"
    save_figure(figure, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
