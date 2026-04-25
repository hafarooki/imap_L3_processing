#!/usr/bin/env python3
"""
Plot the SWAPI passband at a 2 keV beam energy using SWAPIResponse.create_passband_grid.

Output: docs/swapi/figures/passband_2keV.png
Usage:  python docs/swapi/figure_src/plot_passband_2kev.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imap_l3_processing.swapi.l3a.science.swapi_response import (
    SWAPI_K_FACTOR, SWAPIResponse, eval_boundary_min, eval_boundary_max,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_BEAM_ENERGY_EV = 2000.0


def main():
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal_transmission.csv",
        _INSTRUMENT_DATA / "imap_swapi_central_effective_area.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband_fit_coefficients.csv",
    )

    esa_voltage = _BEAM_ENERGY_EV / SWAPI_K_FACTOR
    grid = swapi_response.create_passband_grid(esa_voltage)

    n_elev, n_speed = grid.values_open_aperture.shape
    elevations = grid.min_elevation + np.arange(n_elev) * grid.elevation_spacing
    speed_ratios = grid.min_speed_ratio + np.arange(n_speed) * grid.speed_ratio_spacing

    extent = [speed_ratios[0], speed_ratios[-1], elevations[0], elevations[-1]]
    vmax = max(grid.values_open_aperture.max(), grid.values_sunglasses.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, values, bnd_min, bnd_max, title in [
        (axes[0], grid.values_open_aperture, grid.min_OA_boundary, grid.max_OA_boundary, "Open Aperture (OA)"),
        (axes[1], grid.values_sunglasses, grid.min_SG_boundary, grid.max_SG_boundary, "Sunglasses (SG)"),
    ]:
        im = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", vmin=0, vmax=vmax)
        elev_active = np.linspace(bnd_min[0, 0], bnd_min[0, -1], 200)
        ax.plot(eval_boundary_min(bnd_min, elev_active), elev_active, color="white", linewidth=1.2, label="boundary")
        ax.plot(eval_boundary_max(bnd_max, elev_active), elev_active, color="white", linewidth=1.2)
        ax.set_xlabel("Speed ratio (v / v_central)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("Elevation (deg)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Passband value")

    fig.suptitle(
        f"SWAPI passband at {_BEAM_ENERGY_EV / 1000:.1f} keV "
        f"(ESA voltage = {esa_voltage:.1f} V, central speed = {grid.central_speed:.1f} km/s)",
        fontsize=11,
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "passband_2keV.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
