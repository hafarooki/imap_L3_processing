#!/usr/bin/env python3
"""
Plot the SWAPI passband and integration boundaries at three representative ESA voltages
for each region (Open Aperture and Sunglasses).

Output: docs/swapi/figures/passband_boundaries.png
Usage:  python docs/swapi/figure_src/plot_passband_boundaries.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from imap_l3_processing.swapi.l3a.science.swapi_response import (
    SWAPIResponse, eval_boundary_min, eval_boundary_max,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_REGIONS = [
    ("Open Aperture (OA)", "min_OA_boundary", "max_OA_boundary", "values_open_aperture"),
    ("Sunglasses (SG)",    "min_SG_boundary", "max_SG_boundary", "values_sunglasses"),
]


def _plot_passband(ax, grid, values, bnd_min, bnd_max, label):
    n_elev, n_speed = values.shape
    elevations = grid.min_elevation + np.arange(n_elev) * grid.elevation_spacing
    speed_ratios = grid.min_speed_ratio + np.arange(n_speed) * grid.speed_ratio_spacing
    extent = [speed_ratios[0], speed_ratios[-1], elevations[0], elevations[-1]]

    im = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                   cmap="gist_heat", norm=mcolors.LogNorm(vmin=1e-3, vmax=1))

    sr_lo, sr_hi = speed_ratios[0], speed_ratios[-1]
    el_lo, el_hi = elevations[0], elevations[-1]
    elev_active = np.linspace(bnd_min[0, 0], bnd_min[0, -1], 300)
    min_vals = eval_boundary_min(bnd_min, elev_active)
    max_vals = eval_boundary_max(bnd_max, elev_active)

    ax.set_facecolor("black")
    ax.fill_betweenx(elev_active, sr_lo, min_vals, color="black")
    ax.fill_betweenx(elev_active, max_vals, sr_hi, color="black")
    ax.axhspan(el_lo, bnd_min[0, 0], color="black")
    ax.axhspan(bnd_min[0, -1], el_hi, color="black")

    x_closed = np.concatenate([min_vals, max_vals[::-1], [min_vals[0]]])
    y_closed = np.concatenate([elev_active, elev_active[::-1], [elev_active[0]]])
    ax.plot(x_closed, y_closed, color="tab:blue", linewidth=2.5)

    ax.set_xlabel("Speed ratio (v / v_central)")
    ax.set_title(label)
    ax.set_ylim(-15, 15)
    return im


def main():
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal_transmission.csv",
        _INSTRUMENT_DATA / "imap_swapi_central_effective_area.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband_fit_coefficients.csv",
    )

    all_limits = swapi_response.passband_esa_voltage_limits
    v_min = min(lo for lo, _ in all_limits.values())
    v_max = max(hi for _, hi in all_limits.values())
    v_mid = np.sqrt(v_min * v_max)
    esa_voltages = [v_min, v_mid, v_max]

    n_voltages = len(esa_voltages)
    n_regions = len(_REGIONS)
    fig, axes = plt.subplots(n_regions, n_voltages, figsize=(5 * n_voltages, 4 * n_regions), sharey=True)

    for col, esa_voltage in enumerate(esa_voltages):
        grid = swapi_response.create_passband_grid(esa_voltage)

        for row, (region_label, bnd_min_attr, bnd_max_attr, values_attr) in enumerate(_REGIONS):
            values = getattr(grid, values_attr)
            bnd_min = getattr(grid, bnd_min_attr)
            bnd_max = getattr(grid, bnd_max_attr)

            ax = axes[row, col]
            title = f"{region_label}  |  {esa_voltage:.1f} V  ({grid.central_speed:.0f} km/s)"
            im = _plot_passband(ax, grid, values, bnd_min, bnd_max, title)
            if col == 0:
                ax.set_ylabel("Elevation (deg)")

    fig.suptitle(
        "SWAPI passband and integration region at three representative ESA voltages",
        fontsize=12,
    )
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Passband value")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "passband_boundaries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
