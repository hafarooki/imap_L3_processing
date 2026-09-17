#!/usr/bin/env python3
"""
Plot the SWAPI central effective area and azimuthal transmission calibration curves.

Output: docs/swapi/figures/calibration_curves.png
Usage:  python docs/swapi/figure_src/plot_calibration_curves.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import FIGURES_DIR, load_swapi_response, save_figure

_AREA_CSV = (
    Path(__file__).resolve().parents[3]
    / "instrument_team_data"
    / "swapi"
    / "imap_swapi_central-effective-area_20260425_v001.csv"
)


def main():
    swapi_response = load_swapi_response()

    area_df = pd.read_csv(_AREA_CSV)
    voltages = area_df["esa_voltage"].to_numpy()
    eff_area = area_df["effective_area"].to_numpy()

    transmission_grid = swapi_response._azimuthal_transmission_grid
    transmission = transmission_grid.values
    azimuths = np.arange(len(transmission)) * transmission_grid.spacing

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(voltages, eff_area, color="steelblue", linewidth=1.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("ESA Voltage ($|V|$) [V]")
    ax1.set_ylabel("Central Effective Area $\\mathcal{A}_0(V)$ (cm²)")
    ax1.grid(True, which="both", alpha=0.3)

    full_az = np.concatenate([-azimuths[::-1], azimuths])
    full_tx = np.concatenate([transmission[::-1], transmission])
    ax2.semilogy(
        full_az,
        full_tx,
        color="darkorange",
        linewidth=1.5,
    )
    ax2.set_xlabel(r"Azimuth Angle ($\phi$) [$^\circ$]")
    ax2.set_ylabel("Azimuthal Transmission $T(\\phi)$")
    ax2.axvspan(-20, 20, alpha=0.12, color="steelblue", label="Sunglasses (SG)")
    ax2.axvspan(20, 150, alpha=0.12, color="darkorange", label="Open aperture (OA)")
    ax2.axvspan(-150, -20, alpha=0.12, color="darkorange")
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(1e-5, 1.5e0)
    ax2.set_xticks([-150, -90, -20, 0, 20, 90, 150])
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "calibration_curves.png"
    save_figure(fig, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
