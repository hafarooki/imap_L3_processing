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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"


def _load_csv(path):
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    ea_rows = _load_csv(_INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv")
    voltages = np.array([float(r["esa_voltage"]) for r in ea_rows])
    eff_area = np.array([float(r["effective_area"]) for r in ea_rows])

    tx_rows = _load_csv(_INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv")
    azimuths = np.array([float(r["abs_azimuth"]) for r in tx_rows])
    transmission = np.array([float(r["transmission"]) if r["transmission"] else 0.0 for r in tx_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(voltages, eff_area, color="steelblue", linewidth=1.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("ESA voltage (V)")
    ax1.set_ylabel("Central effective area (cm²)")
    ax1.set_title("Central Effective Area $\\mathcal{A}_0(V)$")
    ax1.grid(True, which="both", alpha=0.3)

    # Full azimuth range with symmetric +/- display
    full_az = np.concatenate([-azimuths[::-1], azimuths])
    full_tx = np.concatenate([transmission[::-1], transmission])
    ax2.semilogy(full_az, np.where(full_tx > 0, full_tx, np.nan), color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Azimuth angle (deg)")
    ax2.set_ylabel("Transmission $T(\\phi)$")
    ax2.set_title("Azimuthal Transmission $T(\\phi)$")
    ax2.axvspan(-20, 20, alpha=0.12, color="steelblue", label="Sunglasses (SG)")
    ax2.axvspan(20, 150, alpha=0.12, color="darkorange", label="Open aperture (OA)")
    ax2.axvspan(-150, -20, alpha=0.12, color="darkorange")
    ax2.set_xlim(-180, 180)
    ax2.set_xticks([-150, -90, -20, 0, 20, 90, 150])
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "calibration_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
