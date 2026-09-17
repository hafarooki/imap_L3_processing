#!/usr/bin/env python3
"""
Plot ground-truth vs optimized count-rate spectra for several illustrative SW
configurations chosen to exercise the integrator's edges:
  - cold (narrow Maxwellian, dynamic-limit speed window collapses to a sliver)
  - hot (broad Maxwellian, fills the passband)
  - bulk elevation past the SG passband edge (per-region elevation clamping)
  - bulk azimuth straddling the SG/OA boundary (multi-region azimuth split)
  - fast solar wind (passband shape at high beam energy)

Each panel sweeps ESA voltage across the proton peak and overlays the dynamic-
limit JIT integrator (calculate_integral) against the fixed-limit, high-resolution
reference integrator (reference_integral_fixed_limits).

Output: docs/swapi/figures/spectra.png
Usage:  python docs/swapi/figure_src/plot_spectra.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    EV_TO_KELVIN,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.species import Species
from scripts.swapi.reference_integral import reference_integral_fixed_limits
from figure_utils import (
    FIGURES_DIR,
    NOMINAL_EPOCH_TT2000,
    velocity_rtn_from_swapi_angles,
    load_swapi_response,
    peak_esa_voltage_for_proton_bulk_speed,
    save_figure,
)


# (label, bulk_speed, T_K, bulk_azimuth, bulk_elevation, density)
CASES = [
    (
        "Nominal: 450 km/s, T = 116,045 K, on-axis",
        450,
        10.0 * EV_TO_KELVIN,
        0.0,
        0.0,
        5.0,
    ),
    ("Cold: 450 km/s, T = 11,605 K, on-axis", 450, 1.0 * EV_TO_KELVIN, 0.0, 0.0, 5.0),
    (
        "Hot: 450 km/s, T = 1,160,452 K, on-axis",
        450,
        100.0 * EV_TO_KELVIN,
        0.0,
        0.0,
        5.0,
    ),
    (
        "Off-axis elevation: $\\theta_b = 9\\degree$ (past SG edge)",
        450,
        10.0 * EV_TO_KELVIN,
        0.0,
        9.0,
        5.0,
    ),
    (
        "Off-axis azimuth: $\\phi_b = 18\\degree$ (near SG/OA edge)",
        450,
        10.0 * EV_TO_KELVIN,
        18.0,
        0.0,
        5.0,
    ),
    (
        "Fast SW: 700 km/s, T = 232,090 K, on-axis",
        700,
        20.0 * EV_TO_KELVIN,
        0.0,
        0.0,
        5.0,
    ),
]


def main():
    print("Loading calibration data...")
    swapi_response = load_swapi_response()

    n_voltages = 60
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    rotation_matrix = np.eye(3)

    handles_for_legend = None
    for ax, (label, v_b, T_k, az, el, density) in zip(axes.flat, CASES):
        v_peak = peak_esa_voltage_for_proton_bulk_speed(v_b)
        esa_voltages = np.logspace(
            np.log10(0.4 * v_peak), np.log10(2.5 * v_peak), n_voltages
        )
        sw = SolarWindParams(
            density=density,
            velocity_rtn=velocity_rtn_from_swapi_angles(v_b, az, el),
            temperature=T_k,
            mass=PROTON_MASS_KG,
        )

        optimized = np.empty(n_voltages)
        reference = np.empty(n_voltages)
        swapi_response.warm_cache(esa_voltages)
        for i, v in enumerate(esa_voltages):
            response_grid = swapi_response.get_response_grid(
                NOMINAL_EPOCH_TT2000, float(v), Species.PROTON
            )
            rate, _ = calculate_integral(sw, response_grid, rotation_matrix)
            optimized[i] = rate
            reference[i] = reference_integral_fixed_limits(
                response_grid, sw, rotation_matrix
            )

        (h_ref,) = ax.plot(
            esa_voltages,
            reference,
            "k-",
            linewidth=2,
            label="Ground truth (fixed limits)",
        )
        (h_prod,) = ax.plot(
            esa_voltages,
            optimized,
            "o",
            color="tab:orange",
            markersize=4,
            markerfacecolor="none",
            markeredgewidth=1.2,
            label="Optimized (dynamic limits)",
        )
        h_peak = ax.axvline(
            v_peak,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
            label="$v_b$ central voltage",
        )
        if handles_for_legend is None:
            handles_for_legend = [h_ref, h_prod, h_peak]

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ESA voltage (V)")
        ax.set_ylabel("Count rate (Hz)")
        ax.set_title(label, fontsize=10)

        peak_rate = max(reference.max(), optimized.max(), 1.0)
        ax.set_ylim(peak_rate * 1e-6, peak_rate * 3)
        ax.grid(True, which="both", alpha=0.3)

        nz = reference > 1.0
        if nz.any():
            rel = np.abs(optimized[nz] - reference[nz]) / reference[nz]
            print(f"  {label}: max |rel err| above 1 Hz = {rel.max():.2%}")

    fig.suptitle(
        "Production vs ground-truth count-rate spectra for representative SW configurations",
        fontsize=13,
    )
    fig.legend(
        handles=handles_for_legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "spectra.png"
    save_figure(fig, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
