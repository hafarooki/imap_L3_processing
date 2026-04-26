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

from imap_l3_processing.constants import METERS_PER_KILOMETER, PROTON_CHARGE_COULOMBS, PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    SWParams, calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from tests.swapi.l3a.science.reference_integral import reference_integral_fixed_limits

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"


def _thermal_speed(temperature_ev):
    return float(np.sqrt(temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER)


def _peak_voltage(bulk_speed_km_s):
    return float(PROTON_MASS_KG * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
                 / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS))


# (label, bulk_speed, T_eV, bulk_azimuth, bulk_elevation, density)
CASES = [
    ("Nominal: 450 km/s, T = 10 eV, on-axis",                450, 10.0,  0.0, 0.0, 5.0),
    ("Cold: 450 km/s, T = 1 eV, on-axis",                    450,  1.0,  0.0, 0.0, 5.0),
    ("Hot: 450 km/s, T = 100 eV, on-axis",                   450, 100.0, 0.0, 0.0, 5.0),
    ("Off-axis elevation: $\\theta_b = 9\\degree$ (past SG edge)", 450, 10.0,  0.0, 9.0, 5.0),
    ("Off-axis azimuth: $\\phi_b = 18\\degree$ (near SG/OA edge)", 450, 10.0, 18.0, 0.0, 5.0),
    ("Fast SW: 700 km/s, T = 20 eV, on-axis",                700, 20.0,  0.0, 0.0, 5.0),
]


def main():
    print("Loading calibration data...")
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-passband-fit-coefficients_20260425_v001.csv",
    )

    n_voltages = 60
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    handles_for_legend = None
    for ax, (label, v_b, T_ev, az, el, density) in zip(axes.flat, CASES):
        v_peak = _peak_voltage(v_b)
        esa_voltages = np.logspace(np.log10(0.4 * v_peak), np.log10(2.5 * v_peak), n_voltages)
        sw = SWParams(
            density=density, bulk_speed=v_b, bulk_azimuth=az,
            bulk_elevation=el, thermal_speed=_thermal_speed(T_ev),
        )

        optimized = np.empty(n_voltages)
        reference = np.empty(n_voltages)
        for i, v in enumerate(esa_voltages):
            grid = swapi_response.create_passband_grid(float(v))
            optimized[i] = calculate_integral(grid, sw)
            reference[i] = reference_integral_fixed_limits(grid, sw)

        h_ref, = ax.plot(esa_voltages, reference, 'k-', linewidth=2, label='Ground truth (fixed limits)')
        h_prod, = ax.plot(esa_voltages, optimized, 'o', color='tab:orange', markersize=4,
                          markerfacecolor='none', markeredgewidth=1.2, label='Optimized (dynamic limits)')
        h_peak = ax.axvline(v_peak, color='gray', linestyle=':', linewidth=0.8, alpha=0.7,
                            label='$v_b$ central voltage')
        if handles_for_legend is None:
            handles_for_legend = [h_ref, h_prod, h_peak]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("ESA voltage (V)")
        ax.set_ylabel("Count rate (Hz)")
        ax.set_title(label, fontsize=10)

        peak_rate = max(reference.max(), optimized.max(), 1.0)
        ax.set_ylim(peak_rate * 1e-6, peak_rate * 3)
        ax.grid(True, which='both', alpha=0.3)

        nz = reference > 1.0
        if nz.any():
            rel = np.abs(optimized[nz] - reference[nz]) / reference[nz]
            print(f"  {label}: max |rel err| above 1 Hz = {rel.max():.2%}")

    fig.suptitle("Production vs ground-truth count-rate spectra for representative SW configurations",
                 fontsize=13)
    fig.legend(handles=handles_for_legend, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01),
               fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "spectra.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
