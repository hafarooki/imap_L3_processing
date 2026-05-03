#!/usr/bin/env python3
"""
Histogram of (optimized / reference) ratios across 10000 random SW conditions,
stacked by reference count rate magnitude.

Reference: fixed-limit reference integrals from tests/.../reference_integrals.csv
Optimized: dynamic-limit integral at the N values defined in
           calculate_proton_solar_wind_moments.

Output: docs/swapi/figures/reference_vs_optimized_histogram.png
Usage:  python scripts/swapi/reference_vs_optimized_histogram.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    SWParams,
    calculate_integral,
    N_ELEVATION,
    N_AZIMUTH,
    N_SPEED,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"
_REFERENCE_INTEGRALS_PATH = (
    _REPO_ROOT / "tests" / "swapi" / "l3a" / "science" / "reference_integrals.csv"
)


def _peak_voltage(bulk_speed_km_s):
    return (
        PROTON_MASS_KG
        * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def main():
    print("Loading calibration data...")
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    df = pd.read_csv(_REFERENCE_INTEGRALS_PATH)
    references = df["integral"].to_numpy()
    optimized = np.empty(len(df))

    az_trans = np.asarray(swapi_response.azimuthal_transmission, dtype=float)
    az_trans_spacing = float(swapi_response.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    print(f"Warming passband cache for {len(df)} rows...")
    peak_voltages = [
        _peak_voltage(float(row.bulk_speed)) for row in df.itertuples(index=False)
    ]
    swapi_response.warm_cache(peak_voltages)

    print(f"Computing {len(df)} optimized integrals...")
    for i, row in enumerate(df.itertuples(index=False)):
        thermal_speed = float(
            np.sqrt(row.temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        sw = SWParams(
            density=float(row.density),
            bulk_speed=float(row.bulk_speed),
            bulk_azimuth=float(row.bulk_azimuth),
            bulk_elevation=float(row.bulk_elevation),
            thermal_speed=thermal_speed,
        )
        v_peak = _peak_voltage(float(row.bulk_speed))
        grid = swapi_response.create_passband_grid(v_peak)
        cs = swapi_response.central_speed(v_peak, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        cea = swapi_response.get_central_effective_area(v_peak)
        optimized[i] = calculate_integral(grid, sw, cs, cea, az_trans, az_trans_spacing)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    # Ratio (optimized / reference); skip cases where reference is exactly zero.
    valid = references > 0
    ratios = np.where(valid, optimized / np.maximum(references, 1e-300), np.nan)
    ratios = ratios[valid]
    references_valid = references[valid]

    rel_errors = np.abs(ratios - 1.0)
    print(f"\nMax |ratio - 1|:   {rel_errors.max():.2%}")
    print(f"Median |ratio - 1|: {np.median(rel_errors):.2%}")
    print(f"95th pct |ratio - 1|: {np.percentile(rel_errors, 95):.2%}")

    # Bin references by magnitude (decades) and bin ratios.
    reference_edges = [0, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, np.inf]
    reference_labels = [
        "< 0.1",
        "0.1–1",
        "1–10",
        "10–10²",
        "10²–10³",
        "10³–10⁴",
        "10⁴–10⁵",
        "≥ 10⁵",
    ]
    reference_bin_idx = (
        np.digitize(references_valid, reference_edges) - 1
    )  # 0..n_reference_bins-1
    n_reference_bins = len(reference_labels)

    # Ratio bins centered on 1.0, log-spaced symmetric around 1
    ratio_edges = np.array(
        [
            0.0,
            0.5,
            0.7,
            0.85,
            0.92,
            0.96,
            0.98,
            0.99,
            1.0,
            1.01,
            1.02,
            1.04,
            1.08,
            1.15,
            1.30,
            1.5,
            2.0,
            np.inf,
        ]
    )
    ratio_centers = 0.5 * (ratio_edges[:-1] + ratio_edges[1:])
    n_ratio_bins = len(ratio_edges) - 1

    # Build (n_ratio_bins, n_reference_bins) count matrix
    counts = np.zeros((n_ratio_bins, n_reference_bins), dtype=int)
    for r, t in zip(ratios, reference_bin_idx):
        ri = np.clip(
            np.searchsorted(ratio_edges, r, side="right") - 1, 0, n_ratio_bins - 1
        )
        if 0 <= t < n_reference_bins:
            counts[ri, t] += 1

    # Plot stacked bars with separation between bins
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = matplotlib.colormaps["viridis"]
    bar_colors = [
        cmap(i / max(1, n_reference_bins - 1)) for i in range(n_reference_bins)
    ]
    x = np.arange(n_ratio_bins)
    bottom = np.zeros(n_ratio_bins)
    bar_width = 0.78  # < 1.0 leaves a visible gap between adjacent bins
    for j in range(n_reference_bins):
        ax.bar(
            x,
            counts[:, j],
            width=bar_width,
            bottom=bottom,
            color=bar_colors[j],
            label=reference_labels[j],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += counts[:, j]

    # Tick labels are the ratio bin edges (e.g., "0.96–0.98")
    bin_label = lambda lo, hi: (
        f"<{hi:g}"
        if lo == 0
        else f">{lo:g}"
        if not np.isfinite(hi)
        else f"{lo:g}–{hi:g}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [bin_label(ratio_edges[i], ratio_edges[i + 1]) for i in range(n_ratio_bins)],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    # Highlight the "ratio = 1" bin
    one_idx = np.searchsorted(ratio_edges, 1.0) - 1
    ax.axvline(
        one_idx + 0.5,
        color="red",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="ratio = 1",
    )

    ax.set_xlabel("Ratio (optimized / reference)")
    ax.set_ylabel("Number of cases")
    ax.legend(title="Reference count rate (Hz)", loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Annotate each bar with its total count.
    totals = counts.sum(axis=1)
    y_pad = totals.max() * 0.012
    for i, total in enumerate(totals):
        if total == 0:
            continue
        ax.text(
            i,
            total + y_pad,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "reference_vs_optimized_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
