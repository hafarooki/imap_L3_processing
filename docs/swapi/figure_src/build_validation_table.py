#!/usr/bin/env python3
"""Generate the integrator-validation table in `docs/swapi/solar-wind-moments.md`.

Computes |ratio - 1| where ratio = optimized / reference for the 10000
configurations in `tests/swapi/l3a/science/reference_integrals.csv`,
stratifies by reference count rate, and reports median / 95th / 99th / max
per band. Writes the markdown table in-place between the
`BEGIN: validation_table` and `END: validation_table` HTML comment markers
in `solar-wind-moments.md`.

Usage:  python docs/swapi/figure_src/build_validation_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    SWParams,
    calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_K_FACTOR

from figure_utils import load_swapi_response

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_INTEGRALS_PATH = (
    _REPO_ROOT / "tests" / "swapi" / "l3a" / "science" / "reference_integrals.csv"
)
_DOC_PATH = _REPO_ROOT / "docs" / "swapi" / "solar-wind-moments.md"
_TABLE_BEGIN = "<!-- BEGIN: validation_table"
_TABLE_END = "<!-- END: validation_table -->"

# Reference count-rate bands used for stratification.
_BANDS = [
    ("$< 0.1$", 0.0, 0.1),
    ("$0.1$ – $1$", 0.1, 1.0),
    ("$1$ – $10$", 1.0, 10.0),
    ("$10$ – $10^2$", 10.0, 100.0),
    ("$10^2$ – $10^3$", 100.0, 1000.0),
    ("$10^3$ – $10^4$", 1000.0, 1e4),
    ("$10^4$ – $10^5$", 1e4, 1e5),
    ("$\\geq 10^5$", 1e5, np.inf),
]


def _peak_voltage(bulk_speed_km_s):
    return (
        PROTON_MASS_KG
        * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _build_table(rel: np.ndarray, refs_v: np.ndarray) -> str:
    lines = [
        "| Reference (Hz)  |     N |  Median |     95% |     99% |     Max |",
        "|-----------------|-------|---------|---------|---------|---------|",
    ]
    for label, lo, hi in _BANDS:
        mask = (refs_v >= lo) & (refs_v < hi)
        n = int(mask.sum())
        if n == 0:
            lines.append(
                f"| {label:<15s} | {n:>5d} |       — |       — |       — |       — |"
            )
            continue
        band = rel[mask]
        med = np.median(band) * 100
        p95 = np.percentile(band, 95) * 100
        p99 = np.percentile(band, 99) * 100
        mx = band.max() * 100
        lines.append(
            f"| {label:<15s} | {n:>5d} | {med:>6.2f}% | {p95:>6.2f}% | {p99:>6.2f}% | {mx:>6.2f}% |"
        )
    return "\n".join(lines)


def _update_doc(table_md: str) -> None:
    text = _DOC_PATH.read_text()
    begin_idx = text.find(_TABLE_BEGIN)
    end_idx = text.find(_TABLE_END)
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise RuntimeError(
            f"Could not find '{_TABLE_BEGIN}' / '{_TABLE_END}' markers in {_DOC_PATH}"
        )
    begin_line_end = text.find("\n", begin_idx) + 1
    new_text = text[:begin_line_end] + table_md + "\n" + text[end_idx:]
    _DOC_PATH.write_text(new_text)


def main():
    print("Loading calibration data...")
    swapi_response = load_swapi_response()
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
        v_peak = peak_voltages[i]
        grid = swapi_response.create_passband_grid(v_peak)
        cs = swapi_response.central_speed(v_peak, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        cea = swapi_response.get_central_effective_area(v_peak)
        optimized[i] = calculate_integral(grid, sw, cs, cea, az_trans, az_trans_spacing)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    valid = references > 0
    refs_v = references[valid]
    rel = np.abs(optimized[valid] / refs_v - 1.0)
    # Cases where both integrators round to ~0 give meaningless ratios; clamp.
    rel = np.minimum(rel, 1.0)

    print(f"\nMax |ratio - 1|:    {rel.max():.2%}")
    print(f"Median |ratio - 1|: {np.median(rel):.2%}")
    print(f"95th pct:           {np.percentile(rel, 95):.2%}")
    print(f"99th pct:           {np.percentile(rel, 99):.2%}")

    table_md = _build_table(rel, refs_v)
    print()
    print(table_md)
    _update_doc(table_md)
    print(f"\nUpdated {_DOC_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
