#!/usr/bin/env python3
"""Generate the integrator-validation scatter plot for `docs/swapi/solar-wind-forward-model.md`.

Computes the optimized JIT integrator (`calculate_integral`) for every row in
`reference_integrals.csv`, then plots optimized vs. reference count rate on
log-log axes with the 1:1 line, coloring points by relative-error band
(within ±1%, 1–5%, and >5%).

Output: docs/swapi/figures/validation_scatter.png
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.species import Species

from figure_utils import (
    ANCHOR_ROTATION_SWAPI_TO_RTN,
    FIGURES_DIR,
    NOMINAL_EPOCH_TT2000,
    REPO_ROOT,
    load_swapi_response,
    peak_esa_voltage_for_proton_bulk_speed,
    run_parallel_map,
    save_figure,
)

_worker_state: types.SimpleNamespace | None = None

_REFERENCE_INTEGRALS_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "reference_integrals.csv"
)
_OUTPUT_PATH = FIGURES_DIR / "validation_scatter.png"


def _initialize_worker_state(
    rows: list[tuple[float, float, float, float, float]],
    peak_voltages: np.ndarray,
    swapi_response,
) -> None:
    global _worker_state
    _worker_state = types.SimpleNamespace(
        rows=rows,
        peak_voltages=peak_voltages,
        swapi_response=swapi_response,
        rotation_matrix=ANCHOR_ROTATION_SWAPI_TO_RTN,
    )


def _process_one(i: int) -> float:
    state = _worker_state
    v_r, v_t, v_n, density, temperature_k = state.rows[i]
    sw = SolarWindParams(
        density=density,
        velocity_rtn=np.array([v_r, v_t, v_n]),
        temperature=temperature_k,
        mass=PROTON_MASS_KG,
    )
    response_grid = state.swapi_response.get_response_grid(
        NOMINAL_EPOCH_TT2000, state.peak_voltages[i], Species.PROTON
    )
    return calculate_integral(sw, response_grid, state.rotation_matrix)[0]


def _plot_scatter(reference: np.ndarray, optimized: np.ndarray) -> None:
    positive = (reference > 0) & (optimized > 0)
    reference_positive = reference[positive]
    optimized_positive = optimized[positive]
    relative_error = np.abs(optimized_positive / reference_positive - 1.0)
    within_band = relative_error <= 0.01
    mid_band = (relative_error > 0.01) & (relative_error <= 0.05)
    outside_band = relative_error > 0.05

    axis_min = min(reference_positive.min(), optimized_positive.min())
    axis_max = max(reference_positive.max(), optimized_positive.max())
    line = np.array([axis_min, axis_max])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        reference_positive[within_band],
        optimized_positive[within_band],
        s=6,
        alpha=0.6,
        color="0.25",
        edgecolor="none",
        label=f"within ±1% ({within_band.sum()})",
        zorder=2,
    )
    ax.scatter(
        reference_positive[mid_band],
        optimized_positive[mid_band],
        s=8,
        alpha=0.75,
        color="tab:orange",
        edgecolor="none",
        label=f"1–5% error ({mid_band.sum()})",
        zorder=3,
    )
    ax.scatter(
        reference_positive[outside_band],
        optimized_positive[outside_band],
        s=10,
        alpha=0.8,
        color="tab:red",
        edgecolor="none",
        label=f">5% error ({outside_band.sum()})",
        zorder=4,
    )
    ax.plot(
        line,
        line,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="1:1",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference count rate (Hz)")
    ax.set_ylabel("Optimized count rate (Hz)")
    ax.set_title(
        "Optimized vs. reference integrator across 10,000 random SW configurations"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_figure(fig, _OUTPUT_PATH)
    print(f"Saved {_OUTPUT_PATH.relative_to(REPO_ROOT)}")


def main():
    print("Loading calibration data...")
    swapi_response = load_swapi_response()
    df = pd.read_csv(_REFERENCE_INTEGRALS_PATH)
    references = df["integral_hz"].to_numpy()
    rows = [
        (
            float(row.v_R_km_s),
            float(row.v_T_km_s),
            float(row.v_N_km_s),
            float(row.density_cm3),
            float(row.temperature_K),
        )
        for row in df.itertuples(index=False)
    ]
    bulk_speeds = df["bulk_speed_km_s"].to_numpy(dtype=float)

    print(f"Warming passband cache for {len(df)} rows...")
    peak_voltages = np.array(
        [peak_esa_voltage_for_proton_bulk_speed(v) for v in bulk_speeds]
    )
    swapi_response.warm_cache(peak_voltages)
    for unique_voltage in np.unique(peak_voltages):
        swapi_response.get_response_grid(
            NOMINAL_EPOCH_TT2000, float(unique_voltage), Species.PROTON
        )

    _initialize_worker_state(rows, peak_voltages, swapi_response)
    optimized = np.array(
        run_parallel_map(_process_one, len(df), desc="integrals", chunksize=64)
    )

    valid = (references > 0) & (optimized > 0)
    rel = np.abs(optimized[valid] / references[valid] - 1.0)
    print(f"\nMax  |ratio - 1|: {rel.max():.2%}")
    print(f"99th |ratio - 1|: {np.percentile(rel, 99):.2%}")
    print(f"95th |ratio - 1|: {np.percentile(rel, 95):.2%}")
    print(f"Median           : {np.median(rel):.2%}")
    print(f"Fraction within ±1%: {(rel <= 0.01).mean():.2%}")

    _plot_scatter(references, optimized)


if __name__ == "__main__":
    main()
