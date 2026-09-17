import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imap_l3_processing.swapi.response.passband_grid import PassbandGrid
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species
from figure_utils import (
    FIGURES_DIR,
    NOMINAL_EPOCH_TT2000,
    load_swapi_response,
    save_figure,
)

_ELEVATION_DISPLAY_LIMIT_DEG = 15.0
_ACTIVE_ELEVATION_SAMPLE_COUNT = 300


def main():
    swapi_response = load_swapi_response()
    esa_voltages = representative_esa_voltages(swapi_response)
    swapi_response.warm_cache(esa_voltages)

    n_columns = len(esa_voltages)
    n_rows = 2
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(5 * n_columns, 4 * n_rows), sharey=True
    )

    last_image = None
    for column, esa_voltage in enumerate(esa_voltages):
        response_grid = swapi_response.get_response_grid(
            NOMINAL_EPOCH_TT2000, esa_voltage, Species.PROTON
        )
        central_speed = response_grid.central_speed

        for row, region in enumerate(["open_aperture", "sunglasses"]):
            grid = (
                response_grid.oa_passband
                if region == "open_aperture"
                else response_grid.sg_passband
            )
            axis = axes[row, column]
            last_image = plot_region_panel(
                axis, grid, region, esa_voltage, central_speed
            )
            if column == 0:
                axis.set_ylabel("Elevation (deg)")

    figure.tight_layout()
    colorbar = figure.colorbar(last_image, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Passband value")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "passband_boundaries.png"
    save_figure(figure, output_path)
    print(f"Saved {output_path}")


def representative_esa_voltages(swapi_response: SwapiResponse) -> list[float]:
    voltage_limits = swapi_response._passband_esa_voltage_limits
    voltage_minimum = min(low for low, _ in voltage_limits.values())
    voltage_maximum = max(high for _, high in voltage_limits.values())
    voltage_geometric_mean = float(np.sqrt(voltage_minimum * voltage_maximum))
    return [voltage_minimum, voltage_geometric_mean, voltage_maximum]


def plot_region_panel(
    axis,
    grid: PassbandGrid,
    region: str,
    esa_voltage: float,
    central_speed: float,
):
    label = "Open Aperture (OA)" if region == "open_aperture" else "Sunglasses (SG)"

    elevations, speed_ratios = grid_axis_coordinates(grid, grid.values)

    active_elevations = np.linspace(
        grid.elevation_range[0], grid.elevation_range[1], _ACTIVE_ELEVATION_SAMPLE_COUNT
    )
    lower_speed_ratios = _vectorized_bracket(
        grid, grid.min_boundary, active_elevations, np.minimum
    )
    upper_speed_ratios = _vectorized_bracket(
        grid, grid.max_boundary, active_elevations, np.maximum
    )

    image = draw_transmission_heatmap(axis, grid.values, elevations, speed_ratios)
    draw_integration_window_outline(
        axis, active_elevations, lower_speed_ratios, upper_speed_ratios
    )

    axis.set_xlabel("Speed ratio (v / v_central)")
    axis.set_title(f"{label}  |  {esa_voltage:.1f} V  ({central_speed:.0f} km/s)")
    axis.set_ylim(-_ELEVATION_DISPLAY_LIMIT_DEG, _ELEVATION_DISPLAY_LIMIT_DEG)
    return image


def grid_axis_coordinates(
    grid: PassbandGrid, transmission_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_elevations, n_speed_ratios = transmission_values.shape
    elevations = grid.min_elevation + np.arange(n_elevations) * grid.elevation_spacing
    speed_ratios = (
        grid.min_speed_ratio + np.arange(n_speed_ratios) * grid.speed_ratio_spacing
    )
    return elevations, speed_ratios


def draw_transmission_heatmap(
    axis,
    transmission_values: np.ndarray,
    elevations: np.ndarray,
    speed_ratios: np.ndarray,
):
    extent = [speed_ratios[0], speed_ratios[-1], elevations[0], elevations[-1]]
    return axis.imshow(
        transmission_values,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gist_heat",
        interpolation="nearest",
    )


def _vectorized_bracket(
    grid: PassbandGrid, boundary: np.ndarray, elevations: np.ndarray, combine
) -> np.ndarray:
    """Per-elevation speed-ratio boundary as `combine` (np.minimum or np.maximum)
    of the two grid rows whose elevations bracket each query. Boundaries are
    indexed on the grid's uniform elevation axis (same as `interpolate_passband`);
    off-grid queries clamp to row 0 / row n-1."""
    n_rows = boundary.shape[0]
    last = n_rows - 1
    i_float = (elevations - grid.min_elevation) / grid.elevation_spacing
    lower = np.clip(np.floor(i_float).astype(int), 0, last)
    upper = np.clip(lower + 1, 0, last)
    return combine(boundary[lower], boundary[upper])


def draw_integration_window_outline(
    axis,
    active_elevations: np.ndarray,
    lower_speed_ratios: np.ndarray,
    upper_speed_ratios: np.ndarray,
):
    closed_x = np.concatenate(
        [lower_speed_ratios, upper_speed_ratios[::-1], [lower_speed_ratios[0]]]
    )
    closed_y = np.concatenate(
        [active_elevations, active_elevations[::-1], [active_elevations[0]]]
    )
    axis.plot(closed_x, closed_y, color="tab:blue", linewidth=2.5)


if __name__ == "__main__":
    main()
