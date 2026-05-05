import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imap_l3_processing.swapi.response.passband_grid import (
    PassbandGrid,
    eval_boundary_max,
    eval_boundary_min,
)
from imap_l3_processing.swapi.response.swapi_response import SWAPIResponse
from figure_utils import load_swapi_response

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "swapi" / "figures"

ELEVATION_DISPLAY_LIMIT_DEG = 15.0
ACTIVE_ELEVATION_SAMPLE_COUNT = 300


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
        grid = swapi_response.create_passband_grid(esa_voltage)
        central_speed = swapi_response.central_speed(esa_voltage, 1.0)

        for row, region in enumerate(["open_aperture", "sunglasses"]):
            axis = axes[row, column]
            last_image = plot_region_panel(
                axis, grid, region, esa_voltage, central_speed
            )
            if column == 0:
                axis.set_ylabel("Elevation (deg)")

    figure.tight_layout()
    colorbar = figure.colorbar(last_image, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Passband value")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "passband_boundaries.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


def representative_esa_voltages(swapi_response: SWAPIResponse) -> list[float]:
    # min/max across both regions and geometric mean
    voltage_limits = swapi_response.passband_esa_voltage_limits
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
    if region == "open_aperture":
        label = "Open Aperture (OA)"
        transmission_values = grid.values_open_aperture
        lower_boundary = grid.min_OA_boundary
        upper_boundary = grid.max_OA_boundary
    elif region == "sunglasses":
        label = "Sunglasses (SG)"
        transmission_values = grid.values_sunglasses
        lower_boundary = grid.min_SG_boundary
        upper_boundary = grid.max_SG_boundary

    elevations, speed_ratios = grid_axis_coordinates(grid, transmission_values)

    active_elevations = np.linspace(
        lower_boundary[0, 0], lower_boundary[0, -1], ACTIVE_ELEVATION_SAMPLE_COUNT
    )
    lower_speed_ratios = eval_boundary_min(lower_boundary, active_elevations)
    upper_speed_ratios = eval_boundary_max(upper_boundary, active_elevations)

    image = draw_transmission_heatmap(
        axis, transmission_values, elevations, speed_ratios
    )
    draw_integration_window_outline(
        axis, active_elevations, lower_speed_ratios, upper_speed_ratios
    )

    axis.set_xlabel("Speed ratio (v / v_central)")
    axis.set_title(f"{label}  |  {esa_voltage:.1f} V  ({central_speed:.0f} km/s)")
    axis.set_ylim(-ELEVATION_DISPLAY_LIMIT_DEG, ELEVATION_DISPLAY_LIMIT_DEG)
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
