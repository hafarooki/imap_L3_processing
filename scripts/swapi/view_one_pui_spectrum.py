"""View one chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production-model (replayed) rate.

Loads the cached pickle output of fit_and_plot_pui.py (observed and
production-model spectrograms, fit parameters) and plots the selected chunk's
sweep-averaged spectrum above per-sweep spectrograms of the observed and
modeled rates over the PUI fit energy window. The modeled rate already
includes the constant instrument background, so it is directly comparable to
the observed coincidence rate.

Usage:
    scripts/swapi/view_one_pui_spectrum.py <YYYY-MM-DD> <HH:MM[:SS]>

Selects the 50-sweep chunk whose central epoch is nearest to the requested
date and time. Run `fit_and_plot_pui.py <date>` first to populate the pickle
caches; everything plotted here comes from those caches, so no SDC access is
needed.
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from spacepy.pycdf import lib as cdf_library

from imap_processing.swapi.l2 import swapi_l2

from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_L2_K_FACTOR,
    SWAPI_PUI_COOLING_INDEX,
)

argument_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
argument_parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
argument_parser.add_argument(
    "time",
    help="UTC time in HH:MM[:SS] format; picks the 50-sweep "
    "chunk whose central epoch is nearest.",
)
argument_parser.add_argument(
    "--output-path",
    type=Path,
    default=None,
    help="If set, save the figure to this path and skip plt.show().",
)
arguments = argument_parser.parse_args()


def _parse_hh_mm_ss(time_string: str):
    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_string, time_format).time()
        except ValueError:
            continue
    raise SystemExit(f"Could not parse time {time_string!r}; expected HH:MM[:SS].")


target_date = datetime.strptime(arguments.date, "%Y-%m-%d")
target_datetime_utc = datetime.combine(
    target_date.date(), _parse_hh_mm_ss(arguments.time)
)
compact_date = target_date.strftime("%Y%m%d")
fit_cache_path = Path(f"/tmp/swapi_pui_fit_{compact_date}.pkl")
spectrogram_cache_path = Path(
    f"/tmp/swapi_pui_spectrograms_per_sweep_{compact_date}.pkl"
)
if not fit_cache_path.exists() or not spectrogram_cache_path.exists():
    sys.exit(
        f"Missing pickle caches for {arguments.date}. "
        f"Run scripts/swapi/fit_and_plot_pui.py {arguments.date} first."
    )

with fit_cache_path.open("rb") as cache_file:
    pickup_ion_data, _pui_fit_input_by_chunk_epoch = pickle.load(cache_file)
with spectrogram_cache_path.open("rb") as spectrogram_file:
    (
        energies_per_sweep_ev,
        observed_spectrogram,
        model_spectrogram,
        _sweep_epochs_tt2000,
    ) = pickle.load(spectrogram_file)

chunk_central_datetimes = np.array(
    [cdf_library.tt2000_to_datetime(int(t)) for t in pickup_ion_data.epoch]
)
seconds_from_target = np.array(
    [
        abs((central - target_datetime_utc).total_seconds())
        for central in chunk_central_datetimes
    ]
)
chunk_index = int(np.argmin(seconds_from_target))
print(
    f"Selected chunk {chunk_index} (central epoch "
    f"{chunk_central_datetimes[chunk_index].isoformat()}, "
    f"{seconds_from_target[chunk_index]:.1f} s from requested "
    f"{target_datetime_utc.isoformat()})"
)

if not np.isfinite(pickup_ion_data.ionization_rate[chunk_index].n):
    sys.exit(f"Chunk {chunk_index} has no successful PUI fit.")

fit_cutoff_speed_kms = float(pickup_ion_data.cutoff_speed[chunk_index].n)
fit_ionization_rate_hz = float(pickup_ion_data.ionization_rate[chunk_index].n)
# Claude: the background is no longer fitted; production adds this constant to
# Claude: the modeled rate before comparing against the observed rate.
background_offset_hz = SWAPI_BACKGROUND_RATE

chunk_sweep_slice = slice(chunk_index * 50, (chunk_index + 1) * 50)
chunk_observed_per_sweep = observed_spectrogram[chunk_sweep_slice]
chunk_model_per_sweep = model_spectrogram[chunk_sweep_slice]
chunk_voltages_per_sweep = energies_per_sweep_ev[chunk_sweep_slice] / SWAPI_L2_K_FACTOR
n_sweeps_in_chunk = chunk_observed_per_sweep.shape[0]

chunk_observed_mean = np.nanmean(chunk_observed_per_sweep, axis=0)
chunk_model_mean = np.nanmean(chunk_model_per_sweep, axis=0)
chunk_energies_mean_ev = (
    np.nanmean(chunk_voltages_per_sweep, axis=0) * SWAPI_L2_K_FACTOR
)

# Poisson uncertainty on the chunk-mean rate at each bin: the mean of N
# independent counting measurements over a livetime each, so total counts
# = N · mean_rate · livetime and σ(mean_rate) = sqrt(mean_rate / (N · livetime)).
valid_sweeps_per_bin = np.sum(np.isfinite(chunk_observed_per_sweep), axis=0)
chunk_observed_uncertainty = np.sqrt(
    chunk_observed_mean / (valid_sweeps_per_bin * swapi_l2.SWAPI_LIVETIME)
)

pickup_ion_window_bin_mask = ~np.all(np.isnan(chunk_model_per_sweep), axis=0)

mean_energies_ev_per_step = np.nanmean(
    chunk_voltages_per_sweep * SWAPI_L2_K_FACTOR, axis=0
)
pickup_ion_bin_indices = np.where(pickup_ion_window_bin_mask)[0]
fit_window_energies_ev = mean_energies_ev_per_step[pickup_ion_bin_indices]
energy_sort_order_within_window = np.argsort(fit_window_energies_ev)
mean_energies_ev_sorted = fit_window_energies_ev[energy_sort_order_within_window]
sorted_pickup_ion_bin_indices = pickup_ion_bin_indices[energy_sort_order_within_window]
observed_spectrogram_sorted = chunk_observed_per_sweep[:, sorted_pickup_ion_bin_indices]
model_spectrogram_sorted = chunk_model_per_sweep[:, sorted_pickup_ion_bin_indices]

positive_spectrogram_values = np.concatenate(
    [
        array[np.isfinite(array) & (array > 0)]
        for array in (observed_spectrogram_sorted, model_spectrogram_sorted)
    ]
)
if positive_spectrogram_values.size:
    spectrogram_vmin = max(float(np.percentile(positive_spectrogram_values, 1)), 1e-3)
    spectrogram_vmax = float(positive_spectrogram_values.max())
else:
    spectrogram_vmin, spectrogram_vmax = 1e-3, 1.0
spectrogram_norm = LogNorm(vmin=spectrogram_vmin, vmax=spectrogram_vmax)

figure = plt.figure(figsize=(10, 9), constrained_layout=True)
grid_spec = figure.add_gridspec(3, 1, height_ratios=[1.3, 0.8, 0.8])
line_axis = figure.add_subplot(grid_spec[0])
observed_spectrogram_axis = figure.add_subplot(grid_spec[1])
model_spectrogram_axis = figure.add_subplot(
    grid_spec[2], sharex=observed_spectrogram_axis, sharey=observed_spectrogram_axis
)

line_axis.errorbar(
    chunk_energies_mean_ev,
    chunk_observed_mean,
    yerr=chunk_observed_uncertainty,
    fmt=".",
    color="tab:blue",
    elinewidth=0.8,
    capsize=2,
    label="Observed (chunk mean)",
)
line_axis.plot(
    chunk_energies_mean_ev,
    chunk_model_mean,
    "x-",
    color="tab:orange",
    label="Model + background",
)
line_axis.set_xscale("log")
line_axis.set_yscale("log")
line_axis.set_xlabel("Energy / eV")
line_axis.set_ylabel("Coincidence rate [Hz]")
line_axis.set_title(
    f"SWAPI He+ PUI spectrum — {chunk_central_datetimes[chunk_index].isoformat()} UT"
    f" (chunk {chunk_index})\n"
    f"α={SWAPI_PUI_COOLING_INDEX:.2f} (fixed)  "
    f"v_b={fit_cutoff_speed_kms:.0f} km/s  "
    f"β_E={fit_ionization_rate_hz:.2e} 1/s  "
    f"bg={background_offset_hz:.3f} Hz (fixed)"
)
line_axis.grid(True, which="both", alpha=0.3)
line_axis.legend()

sweep_indices_in_chunk = np.arange(n_sweeps_in_chunk)
for axis_for_spectrogram, spectrogram_values, label in (
    (
        observed_spectrogram_axis,
        observed_spectrogram_sorted,
        "Observed (50 sweeps × PUI fit window)",
    ),
    (model_spectrogram_axis, model_spectrogram_sorted, "Model"),
):
    mesh = axis_for_spectrogram.pcolormesh(
        sweep_indices_in_chunk,
        mean_energies_ev_sorted,
        np.ma.masked_invalid(spectrogram_values).T,
        shading="nearest",
        cmap="viridis",
        norm=spectrogram_norm,
        rasterized=True,
    )
    axis_for_spectrogram.set_yscale("log")
    axis_for_spectrogram.set_ylabel("Energy / eV")
    axis_for_spectrogram.text(
        0.01,
        0.95,
        label,
        transform=axis_for_spectrogram.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"),
    )
plt.setp(observed_spectrogram_axis.get_xticklabels(), visible=False)
model_spectrogram_axis.set_xlabel("Sweep index within chunk")
figure.colorbar(
    mesh,
    ax=[observed_spectrogram_axis, model_spectrogram_axis],
    label="Coincidence rate [Hz]",
)

if arguments.output_path is not None:
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_path, bbox_inches="tight")
    print(f"Saved {arguments.output_path}")
else:
    plt.show()
