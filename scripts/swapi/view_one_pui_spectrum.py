"""View one chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production-model (replayed) rate.

Loads the cached pickle output of fit_and_plot_pui.py (observed and
production-model spectrograms, fit parameters) and plots the selected chunk's
sweep-averaged spectrum above per-sweep spectrograms of the observed and
modeled rates over the goodness-of-fit window (every coarse step above the
lower fitting boundary). The modeled rate already includes the constant
instrument background, so it is directly comparable to the observed
coincidence rate.

The spectrum panel shades the energy range the fit minimises over and marks
which observed steps feed each goodness-of-fit metric: the mean relative error
up to E_1/4, and the past-peak ratio above it.

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

from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_fit_energy_range,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import (
    MAX_CUTOFF_SPEED_KMS,
    MAX_CUTOFF_SPEED_RATIO,
    MAX_IONIZATION_RATE,
    MAX_MEAN_RELATIVE_ERROR,
    MAX_PAST_PEAK_RATIO,
    MIN_CUTOFF_SPEED_RATIO,
    MIN_IONIZATION_RATE,
    goodness_of_fit_metrics,
    goodness_of_fit_upper_energy,
)
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
    f"/tmp/swapi_pui_gof_window_spectrograms_{compact_date}.pkl"
)
if not fit_cache_path.exists() or not spectrogram_cache_path.exists():
    sys.exit(
        f"Missing pickle caches for {arguments.date}. "
        f"Run scripts/swapi/fit_and_plot_pui.py {arguments.date} first."
    )

with fit_cache_path.open("rb") as cache_file:
    pickup_ion_data, pui_fit_input_by_chunk_epoch = pickle.load(cache_file)
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

# Claude: rebuild the production windows from the same per-chunk fit input the
# Claude: fitter saw, so the step selection matches calculate_pickup_ion_values.
fit_input = pui_fit_input_by_chunk_epoch[int(pickup_ion_data.epoch[chunk_index])]
solar_wind_speed_kms = float(np.linalg.norm(fit_input.solar_wind_velocity_rtn_sun))
fit_lower_energy_ev, fit_upper_energy_ev = calculate_pickup_ion_fit_energy_range(
    solar_wind_speed_kms
)
goodness_window_step_mask = fit_input.esa_energies > fit_lower_energy_ev
# Claude: is_good_fit takes the model rate without background.
goodness_upper_energy_ev = goodness_of_fit_upper_energy(
    fit_input.esa_energies[goodness_window_step_mask],
    chunk_model_mean[goodness_window_step_mask] - SWAPI_BACKGROUND_RATE,
)
relative_error_step_mask = goodness_window_step_mask & (
    fit_input.esa_energies <= goodness_upper_energy_ev
)
past_peak_step_mask = goodness_window_step_mask & (
    fit_input.esa_energies > goodness_upper_energy_ev
)

mean_relative_error, past_peak_ratio = goodness_of_fit_metrics(
    fit_input.esa_energies[goodness_window_step_mask],
    chunk_model_per_sweep[:, goodness_window_step_mask] - SWAPI_BACKGROUND_RATE,
    fit_input.coincidence_count_rates[:, goodness_window_step_mask],
)
cutoff_speed_ratio = fit_cutoff_speed_kms / solar_wind_speed_kms

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

figure = plt.figure(figsize=(11, 7.5), constrained_layout=True)
grid_spec = figure.add_gridspec(2, 1, height_ratios=[1.3, 1])
spectrum_row_grid_spec = grid_spec[0].subgridspec(1, 2, width_ratios=[1.6, 1.2])
line_axis = figure.add_subplot(spectrum_row_grid_spec[0])
table_axis = figure.add_subplot(spectrum_row_grid_spec[1])
spectrogram_row_grid_spec = grid_spec[1].subgridspec(1, 2)
observed_spectrogram_axis = figure.add_subplot(spectrogram_row_grid_spec[0])
model_spectrogram_axis = figure.add_subplot(
    spectrogram_row_grid_spec[1],
    sharex=observed_spectrogram_axis,
    sharey=observed_spectrogram_axis,
)

line_axis.axvspan(
    fit_lower_energy_ev,
    fit_upper_energy_ev,
    color="0.85",
    zorder=0,
    label="Fitting range",
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
line_axis.plot(
    chunk_energies_mean_ev[relative_error_step_mask],
    chunk_observed_mean[relative_error_step_mask],
    "o",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor="tab:green",
    label=r"Mean relative error steps ($E \leq E_{1/4}$)",
)
line_axis.plot(
    chunk_energies_mean_ev[past_peak_step_mask],
    chunk_observed_mean[past_peak_step_mask],
    "s",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor="tab:red",
    label=r"Past-peak ratio steps ($E > E_{1/4}$)",
)
line_axis.set_xscale("log")
# Claude: pin the limits to the measured steps; the shaded fitting range would
# Claude: otherwise stretch the axis past the last step when 16 E_p exceeds it.
line_axis.set_xlim(
    np.nanmin(chunk_energies_mean_ev) / 1.15, np.nanmax(chunk_energies_mean_ev) * 1.15
)
line_axis.set_yscale("log")
line_axis.set_xlabel("Energy / eV")
line_axis.set_ylabel("Coincidence rate [Hz]")
line_axis.grid(True, which="both", alpha=0.3)
line_axis.legend(fontsize=8)


def _scientific_mathtext(value: float, digits: int = 2) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def _uncertain_mathtext(quantity, scientific: bool) -> str:
    """`quantity` (a ufloat) as mathtext, rounded to two significant figures of
    its uncertainty."""
    value, sigma = float(quantity.n), float(quantity.s)
    exponent = int(np.floor(np.log10(abs(value)))) if scientific else 0
    value, sigma = value / 10**exponent, sigma / 10**exponent
    decimals = max(0, 1 - int(np.floor(np.log10(sigma))))
    body = rf"{value:.{decimals}f} \pm {sigma:.{decimals}f}"
    if not scientific:
        return f"${body}$"
    return rf"$({body}) \times 10^{{{exponent}}}$"


parameter_rows = [
    ("Chunk centre (UT)", f"{chunk_central_datetimes[chunk_index]:%Y-%m-%d %H:%M:%S}"),
    (
        r"$v_b$",
        _uncertain_mathtext(pickup_ion_data.cutoff_speed[chunk_index], False) + " km/s",
    ),
    (
        r"$\beta_E$",
        _uncertain_mathtext(pickup_ion_data.ionization_rate[chunk_index], True)
        + r" s$^{-1}$",
    ),
    (
        r"$n_\mathrm{PUI}$ (derived)",
        _uncertain_mathtext(pickup_ion_data.density[chunk_index], True) + r" cm$^{-3}$",
    ),
    (
        r"$T_\mathrm{PUI}$ (derived)",
        _uncertain_mathtext(pickup_ion_data.temperature[chunk_index], True) + " K",
    ),
    (r"$\alpha_\mathrm{PUI}$ (fixed)", f"{SWAPI_PUI_COOLING_INDEX:.2f}"),
    (r"$C_\mathrm{bg}$ (fixed)", f"{background_offset_hz:.3f} Hz"),
    (r"$v_\mathrm{sw}$ (input)", f"{solar_wind_speed_kms:.0f} km/s"),
]
# Claude: each criterion is written with this chunk's value in place of the
# Claude: symbol, and paired with whether it passes, as is_good_fit checks it.
criterion_rows = [
    (
        rf"${fit_cutoff_speed_kms:.0f} \leq {MAX_CUTOFF_SPEED_KMS:.0f}$ km/s",
        r"$v_b$",
        fit_cutoff_speed_kms <= MAX_CUTOFF_SPEED_KMS,
    ),
    (
        rf"${MIN_CUTOFF_SPEED_RATIO} \leq {cutoff_speed_ratio:.2f}"
        rf" \leq {MAX_CUTOFF_SPEED_RATIO}$",
        r"$v_b / v_\mathrm{sw}$",
        MIN_CUTOFF_SPEED_RATIO <= cutoff_speed_ratio <= MAX_CUTOFF_SPEED_RATIO,
    ),
    (
        rf"${_scientific_mathtext(MIN_IONIZATION_RATE, 1)} \leq"
        rf" {_scientific_mathtext(fit_ionization_rate_hz)} \leq"
        rf" {_scientific_mathtext(MAX_IONIZATION_RATE, 1)}$",
        r"$\beta_E$ [s$^{-1}$]",
        MIN_IONIZATION_RATE <= fit_ionization_rate_hz <= MAX_IONIZATION_RATE,
    ),
    (
        rf"${mean_relative_error:.3f} \leq {MAX_MEAN_RELATIVE_ERROR}$",
        r"$\Delta_\mathrm{rel}$",
        mean_relative_error <= MAX_MEAN_RELATIVE_ERROR,
    ),
    (
        rf"${past_peak_ratio:.3f} \leq {MAX_PAST_PEAK_RATIO}$",
        r"$R_\mathrm{past\ peak}$",
        past_peak_ratio <= MAX_PAST_PEAK_RATIO,
    ),
]

table_axis.axis("off")
parameter_table = table_axis.table(
    cellText=[[name, value] for name, value in parameter_rows],
    colLabels=["Parameter", "Value"],
    colWidths=[0.38, 0.62],
    cellLoc="left",
    bbox=[0.0, 0.42, 1.0, 0.58],
)
criterion_table = table_axis.table(
    cellText=[[name, inequality] for inequality, name, _ in criterion_rows],
    colLabels=["Goodness of fit", "Criterion"],
    colWidths=[0.26, 0.74],
    cellLoc="left",
    bbox=[0.0, 0.0, 1.0, 0.36],
)
for table in (parameter_table, criterion_table):
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.9")
for row, (_inequality, _name, passes) in enumerate(criterion_rows, start=1):
    criterion_table[row, 1].get_text().set_color("tab:green" if passes else "tab:red")

sweep_indices_in_chunk = np.arange(n_sweeps_in_chunk)
for axis_for_spectrogram, spectrogram_values, label in (
    (
        observed_spectrogram_axis,
        observed_spectrogram_sorted,
        "Observed",
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
    axis_for_spectrogram.set_xlabel("Sweep index within chunk")
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
observed_spectrogram_axis.set_ylabel("Energy / eV")
model_spectrogram_axis.tick_params(axis="y", which="both", labelleft=False)
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
