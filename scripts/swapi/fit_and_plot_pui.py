"""Download one day of SWAPI L2 + dependencies, run the production helium PUI
fit via PuiChunkFitter, cache the result, and plot the observed coincidence
rate spectrogram alongside the fitted PUI model coincidence rate spectrogram,
followed by the derived PUI parameters (ionization rate, cutoff speed, density,
temperature) as timeseries.

Usage:
    scripts/swapi/fit_and_plot_pui.py <YYYY-MM-DD> [--use-cache] [--output-dir DIR]

Requires the environment variable IMAP_API_KEY to be set.
Downloads land in /tmp/swapi_fit_and_plot_data; the fit pickle lands in
/tmp/swapi_pui_fit_<YYYYMMDD>.pkl and the spectrogram pickle in
/tmp/swapi_pui_gof_window_spectrograms_<YYYYMMDD>.pkl. --use-cache reuses both pickles
and skips the SWAPI network fetch. --output-dir saves the figure as
pui_fit_<YYYYMMDD>.png in the given directory instead of opening an
interactive window.
"""

import argparse
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
import numpy as np
import spiceypy
from imap_data_access import ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input
from spacepy.pycdf import CDF
from spacepy.pycdf import lib as cdf_library
from uncertainties import unumpy

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    calculate_coincidence_rate,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_fit_energy_range,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    build_chunk_collapsed_response,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import (
    MAX_CUTOFF_SPEED_RATIO,
)
from imap_l3_processing.swapi.l3a.chunk_fits import (
    ParallelChunkRunner,
    ProtonChunkFitter,
    PuiChunkFitter,
)
from imap_l3_processing.swapi.l3a.models import SwapiL3PickupIonData
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import chunk_l2_data
from imap_l3_processing.utils import SpiceKernelTypes, get_spice_kernels_file_names
from imap_l3_processing.swapi.species import Species


def replay_chunk_spectrum(dependencies, fit_input, ionization_rate, cutoff_speed):
    """Re-evaluate the forward model for one 50-sweep chunk on every coarse step
    above the lower fitting boundary.

    That is the window the production goodness-of-fit check scores (the fit
    itself also stops at the nominal 16 E_p upper boundary). Mirrors the grid
    bounds in calculate_pickup_ion_values and adds the same constant
    background, so the modeled rate matches what the fitter compared against.
    """
    solar_wind_speed_inertial_frame = float(
        np.linalg.norm(fit_input.solar_wind_velocity_rtn_sun)
    )

    lower_energy_cutoff, _upper_energy_cutoff = calculate_pickup_ion_fit_energy_range(
        solar_wind_speed_inertial_frame
    )
    bin_mask = fit_input.esa_energies > lower_energy_cutoff
    shared_energies = fit_input.esa_energies[bin_mask]

    chunk_response = build_chunk_collapsed_response(
        swapi_response=dependencies.swapi_response,
        voltages_v=shared_energies / SWAPI_L2_K_FACTOR,
        bulk_sw_per_bin_kms=fit_input.bulk_sw_per_bin_swapi_kms[:, bin_mask, :],
        time_as_tt2000=fit_input.time_as_tt2000,
        species=Species.HELIUM_PLUS,
        cutoff_speed_max_kms=(
            solar_wind_speed_inertial_frame * MAX_CUTOFF_SPEED_RATIO * 1.1
        ),
    )
    modeled_per_sweep = (
        calculate_coincidence_rate(
            chunk_response,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            distance=fit_input.distance,
            inflow_angle=fit_input.inflow_angle,
            solar_wind_speed_inertial_frame=solar_wind_speed_inertial_frame,
            density_of_neutral_helium_lookup_table=(
                dependencies.density_of_neutral_helium_calibration_table
            ),
        )
        + SWAPI_BACKGROUND_RATE
    )
    return {
        "energies_ev": shared_energies,
        "observed_rate_per_sweep": fit_input.coincidence_count_rates[:, bin_mask],
        "modeled_rate_per_sweep": modeled_per_sweep,
        "bin_mask": bin_mask,
    }


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
    argument_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip download and refit; load the cached fit from /tmp.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, save the figure as pui_fit_<YYYYMMDD>.png in this directory and skip plt.show().",
    )
    arguments = argument_parser.parse_args()

    if "IMAP_API_KEY" not in os.environ:
        sys.exit("IMAP_API_KEY environment variable is required.")

    target_date = datetime.strptime(arguments.date, "%Y-%m-%d")
    compact_date = target_date.strftime("%Y%m%d")
    cache_path = Path(f"/tmp/swapi_pui_fit_{compact_date}.pkl")

    imap_data_access.config["DATA_DIR"] = Path("/tmp/swapi_fit_and_plot_data")
    imap_data_access.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

    ancillary_template_path = (
        Path(__file__).parent / "imap_swapi_l3a_proton-sw_dependency_template.json"
    )

    science_query_results = imap_data_access.query(
        instrument="swapi",
        data_level="l2",
        descriptor="sci",
        start_date=compact_date,
        end_date=compact_date,
        version="latest",
    )
    if not science_query_results:
        sys.exit(f"No SWAPI L2 sci file at the SDC for {arguments.date}")

    published_l3a_query_results = imap_data_access.query(
        instrument="swapi",
        data_level="l3a",
        descriptor="pui-he",
        start_date=compact_date,
        end_date=compact_date,
        version="latest",
    )
    if published_l3a_query_results:
        published_l3a_cdf_path = imap_data_access.download(
            published_l3a_query_results[0]["file_path"]
        )
        print(f"Downloaded published L3A pui-he from SDC: {published_l3a_cdf_path}")
    else:
        published_l3a_cdf_path = None
        print(f"No published SWAPI L3A pui-he file at the SDC for {arguments.date}")

    ephemeris_kernel_types = [
        SpiceKernelTypes.EphemerisReconstructed,
        SpiceKernelTypes.EphemerisPredicted,
    ]
    non_ephemeris_kernel_types = [
        t for t in SpiceKernelTypes if t not in ephemeris_kernel_types
    ]
    spice_kernel_filenames = list(
        dict.fromkeys(
            os.path.basename(path)
            for path in (
                get_spice_kernels_file_names(
                    target_date - timedelta(days=1),
                    target_date + timedelta(days=2),
                    non_ephemeris_kernel_types,
                )
                + get_spice_kernels_file_names(
                    target_date - timedelta(days=90),
                    target_date + timedelta(days=2),
                    ephemeris_kernel_types,
                )
            )
        )
    )

    dynamic_filenames = [
        os.path.basename(science_query_results[0]["file_path"]),
        *spice_kernel_filenames,
    ]
    processing_input_collection = ProcessingInputCollection(
        *[generate_imap_input(filename) for filename in dynamic_filenames]
    )
    processing_input_collection.deserialize(ancillary_template_path.read_text())
    processing_input_collection.download_all_files()

    for kernel_path in processing_input_collection.get_file_paths(data_type="spice"):
        spiceypy.furnsh(str(imap_data_access.download(kernel_path)))

    dependencies = SwapiL3ADependencies.fetch_dependencies(processing_input_collection)

    input_metadata = InputMetadata(
        instrument="swapi",
        data_level="l3a",
        start_date=target_date,
        end_date=target_date,
        version="v001",
        descriptor="pui-he",
    )

    if not arguments.use_cache or not cache_path.exists():
        dependencies.swapi_response.warm_cache(
            dependencies.data.energy / SWAPI_L2_K_FACTOR
        )
        runner = ParallelChunkRunner(
            dependencies.swapi_response, dependencies.efficiency_calibration_table
        )
        pui_chunks = list(chunk_l2_data(dependencies.data, 50))
        proton_results = runner.run(
            list(chunk_l2_data(dependencies.data, 5)), ProtonChunkFitter()
        )
        pui_fitter = PuiChunkFitter(
            density_of_neutral_helium_lookup_table=dependencies.density_of_neutral_helium_calibration_table,
            hydrogen_inflow_vector=dependencies.hydrogen_inflow_vector,
            helium_inflow_vector=dependencies.helium_inflow_vector,
            proton_sw_results=proton_results,
        )
        pui_result = runner.run(pui_chunks, pui_fitter)
        # Claude: the replay below needs the same per-chunk geometry the fit saw.
        # Claude: runner.run builds it internally, so this repeats the (cheap)
        # Claude: SPICE work rather than reaching into the pool.
        pui_fit_input_by_chunk_epoch = {
            int(fit_input.time_as_tt2000): fit_input
            for fit_input, _quality_flag in pui_fitter.precompute_geometry(pui_chunks)
            if fit_input is not None
        }
        pickup_ion_data = SwapiL3PickupIonData(
            replace(input_metadata, descriptor="pui-he"),
            **pui_result,
        )
        with cache_path.open("wb") as cache_file:
            pickle.dump((pickup_ion_data, pui_fit_input_by_chunk_epoch), cache_file)
        print(f"Cached fit to {cache_path}")

    with cache_path.open("rb") as cache_file:
        pickup_ion_data, pui_fit_input_by_chunk_epoch = pickle.load(cache_file)

    epochs_tt2000 = np.asarray(pickup_ion_data.epoch, dtype=np.int64)
    timestamps = np.array(
        [cdf_library.tt2000_to_datetime(int(t)) for t in epochs_tt2000]
    )

    spectrogram_cache_path = Path(
        f"/tmp/swapi_pui_gof_window_spectrograms_{compact_date}.pkl"
    )
    all_chunks = list(chunk_l2_data(dependencies.data, 50))

    if not arguments.use_cache or not spectrogram_cache_path.exists():
        print(
            "Building per-sweep observed and modeled spectrograms (one replay per chunk)..."
        )
        dependencies.swapi_response.warm_cache(
            dependencies.data.energy / SWAPI_L2_K_FACTOR
        )

        chunk_count = min(len(all_chunks), len(epochs_tt2000))
        per_sweep_voltages = []
        per_sweep_observed = []
        per_sweep_model = []
        per_sweep_epoch_tt2000 = []

        for chunk_index in range(chunk_count):
            data_chunk = all_chunks[chunk_index]
            voltages_2d = (
                data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
            )
            rates_2d = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
            per_sweep_voltages.append(voltages_2d)
            per_sweep_observed.append(rates_2d)
            per_sweep_epoch_tt2000.append(
                np.asarray(data_chunk.sci_start_time, dtype=np.int64)
            )

            model_rates_for_chunk = np.full_like(rates_2d, np.nan, dtype=float)

            chunk_epoch = int(epochs_tt2000[chunk_index])
            fit_input = pui_fit_input_by_chunk_epoch.get(chunk_epoch)
            if (
                fit_input is not None
                and np.isfinite(pickup_ion_data.ionization_rate[chunk_index].n)
                and np.isfinite(pickup_ion_data.cutoff_speed[chunk_index].n)
            ):
                try:
                    replay = replay_chunk_spectrum(
                        dependencies,
                        fit_input,
                        pickup_ion_data.ionization_rate[chunk_index].n,
                        pickup_ion_data.cutoff_speed[chunk_index].n,
                    )
                except Exception as replay_error:
                    print(f"chunk {chunk_index} replay failed: {replay_error}")
                else:
                    model_rates_for_chunk[:, replay["bin_mask"]] = replay[
                        "modeled_rate_per_sweep"
                    ]
            per_sweep_model.append(model_rates_for_chunk)

        energies_per_sweep_ev = (
            np.concatenate(per_sweep_voltages, axis=0) * SWAPI_L2_K_FACTOR
        )
        observed_spectrogram = np.concatenate(per_sweep_observed, axis=0)
        model_spectrogram = np.concatenate(per_sweep_model, axis=0)
        sweep_epochs_tt2000 = np.concatenate(per_sweep_epoch_tt2000, axis=0)

        with spectrogram_cache_path.open("wb") as spectrogram_file:
            pickle.dump(
                (
                    energies_per_sweep_ev,
                    observed_spectrogram,
                    model_spectrogram,
                    sweep_epochs_tt2000,
                ),
                spectrogram_file,
            )
        print(f"Cached spectrograms to {spectrogram_cache_path}")

    with spectrogram_cache_path.open("rb") as spectrogram_file:
        (
            energies_per_sweep_ev,
            observed_spectrogram,
            model_spectrogram,
            sweep_epochs_tt2000,
        ) = pickle.load(spectrogram_file)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    mean_energies_ev = np.nanmean(energies_per_sweep_ev, axis=0)
    energy_sort_order = np.argsort(mean_energies_ev)
    mean_energies_ev_sorted = mean_energies_ev[energy_sort_order]
    observed_sorted = observed_spectrogram[:, energy_sort_order]
    model_sorted = model_spectrogram[:, energy_sort_order]

    positive_rate_values = np.concatenate(
        [
            array[np.isfinite(array) & (array > 0)]
            for array in (observed_sorted, model_sorted)
        ]
    )
    if positive_rate_values.size:
        rate_vmin = max(float(np.percentile(positive_rate_values, 1)), 1e-3)
        rate_vmax = float(positive_rate_values.max())
    else:
        rate_vmin, rate_vmax = 1e-3, 1.0

    sweep_timestamps = np.array(
        [cdf_library.tt2000_to_datetime(int(t)) for t in sweep_epochs_tt2000]
    )

    CUTOFF_SPEED_PANEL_INDEX = 1
    timeseries_panels = [
        (
            "Ionization rate [s$^{-1}$]",
            "log",
            pickup_ion_data.ionization_rate,
            "pui_ionization_rate",
        ),
        (
            "Cutoff speed [km/s]",
            "linear",
            pickup_ion_data.cutoff_speed,
            "pui_cutoff_speed",
        ),
        ("Density [cm$^{-3}$]", "log", pickup_ion_data.density, "pui_density"),
        ("Temperature [K]", "log", pickup_ion_data.temperature, "pui_temperature"),
    ]

    figure = plt.figure(figsize=(11, 13), layout="constrained")
    grid_spec = figure.add_gridspec(
        2 + len(timeseries_panels),
        1,
        height_ratios=[1.6, 1.6] + [1] * len(timeseries_panels),
    )

    observed_axis = figure.add_subplot(grid_spec[0])
    observed_mesh = observed_axis.pcolormesh(
        sweep_timestamps,
        mean_energies_ev_sorted,
        np.ma.masked_invalid(observed_sorted).T,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=rate_vmin, vmax=rate_vmax),
    )
    observed_axis.set_yscale("log")
    observed_axis.set_ylabel("Energy / eV")
    observed_axis.set_title("Observed coincidence count rate (per 12 s sweep)")
    figure.colorbar(observed_mesh, ax=observed_axis, label="rate / s$^{-1}$")
    plt.setp(observed_axis.get_xticklabels(), visible=False)

    model_axis = figure.add_subplot(
        grid_spec[1], sharex=observed_axis, sharey=observed_axis
    )
    model_mesh = model_axis.pcolormesh(
        sweep_timestamps,
        mean_energies_ev_sorted,
        np.ma.masked_invalid(model_sorted).T,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=rate_vmin, vmax=rate_vmax),
    )
    model_axis.set_yscale("log")
    model_axis.set_ylabel("Energy / eV")
    model_axis.set_title(
        "Fitted helium PUI model coincidence count rate (per 12 s sweep)"
    )
    figure.colorbar(model_mesh, ax=model_axis, label="rate / s$^{-1}$")
    plt.setp(model_axis.get_xticklabels(), visible=False)

    # Claude: the cutoff-speed panel overlays the Sun-frame bulk speed the fit
    # Claude: bounded the cutoff against, so read it back from the fit inputs.
    sw_speed_kms = np.array(
        [
            np.linalg.norm(
                pui_fit_input_by_chunk_epoch[int(epoch)].solar_wind_velocity_rtn_sun
            )
            if int(epoch) in pui_fit_input_by_chunk_epoch
            else np.nan
            for epoch in epochs_tt2000
        ]
    )

    published_l3a_timestamps = None
    published_l3a_values_by_var = {}
    if published_l3a_cdf_path is not None:
        with CDF(str(published_l3a_cdf_path)) as published_cdf:
            published_epoch_tt2000 = np.asarray(
                published_cdf.raw_var("epoch")[...], dtype=np.int64
            )
            published_l3a_timestamps = np.array(
                [cdf_library.tt2000_to_datetime(int(t)) for t in published_epoch_tt2000]
            )
            for _, _, _, cdf_var_name in timeseries_panels:
                published_l3a_values_by_var[cdf_var_name] = (
                    np.asarray(published_cdf[cdf_var_name][...], dtype=float),
                    np.asarray(
                        published_cdf[f"{cdf_var_name}_uncert"][...], dtype=float
                    ),
                )

    local_flags = np.asarray(pickup_ion_data.quality_flags, dtype=np.int64)
    local_flagged = local_flags != 0

    for panel_index, (ylabel, yscale, values, cdf_var_name) in enumerate(
        timeseries_panels
    ):
        axis = figure.add_subplot(grid_spec[panel_index + 2], sharex=observed_axis)
        nominal_values_array = unumpy.nominal_values(values)
        std_devs_array = unumpy.std_devs(values)
        show_legend = (
            panel_index == CUTOFF_SPEED_PANEL_INDEX
            or published_l3a_timestamps is not None
        )
        axis.errorbar(
            timestamps,
            nominal_values_array,
            yerr=std_devs_array,
            fmt=".",
            capsize=2,
            color="tab:blue",
            label="This run" if show_legend else None,
        )
        if np.any(local_flagged):
            axis.scatter(
                timestamps[local_flagged],
                nominal_values_array[local_flagged],
                s=60,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                zorder=3,
                label="flag != 0" if show_legend else None,
            )
        if published_l3a_timestamps is not None:
            published_nominal, published_uncert = published_l3a_values_by_var[
                cdf_var_name
            ]
            valid = (
                np.isfinite(published_nominal)
                & (published_uncert >= 0)
                & np.isfinite(published_uncert)
            )
            axis.errorbar(
                published_l3a_timestamps[valid],
                published_nominal[valid],
                yerr=published_uncert[valid],
                fmt="x",
                capsize=2,
                color="red",
                label="SDC L3A",
            )
        if panel_index == CUTOFF_SPEED_PANEL_INDEX:
            axis.plot(
                timestamps,
                sw_speed_kms,
                color="tab:orange",
                linewidth=1.0,
                label="Proton SW speed",
            )
            axis.fill_between(
                timestamps,
                0.8 * sw_speed_kms,
                1.2 * sw_speed_kms,
                color="tab:orange",
                alpha=0.15,
                linewidth=0,
                label=r"$\pm 20\%$ SW speed",
            )
        if (
            panel_index == CUTOFF_SPEED_PANEL_INDEX
            or published_l3a_timestamps is not None
        ):
            axis.legend(loc="best", fontsize="small")
        axis.set_ylabel(ylabel)
        axis.set_yscale(yscale)
        if panel_index < len(timeseries_panels) - 1:
            plt.setp(axis.get_xticklabels(), visible=False)
        else:
            axis.set_xlabel("Time (UTC)")

    figure.suptitle(f"SWAPI helium PUI fit — {arguments.date}")
    if arguments.output_dir is not None:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = arguments.output_dir / f"pui_fit_{compact_date}.png"
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
