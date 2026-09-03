#!/usr/bin/env python3
"""
Visualise the MC validation that `MonteCarloFitPickupIonTest` runs against
`tests/test_data/swapi/pui_count_rate_reference_50sweep.h5`.

The figure has six rows.

Rows 1–3 are spectrograms over the (sweep, ESA voltage) grid, each on its own
row and sharing one log color bar:
    1. Truth model rate (He+ PUI + proton/alpha Maxwellian shoulder +
       background, with the forward deadtime factor applied) — the
       `expected_coincidence_rate_hz` baked into the h5 fixture.
    2. One Poisson realization of that truth (seed 0).
    3. The fitted forward-model rate from that realization: the *truth*
       proton + alpha ideal rate (the PUI fit does not refit proton/alpha)
       plus the fitted PUI coincidence rate plus the fitted background, all
       run through the deadtime factor — directly comparable to row 1 and
       isolating how well the PUI parameters were recovered.

Rows 4–6 are four-panel histograms over the MC realizations, one column per
PUI fit parameter:
    Row 4 — distribution of fitted nominal values with truth and MC mean overlaid.
    Row 5 — distribution of per-fit reported standard errors σ̂ with the
        empirical scatter of the nominal values overlaid for comparison.
    Row 6 — distribution of `(fit − truth) / σ̂`; a correctly-calibrated σ̂
        estimator places these on the unit Gaussian shown for reference.

Output: docs/swapi/figures/pui_mc_validation.svg
Usage:  uv run python docs/swapi/figure_src/plot_pui_mc_validation.py
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from matplotlib.colors import LogNorm

from figure_utils import FIGURES_DIR

from uncertainties import ufloat

from imap_l3_processing.constants import (
    FIVE_MINUTES_IN_NANOSECONDS,
    HE_PUI_PARTICLE_MASS_KG,
    ONE_AU_IN_KM,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_LIVETIME_S,
)
from imap_l3_processing.swapi.l3a.chunk_fits import ParallelChunkRunner, PuiChunkFitter, _shared
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    calculate_coincidence_rate,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    build_chunk_collapsed_response,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import (
    calculate_pui_energy_cutoff,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    build_vasyliunas_siscoe_distribution,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.utils import SpiceKernelTypes, furnish_spice_metakernel
from tests.swapi._helpers import load_swapi_response

_MC_N_SAMPLES = 1000
_EXAMPLE_SEED = 0

_HELIUM_MASS_PER_CHARGE_M_P_PER_E = 4.0

_REFERENCE_50SWEEP_H5_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "pui_count_rate_reference_50sweep.h5"
)
_DENSITY_LUT_PATH = (
    REPO_ROOT / "instrument_team_data" / "swapi" / "density-of-neutral-helium-lut.dat"
)
_HYDROGEN_INFLOW_PATH = (
    REPO_ROOT
    / "tests"
    / "test_data"
    / "swapi"
    / "imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
)
_HELIUM_INFLOW_PATH = (
    REPO_ROOT
    / "tests"
    / "test_data"
    / "swapi"
    / "imap_swapi_helium-inflow-vector_20100101_v001.dat"
)

_SPICE_KERNEL_TYPES = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.IMAPFrames,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.EphemerisReconstructed,
    SpiceKernelTypes.PlanetaryEphemeris,
    SpiceKernelTypes.PlanetaryConstants,
]

# Module-level metakernel path so a fork-spawned worker can re-furnish SPICE
# without re-running `furnish_spice_metakernel` (which would hit the network).
_metakernel_path: str = ""


def _ensure_spice_furnished_in_worker():
    if spiceypy.ktotal("ALL") == 0:
        spiceypy.furnsh(_metakernel_path)


class _FitOnlyPuiChunkFitter(PuiChunkFitter):
    """`PuiChunkFitter` variant that bypasses the proton fit and skips the
    density/temperature integrals. `precompute_geometry` injects the known
    truth bulk velocity and per-bin SWAPI-frame velocities directly, and
    `fit_chunk` returns only the four fit parameters needed by the MC
    validation plot."""

    def __init__(
        self,
        density_of_neutral_helium_lookup_table,
        hydrogen_inflow_vector,
        helium_inflow_vector,
        sw_velocity_rtn,
        bulk_sw_per_bin_swapi,
    ):
        self.density_of_neutral_helium_lookup_table = (
            density_of_neutral_helium_lookup_table
        )
        self.hydrogen_inflow_vector = hydrogen_inflow_vector
        self.helium_inflow_vector = helium_inflow_vector
        self._sw_velocity_rtn = np.asarray(sw_velocity_rtn, dtype=float)
        self._bulk_sw_per_bin_swapi = bulk_sw_per_bin_swapi

    def precompute_geometry(self, chunks):
        # Truth values are identical across MC realizations, so the SPICE-
        # dependent products (cutoffs, V-S distribution) only need to be
        # computed once for the shared chunk-center epoch and reused.
        first_epoch = (
            int(chunks[0].sci_start_time[0]) + FIVE_MINUTES_IN_NANOSECONDS
        )
        chunk_ephemeris_time = spiceypy.unitim(
            first_epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"
        )
        lower_energy_cutoff = 1.25 * calculate_pui_energy_cutoff(
            PROTON_MASS_KG,
            chunk_ephemeris_time,
            self._sw_velocity_rtn,
            self.hydrogen_inflow_vector,
        )
        upper_energy_cutoff = 1.2 * calculate_pui_energy_cutoff(
            HE_PUI_PARTICLE_MASS_KG,
            chunk_ephemeris_time,
            self._sw_velocity_rtn,
            self.helium_inflow_vector,
        )
        vasyliunas_siscoe_distribution = build_vasyliunas_siscoe_distribution(
            chunk_ephemeris_time,
            self._sw_velocity_rtn,
            self.density_of_neutral_helium_lookup_table,
            self.helium_inflow_vector,
        )

        geometries = []
        for chunk in chunks:
            epoch = (
                int(chunk.sci_start_time[0]) + FIVE_MINUTES_IN_NANOSECONDS
            )
            geometries.append((
                epoch,
                self._sw_velocity_rtn,
                self._bulk_sw_per_bin_swapi,
                0,
                lower_energy_cutoff,
                upper_energy_cutoff,
                vasyliunas_siscoe_distribution,
            ))
        return geometries

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        sw_velocity_rtn,
        bulk_sw_per_bin_swapi,
        proton_sw_quality_flag,
        lower_energy_cutoff,
        upper_energy_cutoff,
        vasyliunas_siscoe_distribution,
    ):
        _ensure_spice_furnished_in_worker()
        nan = ufloat(np.nan, np.nan)
        cooling_index = nan
        ionization_rate = nan
        cutoff_speed = nan
        background_rate = nan

        try:
            count_rates_window = data_chunk.coincidence_count_rate[
                :, SWAPI_COARSE_SWEEP_BINS
            ]
            if (
                vasyliunas_siscoe_distribution is None
                or np.any(np.isnan(count_rates_window))
                or np.any(np.isnan(sw_velocity_rtn))
                or np.any(np.isnan(bulk_sw_per_bin_swapi))
            ):
                raise ValueError("Fill values in input data")
            voltages = (
                data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
            ).flatten()
            count_rates = count_rates_window.flatten()
            central_effective_area_scale = _shared[
                "efficiency_table"
            ].central_effective_area_scale_for(epoch, "helium")
            fit_result = calculate_pickup_ion_values(
                _shared["swapi_response"],
                voltages,
                count_rates,
                sw_velocity_rtn,
                bulk_sw_per_bin_swapi,
                self.density_of_neutral_helium_lookup_table,
                lower_energy_cutoff,
                upper_energy_cutoff,
                vasyliunas_siscoe_distribution,
                central_effective_area_scale=central_effective_area_scale,
            )
            fit_params = fit_result.fitting_params
            cooling_index = fit_params.cooling_index
            ionization_rate = fit_params.ionization_rate
            cutoff_speed = fit_params.cutoff_speed
            background_rate = fit_params.background_count_rate
        except Exception:
            pass

        return dict(
            epoch=epoch,
            cooling_index=cooling_index,
            ionization_rate=ionization_rate,
            cutoff_speed=cutoff_speed,
            background_rate=background_rate,
        )


def main():
    os.environ.setdefault(
        "IMAP_API_KEY",
        subprocess.check_output(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ["USER"],
                "-s",
                "imap-api-key",
                "-w",
            ],
            text=True,
        ).strip(),
    )
    furnish_output = furnish_spice_metakernel(
        start_date=datetime(2026, 4, 25, 0, 0, 0),
        end_date=datetime(2026, 4, 25, 0, 11, 0),
        kernel_types=_SPICE_KERNEL_TYPES,
    )
    # `_FitOnlyPuiChunkFitter` workers furnish SPICE fresh in their own
    # process via `_ensure_spice_furnished_in_worker`, which reads
    # `_metakernel_path` from this module (inherited across fork).
    global _metakernel_path
    _metakernel_path = str(furnish_output.metakernel_path)

    with h5py.File(_REFERENCE_50SWEEP_H5_PATH, "r") as h5:
        # h5 stores voltage in ascending order, but production SWAPI sweeps go
        # high→low. Flip along the bin axis once at ingest so the rest of the
        # script operates in the production canonical order; the spectrogram
        # plot uses a log y-axis so high voltage naturally sits at the top.
        voltage_v = h5["voltage_v"][...].astype(float)[::-1].copy()
        energy_ev = h5["energy_ev"][...].astype(float)[::-1].copy()
        expected_rate_hz_per_bin = h5[
            "expected_coincidence_rate_hz"
        ][...].astype(float)[:, ::-1].copy()
        proton_alpha_rate_hz_per_bin = h5[
            "proton_alpha_coincidence_rate_hz"
        ][...].astype(float)[:, ::-1].copy()
        bulk_sw_per_bin_swapi_kms = h5[
            "bulk_sw_per_bin_swapi_kms"
        ][...].astype(float)[:, ::-1, :].copy()
        sci_start_time = h5["sci_start_time_tt2000_ns"][...].astype(np.int64)
        sw_velocity_rtn = np.array(h5.attrs["bulk_sw_rtn_kms"], dtype=float)
        cooling_index_truth = float(h5.attrs["cooling_index"])
        cutoff_speed_truth_kms = float(h5.attrs["cutoff_speed_kms"])
        ionization_rate_truth_hz = float(h5.attrs["ionization_rate_hz"])
        background_rate_truth_hz = float(h5.attrs["background_rate_hz"])
        helium_efficiency_ratio = float(h5.attrs["helium_efficiency_ratio"])

    n_sweeps = expected_rate_hz_per_bin.shape[0]
    expected_counts_coarse = np.maximum(
        expected_rate_hz_per_bin * SWAPI_LIVETIME_S, 0.0
    )

    energy_full_template = np.zeros((n_sweeps, 72))
    energy_full_template[:, SWAPI_COARSE_SWEEP_BINS] = energy_ev[np.newaxis, :]

    # Build one SwapiL2Data chunk per MC realization. Each chunk gets a
    # unique sci_start_time offset (and therefore a unique chunk-center
    # epoch) so the fitter's per-chunk geometry maps stay one-to-one with
    # the chunks. The same pattern is in the test.
    print(f"Building {_MC_N_SAMPLES} Poisson-resampled chunks...")
    chunks = []
    for seed in range(_MC_N_SAMPLES):
        mc_rng = np.random.default_rng(seed)
        observed_counts = mc_rng.poisson(expected_counts_coarse)
        observed_rate = observed_counts.astype(float) / SWAPI_LIVETIME_S
        rate_full = np.zeros((n_sweeps, 72))
        rate_full[:, SWAPI_COARSE_SWEEP_BINS] = observed_rate

        chunk_start_time = sci_start_time + (seed * 1_000_000)  # +1 ms per seed
        chunk = SwapiL2Data(
            sci_start_time=chunk_start_time,
            energy=energy_full_template.copy(),
            coincidence_count_rate=rate_full,
            coincidence_count_rate_uncertainty=np.zeros_like(rate_full),
        )
        chunks.append(chunk)

    print("Loading swapi_response, density LUT, and inflow vectors...")
    swapi_response = load_swapi_response(warm_cache_voltages=voltage_v)
    density_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(_DENSITY_LUT_PATH)
    hydrogen_inflow_vector = InflowVector.from_file(_HYDROGEN_INFLOW_PATH)
    helium_inflow_vector = InflowVector.from_file(_HELIUM_INFLOW_PATH)

    # Efficiency table with alpha/proton ratio matching the h5 fixture. The
    # ParallelChunkRunner reads this out of `_shared` in workers.
    proton_eff = 0.02348
    alpha_eff = proton_eff * helium_efficiency_ratio
    efficiency_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False
    )
    efficiency_file.write(
        f"2000-01-01T11:00:00  0  {proton_eff:.10f}  {alpha_eff:.10f}\n"
    )
    efficiency_file.close()
    efficiency_table = EfficiencyCalibrationTable(efficiency_file.name)
    os.unlink(efficiency_file.name)

    fitter = _FitOnlyPuiChunkFitter(
        density_of_neutral_helium_lookup_table=density_lookup_table,
        hydrogen_inflow_vector=hydrogen_inflow_vector,
        helium_inflow_vector=helium_inflow_vector,
        sw_velocity_rtn=sw_velocity_rtn,
        bulk_sw_per_bin_swapi=bulk_sw_per_bin_swapi_kms,
    )
    runner = ParallelChunkRunner(
        swapi_response=swapi_response, efficiency_table=efficiency_table
    )

    print(f"Running MC fits ({_MC_N_SAMPLES} samples)...")
    # Truth `background_rate_hz` (1.5) exceeds production's 1.0-Hz hard cap, so
    # `_set_background_to_fill_if_too_high` would null every background fit and
    # the MC validation could never recover the background parameter. Patch it
    # to a no-op for the duration of the MC run; fork-pool workers inherit the
    # patched module attribute.
    with patch(
        "imap_l3_processing.swapi.l3a.science.pickup_ion."
        "calculate_pickup_ion_values._set_background_to_fill_if_too_high",
        lambda _param_vals: None,
    ):
        result = runner.run(chunks, fitter)

    cooling_index_arr = result["cooling_index"]
    ionization_rate_arr = result["ionization_rate"]
    cutoff_speed_arr = result["cutoff_speed"]
    background_rate_arr = result["background_rate"]
    fits = np.array(
        [
            [u.nominal_value for u in cooling_index_arr],
            [u.nominal_value for u in ionization_rate_arr],
            [u.nominal_value for u in cutoff_speed_arr],
            [u.nominal_value for u in background_rate_arr],
        ]
    ).T
    sigmas = np.array(
        [
            [u.std_dev for u in cooling_index_arr],
            [u.std_dev for u in ionization_rate_arr],
            [u.std_dev for u in cutoff_speed_arr],
            [u.std_dev for u in background_rate_arr],
        ]
    ).T
    good = np.all(np.isfinite(fits) & np.isfinite(sigmas), axis=1)
    n_good = int(good.sum())
    print(f"  {n_good}/{_MC_N_SAMPLES} fits succeeded")

    # Pick the example seed for the spectrogram row, falling back from
    # `_EXAMPLE_SEED` to the first seed whose fit succeeded — seed 0 is not
    # guaranteed to converge, and `_evaluate_fitted_total_rate` needs finite
    # fit parameters to forward-model the spectrogram.
    example_seed = _EXAMPLE_SEED if good[_EXAMPLE_SEED] else int(np.argmax(good))
    example_fits = fits[example_seed]
    mc_rng = np.random.default_rng(example_seed)
    example_observed_counts = mc_rng.poisson(expected_counts_coarse)
    example_rate_coarse = example_observed_counts.astype(float) / SWAPI_LIVETIME_S
    print(
        f"Example realization (seed {example_seed}): "
        f"cooling_index={example_fits[0]:.3f}, "
        f"ionization_rate={example_fits[1]:.3e}, "
        f"cutoff_speed={example_fits[2]:.1f} km/s, "
        f"background={example_fits[3]:.3f} Hz"
    )

    example_epoch = (
        int(sci_start_time[0]) + example_seed * 1_000_000 + FIVE_MINUTES_IN_NANOSECONDS
    )
    fitted_total_rate = _evaluate_fitted_total_rate(
        voltage_v=voltage_v,
        bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
        proton_alpha_rate_hz_per_bin=proton_alpha_rate_hz_per_bin,
        sw_velocity_rtn=sw_velocity_rtn,
        epoch=example_epoch,
        helium_efficiency_ratio=helium_efficiency_ratio,
        swapi_response=swapi_response,
        density_lookup_table=density_lookup_table,
        helium_inflow_vector=helium_inflow_vector,
        example_fits=example_fits,
    )

    truth_values = (
        cooling_index_truth,
        ionization_rate_truth_hz,
        cutoff_speed_truth_kms,
        background_rate_truth_hz,
    )
    _plot(
        voltage_v,
        n_sweeps,
        expected_rate_hz_per_bin,
        example_rate_coarse,
        fitted_total_rate,
        fits[good],
        sigmas[good],
        truth_values,
    )


def _evaluate_fitted_total_rate(
    *,
    voltage_v: np.ndarray,
    bulk_sw_per_bin_swapi_kms: np.ndarray,
    proton_alpha_rate_hz_per_bin: np.ndarray,
    sw_velocity_rtn: np.ndarray,
    epoch: int,
    helium_efficiency_ratio: float,
    swapi_response,
    density_lookup_table,
    helium_inflow_vector,
    example_fits: np.ndarray,
) -> np.ndarray:
    """Forward-model row 3: truth proton/alpha ideal rate + fitted PUI ideal
    rate (over the full voltage grid) + fitted background, all wrapped in the
    same deadtime factor used to build the truth fixture. Mirrors the truth
    construction so rows 1 and 3 are directly comparable.

    `calculate_pickup_ion_values` masks bins to the production PUI fit window
    (above the proton-PUI cutoff and below 1.2× the helium-PUI cutoff); we
    rebuild the chunk response over the full voltage grid so the spectrogram
    covers the same axis as rows 1 and 2. Below the proton-PUI cutoff the PUI
    model contributes ~0, so the full-grid extension just zero-pads the
    low-energy tail.
    """
    cooling_index, ionization_rate, cutoff_speed_kms, background_rate = example_fits
    chunk_ephemeris_time = spiceypy.unitim(
        epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"
    )
    sw_speed_kms = float(np.linalg.norm(sw_velocity_rtn))

    vasyliunas_siscoe_distribution = build_vasyliunas_siscoe_distribution(
        chunk_ephemeris_time,
        sw_velocity_rtn,
        density_lookup_table,
        helium_inflow_vector,
    )
    # Production fit uses sw_speed * 1.2 as the v' grid ceiling; widen if the
    # fitted cutoff landed above that so the grid still reaches the cutoff.
    cutoff_speed_max_kms = max(sw_speed_kms * 1.2, cutoff_speed_kms)
    radius_in_au = vasyliunas_siscoe_distribution.distance_km / ONE_AU_IN_KM
    min_speed_kms = max(
        1.0,
        sw_speed_kms * 0.8 * density_lookup_table.get_minimum_distance() / radius_in_au,
    )

    chunk_response_full = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=voltage_v,
        bulk_sw_per_bin_kms=bulk_sw_per_bin_swapi_kms,
        mass_per_charge_m_p_per_e=_HELIUM_MASS_PER_CHARGE_M_P_PER_E,
        cutoff_speed_max_kms=cutoff_speed_max_kms,
        min_speed_kms=min_speed_kms,
        central_effective_area_scale=helium_efficiency_ratio,
    )

    # `calculate_coincidence_rate` does integer arithmetic on cutoff_speed, so
    # pass nominal floats. Add the fitted background once at the total level
    # before deadtime, matching the truth fixture.
    pui_nominal_params = FittingParameters(
        cooling_index=cooling_index,
        ionization_rate=ionization_rate,
        cutoff_speed=cutoff_speed_kms,
        background_count_rate=0.0,
    )
    pui_rate = calculate_coincidence_rate(
        chunk_response_full, vasyliunas_siscoe_distribution, pui_nominal_params
    )

    ideal_total = proton_alpha_rate_hz_per_bin + pui_rate + background_rate
    return ideal_total * deadtime_factor(ideal_total)


def _plot(
    voltage_v: np.ndarray,
    n_sweeps: int,
    truth_rate: np.ndarray,
    noisy_rate: np.ndarray,
    fitted_rate: np.ndarray,
    fits: np.ndarray,
    sigmas: np.ndarray,
    truth_values: tuple,
) -> None:
    fig = plt.figure(figsize=(14, 17))
    # Two separate gridspecs so the spectrograms can share x with no hspace
    # while the histograms share y with no wspace; a vertical gap in between
    # keeps the two blocks visually separated.
    spectrogram_gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.35,
        hspace=0.0,
        left=0.06,
        right=0.94,
        top=0.95,
        bottom=0.60,
    )
    histogram_gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.0,
        hspace=0.55,
        left=0.06,
        right=0.94,
        top=0.52,
        bottom=0.04,
    )

    # One spectrogram per row, spanning the first three columns. The fourth
    # column on row 0 hosts a single colorbar shared across all three.
    ax_truth = fig.add_subplot(spectrogram_gs[0, 0:3])
    ax_noisy = fig.add_subplot(spectrogram_gs[1, 0:3], sharex=ax_truth, sharey=ax_truth)
    ax_fitted = fig.add_subplot(spectrogram_gs[2, 0:3], sharex=ax_truth, sharey=ax_truth)
    plt.setp(ax_truth.get_xticklabels(), visible=False)
    plt.setp(ax_noisy.get_xticklabels(), visible=False)

    positive_truth = truth_rate[truth_rate > 0]
    vmin = max(1e-2, float(np.nanmin(positive_truth)))
    vmax = float(np.nanmax(positive_truth))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    sweep_edges = np.arange(n_sweeps + 1)
    voltage_edges = _bin_edges_geometric(voltage_v)

    panels = (
        (ax_truth, truth_rate, "Truth model rate (He$^+$ PUI + p/α + bg, deadtime applied)"),
        (ax_noisy, noisy_rate, f"Example Poisson realization (seed {_EXAMPLE_SEED})"),
        (
            ax_fitted,
            fitted_rate,
            "Fitted total rate (truth p/α + fitted PUI + fitted bg, deadtime applied)",
        ),
    )
    mesh = None
    for ax, data, title in panels:
        plot_data = np.where(data > 0, data, np.nan)
        # Sweep index on the x-axis, ESA voltage on a (non-inverted) log y-axis
        # so higher voltages sit at the top of the panel.
        mesh = ax.pcolormesh(
            sweep_edges, voltage_edges, plot_data.T,
            norm=norm, cmap="viridis", shading="auto", rasterized=True,
        )
        ax.set_yscale("log")
        ax.set_ylabel("ESA voltage [V]")
        # With `hspace=0` the spectrograms are stacked flush, so there's no
        # room for an external title above each row. Place the label inside
        # the axes (top-left, with a semi-opaque background) to keep them
        # legible against the viridis spectrogram.
        ax.text(
            0.01, 0.95, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"),
        )
    ax_fitted.set_xlabel("Sweep index")

    # Shared vertical colorbar spanning all three spectrogram rows.
    cbar_top = ax_truth.get_position().y1
    cbar_bottom = ax_fitted.get_position().y0
    cbar_left = ax_truth.get_position().x1 + 0.015
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, 0.012, cbar_top - cbar_bottom])
    fig.colorbar(mesh, cax=cbar_ax, label="Coincidence rate [Hz]")

    param_titles = (
        "Cooling Index",
        "Ionization Rate [Hz]",
        "Cutoff Speed [km/s]",
        "Background [Hz]",
    )

    truth_array = np.asarray(truth_values)
    normalized_errors_all = (fits - truth_array) / sigmas

    # Row 4 of the grid: distribution of fitted nominal values. Column title
    # goes here only; the σ̂ and normalized-error rows share it positionally.
    row4_axes = []
    for k in range(4):
        sharey_ax = row4_axes[0] if row4_axes else None
        ax = fig.add_subplot(histogram_gs[0, k], sharey=sharey_ax)
        row4_axes.append(ax)
        if k > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        values = fits[:, k]
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=40, color="tab:blue", alpha=0.8, edgecolor="white")
        ax.axvline(truth_values[k], color="k", ls="--", lw=1.2, label="truth")
        mean_fit = float(np.mean(finite))
        ax.axvline(mean_fit, color="tab:red", ls="-", lw=1.0, alpha=0.85, label="mean")
        ax.set_title(param_titles[k], fontsize=10)
        ax.set_xlabel(param_titles[k], fontsize=9)
        if k == 0:
            ax.set_ylabel("Fitted value\nMC count", fontsize=9)
        std = float(np.std(finite, ddof=1))
        rel_bias = (mean_fit - truth_values[k]) / truth_values[k]
        ax.annotate(
            f"truth = {truth_values[k]:g}\n"
            f"mean = {mean_fit:g}\n"
            f"σ = {std:g}\n"
            f"bias = {rel_bias:+.2%}",
            xy=(0.04, 0.96),
            xycoords="axes fraction",
            va="top", ha="left",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # Row 5 of the grid: distribution of per-fit reported σ̂.
    row5_axes = []
    for k in range(4):
        sharey_ax = row5_axes[0] if row5_axes else None
        ax = fig.add_subplot(histogram_gs[1, k], sharey=sharey_ax)
        row5_axes.append(ax)
        if k > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        finite_sigma = sigmas[:, k][np.isfinite(sigmas[:, k])]
        sigma_upper = float(np.percentile(finite_sigma, 99))
        clipped_sigma = finite_sigma[finite_sigma <= sigma_upper]
        n_clipped = int(finite_sigma.size - clipped_sigma.size)
        ax.hist(clipped_sigma, bins=40, color="tab:orange", alpha=0.8, edgecolor="white")
        mean_sigma = float(np.mean(finite_sigma))
        ax.axvline(
            mean_sigma, color="tab:red", ls="-", lw=1.0, alpha=0.85,
            label="mean σ̂",
        )
        empirical_std = float(np.std(fits[:, k][np.isfinite(fits[:, k])], ddof=1))
        ax.axvline(
            empirical_std, color="k", ls="--", lw=1.2,
            label="empirical std",
        )
        ax.set_xlabel(f"σ̂({param_titles[k]})", fontsize=9)
        if k == 0:
            ax.set_ylabel("Reported σ̂\nMC count", fontsize=9)
        rel_sigma_error = (mean_sigma - empirical_std) / empirical_std
        annotation_lines = [
            f"mean σ̂ = {mean_sigma:g}",
            f"empirical std = {empirical_std:g}",
            f"σ̂/std − 1 = {rel_sigma_error:+.2%}",
        ]
        if n_clipped:
            annotation_lines.append(f"clipped: {n_clipped}/{finite_sigma.size}")
        ax.annotate(
            "\n".join(annotation_lines),
            xy=(0.04, 0.96),
            xycoords="axes fraction",
            va="top", ha="left",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # Row 6 of the grid: normalized error distribution (fit − truth) / σ̂.
    row6_axes = []
    for k in range(4):
        sharey_ax = row6_axes[0] if row6_axes else None
        ax = fig.add_subplot(histogram_gs[2, k], sharey=sharey_ax)
        row6_axes.append(ax)
        if k > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        normalized_errors = normalized_errors_all[:, k]
        finite = normalized_errors[np.isfinite(normalized_errors)]
        n_finite = finite.size

        x_max = max(4.0, float(np.percentile(np.abs(finite), 99)))
        x_max = min(x_max, 8.0)
        bins_norm = np.linspace(-x_max, x_max, 51)
        bin_width = bins_norm[1] - bins_norm[0]
        n_clipped = int(np.sum(np.abs(finite) > x_max))

        ax.hist(
            np.clip(finite, -x_max, x_max),
            bins=bins_norm, color="tab:blue",
            alpha=0.8, edgecolor="white",
        )
        x_ref = np.linspace(-x_max, x_max, 200)
        unit_gaussian_pdf = (
            np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
        ) * n_finite * bin_width
        ax.plot(
            x_ref, unit_gaussian_pdf,
            color="k", ls="--", lw=1.2, alpha=0.8,
            label="N(0, 1) (calibrated)",
        )
        ax.set_xlim(-x_max, x_max)

        median_z = float(np.median(finite))
        mad_std = float(1.4826 * np.median(np.abs(finite - median_z)))
        ax.set_xlabel("(fit − truth) / σ̂", fontsize=9)
        if k == 0:
            ax.set_ylabel("(fit − truth) / σ̂\nMC count", fontsize=9)
        annotation_lines = [
            f"std (MAD) = {mad_std:.2f}",
            f"median = {median_z:+.2f}",
        ]
        if n_clipped:
            annotation_lines.append(f"clipped: {n_clipped}/{n_finite}")
        ax.annotate(
            "\n".join(annotation_lines),
            xy=(0.04, 0.96),
            xycoords="axes fraction",
            va="top", ha="left",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    fig.suptitle(
        f"PUI fit MC validation on `pui_count_rate_reference_50sweep.h5` "
        f"({fits.shape[0]} of {_MC_N_SAMPLES} Poisson realizations recovered)",
        fontsize=11,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "pui_mc_validation.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def _bin_edges_geometric(centers: np.ndarray) -> np.ndarray:
    """Edges for a log-spaced pcolormesh axis: geometric midpoints between
    centers, outermost edges mirrored."""
    log_c = np.log(centers)
    mid = 0.5 * (log_c[:-1] + log_c[1:])
    first = log_c[0] - (mid[0] - log_c[0])
    last = log_c[-1] + (log_c[-1] - mid[-1])
    return np.exp(np.concatenate([[first], mid, [last]]))


if __name__ == "__main__":
    main()
