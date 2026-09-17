#!/usr/bin/env python3
"""
Plot alpha peak-finding + moments fit on real L2 SWAPI spectra.

Three side-by-side panels (one per converged real-data chunk) showing the
5-sweep-averaged observed count rate vs ESA voltage overlaid with: the
Stage-1 proton model, the Stage-2 alpha-only model, the combined (p + α)
fit, and the alpha peak window detected by `calculate_initial_guess`.

The L2 CDF, MAG L2 (with L1D fallback), and SPICE kernels are downloaded
via imap-data-access; the IMAP API key is read from the IMAP_API_KEY
environment variable.

Output: docs/swapi/figures/alpha_peak_finding.png
Usage:  python docs/swapi/figure_src/plot_alpha_peak_finding.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import imap_data_access
import matplotlib
import numpy as np
import requests
import spacepy.pycdf
import spiceypy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_MASS_KG,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_BIN_PERIOD_S,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_LIVETIME_CENTER_OFFSET_S,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.utils import (
    compute_direction_of_mean_magnetic_field_over_chunk,
    get_swapi_geometry,
    read_mag_rtn_data,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.utils import SpiceKernelTypes
from imap_l3_processing.swapi.species import Species
from figure_utils import (
    require_imap_api_key,
    FIGURES_DIR,
    NOMINAL_EPOCH_TT2000,
    find_sweep_start_index,
    load_swapi_response,
    save_figure,
)

DATE_YYYYMMDD = "20260101"
N_SWEEPS = 5

_CASE_SWEEP_START_TIMES_UTC = [
    "2026-01-01T07:54:04.981",
    "2026-01-01T04:09:04.993",
    "2026-01-01T09:09:04.977",
]


# --------------------------------------------------------------------------- #
# imap-data-access auth + downloads (mirrors plot_real_data_fit.py)
# --------------------------------------------------------------------------- #


def _download_l2(date_yyyymmdd: str) -> Path:
    matches = imap_data_access.query(
        instrument="swapi",
        data_level="l2",
        descriptor="sci",
        start_date=date_yyyymmdd,
        end_date=date_yyyymmdd,
        version="latest",
    )
    if not matches:
        raise SystemExit(f"No SWAPI L2 file for {date_yyyymmdd}")
    file_name = matches[0]["file_path"].split("/")[-1]
    return imap_data_access.download(file_name)


def _download_mag_rtn(date_yyyymmdd: str) -> Path | None:
    """Prefer MAG L2 (norm-rtn); fall back to L1D. Returns None if neither exists."""
    for level in ("l2", "l1d"):
        matches = imap_data_access.query(
            instrument="mag",
            data_level=level,
            descriptor="norm-rtn",
            start_date=date_yyyymmdd,
            end_date=date_yyyymmdd,
            version="latest",
        )
        if matches:
            file_name = matches[0]["file_path"].split("/")[-1]
            print(f"Using MAG {level.upper()} ({file_name})")
            return imap_data_access.download(file_name)
    return None


def _furnish_spice_around(date_yyyymmdd: str) -> None:
    if spiceypy.ktotal("ALL") > 5:
        return

    target = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    start = target - timedelta(days=1)
    end = target + timedelta(days=2)
    spk_start = target - timedelta(days=90)

    metakernel_url = (
        urlparse(imap_data_access.config["DATA_ACCESS_URL"])
        ._replace(path="metakernel")
        .geturl()
    )
    j2000 = datetime(2000, 1, 1, 12)

    def query_kernels(types, t0, t1):
        params = {
            "file_types": [t.value for t in types],
            "start_time": str(int((t0 - j2000).total_seconds())),
            "end_time": str(int((t1 - j2000).total_seconds())),
            "list_files": "true",
        }
        resp = requests.get(metakernel_url, params=params, timeout=30)
        resp.raise_for_status()
        return [Path(p).name for p in json.loads(resp.text)]

    imap_spk_types = {
        SpiceKernelTypes.EphemerisReconstructed,
        SpiceKernelTypes.EphemerisPredicted,
    }
    non_spk = [t for t in SpiceKernelTypes if t not in imap_spk_types]

    narrow = query_kernels(non_spk, start, end)
    spk = query_kernels(list(imap_spk_types), spk_start, end)

    seen = set()
    kernels = []
    for k in narrow + spk:
        if k not in seen:
            seen.add(k)
            kernels.append(k)
    print(f"Downloading {len(kernels)} SPICE kernel(s)…")
    for k in kernels:
        path = imap_data_access.download(k)
        spiceypy.furnsh(str(path))


# --------------------------------------------------------------------------- #
# L2 reading
# --------------------------------------------------------------------------- #


def _read_5_sweep_block(
    cdf_path: Path, sweep_start: int, n_sweeps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with spacepy.pycdf.CDF(str(cdf_path)) as cdf:
        epoch_ns = np.asarray(
            cdf.raw_var("epoch")[sweep_start : sweep_start + n_sweeps], dtype=np.int64
        )
        cr = np.asarray(
            cdf["swp_coin_rate"][sweep_start : sweep_start + n_sweeps, :], dtype=float
        )
        esa = np.asarray(
            cdf["esa_energy"][sweep_start : sweep_start + n_sweeps, :], dtype=float
        )
    return epoch_ns, cr, esa


def _coarse_measurement_times_ns(epoch_ns: np.ndarray) -> np.ndarray:
    # Claude: TODO `epoch` is the sweep centre, not the sweep start, so every
    # bin time below lands ~6 s late. Build these from `sci_start_time` instead.
    coarse_bins = np.arange(SWAPI_COARSE_SWEEP_BINS.start, SWAPI_COARSE_SWEEP_BINS.stop)
    seconds_into_sweep = (
        coarse_bins * SWAPI_BIN_PERIOD_S + SWAPI_LIVETIME_CENTER_OFFSET_S
    )
    return (
        epoch_ns[:, None] + seconds_into_sweep * ONE_SECOND_IN_NANOSECONDS
    ).flatten()


# --------------------------------------------------------------------------- #
# Per-chunk plot
# --------------------------------------------------------------------------- #


def _plot_case(ax, swapi_response, cdf_path, mag_data, sweep_start_time_utc):
    sweep_start = find_sweep_start_index(cdf_path, sweep_start_time_utc)
    epoch_ns, count_rate_full, esa_energy_full = _read_5_sweep_block(
        cdf_path, sweep_start, N_SWEEPS
    )
    count_rates = count_rate_full[:, SWAPI_COARSE_SWEEP_BINS]
    voltages = esa_energy_full[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
    n_sweeps, n_bins = count_rates.shape

    times_ns = _coarse_measurement_times_ns(epoch_ns)
    rotation_matrices = get_swapi_geometry(times_ns)

    swapi_response.warm_cache(voltages.flatten())

    proton_ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=voltages,
        swapi_response=swapi_response,
        time_as_tt2000=NOMINAL_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=rotation_matrices,
    )
    proton_moments = fit_solar_wind_proton_model(proton_ctx)
    proton_velocity_rtn = proton_moments.velocity_rtn_nominal()
    proton_sw = SolarWindParams(
        density=proton_moments.density.nominal_value,
        velocity_rtn=proton_velocity_rtn,
        temperature=proton_moments.temperature.nominal_value,
        mass=PROTON_MASS_KG,
    )
    proton_true, _ = model_solar_wind_ideal_coincidence_rates(proton_sw, proton_ctx)
    count_avg = count_rates.mean(axis=0)

    alpha_ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=voltages,
        swapi_response=swapi_response,
        time_as_tt2000=NOMINAL_EPOCH_TT2000,
        species=Species.ALPHA,
        rotation_matrices=rotation_matrices,
    )
    seed = calculate_initial_guess(
        alpha_ctx=alpha_ctx,
        proton_true_rate=proton_true,
        proton_temperature=proton_moments.temperature.nominal_value,
        proton_velocity_rtn=proton_velocity_rtn,
    )
    has_peak = seed is not None
    peak_idx = seed[3] if has_peak else np.array([], dtype=int)

    chunk_center_ns = int(epoch_ns[0]) + 30 * ONE_SECOND_IN_NANOSECONDS
    if mag_data is None:
        b_hat = np.full(3, np.nan)
    else:
        b_hat = compute_direction_of_mean_magnetic_field_over_chunk(
            mag_data, chunk_center_ns, 30 * ONE_SECOND_IN_NANOSECONDS
        )
    alpha_moments = fit_solar_wind_alpha_model(
        proton_ctx=proton_ctx,
        alpha_ctx=alpha_ctx,
        proton_moments=proton_moments,
        magnetic_field_direction=b_hat,
    )
    alpha_velocity_rtn = np.array([c.nominal_value for c in alpha_moments.velocity_rtn])
    alpha_sw = SolarWindParams(
        density=alpha_moments.density.nominal_value,
        velocity_rtn=alpha_velocity_rtn,
        temperature=alpha_moments.temperature.nominal_value,
        mass=ALPHA_PARTICLE_MASS_KG,
    )
    alpha_true_fit, _ = model_solar_wind_ideal_coincidence_rates(alpha_sw, alpha_ctx)
    combined_true = proton_true + alpha_true_fit
    combined_deadtime = deadtime_factor(combined_true)
    proton_obs = proton_true * combined_deadtime
    alpha_obs = alpha_true_fit * combined_deadtime
    combined_obs = combined_true * combined_deadtime
    proton_avg = proton_obs.reshape(n_sweeps, n_bins).mean(axis=0)
    alpha_avg = alpha_obs.reshape(n_sweeps, n_bins).mean(axis=0)
    combined_avg = combined_obs.reshape(n_sweeps, n_bins).mean(axis=0)

    voltage_per_sweep = np.abs(voltages[0])
    sort_idx = np.argsort(voltage_per_sweep)
    voltage_sorted = voltage_per_sweep[sort_idx]

    ax.plot(
        voltage_sorted,
        count_avg[sort_idx],
        ".",
        color="black",
        markersize=4,
        label="Observed (5-sweep avg)",
        zorder=3,
    )
    ax.plot(
        voltage_sorted,
        proton_avg[sort_idx],
        color="tab:blue",
        lw=1.5,
        label="Proton model",
        zorder=2,
    )
    ax.plot(
        voltage_sorted,
        alpha_avg[sort_idx],
        color="tab:red",
        lw=1.5,
        label="Alpha model",
        zorder=2,
    )
    ax.plot(
        voltage_sorted,
        combined_avg[sort_idx],
        color="tab:purple",
        lw=1.5,
        linestyle="--",
        label="Combined fit (p + α)",
        zorder=2,
    )
    if has_peak:
        peak_voltages = voltage_per_sweep[peak_idx]
        ax.axvspan(
            peak_voltages.min(),
            peak_voltages.max(),
            alpha=0.15,
            color="tab:red",
            zorder=1,
        )
        ax.plot(
            peak_voltages,
            count_avg[peak_idx],
            "o",
            color="tab:red",
            markersize=5,
            markerfacecolor="none",
            lw=1.2,
            label="Peak bins",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel("ESA voltage [V]")
    ax.grid(True, which="both", alpha=0.2)


def main():
    require_imap_api_key()
    print(f"Downloading L2 for {DATE_YYYYMMDD}…")
    cdf_path = _download_l2(DATE_YYYYMMDD)
    print("Downloading MAG RTN…")
    mag_path = _download_mag_rtn(DATE_YYYYMMDD)
    mag_data = read_mag_rtn_data(mag_path) if mag_path is not None else None
    if mag_data is None:
        print("No MAG file available — skipping alpha LM overlay.")
    _furnish_spice_around(DATE_YYYYMMDD)

    print("Loading calibration data…")
    swapi_response = load_swapi_response()

    n_cases = len(_CASE_SWEEP_START_TIMES_UTC)
    fig, axes = plt.subplots(
        1,
        n_cases,
        figsize=(4.5 * n_cases, 4.0),
        sharey=True,
        gridspec_kw={"wspace": 0},
    )

    for ax, sweep_start_time_utc in zip(axes, _CASE_SWEEP_START_TIMES_UTC):
        print(f"Plotting chunk starting {sweep_start_time_utc}…")
        _plot_case(ax, swapi_response, cdf_path, mag_data, sweep_start_time_utc)

    axes[0].set_ylabel("Count rate [Hz]")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "alpha_peak_finding.png"
    save_figure(fig, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
