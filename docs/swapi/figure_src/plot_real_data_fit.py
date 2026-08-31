#!/usr/bin/env python3
"""
Real-data example: SWAPI proton-moments fit on 5 sweeps + averaged sweep.

Picks a real L2 date/sweep block where the fine-sweep bins (63..71) bracket
the proton peak, fits twice per chunk (full SCIENCE bins vs coarse-only bins),
and overlays both forward-modelled spectra on each sweep's measured count
rate. The 6th column shows the average of the 5 sweeps with the same overlays.

The script downloads the L2 CDF and SPICE kernels itself via imap-data-access.
The IMAP API key is read from the IMAP_API_KEY environment variable.
WIND/SWE 2-min ground truth at the chunk epoch is fetched the same way as
plot_fit_accuracy.py — cached under ~/.cache/imap_l3/wind_swe_2m/.

Output: docs/swapi/figures/real_data_fit.svg
Usage:  conda run -n imapL3 python docs/swapi/figure_src/plot_real_data_fit.py
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import imap_data_access
import matplotlib
import numpy as np
import requests
import spacepy.pycdf
import spiceypy
from matplotlib.transforms import blended_transform_factory

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
import scipy.optimize

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.l3a.utils import (
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.constants import (
    SWAPI_BIN_PERIOD_S,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_FINE_SWEEP_BINS,
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_LIVETIME_CENTER_OFFSET_S,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.utils import SpiceKernelTypes
from figure_utils import FIGURES_DIR, REPO_ROOT, load_swapi_response

_DOC_PATH = REPO_ROOT / "docs" / "swapi" / "proton-sw.md"
_TABLE_BEGIN = "<!-- BEGIN: real_data_table"
_TABLE_END = "<!-- END: real_data_table -->"

# 2026-02-01: 5-sweep block at sweep_start=6853, identified by inspecting the
# day's fine-sweep voltages — fine bins span 5..334 V and bracket the ~257 V
# proton peak. 60-second chunk covers 22:50:36 .. 22:51:36 UT.
DATE_YYYYMMDD = "20260201"
SWEEP_START = 6853
N_SWEEPS = 5

WIND_YEAR = 2026
WIND_CACHE_DIR = Path.home() / ".cache" / "imap_l3" / "wind_swe_2m"
WIND_BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min"


def main():
    if not os.environ.get("IMAP_API_KEY"):
        raise SystemExit(
            "IMAP_API_KEY environment variable is not set. "
            "Export it before running this script."
        )
    cdf_path = _download_l2(DATE_YYYYMMDD)
    _furnish_spice_around(DATE_YYYYMMDD)

    sweep_start = SWEEP_START
    epoch_ns, count_rate, esa_voltage = _read_5_sweep_block(
        cdf_path, sweep_start, N_SWEEPS
    )
    chunk_center_ns = int(epoch_ns[0]) + 30 * ONE_SECOND_IN_NANOSECONDS
    chunk_center_dt = _tt2000ns_to_datetime(chunk_center_ns)
    print(
        f"Chunk: {DATE_YYYYMMDD} sweeps {sweep_start}..{sweep_start + N_SWEEPS - 1} "
        f"(center {chunk_center_dt.isoformat()} UT)"
    )

    swapi_response = load_swapi_response()
    swapi_response.warm_cache(esa_voltage.flatten() / SWAPI_L2_K_FACTOR)

    science_result = _fit_for_bins(
        SWAPI_SCIENCE_BINS, count_rate, esa_voltage, epoch_ns, swapi_response
    )
    coarse_result = _fit_for_bins(
        SWAPI_COARSE_SWEEP_BINS, count_rate, esa_voltage, epoch_ns, swapi_response
    )

    sc_velocity_rtn = get_spacecraft_velocity_rtn(chunk_center_ns)
    science_result["velocity_rtn_sun"] = (
        np.array(
            [c.nominal_value for c in science_result["fit_result"].velocity_rtn]
        )
        + sc_velocity_rtn
    )
    coarse_result["velocity_rtn_sun"] = (
        np.array(
            [c.nominal_value for c in coarse_result["fit_result"].velocity_rtn]
        )
        + sc_velocity_rtn
    )

    print("\n--- Fit results ---")
    print(f"  Science bins (1..71, includes fine):")
    _print_fit(science_result, sc_velocity_rtn)
    print(f"  Coarse-only bins (1..62):")
    _print_fit(coarse_result, sc_velocity_rtn)

    print("\n--- Bootstrap σ on the all-bins fit (B=300, warm-LM, xtol=1e-3) ---")
    boot = _bootstrap_sigmas(
        science_result,
        B=300,
        xtol=1e-3,
        seed=42,
        sc_velocity_rtn=sc_velocity_rtn,
    )
    print(
        f"  kept {boot['B_kept']}/{boot['B_total']} resamples\n"
        f"  σ_n  = {boot['n_sigma']:.4f} cm^-3   (HC3: {science_result['fit_result'].density.std_dev:.4f})\n"
        f"  σ_T  = {boot['T_sigma']:.3e} K       (HC3: {science_result['fit_result'].temperature.std_dev:.3e})\n"
        f"  σ_vR = {boot['vR_sigma']:.3f} km/s   (HC3: {science_result['fit_result'].velocity_rtn[0].std_dev:.3f})\n"
        f"  σ_vT = {boot['vT_sigma']:.3f} km/s   (HC3: {science_result['fit_result'].velocity_rtn[1].std_dev:.3f})\n"
        f"  σ_vN = {boot['vN_sigma']:.3f} km/s   (HC3: {science_result['fit_result'].velocity_rtn[2].std_dev:.3f})"
    )

    wind = _wind_value_at(chunk_center_dt)
    print("\n--- WIND/SWE 2-min comparison ---")
    if wind is None:
        print("  No high-quality WIND/SWE fit within ±2 min of the chunk centre.")
    else:
        print(
            f"  WIND/SWE @ {wind['time'].isoformat()}\n"
            f"    n=({wind['density']:5.3f} ± {wind['density_sigma']:.3f}) cm^-3, "
            f"T=({wind['temperature']:.3e} ± {wind['temperature_sigma']:.1e}) K, "
            f"|v|={wind['speed']:6.1f} ± {wind['speed_sigma']:.2f} km/s,\n"
            f"    v_RTN=[{wind['v_R']:7.2f}±{wind['v_R_sigma']:.2f}, "
            f"{wind['v_T']:6.2f}±{wind['v_T_sigma']:.2f}, "
            f"{wind['v_N']:6.2f}±{wind['v_N_sigma']:.2f}]"
        )

    table_md = _build_intro_table(science_result, coarse_result, boot, wind)
    _update_doc(table_md)
    print(f"\nUpdated table block in {_DOC_PATH.relative_to(REPO_ROOT)}")

    _make_plot(
        epoch_ns,
        count_rate,
        esa_voltage,
        swapi_response,
        science_result,
        coarse_result,
    )


# --------------------------------------------------------------------------- #
# imap-data-access auth + downloads
# --------------------------------------------------------------------------- #


def _download_l2(date_yyyymmdd: str) -> Path:
    """Find the latest SWAPI L2 file for the given date and download it."""
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


def _furnish_spice_around(date_yyyymmdd: str) -> None:
    """Download all SPICE kernels covering [D-1, D+2] (and a 90-day lookback
    for the IMAP SPK so the reconstructed kernel is included), then furnsh.

    Mirrors the two-query strategy in ~/projects/imap-validation/process_swapi_l3.sh.
    """
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
    """Return (epoch_tt2000_ns, swp_coin_rate, esa_energy) for the 5-sweep slice."""
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


def _measurement_times_ns(epoch_ns: np.ndarray, bin_slice: slice) -> np.ndarray:
    bins = np.arange(bin_slice.start, bin_slice.stop)
    seconds_into_sweep = bins * SWAPI_BIN_PERIOD_S + SWAPI_LIVETIME_CENTER_OFFSET_S
    return (epoch_ns[:, None] + seconds_into_sweep * ONE_SECOND_IN_NANOSECONDS).flatten()


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _fit_for_bins(
    bin_slice: slice,
    count_rate: np.ndarray,
    esa_voltage: np.ndarray,
    epoch_ns: np.ndarray,
    swapi_response,
):
    cr = count_rate[:, bin_slice]
    voltages = esa_voltage[:, bin_slice] / SWAPI_L2_K_FACTOR
    times = _measurement_times_ns(epoch_ns, bin_slice)
    rotation_matrices = get_swapi_geometry(times)
    ctx = build_solar_wind_fit_context(
        count_rate=cr,
        esa_voltage=voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    fit_result = fit_solar_wind_proton_model(ctx)
    return {
        "fit_result": fit_result,
        "ctx": ctx,
        "bin_slice": bin_slice,
        "swapi_response": swapi_response,
    }


def _bootstrap_sigmas(
    result,
    B: int = 300,
    xtol: float = 1e-3,
    seed: int = 42,
    sc_velocity_rtn: np.ndarray | None = None,
) -> dict:
    """Warm-started LM bootstrap (no basin hopping) for σ on (n, T, v_RTN)."""
    ctx = result["ctx"]
    fit = result["fit_result"]
    state0 = SolarWindParams(
        density=fit.density.nominal_value,
        velocity_rtn=np.array([c.nominal_value for c in fit.velocity_rtn]),
        temperature=fit.temperature.nominal_value,
        mass=PROTON_MASS_KG,
    ).to_vector()
    cr = ctx.count_rate.ravel()
    v = ctx.esa_voltage.ravel()
    rm = ctx.rotation_matrices
    swapi_response_obj = result["swapi_response"]

    rng = np.random.default_rng(seed)
    fits_n, fits_T, fits_vR, fits_vT, fits_vN = [], [], [], [], []
    for _ in range(B):
        idx = rng.choice(len(cr), len(cr), replace=True)
        ctx_b = build_solar_wind_fit_context(
            count_rate=cr[idx],
            esa_voltage=v[idx],
            swapi_response=swapi_response_obj,
            central_effective_area_scale=1.0,
            rotation_matrices=rm[idx],
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )

        cache = {}

        def _eval(state):
            sw = SolarWindParams.from_vector(state, ctx_b.mass_kg)
            ri, ji = model_solar_wind_ideal_coincidence_rates(sw, ctx_b)
            df = deadtime_factor(ri)
            cache["state"] = state.copy()
            cache["r"] = ri * df - ctx_b.count_rate
            cache["j"] = ji * np.square(df)[:, None]

        def _residues(state):
            if "state" not in cache or not np.array_equal(state, cache["state"]):
                _eval(state)
            return cache["r"]

        def _jacobian(state):
            if "state" not in cache or not np.array_equal(state, cache["state"]):
                _eval(state)
            return cache["j"]

        try:
            raw = scipy.optimize.least_squares(
                _residues,
                state0.copy(),
                jac=_jacobian,
                method="lm",
                xtol=xtol,
            )
        except Exception:
            continue
        if not raw.success:
            continue
        sw = SolarWindParams.from_vector(raw.x, ctx_b.mass_kg)
        fits_n.append(sw.density)
        fits_T.append(sw.temperature)
        fits_vR.append(float(sw.velocity_rtn[0]))
        fits_vT.append(float(sw.velocity_rtn[1]))
        fits_vN.append(float(sw.velocity_rtn[2]))

    arr_n = np.array(fits_n)
    arr_T = np.array(fits_T)
    arr_vR = np.array(fits_vR)
    arr_vT = np.array(fits_vT)
    arr_vN = np.array(fits_vN)
    if sc_velocity_rtn is not None and len(arr_n):
        sun_speed = np.sqrt(
            (arr_vR + sc_velocity_rtn[0]) ** 2
            + (arr_vT + sc_velocity_rtn[1]) ** 2
            + (arr_vN + sc_velocity_rtn[2]) ** 2
        )
        sun_speed_sigma = float(np.std(sun_speed, ddof=1))
    else:
        sun_speed_sigma = float("nan")
    return {
        "B_kept": len(fits_n),
        "B_total": B,
        "n_sigma": float(np.std(arr_n, ddof=1)),
        "T_sigma": float(np.std(arr_T, ddof=1)),
        "vR_sigma": float(np.std(arr_vR, ddof=1)),
        "vT_sigma": float(np.std(arr_vT, ddof=1)),
        "vN_sigma": float(np.std(arr_vN, ddof=1)),
        "sun_speed_sigma": sun_speed_sigma,
    }


def _build_intro_table(science_result, coarse_result, boot, wind):
    """Build the markdown table consumed by the Introduction section."""
    sci_fit = science_result["fit_result"]
    coa_fit = coarse_result["fit_result"]
    sci_v_sun = science_result["velocity_rtn_sun"]
    coa_v_sun = coarse_result["velocity_rtn_sun"]
    # Sun-frame |v| uncertainty: SC velocity is treated as deterministic so
    # |v_sc + v_sun_offset| inherits std from the SC-frame velocity ufloat.
    sci_speed_uf = sum(c**2 for c in sci_fit.velocity_rtn) ** 0.5
    coa_speed_uf = sum(c**2 for c in coa_fit.velocity_rtn) ** 0.5
    sci_speed_sun = float(np.linalg.norm(sci_v_sun))
    coa_speed_sun = float(np.linalg.norm(coa_v_sun))

    if wind is None:
        wind_n = wind_T = wind_v = wind_vR = wind_vT = wind_vN = "—"
        wind_header = "WIND/SWE 2-min (unavailable)"
    else:
        wind_t = wind["time"].strftime("%H:%M:%S")
        wind_header = f"WIND/SWE 2-min @ {wind_t} UT"
        wind_n = f"{wind['density']:.3f} ± {wind['density_sigma']:.3f}"
        wind_T = (
            f"$`({wind['temperature'] / 1e4:.3f} \\pm "
            f"{wind['temperature_sigma'] / 1e4:.2f})\\times10^{{4}}`$"
        )
        wind_v = f"{wind['speed']:.1f} ± {wind['speed_sigma']:.2f}"
        wind_vR = f"{wind['v_R']:+.2f} ± {wind['v_R_sigma']:.2f}"
        wind_vT = f"{wind['v_T']:+.2f} ± {wind['v_T_sigma']:.2f}"
        wind_vN = f"{wind['v_N']:+.2f} ± {wind['v_N_sigma']:.2f}"

    rows = [
        (
            "Density $`n`$ (cm⁻³)",
            f"{sci_fit.density.nominal_value:.3f} ± {sci_fit.density.std_dev:.3f}",
            f"± {boot['n_sigma']:.3f}",
            f"{coa_fit.density.nominal_value:.3f} ± {coa_fit.density.std_dev:.3f}",
            wind_n,
        ),
        (
            "Temperature $`T`$ (K)",
            f"$`({sci_fit.temperature.nominal_value / 1e4:.3f} \\pm "
            f"{sci_fit.temperature.std_dev / 1e4:.3f})\\times10^{{4}}`$",
            f"$`\\pm {boot['T_sigma'] / 1e4:.3f}\\times10^{{4}}`$",
            f"$`({coa_fit.temperature.nominal_value / 1e4:.3f} \\pm "
            f"{coa_fit.temperature.std_dev / 1e4:.3f})\\times10^{{4}}`$",
            wind_T,
        ),
        (
            "Inertial-frame $`\\lvert v \\rvert`$ (km/s)",
            f"{sci_speed_sun:.1f} ± {sci_speed_uf.std_dev:.2f}",
            f"± {boot['sun_speed_sigma']:.2f}",
            f"{coa_speed_sun:.1f} ± {coa_speed_uf.std_dev:.2f}",
            wind_v,
        ),
        (
            "$`v_{R}`$ (km/s, inertial RTN)",
            f"{sci_v_sun[0]:.2f} ± {sci_fit.velocity_rtn[0].std_dev:.2f}",
            f"± {boot['vR_sigma']:.2f}",
            f"{coa_v_sun[0]:.2f} ± {coa_fit.velocity_rtn[0].std_dev:.2f}",
            wind_vR,
        ),
        (
            "$`v_{T}`$ (km/s, inertial RTN)",
            f"{sci_v_sun[1]:+.2f} ± {sci_fit.velocity_rtn[1].std_dev:.2f}",
            f"± {boot['vT_sigma']:.2f}",
            f"{coa_v_sun[1]:+.2f} ± {coa_fit.velocity_rtn[1].std_dev:.2f}",
            wind_vT,
        ),
        (
            "$`v_{N}`$ (km/s, inertial RTN)",
            f"{sci_v_sun[2]:+.2f} ± {sci_fit.velocity_rtn[2].std_dev:.2f}",
            f"± {boot['vN_sigma']:.2f}",
            f"{coa_v_sun[2]:+.2f} ± {coa_fit.velocity_rtn[2].std_dev:.2f}",
            wind_vN,
        ),
    ]

    header_cells = [
        "Quantity",
        "SWAPI all bins (HC3 σ)",
        "SWAPI all bins (boot σ)",
        "SWAPI coarse only (HC3 σ)",
        wind_header,
    ]
    aligns = ["---", "---:", "---:", "---:", "---:"]

    md = ["| " + " | ".join(header_cells) + " |"]
    md.append("|" + "|".join(aligns) + "|")
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


def _update_doc(table_md: str) -> None:
    text = _DOC_PATH.read_text()
    begin = text.find(_TABLE_BEGIN)
    end = text.find(_TABLE_END)
    if begin < 0 or end < 0 or end <= begin:
        raise RuntimeError(
            f"Could not find '{_TABLE_BEGIN}' / '{_TABLE_END}' markers in {_DOC_PATH}"
        )
    line_end = text.find("\n", begin) + 1
    new_text = text[:line_end] + table_md + "\n" + text[end:]
    _DOC_PATH.write_text(new_text)


def _print_fit(result, sc_velocity_rtn: np.ndarray):
    fit = result["fit_result"]
    n = fit.density
    T = fit.temperature
    v_sc = fit.velocity_rtn
    v_sc_nom = np.array([c.nominal_value for c in v_sc])
    v_sun_nom = v_sc_nom + sc_velocity_rtn
    # Inertial speed σ ≈ σ on |v_sc| (sc velocity is ~deterministic per epoch).
    sun_speed_uf = sum(c**2 for c in v_sc) ** 0.5
    print(
        f"    n=({n.nominal_value:6.3f} ± {n.std_dev:.3f}) cm^-3, "
        f"T=({T.nominal_value:.3e} ± {T.std_dev:.1e}) K, "
        f"|v|_sun={float(np.linalg.norm(v_sun_nom)):6.1f} ± {sun_speed_uf.std_dev:.2f} km/s, "
        f"v_RTN_sun=[{v_sun_nom[0]:7.2f}±{v_sc[0].std_dev:.2f}, "
        f"{v_sun_nom[1]:6.2f}±{v_sc[1].std_dev:.2f}, "
        f"{v_sun_nom[2]:6.2f}±{v_sc[2].std_dev:.2f}], "
        f"bad_fit={int(fit.quality_flag)}"
    )


def _model_rates_with_deadtime(
    fit_result, voltages_flat: np.ndarray, rotation_matrices: np.ndarray, swapi_response
) -> np.ndarray:
    """Forward-model count rates at the supplied voltages using a fit's params."""
    swapi_response.warm_cache(voltages_flat)
    ctx = build_solar_wind_fit_context(
        count_rate=np.ones_like(voltages_flat),
        esa_voltage=voltages_flat,
        swapi_response=swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    sw = SolarWindParams(
        density=fit_result.density.nominal_value,
        velocity_rtn=np.array(
            [c.nominal_value for c in fit_result.velocity_rtn]
        ),
        temperature=fit_result.temperature.nominal_value,
        mass=PROTON_MASS_KG,
    )
    rate_ideal, _ = model_solar_wind_ideal_coincidence_rates(sw, ctx)
    return rate_ideal * deadtime_factor(rate_ideal)


# --------------------------------------------------------------------------- #
# WIND value at chunk centre
# --------------------------------------------------------------------------- #


def _wind_value_at(target_dt: datetime) -> dict | None:
    """Return WIND/SWE 2-min ground truth nearest to `target_dt` (within ±2 min).

    Layout of the ASCII file matches scripts/swapi/sample_wind_solar_wind.py.
    """
    asc = WIND_CACHE_DIR / f"wind_swe_2m_sw{WIND_YEAR}.asc"
    if not asc.exists():
        WIND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = f"{WIND_BASE_URL}/{asc.name}"
        print(f"Downloading {url} -> {asc}")
        urllib.request.urlretrieve(url, asc)

    data = np.loadtxt(asc, comments=";")

    fdoy_target = (
        target_dt - datetime(target_dt.year, 1, 1)
    ).total_seconds() / 86400.0 + 1.0
    year = data[:, 0].astype(int)
    fdoy = data[:, 1]
    in_year = year == target_dt.year
    if not in_year.any():
        return None

    delta = np.abs(fdoy[in_year] - fdoy_target)
    idx_in_year = int(np.argmin(delta))
    if delta[idx_in_year] > 2.0 / (60 * 24):  # >2 min
        return None
    row = data[in_year][idx_in_year]
    if int(row[2]) != 10:
        return None
    if any(abs(row[c]) >= 9e4 for c in (3, 5, 7, 9, 11, 21)):
        return None

    # WIND/SWE 2-min ASCII layout: each value is followed by its 1σ uncertainty.
    # Column indices (0-based): 3 V, 4 σV, 5 Vx, 6 σVx, 7 Vy, 8 σVy, 9 Vz,
    # 10 σVz, 11 Wscalar, 12 σWscalar, 21 Np, 22 σNp.
    w_th = row[11]
    sigma_w = row[12]
    T_K = (
        PROTON_MASS_KG
        * (w_th * 1e3) ** 2
        / (2.0 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
    )
    # Linear-error propagation: T = m·w²/(2k)  ⇒  σT = (2T/w)·σw.
    sigma_T = T_K * 2.0 * sigma_w / w_th if w_th > 0 else float("nan")
    return {
        "time": datetime(target_dt.year, 1, 1) + timedelta(days=row[1] - 1),
        "speed": float(row[3]),
        "speed_sigma": float(row[4]),
        "v_R": float(-row[5]),
        "v_R_sigma": float(row[6]),
        "v_T": float(-row[7]),
        "v_T_sigma": float(row[8]),
        "v_N": float(row[9]),
        "v_N_sigma": float(row[10]),
        "temperature": float(T_K),
        "temperature_sigma": float(sigma_T),
        "density": float(row[21]),
        "density_sigma": float(row[22]),
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def _make_plot(
    epoch_ns: np.ndarray,
    count_rate: np.ndarray,
    esa_voltage: np.ndarray,
    swapi_response,
    science_result,
    coarse_result,
):
    voltages = esa_voltage / SWAPI_L2_K_FACTOR
    valid_full = (voltages > 0) & np.isfinite(voltages)
    times_full = _measurement_times_ns(epoch_ns, slice(0, 72))
    rotation_full = get_swapi_geometry(times_full)

    voltages_flat = voltages.flatten()
    valid_flat = valid_full.flatten()
    rotation_valid = rotation_full[valid_flat]
    voltages_valid = voltages_flat[valid_flat]

    science_curve_full = np.full_like(voltages_flat, np.nan)
    coarse_curve_full = np.full_like(voltages_flat, np.nan)
    science_curve_full[valid_flat] = _model_rates_with_deadtime(
        science_result["fit_result"], voltages_valid, rotation_valid, swapi_response
    )
    coarse_curve_full[valid_flat] = _model_rates_with_deadtime(
        coarse_result["fit_result"], voltages_valid, rotation_valid, swapi_response
    )

    science_2d = science_curve_full.reshape(voltages.shape)
    coarse_2d = coarse_curve_full.reshape(voltages.shape)

    avg_voltage = np.nanmean(np.where(voltages > 0, voltages, np.nan), axis=0)

    n_sweeps = count_rate.shape[0]
    n_panels = n_sweeps
    n_rows = n_panels

    fig = plt.figure(figsize=(10.0, 1.55 * n_rows))
    outer = fig.add_gridspec(n_rows, 1, hspace=0.0)
    panel_axes: list[tuple] = []
    for i in range(n_panels):
        inner = outer[i, 0].subgridspec(1, 2, width_ratios=[1.4, 4.5], wspace=0)
        ax_f = fig.add_subplot(inner[0])
        ax_c = fig.add_subplot(inner[1], sharey=ax_f)
        panel_axes.append((ax_c, ax_f))

    color_data = "k"
    color_full = "tab:blue"
    color_coarse = "tab:orange"

    bin_indices = np.arange(72)

    # Fine sub-axis restricted to V > 100 V: the 3 low-V fine bins (~5–35 V)
    # plot at effectively zero rate and would introduce a false drop in the line.
    fine_step_lo = SWAPI_FINE_SWEEP_BINS.start
    fine_step_hi = SWAPI_FINE_SWEEP_BINS.stop - 1
    fine_dense_step = np.linspace(fine_step_lo, fine_step_hi, 200)

    def _dense_fine(values_per_bin, v_per_bin):
        keep = (v_per_bin > 100.0) & np.isfinite(values_per_bin)
        if keep.sum() < 2:
            return fine_dense_step, np.full_like(fine_dense_step, np.nan)
        bins_keep = bin_indices[
            SWAPI_FINE_SWEEP_BINS.start : SWAPI_FINE_SWEEP_BINS.stop
        ][keep[SWAPI_FINE_SWEEP_BINS]]
        order = np.argsort(bins_keep)
        return (
            fine_dense_step,
            np.interp(
                fine_dense_step,
                bins_keep[order],
                values_per_bin[bins_keep[order]],
                left=np.nan,
                right=np.nan,
            ),
        )

    def _draw(panel, cr, s_curve, c_curve, v, *, draw_legend=False):
        ax_c, ax_f = panel
        ok = (v > 0) & np.isfinite(v)
        coarse_mask = ok & (bin_indices < SWAPI_FINE_SWEEP_BINS.start)
        fine_mask_useful = (
            ok & (bin_indices >= SWAPI_FINE_SWEEP_BINS.start) & (v > 100.0)
        )

        ax_c.semilogy(
            bin_indices[coarse_mask],
            np.maximum(cr[coarse_mask], 0.1),
            "o",
            color=color_data,
            ms=3,
            mfc="none",
            mew=0.8,
            label="L2 coarse bin" if draw_legend else None,
        )
        ax_f.semilogy(
            bin_indices[fine_mask_useful],
            np.maximum(cr[fine_mask_useful], 0.1),
            "o",
            color=color_data,
            ms=4.0,
            mfc=color_data,
            mew=0,
            label="L2 fine-sweep bin" if draw_legend else None,
        )

        coarse_steps = bin_indices[coarse_mask]
        for curve, color, ls, label in (
            (s_curve, color_full, "-", "Fit (bins 1..71, incl. fine sweep)"),
            (c_curve, color_coarse, "--", "Fit (bins 1..62, no fine sweep)"),
        ):
            ax_c.semilogy(
                coarse_steps,
                np.maximum(curve[coarse_steps], 0.1),
                ls,
                color=color,
                lw=1.4,
                label=label if draw_legend else None,
            )
            x_dense, y_dense = _dense_fine(curve, v)
            ax_f.semilogy(
                x_dense,
                np.maximum(y_dense, 0.1),
                ls,
                color=color,
                lw=1.4,
            )

    def _peak_centroid(cr, v, peak_slice: slice):
        """Sub-bin peak centroid (3-bin count-rate-weighted mean around the
        brightest bin in `peak_slice`). Returns (bin_centroid, peak_rate)."""
        bins = np.arange(peak_slice.start, peak_slice.stop)
        sub_v = v[peak_slice]
        sub_cr = cr[peak_slice]
        ok = (sub_v > 0) & np.isfinite(sub_v) & np.isfinite(sub_cr)
        if ok.sum() < 3:
            return None, None
        idx_local = int(np.argmax(np.where(ok, sub_cr, -np.inf)))
        idx_global = bins[idx_local]
        lo = max(peak_slice.start, idx_global - 1)
        hi = min(peak_slice.stop, idx_global + 2)
        win_bins = np.arange(lo, hi)
        weights = np.maximum(cr[lo:hi], 0.0)
        if weights.sum() == 0:
            return None, None
        return (
            float((win_bins * weights).sum() / weights.sum()),
            float(cr[idx_global]),
        )

    def _add_peak_markers(panel, cr, v, *, draw_legend=False):
        """'+' markers at the coarse-bin peak (purple) and fine-bin peak
        (red), each annotated above the marker with E/q and rate."""
        ax_c, ax_f = panel
        bin_grid = np.arange(72, dtype=float)
        for peak_slice, sub_ax, color, marker_label in (
            (SWAPI_COARSE_SWEEP_BINS, ax_c, "tab:purple", "Coarse-bin peak"),
            (SWAPI_FINE_SWEEP_BINS, ax_f, "tab:red", "Fine-bin peak"),
        ):
            peak_bin, peak_rate = _peak_centroid(cr, v, peak_slice)
            if peak_bin is None:
                continue
            v_region = v[peak_slice]
            bins_region = bin_grid[peak_slice]
            ok_region = (v_region > 0) & np.isfinite(v_region)
            v_at_peak = float(
                np.interp(peak_bin, bins_region[ok_region], v_region[ok_region])
            )
            e_over_q_eV = SWAPI_K_FACTOR * v_at_peak
            log10_rate = float(np.log10(max(peak_rate, 1.0)))
            x_marker = peak_bin
            sub_ax.plot(
                [x_marker],
                [peak_rate],
                marker="+",
                color=color,
                ms=10,
                mew=1.6,
                zorder=5,
                label=marker_label if draw_legend else None,
            )
            text_transform = blended_transform_factory(
                sub_ax.transData, sub_ax.transAxes
            )
            sub_ax.text(
                x_marker,
                0.98,
                rf"${e_over_q_eV:.4g}\,\frac{{\mathrm{{eV}}}}{{k^*}},\ "
                rf"10^{{{log10_rate:.2f}}}\,\mathrm{{Hz}}$",
                transform=text_transform,
                color=color,
                fontsize=10,
                va="top",
                ha="center",
                zorder=5,
                clip_on=False,
            )

    panel_titles = [f"Sweep {i + 1}" for i in range(n_sweeps)]
    panel_data = list(zip(voltages, count_rate, science_2d, coarse_2d))
    for i, (v, cr, s_curve, c_curve) in enumerate(panel_data):
        panel = panel_axes[i]
        _draw(panel, cr, s_curve, c_curve, v, draw_legend=(i == 0))
        _add_peak_markers(panel, cr, v, draw_legend=(i == 0))
        panel[1].set_ylabel(f"{panel_titles[i]}\nrate (Hz)", fontsize=9)

    avg_v = np.where(np.isnan(avg_voltage), 0.0, avg_voltage)

    peak = float(np.nanmax(count_rate))
    last_panel_idx = len(panel_axes) - 1
    seam_color = "k"
    for i, (ax_c, ax_f) in enumerate(panel_axes):
        ax_f.set_xscale("linear")
        ax_f.set_xlim(fine_step_hi + 0.5, 65.5)
        ax_c.set_xscale("linear")
        ax_c.set_xlim(SWAPI_FINE_SWEEP_BINS.start - 0.5, 0.5)
        ax_f.set_ylim(max(0.1, peak * 1e-5), peak * 60)

        ax_f.spines["right"].set_visible(True)
        ax_f.spines["right"].set_color(seam_color)
        ax_f.spines["right"].set_linewidth(1.0)
        ax_c.spines["left"].set_visible(False)
        ax_c.tick_params(left=False, labelleft=False)
        for ax in (ax_f, ax_c):
            ax.tick_params(labelsize=8)
            ax.grid(True, which="both", alpha=0.25)
        is_last = i == last_panel_idx
        if is_last:
            ax_f.set_xlabel("Fine Steps", fontsize=10)
            ax_c.set_xlabel("Coarse Steps", fontsize=10)
            ax_f.set_xticks([66, 68, 70, 71])
            ax_f.set_xticklabels(["66", "68", "70", "71"], fontsize=8)
            ax_c.set_xticks([1, 10, 20, 30, 40, 50, 60])
            ax_c.set_xticklabels(["1", "10", "20", "30", "40", "50", "60"], fontsize=8)
            ax_f.tick_params(labelbottom=True, bottom=True)
            ax_c.tick_params(labelbottom=True, bottom=True)
        else:
            ax_f.tick_params(labelbottom=False)
            ax_c.tick_params(labelbottom=False)

    fine_label_bins = [b for b in [66, 68, 70, 71] if avg_v[b] > 0]
    coarse_label_bins = [b for b in [1, 5, 10, 20, 30, 40, 50, 60] if avg_v[b] > 0]
    ax_c_top, ax_f_top = panel_axes[0]
    top_f = ax_f_top.secondary_xaxis("top")
    top_f.set_xticks(fine_label_bins)
    top_f.set_xticklabels([f"{avg_v[b]:.0f}V" for b in fine_label_bins], fontsize=7)
    top_f.tick_params(labelsize=7, pad=1)
    top_c = ax_c_top.secondary_xaxis("top")
    top_c.set_xticks(coarse_label_bins)
    top_c.set_xticklabels([f"{avg_v[b]:.0f}V" for b in coarse_label_bins], fontsize=7)
    top_c.set_xlabel("ESA voltage (V)", fontsize=9, labelpad=4)
    top_c.tick_params(labelsize=7, pad=1)

    handles, labels = panel_axes[0][0].get_legend_handles_labels()
    handles_f, labels_f = panel_axes[0][1].get_legend_handles_labels()
    handles += handles_f
    labels += labels_f
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "real_data_fit.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")


def _tt2000ns_to_datetime(tt2000_ns: int) -> datetime:
    """Convert TT2000 nanoseconds to a UTC datetime via spacepy."""
    return spacepy.pycdf.lib.tt2000_to_datetime(int(tt2000_ns))


if __name__ == "__main__":
    main()
