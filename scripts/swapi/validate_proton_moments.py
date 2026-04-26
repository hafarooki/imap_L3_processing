"""
Validate the SWAPI L3a proton moments fitter against real L2 data, overlaying
OMNI reference moments and observed/model count-rate spectrograms.

Usage: python scripts/swapi/validate_proton_moments.py 2026-01-01 [--data-dir DIR]
Data dir needs: data/l2-dataset.h5, kernels/*, data/wind/omni_hro2_5min_*,
                data/wind/omni_coho1hr_merged_mag_plasma_*
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import spiceypy
import xarray as xr
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from spacepy import pycdf
from tqdm import tqdm

from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _model_count_rates,
    _optimize,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_SCIENCE_BINS
from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

K = 1.89  # SWAPI ESA k-factor eV/V
T_LIVE = 0.145  # s per passband
N_SW = 5  # sweeps per fit
V_LO, V_HI = 0.75, 1.35  # proton-core fit window as fraction of peak V
INSTR = REPO / "instrument_team_data/swapi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument(
        "--data-dir", default=str(Path.home() / "projects/swapi-calibration")
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    date = pd.Timestamp(args.date).date()
    data_dir = Path(args.data_dir)
    out = (
        Path(args.out)
        if args.out
        else REPO / f"docs/swapi/figures/validate_proton_moments_{date}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading SPICE...")
    load_spice(data_dir / "kernels")

    print("Loading L2...")
    ds = xr.open_dataset(data_dir / "data/l2-dataset.h5")
    ds = ds.drop_duplicates("epoch").where(
        lambda x: x.epoch.diff("epoch") >= pd.Timedelta("11s"), drop=True
    )
    counts = (
        (ds["swp_coin_rate"] * T_LIVE)
        .round(0)
        .astype(int)
        .assign_coords(esa_energy=ds.esa_energy)
    )
    t0 = pd.Timestamp(date).to_datetime64()
    t1 = (pd.Timestamp(date) + pd.Timedelta("1D")).to_datetime64()
    day = counts.where((counts.epoch >= t0) & (counts.epoch < t1), drop=True)
    day = day.isel(epoch=slice(0, len(day.epoch) // N_SW * N_SW))
    if not len(day.epoch):
        sys.exit(f"No data for {date}")
    print(f"  {len(day.epoch)} sweeps")

    resp = SWAPIResponse.from_files(
        azimuthal_transmission_path=INSTR
        / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        central_effective_area_path=INSTR
        / "imap_swapi_central-effective-area_20260425_v001.csv",
        passband_fit_coefficients_path=INSTR
        / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )

    sci_V = day.esa_energy.values.mean(0)[SWAPI_SCIENCE_BINS] / K
    pb_mask = np.isfinite(sci_V) & (sci_V > 0)
    rep_V = sci_V[pb_mask]
    n_pb, n_g = len(rep_V), len(day.epoch) // N_SW
    n_m = N_SW * n_pb

    # Per-passband TT2000 times: epoch is sweep midpoint; subtract 6 s, add step offset.
    pb_idx = (np.where(pb_mask)[0] + SWAPI_SCIENCE_BINS.start).astype("int64")
    sweep_tt = np.asarray(
        pycdf.lib.v_datetime_to_tt2000(
            pd.to_datetime(day.epoch.values).to_pydatetime()
        ),
        dtype="int64",
    )
    meas_tt = (sweep_tt[:, None] + pb_idx * int(round(12e9 / 72)) - int(6e9)).reshape(
        n_g, n_m
    )

    print("Precomputing SPICE geometry...")
    rots = np.zeros((n_g, n_m, 3, 3))
    scvs = np.zeros((n_g, 3))
    ok = np.zeros(n_g, dtype=bool)
    for g in tqdm(range(n_g), desc="SPICE"):
        try:
            rots[g], scvs[g] = get_swapi_geometry(meas_tt[g])
            ok[g] = True
        except Exception:
            pass

    rates = (
        np.maximum(
            day.values.astype(float).reshape(n_g, N_SW, 72)[:, :, SWAPI_SCIENCE_BINS][
                :, :, pb_mask
            ],
            0,
        )
        / T_LIVE
    ).reshape(n_g, n_m)
    v_flat = np.tile(rep_V, N_SW)
    mid_times = (
        day.epoch.values.astype("datetime64[ns]")
        .astype("int64")
        .reshape(n_g, N_SW)
        .mean(1)
        .astype("int64")
        .astype("datetime64[ns]")
    )

    def fit_one(i):
        r = rates[i]
        if not (np.all(np.isfinite(r)) and r.max() > 0 and ok[i]):
            return None
        vp = rep_V[r.reshape(N_SW, n_pb).sum(0).argmax()]
        sel_pb = np.where((rep_V >= V_LO * vp) & (rep_V <= V_HI * vp))[0]
        if len(sel_pb) < 5:
            return None
        sel = (np.arange(N_SW)[:, None] * n_pb + sel_pb[None, :]).ravel()
        vw = v_flat[sel]
        # typed.List can't pickle across loky workers; rebuild from cached SWAPIResponse.
        gw = numba.typed.List([resp.create_passband_grid(v) for v in vw])
        try:
            ig = _get_initial_guess(r[sel], vw, gw, rots[i][sel], scvs[i])
            res = _optimize(r[sel], gw, rots[i][sel], scvs[i], ig)
        except Exception:
            return None
        if not np.linalg.norm(res.bulk_velocity_rtn):
            return None
        gf = numba.typed.List([resp.create_passband_grid(v) for v in v_flat])
        model = _model_count_rates(
            res.density, res.temperature, res.bulk_velocity_rtn, gf, rots[i], scvs[i]
        )
        return res, model.reshape(N_SW, n_pb)

    print("Fitting (parallel)...")
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(fit_one)(i) for i in tqdm(range(n_g))
    )

    fits = {k: [] for k in ("epoch", "density", "T_eV", "speed", "vR", "vT", "vN")}
    obs = np.full((n_g, n_pb), np.nan)
    mod = np.full((n_g, n_pb), np.nan)
    for i, res in enumerate(results):
        obs[i] = rates[i].reshape(N_SW, n_pb).mean(0)
        if res is None:
            continue
        r, m2d = res
        mod[i] = m2d.mean(0)
        vr, vt, vn = (float(x) for x in r.bulk_velocity_rtn)
        fits["epoch"].append(pd.Timestamp(mid_times[i]))
        fits["density"].append(r.density)
        fits["T_eV"].append(r.temperature)
        fits["speed"].append(float(np.linalg.norm(r.bulk_velocity_rtn)))
        fits["vR"].append(vr)
        fits["vT"].append(vt)
        fits["vN"].append(vn)
    if not fits["epoch"]:
        sys.exit("No successful fits.")
    print(f"Fit {len(fits['epoch'])} / {n_g} groups")

    _plot(
        out,
        date,
        rep_V,
        mid_times,
        obs,
        mod,
        fits,
        load_omni_5min(data_dir, date),
        load_omni_1hr(data_dir, date),
    )


# --- helpers ---


def load_spice(kernels_dir):
    if (k := REPO / "spice_kernels/imap_wkcp.tf").exists():
        spiceypy.furnsh(str(k))
    for pat in ("*.tpc", "*.tls", "*.tf", "*.tsc", "*.bsp", "*.ah.bc"):
        for k in sorted(kernels_dir.glob(pat)):
            spiceypy.furnsh(str(k))


def _clean(v, hi=9e4):
    v = np.asarray(v, dtype=float)
    return np.where((v >= hi) | (v <= -1e30), np.nan, v)


def _cdf_day(data_dir, glob, date):
    import cdflib

    f = next(
        (
            p
            for p in sorted((data_dir / "data/wind").glob(glob))
            if date.strftime("%Y%m") in p.name
        ),
        None,
    )
    if f is None:
        return None, None, None
    c = cdflib.CDF(str(f))
    ep = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")))
    m = (ep >= pd.Timestamp(date)) & (ep < pd.Timestamp(date) + pd.Timedelta("1D"))
    return c, ep, m


def load_omni_5min(data_dir, date):
    c, ep, _ = _cdf_day(data_dir, "omni_hro2_5min_*_v*.cdf", date)
    if c is None:
        return None
    ts = _clean(c.varget("Timeshift"))
    ep = ep - pd.to_timedelta(np.nan_to_num(ts), unit="s")
    m = (ep >= pd.Timestamp(date)) & (ep < pd.Timestamp(date) + pd.Timedelta("1D"))
    Vx, Vy, Vz = (_clean(c.varget(k))[m] for k in ("Vx", "Vy", "Vz"))
    return {
        "epoch": ep[m],
        "speed": _clean(c.varget("flow_speed"))[m],
        "density": _clean(c.varget("proton_density"), hi=999)[m],
        "T_eV": _clean(c.varget("T"), hi=9e6)[m] * 8.617e-5,
        "vR": -Vx,
        "vT": -Vy,
        "vN": Vz,
    }


def load_omni_1hr(data_dir, date):
    c, ep, m = _cdf_day(data_dir, "omni_coho1hr_merged_mag_plasma_*_v*.cdf", date)
    if c is None:
        return None
    sp = _clean(c.varget("V"))[m]
    az = np.deg2rad(_clean(c.varget("azimuthAngle"), hi=999)[m])
    el = np.deg2rad(_clean(c.varget("elevAngle"), hi=999)[m])
    return {
        "epoch": ep[m],
        "speed": sp,
        "density": _clean(c.varget("N"), hi=999)[m],
        "T_eV": _clean(c.varget("T"), hi=1e10)[m] * 8.617e-5,
        "vR": sp * np.cos(el) * np.cos(az),
        "vT": sp * np.cos(el) * np.sin(az),
        "vN": sp * np.sin(el),
    }


# --- plot ---


def _plot(out, date, rep_V, mid_times, obs, mod, fits, omni5, omni1):
    fig = plt.figure(figsize=(13, 14))
    gs = GridSpec(
        6,
        2,
        figure=fig,
        width_ratios=[20, 1],
        height_ratios=[1.3, 1.3, 1, 1, 1, 1],
        hspace=0.35,
        wspace=0.05,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    caxes = [fig.add_subplot(gs[i, 1]) for i in range(2)]
    for i in range(2, 6):
        fig.add_subplot(gs[i, 1]).set_visible(False)
    fig.suptitle(f"SWAPI proton moments vs OMNI — {date}")

    ope = [pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]

    def omni_lines(ax, field):
        if omni5 is not None:
            ax.plot(
                omni5["epoch"],
                omni5[field],
                "-",
                color="C1",
                lw=2,
                path_effects=ope,
                label="OMNI 5-min",
            )
        if omni1 is not None:
            ax.plot(
                omni1["epoch"],
                omni1[field],
                "-",
                color="k",
                lw=1.5,
                alpha=0.7,
                label="OMNI 1-hr",
            )

    # Spectrograms
    Vi = np.argsort(rep_V)
    lV = np.log10(rep_V[Vi])
    Ve = (
        10
        ** np.r_[
            lV[0] - 0.5 * (lV[1] - lV[0]),
            0.5 * (lV[:-1] + lV[1:]),
            lV[-1] + 0.5 * (lV[-1] - lV[-2]),
        ]
    )
    gns = mid_times.astype("int64")
    te = np.r_[
        gns[0] - (gns[1] - gns[0]) // 2,
        (gns[:-1] + gns[1:]) // 2,
        gns[-1] + (gns[-1] - gns[-2]) // 2,
    ].astype("datetime64[ns]")
    vmax = float(
        np.nanpercentile(
            np.concatenate([obs[obs > 0].ravel(), mod[mod > 0].ravel()]), 99.5
        )
    )
    norm = Normalize(vmin=0, vmax=vmax)

    for ax, cax, data, title in [
        (axes[0], caxes[0], obs, "Measurement"),
        (axes[1], caxes[1], mod, "Model"),
    ]:
        a = np.where(data[:, Vi].T > 0, data[:, Vi].T, np.nan)
        fig.colorbar(
            ax.pcolormesh(te, Ve, a, norm=norm, cmap="viridis", shading="flat"),
            cax=cax,
            label="count rate (Hz)",
        )
        ax.set(yscale="log", ylabel="ESA V (V)")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")

    def v2V(v):
        return 0.5 * 0.01044 * np.asarray(v) ** 2 / K

    Vfit = v2V(fits["speed"])
    pkV = (
        pd.Series(
            [
                rep_V[np.nanargmax(o)] if np.any(np.isfinite(o) & (o > 0)) else np.nan
                for o in obs
            ],
            index=pd.DatetimeIndex(mid_times),
        )
        .rolling("1min", center=True, min_periods=1)
        .mean()
    )
    for ax in axes[:2]:
        ax.plot(pkV.index, pkV.values, "-", color="cyan", lw=1.2, alpha=0.85)
        ax.plot(fits["epoch"], Vfit, "-", color="red", lw=1.0, alpha=0.8)
    axes[0].plot([], [], "-", color="cyan", lw=1.2, label="peak V (1min mean)")
    axes[0].plot([], [], "-", color="red", lw=1.0, label="fitted V_bulk")
    if omni5 is not None:
        axes[0].plot(
            omni5["epoch"],
            v2V(omni5["speed"]),
            "-",
            color="white",
            lw=2,
            path_effects=ope,
            label="OMNI 5-min V_bulk",
        )
    if omni1 is not None:
        axes[0].plot(
            omni1["epoch"],
            v2V(omni1["speed"]),
            "--",
            color="white",
            lw=1.5,
            alpha=0.7,
            label="OMNI 1-hr V_bulk",
        )
    axes[0].legend(loc="upper right", fontsize=8)
    if np.any(np.isfinite(Vfit)):
        for ax in axes[:2]:
            ax.set_ylim(
                float(np.nanpercentile(Vfit, 1) * 0.4),
                float(np.nanpercentile(Vfit, 99) * 2.2),
            )

    for ax, key, ylabel in [
        (axes[2], "speed", "Speed (km/s)"),
        (axes[3], "density", "Density (cm⁻³)"),
        (axes[4], "T_eV", "Temperature (eV)"),
    ]:
        ax.plot(fits["epoch"], fits[key], ".", ms=4, label="SWAPI")
        omni_lines(ax, key)
        p1, p99 = np.percentile(fits[key], [1, 99])
        ax.set(
            ylabel=ylabel,
            ylim=(p1 - 0.1 * max(p99 - p1, 1e-9), p99 + 0.1 * max(p99 - p1, 1e-9)),
        )
    axes[2].legend(loc="upper right", fontsize=8)

    for color, comp in zip(("C0", "C1", "C2"), ("vR", "vT", "vN")):
        axes[5].plot(fits["epoch"], fits[comp], ".", ms=4, color=color, label=comp)
        if omni5 is not None:
            axes[5].plot(
                omni5["epoch"], omni5[comp], "-", color=color, lw=2, path_effects=ope
            )
        if omni1 is not None:
            axes[5].plot(
                omni1["epoch"], omni1[comp], "--", color=color, lw=1.5, alpha=0.7
            )
    axes[5].set(ylabel="Velocity RTN (km/s)")
    axes[5].legend(loc="upper right", ncol=3, fontsize=7)

    for ax in axes[2:]:
        ax.grid(True, alpha=0.3)
    fig.align_ylabels(axes)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
