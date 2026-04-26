"""
Validate the IMAP SWAPI L3a proton solar wind moments fitter against real L2
data, overlaying WIND/OMNI reference moments and the observed / model count
rate spectrograms.

Usage:
    python scripts/swapi/validate_proton_moments.py 2026-01-01 \\
        --data-dir /path/to/swapi-calibration

Data expected under ``data-dir``:
    ./data/l2-dataset.h5                                    L2 sweep counts
    ./kernels/*.tpc, *.tls, *.tf, *.tsc, *.bsp, *.ah.bc     SPICE
    ./data/wind/omni_coho1hr_merged_mag_plasma_*_v*.cdf     OMNI reference

SWAPI response calibration files are loaded from instrument_team_data/swapi/ in the repo.
"""
import argparse
import sys
from pathlib import Path

_THIS_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_THIS_REPO))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import spiceypy
import xarray as xr
from imap_processing.spice.geometry import SpiceFrame, get_rotation_matrix, imap_state
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import (
    _get_initial_guess,
    _model_count_rates,
    _optimize,
)
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse
from imap_l3_processing.swapi.l3a.science.speed_calculation import SWAPI_SCIENCE_BINS

REPO_ROOT = _THIS_REPO

SWAPI_K_FACTOR = 1.89  # eV/V
SAMPLE_TIME = 0.145    # s per passband per sweep

# Restrict fit to the proton thermal core. Alphas appear at V ~ 2*V_peak and
# non-thermal tails bleed into passbands far from peak; including them drags
# the fit away from the Maxwellian core.
V_LOWER_FACTOR = 0.75
V_UPPER_FACTOR = 1.35


def load_spice_kernels(kernels_dir: Path):
    for kernel in [
        kernels_dir / 'pck00011.tpc',
        kernels_dir / 'naif0012.tls',
        kernels_dir / 'imap_130.tf',
        kernels_dir / 'imap_science_110.tf',
        REPO_ROOT / 'spice_kernels/imap_wkcp.tf',
        kernels_dir / 'imap_sclk_0084.tsc',
        kernels_dir / 'de440.bsp',
        kernels_dir / 'imap_pred_od015_20251216_20260127_v01.bsp',
        kernels_dir / 'imap_pred_od016_20260105_20260216_v01.bsp',
        kernels_dir / 'imap_recon_20250925_20251216_v01.bsp',
    ]:
        if kernel.exists():
            spiceypy.furnsh(str(kernel))
    for ck in sorted(kernels_dir.glob('*.ah.bc')):
        spiceypy.furnsh(str(ck))


def epochs_to_et(epochs):
    datetimes = pd.to_datetime(epochs.astype('datetime64[ms]')).to_pydatetime()
    return np.array(spiceypy.datetime2et(datetimes))


def _load_coho1hr(data_dir: Path, target_date):
    """Load OMNI COHO 1-hour merged mag+plasma reference (RTN coordinates)."""
    import cdflib

    wind_dir = data_dir / 'data' / 'wind'
    target_month = pd.Timestamp(target_date).strftime('%Y%m')
    files = sorted(wind_dir.glob('omni_coho1hr_merged_mag_plasma_*_v*.cdf'))
    omni_file = next((p for p in files if target_month in p.name), None)
    if omni_file is None:
        return None

    c = cdflib.CDF(str(omni_file))

    def _clean(v, fill_above=9e4):
        v = np.asarray(v, dtype=float)
        return np.where((v >= fill_above) | (v <= -1e30), np.nan, v)

    ep = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget('Epoch')))
    day_mask = (ep >= pd.Timestamp(target_date)) & \
               (ep < pd.Timestamp(target_date) + pd.Timedelta('1D'))
    speed = _clean(c.varget('V'))
    az_rad = np.deg2rad(_clean(c.varget('azimuthAngle'), fill_above=999))
    el_rad = np.deg2rad(_clean(c.varget('elevAngle'), fill_above=999))
    v = speed[day_mask]
    return {
        'epoch': ep[day_mask],
        'speed': v,
        'density': _clean(c.varget('N'), fill_above=999)[day_mask],
        'T_eV': _clean(c.varget('T'), fill_above=1e10)[day_mask] * 8.617e-5,
        'vR': v * np.cos(el_rad[day_mask]) * np.cos(az_rad[day_mask]),
        'vT': v * np.cos(el_rad[day_mask]) * np.sin(az_rad[day_mask]),
        'vN': v * np.sin(el_rad[day_mask]),
    }


def load_omni(data_dir: Path, target_date):
    """Load OMNI HRO2 5-minute data, reversing the L1-to-bow-shock timeshift."""
    import cdflib

    wind_dir = data_dir / 'data' / 'wind'
    target_month = pd.Timestamp(target_date).strftime('%Y%m')
    hro2_files = sorted(wind_dir.glob('omni_hro2_5min_*_v*.cdf'))
    omni_file = next((p for p in hro2_files if target_month in p.name), None)
    if omni_file is None:
        return None

    c = cdflib.CDF(str(omni_file))

    def _clean(v, fill_above=9e4):
        v = np.asarray(v, dtype=float)
        return np.where((v >= fill_above) | (v <= -1e30), np.nan, v)

    ep = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget('Epoch')))

    # Timeshift is the propagation delay (s) that was ADDED to L1 times
    # to align with Earth's bow shock nose.  Subtract it to recover L1 time.
    timeshift_s = np.asarray(c.varget('Timeshift'), dtype=float)
    timeshift_s = np.where(timeshift_s >= 9e4, np.nan, timeshift_s)
    ep_l1 = ep - pd.to_timedelta(np.nan_to_num(timeshift_s, nan=0.0), unit='s')

    day_mask = (ep_l1 >= pd.Timestamp(target_date)) & \
               (ep_l1 < pd.Timestamp(target_date) + pd.Timedelta('1D'))

    Vx = _clean(c.varget('Vx'))
    Vy = _clean(c.varget('Vy'))
    Vz = _clean(c.varget('Vz'))
    speed = _clean(c.varget('flow_speed'))

    return {
        'epoch': ep_l1[day_mask],
        'speed': speed[day_mask],
        'density': _clean(c.varget('proton_density'), fill_above=999)[day_mask],
        'T_eV': _clean(c.varget('T'), fill_above=9e6)[day_mask] * 8.617e-5,
        # GSE→RTN approximation: R≈−X, T≈−Y (dusk→prograde flip), N≈Z
        'vR': -Vx[day_mask],
        'vT': -Vy[day_mask],
        'vN': Vz[day_mask],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('date', help='target date YYYY-MM-DD')
    parser.add_argument('--data-dir', default=str(Path.home() / 'projects' / 'swapi-calibration'),
                        help='directory containing data/, kernels/, paper/model_inputs/')
    parser.add_argument('--out', default=None,
                        help='output PNG path (default: validate_proton_moments_<date>.png)')
    parser.add_argument('--group-stride', type=int, default=2,
                        help='fit every Nth 5-sweep block (1 for full resolution)')
    args = parser.parse_args()

    target_date = pd.Timestamp(args.date).date()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / 'docs' / 'swapi' / 'figures' / f'validate_proton_moments_{target_date}.png'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('Loading SPICE kernels...')
    load_spice_kernels(data_dir / 'kernels')

    print('Loading L2 dataset...')
    dataset = xr.open_dataset(data_dir / 'data' / 'l2-dataset.h5')
    dataset = dataset.drop_duplicates('epoch').where(
        lambda x: x.epoch.diff('epoch') >= pd.Timedelta('11s'), drop=True
    )
    count = (dataset['swp_coin_rate'] * SAMPLE_TIME).round(0).astype(int)
    count = count.assign_coords(esa_energy=dataset.esa_energy)

    count_day = None
    for date, count_date in count.groupby(count.epoch.dt.date):
        if date == target_date:
            count_day = count_date
            break
    if count_day is None:
        sys.exit(f'No data for {target_date}')
    print(f'Found {len(count_day.epoch)} sweeps for {target_date}')

    # Group 5 consecutive sweeps per fit; stride selects every Nth block.
    sweeps_per_fit = 5
    block = sweeps_per_fit * args.group_stride
    n_day = len(count_day.epoch) // block * block
    count_day = count_day.isel(epoch=slice(0, n_day))
    keep = np.concatenate([np.arange(sweeps_per_fit) + i * block
                           for i in range(n_day // block)])
    count_day = count_day.isel(epoch=keep)

    print('Loading SWAPI response model...')
    _instrument_data = _THIS_REPO / 'instrument_team_data' / 'swapi'
    swapi_response = SWAPIResponse.from_files(
        azimuthal_transmission_path=_instrument_data / 'imap_swapi_proton-sw-azimuthal-transmission_20260425_v001.csv',
        central_effective_area_path=_instrument_data / 'imap_swapi_proton-sw-central-effective-area_20260425_v001.csv',
        passband_fit_coefficients_path=_instrument_data / 'imap_swapi_proton-sw-passband-fit-coefficients_20260425_v001.csv',
    )

    print('Precomputing passband grids...')
    # Only science bins (1–71): index 0 is always discarded; bins 1–62 coarse, 63–71 fine sweep.
    science_esa_voltage = count_day.esa_energy.values.mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_K_FACTOR
    passband_mask = np.isfinite(science_esa_voltage) & (science_esa_voltage > 0)
    rep_V = science_esa_voltage[passband_mask]
    base_grids = numba.typed.List([swapi_response.create_passband_grid(v) for v in rep_V])
    n_passbands = int(passband_mask.sum())
    n_groups = len(count_day.epoch) // sweeps_per_fit
    n_meas = sweeps_per_fit * n_passbands
    print(f'  {n_passbands} passbands x {sweeps_per_fit} sweeps = {n_meas} measurements/fit, {n_groups} fits')

    fit_grids = numba.typed.List()
    for _ in range(sweeps_per_fit):
        for g in base_grids:
            fit_grids.append(g)

    print('Precomputing SPICE quantities...')
    # passband_indices are original bin numbers (1–71) needed for measurement timing
    passband_indices = np.where(passband_mask)[0] + SWAPI_SCIENCE_BINS.start
    step_duration = 12.0 / 72

    # CDF epoch is the MIDDLE of the 12-s sweep, so subtract 6 s to get sweep start
    # before adding the per-passband offset (passband_idx * 12/72 s).
    sweep_ets = epochs_to_et(count_day.epoch.values).reshape(n_groups, sweeps_per_fit)
    meas_ets = (sweep_ets[:, :, None]
                + passband_indices[None, None, :] * step_duration
                - 6.0)

    flat_ets = meas_ets.reshape(-1)
    try:
        flat_rot = get_rotation_matrix(flat_ets, SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI)
        all_rot = flat_rot.reshape(n_groups, sweeps_per_fit, n_passbands, 3, 3)
        ck_valid = np.ones(n_groups, dtype=bool)
    except Exception:
        all_rot = np.zeros((n_groups, sweeps_per_fit, n_passbands, 3, 3))
        ck_valid = np.zeros(n_groups, dtype=bool)
        for g in range(n_groups):
            try:
                all_rot[g] = get_rotation_matrix(
                    meas_ets[g].reshape(-1),
                    SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_SWAPI,
                ).reshape(sweeps_per_fit, n_passbands, 3, 3)
                ck_valid[g] = True
            except Exception:
                pass
    all_rot_flat = all_rot.reshape(n_groups, n_meas, 3, 3)

    # Spacecraft velocity: ~30 km/s +T, varies < 1 km/s per day — use one value.
    sc_velocity = np.zeros(3)
    for g in range(n_groups):
        if not ck_valid[g]:
            continue
        try:
            et = float(sweep_ets[g, 0])
            state_ecl = imap_state(et, SpiceFrame.ECLIPJ2000)
            rtn_from_ecl = get_rotation_matrix(et, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN)
            sc_velocity = np.einsum('ij,j->i', rtn_from_ecl, state_ecl[3:])
            break
        except Exception:
            pass

    all_counts = count_day.values.astype(float).reshape(n_groups, sweeps_per_fit, 72)
    all_counts_masked = np.maximum(all_counts[:, :, SWAPI_SCIENCE_BINS][:, :, passband_mask], 0.0)
    count_rates = (all_counts_masked / SAMPLE_TIME).reshape(n_groups, n_meas)

    v_flat = np.tile(rep_V, sweeps_per_fit)

    epoch_ns = count_day.epoch.values.astype('datetime64[ns]').astype('int64')
    group_mid_times = (epoch_ns.reshape(n_groups, sweeps_per_fit).mean(axis=1)
                       .astype('int64').astype('datetime64[ns]'))

    def fit_one(i):
        rate = count_rates[i]
        if not (np.all(np.isfinite(rate)) and rate.max() > 0 and ck_valid[i]):
            return None
        rate_2d = rate.reshape(sweeps_per_fit, n_passbands)
        peak = int(rate_2d.sum(axis=0).argmax())
        v_peak = rep_V[peak]
        pb_sel = np.where((rep_V <= V_UPPER_FACTOR * v_peak)
                          & (rep_V >= V_LOWER_FACTOR * v_peak))[0]
        if len(pb_sel) < 5:
            return None

        flat_sel = (np.arange(sweeps_per_fit)[:, None] * n_passbands
                    + pb_sel[None, :]).ravel()
        rate_w = rate[flat_sel]
        v_w = v_flat[flat_sel]
        rot_w = all_rot_flat[i][flat_sel]
        grids_w = numba.typed.List([fit_grids[j] for j in flat_sel])

        try:
            ig = _get_initial_guess(rate_w, v_w, grids_w, rot_w, sc_velocity)
            res = _optimize(rate_w, grids_w, rot_w, sc_velocity, ig)
        except Exception:
            return None
        # Also compute model rate over ALL passbands (not just the fit window)
        # for spectrogram overlay.
        if np.linalg.norm(res.bulk_velocity_rtn) == 0:
            return None
        model_full = _model_count_rates(
            res.density, res.temperature, res.bulk_velocity_rtn, fit_grids, all_rot_flat[i], sc_velocity)
        return res, model_full.reshape(sweeps_per_fit, n_passbands)

    print('Fitting groups (parallel)...')
    for i in range(n_groups):
        if ck_valid[i] and count_rates[i].max() > 0:
            fit_one(i)
            break
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(fit_one)(i) for i in tqdm(range(n_groups))
    )

    epochs_out, densities, temperatures, speeds, v_r, v_t, v_n = [], [], [], [], [], [], []
    # Per-group observed/model spectrograms (sweep-summed, one per group)
    obs_spec = np.full((n_groups, n_passbands), np.nan)
    mod_spec = np.full((n_groups, n_passbands), np.nan)
    for i, out in enumerate(results):
        obs_spec[i] = count_rates[i].reshape(sweeps_per_fit, n_passbands).mean(axis=0)
        if out is None:
            continue
        res, model_2d = out
        mod_spec[i] = model_2d.mean(axis=0)
        epochs_out.append(pd.Timestamp(group_mid_times[i]))
        densities.append(res.density)
        temperatures.append(res.temperature)
        vr, vt, vn = res.bulk_velocity_rtn
        speeds.append(float(np.linalg.norm(res.bulk_velocity_rtn)))
        v_r.append(float(vr))
        v_t.append(float(vt))
        v_n.append(float(vn))

    if not epochs_out:
        sys.exit('No successful fits.')
    print(f'Successfully fit {len(epochs_out)} groups')

    ref = load_omni(data_dir, target_date)
    ref1hr = _load_coho1hr(data_dir, target_date)

    # === Plot ===
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13, 14))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[20, 1],
                  height_ratios=[1.3, 1.3, 1, 1, 1, 1], hspace=0.35, wspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    cax0 = fig.add_subplot(gs[0, 1])
    cax1 = fig.add_subplot(gs[1, 1])
    for i in range(2, 6):
        fig.add_subplot(gs[i, 1]).set_visible(False)
    fig.suptitle(f'SWAPI proton moments vs OMNI — {target_date}')

    # Spectrogram voltage edges
    V_sorted_idx = np.argsort(rep_V)
    V_sorted = rep_V[V_sorted_idx]
    V_log = np.log10(V_sorted)
    v_edges = np.concatenate([[V_log[0] - 0.5 * (V_log[1] - V_log[0])],
                               0.5 * (V_log[:-1] + V_log[1:]),
                               [V_log[-1] + 0.5 * (V_log[-1] - V_log[-2])]])
    V_edges = 10 ** v_edges

    t_edges_ns = np.concatenate([
        [epoch_ns.reshape(n_groups, sweeps_per_fit)[:, 0].min() - int(60e9)],
        group_mid_times.astype('int64'),
        [group_mid_times.astype('int64')[-1] + int(120e9)],
    ])
    # Use group mid-times as cell centers; build edges between them
    gt_ns = group_mid_times.astype('int64')
    if len(gt_ns) >= 2:
        t_mid_edges = np.concatenate([
            [gt_ns[0] - (gt_ns[1] - gt_ns[0]) // 2],
            (gt_ns[:-1] + gt_ns[1:]) // 2,
            [gt_ns[-1] + (gt_ns[-1] - gt_ns[-2]) // 2],
        ])
    else:
        t_mid_edges = np.array([gt_ns[0] - int(60e9), gt_ns[0] + int(60e9)])
    t_edges = t_mid_edges.astype('datetime64[ns]')

    # Shared color scale across obs and model for direct comparison.
    _pos = np.concatenate([obs_spec[obs_spec > 0].ravel(), mod_spec[mod_spec > 0].ravel()])
    vmin_shared = max(np.nanpercentile(_pos, 5), 0.1)
    vmax_shared = np.nanpercentile(_pos, 99.5)

    spec_norm = Normalize(vmin=0, vmax=vmax_shared)

    def plot_spec(ax, cax, spec, title):
        arr = spec[:, V_sorted_idx].T  # (V, time)
        arr_plot = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
        mesh = ax.pcolormesh(t_edges, V_edges, arr_plot,
                             norm=spec_norm, cmap='viridis', shading='flat')
        ax.set_yscale('log')
        ax.set_ylabel('ESA V (V)')
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
        fig.colorbar(mesh, cax=cax, label='count rate (Hz)')
        return mesh

    plot_spec(axes[0], cax0, obs_spec, 'Measurement')
    # Overlay the passband V corresponding to the fitted bulk speed, per group.
    # V_bulk = m_p * v_bulk^2 / (2 * k_star * q), with v in km/s, V in V
    # T[eV] = 0.5 * 0.01044 * (v_km_s)^2  so V = T / k_star
    speeds_arr = np.array(speeds)
    V_fit_line = 0.5 * 0.01044 * speeds_arr ** 2 / SWAPI_K_FACTOR

    # 1-min rolling mean of peak ESA voltage from observed spectrogram.
    peak_V_all = np.array([
        rep_V[np.nanargmax(obs_spec[i])] if np.any(np.isfinite(obs_spec[i]) & (obs_spec[i] > 0)) else np.nan
        for i in range(n_groups)
    ])
    peak_ser = pd.Series(peak_V_all, index=pd.DatetimeIndex(group_mid_times))
    peak_roll = peak_ser.rolling('1min', center=True, min_periods=1).mean()

    _omni_pe = [pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]

    axes[0].plot(peak_roll.index, peak_roll.values, '-', color='cyan', lw=1.2, alpha=0.85, label='peak V (1min mean)')
    axes[0].plot(epochs_out, V_fit_line, '-', color='red', lw=1.0, alpha=0.8, label='fitted V_bulk')
    if ref is not None:
        V_omni_line = 0.5 * 0.01044 * np.asarray(ref['speed']) ** 2 / SWAPI_K_FACTOR
        axes[0].plot(ref['epoch'], V_omni_line, '-', color='white', lw=2.0,
                     path_effects=_omni_pe, label='OMNI 5-min V_bulk')
    if ref1hr is not None:
        V_omni_1hr = 0.5 * 0.01044 * np.asarray(ref1hr['speed']) ** 2 / SWAPI_K_FACTOR
        axes[0].plot(ref1hr['epoch'], V_omni_1hr, '--', color='white', lw=1.5,
                     alpha=0.7, label='OMNI 1-hr V_bulk')
    axes[0].legend(loc='upper right', fontsize=8)

    plot_spec(axes[1], cax1, mod_spec, 'Model')
    axes[1].plot(peak_roll.index, peak_roll.values, '-', color='cyan', lw=1.2, alpha=0.85)
    axes[1].plot(epochs_out, V_fit_line, '-', color='red', lw=1.0, alpha=0.8)

    # Zoom spectrograms to the proton core band (based on fitted V_bulk range).
    if np.any(np.isfinite(V_fit_line)):
        v_lo = np.nanpercentile(V_fit_line, 1) * 0.4
        v_hi = np.nanpercentile(V_fit_line, 99) * 2.2
        axes[0].set_ylim(v_lo, v_hi)
        axes[1].set_ylim(v_lo, v_hi)

    def rlim(arr, lo=1, hi=99, pad=0.1):
        a = np.asarray(arr)
        p_lo, p_hi = np.percentile(a, [lo, hi])
        span = max(p_hi - p_lo, 1e-9)
        return p_lo - pad * span, p_hi + pad * span

    _omni_kw = dict(lw=2.0, path_effects=_omni_pe)

    axes[2].plot(epochs_out, speeds, '.', ms=4, label='SWAPI')
    if ref is not None:
        axes[2].plot(ref['epoch'], ref['speed'], '-', color='C1', **_omni_kw, label='OMNI 5-min')
    if ref1hr is not None:
        axes[2].plot(ref1hr['epoch'], ref1hr['speed'], '-', color='k', lw=1.5, alpha=0.7, label='OMNI 1-hr')
    axes[2].set_ylabel('Speed (km/s)')
    axes[2].set_ylim(rlim(speeds))
    axes[2].legend(loc='upper right', fontsize=8)

    axes[3].plot(epochs_out, densities, '.', ms=4, label='SWAPI')
    if ref is not None:
        axes[3].plot(ref['epoch'], ref['density'], '-', color='C1', **_omni_kw, label='OMNI 5-min')
    if ref1hr is not None:
        axes[3].plot(ref1hr['epoch'], ref1hr['density'], '-', color='k', lw=1.5, alpha=0.7, label='OMNI 1-hr')
    axes[3].set_ylabel('Density (cm⁻³)')
    axes[3].set_ylim(rlim(densities))

    axes[4].plot(epochs_out, temperatures, '.', ms=4, label='SWAPI')
    if ref is not None:
        axes[4].plot(ref['epoch'], ref['T_eV'], '-', color='C1', **_omni_kw, label='OMNI 5-min')
    if ref1hr is not None:
        axes[4].plot(ref1hr['epoch'], ref1hr['T_eV'], '-', color='k', lw=1.5, alpha=0.7, label='OMNI 1-hr')
    axes[4].set_ylabel('Temperature (eV)')
    axes[4].set_ylim(rlim(temperatures))

    axes[5].plot(epochs_out, v_r, '.', ms=4, color='C0', label='V_R')
    axes[5].plot(epochs_out, v_t, '.', ms=4, color='C1', label='V_T')
    axes[5].plot(epochs_out, v_n, '.', ms=4, color='C2', label='V_N')
    if ref is not None:
        axes[5].plot(ref['epoch'], ref['vR'], '-', color='C0', **_omni_kw)
        axes[5].plot(ref['epoch'], ref['vT'], '-', color='C1', **_omni_kw)
        axes[5].plot(ref['epoch'], ref['vN'], '-', color='C2', **_omni_kw)
    if ref1hr is not None:
        axes[5].plot(ref1hr['epoch'], ref1hr['vR'], '--', color='C0', lw=1.5, alpha=0.7)
        axes[5].plot(ref1hr['epoch'], ref1hr['vT'], '--', color='C1', lw=1.5, alpha=0.7)
        axes[5].plot(ref1hr['epoch'], ref1hr['vN'], '--', color='C2', lw=1.5, alpha=0.7)
    axes[5].set_ylabel('Velocity RTN (km/s)')
    axes[5].legend(loc='upper right', ncol=3, fontsize=7)

    for ax in axes[2:]:
        ax.grid(True, alpha=0.3)

    fig.align_ylabels(axes)
    plt.show()
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
