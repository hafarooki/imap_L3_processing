#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
Benchmark the sensitivity of calculate_integral to the number of quadrature points
per integration dimension (elevation, azimuth-SG, azimuth-OA, speed).

For each dimension, N varies from 5 to 201 while the other three dimensions are held
at N_REF=200. Relative error is computed against the N_REF reference across a suite
of representative solar wind conditions.

Output: docs/swapi/figures/integration_benchmark.png

Usage (from repo root):
    python scripts/swapi/integration_benchmark.py
"""
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imap_l3_processing.constants import METERS_PER_KILOMETER, PROTON_CHARGE_COULOMBS, PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_moments import SWParams
from imap_l3_processing.swapi.l3a.science.swapi_response import SWAPIResponse

_SWAPI_K_FACTOR = 1.89  # eV/V

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTRUMENT_DATA = _REPO_ROOT / "instrument_team_data" / "swapi"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "swapi" / "figures"

_ANG_SPACING_REF = 0.1  # degrees; used to derive reference N for elevation and azimuth
_N_SP_REF = 100         # fixed reference N for speed
_N_SWEEP = [5, 7, 10, 15, 21, 31, 51, 101, 151, 201]

# Current production N values, drawn as vertical reference lines
_CURRENT_N = {
    "elevation":  31,
    "azimuth_sg": 31,
    "azimuth_oa": 31,
    "speed":      31,
}

# (label, bulk_speed km/s, temperature eV, bulk_azimuth deg, bulk_elevation deg)
# bulk_azimuth is always within the sunglasses FOV (±20°)
_TEST_CASES = [
    ("slow cold",     300,  5,   0,   0),
    ("typical",       450, 10,  15,  -5),
    ("fast hot",      700, 50,   0,   0),
    ("SG edge az",    450, 10,  18,   0),
    ("off elevation", 450, 10,   0,   8),
    ("hot off-axis",  600, 30,  15,  -8),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_voltage(bulk_speed_km_s: float) -> float:
    return (PROTON_MASS_KG * (bulk_speed_km_s * METERS_PER_KILOMETER) ** 2
            / (2 * _SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS))


def _make_sw_params(bulk_speed, temperature_ev, bulk_azimuth, bulk_elevation, density=5.0) -> SWParams:
    thermal_speed = float(
        np.sqrt(temperature_ev * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG) / METERS_PER_KILOMETER
    )
    return SWParams(
        density=density,
        bulk_speed=bulk_speed,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
        thermal_speed=thermal_speed,
    )


def _trapz_w(a: float, b: float, n: int) -> np.ndarray:
    w = np.full(n, (b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _dyn(center: float, width: float, lo: float, hi: float) -> tuple[float, float]:
    return _clamp(center - width, lo, hi), _clamp(center + width, lo, hi)


def _angular_limits(sw: SWParams, region: int, grid, epsilon_oa: float = 1e-3) -> tuple[float, float, float, float]:
    eps = 1e-3 if region == 0 else epsilon_oa
    arg = sw.thermal_speed ** 2 * math.log(eps) / (grid.central_speed * sw.bulk_speed) + 1
    w = math.degrees(math.acos(max(-1.0, min(1.0, arg))))
    if region == 0:
        el_lim = _dyn(sw.bulk_elevation, w, grid.min_SG_elevation, grid.max_SG_elevation)
        az_lim = _dyn(sw.bulk_azimuth, w, -20.0, 20.0)
    elif region == -1:
        el_lim = _dyn(sw.bulk_elevation, w, grid.min_OA_elevation, grid.max_OA_elevation)
        az_lim = _dyn(sw.bulk_azimuth, w, -150.0, -20.0)
    else:
        el_lim = _dyn(sw.bulk_elevation, w, grid.min_OA_elevation, grid.max_OA_elevation)
        az_lim = _dyn(sw.bulk_azimuth, w, 20.0, 150.0)
    return el_lim + az_lim


def _passband_np(grid, is_sg: bool, el_arr: np.ndarray, sp_arr: np.ndarray) -> np.ndarray:
    """Bilinear passband interpolation. Returns shape (len(el_arr), len(sp_arr))."""
    gv = grid.values_sunglasses if is_sg else grid.values_open_aperture
    i = (el_arr[:, None] - grid.min_elevation) / grid.elevation_spacing
    j = (sp_arr[None, :] / grid.central_speed - grid.min_speed_ratio) / grid.speed_ratio_spacing
    mask = (i >= 0) & (i + 1 < gv.shape[0]) & (j >= 0) & (j + 1 < gv.shape[1])
    i0 = np.clip(i.astype(int), 0, gv.shape[0] - 2)
    j0 = np.clip(j.astype(int), 0, gv.shape[1] - 2)
    wi, wj = i - i0, j - j0
    return np.where(
        mask,
        (1 - wi) * ((1 - wj) * gv[i0, j0] + wj * gv[i0, j0 + 1])
        + wi * ((1 - wj) * gv[i0 + 1, j0] + wj * gv[i0 + 1, j0 + 1]),
        0.0,
    )


def _transmission_np(grid, az_arr: np.ndarray) -> np.ndarray:
    az = (az_arr + 180) % 360 - 180
    i = np.abs(az) / grid.azimuthal_transmission_spacing
    i0 = np.clip(i.astype(int), 0, len(grid.azimuthal_transmission) - 1)
    i1 = np.clip(i0 + 1, 0, len(grid.azimuthal_transmission) - 1)
    return grid.azimuthal_transmission[i0] * (i1 - i) + grid.azimuthal_transmission[i1] * (i - i0)


# ---------------------------------------------------------------------------
# Parameterized integral
# ---------------------------------------------------------------------------

def _region_integral(grid, sw: SWParams, region: int, n_el: int, n_az: int, n_sp: int,
                     epsilon_oa: float, az_outer: bool = False) -> float:
    """
    Trapezoid integral over one azimuth region (0=SG, -1=left OA, +1=right OA).
    az_outer=False: elevation outer loop (matches production code order).
    az_outer=True:  azimuth outer loop (reference for azimuth N sweeps).
    """
    is_sg = region == 0
    min_el, max_el, min_az, max_az = _angular_limits(sw, region, grid, epsilon_oa=epsilon_oa)
    if max_el <= min_el or max_az <= min_az:
        return 0.0

    passband_norm = float(
        _passband_np(grid, is_sg, np.array([0.0]), np.array([grid.central_speed]))[0, 0]
    )
    if passband_norm == 0.0:
        return 0.0

    poly_min = grid.min_SG_poly if is_sg else grid.min_OA_poly
    poly_max = grid.max_SG_poly if is_sg else grid.max_OA_poly

    sin_be = math.sin(math.radians(sw.bulk_elevation))
    cos_be = math.cos(math.radians(sw.bulk_elevation))

    el = np.linspace(min_el, max_el, n_el)
    az = np.linspace(min_az, max_az, n_az)
    el_w = _trapz_w(min_el, max_el, n_el)
    az_w = _trapz_w(min_az, max_az, n_az)
    cos_el = np.cos(np.radians(el))
    sin_el = np.sin(np.radians(el))
    trans = _transmission_np(grid, az)
    cos_alpha = (
        sin_be * sin_el[:, None]
        + cos_be * cos_el[:, None] * np.cos(np.radians(az[None, :] - sw.bulk_azimuth))
    )

    if not az_outer:
        result = 0.0
        for i_el in range(n_el):
            sp_lo = grid.central_speed * float(np.polyval(poly_min, el[i_el]))
            sp_hi = grid.central_speed * float(np.polyval(poly_max, el[i_el]))
            sp_lo, sp_hi = _dyn(sw.bulk_speed, sw.thermal_speed * 5, sp_lo, sp_hi)
            if sp_hi <= sp_lo:
                continue
            sp = np.linspace(sp_lo, sp_hi, n_sp)
            sp_w = _trapz_w(sp_lo, sp_hi, n_sp)
            pb_sp3 = _passband_np(grid, is_sg, el[i_el:i_el+1], sp)[0] * sp**3 / passband_norm
            exp_vals = np.exp(
                -(sp[None,:]**2 + sw.bulk_speed**2 - 2*sp[None,:]*sw.bulk_speed*cos_alpha[i_el,:,None])
                / (2*sw.thermal_speed**2)
            )
            result += el_w[i_el] * cos_el[i_el] * np.dot(az_w * trans, (pb_sp3 * exp_vals) @ sp_w)
    else:
        result = 0.0
        for i_az in range(n_az):
            el_sp_integral = 0.0
            for i_el in range(n_el):
                sp_lo = grid.central_speed * float(np.polyval(poly_min, el[i_el]))
                sp_hi = grid.central_speed * float(np.polyval(poly_max, el[i_el]))
                sp_lo, sp_hi = _dyn(sw.bulk_speed, sw.thermal_speed * 5, sp_lo, sp_hi)
                if sp_hi <= sp_lo:
                    continue
                sp = np.linspace(sp_lo, sp_hi, n_sp)
                sp_w = _trapz_w(sp_lo, sp_hi, n_sp)
                pb_sp3 = _passband_np(grid, is_sg, el[i_el:i_el+1], sp)[0] * sp**3 / passband_norm
                exp_vals = np.exp(
                    -(sp**2 + sw.bulk_speed**2 - 2*sp*sw.bulk_speed*cos_alpha[i_el, i_az])
                    / (2*sw.thermal_speed**2)
                )
                el_sp_integral += el_w[i_el] * cos_el[i_el] * float(pb_sp3 @ (exp_vals * sp_w))
            result += az_w[i_az] * trans[i_az] * el_sp_integral

    return result * (
        grid.central_effective_area
        * sw.density
        * (np.sqrt(2 * np.pi) * sw.thermal_speed) ** -3
        * 1e5
        * (math.pi / 180) ** 2
    )


def _integral(grid, sw: SWParams, n_el: int, n_az_sg: int, n_az_oa: int, n_sp: int,
              epsilon_oa: float = 1e-3, az_outer: bool = False) -> float:
    """
    Full triple integral summed over all three azimuth regions.
    az_outer=True uses azimuth as the outermost loop (reference for azimuth N sweeps).
    """
    total = 0.0
    for region in (0, -1, 1):
        n_az = n_az_sg if region == 0 else n_az_oa
        total += _region_integral(grid, sw, region, n_el, n_az, n_sp, epsilon_oa, az_outer)
    return total


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

_DIM_IDX = {"elevation": 0, "azimuth_sg": 1, "azimuth_oa": 2, "speed": 3}


def _ref_n_angular(grid, sw: SWParams, epsilon_oa: float = 1e-3) -> tuple[int, int, int]:
    """
    Returns (n_el_ref, n_az_sg_ref, n_az_oa_ref) using _ANG_SPACING_REF degree spacing
    over the actual dynamic angular limits for this test case.
    """
    n_el_ref = 2
    n_az_sg_ref = 2
    n_az_oa_ref = 2
    for region in (0, -1, 1):
        min_el, max_el, min_az, max_az = _angular_limits(sw, region, grid, epsilon_oa=epsilon_oa)
        n_el = max(2, round((max_el - min_el) / _ANG_SPACING_REF) + 1)
        n_az = max(2, round((max_az - min_az) / _ANG_SPACING_REF) + 1)
        n_el_ref = max(n_el_ref, n_el)
        if region == 0:
            n_az_sg_ref = n_az
        else:
            n_az_oa_ref = max(n_az_oa_ref, n_az)
    return n_el_ref, n_az_sg_ref, n_az_oa_ref


def _sweep(grid, sw: SWParams, dim: str, epsilon_oa: float = 1e-3) -> np.ndarray:
    """
    Returns relative errors (fraction) for each N in _N_SWEEP.
    The varying dimension takes each sweep value; the other three are held at
    their reference values (0.1-degree angular spacing, N=_N_SP_REF for speed).
    Azimuth sweeps use az_outer=True for the reference so the reference integrand
    order matches the dimension being assessed.
    """
    az_dim = dim in ("azimuth_sg", "azimuth_oa")
    n_el_ref, n_az_sg_ref, n_az_oa_ref = _ref_n_angular(grid, sw, epsilon_oa=epsilon_oa)
    ns_ref = [n_el_ref, n_az_sg_ref, n_az_oa_ref, _N_SP_REF]
    ref = _integral(grid, sw, *ns_ref, epsilon_oa=epsilon_oa, az_outer=az_dim)
    errors = np.empty(len(_N_SWEEP))
    idx = _DIM_IDX[dim]
    for k, n in enumerate(_N_SWEEP):
        ns = list(ns_ref)
        ns[idx] = n
        result = _integral(grid, sw, *ns, epsilon_oa=epsilon_oa, az_outer=az_dim)
        errors[k] = abs(result - ref) / max(abs(ref), 1e-30)
    return errors


_EPSILON_OA_VALUES = [1e-6, 1e-5, 1e-4, 1e-3]


def _epsilon_oa_sweep(grid, sw: SWParams) -> tuple[np.ndarray, np.ndarray]:
    """
    For each epsilon_OA value: returns
      - relative change in integral value vs epsilon_OA=1e-6 (at reference N)
      - OA azimuth N-sweep errors (fraction) against the 0.1-degree reference for that epsilon
    """
    # High-N baseline at epsilon_OA=1e-6
    n_el_ref, n_az_sg_ref, n_az_oa_ref = _ref_n_angular(grid, sw, epsilon_oa=1e-6)
    baseline = _integral(grid, sw, n_el_ref, n_az_sg_ref, n_az_oa_ref, _N_SP_REF, epsilon_oa=1e-6)

    value_changes = np.empty(len(_EPSILON_OA_VALUES))
    n_sweep_errors = np.empty((len(_EPSILON_OA_VALUES), len(_N_SWEEP)))

    for i, eps in enumerate(_EPSILON_OA_VALUES):
        n_el_r, n_az_sg_r, n_az_oa_r = _ref_n_angular(grid, sw, epsilon_oa=eps)
        ref = _integral(grid, sw, n_el_r, n_az_sg_r, n_az_oa_r, _N_SP_REF, epsilon_oa=eps, az_outer=True)
        value_changes[i] = abs(ref - baseline) / max(abs(baseline), 1e-30)
        n_sweep_errors[i] = _sweep(grid, sw, "azimuth_oa", epsilon_oa=eps)

    return value_changes, n_sweep_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading calibration data...")
    swapi_response = SWAPIResponse.from_files(
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-azimuthal-transmission_20250101_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-central-effective-area_20250101_v001.csv",
        _INSTRUMENT_DATA / "imap_swapi_proton-sw-passband-fit-coefficients_20250101_v001.csv",
    )

    dims = ["elevation", "azimuth_sg", "azimuth_oa", "speed"]
    dim_labels = ["Elevation", "Azimuth (SG region)", "Azimuth (OA region)", "Speed"]

    n_cases = len(_TEST_CASES)
    n_dims = len(dims)
    n_n = len(_N_SWEEP)
    all_errors = {dim: np.empty((n_cases, n_n)) for dim in dims}
    case_labels = []

    total_evals = n_cases * n_dims * (n_n + 1)  # +1 for the ref per sweep
    print(
        f"Running {n_cases} test cases × {n_dims} dimensions × {n_n} N values "
        f"= {total_evals} integral evaluations "
        f"(reference: {_ANG_SPACING_REF}° angular spacing, N_speed={_N_SP_REF})..."
    )

    for i, (label, bulk_speed, temperature_ev, bulk_azimuth, bulk_elevation) in enumerate(_TEST_CASES):
        grid = swapi_response.create_passband_grid(_peak_voltage(bulk_speed))
        sw = _make_sw_params(bulk_speed, temperature_ev, bulk_azimuth, bulk_elevation)
        case_labels.append(label)
        for dim in dims:
            print(f"  [{i + 1}/{n_cases}] {label:15s}  dim={dim}", flush=True)
            all_errors[dim][i] = _sweep(grid, sw, dim)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flat
    colors = plt.cm.tab10(np.linspace(0, 1, n_cases))

    for ax, dim, dim_label in zip(axes, dims, dim_labels):
        errors_pct = all_errors[dim] * 100  # (n_cases, n_n)

        for i, label in enumerate(case_labels):
            ax.semilogy(_N_SWEEP, errors_pct[i], color=colors[i], linewidth=1.2, label=label)

        # Max across test cases
        ax.semilogy(_N_SWEEP, errors_pct.max(axis=0), color="black", linewidth=2.0,
                    linestyle="-", label="worst case", zorder=5)

        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.0, label="1% threshold")
        ax.axvline(_CURRENT_N[dim], color="gray", linestyle=":", linewidth=1.2,
                   label=f"current N={_CURRENT_N[dim]}")

        ax.set_title(dim_label)
        ax.set_ylabel("Relative error (%)")
        ax.set_xlim(_N_SWEEP[0], _N_SWEEP[-1])
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes[2:]:
        ax.set_xlabel("N (quadrature points)")

    # Legend below the figure
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.suptitle(
        f"Integration accuracy vs N quadrature points\n"
        f"(reference: {_ANG_SPACING_REF}° angular spacing, N_speed={_N_SP_REF})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    out = _OUTPUT_DIR / "integration_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    # -----------------------------------------------------------------------
    # Epsilon-OA sweep figure
    # -----------------------------------------------------------------------
    print("\nRunning epsilon_OA sweep...")
    eps_value_changes = np.empty((n_cases, len(_EPSILON_OA_VALUES)))
    eps_n_errors = np.empty((n_cases, len(_EPSILON_OA_VALUES), len(_N_SWEEP)))

    for i, (label, bulk_speed, temperature_ev, bulk_azimuth, bulk_elevation) in enumerate(_TEST_CASES):
        print(f"  [{i + 1}/{n_cases}] {label}", flush=True)
        grid = swapi_response.create_passband_grid(_peak_voltage(bulk_speed))
        sw = _make_sw_params(bulk_speed, temperature_ev, bulk_azimuth, bulk_elevation)
        eps_value_changes[i], eps_n_errors[i] = _epsilon_oa_sweep(grid, sw)

    fig2, (ax_val, ax_n) = plt.subplots(1, 2, figsize=(11, 5))
    eps_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(_EPSILON_OA_VALUES)))
    case_colors = plt.cm.tab10(np.linspace(0, 1, n_cases))

    # Left: integral value change vs epsilon_OA for each test case
    for i, label in enumerate(case_labels):
        ax_val.semilogy(_EPSILON_OA_VALUES, eps_value_changes[i] * 100,
                        color=case_colors[i], marker="o", linewidth=1.2, label=label)
    ax_val.axhline(1.0, color="crimson", linestyle="--", linewidth=1.0, label="1% threshold")
    ax_val.set_xscale("log")
    ax_val.set_xlabel("epsilon_OA")
    ax_val.set_ylabel("Integral change vs epsilon=1e-6 (%)")
    ax_val.set_title("Signal loss from raising epsilon_OA")
    ax_val.grid(True, which="both", alpha=0.3)
    ax_val.legend(fontsize=8)

    # Right: OA azimuth N sweep worst-case error for each epsilon_OA
    for j, eps in enumerate(_EPSILON_OA_VALUES):
        worst = eps_n_errors[:, j, :].max(axis=0) * 100
        ax_n.semilogy(_N_SWEEP, worst, color=eps_colors[j], linewidth=1.5,
                      label=f"eps={eps:.0e}")
    ax_n.axhline(1.0, color="crimson", linestyle="--", linewidth=1.0, label="1% threshold")
    ax_n.axvline(_CURRENT_N["azimuth_oa"], color="gray", linestyle=":", linewidth=1.2,
                 label=f"current N={_CURRENT_N['azimuth_oa']}")
    ax_n.set_xlabel("N (OA azimuth quadrature points)")
    ax_n.set_ylabel("Worst-case relative error (%)")
    ax_n.set_title("OA azimuth accuracy vs N, by epsilon_OA")
    ax_n.grid(True, which="both", alpha=0.3)
    ax_n.legend(fontsize=8)

    fig2.suptitle("Effect of epsilon_OA on integral value and required N", fontsize=11)
    fig2.tight_layout()

    out2 = _OUTPUT_DIR / "integration_benchmark_epsilon_oa.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    # -----------------------------------------------------------------------
    # Integrand shape comparison (typical case: 450 km/s, 10 eV, az=15°, el=-5°)
    # -----------------------------------------------------------------------
    print("\nPlotting integrand shapes...")
    grid_typ = swapi_response.create_passband_grid(_peak_voltage(450.0))
    sw_typ = _make_sw_params(450, 10, 15, -5)
    min_el, max_el, min_az, max_az = _angular_limits(sw_typ, 0, grid_typ)  # SG region

    N_PLOT = 300  # dense points for smooth curves

    # Elevation marginal: f_el(el) = cos(el) * integral_{az,sp} T(az)*pb*exp
    # For each elevation, integrate over az and sp at reference resolution
    el_pts = np.linspace(min_el, max_el, N_PLOT)
    n_el_r, n_az_sg_r, _ = _ref_n_angular(grid_typ, sw_typ)
    az_ref = np.linspace(min_az, max_az, n_az_sg_r)
    trans_ref = _transmission_np(grid_typ, az_ref)
    az_w_ref = _trapz_w(min_az, max_az, n_az_sg_r)
    sin_be = math.sin(math.radians(sw_typ.bulk_elevation))
    cos_be = math.cos(math.radians(sw_typ.bulk_elevation))

    f_el = np.empty(N_PLOT)
    for k, e in enumerate(el_pts):
        cos_e = math.cos(math.radians(e))
        sin_e = math.sin(math.radians(e))
        sp_lo = grid_typ.central_speed * float(np.polyval(grid_typ.min_SG_poly, e))
        sp_hi = grid_typ.central_speed * float(np.polyval(grid_typ.max_SG_poly, e))
        sp_lo, sp_hi = _dyn(sw_typ.bulk_speed, sw_typ.thermal_speed * 5, sp_lo, sp_hi)
        sp = np.linspace(sp_lo, sp_hi, _N_SP_REF)
        sp_w = _trapz_w(sp_lo, sp_hi, _N_SP_REF)
        pb_norm = float(_passband_np(grid_typ, True, np.array([0.0]), np.array([grid_typ.central_speed]))[0, 0])
        pb_sp3 = _passband_np(grid_typ, True, np.array([e]), sp)[0] * sp**3 / pb_norm
        cos_alpha = sin_be * sin_e + cos_be * cos_e * np.cos(np.radians(az_ref - sw_typ.bulk_azimuth))
        exp_vals = np.exp(
            -(sp[None,:]**2 + sw_typ.bulk_speed**2 - 2*sp[None,:]*sw_typ.bulk_speed*cos_alpha[:,None])
            / (2*sw_typ.thermal_speed**2)
        )
        sp_int = (pb_sp3 * exp_vals) @ sp_w  # (n_az,)
        f_el[k] = cos_e * np.dot(az_w_ref * trans_ref, sp_int)

    # Azimuth marginal (az_outer): f_az(az) = T(az) * integral_{el,sp} cos(el)*pb*exp
    az_pts = np.linspace(min_az, max_az, N_PLOT)
    el_ref = np.linspace(min_el, max_el, n_el_r)
    el_w_ref = _trapz_w(min_el, max_el, n_el_r)
    cos_el_ref = np.cos(np.radians(el_ref))
    sin_el_ref = np.sin(np.radians(el_ref))
    trans_pts = _transmission_np(grid_typ, az_pts)

    f_az = np.empty(N_PLOT)
    for k, a in enumerate(az_pts):
        el_sp_int = 0.0
        cos_alpha_col = sin_be * sin_el_ref + cos_be * cos_el_ref * math.cos(math.radians(a - sw_typ.bulk_azimuth))
        for j, e in enumerate(el_ref):
            sp_lo = grid_typ.central_speed * float(np.polyval(grid_typ.min_SG_poly, e))
            sp_hi = grid_typ.central_speed * float(np.polyval(grid_typ.max_SG_poly, e))
            sp_lo, sp_hi = _dyn(sw_typ.bulk_speed, sw_typ.thermal_speed * 5, sp_lo, sp_hi)
            sp = np.linspace(sp_lo, sp_hi, _N_SP_REF)
            sp_w = _trapz_w(sp_lo, sp_hi, _N_SP_REF)
            pb_norm = float(_passband_np(grid_typ, True, np.array([0.0]), np.array([grid_typ.central_speed]))[0, 0])
            pb_sp3 = _passband_np(grid_typ, True, np.array([e]), sp)[0] * sp**3 / pb_norm
            exp_vals = np.exp(
                -(sp**2 + sw_typ.bulk_speed**2 - 2*sp*sw_typ.bulk_speed*cos_alpha_col[j])
                / (2*sw_typ.thermal_speed**2)
            )
            el_sp_int += el_w_ref[j] * cos_el_ref[j] * float(pb_sp3 @ (exp_vals * sp_w))
        f_az[k] = trans_pts[k] * el_sp_int

    # Normalize both to their max for shape comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    axes3[0].plot(el_pts, f_el / np.max(np.abs(f_el)))
    axes3[0].set_xlabel("elevation (deg)")
    axes3[0].set_ylabel("normalized integrand")
    axes3[0].set_title("Elevation marginal integrand\n(inner: az+sp at ref N)")
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(az_pts, f_az / np.max(np.abs(f_az)))
    axes3[1].set_xlabel("azimuth (deg)")
    axes3[1].set_ylabel("normalized integrand")
    axes3[1].set_title("Azimuth marginal integrand\n(inner: el+sp at ref N)")
    axes3[1].grid(True, alpha=0.3)

    fig3.suptitle("Typical case: 450 km/s, 10 eV, az=15°, el=−5° (SG region)", fontsize=11)
    fig3.tight_layout()
    out3 = _OUTPUT_DIR / "integrand_shapes.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()
