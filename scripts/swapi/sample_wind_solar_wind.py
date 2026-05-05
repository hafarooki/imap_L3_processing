#!/usr/bin/env python3
"""
Sample real solar wind conditions from WIND/SWE 2-min ASCII data.

Downloads the WIND/SWE 2-min "all_orbit_phases"-excluded ASCII file for a given
year (default 2025), filters for high-quality bimaxwellian fits (fit_flag == 10
unless --keep-flags is given), drops rows with fill values in the quantities we
care about, randomly samples N rows, and writes a CSV usable as ground-truth
solar wind parameters for plot_fit_accuracy.py and similar scripts.

Source: https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/

Frame note: WIND publishes proton velocity in GSE. We add an approximate RTN
projection (v_R = -Vx_GSE, v_T = -Vy_GSE, v_N = Vz_GSE) which is accurate at
L1 to within the ~7° tilt between GSE-Z and the solar rotation axis — fine for
sampling realistic bulk_speed / vT / vN ranges.

Output columns (units in header row):
  year, fdoy, fit_flag,
  proton_bulk_speed_km_s, proton_thermal_speed_km_s, proton_density_cm3,
  proton_vx_gse_km_s, proton_vy_gse_km_s, proton_vz_gse_km_s,
  v_R_km_s, v_T_km_s, v_N_km_s, proton_temperature_K

Usage:
  conda run -n imapL3 python scripts/swapi/sample_wind_solar_wind.py
  conda run -n imapL3 python scripts/swapi/sample_wind_solar_wind.py \
      --year 2025 --n 1000 --seed 7 \
      --output docs/swapi/figure_src/wind_solar_wind_samples_2025.csv
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    PROTON_MASS_KG,
)

_BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imap_l3" / "wind_swe_2m"
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "docs" / "swapi" / "figure_src" / "wind_solar_wind_samples_2025.csv"
)

# Column indices (0-based) into the WIND ASCII record. See documentation_jck.txt
# at the source URL for the full layout.
_COL_YEAR = 0
_COL_FDOY = 1
_COL_FIT_FLAG = 2
_COL_VBULK = 3  # proton bulk speed [km/s]
_COL_VX_GSE = 5  # proton Vx GSE [km/s]
_COL_VY_GSE = 7
_COL_VZ_GSE = 9
_COL_W_SCALAR = 11  # proton scalar thermal speed [km/s]
_COL_NP = 21  # proton density [cm^-3]

# WIND fill values used in this dataset.
_FILL_F7 = 99999.9
_FILL_F9 = 99999.999


def _download(year: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"wind_swe_2m_sw{year}.asc"
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Using cached {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst
    url = f"{_BASE_URL}/{name}"
    print(f"Downloading {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        size = int(resp.headers.get("Content-Length", "0"))
        chunk = 1 << 20
        bytes_read = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            bytes_read += len(buf)
            if size:
                pct = 100.0 * bytes_read / size
                print(
                    f"  {bytes_read / 1e6:7.1f} / {size / 1e6:.1f} MB ({pct:5.1f}%)",
                    end="\r",
                )
        print()
    tmp.rename(dst)
    return dst


def _load(path: Path) -> np.ndarray:
    """Load the WIND 2-min ASCII file. Comment-prefixed lines start with ';'."""
    print(f"Parsing {path} ...")
    data = np.loadtxt(path, comments=";")
    print(f"  {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def _thermal_speed_to_temperature_K(w_km_s: np.ndarray) -> np.ndarray:
    """Convert scalar thermal speed (km/s) to temperature (K) for protons.

    Using w_th = sqrt(2 k T / m), so T = m w_th^2 / (2 k).
    """
    w_m_s = w_km_s * 1e3
    return PROTON_MASS_KG * w_m_s**2 / (2.0 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--n", type=int, default=1000, help="Number of samples to draw")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--keep-flags",
        type=int,
        nargs="+",
        default=[10],
        help="Fit flags to keep (default: 10 = ideal proton fits). "
        "See documentation_jck.txt at the source URL.",
    )
    p.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument(
        "--no-cache-download",
        action="store_true",
        help="Skip download even if the cached file is missing (errors instead).",
    )
    args = p.parse_args()

    src = args.cache_dir / f"wind_swe_2m_sw{args.year}.asc"
    if not src.exists():
        if args.no_cache_download:
            sys.exit(f"Missing cached file {src} and --no-cache-download was set.")
        src = _download(args.year, args.cache_dir)

    data = _load(src)

    fit_flag = data[:, _COL_FIT_FLAG].astype(int)
    keep = np.isin(fit_flag, np.asarray(args.keep_flags, dtype=int))
    print(f"  After fit-flag filter (flags={args.keep_flags}): {keep.sum()} rows")

    # Drop rows with fill values in the proton quantities we sample.
    cols_required = [
        _COL_VBULK,
        _COL_VX_GSE,
        _COL_VY_GSE,
        _COL_VZ_GSE,
        _COL_W_SCALAR,
        _COL_NP,
    ]
    finite = np.ones(data.shape[0], dtype=bool)
    for c in cols_required:
        col = data[:, c]
        # Both fill values appear in this dataset; treat anything ≥ 9.99e4 as fill.
        finite &= np.isfinite(col) & (np.abs(col) < 9e4)
    keep &= finite
    print(f"  After fill-value filter: {keep.sum()} rows")

    if keep.sum() < args.n:
        sys.exit(
            f"Only {keep.sum()} rows survive filters but {args.n} samples requested."
        )

    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(keep)
    chosen = rng.choice(candidates, size=args.n, replace=False)
    chosen.sort()
    sub = data[chosen]

    year = sub[:, _COL_YEAR].astype(int)
    fdoy = sub[:, _COL_FDOY]
    flag = sub[:, _COL_FIT_FLAG].astype(int)
    v_bulk = sub[:, _COL_VBULK]
    vx_gse = sub[:, _COL_VX_GSE]
    vy_gse = sub[:, _COL_VY_GSE]
    vz_gse = sub[:, _COL_VZ_GSE]
    w_scalar = sub[:, _COL_W_SCALAR]
    n_p = sub[:, _COL_NP]

    # GSE -> RTN approximation valid at L1 (within ~7° tilt between GSE-Z and
    # the solar rotation axis). Sufficient for sampling realistic SW ranges.
    v_R = -vx_gse
    v_T = -vy_gse
    v_N = vz_gse

    T_K = _thermal_speed_to_temperature_K(w_scalar)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "year,fdoy,fit_flag,"
        "proton_bulk_speed_km_s,proton_thermal_speed_km_s,proton_density_cm3,"
        "proton_vx_gse_km_s,proton_vy_gse_km_s,proton_vz_gse_km_s,"
        "v_R_km_s,v_T_km_s,v_N_km_s,proton_temperature_K"
    )
    out = np.column_stack(
        [
            year,
            fdoy,
            flag,
            v_bulk,
            w_scalar,
            n_p,
            vx_gse,
            vy_gse,
            vz_gse,
            v_R,
            v_T,
            v_N,
            T_K,
        ]
    )
    fmt = "%d,%.6f,%d,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e"
    np.savetxt(args.output, out, fmt=fmt, header=header, comments="")
    print(f"Wrote {args.n} samples -> {args.output}")

    print("\nSummary of sampled distributions:")
    for label, arr, unit in [
        ("bulk_speed", v_bulk, "km/s"),
        ("thermal_speed", w_scalar, "km/s"),
        ("temperature", T_K, "K"),
        ("density", n_p, "cm^-3"),
        ("v_R", v_R, "km/s"),
        ("v_T", v_T, "km/s"),
        ("v_N", v_N, "km/s"),
    ]:
        q = np.percentile(arr, [1, 50, 99])
        print(
            f"  {label:>13s} [{unit}]  p1={q[0]:9.3g}  median={q[1]:9.3g}  p99={q[2]:9.3g}"
        )


if __name__ == "__main__":
    main()
