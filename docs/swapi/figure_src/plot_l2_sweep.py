#!/usr/bin/env python3
"""
Plot a single raw L2 ESA sweep from a real SWAPI CDF, unmodified.

Output: docs/swapi/figures/l2_sweep.png
Usage:  python docs/swapi/figure_src/plot_l2_sweep.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"

SWAPI_L2_K_FACTOR = 1.93

# imap_swapi_l2_sci_20260204_v002.cdf, sweep 0 (2026-02-03T23:59:08Z)
ESA_ENERGY = np.array([
    1.97632000e+04, 1.97632000e+04, 1.81225898e+04, 1.66181723e+04,
    1.52386415e+04, 1.39736303e+04, 1.28136319e+04, 1.17499289e+04,
    1.07745275e+04, 9.88009751e+03, 9.05991715e+03, 8.30782275e+03,
    7.61816225e+03, 6.98575280e+03, 6.40584180e+03, 5.87407118e+03,
    5.38644464e+03, 4.93929762e+03, 4.52926980e+03, 4.15327977e+03,
    3.80850196e+03, 3.49234531e+03, 3.20243390e+03, 2.93658902e+03,
    2.69281281e+03, 2.46927330e+03, 2.26429056e+03, 2.07632414e+03,
    1.90396144e+03, 1.74590715e+03, 1.60097348e+03, 1.46807125e+03,
    1.34620169e+03, 1.23444893e+03, 1.13197314e+03, 1.03800422e+03,
    9.51835971e+02, 8.72820843e+02, 8.00365029e+02, 7.33924018e+02,
    6.72998499e+02, 6.17130615e+02, 5.65900512e+02, 5.18923194e+02,
    4.75845622e+02, 4.36344066e+02, 4.00121667e+02, 3.66906212e+02,
    3.36448085e+02, 3.08518390e+02, 2.82907234e+02, 2.59422146e+02,
    2.37886635e+02, 2.18138860e+02, 2.00030415e+02, 1.83425212e+02,
    1.68198464e+02, 1.54235740e+02, 1.41432109e+02, 1.29691351e+02,
    1.18925232e+02, 1.09052846e+02, 1.00000000e+02, 6.75500000e+01,
    3.86000000e+01, 9.65000000e+00, 5.90960685e+02, 5.41903037e+02,
    4.96917831e+02, 4.55666999e+02, 4.17840538e+02, 3.83154180e+02,
])

COIN_RATE = np.array([
    -9.9999998e+30, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
     0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
     0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
     0.0000000e+00, 0.0000000e+00, 6.8965521e+00, 0.0000000e+00,
     6.8965521e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
     6.8965521e+00, 2.0689655e+01, 2.0689655e+01, 4.1379311e+01,
     3.4482761e+01, 2.0689655e+01, 1.3793104e+01, 3.4482761e+01,
     2.0689655e+01, 7.5862068e+01, 5.5172417e+01, 4.1379311e+01,
     8.9655174e+01, 1.1034483e+02, 2.8275864e+02, 4.0689658e+02,
     7.5172418e+02, 7.7931036e+02, 6.7586206e+02, 7.7931036e+02,
     1.2827587e+03, 2.5793103e+03, 6.2000000e+03, 1.5289655e+04,
     2.4558621e+04, 2.0965518e+04, 1.0772414e+04, 3.4758621e+03,
     7.7931036e+02, 1.9310345e+02, 7.5862068e+01, 6.2068966e+01,
     5.5172417e+01, 2.7586208e+01, 6.8965521e+00, 2.0689655e+01,
     2.7586208e+01, 6.8965521e+00, 5.5172417e+01, 2.0689655e+01,
     4.1379311e+01, 2.0689655e+01, 1.3793104e+01, 0.0000000e+00,
     0.0000000e+00, 0.0000000e+00, 2.9241379e+03, 5.6896553e+03,
     9.5172412e+03, 9.4275869e+03, 5.8344829e+03, 2.1034482e+03,
], dtype=np.float32)


def main():
    esa = ESA_ENERGY
    coin = COIN_RATE

    bins = np.arange(72)
    science_mask = (bins >= 1) & (esa > 0)

    bin_nums = bins[science_mask]
    coin_sci = coin[science_mask]
    esa_sci = esa[science_mask]
    voltage = esa_sci / SWAPI_L2_K_FACTOR

    # Sort by increasing energy (= increasing |V| = decreasing bin index)
    plot_order = np.argsort(esa_sci)
    bin_nums_sorted = bin_nums[plot_order]
    coin_sorted = coin_sci[plot_order]
    voltage_sorted = voltage[plot_order]

    is_fine = bin_nums_sorted >= 63
    is_coarse = bin_nums_sorted < 63

    fig, (ax, ax_b) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    y_floor = 0.5
    pos_coarse = is_coarse & (coin_sorted > 0)
    zero_coarse = is_coarse & (coin_sorted == 0)
    pos_fine = is_fine & (coin_sorted > 0)
    zero_fine = is_fine & (coin_sorted == 0)

    # Top subplot: count rate vs ESA voltage
    ax.semilogy(
        voltage_sorted[pos_coarse], coin_sorted[pos_coarse],
        "o", color="C0", markersize=4, label="Coarse sweep (bins 1–62)",
    )
    ax.semilogy(
        voltage_sorted[zero_coarse], np.full(zero_coarse.sum(), y_floor),
        "o", color="C0", markersize=4, fillstyle="none",
    )
    ax.semilogy(
        voltage_sorted[pos_fine], coin_sorted[pos_fine],
        "s", color="C1", markersize=5, label="Fine sweep (bins 63–71)",
    )
    ax.semilogy(
        voltage_sorted[zero_fine], np.full(zero_fine.sum(), y_floor),
        "s", color="C1", markersize=5, fillstyle="none",
    )

    ax.set_ylabel("Coincidence count rate (Hz)")
    ax.set_title("SWAPI L2 raw sweep (imap_swapi_l2_sci_20260204, sweep 0)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="major")
    ax.set_ylim(bottom=y_floor)

    # Lower subplot: bin index vs ESA voltage
    ax_b.plot(voltage_sorted[is_coarse], bin_nums_sorted[is_coarse], "o", color="C0", markersize=4)
    ax_b.plot(voltage_sorted[is_fine],   bin_nums_sorted[is_fine],   "s", color="C1", markersize=5)
    ax_b.set_ylabel("Bin index")
    ax_b.grid(True, alpha=0.3, which="major")
    ax_b.set_xlabel("ESA voltage |V| (V)")

    # Shared log x-axis
    ax.set_xscale("log")
    ax_b.set_xscale("log")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / "l2_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Saved {_OUTPUT_DIR / 'l2_sweep.png'}")


if __name__ == "__main__":
    main()
