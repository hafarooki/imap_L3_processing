#!/usr/bin/env python3
"""
Plot one flight chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production forward model, above per-sweep spectrograms of the observed and
modeled rates over the PUI fit energy window.

Runs the production helium PUI fit over the whole day via
`scripts/swapi/fit_and_plot_pui.py`, then renders the single chunk nearest
15:54:05 UT via `scripts/swapi/view_one_pui_spectrum.py`.

Requires the environment variable IMAP_API_KEY to be set (the fit downloads its
L2 and SPICE inputs from the SDC).

TODO: refactor this into a self-contained figure_src script, the way
plot_alpha_peak_finding.py resolves and downloads its own inputs, instead of
driving two scripts/swapi/ entry points over a /tmp pickle handoff. Doing that
would also let the figure be built and saved here through
`figure_utils.save_figure`, and let the chunk be pinned to an exact
`sci_start_time` via `figure_utils.find_sweep_start_index` rather than to
whichever chunk centre happens to be nearest the requested time. Two things
have to be sorted out first:
  1. view_one_pui_spectrum.py reads the whole-day fit and spectrogram pickles
     that fit_and_plot_pui.py leaves in /tmp; only the one selected chunk is
     actually needed, so the fit could be run for that chunk alone.
  2. Both scripts do all their work at module level under argparse, so neither
     can be imported and called directly.

Output: docs/swapi/figures/pui_flight_model_comparison.png
Usage:  uv run python docs/swapi/figure_src/plot_pui_flight_model_comparison.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from figure_utils import DEFAULT_PNG_PALETTE_COLORS, FIGURES_DIR, require_imap_api_key

_DATE = "2026-01-01"
_CHUNK_TIME = "15:54:05"
_OUTPUT_PATH = FIGURES_DIR / "pui_flight_model_comparison.png"

_SCRIPTS_DIR = REPO_ROOT / "scripts" / "swapi"
_FIT_SCRIPT = _SCRIPTS_DIR / "fit_and_plot_pui.py"
_VIEW_SCRIPT = _SCRIPTS_DIR / "view_one_pui_spectrum.py"


def _run(command: list[str]) -> None:
    # Claude: both scripts import from the `scripts.swapi` package, which is not installed into the venv, so the repo root has to be on the child's path.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("MPLBACKEND", "Agg")

    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    if result.returncode != 0:
        sys.exit(f"{Path(command[1]).name} failed with exit code {result.returncode}")


def _quantize_in_place(png_path: Path) -> None:
    """Reduce `png_path` to an indexed palette, as `figure_utils.save_figure` does.

    view_one_pui_spectrum.py writes the file with a plain `figure.savefig`, so
    the palette reduction that every other figure gets has to be applied after
    the fact until the plotting moves into this script.
    """
    with Image.open(png_path) as rendered:
        # Claude: quantize() needs RGB, and the flattened figure is opaque.
        opaque = rendered.convert("RGB")
        # Claude: median cut spends the palette on the spectrogram gradients and
        # Claude: shifts the small red and orange markers; octree keeps them.
        quantized = opaque.quantize(
            colors=DEFAULT_PNG_PALETTE_COLORS, method=Image.Quantize.FASTOCTREE
        )
    quantized.save(png_path, optimize=True)


def main() -> None:
    require_imap_api_key()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Claude: fit_and_plot_pui.py calls plt.show() and blocks unless --output-dir is set; its own PNG is a byproduct we do not keep.
    with tempfile.TemporaryDirectory() as scratch_dir:
        _run([sys.executable, str(_FIT_SCRIPT), _DATE, "--output-dir", scratch_dir])

    _run(
        [
            sys.executable,
            str(_VIEW_SCRIPT),
            _DATE,
            _CHUNK_TIME,
            "--output-path",
            str(_OUTPUT_PATH),
        ]
    )

    # Claude: view_one_pui_spectrum.py already printed the "Saved" line for this path.
    _quantize_in_place(_OUTPUT_PATH)
    print(f"Reduced to an indexed palette ({_OUTPUT_PATH.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
