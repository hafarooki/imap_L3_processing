# imap_L3_processing

## Environment

Run `source ~/imap_api_key.sh` before anything that reaches the SDC. It exports
`IMAP_API_KEY`, which `imap-data-access` needs to query and download L2 science
files, SPICE kernels, and ancillary files. Without it, `docs/update_figures.py`
refuses to start and the SDC-backed figure scripts exit early.

Use `uv run python ...` for everything. Note that a bare `uv sync` prunes the
dev and test extras plus `pint-xarray`; restore them with
`uv sync --extra dev --extra test && uv pip install pint-xarray`.

## SWAPI documentation figures

`uv run docs/update_figures.py` regenerates every figure under
`docs/*/figures/`, sequentially in one interpreter so numba compiles the shared
forward-model kernels once. Pass filename substrings to build a subset, or
`--list` to see the scripts. Per-script logs land in `.figure_logs/`.

Figures are PNGs written through `figure_utils.save_figure`, which renders at a
fixed DPI and reduces to an indexed palette so an unchanged figure is
byte-identical across runs. The driver reports which files actually changed.
