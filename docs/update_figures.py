#!/usr/bin/env python
"""Regenerate the documentation figures sequentially, in one interpreter.

    uv run docs/update_figures.py               # every figure
    uv run docs/update_figures.py spectra pui   # only scripts matching a substring
    uv run docs/update_figures.py --list

Running every figure in a single process means numba compiles the shared
forward-model kernels once rather than once per script, which is most of the
fixed cost of a full rebuild. Shows a live timer per script, then reports how
long each one took, which figures changed, and which scripts failed. Exits
non-zero if any script failed.
"""

import argparse
import contextlib
import hashlib
import os
import runpy
import sys
import threading
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Claude: set before matplotlib is imported anywhere, including by a figure.
os.environ.setdefault("MPLBACKEND", "Agg")
# Claude: the run is sequential, so each script's own worker pool is free to
# use the whole machine. figure_utils reads this.
os.environ.setdefault("FIGURE_WORKER_PROCESSES", str(os.cpu_count() or 1))


def discover_scripts() -> list[Path]:
    return sorted(REPO_ROOT.glob("docs/*/figure_src/plot_*.py"))


def figure_files() -> list[Path]:
    return sorted(path for path in REPO_ROOT.glob("docs/*/figures/*") if path.is_file())


def hash_figures() -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in figure_files()
    }


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"
    return f"{seconds:.1f}s"


class LiveTimer:
    """Redraw `running <label> <elapsed>` in place until the script finishes."""

    def __init__(self, label: str, enabled: bool):
        self.label = label
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LiveTimer":
        if self.enabled:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def _tick(self) -> None:
        start = time.monotonic()
        while not self._stop.wait(1.0):
            elapsed = format_duration(time.monotonic() - start)
            sys.stdout.write(f"\r  running {self.label:<44} {elapsed:>8}")
            sys.stdout.flush()

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stdout.write("\r" + " " * 66 + "\r")
            sys.stdout.flush()


def reset_shared_state() -> None:
    """Undo the global state a figure script leaves behind.

    All figures share this interpreter, so pyplot's figure registry and the
    furnished SPICE kernel pool have to be cleared between scripts -- notably
    because a script's kernel-furnishing guard checks whether any kernels are
    already loaded and skips its own setup if so.
    """
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
    if "spiceypy" in sys.modules:
        with contextlib.suppress(Exception):
            sys.modules["spiceypy"].kclear()


def purge_modules_from(script_dir: Path) -> None:
    """Forget modules imported from `script_dir`.

    Each `figure_src` directory has its own `figure_utils`; leaving the first
    one in `sys.modules` would silently hand it to a script from a different
    directory. Modules from the installed package stay cached, which is what
    keeps numba from recompiling.
    """
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).parent == script_dir:
            del sys.modules[name]


def run_figure_script(path: Path, log_path: Path) -> str | None:
    """Execute one figure script; return a traceback string, or None on success."""
    import matplotlib

    # Claude: `python script.py` puts the script's directory on sys.path but
    # Claude: runpy does not, and the figure scripts import figure_utils from there.
    script_dir = path.parent
    sys.path.insert(0, str(script_dir))
    # Claude: runpy leaves sys.argv alone, so a script with its own argparse
    # Claude: would otherwise see this driver's filter arguments and bail out.
    driver_argv = sys.argv
    sys.argv = [str(path)]
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            # Claude: run_path gives the script its own module globals, so
            # module-level worker state cannot leak between figures, and
            # rc_context keeps a script's style tweaks from doing the same.
            try:
                with matplotlib.rc_context():
                    runpy.run_path(str(path), run_name="__main__")
            except SystemExit as exit_request:
                if exit_request.code not in (None, 0):
                    return f"SystemExit: {exit_request.code}"
            except BaseException:
                return traceback.format_exc()
            finally:
                reset_shared_state()
                purge_modules_from(script_dir)
                sys.argv = driver_argv
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(script_dir))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the documentation figures sequentially."
    )
    parser.add_argument(
        "filters",
        nargs="*",
        help="only run scripts whose filename contains one of these substrings",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the figure scripts and exit"
    )
    arguments = parser.parse_args()

    scripts = discover_scripts()
    if arguments.list:
        for path in scripts:
            print(f"  {path.relative_to(REPO_ROOT)}")
        return 0

    if arguments.filters:
        scripts = [
            path
            for path in scripts
            if any(needle in path.name for needle in arguments.filters)
        ]
        if not scripts:
            print(f"No figure script matches {arguments.filters}", file=sys.stderr)
            return 1

    # Claude: several figures are built from SDC data. Without the key they fail
    # Claude: deep inside a download, so refuse before running anything.
    if not os.environ.get("IMAP_API_KEY"):
        print(
            "IMAP_API_KEY is not set; the SDC-backed figures cannot be built.",
            file=sys.stderr,
        )
        return 1

    # Claude: keep the progress lines streaming when stdout is a pipe.
    sys.stdout.reconfigure(line_buffering=True)

    log_dir = REPO_ROOT / ".figure_logs"
    log_dir.mkdir(exist_ok=True)
    interactive = sys.stdout.isatty()

    before = hash_figures()
    run_start = time.monotonic()
    durations: list[tuple[float, str]] = []
    failures: list[tuple[str, str, Path]] = []

    print(
        f"Regenerating {len(scripts)} figure(s) sequentially "
        f"on {os.environ['FIGURE_WORKER_PROCESSES']} core(s)..."
    )
    for path in scripts:
        log_path = log_dir / f"{path.stem}.log"
        start = time.monotonic()
        with LiveTimer(path.name, interactive):
            failure = run_figure_script(path, log_path)
        elapsed = time.monotonic() - start
        durations.append((elapsed, path.name))
        status = "ok     " if failure is None else "FAILED "
        print(f"  {status} {path.name:<44} {format_duration(elapsed):>8}")
        if failure is not None:
            failures.append((path.name, failure, log_path))

    run_elapsed = time.monotonic() - run_start
    after = hash_figures()
    updated = [
        f"{path.relative_to(REPO_ROOT)}" + ("  (new)" if path not in before else "")
        for path, digest in after.items()
        if before.get(path) != digest
    ]

    print("\nSlowest first:")
    for elapsed, name in sorted(durations, reverse=True):
        print(f"  {name:<44} {format_duration(elapsed):>8}")

    print(f"\nUpdated ({len(updated)}):")
    for line in sorted(updated):
        print(f"  {line}")

    if failures:
        print(f"\nFailed ({len(failures)}):")
        for name, failure, log_path in failures:
            print(f"  {name}")
            for line in failure.strip().splitlines()[-12:]:
                print(f"    | {line}")
            print(f"    | (full output: {log_path.relative_to(REPO_ROOT)})")

    print(f"\nUnchanged: {len(after) - len(updated)} figure(s) already up to date.")
    print(f"Total: {format_duration(run_elapsed)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
