# IMAP L3 Data Processing

The level 3 processing code is a command-line program that will run inside of a docker container.
The code snippet below will build the docker image and run the container and remove it after execution is completed.

## Running the Processor using SDC Infrastructure

### Uploading an L2 cdf file to trigger the L3 pipeline:

There is a program named `imap-data-access` which is used to upload and download cdf files from the SDC.
Start by running the following commands in a terminal to create a virtual environment. If you've pulled the repository,
open the terminal from the imap_l3_processing folder.

- `python -m venv venv`
- `source venv/Scripts/activate`

Continue following the installation instructions from the IMAP Science Operations Center. Installation instructions are
found at: https://github.com/IMAP-Science-Operations-Center/imap-data-access

The SDC expects cdf files to follow a specific naming
convention: `imap_swapi_{data_level}_{descriptor}_{start_date}_{version}.cdf`
An example file would be called: `imap_swapi_l2_sci_20111231_v001.cdf`<br/>
Note: Currently the only acceptable descriptor is 'sci' <br/>
Note: File versions must be 3 characters long. (i.e. v003)

The command to upload a cdf file and trigger the pipeline
is : `imap-data-access upload imap_swapi_l2_sci_20111231_v001.cdf`.
The command to see the newly created L3a data files
is : `imap-data-access query --instrument swapi --data-level l3a --version v001`

You should see two L3a files:

* imap_swapi_l3a_proton-sw-fake-menlo-{GUID}_20111231_v001.cdf
* imap_swapi_l3a_alpha-sw-fake-menlo-{GUID}_20111231_v001.cdf

## Running the Processor Locally

### Setup

1. Install Docker Desktop
2. Start Docker Desktop
3. Pull this repository

### How to run the docker container from the command line

The arguments passed after the build command describe the inputs to the level 3 processing code.
The volume command-line argument mounts the local spice kernel folder to the docker container.
The remaining arguments match the inputs that we expect to receive from the SDC batch job run.

`docker run --rm --mount type=bind,src="$(pwd)/spice_kernels",dst="/mnt/spice" --mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" $(docker build -q -f Dockerfile_run_local .) $@`

Alternatively, running run_local_using_docker.sh in Git Bash will execute the above command for you:

`./run_local_using_docker.sh {instrument-name} {data-level}`

### Getting data from Dev data from SDC (Science Data Center)

We created a tool to retrieve the latest data from the SDC to assist with testing. The tool takes in command line
arguments, the instrument level, and the number of files to retrieve. For example:

`python fetch_latest_data.py --instrument swapi --level l3a --count 4`

This will copy the .cdf files into your repo folder under the data folder. 

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

### Setup

- `uv sync --extra test` installs the runtime dependencies plus unit-testing tools (`pytest`, `pytest-xdist`).
- `uv sync --extra dev` installs formatting/linting tools (`ruff`) plus other developer dependencies.
- `uv sync --extra test --extra dev` installs both.

### Test data (Git LFS)

Many test fixtures (`.dat`, `.bsp`, `.tls`, `.tsc`, `.tf` files — calibration tables and SPICE
kernels) are stored via [Git LFS](https://git-lfs.com/), not checked directly into git. Before
running the test suite for the first time:

- Install the `git-lfs` CLI (e.g. `brew install git-lfs`, or see the link above).
- Run `git lfs install` once per machine to register the LFS filters with git. Without it,
  `git lfs pull` silently fetches objects but does not check them out.
- Run `git lfs pull` to replace the LFS-tracked files with their real content.

If this step is skipped, those files stay as small text pointer stubs (starting with
`version https://git-lfs.github.com/spec/v1`), and tests that read them fail with confusing
errors that don't look LFS-related, e.g. `ValueError: could not convert string 'version' to
float64...` or `spiceypy` errors like `SpiceNOLEAPSECONDS`/`SpiceMISSINGTIMEINFO`.

### Running tests

- Running the test suite requires the Git LFS data described above: run `git lfs pull` first.
  Skipping this step will cause tests to fail.
- `uv run pytest` runs the full suite. `uv run python -m unittest discover tests` also works,
  since the suite is written with `unittest.TestCase`.
- To run tests for a single instrument, point either of the above at its directory, e.g.
  `uv run pytest tests/swapi`.
- `pytest-xdist` is installed (`-n auto`), but is not currently used in CI or recommended locally,
  as some tests are not yet safe to run in parallel (TODO).

### Linting

- `uv run ruff check .` lints the codebase.
- `uv run ruff format --check .` checks formatting without modifying files (drop `--check` to apply formatting).

### Pre-commit hooks

This repo uses [`pre-commit`](https://pre-commit.com/) to run `ruff check` and `ruff format` on
staged files before each commit (see `.pre-commit-config.yaml`).

- `uv run pre-commit install` sets up the git hook once per clone.
- `uv run pre-commit run --all-files` runs the hooks against the whole repo (useful the first
  time, or after changing the hook config).

Both hooks also run in CI, so failures caught locally would otherwise fail there too.
