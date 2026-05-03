# Handoff: Replace Instrument Response LUT with SWAPIResponse-derived response

## Goal

Remove the legacy `InstrumentResponseLookupTableCollection` (a zip of 62 SIMION-generated
`response_ESA<N>.dat` files) and derive the same per-passband instrument response on the
fly from the analytic `SWAPIResponse` model (passband polynomials + central effective area
+ azimuthal transmission). This is the natural follow-on to the geometric factor work that
already replaced the GF LUTs with `calculate_geometric_factor()`.

## What the legacy LUT provides

`InstrumentResponseLookupTableCollection` is a zip containing one file per ESA step
(62 total). Each file is a 7-column table: `(energy, elevation, azimuth, d_energy,
d_elevation, d_azimuth, response)` — a Monte Carlo–sampled instrument response at
~71k grid points per passband. On load, `InstrumentResponseLookupTable.__post_init__`
precomputes an `integral_factor` array:

```python
integral_factor = response * speed**4 * d_energy * cos(elevation) * d_azimuth * d_elevation / denominator
```

where `denominator = Σ(d_energy * cos(elevation) * d_elevation * d_azimuth)` normalizes
the phase-space volume.

## Where the LUT is used (only PUI fitting)

The LUT is consumed **only** in the PUI He+ forward model
(`calculate_pickup_ion.py`). It is **not** used by:
- Proton moments fitter (uses `SWAPIResponse` directly)
- Alpha moments fitter (uses `SWAPIResponse` directly)
- L3B VDF / differential flux (now uses `calculate_geometric_factor`)

### Call chain

1. **`swapi_processor.py:626`** passes `dependencies.instrument_response_calibration_table`
   to `calculate_pickup_ion_values()`.
2. **`calculate_pickup_ion_values()`** (`calculate_pickup_ion.py:52`) stores it in
   `ModelCountRateCalculator.response_lookup_table_collection`.
3. **`ModelCountRateCalculator.model_one_count_rate()`** (`calculate_pickup_ion.py:434`):
   - Gets the per-bin LUT via `.get_table_for_energy_bin(energy_bin_index)`.
   - Calls `get_speed_grid(response_lookup_table, ephemeris_time)` which uses the LUT's
     `energy`, `elevation`, `azimuth` arrays to build a 3-D speed grid in the solar wind
     rest frame (via SPICE frame rotations).
   - Calls `model_count_rate_integral(response_lookup_table, forward_model, speed_grid)`
     which evaluates the PUI distribution function `f(speed)` at every grid point and
     dot-products with `response_lookup_table.integral_factor`.
4. The result is combined with the dynamic geometric factor:
   `eff_correction * (gf / 2) * integral + background`.

### Key observation

The integral computed in step 3 is a **normalized** convolution of the PUI VDF with the
instrument response. The normalization (dividing by `denominator`) means the integral
represents the fraction of the response-weighted phase space that sees PUI flux. The
absolute calibration comes from `gf / 2` (already migrated to `SWAPIResponse`).

## What SWAPIResponse already provides

`SWAPIResponse` (`swapi_response.py`) encapsulates:
- `get_central_effective_area(voltage)` — A₀(V) in cm²
- `create_passband_grid(voltage)` — returns a `PassbandGrid` with `oa_grid` and `sg_grid`
  (structured arrays of `(energy_ratio, elevation, passband_value)`) at analytically
  evaluated grid points
- `get_azimuthal_transmission(azimuth_deg)` — T(φ), interpolated from the loaded CSV
- `eval_boundary_min(elevation)` / `eval_boundary_max(elevation)` — passband speed-ratio
  boundaries per elevation

The proton and alpha moments fitters already use this to compute model count rates via
Gauss-Legendre quadrature over `(speed_ratio, elevation, azimuth)`. The PUI fitter
should be convertible to the same approach.

## Suggested approach

### Option A: Analytic quadrature (recommended)

Replace the Monte Carlo grid with Gauss-Legendre quadrature, mirroring the proton moments
fitter. For each ESA step:

1. Get the `PassbandGrid` via `swapi_response.create_passband_grid(voltage)`.
2. Set up GL nodes over `(speed_ratio, elevation, azimuth)` — same node arrays as
   `calculate_proton_solar_wind_moments.py`.
3. At each quadrature point:
   - Compute the passband value via bilinear interpolation on the `PassbandGrid`.
   - Compute T(|φ|) from the azimuthal transmission.
   - Convert `(speed_ratio, elevation, azimuth)` → `(energy, elevation_deg, azimuth_deg)`
     → speed in instrument frame → speed in SW rest frame (via SPICE, same as current
     `get_speed_grid`).
   - Evaluate the PUI VDF `f(speed)`.
   - Accumulate: `f(speed) * passband * T(|φ|) * cos(θ) * speed_ratio * weight`.
4. The result replaces `model_count_rate_integral()`.

This removes the zip file dependency, the `InstrumentResponseLookupTable` class, and the
`InstrumentResponseLookupTableCollection` class entirely.

### Option B: Generate the LUT on the fly

Use `SWAPIResponse` to evaluate the response at the same ~71k grid points that the
SIMION LUT uses, producing an equivalent `InstrumentResponseLookupTable` without reading
the zip. This is simpler but retains the LUT data structure. Less clean.

**Recommendation**: Option A. The proton fitter already proves the quadrature approach
works, and the PUI fitter doesn't need the full ~71k-point grid.

## Files to modify

### Production code
- **`imap_l3_processing/swapi/l3a/science/calculate_pickup_ion.py`**
  - Remove `InstrumentResponseLookupTable` and `InstrumentResponseLookupTableCollection`
    imports.
  - Rewrite `ModelCountRateCalculator`: drop `response_lookup_table_collection` field,
    replace `get_speed_grid()` and `model_one_count_rate()` with quadrature-based
    equivalents using `SWAPIResponse`.
  - Remove `model_count_rate_integral()` function.
  - Update `calculate_pickup_ion_values()` signature: drop
    `instrument_response_lookup_table` parameter.
- **`imap_l3_processing/swapi/swapi_processor.py:626-627`**
  - Remove `dependencies.instrument_response_calibration_table` from the call.
- **`imap_l3_processing/swapi/l3a/swapi_l3a_dependencies.py`**
  - Remove `instrument_response_calibration_table` field and its fetch/load logic.
  - Remove `InstrumentResponseLookupTableCollection` import.
- **`imap_l3_processing/swapi/descriptors.py`**
  - Remove `INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR`.

### Files to delete
- **`imap_l3_processing/swapi/l3b/science/instrument_response_lookup_table.py`** — the
  entire module (both `InstrumentResponseLookupTable` and
  `InstrumentResponseLookupTableCollection` classes).
- **`tests/swapi/l3b/science/test_instrument_response_lookup_table.py`**
- **`tests/test_data/swapi/imap_swapi_instrument-response-lut_20241023_v000.zip`**

### Tests to update
- **`tests/swapi/l3a/science/test_calculate_pickup_ion.py`** — remove all
  `InstrumentResponseLookupTableCollection` usage; update `ModelCountRateCalculator`
  construction. Two tests load the zip directly
  (`test_calculate_pickup_ions_with_minimize_mocked` at line 137,
  `test_calculate_pickup_ions_with_minimize` which is currently skipped).
- **`tests/swapi/test_swapi_processor.py`** — remove
  `INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR` from imports and all `input_file_names`
  lists (lines 153, 461, 592, 796, 1090). Remove `instrument_response_lut` from mock
  dependency setup (lines 226, 239).
- **`tests/swapi/l3a/test_swapi_l3a_dependencies.py`** — remove descriptor from expected
  list and file path from `test_from_file_paths`.
- **`tests/integration/test_swapi_processor_integration.py`** — remove
  `instrument-response-lut` from inline dependency JSON (line 115).
- **`tests/integration/test_data/swapi/imap_swapi_l3a_proton-sw_20260425_v001.json`** —
  remove the `instrument-response-lut` entry (lines 30-32).

### Scripts and config
- **`run_local.py`** — remove instrument-response-lut path from L3A dependency args
  (line ~1428).
- **`run_local_with_upload_and_download.py`** — remove from all SWAPI dependency strings.
- **`run_with_download_and_upload_using_docker.sh`** — remove from dependency JSON.
- **`scripts/swapi/upload_swapi_l3_dependencies.py`** — remove the zip from upload list
  (line 13).
- **`scripts/swapi/imap_swapi_l3a_proton-sw_dependency_template.json`** — remove entry
  (lines 9-12).
- **`scripts/swapi/pui_fitting_experiments.py`** — remove
  `InstrumentResponseLookupTableCollection` import and loading; update
  `ModelCountRateCalculator` construction.

## Key physics to preserve

1. **Speed grid in SW rest frame**: The current code converts each LUT grid point
   `(energy, elevation, azimuth)` → speed in instrument frame → SPICE rotation to
   ECLIPJ2000 → subtract SW velocity → magnitude = speed in SW frame. The quadrature
   replacement must do the same transform at each GL node.

2. **Integral normalization**: `integral_factor` divides by the total phase-space volume
   `denominator`. The quadrature weights naturally handle this if the integration is set
   up correctly (the GL weights over the full domain sum to the domain volume).

3. **Speed⁴ weighting**: The `integral_factor` includes `speed**4`. This converts from
   the instrument's `(energy, θ, φ)` coordinates to the VDF's speed-space Jacobian.
   With the analytic passband, the equivalent is the `r · cos(θ)` integrand factor
   (speed ratio times cosine elevation) multiplied by the passband value, similar to
   `calculate_geometric_factor`.

4. **The `/2` in `gf/2`**: Rankin Eq 6 normalization. This stays regardless of how the
   integral is computed.

## Verification

- The PUI test `test_calculate_pickup_ions_with_minimize_mocked` is the key regression
  test. It loads real response data and fits synthetic PUI spectra. After migration, the
  fitted parameters should agree within tolerance.
- Run: `conda run -n imapL3 python -m pytest tests/swapi/ -v`
- Run: `conda run -n imapL3 python -m pytest tests/integration/ -v`

## Current branch state

Branch `dynamic-geometric-factor` in worktree `~/projects/imap_L3_processing_gf_branch`.
All 170 SWAPI unit tests pass + 13 integration tests pass. No uncommitted dependency on
the instrument response LUT from GF code — the LUT is now isolated to PUI fitting only.
