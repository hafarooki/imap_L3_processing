# SWAPI Solar Wind Moments Algorithm Notes

## Code Files

- `imap_l3_processing/swapi/l3a/science/swapi_response.py` — Instrument response model. Loads calibration tables (azimuthal transmission, central effective area, passband polynomial fits) and builds a `PassbandGrid` per ESA voltage step via `SWAPIResponse.create_passband_grid`.
- `imap_l3_processing/swapi/l3a/science/calculate_proton_solar_wind_moments.py` — Core fitting algorithm. `fit_solar_wind_proton_moments` implements the three-step procedure (SPICE → initial guess → Levenberg–Marquardt); `calculate_integral` is the Numba-JIT count-rate integral over elevation, azimuth, and speed.
- `imap_l3_processing/swapi/l3a/science/speed_calculation.py` — ESA bin layout constants (`SWAPI_SCIENCE_BINS`, `SWAPI_COARSE_SWEEP_BINS`, `SWAPI_FINE_SWEEP_BINS`) and the `esa_voltage_to_proton_speed` conversion.
- `imap_l3_processing/swapi/swapi_processor.py` — Production pipeline entry point. `SwapiProcessor.process_l3a_proton` slices science bins, computes per-bin measurement times, calls `fit_solar_wind_proton_moments`, and propagates uncertainties to scalar speed, clock angle, and deflection angle before writing the CDF.
- `scripts/swapi/validate_proton_moments.py` — Offline validation script. Runs the fitter over a full day of real L2 data and produces a six-panel figure comparing fitted moments against OMNI reference data.

## TODO

- [ ] Validate alphas
- [ ] Empirically validate alpha-species correction $\mathcal{A}_0^\alpha/\mathcal{A}_0^p = \varepsilon_\alpha/\varepsilon_p$ against OMNI alpha density (the paper does not prove this; alpha ABM measurements weren't taken)
- [ ] Coordinate with cal-file format owner to add a `lab_time` field to the efficiency LUT.
Interim: `eps_p_lab` is pinned to the first row at/after 2025-11-01 because the pre-2025-11 rows in the current LUT are placeholder values (0.02348 repeated) and using them as the lab denominator drove proton density 6× too low.
Replace with the proper `lab_time` lookup once the LUT format gains the field (see `EfficiencyCalibrationTable.eps_p_lab`).
- [ ] Remove old SW model from pickup ion code (?)
- [ ] Dynamic calculation of pickup ion geometric factor (?)
- [ ] Choose bad fit flags
- [ ] Fix the integration edge cases 
- [ ] finalize document
- [ ] **clean up/refactor code**
- [ ] Human review: run integration test output through a CDF reader and sanity-check variable names, units, and fill values (test passes automatically)
- [ ] Human review: read through unit tests top-to-bottom and verify coverage is meaningful, not just passing (all 167 pass automatically)
- [ ] Try Poisson MLE instead of least squares
- [ ] Compare with WIND with production-like testing (unlike current validation script)
- [ ] Clarify behavior of partial last chunk from `chunk_l2_data` — if the day's sweep count isn't a multiple of 5, the final group has fewer sweeps and its epoch (`sci_start_time[0] + 30s`) is off-center; decide whether to drop it, pad it, or accept the timestamp offset
- [ ] validate more cases
- [ ] Confirm output format for bulk velocity — currently written as scalar speed + clock angle + deflection angle (with uncertainties); decide if RTN vector components should also be in the CDF

## Model

The coincidence count rate at ESA voltage $V$ is
$$C(V) = \sum_s \int d^3v \; v \, f^s(\mathbf{v}) \, \mathcal{A}^s(\mathbf{v}, V),$$
where $f^s$ is the VDF of species $s$ and $\mathcal{A}^s$ is the effective area.

The effective area is decomposed as
$$\mathcal{A}^s(v, \theta, \phi, V) = \mathcal{A}_0^s(V) \cdot P^s\!\left(\dfrac{v}{v_0^s},\, \theta,\, \phi,\, V\right) \cdot T(\phi),$$
where $v_0^s = \sqrt{2 k^* q^s |V| / m^s}$ is the central speed ($k^* = 1.89$ eV/V; see "Two k-factors" below), $\mathcal{A}_0^s(V)$ is the central effective area (from lab measurements, interpolated at each $V$), $P^s$ is the energy-angle passband (from SIMION, separate sunglasses/open-aperture grids, normalized to $P(1, 0°, V) = 1$), and $T(\phi)$ is the azimuthal transmission factor ($T \approx 10^{-3}$ at $|\phi| < 9°$). $T$ is an even function; the calibration table covers $|\phi|$ only, and `_interpolate_transmission` indexes it by $|\phi|$ after wrapping $\phi$ to $(-180°, 180°]$.

#### Two k-factors

The L2 product labels its energy axis using an outdated SWAPI k-factor:
$$\texttt{esa\_energy}_\text{L2} = k_\text{L2} \cdot |V|, \qquad k_\text{L2} = 1.93\ \text{eV/V}.$$
The L3 fitter expects true ESA voltage $V$ as input, so any code that reads L2's `esa_energy` and feeds it to `fit_solar_wind_proton_moments`, `create_passband_grid`, or `esa_voltage_to_proton_speed` must first divide by $k_\text{L2}$.

All internal L3 physics — passband normalization, central speed $v_0^s$, the polynomial fits in $\log(k^*|V|)$ — uses the revised k-factor $k^* = 1.89$ eV/V from high-resolution SIMION simulations. The two values are exposed as `SWAPI_L2_K_FACTOR` and `SWAPI_K_FACTOR` in `imap_l3_processing/swapi/l3a/science/speed_calculation.py`. Mixing them silently shifts the fitted moments by ~1–2%.

![Central effective area and azimuthal transmission](figures/calibration_curves.png)

*Generated by `docs/swapi/figure_src/plot_calibration_curves.py`.*

$P^s$ is voltage-dependent. SIMION results at discrete energies are stored as polynomial fits of $\log P$ in $\log(k^* |V|)$; `get_passband_values` reconstructs $P$ via `np.exp(np.polyval(coeffs, log(k*|V|)))`. When the requested voltage falls outside the calibrated range stored in the CSV, it is silently clamped to the nearest endpoint — bins far from the proton peak receive the boundary passband value. For each $V_i$, `SWAPIResponse.create_passband_grid` evaluates these fits onto a uniform (elevation, speed-ratio) grid (`_TARGET_ELEVATIONS` = −12° to 10.5° in 0.5° steps, `_TARGET_SPEED_RATIOS` = 101 points from 0.9 to 1.1) and stores the result in a `PassbandGrid` struct, precomputed once per ESA step.

The solar wind proton VDF is modeled as a drifting Maxwellian:
$$f_p(\mathbf{v}) = \frac{n}{(\sqrt{2\pi}\, v_\text{th})^3} \exp\!\left(-\frac{v^2 + v_b^2 - 2 v\, v_b \cos\alpha}{2 v_\text{th}^2}\right),$$
where $\cos\alpha = \sin\theta_b \sin\theta + \cos\theta_b \cos\theta \cos(\phi - \phi_b)$ and $v_\text{th} = \sqrt{T/m_p}$ ($T$ in energy units).

Substituting into the count rate integral in spherical velocity coordinates $(v, \theta, \phi)$:
$$C(V) = \frac{n\, \mathcal{A}_0(V)}{(\sqrt{2\pi}\, v_\text{th})^3} \sum_\text{region} \int \cos\theta\, d\theta \int T(\phi)\, d\phi \int v^3\, P\!\left(\tfrac{v}{v_0}, \theta\right) \exp\!\left(-\frac{v^2 + v_b^2 - 2vv_b\cos\alpha}{2v_\text{th}^2}\right) dv.$$

The sum runs over three azimuth regions: sunglasses (SG, $|\phi| \leq 20°$), left open aperture (OA, $-150° < \phi < -20°$), and right open aperture ($20° < \phi < 150°$).

### Angular limits

Integration limits are centered on $(\theta_b, \phi_b)$ and clamped per region to the bilinear-interpolation extent of that region's passband. The half-width is where the Maxwellian drops below $\varepsilon$:
$$\Delta\alpha = \arccos\!\left(\mathrm{clip}\!\left(\frac{v_\text{th}^2 \ln\varepsilon}{v_0 v_b} + 1;\; -1,\; 1\right)\right),$$
with $\varepsilon_\text{SG} = \varepsilon_\text{OA} = 10^{-6}$. The integration window is $[\theta_b - \Delta\alpha,\; \theta_b + \Delta\alpha]$ in elevation and $[\phi_b - \Delta\alpha,\; \phi_b + \Delta\alpha]$ in azimuth, then clamped per region:

| Region | Elevation bounds | Azimuth bounds |
|--------|------------------|----------------|
| SG     | $[-10.5°,\; 7°]$    | $[-20°,\; 20°]$ |
| OA−    | $[-12°,\; 10.5°]$   | $[-150°,\; -20°]$ (overridden by scan; see below) |
| OA+    | $[-12°,\; 10.5°]$   | $[20°,\; 150°]$ (overridden by scan; see below) |

Elevation bounds extend one half-cell beyond the nonzero stored rows of each passband (the bilinear-interp extent). Truncating earlier misses a small "second peak" near the FOV edge where the rising Maxwellian (toward $\theta_b$) outweighs the falling passband.

#### OA azimuth: integrand-aware trim

For OA, the gaussian-only $\Delta\alpha$ above is replaced by a transmission-aware scan (`_trim_oa_azimuth_by_integrand`). The motivation: $T(\phi)$ is essentially zero from $20°$ to $25°$ and only rises to its plateau by $30°$, so the standard $\Delta\alpha = 5.26\,\sigma_\alpha$ window (from $\varepsilon = 10^{-6}$) routinely opens a 1°–10° sliver of OA where the integrand $\rho \times T \approx 0$ — wasting integration nodes on a dead zone. Conversely, when $\phi_b$ sits past $\sim 15°$, the OA window must extend well past $\Delta\alpha$ to capture the high-$T$ region where the gaussian tail × full transmission still contributes.

The trim works in three steps:

1. **Scan** $\rho(\phi) \cdot T(\phi)$ at $(\theta = \mathrm{clip}(\theta_b, \theta_\text{lo}, \theta_\text{hi}),\; v = v_0)$ across the full OA passband ($[20°, 150°]$ for OA+, $[-150°, -20°]$ for OA−), at adaptive spacing $\Delta\phi_\text{scan} = \mathrm{clip}(\sigma_\alpha / 2,\; 0.1°,\; 1°)$. Spacing tracks the gaussian width so cold-plasma peaks (sub-degree wide) aren't missed by a coarse grid. The "elevation peak inside the window" is where density is maximal at fixed $\phi$.
2. **Skip** the OA region entirely if $\max(\rho T) < 10^{-9}$ — gaussian tail is too far from any meaningful-transmission $\phi$.
3. **Anchor** the integration window at the OA inner boundary ($\pm 20°$, on the SG side) and trim only the *far* end at the threshold $\rho T > 10^{-3} \times \max$. Anchoring is essential: $T(\pm 20°) = 0$ by construction, so the rising-edge peak (at $\sim \pm 21°$ for $\phi_b$ near zero) sits between the boundary and the first scan grid point that exceeds threshold; trimming the boundary side would cut off real signal.

For typical solar wind at $T \sim 10$ eV and $|\phi_b| < 6°$, the trim collapses the OA window to a few degrees adjacent to the SG/OA transition. For high-deflection cases ($|\phi_b| > 15°$), the scan finds the peak at $\phi \sim 25°$–$30°$ and the trim returns essentially the same window as the gaussian-only $\Delta\alpha$ would. Median chunk fit time drops from ~963 ms (with the gaussian-only $\Delta\alpha$ feeding 41 OA azimuth nodes) to ~175 ms — a ~5.5× speedup with no loss of accuracy on the reference-integral histogram (max ratio error stays at 10% for low-rate edge cases, identical to the un-trimmed integrator).

### Speed limits

The speed window is computed per elevation by clamping the Maxwellian's effective support $[v_b - 10v_\text{th},\; v_b + 10v_\text{th}]$ to the passband's per-elevation support $[r_\text{min}(\theta),\, r_\text{max}(\theta)]\,v_0$:
$$v_\text{min}(\theta) = \mathrm{clip}\!\left(v_b - 10v_\text{th};\; r_\text{min}(\theta)\,v_0,\; r_\text{max}(\theta)\,v_0\right), \qquad v_\text{max}(\theta) = \mathrm{clip}\!\left(v_b + 10v_\text{th};\; r_\text{min}(\theta)\,v_0,\; r_\text{max}(\theta)\,v_0\right).$$
$r_\text{min}(\theta)$ and $r_\text{max}(\theta)$ (blue curves below) are read from `min_*_boundary` / `max_*_boundary` via `_eval_boundary`. When the Maxwellian's support falls entirely outside the passband at a given elevation, the window collapses and the elevation contributes nothing.

Two implementation details about the boundaries:

- **Voltage-independent.** The boundaries are computed once at `SWAPIResponse.from_files` time using a representative voltage, not per `create_passband_grid` call. This is valid because `exp(polyval(...))` is always strictly positive, so the same (elevation, speed-ratio) cells are nonzero for every in-range voltage.
- **Expanding interpolation.** `_eval_boundary` does not linearly interpolate between stored grid points. Instead, for a query elevation between two stored points, it returns the more expansive of the two: `min` of the two min-boundaries, `max` of the two max-boundaries. This guarantees the integration window brackets the full nonzero passband at all elevations, at the cost of a small over-integration in the gap between stored points.

![SWAPI passband and integration region at three beam energies](figures/passband_boundaries.png)

*Generated by `docs/swapi/figure_src/plot_passband_boundaries.py`.*

The integral is evaluated as nested elevation $\to$ azimuth $\to$ speed loops with **Gauss-Legendre quadrature** in every dimension at $(N_\text{elev}, N_\text{az,SG}, N_\text{az,OA}, N_\text{speed}) = (21, 21, 21, 11)$. OA azimuth nodes are now equal in count to SG since the transmission-aware trim above produces a tighter integration window — 21 nodes over the trimmed range gives equivalent or better accuracy than the previous 41 nodes over the wider gaussian-only window. The per-elevation row $v^3 P(v/v_0, \theta)$ is precomputed once and reused across all azimuths (azimuth enters only through the Maxwellian). The passband is renormalized at runtime so that $P(1, 0°, V) = 1$ (`PassbandGrid` stores the raw polynomial-evaluated SIMION values). JIT-compiled with Numba. See `calculate_integral(PassbandGrid, SWParams, central_speed, central_effective_area, azimuthal_transmission, transmission_spacing)`.

Representative model spectra below exercise the integrator's edges. The off-axis azimuth case ($\phi_b = 18°$) is the worst overall (18.8%): the bulk sits adjacent to the SG/OA transition where the azimuthal transmission rises five orders of magnitude across 10°, and the optimized integrator cannot resolve that transition as densely as the reference's 0.1° spacing.

![Production vs ground-truth spectra for six representative SW configurations](figures/spectra.png)

*Generated by `docs/swapi/figure_src/plot_spectra.py`.*

### Integrator Validation

The optimized integrator is validated against a high-resolution fixed-limit reference (`reference_integral_fixed_limits`) over 10000 random solar-wind configurations (`reference_integrals.csv`). Each configuration is evaluated at the ESA voltage whose central proton speed equals its `bulk_speed`. The histogram below bins the resulting (optimized / reference) ratio, stacked by reference count rate.

High-rate cases ($\geq 10^3$ Hz) cluster within $\pm 1\%$ of unity. The tail at ratio $< 0.5$ is configurations with reference $< 0.1$ Hz where the bulk direction sits many sigma outside the FOV — both integrators round to ~0, well below the noise floor.

![Optimized / reference ratio histogram, stacked by reference count rate](figures/reference_vs_optimized_histogram.png)

*Generated by `scripts/swapi/reference_vs_optimized_histogram.py`.*

## ESA Sweep Bin Layout

Each 12-second ESA sweep contains **72 bins** (indices 0–71). Their roles are:

| Indices | Count | Description |
|---------|-------|-------------|
| 0       | 1     | **Always discarded.** Hardware artifact; contains no science data. |
| 1–62    | 62    | **Coarse sweep.** Uniform logarithmic energy steps covering the full proton solar wind range. |
| 63–71   | 9     | **Fine sweep.** Higher-resolution scan of the proton peak, using smaller voltage steps for better moment accuracy. |

The fine-sweep bins are currently used in the moments fit (they overlap the peak and add resolution). The production processor and validation scripts use `SWAPI_SCIENCE_BINS = slice(1, 72)` (i.e., `SWAPI_COARSE_SWEEP_BINS + SWAPI_FINE_SWEEP_BINS`) and always exclude bin 0 explicitly.
Occasionally, one or more fine sweep bins will have zero ESA voltage; we exclude these from the fit.
For solar wind protons, we also restrict the fit to points that are within half of the maximum count rate to avoid including other populations.
Named constants are defined in `imap_l3_processing/swapi/l3a/science/speed_calculation.py`:

```python
SWAPI_COARSE_SWEEP_BINS = slice(1, 63)   # indices 1–62
SWAPI_FINE_SWEEP_BINS   = slice(63, 72)  # indices 63–71
SWAPI_SCIENCE_BINS      = slice(1, 72)   # indices 1–71 (all usable bins)
```

### Measurement timing

The start time for each sweep is denoted $t_\text{epoch}$. For bin $i$ (0-indexed, although recall that bin 0 is skipped), the measurement time is:
$$t_i = t_\text{epoch} + i \cdot \tfrac{12}{72}\,\text{s} = t_\text{epoch} + i \cdot 0.1\overline{6}\,\text{s}.$$

## Fitting Procedure

Given $N$ measurements $(C_i, V_i, t_i)$, the solar wind moments $(n, T, \mathbf{v}_b^\text{RTN})$ are fit in three steps:
1. Obtain RTN $\rightarrow$ SWAPI rotation matrices $R_i$ and spacecraft velocity $\mathbf{v}_\text{sc}^\text{RTN}$ from SPICE.
2. Compute an initial guess: temperature and bulk speed from a Gaussian fit to $C_i$ vs. $v_i$, with bulk velocity assumed purely anti-sunward.
3. Refine by nonlinear least squares.
> **TODO** alphas

### Step 1: SPICE

$R_i$ (shape $N \times 3 \times 3$) are obtained from `get_rotation_matrix(IMAP_RTN, IMAP_SWAPI)` at each measurement time. TT2000 nanoseconds are converted to ET via $t^\text{ET} = \text{unitim}(t / 10^9,\, \text{TT},\, \text{ET})$.

$\mathbf{v}_\text{sc}^\text{RTN}$ (km/s) is obtained at the median time from `imap_state` in ECLIPJ2000 and rotated into RTN:
$$\mathbf{v}_\text{sc}^\text{RTN} = M_{\text{ECL} \rightarrow \text{RTN}} \, \mathbf{v}_\text{sc}^\text{ECL}.$$

### Step 2: Initial guess

**Temperature and speed magnitude** are obtained from a Gaussian fit to $C_i$ vs. $v_i = \sqrt{2 k^* q V_i / m_p}$:
$$C_i \approx A \exp\!\left(-\frac{(v_i - v_b)^2}{2 \sigma_v^2}\right), \qquad A, v_b, \sigma_v > 0.$$
The fitted width $\sigma_v$ is used directly as the thermal width (no passband subtraction), with a floor $\sigma_{\text{floor}, v}$ corresponding to a 1 eV temperature floor:
$$\sigma_{\text{thermal}, v} = \max(\sigma_v,\, \sigma_{\text{floor}, v}).$$
The initial temperature is $T_0 = m_p \sigma_{\text{thermal}, v}^2$.

**Velocity direction** is set to purely anti-sunward, $\mathbf{v}_b^\text{RTN} = (v_b, 0, 0)$. The optimizer in Step 3 recovers the transverse components $v_T$ and $v_N$ from the small spin-phase modulation of the bulk azimuth/elevation in the instrument frame (the spin axis is the SWAPI boresight, so an anti-sunward bulk projects exactly onto $-Y_\text{SWAPI}$ at every spin phase, while non-zero $v_T,\,v_N$ produce a sinusoidal wobble of order $\arcsin(\sqrt{v_T^2+v_N^2}/v_R)$ around that direction).

The initial density is scaled to match the mean observed count rate:
$$n_0 = \frac{\langle C_i \rangle}{\langle C_i^\text{model}(n=1) \rangle}.$$

Figure below shows initial-guess and final-fit accuracy across 10000 random solar wind configurations (bulk speed 300–800 km/s, temperature 2–50 eV log-uniform, density 2–20 cm⁻³, $v_T, v_N \in [-50, 50]$ km/s). Synthetic count rates are produced from the forward model using the real SWAPI 71-bin science voltage sweep (from the L2 CDF), 5 sweeps per fit, realistic spin geometry (spin axis = boresight, 15 s period), and Poisson noise — matching the production processor exactly. The velocity initial guess for $v_T$ and $v_N$ is zero by construction; the final optimizer recovers them from the spin-phase modulation. 0 bad-fit flags across all 10000 cases. Generated by `docs/swapi/figure_src/plot_initial_guess_accuracy.py`.

![Initial-guess vs. final-optimizer accuracy for 10000 synthetic solar wind cases](figures/initial_guess_accuracy.png)

### Step 3: Optimization

Parameters $[\log n,\, \log T,\, v_R,\, v_T,\, v_N]$ (with $\mathbf{v}_b^\text{RTN} = (v_R, v_T, v_N)$ in the inertial RTN frame) are fit by `scipy.optimize.least_squares` using the Levenberg–Marquardt algorithm (`method='lm'`, `diff_step=1e-4`) with Poisson-weighted residuals over a half-mean mask
$$\mathcal{M} = \{i : C_i \geq \tfrac{1}{2} \langle C \rangle\}, \qquad r_i = \frac{C_i^\text{model} - C_i}{\sigma_i}\ \ \text{for}\ i \in \mathcal{M}, \qquad \sigma_i = \frac{\sqrt{\max(C_i \cdot \Delta t,\; 1)}}{\Delta t},$$
where $\Delta t = 0.145\ \text{s}$ is the livetime per ESA energy step. The Poisson standard deviation of a count rate uses $\max(C_i \Delta t, 1)$ — the observed photon count clamped to at least 1 (so $\sigma_i$ is never zero), divided by $\Delta t$ to return to count-rate units.

Bins below half the mean count rate are dropped: they carry essentially no proton signal (the proton peak sits in a narrow speed window — most ESA bins are noise floor and off-axis leakage), but with N ≈ 355 bins per 5-sweep fit they still contribute non-trivial residuals that can pull the moments. The mask is global across all sweeps × bins fed to one fit, and is computed once from the observed count rates only — the model never re-evaluates it. Initial-guess construction (the Gaussian curve fit and the mean-rate density scaling) is unaffected and uses all bins. If fewer than 5 bins survive the mask (fewer constraints than free parameters), the mask is dropped and all bins are fit, which is purely a guard against pathological inputs (e.g. all-zero count rates).

Density and temperature are parameterized in log-space to keep them positive throughout optimization. The optimizer's `success` flag is mapped to `bad_fit_flag`: failure sets `HI_CHI_SQ`.

#### Wrong-basin detection (post-fit flip check)

The forward model is approximately invariant under the spin-axis mirror $(v_T, v_N) \rightarrow (-v_T, -v_N)$, broken only weakly by the SG passband elevation asymmetry ($[-10.5°, +7°]$) and (in real data) spacecraft velocity. The two basins are *not* truly degenerate — the truth is the global minimum, and the mirror is a local minimum with $\chi^2$ typically $100$–$500\times$ higher. But starting LM from $(v_T, v_N) = (0, 0)$ puts the optimizer on the saddle between them, so noise can push it into either basin, and once committed it stays trapped at the local minimum.

Detection is cheap: after LM converges to $\hat{\mathbf{x}} = (\log n, \log T, \mathbf{v}_b)$, build the flipped solution by rotating the bulk velocity 180° about the spin axis ($\hat{\mathbf{s}}$ in RTN, recovered as the second row of any rotation matrix since $R_i \hat{\mathbf{s}} = \hat{\mathbf{y}}_\text{SWAPI}$ by design):
$$\mathbf{v}_b' = 2(\mathbf{v}_b \cdot \hat{\mathbf{s}})\,\hat{\mathbf{s}} - \mathbf{v}_b.$$
This works for any spin axis orientation, not just radial. Evaluate the residual at $\hat{\mathbf{x}}' = (\log n, \log T, \mathbf{v}_b')$. If $\chi^2(\hat{\mathbf{x}}') < \chi^2(\hat{\mathbf{x}})$, LM was in the wrong basin — re-run LM from $\hat{\mathbf{x}}'$ to land at the correct local minimum. The cost is one extra residual evaluation in the typical case, plus one extra LM run for the $\sim 1\%$ of cases that need it. On a synthetic benchmark (see `plot_initial_guess_accuracy.py`, 10000 cases), the flip check correctly redirects all wrong-basin fits with no false positives.

![χ² landscape in the (v_T, v_N) plane showing the truth and spin-axis-mirror minima](figures/wrong_basin.png)

*Generated by `docs/swapi/figure_src/plot_wrong_basin.py`. The mirror minimum has $\chi^2$ roughly $200\times$ the truth — easily distinguished by a single residual evaluation at the flipped solution.*

> **Note on `diff_step`.** The default finite-difference step in `least_squares` scales with the parameter magnitude, producing steps of $\sim 10^{-8}\ \text{km/s}$ for $v_T,\, v_N$ near zero. For cold plasma ($T \lesssim 5\ \text{eV}$), the resulting count-rate perturbation falls below the GL-quadrature noise floor, making the numerical Jacobian for $v_T$ and $v_N$ pure noise and inflating the LM damping factor to $\sim 10^{13}$, which freezes all parameters. `diff_step=1e-4` is the empirical optimum over $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$: it sits above the noise floor (giving a clean Jacobian and correct convergence) while keeping the linearization error small enough not to degrade accuracy ($10^{-2}$ degrades $v_N$ RMSE; $10^{-1}$ produces bad fits and a 5× slowdown).

Inside the model, the spacecraft velocity is subtracted and the result rotated into instrument coordinates:
$$\mathbf{v}_{b,i}^\text{xyz} = R_i (\mathbf{v}_b^\text{RTN} - \mathbf{v}_\text{sc}^\text{RTN}).$$
The azimuth and elevation angles fed to the integral are then
$$\phi_{b,i} = \operatorname{arctan2}(-v_{b,i,x},\, -v_{b,i,y}), \qquad \theta_{b,i} = \arcsin\!\left(-\frac{v_{b,i,z}}{v_b}\right).$$
(Sign conventions and coordinate system follow the instrument paper.)

#### Deadtime correction

The detector deadtime is $\tau = 183.7\ \text{ns}$. Following Tsoulfanidis (1995), p. 74, the true count rate $n$ and measured rate $g$ are related by $n = g / (1 - g\tau)$. Rearranged for the forward model, the true model rate $C^\text{model}$ is mapped to the predicted observed rate before computing residuals:
$$C^\text{observed}_i = \frac{C^\text{model}_i}{1 + \tau\, C^\text{model}_i}.$$
This deadtime correction is often non-negligible.
It reaches 5% at $C \approx 2.7\times 10^5 \text{Hz}$, which is not an uncommonly high coincidence rate.
Not accounting for this would result in an overestimate of the model count rate and thus in underestimate of the density in such cases.

#### Parameter uncertainties

The Jacobian $J$ of the normalized residuals $r_i$ with respect to $[\log n,\, \log T,\, v_R,\, v_T,\, v_N]$ at the solution is returned by the optimizer. The covariance matrix in parameter space is estimated as
$$\Sigma_x = (J^\top J)^+,$$
where ${}^+$ denotes the Moore–Penrose pseudoinverse. Uncertainty in physical parameters follows by Gaussian propagation:
$$\sigma_n = n\,\sqrt{\Sigma_{x,00}}, \qquad \sigma_T = T\,\sqrt{\Sigma_{x,11}},$$
and for derived quantities:
$$\sigma_{|\mathbf{v}|} = \sqrt{\hat{\mathbf{v}}^\top \Sigma_v \hat{\mathbf{v}}},$$
$$\sigma_{\phi_c} = \sqrt{\mathbf{g}_c^\top \Sigma_v \mathbf{g}_c}, \quad \mathbf{g}_c = \frac{1}{v_T^2+v_N^2}\begin{pmatrix}0\\v_N\\-v_T\end{pmatrix},$$
$$\sigma_{\phi_d} = \sqrt{\mathbf{g}_d^\top \Sigma_v \mathbf{g}_d}, \quad \mathbf{g}_d = \frac{1}{|\mathbf{v}|^2}\begin{pmatrix}-\sqrt{v_T^2+v_N^2}\\v_R v_T/\sqrt{v_T^2+v_N^2}\\v_R v_N/\sqrt{v_T^2+v_N^2}\end{pmatrix},$$
where $\Sigma_v = \Sigma_x[2\!:\!5,\,2\!:\!5]$ is the $3\times 3$ velocity block, $\hat{\mathbf{v}} = \mathbf{v}/|\mathbf{v}|$, $\phi_c = \arctan2(v_T, v_N)$ is the clock angle, and $\phi_d = \arctan2(\sqrt{v_T^2+v_N^2},\,v_R)$ is the deflection angle.

These uncertainties reflect Poisson shot noise only; model imperfection (non-Maxwellian features, alpha contamination, temporal variability within the fit window) is not captured.

When $v_T = v_N = 0$ exactly, $\mathbf{g}_c$ and $\mathbf{g}_d$ are undefined (division by $v_T^2 + v_N^2 = 0$); the processor sets both $\sigma_{\phi_c}$ and $\sigma_{\phi_d}$ to NaN in that case.

## Alpha Particle Moments

The alpha solar wind moments fitter (`calculate_alpha_solar_wind_moments.py`) reuses the proton forward model (`_model_count_rates`) and adds a 3-DOF Levenberg–Marquardt fit over $(n_\alpha, T_\alpha, \Delta v)$ where alphas are constrained to drift along the local magnetic field:
$$\mathbf{v}_\alpha = \mathbf{v}_p^* + \Delta v \, \hat{\mathbf{B}}.$$
This encodes the observed solar-wind fact that non-field-aligned differential drift is quenched by firehose/mirror instabilities.

**Sign convention.** $\Delta v > 0$ means the alpha drifts along $+\hat{\mathbf{B}}$ relative to protons. Since MAG provides $\mathbf{B}$ with a physical polarity, $\Delta v$'s sign is interpretable.

### Two-stage strategy

The pipeline processes 5-sweep chunks (matching the existing alpha LUT cadence). Each chunk is sliced to the **62 coarse-sweep bins** (`SWAPI_COARSE_SWEEP_BINS`) and flattened over (sweep, bin) to a 310-residual axis. The 9 fine-sweep bins (63–71) are dropped because they cluster near the proton peak and inform proton thermal width only — useless when protons are held fixed.

- **Stage 1**: Re-run `fit_solar_wind_proton_moments` on the 310-residual axis to get $(n_p^*, T_p^*, \mathbf{v}_p^{*\,\text{RTN}})$. This is independent of the per-sweep proton L3A product (which uses 71 bins) and is persisted as `reference_proton_*` fields in the alpha CDF.
- **Stage 2**: Fit $(n_\alpha, T_\alpha, \Delta v)$ on the same 310-residual axis with proton parameters held fixed. The combined observed model is
  $$C_\text{obs}(V) = \text{deadtime}\!\left(C_p^\text{true}(V; \theta_p^*) + C_\alpha^\text{true}(V; \theta_\alpha)\right),$$
  so deadtime acts on the sum (important in proton-peak bins where alphas are negligible but deadtime is at its largest). Stage 2 reuses the SPICE rotation matrices and spacecraft velocity computed for Stage 1.

Joint refit of both species is deferred (Stage 2 does not resolve alpha contamination of the proton fit).

### Species-dependent pieces

The same `PassbandGrid` infrastructure works for both species — only $v_0^s = \sqrt{2 k^* (q/m)^s |V|}$ changes. `SWAPIResponse.create_passband_grid(V, m, q)` is keyed by $(V, m, q)$ in `_grid_cache`, so proton and alpha grids cache independently. At the same $V$,
$$\frac{v_0^\alpha}{v_0^p} = \sqrt{\frac{q_\alpha m_p}{q_p m_\alpha}} = \sqrt{\frac{2 m_p}{4 m_p}} = \frac{1}{\sqrt 2}.$$
Thermal speed uses the elementary charge regardless of species — "$T_\alpha$ in eV" means $k_B T_\alpha = T_\alpha \cdot e$ Joules:
$$v_{th}^\alpha = \sqrt{\frac{T_\alpha \cdot e}{m_\alpha}}.$$

### Two k-factors (continued)

The same proton-revised $k^* = 1.89$ eV/V applies to both species inside the fitter. The L2 product still uses the outdated $k_{L2} = 1.93$ to label its `esa_energy` field, so the alpha pipeline divides L2 `esa_energy` by `SWAPI_L2_K_FACTOR` before passing voltages in.

### Effective-area scaling

`EfficiencyCalibrationTable` stores **absolute** detection efficiencies for each species ($\sim 0.11$ for protons, $\sim 0.15$ for alphas). The lab-calibrated `central_effective_area(V)` table is the proton ABM-derived $\mathcal{A}_0^p(V_\text{lab})$. At the integration site we apply a per-species ratio to convert this to the runtime species/time effective area:
$$\text{proton}: \quad \texttt{central\_effective\_area\_scale} = \frac{\varepsilon_p(t)}{\varepsilon_p(t_\text{lab})}$$
$$\text{alpha}: \quad \texttt{central\_effective\_area\_scale} = \frac{\varepsilon_\alpha(t)}{\varepsilon_p(t_\text{lab})}$$
The proton-lab denominator is used for **both** species so the alpha scale folds the species correction $\mathcal{A}_0^\alpha/\mathcal{A}_0^p \approx \varepsilon_\alpha/\varepsilon_p$ together with alpha time drift into a single ratio. This factor is exposed as `EfficiencyCalibrationTable.eps_p_lab`; until the cal-file format adds a `lab_time` field, it falls back to the earliest entry in the LUT.

When the LUT contains only the lab row, `proton_eff_scale = 1.0` exactly and proton outputs are unchanged from the pre-wiring code.

### Initial guess

Stage 2's initial guess uses the alpha bump remaining after subtracting the deadtime-applied proton background:
1. Average the 5 sweeps to find the alpha peak via `get_alpha_peak_indices` (which assumes decreasing energy ordering).
2. Compute the residual `count_avg − deadtime(proton_true).mean()` at peak bins, clip to ≥ 0.
3. Gaussian fit on speed-axis $\sqrt{2 k^* q_\alpha |V| / m_\alpha}$. Floor the thermal width by the 1 eV temperature floor.
4. Density: scale a unit-density alpha model to the residual mean at peak bins; clip to $\geq 10^{-3}$ to keep $\log n$ finite.
5. $\Delta v_0 = 0$ (typical $|\Delta v| \lesssim v_A \sim 10\text{–}50$ km/s).

### Wrong-basin: signed-Δv flip

The alpha velocity is constrained to the 1-DOF line $\mathbf{v}_p^* + \Delta v \hat{\mathbf{B}}$, so the only directional ambiguity is the sign of $\Delta v$ (no 3D rotation about the spin axis as in the proton fit). After LM converges, evaluate residuals at $\Delta v \to -\Delta v$; if $\chi^2$ is lower there, re-run LM from the flipped point. Cost: 1 extra residual eval typically; 1 extra LM run for the rare basin trap.

### Uncertainty propagation

$$\Sigma_\text{stage 2} = (J^\top J)^+\quad\text{(3×3, in $(\log n_\alpha, \log T_\alpha, \Delta v)$ space)}$$
$$\sigma_{n_\alpha} = n_\alpha \sqrt{\Sigma_\text{stage 2}[0,0]}, \qquad \sigma_{T_\alpha} = T_\alpha \sqrt{\Sigma_\text{stage 2}[1,1]}, \qquad \sigma_{\Delta v} = \sqrt{\Sigma_\text{stage 2}[2,2]}$$
$$\Sigma_{\mathbf{v}_\alpha} = \Sigma_{\mathbf{v}_p} + \sigma_{\Delta v}^2 \, \hat{\mathbf{B}}\hat{\mathbf{B}}^\top$$
This **ignores proton-parameter uncertainty's effect on Stage 2 residuals**, so $\sigma_{n_\alpha}, \sigma_{T_\alpha}$ are lower bounds.

### Quality flags (alpha-specific)

- `STALE_PROTON` (= 32): Stage 1 proton fit failed (proton `bad_fit_flag != NONE`). Stage 2 returns NaN moments without trying.
- `MAG_GAP` (= 64): MAG L1D gave NaN or $|B| < 10^{-12}$ at the chunk center. Stage 2 returns NaN moments.
- `HI_CHI_SQ` (= 8): peak-finding failed or LM did not converge.

### Magnetic-field rotation

$\hat{\mathbf{B}}^\text{RTN}$ is computed per-chunk from MAG L1D's `b_dsrf` (despun spacecraft frame) using `MagL1dData.rebin_to(chunk_center, 30s)` and `get_swapi_dsrf_to_rtn` (SPICE: `IMAP_DPS → IMAP_RTN`). NaN propagation: rebin returning non-finite or near-zero magnitude → `MAG_GAP`.

### Known limitations

- **Alpha species correction unproven**: $\mathcal{A}_0^\alpha(V_\text{lab}) / \mathcal{A}_0^p(V_\text{lab}) = \varepsilon_\alpha/\varepsilon_p$ is the natural-default extension because separate alpha ABM measurements weren't taken. Validate against OMNI alpha density.
- **Voltage-independent species efficiency ratio**: `EfficiencyCalibrationTable` stores scalar-per-species-per-time; the V-dependence of $\varepsilon_\alpha/\varepsilon_p$ is not modeled.
- **Frozen-proton uncertainty propagation**: alpha $n, T$ error bars from Stage 2 do not include proton-parameter uncertainty.
- **Alpha contamination of proton fit**: Stage 2 holds protons fixed, so their fit absorbs whatever bias the alpha bump caused in Stage 1. A future joint refit will address this.
- **Field-aligned-only drift**: transient non-field-aligned drifts (e.g. during CME shocks) will be fit as $\Delta v \approx 0$ plus elevated $\chi^2$.
- **Bin-axis split**: Stage 1 inside the alpha processor uses 62 coarse bins, while the per-sweep proton L3A uses 71. Reference proton moments in the alpha product will differ slightly from the per-sweep L3A — compare per-chunk-mean of L3A vs `reference_proton_*` as a sanity diagnostic.

## References

- Rankin, J. S., McComas, D. J., et al. (2025). Solar Wind and Pickup Ion (SWAPI) Instrument on NASA's Interstellar Mapping and Acceleration Probe (IMAP). *Space Science Reviews*, 221(8), 108. https://doi.org/10.1007/s11214-025-01229-8 — SWAPI instrument paper; sign conventions and coordinate system (Step 3).
- Tsoulfanidis, N. (1995). *Measurement and Detection of Radiation* (2nd ed.). Taylor & Francis. p. 74. — Deadtime formula: $n = g / (1 - g\tau)$.