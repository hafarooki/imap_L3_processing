# SWAPI Solar Wind Moments Algorithm Notes

## Code Files

- `imap_l3_processing/swapi/l3a/science/swapi_response.py` — Instrument response model. Loads calibration tables (azimuthal transmission, central effective area, passband polynomial fits) and builds a `PassbandGrid` per ESA voltage step via `SWAPIResponse.create_passband_grid`.
- `imap_l3_processing/swapi/l3a/science/calculate_proton_solar_wind_moments.py` — Core proton fitting algorithm. `fit_solar_wind_proton_moments` implements the three-step procedure (SPICE → initial guess → Levenberg–Marquardt); `calculate_integral` is the Numba-JIT count-rate integral over elevation, azimuth, and speed.
- `imap_l3_processing/swapi/l3a/science/calculate_alpha_solar_wind_moments.py` — Alpha particle moments fitter (Stage 2 of the two-stage proton-frozen scheme). `fit_solar_wind_alpha_moments` fits $(n_\alpha, T_\alpha, \Delta v)$ with proton parameters held fixed.
- `imap_l3_processing/swapi/l3a/science/speed_calculation.py` — ESA step layout constants (`SWAPI_SCIENCE_BINS`, `SWAPI_COARSE_SWEEP_BINS`, `SWAPI_FINE_SWEEP_BINS`), k-factor constants, `esa_voltage_to_proton_speed`/`esa_voltage_to_alpha_speed` conversions, and `get_alpha_peak_indices` for locating the alpha bump in the count-rate spectrum.
- `imap_l3_processing/swapi/swapi_processor.py` and `imap_l3_processing/swapi/l3a/chunk_fits.py` — Production pipeline entry points. `SwapiProcessor` dispatches on descriptor (`proton-sw`, `alpha-sw`, `pui-he`). Each product's processing method precomputes SPICE geometry, then distributes 5-sweep chunks across fork-based multiprocessing. Module-level worker functions (`proton_chunk_worker`, `alpha_chunk_worker`, `pui_proton_chunk_worker`) receive shared state (`SWAPIResponse`, `EfficiencyCalibrationTable`, MAG data) via an initializer. `derive_velocity_angles` (in `calculate_proton_solar_wind_moments.py`) converts fitted RTN velocity to spacecraft-frame speed, clock angle, and deflection angle in the DPS frame; speed σ uses the closed-form delta method, while the two angles' σ are propagated by Monte-Carlo sampling from the fitted velocity covariance. The proton worker also writes the Sun-frame bulk velocity vector and Sun-frame scalar speed.

## TODO

- [ ] Refactor code
- [ ] Review/rewrite tests
- [ ] Validate uncertainty
- [ ] Clarify behavior of partial last chunk from `chunk_l2_data` — if the day's sweep count isn't a multiple of 5, the final group has fewer sweeps and its epoch (`sci_start_time[0] + 30s`) is off-center; decide whether to drop it, pad it, or accept the timestamp offset
- [ ] Efficiency stuff
      > Determine alpha efficiency. Maybe empirically validate alpha-species correction $\mathcal{A}_0^\alpha/\mathcal{A}_0^p = \varepsilon_\alpha/\varepsilon_p$ against OMNI alpha density (the paper does not prove this; alpha ABM measurements weren't taken)
      > Coordinate with cal-file format owner to add a `lab_time` field to the efficiency LUT.
      > Interim: `eps_p_lab` is pinned to the first row at/after 2025-11-01 because the pre-2025-11 rows in the current LUT are placeholder values (0.02348 repeated) and using them as the lab denominator drove proton density 6× too low.
      > Replace with the proper `lab_time` lookup once the LUT format gains the field (see `EfficiencyCalibrationTable.eps_p_lab`).
- [ ] Choose bad fit flags
      > Daily repointing data gaps
      > High temperature (but still report speed and maybe pressure, too)
- [ ] Handle the padded SWAPI vs unpadded MAG data temporal mismatch
- [x] Use L2 or L1D depending on what's available https://github.com/IMAP-Science-Operations-Center/imap_L3_processing/issues/13
- [ ] Investigate the degeneracy in clock angle at 8:11 on 2026-02-04

## Input Data

### L2 Science Data

The primary input for `SwapiProcessor` is SWAPI L2 coincidence count-rate data (`imap_swapi_l2_sci`). Each CDF contains time-ordered ESA sweeps with fields:

- `swp_coin_rate` — coincidence count rate (Hz) for each ESA step.
- `esa_energy` — energy-per-charge setting for each ESA step. It is related to the actual ESA voltage setting of the instrument by $V = -\texttt{esa\_energy} / k_\text{L2}$, where $k_\text{L2} = 1.93$ eV/V/e.
- `sci_start_time` — sweep start epoch (TT2000 ns)

Each 12-second ESA sweep contains **72 ESA steps** (indices 0–71). Their roles are:

| Indices | Count | Description |
|---------|-------|-------------|
| 0       | 1     | **Always discarded.** Voltage ramp-up step. |
| 1–62    | 62    | **Coarse steps.** Fixed ESA voltage steps with logarithmic spacing. |
| 63–71   | 9     | **Fine steps.** Depends on instrument mode, but usually provides a higher-resolution scan of the proton peak, using smaller voltage steps. |

To fit protons, we use both the coarse steps and the fine steps, since the fine steps usually provide extra information about the protons.
To fit alphas, we use only the coarse steps.

Occasionally, one or more fine steps will have zero ESA voltage; these are excluded from the fit.
Likewise, only steps near the proton peak are included.

For both protons and alphas, most of the steps are discarded and only the few that are near the peak are actually used.

The CDF provides these 12-second sweeps for one day per file.
 `SwapiProcessor` groups sweeps into non-overlapping 5-sweep chunks (60s cadence), the least common multiple of the spin rate (approximately 15s) and the sweep cadence (12s).
The use of 5 sweeps makes it possible to determine the bulk velocity of the solar wind.
The solar wind fitting algorithms are applied to these 5-sweep chunks individually.

### SPICE Kernels

`SwapiProcessor` requires enough SPICE kernels to be furnished to obtain RTN to SWAPI instrument coordinate rotation matrices at each measurement time.
These are used to rotate the spacecraft-frame bulk velocity into instrument coordinates for the forward model.
It also uses the spacecraft velocity to convert the bulk velocity vectors to the Sun's inertial rest frame.

To get the spice kernel for each measured coincidence rate, the measurement time must be specified.
The start time for each sweep, available from the L2 CDF, is denoted $t_\text{epoch}$.
For ESA step $i$ (0-indexed, although recall that step 0 is skipped), the measurement time is:
$$t_i = t_\text{epoch} + i \cdot \tfrac{12}{72}\,\text{s} = t_\text{epoch} + i \cdot 0.1\overline{6}\,\text{s}.$$

### MAG RTN (alpha only)

The alpha moments depend on the local magnetic field direction because the alpha-proton drift is constrained to lie along $\hat{\mathbf{B}}$.
The processor reads MAG RTN samples and derives $\hat{\mathbf{B}}^\text{RTN}$ for the alpha fit.

The dependency prefers MAG **L2** and falls back to **L1D** when no L2 file is available. MAG is required for `alpha-sw`; the processor raises `ValueError` if neither product is provided, matching SWE's dependency loader behavior. When L1D is the source, every alpha-sw chunk in the run has its `PRELIMINARY_MAG` bit set so the product can be flagged for reprocessing once L2 is available. `proton-sw` and `pui-he` do not consume MAG.

For each 5-sweep alpha chunk, the processor uses the full $60\,\text{s}$ MAG window $[\,t_\text{center} - 30\,\text{s},\; t_\text{center} + 30\,\text{s})$.
The in-window RTN samples are averaged directly, and the mean vector is normalized to produce $\hat{\mathbf{B}}^\text{RTN}$.

If $\hat{\mathbf{B}}^\text{RTN}$ cannot be computed (empty MAG window or fill values among the in-window samples), the chunk is flagged `MAG_GAP` and is assigned fill values.

## SWAPI Response Model

The coincidence count rate at ESA voltage $V$ is
$$C(V) = \sum_s \int d^3v \; v \, f^s(\mathbf{v}) \, \mathcal{A}^s(\mathbf{v}, V),$$
where $f^s$ is the VDF of species $s$ and $\mathcal{A}^s$ is the effective area.

The effective area is decomposed as
$$\mathcal{A}^s(v, \theta, \phi, V) = \mathcal{A}_0^s(V) \cdot P\!\left(\dfrac{v}{v_0^s},\, \theta,\, \phi,\, V\right) \cdot \mathcal{T}(\phi),$$
where:
- $v_0^s = \sqrt{2 k^* q^s |V| / m^s}$ is the central speed;
- $\mathcal{A}_0^s$ is the central effective area;
- $P$ is the energy-angle passband;
- $\mathcal{T}$ is the azimuthal transmission factor.

Copies of these three functions in the form of CSV files are in `instrument_team_data/swapi`.

The normalization of $\mathcal{A}_0^s$ and $P$ are aligned in terms of the value at $\theta = 0$ and $k^* \equiv 1.89$ eV/V/e, the peak $E/|V|$ at $\theta=0^\circ$ based on high-resolution SIMION simulations.
It differs from $k_\text{L2} = 1.93$, which is the $k$-factor determined approximately from lab measurements (Rankin et al. 2025).
They differ primarily due to slight inaccuracy of the beam energy and orientation in the lab measurements. 

> ![](figures/calibration_curves.png)
> *Central effective area and azimuthal transmission.* [[src]](figure_src/plot_calibration_curves.py)

$\mathcal{T}(\phi)$ and $\mathcal{A}_0^s(V)$ are 1D functions stored in simple CSV files with a uniform grid. They are interpolated linearly for practical usage. ESA voltages outside the tabulated range are clamped to the endpoint values.

![SWAPI passband and integration region at three beam energies](figures/passband_boundaries.png)
> *Example passbands.* [[src]](figure_src/plot_passband_boundaries.py)

$P$ for a given $V$ is represented as a `PassbandGrid` object.
The CSV file contains polynomial fits of $\log P$ for each ($\theta$, $v/v_0$) pixel as a function of $\log(k^* |V|)$. Open aperture ($|\phi| > 20°$) and sunglasses ($|\phi| \leq 20°$) have separate fits.

`SWAPIResponse.create_passband_grid` evaluates those fits at the requested $V$ and interpolates them onto a uniform grid: $\theta = -15°$ to $15°$ in 0.5° steps, and $v/v_0 = 0.9$ to $1.1$ in 101 points. The uniform grid is used for fast interpolation inside the integrator. Voltages outside the fitted range are clamped to the nearest endpoint.

`PassbandGrid` also stores the passband region used by the integrator. This includes per-elevation speed-ratio bounds (`min_OA_boundary`, `max_OA_boundary`, `min_SG_boundary`, `max_SG_boundary`) and a per-region elevation range (`oa_elevation_range`, `sg_elevation_range`).

The region is set by a threshold of 1% of the grid maximum. For each elevation row with at least one above-threshold cell, the speed-ratio bounds are the first speed-ratio pixels just outside the above-threshold region. Rows with no above-threshold cell are omitted.

The elevation range is anchored at the interpolated crossing where the row maximum drops below threshold. Both the speed-ratio bounds and elevation range are recomputed for every $V$, since the polynomial fits change shape with voltage.

When an integration elevation falls between stored passband-bound rows, the wider neighboring interval is used. This avoids clipping the passband between rows.

`SwapiProcessor` precomputes the grids for each L2 file before fitting the 5-sweep chunks.

## Solar Wind Model Count Rate Integral
The solar wind proton velocity distribution function (VDF) is modeled as a drifting Maxwellian. In instrument coordinates, it is parameterized by bulk velocity ($v_b, \theta_b, \phi_b$), proton temperature $T_p$ in Kelvin, and density $n$.
The VDF is given by:
$$f_p(\mathbf{v}) = f_p(v, \theta, \phi) = \frac{n}{(\sqrt{2\pi}\, v_\text{th})^3} \exp\!\left(-\frac{v^2 + v_b^2 - 2 v\, v_b \cos\alpha(\theta, \phi)}{2 v_\text{th}^2}\right),$$
where $\theta$ is elevation, $\phi$ is azimuth, $\cos\alpha = \sin\theta_b \sin\theta + \cos\theta_b \cos\theta \cos(\phi - \phi_b)$, and $v_\text{th} = \sqrt{k_B T_p/m_p}$.

Substituting into the count rate integral in spherical velocity coordinates $(v, \theta, \phi)$:
$$C(V) = \frac{n\, \mathcal{A}_0(V)}{(\sqrt{2\pi}\, v_\text{th})^3} \sum_\text{region} \int \cos\theta\, d\theta \int \mathcal{T}(\phi)\, d\phi \int v^3\, P\!\left(\tfrac{v}{v_0}, \theta\right) \exp\!\left(-\frac{v^2 + v_b^2 - 2vv_b\cos\alpha}{2v_\text{th}^2}\right) dv.$$
The $v^3\cos\theta$ factor comes from the velocity-space volume element $v^2\cos\theta\,dv\,d\theta\,d\phi$ times the particle speed $v$ in the flux term.

The sum runs over five azimuth regions: sunglasses (SG, $|\phi| \leq 20°$), and the open aperture split into a vanes-vignetting (VV) sub-region adjacent to each vane ($20° < |\phi| \leq 26°$) plus a full-transmission OA sub-region ($26° < |\phi| < 150°$).
They are integrated separately so that only one passband is used for each integral and because the integral will generally have separate peaks in each of these regions due to the vanes at $\pm 20^\circ$.

Splitting the open aperture at $\pm 26°$ anchors a Gauss-Legendre boundary at the inflection of $\mathcal{T}(\phi)$, which is identically zero at $|\phi| = 20°$ (vanes fully blocking) and rises smoothly to $\sim 1$ by $|\phi| \approx 30°$. Without the split, the steep rise sits in the interior of one wide GL window and is poorly resolved. The $26°$ cut was chosen by sweeping $23°$–$30°$ against `reference_integrals.csv` (best high-rate failure count); $27°$ puts the steepest $d\mathcal{T}/d\phi$ point right at the boundary, undoing the benefit. The constant `VV_OUTER_DEG` in `calculate_proton_solar_wind_moments.py` controls the cut.

### Angular limits

For each azimuth region, the angular cutoff $\Delta\alpha$ is chosen from the VDF angular falloff at the passband central speed $v_0$:
$$\Delta\alpha = \frac{180}{\pi}\arccos\!\left(\mathrm{clamp}\!\left(\frac{v_\text{th}^2 \ln\varepsilon}{v_0 v_b} + 1;\; -1,\; 1\right)\right),$$
with $\varepsilon_\text{SG} = \varepsilon_\text{OA} = 10^{-6}$. If the VDF is broad enough that the arccos argument would leave $[-1, 1]$, the clamp makes $\Delta\alpha = 180^\circ$.

The implementation uses this angular cutoff as a rectangular half-extent in elevation and azimuth:
$$[\theta_b - \Delta\alpha,\; \theta_b + \Delta\alpha] \times [\phi_b - \Delta\alpha,\; \phi_b + \Delta\alpha].$$
This rectangle is conservative: it contains the angular disk of radius $\Delta\alpha$ around the bulk direction.

The window is then clamped to the passband elevation range for the region and to that region's azimuth span:

| Region | Azimuth Range |
|--------|----------------|
| SG     | $[-20°,\; 20°]$ |
| VV−    | $[-26°,\; -20°]$ |
| VV+    | $[20°,\; 26°]$ |
| OA−    | $[-150°,\; -26°]$ |
| OA+    | $[26°,\; 150°]$ |

If either clamped dimension has zero width, that region is skipped.

For OA± only, the azimuth window is trimmed once more using the product of the VDF and azimuthal transmission. `_trim_oa_azimuth_by_integrand` samples 64 points of $f(v_0, \theta_b', \phi)\mathcal{T}(\phi)$ across the clamped OA azimuth window, where $\theta_b'$ is $\theta_b$ clamped into the OA elevation range. It keeps the portion above $10^{-6}$ of its maximum and expands by one sample on each side.

After this trim, OA± is skipped when the heuristic upper estimate
$$\hat{C}_\text{OA} = \mathcal{A}_0(V)\,v_0^3\,\Delta\theta\,\Delta v\,\int_{\phi_\text{lo}}^{\phi_\text{hi}} \mathcal{T}(\phi)\,g(\phi)\,d\phi$$
falls below $\max(0.1\;\text{Hz},\; 10^{-3} C_\text{SG})$. Here $g(\phi) = f(v_0, \theta_b', \phi)$, $\Delta\theta$ is the clamped OA elevation width in radians, and $\Delta v = (r_\text{max}(0) - r_\text{min}(0))v_0$ is the OA passband speed width at $\theta = 0^\circ$.

The VV± regions are not trimmed or skipped: they are bounded ($\leq 6°$ wide), $\mathcal{T}(\phi)$ is small (peak $\sim 0.05$) but rises steeply, and a few GL nodes resolve them cheaply.

### Speed limits

For each Gauss-Legendre elevation node, the speed integral only needs to cover speeds where both of these are true:

1. The VDF is non-negligible.
2. The SWAPI passband is nonzero at that elevation.

The VDF speed interval is taken to be
$$[v_b - \Delta v_\text{VDF},\; v_b + \Delta v_\text{VDF}], \qquad \Delta v_\text{VDF} = 6v_\text{th},$$
where $v_b = |\mathbf{v}_b|$ and $v_\text{th}$ is the thermal speed. For the Maxwellian VDF used here, this captures essentially all of the distribution: at $6\sigma$ the radial factor is $e^{-18} \approx 1.5 \times 10^{-8}$ of peak. A much wider window (e.g.\ the previous $10v_\text{th}$) makes Gauss-Legendre concentrate nodes far from the integrand peak for cold plasma where the passband already extends well beyond $6v_\text{th}$, and the resulting polynomial overshoots the near-delta peak by a few percent at high count rate; a much narrower one ($3v_\text{th}$) clips the Maxwellian wings and fails catastrophically when the angular geometry shifts the per-elevation speed peak off-center.

The passband speed range is stored as speed-ratio bounds relative to the central passband speed $v_0$:
$$[r_\text{min}(\theta)v_0,\; r_\text{max}(\theta)v_0].$$
Here $r_\text{min}(\theta)$ and $r_\text{max}(\theta)$ depend on both elevation and aperture region (SG or OA). They describe where the passband at that elevation remains above the integration cutoff.

The integration limits are the intersection of those two windows:
$$v_\text{lo}(\theta) = \max\!\left(v_b - \Delta v_\text{VDF},\; r_\text{min}(\theta)v_0\right),$$
$$v_\text{hi}(\theta) = \min\!\left(v_b + \Delta v_\text{VDF},\; r_\text{max}(\theta)v_0\right).$$

#### Quadrature behavior

`calculate_integral` evaluates each region as nested Gauss-Legendre quadratures in elevation, azimuth, and speed:
$$(N_\text{elev}, N_\text{az}, N_\text{speed}) = (21,\;21,\;15).$$
SG and OA use the same azimuth-node count; the OA window is already tightened by the transmission-aware trim. $N_\text{speed} = 15$ (rather than 11) is required for cold plasma: when the Maxwellian width $v_\text{th}$ is small relative to the (passband-bounded) speed window, the per-node spacing at $N_\text{speed} = 11$ is comparable to $\sigma$ and GL undersamples the peak. With the $\Delta v_\text{VDF} = 6 v_\text{th}$ window above, $N_\text{speed} = 15$ gives $\sim 0.8\sigma$ spacing — sufficient for GL convergence on the bilinear-interpolated integrand. Bumping further to $N_\text{speed} = 21$ marginally reduces the median error but does not improve the high-rate tail (the residual $\sim 2\%$ failures at $\geq 10^4$ Hz come from elevation-dimension integrand kinks, not speed quadrature).

The loop order is elevation → azimuth → speed. The implementation computes terms in the outermost loop where they are constant as much as possible.

### Integrator Validation

Representative model spectra below exercise the integrator's edges. Most configurations stay within $\sim 1.5\%$ of the reference; the cold-plasma case ($T = 11{,}605$ K, on-axis) reaches $\sim 13\%$ at the spectrum peak because the polynomial-fit passband tail outside the 1%-of-max threshold-crossing carries non-negligible signal that the production integrator deliberately truncates while the fixed-limit reference does not.

![Production vs ground-truth spectra for six representative SW configurations](figures/spectra.png)

*Generated by `docs/swapi/figure_src/plot_spectra.py`.*

The optimized integrator is validated against a high-resolution fixed-limit reference (`reference_integral_fixed_limits`) over 10000 random solar-wind configurations (`reference_integrals.csv`). Each configuration is evaluated at the ESA voltage whose central proton speed equals its `bulk_speed`. The table below summarizes the distribution of $|$ratio $-1|$ stratified by reference count rate.

<!-- BEGIN: validation_table (auto-generated by docs/swapi/figure_src/build_validation_table.py — do not edit by hand) -->
| Reference (Hz)  |     N |  Median |     95% |     99% |     Max |
|-----------------|-------|---------|---------|---------|---------|
| $< 0.1$         |  2132 | 100.00% | 100.00% | 100.00% | 100.00% |
| $0.1$ – $1$     |   235 |   5.32% |  24.60% |  36.49% |  42.88% |
| $1$ – $10$      |   325 |   2.12% |  13.17% |  20.68% |  28.64% |
| $10$ – $10^2$   |   383 |   0.98% |   7.67% |  13.32% |  16.02% |
| $10^2$ – $10^3$ |   525 |   0.55% |   3.69% |   9.31% |  13.96% |
| $10^3$ – $10^4$ |   865 |   0.30% |   1.66% |   5.02% |  11.10% |
| $10^4$ – $10^5$ |  1694 |   0.15% |   0.70% |   1.40% |   2.97% |
| $\geq 10^5$     |  3841 |   0.09% |   0.35% |   0.60% |   1.52% |
<!-- END: validation_table -->

For high-rate cases ($\geq 10^3$ Hz) where the proton fit residuals are dominated by Poisson noise rather than integrator error, $|$ratio $-1|$ stays within a few percent of unity at the 99th percentile. The $< 0.1$ Hz band is configurations where the bulk direction sits many sigma outside the FOV — both integrators round to $\sim 0$, well below the noise floor, so the ratio is meaningless and clamped to $100\%$ above. The remaining percent-level worst-cases at $\geq 10^4$ Hz cluster around bulk elevations within $\sim 1°$ of the SG passband elevation edges ($\pm 5.93°$, $-9.94°$), where the bilinear-interpolated passband has a near-cliff in the integrand that Gauss-Legendre at $N_\text{elev}=21$ does not fully resolve.

*Run `docs/swapi/figure_src/build_validation_table.py` after non-trivial integrator changes to regenerate the table in place.*

## Fitting Procedure

Given $N$ measurements $(C_i, V_i, t_i)$, the solar wind moments $(n, T, \mathbf{v}_b^\text{SC})$ are fit in three steps:
1. Obtain RTN $\rightarrow$ SWAPI rotation matrices $R_i$ from SPICE.
2. Compute an initial guess: temperature and bulk speed from a Gaussian fit to $C_i$ vs. $v_i$, with bulk velocity assumed anti-sunward with a nominal transverse offset.
3. Refine by nonlinear least squares.

The alpha particle moments are fit in a separate two-stage procedure described in [Alpha Particle Moments](#alpha-particle-moments).

### Step 1: SPICE

$R_i$ (shape $N \times 3 \times 3$) are precomputed for each measurement time (see [SPICE Kernels](#spice-kernels)).

### Step 2: Initial guess

**Temperature and speed magnitude** are obtained from a Gaussian fit to $C_i$ vs. $v_i = \sqrt{2 k^* q V_i / m_p}$:
$$C_i \approx A \exp\!\left(-\frac{(v_i - v_b)^2}{2 \sigma_v^2}\right), \qquad A, v_b, \sigma_v > 0.$$
The fitted width $\sigma_v$ is used directly as the thermal width (no passband subtraction), with a floor $\sigma_{\text{floor}, v}$ corresponding to a $\approx 11{,}600$ K temperature floor:
$$\sigma_{\text{thermal}, v} = \max(\sigma_v,\, \sigma_{\text{floor}, v}).$$
The initial temperature is $T_0 = m_p \sigma_{\text{thermal}, v}^2 / k_B$.

**Velocity direction** is initialized with $v_T = -30$ km/s and $v_N = 0$, with the radial component chosen so that the total speed matches the Gaussian-fit $v_b$:
$$v_R^{(0)} = \sqrt{\max(v_b^2 - 30^2,\; 0)}, \qquad \mathbf{v}_b^\text{SC,(0)} = (v_R^{(0)},\, -30,\, 0)\ \text{km/s}.$$
The $v_T = -30$ km/s offset is a nominal aberration-direction seed; in practice it makes no difference to the final result because the dual-LM flip check (see Wrong-basin detection below) always explores both mirror basins regardless of the initial transverse velocity. A $v_T = 0$ seed produces identical fits. Preserving $|\mathbf{v}_b^{(0)}| = v_b$ keeps the initial speed consistent with the Gaussian fit. The optimizer in Step 3 recovers the true transverse components from the small spin-phase modulation of the bulk azimuth/elevation in the instrument frame.

The initial density is scaled to match the mean observed count rate:
$$n_0 = \frac{\langle C_i \rangle}{\langle C_i^\text{model}(n=1) \rangle}.$$

Figure below shows initial-guess and final-fit accuracy across 10000 random solar wind configurations (bulk speed 300–800 km/s, temperature 23,000–580,000 K log-uniform, density 2–20 cm⁻³, $v_T, v_N \in [-50, 50]$ km/s). Synthetic count rates are produced from the forward model using the real SWAPI 71-step science voltage sweep (from the L2 CDF), 5 sweeps per fit, realistic spin geometry (spin axis = boresight, 15 s period), and Poisson noise — matching the production processor exactly. The dual-LM flip check (Step 3) ensures the optimizer always finds the correct basin regardless of the initial transverse velocity. 0 bad-fit flags across all 10000 cases. Generated by `docs/swapi/figure_src/plot_initial_guess_accuracy.py`.

![Initial-guess vs. final-optimizer accuracy for 10000 synthetic solar wind cases](figures/initial_guess_accuracy.png)

### Step 3: Optimization

Parameters $[\log n,\, \log T,\, v_R,\, v_T,\, v_N]$ (with $\mathbf{v}_b^\text{SC} = (v_R, v_T, v_N)$ in the spacecraft RTN frame) are fit by `scipy.optimize.least_squares` using the Levenberg–Marquardt algorithm (`method='lm'`, `diff_step=1e-4`) with unweighted residuals over a 10%-of-peak count-rate mask:
$$\mathcal{M} = \{i : C_i \geq 0.1 \max_j C_j\}, \qquad r_i = C_i^\text{model} - C_i\ \ \text{for}\ i \in \mathcal{M}.$$

Steps below 10% of the peak count rate are dropped: they carry essentially no proton signal (the proton peak sits in a narrow speed window — most ESA steps are noise floor and off-axis leakage), but with N ≈ 355 steps per 5-sweep fit they still contribute non-trivial residuals that can pull the moments. The 10% threshold is chosen to exclude the deep tails (PUI/alpha contamination in production data) while keeping enough steps that the spin-axis-mirror basins remain discriminable — tighter masks (e.g. FWHM at 0.5×max) leave the cold-plasma chi² landscape too noise-degenerate to pick the right basin. The mask is global across all sweeps × steps fed to one fit, and is computed once from the observed count rates only — the model never re-evaluates it. Initial-guess construction (the Gaussian curve fit and the mean-rate density scaling) is unaffected and uses all steps. If fewer than 5 steps survive the mask (fewer constraints than free parameters), the mask is dropped and all steps are fit, which is purely a guard against pathological inputs (e.g. all-zero count rates).

Density and temperature are parameterized in log-space to keep them positive throughout optimization. The optimizer's `success` flag is mapped to `bad_fit_flag`: failure sets `BAD_FIT`.

#### Wrong-basin detection (post-fit flip check)

The forward model is approximately invariant under the spin-axis mirror $(v_T, v_N) \rightarrow (-v_T, -v_N)$, broken only weakly by the SG passband elevation asymmetry ($[-10.5°, +7°]$). The two basins are *not* truly degenerate — the truth is the global minimum, and the mirror is a local minimum with $\chi^2$ typically $100$–$500\times$ higher. A single LM run can converge to either basin depending on the initial guess and noise, and once committed it stays trapped at its local minimum. The initial transverse velocity seed does not reliably control which basin LM enters — the mirror symmetry is in the instrument frame, not in RTN where the seed is specified, so an RTN-space offset does not systematically break the degeneracy.

After LM converges to $\hat{\mathbf{x}} = (\log n, \log T, \mathbf{v}_b)$, build the flipped solution by rotating the bulk velocity 180° about the spin axis ($\hat{\mathbf{s}}$ in RTN, recovered as the second row of any rotation matrix since $R_i \hat{\mathbf{s}} = \hat{\mathbf{y}}_\text{SWAPI}$ by design):
$$\mathbf{v}_b' = 2(\mathbf{v}_b \cdot \hat{\mathbf{s}})\,\hat{\mathbf{s}} - \mathbf{v}_b.$$
This works for any spin axis orientation, not just radial. Then **iteratively re-run LM from the flipped seed** $\hat{\mathbf{x}}' = (\log n, \log T, \mathbf{v}_b')$, accepting the new minimum only if $\chi^2$ improves by more than a relative tolerance ($10^{-3}$), and continue flipping the new minimum until no improvement (cap at 6 flips). A single residual evaluation at the flipped *point* is not a reliable proxy for the mirror basin's depth: the two basins can have different $(n, T)$ at their respective minima (the wrong basin tends to inflate density to compensate for the angular mismatch), so $\chi^2(\hat{\mathbf{x}}')$ — computed at the first basin's $(n, T)$ — is artificially high and hides the fact that the mirror basin's actual minimum has lower $\chi^2$ than the first.
The reason a single flip is not always enough: a poorly conditioned IG (e.g. low density + the 10%-of-peak mask saturating the IG density at its upper bound) can land LM in a third off-axis basin, whose spin-axis mirror is *also* not the truth basin. Iterating the flip walks through the connected family of basins until the deepest reachable one is found. The procedure is monotone in $\chi^2$, so it can only equal-or-improve the single-flip result; on the synthetic benchmark (1000 cases, bulk speed 300–800 km/s, $|\mathbf{v}_\perp| \le 50$ km/s), it eliminates the worst $\sim 100$ km/s mirror-basin failure that affected $\sim 0.1\%$ of cases under the single-flip rule, and reduces the 1000-case worst-case $v_T,\, v_N$ residuals from $\sim 90$–$100$ km/s to $\le 6$ km/s. Typical cost is unchanged at 2 LM runs (the second flip terminates immediately with no improvement); pathological cases need 2–3.

![χ² landscape in the (v_T, v_N) plane showing the truth and spin-axis-mirror minima](figures/wrong_basin.png)

*Generated by `docs/swapi/figure_src/plot_wrong_basin.py`. The mirror minimum has $\chi^2$ roughly $200\times$ the truth — easily distinguished by a single residual evaluation at the flipped solution.*

> **Note on `diff_step`.** The default finite-difference step in `least_squares` scales with the parameter magnitude, producing steps of $\sim 10^{-8}\ \text{km/s}$ for $v_T,\, v_N$ near zero. For cold plasma ($T \lesssim 60{,}000\ \text{K}$), the resulting count-rate perturbation falls below the GL-quadrature noise floor, making the numerical Jacobian for $v_T$ and $v_N$ pure noise and inflating the LM damping factor to $\sim 10^{13}$, which freezes all parameters. `diff_step=1e-4` is the empirical optimum over $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$: it sits above the noise floor (giving a clean Jacobian and correct convergence) while keeping the linearization error small enough not to degrade accuracy ($10^{-2}$ degrades $v_N$ RMSE; $10^{-1}$ produces bad fits and a 5× slowdown).

Inside the model, the spacecraft-frame bulk velocity is rotated into instrument coordinates:
$$\mathbf{v}_{b,i}^\text{xyz} = R_i \, \mathbf{v}_b^\text{SC}.$$
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

The optimizer returns the Jacobian $J$ of the residuals with respect to $[\log n,\, \log T,\, v_R,\, v_T,\, v_N]$ at the solution. The parameter covariance is
$$\Sigma_x = s^2\,(J^\top J)^+, \qquad s^2 = \frac{\sum_i r_i^2}{N - p},$$
where ${}^+$ is the Moore–Penrose pseudoinverse. Residuals are unweighted, so the $s^2$ scaling absorbs both measurement noise and model imperfection (non-Maxwellian features, alpha contamination, intra-window variability) — equivalent to `scipy.optimize.curve_fit` with `absolute_sigma=False`. The directly fitted scalars get
$$\sigma_n = n\,\sqrt{\Sigma_{x,00}}, \qquad \sigma_T = T\,\sqrt{\Sigma_{x,11}}.$$

The fitted velocity $\mathbf{v}_b^\text{SC}$ (RTN) is built as a 3-tuple of correlated `uncertainties.UFloat` components carrying $\Sigma_v = \Sigma_x[2{:}5,\,2{:}5]$. `derive_velocity_angles` rotates this tuple into the IMAP DPS (despun spacecraft) frame so the angles describe plasma flow relative to the spacecraft attitude:
$$\mathbf{u} = R_\text{RTN\to DPS}\,\mathbf{v}_b^\text{SC}, \qquad |\mathbf{v}| = |\mathbf{u}|, \quad \phi_c = \arctan2(u_1,\, u_0) \bmod 360°, \quad \phi_d = \arccos\!\left(-u_2/|\mathbf{u}|\right).$$
The rotation is a linear map applied with `numpy` to the object-dtype UFloat array, so correlations propagate automatically; the DPS covariance is recovered with `uncertainties.covariance_matrix` and equals $R\,\Sigma_v\,R^\top$.

**Speed σ** is propagated through `umath.sqrt(sum(x**2 for x in u_unc))` — the `uncertainties` package's automatic delta method. The linearization is essentially exact whenever $|\mathbf{u}| \gg \sigma$, which is always true for SWAPI bulk speeds vs. fitted scatter, so MC would only add sampling noise.

**Clock and deflection angle σ** are propagated by Monte Carlo. The arctan2 and arccos gradients scale as $1/u_{xy}^2$ and $1/(|\mathbf{u}|^2\,u_{xy})$ and diverge as $u_{xy} \to 0$, where $u_{xy} = \sqrt{u_0^2 + u_1^2}$. SWAPI's bulk velocity is dominated by the spin-axis component, so $u_{xy} \sim \sigma_{xy}$ and the delta method underestimates σ by tens of percent in the typical regime (see `scripts/swapi/compare_angle_propagation.py`: cold spin-aligned plasma shows +49% bias on $\sigma_{\phi_c}$ and +46% on $\sigma_{\phi_d}$, with $\sigma_{\phi_c}$ exceeding the uniform-distribution bound of $\approx 104°$). Instead, we draw `N_VELOCITY_ANGLE_MC_SAMPLES = 1000` samples
$$\mathbf{u}_i \sim \mathcal{N}(\mathbf{u},\, \Sigma_\text{DPS}),$$
recompute $(\phi_c^{(i)}, \phi_d^{(i)})$ per sample, and take the sample standard deviation. Clock-angle σ uses residuals wrapped to $(-180°,\, 180°]$ relative to the nominal $\phi_c$ so the $0°/360°$ branch cut doesn't inflate the spread; deflection σ is the plain sample std (with the arccos argument clipped to $[-1,\,1]$ to absorb numerical overshoots from samples just outside the unit-direction shell). The RNG is seeded per call (`np.random.default_rng(0)`) so outputs are deterministic.

When $\Sigma_\text{DPS}$ is non-finite (failed fit), all three σ are NaN.

### Inertial bulk velocity and speed

The optimizer returns $\mathbf{v}_b^\text{SC}$ in the spacecraft RTN frame. To recover the plasma velocity in the sun's inertial rest frame the spacecraft velocity is added back:
$$\mathbf{v}_b^\text{sun} = \mathbf{v}_b^\text{SC} + \mathbf{v}_\text{sc}^\text{RTN},$$
where $\mathbf{v}_\text{sc}^\text{RTN}$ (km/s) is obtained at the chunk center epoch directly from `imap_state(et, IMAP_RTN)` — SPICE's underlying `sxform`-based 6D state transform produces the kinematic velocity in the dynamic RTN frame (i.e. it includes the rotation-rate term of the rotating frame), so no separate rotation step is applied.
This 3-vector is stored as `proton_sw_bulk_velocity_rtn_sun` (shape $N \times 3$, units km/s) in the proton L3A CDF. Its covariance is stored as `proton_sw_bulk_velocity_rtn_sun_covariance`. Since $\mathbf{v}_\text{sc}^\text{RTN}$ is SPICE-derived and treated as exact, the Sun-frame vector covariance is the fitted spacecraft-frame velocity covariance:
$$\Sigma_v^\text{sun} = \Sigma_v^\text{SC}.$$

The scalar CDF variable `proton_sw_speed` remains the magnitude of the fitted spacecraft-frame velocity. The separate scalar variable `proton_sw_speed_sun` is the magnitude of the Sun-frame vector:
$$v_\text{sun} = \left|\mathbf{v}_b^\text{SC} + \mathbf{v}_\text{sc}^\text{RTN}\right|.$$
Its uncertainty, `proton_sw_speed_sun_uncert`, is propagated with the `uncertainties` package from the correlated fitted velocity components in `result.bulk_velocity_rtn`, after adding the exact spacecraft-velocity offset:
$$\sigma_{v_\text{sun}} = \mathrm{std}\!\left(\sqrt{\sum_j \left(v_{b,j}^\text{SC} + v_{\text{sc},j}^\text{RTN}\right)^2}\right).$$
Equivalently, this is the first-order Gaussian propagation $\sqrt{\mathbf{g}_\text{sun}^\top \Sigma_v \mathbf{g}_\text{sun}}$ with $\mathbf{g}_\text{sun} = \mathbf{v}_b^\text{sun}/|\mathbf{v}_b^\text{sun}|$, but the implementation uses the correlated `UFloat` components directly rather than recomputing from `bulk_velocity_rtn_covariance()`.

## Alpha Particle Moments

The alpha solar wind moments fitter (`calculate_alpha_solar_wind_moments.py`) reuses the proton forward model (`_model_count_rates`) and adds a 3-DOF Levenberg–Marquardt fit over $(n_\alpha, T_\alpha, \Delta v)$ where alphas are constrained to drift along the local magnetic field:
$$\mathbf{v}_\alpha = \mathbf{v}_p^* + \Delta v \, \hat{\mathbf{B}}.$$
This encodes the observed solar-wind fact that non-field-aligned differential drift is quenched by firehose/mirror instabilities.

**Sign convention.** $\Delta v > 0$ means the alpha drifts along $+\hat{\mathbf{B}}$ relative to protons. Since MAG provides $\mathbf{B}$ with a physical polarity, $\Delta v$'s sign is interpretable.

### Two-stage strategy

The pipeline processes 5-sweep chunks (matching the existing alpha LUT cadence). Each chunk is sliced to the **62 coarse-sweep steps** (`SWAPI_COARSE_SWEEP_BINS`) and flattened over (sweep, step) to a 310-element axis. The 9 fine-sweep steps (63–71) are dropped because they cluster near the proton peak and inform proton thermal width only — useless when protons are held fixed.

- **Stage 1**: Re-run `fit_solar_wind_proton_moments` on the 310-element axis to get $(n_p^*, T_p^*, \mathbf{v}_p^{*\,\text{RTN}})$. This is independent of the per-sweep proton L3A product (which uses 71 steps) and is persisted as `reference_proton_*` fields in the alpha CDF.
- **Stage 2**: The initial guess (`_alpha_initial_guess`) identifies the alpha peak steps via `get_alpha_peak_indices`. All per-measurement arrays are then subset to only those peak steps across all sweeps, so the Levenberg–Marquardt fit targets the alpha bump rather than the proton-dominated tails (which create an $n_\alpha{\downarrow}/T_\alpha{\uparrow}$ degeneracy). The residual axis for Stage 2 is therefore `n_sweeps × len(peak_steps)`, not the full 310. The combined observed model is
  $$C_\text{obs}(V) = \text{deadtime}\!\left(C_p^\text{true}(V; \theta_p^*) + C_\alpha^\text{true}(V; \theta_\alpha)\right),$$
  so deadtime acts on the sum (important in proton-peak steps where alphas are negligible but deadtime is at its largest). Stage 2 reuses the SPICE rotation matrices computed for Stage 1.

Joint refit of both species is deferred (Stage 2 does not resolve alpha contamination of the proton fit).

### Species-dependent pieces

The same `PassbandGrid` infrastructure works for both species — the grid is V-only (passband shape depends only on voltage, not species), so `SWAPIResponse.create_passband_grid(V)` is cached by `float(V)` and shared between proton and alpha fits at the same ESA voltage. `SwapiProcessor` calls `SWAPIResponse.warm_cache(data.energy / SWAPI_L2_K_FACTOR)` in the parent process before each `ProcessPoolExecutor`, so the ~1.8 ms pandas pivot inside `_build_passband_array` is paid once per unique voltage in the parent rather than once per worker; under `fork`, children inherit the populated cache and the bulk numpy buffers stay shared via copy-on-write. Species-dependent quantities — central speed $v_0^s$ and scaled central effective area — are computed separately and passed alongside the grid to `calculate_integral`. At the same $V$,
$$\frac{v_0^\alpha}{v_0^p} = \sqrt{\frac{q_\alpha m_p}{q_p m_\alpha}} = \sqrt{\frac{2 m_p}{4 m_p}} = \frac{1}{\sqrt 2}.$$
Thermal speed uses Boltzmann's constant with temperature in Kelvin:
$$v_{th}^\alpha = \sqrt{\frac{k_B T_\alpha}{m_\alpha}}.$$

### Effective-area scaling

`EfficiencyCalibrationTable` stores **absolute** detection efficiencies for each species ($\sim 0.11$ for protons, $\sim 0.15$ for alphas). The lab-calibrated `central_effective_area(V)` table is the proton ABM-derived $\mathcal{A}_0^p(V_\text{lab})$. At the integration site we apply a per-species ratio to convert this to the runtime species/time effective area:
$$\text{proton}: \quad \texttt{central\_effective\_area\_scale} = \frac{\varepsilon_p(t)}{\varepsilon_p(t_\text{lab})}$$
$$\text{alpha}: \quad \texttt{central\_effective\_area\_scale} = \frac{\varepsilon_\alpha(t)}{\varepsilon_p(t_\text{lab})}$$
The proton-lab denominator is used for **both** species so the alpha scale folds the species correction $\mathcal{A}_0^\alpha/\mathcal{A}_0^p \approx \varepsilon_\alpha/\varepsilon_p$ together with alpha time drift into a single ratio. This factor is exposed as `EfficiencyCalibrationTable.eps_p_lab`; until the cal-file format adds a `lab_time` field, it falls back to the earliest entry in the LUT.

When the LUT contains only the lab row, `proton_eff_scale = 1.0` exactly and proton outputs are unchanged from the pre-wiring code.

### Initial guess

Stage 2's initial guess (`_alpha_initial_guess`) locates the alpha bump by subtracting the deadtime-applied proton background from the sweep-averaged count rate:

1. Reshape data into `(n_sweeps, n_steps)` via `_infer_sweep_layout`. Average the 5 sweeps: `count_avg`, `proton_bg_avg` (deadtime-corrected proton model per sweep, then averaged).
2. Convert voltages to energies: $E_i = k^* |V_i|$.
3. Call `get_alpha_peak_indices(count_avg, energies)` to locate the alpha peak. This function finds the proton peak first, then walks backward (toward higher energies / lower indices) to where counts start increasing again (start of the alpha bump), masks steps above this start and below $4\times$ the proton peak energy, and returns the alpha peak slice.
4. Guard: require $\geq 3$ steps in the peak and at least one step with positive residual $C_i - R_i^p > 0$.
5. Gaussian fit on the residual $\max(C_i - R_i^p, 0)$ vs. alpha speed $v_i^\alpha = \sqrt{2 k^* (q_\alpha/m_\alpha) |V_i|}$ at peak steps. Yields bulk speed $v_b^\alpha$ and thermal width $\sigma_v$. Floor $\sigma_v$ by the temperature floor ($\approx 11{,}600$ K equivalent for alpha mass).
6. Density: compute a unit-density alpha forward model at $\Delta v = 0$ (using the proton bulk velocity as the alpha velocity seed), average across sweeps, and scale to match the mean residual at the peak:
   $$n_{\alpha,0} = \max\!\left(\frac{\overline{(C_i - R_i^p)_\text{peak}}}{\overline{R_i^{\alpha,\text{unit}}}},\; 10^{-3}\right)$$
7. Return $(n_{\alpha,0}, T_\alpha, \Delta v = 0, \text{peak\_step\_indices})$. The optimizer starts with $\Delta v = 0$ and the wrong-basin flip (below) handles sign ambiguity. The returned `peak_step_indices` are used to subset the residual axis for Stage 2 (see above).

The figure below shows these steps on three real L2 spectra from `imap_swapi_l2_sci_20260101`. Top row: 5-sweep-averaged observed count rate (blue dots) vs the frozen proton model (orange), with the detected alpha peak region shaded green. Bottom row: residual (observed − proton model) at all steps (grey) and the peak steps (green circles), with the Gaussian fit (red) that yields the initial $v_b^\alpha$.

![Alpha peak-finding on real L2 spectra](figures/alpha_peak_finding.png)

*Generated by `docs/swapi/figure_src/plot_alpha_peak_finding.py`.*

### Wrong-basin detection ($\Delta v$ flip)

The 1-DOF $\Delta v$ parameterization creates a basin ambiguity along $\hat{\mathbf{B}}$: flipping $\Delta v \to -\Delta v$ can yield a comparable $\chi^2$ when the alpha bump sits near the proton thermal tail. After LM converges to $(\log n_\alpha, \log T_\alpha, \Delta v)$, the fit evaluates $\chi^2$ at the flipped point $(\log n_\alpha, \log T_\alpha, -\Delta v)$. If $\chi^2_{\text{flipped}} < \chi^2_{\text{LM}}$, LM re-runs from the flipped point. This costs one extra residual evaluation typically, plus one extra LM run for the cases that need it.

Unlike the proton wrong-basin check (which always re-runs LM from the flipped seed), the alpha flip uses a cheaper evaluate-then-rerun strategy because the 1-DOF flip preserves $(n_\alpha, T_\alpha)$ — unlike the proton case, where the 3-DOF velocity flip changes the $(n, T)$ landscape significantly enough that a single-point $\chi^2$ is a poor proxy.

### Uncertainty propagation

$$\Sigma_\text{stage 2} = s^2\,(J^\top J)^+\quad\text{(3×3, in $(\log n_\alpha, \log T_\alpha, \Delta v)$ space)}, \qquad s^2 = \frac{\sum_i r_i^2}{N - p}$$
$$\sigma_{n_\alpha} = n_\alpha \sqrt{\Sigma_\text{stage 2}[0,0]}, \qquad \sigma_{T_\alpha} = T_\alpha \sqrt{\Sigma_\text{stage 2}[1,1]}, \qquad \sigma_{\Delta v} = \sqrt{\Sigma_\text{stage 2}[2,2]}$$
$$\Sigma_{\mathbf{v}_\alpha} = \Sigma_{\mathbf{v}_p} + \sigma_{\Delta v}^2 \, \hat{\mathbf{B}}\hat{\mathbf{B}}^\top$$
This **ignores proton-parameter uncertainty's effect on Stage 2 residuals**, so $\sigma_{n_\alpha}, \sigma_{T_\alpha}$ are lower bounds.

### Quality flags (alpha-specific)

- `STALE_PROTON` (= 32): Stage 1 proton fit failed (proton `bad_fit_flag != NONE`). Stage 2 returns fill-valued moments without trying.
- `BAD_FIT` (= 8): fit was attempted with valid inputs but failed — reference proton velocity is nonphysical, peak-finding failed, or optimizer did not converge.
- `EPHEMERIS_GAP` (= 4): SPICE could not provide rotation matrices for the chunk's measurement times. The chunk is fill-valued without attempting a fit.
- `MAG_GAP` (= 128): SPICE geometry was available but MAG data is missing or contains fill values across the chunk window. The alpha fit is skipped and moments are fill-valued.
- `PRELIMINARY_MAG` (= 64): MAG L1D was used as the source for this run (L2 was unavailable). Set on every chunk in the run. The product is a candidate for reprocessing once MAG L2 covers the time range.

### Known limitations

- **Alpha species correction unproven**: $\mathcal{A}_0^\alpha(V_\text{lab}) / \mathcal{A}_0^p(V_\text{lab}) = \varepsilon_\alpha/\varepsilon_p$ is the natural-default extension because separate alpha ABM measurements weren't taken. Validate against OMNI alpha density.
- **Voltage-independent species efficiency ratio**: `EfficiencyCalibrationTable` stores scalar-per-species-per-time; the V-dependence of $\varepsilon_\alpha/\varepsilon_p$ is not modeled.
- **Frozen-proton uncertainty propagation**: alpha $n, T$ error bars from Stage 2 do not include proton-parameter uncertainty.
- **Alpha contamination of proton fit**: Stage 2 holds protons fixed, so their fit absorbs whatever bias the alpha bump caused in Stage 1. A future joint refit will address this.
- **Field-aligned-only drift**: transient non-field-aligned drifts (e.g. during CME shocks) will be fit as $\Delta v \approx 0$ plus elevated $\chi^2$.
- **Step-axis split**: Stage 1 inside the alpha processor uses 62 coarse steps, while the per-sweep proton L3A uses 71. Reference proton moments in the alpha product will differ slightly from the per-sweep L3A — compare per-chunk-mean of L3A vs `reference_proton_*` as a sanity diagnostic.

## References

- Rankin, J. S., McComas, D. J., et al. (2025). Solar Wind and Pickup Ion (SWAPI) Instrument on NASA's Interstellar Mapping and Acceleration Probe (IMAP). *Space Science Reviews*, 221(8), 108. https://doi.org/10.1007/s11214-025-01229-8 — SWAPI instrument paper; sign conventions and coordinate system (Step 3).
- Tsoulfanidis, N. (1995). *Measurement and Detection of Radiation* (2nd ed.). Taylor & Francis. p. 74. — Deadtime formula: $n = g / (1 - g\tau)$.
