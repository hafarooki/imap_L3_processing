
# SWAPI Response Model

## Effective Area Function

The coincidence count rate for ESA voltage setting $`V`$ is modeled as
```math
C(V) = \sum_{s} \int d^3v \thinspace v \thinspace f^{s}(\mathbf{v}) \thinspace \mathcal{A}^{s}(\mathbf{v}, V),
```
where $`f^{s}`$ is the VDF of species $`s`$ and $`\mathcal{A}^{s}`$ is the effective area function.

In instrument coordinates (speed $v$, elevation angle $\theta$, azimuth angle $\phi$; [Rankin et al. 2025](https://doi.org/10.1007/s11214-025-01229-8)), $`\mathcal{A}^{s}`$ is decomposed as
```math
\mathcal{A}^{s}(v, \theta, \phi, V) = \mathcal{A}_{0}^{s}(V) \cdot P_{\text{region}(\phi)}\negthinspace \left(\dfrac{v}{v_{0}^{s}},\thinspace \theta,\thinspace V\right) \cdot T(\phi),
```
where:
- $`v_{0}^{s} = \sqrt{2 k^{\ast} q^{s} |V| / m^{s}}`$ is the central speed;
- $`\mathcal{A}_{0}^{s}`$ is the central effective area;
- $`\text{region}(\phi)`$ is `SG` for $|\phi| \leq 20^\circ$, `OA` otherwise;
- $`P_{r}`$ is the region-specific energy-angle passband;
- $`T`$ is the azimuthal transmission factor.

CSV versions of these three functions are in `instrument_team_data/swapi`.
The production code loads them from ancillary files.
They are processed and cached in the `SwapiResponse` class.

The normalizations of $`\mathcal{A}_{0}^{s}`$ and $`P_{r}`$ are aligned in terms of the value at $`\theta = 0^\circ`$ and $`k^{\ast} \equiv 1.89`$ eV/V/e, the peak $`(E/q)/|V|`$ at $`\theta=0^{\circ}`$ based on high-resolution SIMION simulations.
$`k^{\ast}`$ differs from $`k_{\text{L2}} = 1.93\,\text{eV/V/e}`$, which is the $`k`$-factor estimated pre-launch from lab measurements ([Rankin et al. 2025](https://doi.org/10.1007/s11214-025-01229-8)) and used to convert from ESA energy in the L2 CDF files to the actual ESA voltage of the instrument.
$k^{\ast}$ and $k_{\text{L2}}$ differ primarily because of small inaccuracies in the beam energy and orientation in the lab measurements.

## Central Effective Area and Azimuthal Transmission

`SwapiResponse.from_files` loads the azimuthal transmission and central effective area calibration curves from ancillary CSV files.
$`T(\phi)`$ is tabulated as a function of $`|\phi|`$ from $`0^\circ`$ to $`180^\circ`$ at $`0.1^\circ`$ spacing.
Missing transmission entries are treated as zero, and the interpolator uses $`|\phi|`$ after wrapping azimuth into $`[-180^\circ, 180^\circ)`$.
The flat portions of the azimuth response are explicitly included in the interpolator for performance: $`T=10^{-3}`$ for the central sunglasses region ($`|\phi| \le 9^\circ`$), and $`T=1`$ across the main open aperture range ($`31^\circ \le |\phi| \le 115^\circ`$).

The central effective area CSV gives the baseline proton response $`\mathcal{A}_{0,\text{(lab)}}^{\text{H}^+}(V)`$ as a function of ESA voltage.
`SwapiResponse` evaluates this curve by linear interpolation in $`|V|`$ and uses the nearest tabulated value outside the CSV range.

> ![](figures/calibration_curves.svg)
> *Central effective area and azimuthal transmission.* [[src]](figure_src/plot_calibration_curves.py)

$`\mathcal{A}_{0,\text{(lab)}}^{\text{H}^+}(V)`$ describes the central effective area for protons at the beginning of the mission.
The detector efficiency degrades with time and is species-dependent.
Therefore, an efficiency calibration table (see `EfficiencyCalibrationTable`) is used to scale the central effective area.

The efficiency calibration table has two columns: $\varepsilon_\text{H}$ for hydrogen and $\varepsilon_\text{He}$ for helium.
They are used as follows:

**Protons ($`\text{H}^+`$)**
```math
\mathcal{A}_{0}^{\text{H}^+}\!(V) = \mathcal{A}_{0,\text{(lab)}}^{\text{H}^+}\!(V)\, \varepsilon_\text{H}(t),
```
where $`\varepsilon_\text{H}(t)`$ is the most recent entry whose timestamp precedes $t$.

**Alphas ($`\text{He}^{++}`$) & PUIs ($`\text{He}^{+}`$)**
```math
\mathcal{A}_{0}^{\text{He}^{++}}\!(V)
= \mathcal{A}_{0}^{\text{He}^{+}}\!(V)
= \mathcal{A}_{0,\text{(lab)}}^{\text{H}^+}\!(V)\, \varepsilon_\text{He}(t)
```

Initially, the hydrogen column is set to $1$, and the helium column is set to $1.05$.
The ratio of $1.05$ for helium comes from the high-energy limit observed in the lab for $\text{He}^+$ versus $\text{H}^+$.
Above a few keV per charge, the ratio was observed to be consistently $1.05$.
Since both solar wind alphas and pickup ions tend to be above that threshold, the increase of the ratio at low energies has not been accounted for.
It has also been assumed that $\text{He}^+$ and $\text{He}^{++}$ have the same efficiency based on theoretical expectations and preliminary analyses of SWAPI's measurements in space.

## Energy-Angle Passbands

> ![SWAPI passband and integration region at three beam energies](figures/passband_boundaries.svg)
> *Examples of interpolated passbands with integration limits.* [[src]](figure_src/plot_passband_boundaries.py)

The passband coefficient CSV is indexed by `region`, `energy_ratio`, and `elevation`, where `energy_ratio` is $`(E/q)/|V|`$ (particle energy per charge versus ESA voltage magnitude).
For each pixel, the coefficients $c$ give a polynomial fit to $`\ln P`$ as a function of $`\ln(k^{\ast}|V|)`$.
The current ancillary file contains quadratic coefficients, ordered from highest degree to constant term.
In the future, a higher-order polynomial may be used.
When initializing the passband cache for a given $|V|$, for each region, `SwapiResponse` clamps $`|V|`$ to the  `min_esa_voltage`/`max_esa_voltage` limits from the CSV for a given region, then evaluates
```math
P = \exp\left[\operatorname{polyval}\left(c, \ln(k^{\ast}|V_{\mathrm{clamped}}|)\right)\right].
```
The `energy_ratio` axis is converted to speed ratio using
```math
\dfrac{v}{v_{0}^{s}} = \sqrt{\dfrac{(E/q)/|V|}{k^{\ast}}},
```
and the speed ratio and elevation axes are interpolated to a uniform grid, enabling efficient index-based interpolation when fitting.

After interpolating to the uniform grid, the passband is divided by its value at $`(v/v_0=1,\thinspace \theta=0)`$ so that $`P_{r}(1, 0, V) = 1`$ by construction.

The `PassbandGrid` class stores the interpolated values for a given $V$ and $\text{region}(\phi)$ in addition to an integration contour of 1% relative to the maximum value.
