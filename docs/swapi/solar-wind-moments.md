# SWAPI Solar Wind Moments Algorithm Notes

## Model

The coincidence count rate at ESA voltage $V$ is
$$C(V) = \sum_s \int d^3v \; v \, f^s(\mathbf{v}) \, \mathcal{A}^s(\mathbf{v}, V),$$
where $f^s$ is the VDF of species $s$ and $\mathcal{A}^s$ is the effective area (see `docs/swapi/SWAPI Effective Area Function.pdf`).

The effective area is decomposed as
$$\mathcal{A}^s(v, \theta, \phi, V) = \mathcal{A}_0^s(V) \cdot P^s\!\left(\tfrac{v}{v_0^s},\, \theta,\, \phi,\, V\right) \cdot T(\phi),$$
where $v_0^s = \sqrt{2 k^* q^s |V| / m^s}$ is the central speed ($k^* = 1.89$ eV/V), $\mathcal{A}_0^s(V)$ is the central effective area (from lab measurements, interpolated at each $V$), $P^s$ is the energy-angle passband (from SIMION, separate sunglasses/open-aperture grids, normalized to $P(1, 0°, V) = 1$), and $T(\phi)$ is the azimuthal transmission factor ($T \approx 10^{-3}$ at $|\phi| < 9°$).

$P^s$ is voltage-dependent because the ESA's acceptance varies with beam energy. SIMION results at discrete energies are stored as polynomial fits in $\log(k^* |V|)$ and evaluated at runtime. For each $V_i$, `SWAPIResponse.create_passband_grid` evaluates these fits, resamples onto a uniform (elevation, speed-ratio) grid, and fits degree-5 polynomials to the passband speed limits as a function of elevation. The result is stored in a `PassbandGrid` struct, precomputed once per ESA step before optimization.

The solar wind proton VDF is a drifting Maxwellian:
$$f_p(\mathbf{v}) = \frac{n}{(\sqrt{2\pi}\, v_\text{th})^3} \exp\!\left(-\frac{v^2 + v_b^2 - 2 v\, v_b \cos\alpha}{2 v_\text{th}^2}\right),$$
where $\cos\alpha = \sin\theta_b \sin\theta + \cos\theta_b \cos\theta \cos(\phi - \phi_b)$ and $v_\text{th} = \sqrt{T/m_p}$ ($T$ in energy units).

Substituting into the count rate integral in spherical velocity coordinates $(v, \theta, \phi)$:
$$C(V) = \frac{n\, \mathcal{A}_0(V)}{(\sqrt{2\pi}\, v_\text{th})^3} \sum_\text{region} \int \cos\theta\, d\theta \int T(\phi)\, d\phi \int v^3\, P\!\left(\tfrac{v}{v_0}, \theta\right) \exp\!\left(-\frac{v^2 + v_b^2 - 2vv_b\cos\alpha}{2v_\text{th}^2}\right) dv.$$

The sum runs over three azimuth regions: sunglasses (SG, $|\phi| \leq 20°$), left open aperture (OA, $-150° < \phi < -20°$), and right open aperture ($20° < \phi < 150°$).

**Angular limits.** Integration limits are centered on $(\theta_b, \phi_b)$ and clipped to the FOV. The half-width is where the Maxwellian drops below $\varepsilon$:
$$\Delta\alpha = \arccos\!\left(\frac{v_\text{th}^2 \ln\varepsilon}{v_0 v_b} + 1\right),$$
with $\varepsilon = 10^{-3}$. Azimuth FOV bounds: $[-20°, 20°]$ (SG) or $[\pm 20°, \pm 150°]$ (OA). Elevation FOV bounds are derived per region from the passband grid in `create_passband_grid`: each is the range of grid rows where the passband is nonzero, extended by one elevation step on each side to include the first zero.

**Speed limits.** At each elevation the passband's speed acceptance is bounded by polynomials $r_\text{min}(\theta)$, $r_\text{max}(\theta)$ (fit per voltage in `create_passband_grid` from the nonzero grid columns, extended by one speed-ratio step on each side to include the first zero), giving $v \in [v_0\, r_\text{min}(\theta),\; v_0\, r_\text{max}(\theta)]$, intersected with $[v_b - 5v_\text{th},\; v_b + 5v_\text{th}]$.

The integral is evaluated in the order shown (elevation $\to$ azimuth $\to$ speed) for memory locality, using the trapezoid rule with $N = 31$ uniformly spaced points per dimension, and JIT-compiled with Numba. See `calculate_integral(PassbandGrid, SWParams)` in `calculate_proton_solar_wind_moments.py`.

## Fitting Procedure

Given $N$ measurements $(C_i, V_i, t_i)$, the solar wind moments $(n, T, \mathbf{v}_b^\text{RTN})$ are fit in three steps:
1. Obtain RTN $\rightarrow$ SWAPI rotation matrices $R_i$ and spacecraft velocity $\mathbf{v}_\text{sc}^\text{RTN}$ from SPICE.
2. Compute an initial guess from a Gaussian fit to $C_i$ vs. $v_i$.
3. Refine by nonlinear least squares.
- **TODO** alphas

### Step 1: SPICE

$R_i$ (shape $N \times 3 \times 3$) are obtained from `get_rotation_matrix(IMAP_RTN, IMAP_SWAPI)` at each measurement time. TT2000 nanoseconds are converted to ET via $t^\text{ET} = \text{unitim}(t / 10^9,\, \text{TT},\, \text{ET})$.

$\mathbf{v}_\text{sc}^\text{RTN}$ (km/s) is obtained at the median time from `imap_state` in ECLIPJ2000 and rotated into RTN:
$$\mathbf{v}_\text{sc}^\text{RTN} = M_{\text{ECL} \rightarrow \text{RTN}} \, \mathbf{v}_\text{sc}^\text{ECL}.$$

### Step 2: Initial guess

A Gaussian is fit to $C_i$ as a function of the central speed $v_i = \sqrt{2 k^* q V_i / m_p}$:
$$C_i \approx A \exp\!\left(-\frac{(v_i - v_b)^2}{2 v_\text{th}^2}\right), \qquad A, v_b, v_\text{th} > 0.$$
The fitted width gives the initial temperature $T_0 = m_p v_\text{th}^2$ and the bulk velocity is taken as anti-sunward: $\mathbf{v}_b^\text{RTN} = (v_b, 0, 0)$. The initial density is scaled to match the mean observed count rate:
$$n_0 = \frac{\langle C_i \rangle}{\langle C_i^\text{model}(n=1) \rangle}.$$

### Step 3: Optimization

Parameters $[\log n,\, \log T,\, v_R,\, v_T,\, v_N]$ (with $\mathbf{v}_b^\text{RTN} = (v_R, v_T, v_N)$ in the inertial RTN frame) are fit by `scipy.optimize.least_squares` using the Levenberg–Marquardt algorithm (`method='lm'`, default tolerances) with residuals
$$r_i = \frac{C_i^\text{model} - C_i}{\sqrt{\max(C_i,\, 1)}}.$$
Density and temperature are parameterized in log-space to keep them positive throughout optimization. The optimizer's `success` flag is mapped to `bad_fit_flag`: failure sets `HI_CHI_SQ`.
Inside the model, the spacecraft velocity is subtracted and the result rotated into instrument coordinates:
$$\mathbf{v}_{b,i}^\text{xyz} = R_i (\mathbf{v}_b^\text{RTN} - \mathbf{v}_\text{sc}^\text{RTN}).$$
The azimuth and elevation angles fed to the integral are then
$$\phi_{b,i} = \operatorname{arctan2}(-v_{b,i,x},\, -v_{b,i,y}), \qquad \theta_{b,i} = \arcsin\!\left(-\frac{v_{b,i,z}}{v_b}\right).$$
(Sign conventions and coordinate system follow the instrument paper.)

**Deadtime correction.** The detector is non-paralyzable with deadtime $\tau = 183.7\ \text{ns}$ and a per-bin sample time of $t_\text{sample} = 0.145\ \text{s}$. The model count rate $C^\text{model}$ (true rate) is forward-corrected before computing residuals:
$$C^\text{observed}_i = \frac{C^\text{model}_i}{1 + (\tau / t_\text{sample})\, C^\text{model}_i}.$$

> TODO verify this is correct