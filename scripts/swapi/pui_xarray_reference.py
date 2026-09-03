r"""simple_pui_demo.ipynb

# SWAPI He+ PUI forward model (xarray + pint)

## Model Specification

### Distribution Function

The generalized filled-shell distribution model for PUIs for pickup ions at position $(r, \psi)$ in the solar inertial frame is given by
$$
f_\text{PUI}\!\left( r, \psi, w_k \right)
    = \frac{\alpha_{\text{PUI},k}}{4\pi}
      \cdot
      \frac{\beta_{E,k} r_E^2}{r u_\text{sw} v_{b,k}^3}
      \cdot
      w_k^{\alpha_{\text{PUI}, k} - 3}
      \cdot
      n_k (r w_k^{\alpha_{\text{PUI},k}}, \psi)
      \cdot
      \Theta(1 - w_k),
$$
with free parameters:
* Cutoff speed $v_{b,k}$.
* Cooling index $\alpha_{\text{PUI},k}$.
* Ionization rate $\beta_{E,k}$.
* An additive, energy-independent background coincidence rate.

and additional terms:
* $u_\text{sw}$ is the solar wind speed in the inertial frame.
* $w_k \equiv v'/v_{b,k}$, where $v' \equiv \|\mathbf{v} - \mathbf{v}_\text{sw}\|$ is the speed in the solar wind frame.
* $n(r, \psi)$ is the density of interstellar neutrals, precomputed into a lookup table using the hot model (Thomas 1978).
* $r_E \approx 1\,\text{au}$


### Forward Model

The coincidence rate contribution from pickup ions is
$$
C_\text{PUI}(V) = \int \text{d}^3v \thinspace  v \thinspace  f_\text{PUI}(\mathbf{v})  \mathcal{A}^{s}(\mathbf{v}, V).
$$

$\mathcal{A}^{s}$ is specified in instrument coordinates ($\theta, \phi, v/v_0(V)$), so we must integrate over those coordinates using $\mathrm{d}^3v = v^2 \cos\theta \, \mathrm{d} v \, \mathrm{d} \theta \, \mathrm{d} \phi$:
$$
C_\text{PUI}(V) = \int \text{d}v \, \text{d}\phi \, \text{d}\theta \, v^3 \cos\theta f_\text{PUI}(\mathbf{v}) \mathcal{A}^{s}(\mathbf{v}, V).
$$

For each ($\theta, \phi$), one can construct unit vector $\hat{\mathbf{v}}(\theta, \phi)$ to specify $\mathbf{v} = v \hat{\mathbf{v}}$.
With $\mathbf{v}$ and $\mathbf{v}_\text{sw}$ provided in the same coordinate system, and the model's free parameters specified, it is then straightforward to calculate all of the variables needed for the model.

### Coordinates and notation

The integration uses SWAPI instrument coordinates $(\phi, \theta, \xi)$ where $\phi$ is azimuth, $\theta$ is elevation, and $\xi \equiv v/v_0(V)$.

The velocity unit vector in SWAPI's Cartesian coordinate system is


$$
\hat{\mathbf{v}}(\theta, \phi)
  = \begin{pmatrix}
      -\cos\theta \sin\phi \\
      -\cos\theta \cos\phi \\
      -\sin\theta
    \end{pmatrix}
$$
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_xarray  # noqa: F401
import xarray as xr

from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path

OUTPUT_PATH = get_test_data_path("swapi/pui_count_rate_reference.csv")

ureg = pint.UnitRegistry(force_ndarray_like=True)
ureg.define("counts = []")  # pint has no built-in `counts` unit
pint.set_application_registry(ureg)
Q = ureg.Quantity

K_FACTOR = Q(1.89, "eV/V/e")  # E_beam = K · |V|; eV carries the elementary charge
ONE_AU = Q(1.0, "au")
HE_PLUS_MASS_PER_CHARGE = Q(4.0, "m_p/e")  # He+ nucleus, q = +1 e
HELIUM_EFFICIENCY_RATIO = 1.05

"""## Load response files"""

central_effective_area = HELIUM_EFFICIENCY_RATIO * (
    pd.read_csv(get_test_instrument_team_data_path("swapi/imap_swapi_central-effective-area_20260425_v001.csv"))
    .set_index("esa_voltage").effective_area.to_xarray().rename(esa_voltage="V")
    .pint.quantify("cm**2")
)
_voltages_mag = central_effective_area["V"].values
voltages = xr.DataArray(
    Q(_voltages_mag, "V"), dims="V", coords={"V": _voltages_mag},
)

passband_coefficients = (
    pd.read_csv(get_test_instrument_team_data_path("swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"))
    .set_index(["region", "energy_ratio", "elevation"])[["0", "1", "2"]]
    .rename_axis(columns="degree")
    .to_xarray()
    .to_dataarray("degree")  # degree as in polynomial order, not units
    .rename(elevation="elevation_deg")
    .pipe(lambda da: da.assign_coords(degree=da["degree"].astype(int)))
)
source_speed_ratio = np.sqrt(
    passband_coefficients["energy_ratio"].values / K_FACTOR.to('eV/V/e').magnitude
)
passband_coefficients = (
    passband_coefficients.assign_coords(speed_ratio=("energy_ratio", source_speed_ratio))
    .swap_dims({"energy_ratio": "speed_ratio"}).drop_vars("energy_ratio")
)

az_transmission_native = (
    pd.read_csv(get_test_instrument_team_data_path("swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"))
    .fillna(0).set_index("abs_azimuth").transmission.to_xarray()
).coarsen(abs_azimuth=10, boundary='trim').mean()
print("azimuth spacing:", az_transmission_native.abs_azimuth.to_series().diff().dropna().round(4).unique())

density_data = np.loadtxt(get_test_instrument_team_data_path("swapi/density-of-neutral-helium-lut.dat"))
psi_axis, r_axis = np.unique(density_data[:, 0]), np.unique(density_data[:, 1])
density_values = density_data[:, 2].reshape(len(psi_axis), len(r_axis))

r_axis = np.insert(r_axis, 0, 0.0)
density_values = np.insert(density_values, 0, 0.0, axis=1)

density_table = xr.DataArray(
    density_values,
    dims=("psi", "r"), coords={"psi": psi_axis, "r": r_axis},
).pint.quantify("1/cm**3")

"""## Integration grid (native to the calibration files)"""

elevation_deg = passband_coefficients["elevation_deg"]
speed_ratio   = passband_coefficients["speed_ratio"]

# Mirror az_transmission_native (abs_azimuth 0..180) into signed [-180, 180].
_abs_az = az_transmission_native["abs_azimuth"].values
_trans  = az_transmission_native.values
_az_signed = np.concatenate([-_abs_az[:0:-1], _abs_az])
azimuth_deg = xr.DataArray(_az_signed, dims="azimuth_deg", coords={"azimuth_deg": _az_signed})
azimuthal_transmission = xr.DataArray(
    np.concatenate([_trans[:0:-1], _trans]),
    dims="azimuth_deg", coords={"azimuth_deg": _az_signed},
)

azimuth_rad = np.deg2rad(azimuth_deg)
elevation_rad = np.deg2rad(elevation_deg)
direction = xr.concat([
    -np.cos(elevation_rad) * np.sin(azimuth_rad),
    -np.cos(elevation_rad) * np.cos(azimuth_rad),
    -np.sin(elevation_rad) * xr.ones_like(azimuth_rad),
], dim="cartesian").assign_coords(cartesian=["x", "y", "z"])

central_instrument_speed = np.sqrt(2 * K_FACTOR * voltages / HE_PLUS_MASS_PER_CHARGE).pint.to("km/s")

"""## Evaluate passband at each ESA voltage"""

log_beam_energy = np.log(np.abs(K_FACTOR * voltages).pint.to("eV/e").pint.dequantify())
passband_per_region = np.exp(
    xr.polyval(log_beam_energy, passband_coefficients, degree_dim="degree")
).fillna(0)
# A_0 and P share the SIMION normalization at (θ=0, k*=1.89 eV/V/e); enforce
# P(elevation_deg=0, speed_ratio=1) = 1 to match the production passband grid.
passband_per_region = passband_per_region / passband_per_region.interp(
    elevation_deg=0.0, speed_ratio=1.0,
)
az_is_sg = np.abs(azimuth_deg) <= 20
passband_full = xr.where(az_is_sg, passband_per_region.sel(region="SG"),
                                   passband_per_region.sel(region="OA"))

"""## Parameters"""

SW_SPEED_KMS = 450.0
SW_AZ_DEG = 0.0
SW_EL_DEG = -10.0
COOLING_INDEX = 2.0       # α
CUTOFF_SPEED_KMS = 450.0  # v_b
IONIZATION_RATE_HZ = 2e-7 # β_E at 1 AU
HELIO_DIST_AU = 1.0       # r
INFLOW_PSI_DEG = 75.0     # ψ — IMAP-to-helium-inflow longitude offset
SW_SPEED_INERTIAL_KMS = 450.0  # |v_sw| in ECLIPJ2000 (≈ SW_SPEED + 25)

"""## Bulk-SW vector in SWAPI coords
(look-from convention matches `direction`)
"""

sw_az_rad = np.deg2rad(SW_AZ_DEG)
sw_el_rad = np.deg2rad(SW_EL_DEG)
v_sw_vec = Q(SW_SPEED_KMS, "km/s") * np.array([
    -np.cos(sw_el_rad) * np.sin(sw_az_rad),
    -np.cos(sw_el_rad) * np.cos(sw_az_rad),
    -np.sin(sw_el_rad),
])
v_sw = xr.DataArray(v_sw_vec, dims="cartesian", coords={"cartesian": ["x", "y", "z"]})

"""## Per-V speed grid and SW-frame speed"""

speed_grid = (
    speed_ratio * central_instrument_speed
).transpose('V', 'speed_ratio')
v_sw_speed = np.sqrt((v_sw**2).sum("cartesian"))            # scalar
v_dot_vsw  = xr.dot(direction, v_sw, dim="cartesian")       # (az, el)
speed_sw   = np.sqrt(np.maximum(
    speed_grid**2 + v_sw_speed**2 - 2 * speed_grid * v_dot_vsw,
    Q(0, "km**2/s**2"),
))

r"""## V-S PUI VDF

$$
f_\text{PUI}\!\left( r, \psi, w_k \right)
    = \frac{\alpha_{\text{PUI},k}}{4\pi}
      \cdot
      \frac{\beta_{E,k} r_E^2}{r u_\text{sw} v_{b,k}^3}
      \cdot
      w_k^{\alpha_{\text{PUI}, k} - 3}
      \cdot
      n_k (r w_k^{\alpha_{\text{PUI},k}}, \psi)
      \cdot
      \Theta(1 - w_k),
$$
"""

cooling_index = COOLING_INDEX
cutoff_speed = Q(CUTOFF_SPEED_KMS, "km/s")
r = Q(HELIO_DIST_AU, "au")
u_sw = Q(SW_SPEED_INERTIAL_KMS, "km/s")
beta_E = Q(IONIZATION_RATE_HZ, "1/s")
psi = Q(INFLOW_PSI_DEG, "deg")

w = (speed_sw / cutoff_speed).pint.to("dimensionless")

# Term 1  α / (4π)
term1 = cooling_index / (4 * np.pi)

# Term 2  (β_E · r_E²) / (r · u_sw · v_b³)
term2 = (beta_E * ONE_AU**2) / (r * u_sw * cutoff_speed**3)

# Term 3  w^(α−3)
term3 = w ** (cooling_index - 3)

# Term 4  `r·w^α`
term4 = density_table.pint.interp(
    psi=(psi % Q(360, "deg")).magnitude,
    r=(r * w**cooling_index).pint.to("au").pint.dequantify(),
).drop_vars(["psi", "r"]).fillna(0)

# Term 5  Θ(1 − w)
term5 = xr.where((w < 1), 1, 0)

f_PUI = (term1 * term2 * term3 * term4 * term5).pint.to("s**3/km**6")

"""## Coincidence count rate per ESA voltage"""

effective_area = (central_effective_area * passband_full * azimuthal_transmission).clip(min=Q(0, "cm**2"))

flux = f_PUI * speed_grid

deg2_to_sr = (np.pi / 180.0) ** 2
cos_elevation = np.cos(np.deg2rad(elevation_deg))
d3v = speed_grid ** 2 * deg2_to_sr * cos_elevation

integrand = effective_area * flux * d3v

integral = (
    integrand.integrate("azimuth_deg")
             .integrate("elevation_deg")
             .integrate("speed_ratio")
)
count_rate = (integral * central_instrument_speed).pint.to("counts/s")

count_rate.to_series().to_csv(OUTPUT_PATH)

"""## Plot"""

_v_flat = speed_sw.pint.to("km/s").pint.dequantify().values.ravel()
_f_flat = f_PUI.pint.dequantify().values.ravel()
_order = np.argsort(_v_flat)
v_flat_order, f_flat_order = _v_flat[_order], _f_flat[_order]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)

ax0.loglog(_voltages_mag, count_rate.pint.dequantify().values, ".-")
ax0.set_xlabel("ESA voltage [V]")
ax0.set_ylabel("Coincidence Rate [Hz]")
ax0.set_ylim(1e-1, 1e2)
ax0.set_title(
    f"|v_sw|={SW_SPEED_KMS:g} km/s  "
    f"az={SW_AZ_DEG:g}°  el={SW_EL_DEG:g}°"
)
ax0.grid(True, alpha=0.3)

ax1.semilogy(v_flat_order / cutoff_speed, f_flat_order)
ax1.set_xlabel("$w$ [km/s]")
ax1.set_ylabel(r"$f_\mathrm{PUI}$($w$)  [$s^3/km^6$]")
ax1.set_title(
    f"V-S shell  α={COOLING_INDEX:g}  v_b={CUTOFF_SPEED_KMS:g} km/s  "
    f"β_E={IONIZATION_RATE_HZ:g} /s  r={HELIO_DIST_AU:g} au  ψ={INFLOW_PSI_DEG:g}°"
)
ax1.set_xlim(0, 1.2)
ax1.grid(True, alpha=0.3)
plt.show()
