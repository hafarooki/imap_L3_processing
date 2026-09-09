import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.calculate_initial_guess import (
    INITIAL_TEMPERATURE_FLOOR_K,
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000, load_swapi_response


# RTN → SWAPI rotation. Body +Y (the SWAPI boresight / spin axis) in RTN is
# column 1 of the transpose, i.e. -R̂_RTN. The solar wind direction (anti-
# parallel to the spin axis) is therefore +R̂.
_R_BASE_RTN_TO_SWAPI = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)


def _esa_voltage_for_proton_speed(speed_km_s: float) -> float:
    """Inverse of `esa_voltage_to_proton_speed` — pick the ESA voltage whose
    central proton speed is exactly `speed_km_s`. Used to place the count-rate
    peak at a known voltage."""
    return float(
        PROTON_MASS_KG
        * (speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _spin_rotation_matrices(n: int) -> np.ndarray:
    """SWAPI→RTN rotation matrices for `n` consecutive bins. Body +Y (the
    SWAPI boresight / spin axis) lies along -R̂_RTN on every bin, so the
    chunk-mean spin axis is exactly -R̂."""
    return np.tile(_R_BASE_RTN_TO_SWAPI.T, (n, 1, 1))


def _build_proton_ctx(count_rate: np.ndarray, esa_voltage: np.ndarray):
    """Build a real proton-species `SolarWindFitContext` from the shipped CSVs.

    `count_rate`, `esa_voltage`, and the rotation matrices are all (N,)/(N,3,3)
    aligned in the standard L2 layout — one entry per bin, multiple sweeps
    flattened into a single axis."""
    response = load_swapi_response(warm_cache_voltages=esa_voltage)
    ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=_spin_rotation_matrices(len(esa_voltage)),
    )
    return ctx


def _make_synthetic_ctx_at_known_truth(truth: SolarWindParams,
                                       n_bins: int = 71):
    """Build a context whose `count_rate` array is the noiseless ideal forward
    model evaluated at `truth`."""
    bulk_speed = float(np.linalg.norm(truth.velocity_rtn))
    # Wide enough to bracket ±5σ at T=1e5 K, narrow enough to keep all bins
    # on-instrument.
    speed_grid = np.linspace(0.4 * bulk_speed, 1.6 * bulk_speed, n_bins)
    voltages = np.array(
        [_esa_voltage_for_proton_speed(s) for s in speed_grid]
    )

    # Build a placeholder ctx, evaluate the forward model at `truth` to get
    # ideal rates, then build the production ctx with those rates as
    # `count_rate`.
    placeholder = _build_proton_ctx(np.ones_like(voltages), voltages)
    ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(truth, placeholder)
    return _build_proton_ctx(ideal_rates, voltages)


class TestCalculateInitialGuessSeeds(unittest.TestCase):
    """Tests for `calculate_initial_guess` — doc-specified seed construction, with the Gaussian refiner patched so seeds passed to it are observable directly."""

    def _ctx_with_peak_at(self, peak_speed_kms: float):
        """Build a context whose count-rate spectrum has its maximum at a
        bin whose ESA voltage corresponds to `peak_speed_kms`. The shape of
        the spectrum doesn't matter — we patch out the refiner."""
        speed_grid = np.linspace(0.4 * peak_speed_kms, 1.6 * peak_speed_kms,
                                 31)
        voltages = np.array(
            [_esa_voltage_for_proton_speed(s) for s in speed_grid]
        )
        peak_idx = len(speed_grid) // 2  # middle bin
        voltages[peak_idx] = _esa_voltage_for_proton_speed(peak_speed_kms)
        count_rate = np.linspace(0.1, 0.5, len(voltages))
        count_rate[peak_idx] = 100.0  # Unambiguous global maximum.
        return _build_proton_ctx(count_rate, voltages)

    def test_bulk_speed_seed_is_proton_speed_at_peak_voltage(self):
        """A spectrum whose unambiguous maximum sits at the ESA voltage for 480 km/s passes 480 km/s as the bulk-speed seed into the Gaussian refiner."""
        peak_speed = 480.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "calculate_initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, 1e5),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        # Args: (speed, count_rate, bulk_speed_seed, temperature_seed, mass_kg)
        args = patched_refine.call_args.args
        bulk_speed_seed_arg = args[2]
        np.testing.assert_allclose(bulk_speed_seed_arg, peak_speed, rtol=1e-12)

    def test_temperature_seed_uses_documented_speed_squared_formula(self):
        """With a peak bulk speed well above the floor, the temperature seed handed to the refiner is exactly 60_000·(v/400)² K."""
        peak_speed = 480.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "calculate_initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, 1e5),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        temperature_seed_arg = patched_refine.call_args.args[3]
        expected_temperature = 60_000.0 * (peak_speed / 400.0) ** 2
        np.testing.assert_allclose(temperature_seed_arg, expected_temperature,
                                   rtol=1e-12)

    def test_temperature_seed_floors_at_one_ev(self):
        """At a low peak speed where 60_000·(v/400)² is below 1 eV, the temperature seed handed to the refiner is clamped to `INITIAL_TEMPERATURE_FLOOR_K`."""
        peak_speed = 100.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "calculate_initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, INITIAL_TEMPERATURE_FLOOR_K),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        temperature_seed_arg = patched_refine.call_args.args[3]
        self.assertEqual(temperature_seed_arg, INITIAL_TEMPERATURE_FLOOR_K)


class TestCalculateInitialGuessDirection(unittest.TestCase):
    """Tests for `calculate_initial_guess` — chunk-mean spin-axis direction handling for the returned `velocity_rtn`."""

    def test_initial_velocity_is_anti_parallel_to_spin_axis(self):
        """With every rotation matrix aligning body +Y to -R̂, the returned bulk velocity points along +R̂ — the negation of the chunk-mean spin axis."""
        speed_grid = np.linspace(300.0, 600.0, 31)
        voltages = np.array(
            [_esa_voltage_for_proton_speed(s) for s in speed_grid]
        )
        count_rate = np.linspace(0.1, 0.5, len(voltages))
        peak_idx = len(speed_grid) // 2  # middle bin
        count_rate[peak_idx] = 100.0
        ctx = _build_proton_ctx(count_rate, voltages)

        # `_spin_rotation_matrices` aligns body +Y to -R̂ on every bin, so
        # the chunk-mean spin axis is exactly -R̂.
        expected_axis = np.array([-1.0, 0.0, 0.0])

        guess = calculate_initial_guess(ctx)
        v_unit = guess.velocity_rtn / np.linalg.norm(
            guess.velocity_rtn
        )
        np.testing.assert_allclose(v_unit, -expected_axis, atol=1e-12)

    def test_velocity_magnitude_matches_truth_bulk_speed(self):
        """On a noiseless forward-model spectrum at 450 km/s the unmocked Gaussian refine recovers the truth, so the returned `|velocity_rtn|` matches the truth bulk speed to ~2%."""
        truth_bulk_speed = 450.0
        truth = SolarWindParams(
            density=5.0,
            velocity_rtn=np.array([truth_bulk_speed, 0.0, 0.0]),
            temperature=1.0e5,
            mass=PROTON_MASS_KG,
        )
        ctx = _make_synthetic_ctx_at_known_truth(truth)

        guess = calculate_initial_guess(ctx)
        np.testing.assert_allclose(
            np.linalg.norm(guess.velocity_rtn),
            truth_bulk_speed,
            rtol=2e-2,
        )


class TestCalculateInitialGuessDensity(unittest.TestCase):
    """Tests for `calculate_initial_guess` — the returned density is the optimal scale of the unit-density forward model against the observed count rates."""

    def test_density_is_optimal_scale_of_unit_density_forward_model(self):
        """Re-running the unit-density forward model at the guess's own velocity/temperature and feeding it through `optimal_density_scale` reproduces the guess density exactly."""
        truth = SolarWindParams(
            density=4.2,
            velocity_rtn=np.array([470.0, 0.0, 0.0]),
            temperature=1.2e5,
            mass=PROTON_MASS_KG,
        )
        ctx = _make_synthetic_ctx_at_known_truth(truth)

        guess = calculate_initial_guess(ctx)

        # Reproduce the function's density step independently: build a
        # unit-density forward-model evaluation at the *same* bulk-velocity
        # direction and temperature the function uses, then call
        # `optimal_density_scale`. The result must match `guess.density`.
        unit_density_params = SolarWindParams(
            density=1.0,
            velocity_rtn=guess.velocity_rtn,
            temperature=guess.temperature,
            mass=ctx.mass_kg,
        )
        unit_rates, _ = model_solar_wind_ideal_coincidence_rates(
            unit_density_params, ctx
        )
        expected_density = optimal_density_scale(unit_rates, ctx.count_rate)
        np.testing.assert_allclose(guess.density, expected_density, rtol=1e-10)


class TestCalculateInitialGuessRefinerFailure(unittest.TestCase):
    """Tests for `calculate_initial_guess` — pathological spectra that the Gaussian refiner cannot fit surface as a wrapped `RuntimeError`, with the offending peak-bin speed and seed temperature included in the message and the underlying `scipy` error chained as `__cause__`."""

    def test_wraps_runtime_error_on_pathological_edge_spike_spectrum(self):
        """A count-rate spectrum with an extreme isolated spike at the lowest-speed bin (1e10 surrounded by 1e-6) drives `curve_fit` past its `maxfev` budget; the refiner re-raises the resulting `RuntimeError` with a wrapped message that names the peak-bin seed, and the scipy error is preserved on `__cause__`."""
        peak_speed = 250.0
        speed_grid = np.linspace(peak_speed, peak_speed + 600.0, 71)
        voltages = np.array(
            [_esa_voltage_for_proton_speed(s) for s in speed_grid]
        )
        count_rate = np.full(len(voltages), 1e-6)
        count_rate[0] = 1.0e10
        ctx = _build_proton_ctx(count_rate, voltages)

        with self.assertRaises(RuntimeError) as raise_context:
            calculate_initial_guess(ctx)

        message = str(raise_context.exception)
        self.assertIn("Initial-guess Gaussian fit failed", message)
        self.assertIn(f"{peak_speed:.1f}", message)
        self.assertIsInstance(raise_context.exception.__cause__, RuntimeError)
        self.assertIn("maxfev", str(raise_context.exception.__cause__))


if __name__ == "__main__":
    unittest.main()
