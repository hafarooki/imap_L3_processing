"""Check that reported proton and alpha fit uncertainties match observed errors.

Monte Carlo trials use a simple model with:
- Poisson noise,
- 1% log-normal noise
- a 10 Hz Poisson floor.

For each L3a CDF value/sigma pair, the normalized errors,
(fitted - truth) / sigma, must have mean zero and standard deviation one within
the test tolerances.
"""

from __future__ import annotations

import multiprocessing
import os
import types
import unittest
from datetime import timedelta

import numpy as np

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import SWAPI_LIVETIME_S
from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitResult,
    _alpha_moments_from_fit,
    _proton_moments_from_fit,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import (
    NOMINAL_TEST_EPOCH_TT2000,
    NOMINAL_SWAPI_TO_RTN_ROTATION,
    load_swapi_response,
)
from tests.test_helpers import run_periodically


_N_BINS_PER_SWEEP = 62
_N_SWEEPS = 5
_VOLTAGE_AXIS = np.broadcast_to(
    np.logspace(np.log10(3500.0), np.log10(140.0), _N_BINS_PER_SWEEP),
    (_N_SWEEPS, _N_BINS_PER_SWEEP),
).copy()
_N_MEAS = _VOLTAGE_AXIS.size

_SWEEP_DURATION_S = 12.0
_SAMPLE_TIME_PER_BIN_S = _SWEEP_DURATION_S / 72
_SPIN_PERIOD_S = 15.13


def _per_bin_rotation_matrices() -> np.ndarray:
    """Per-bin SWAPI→RTN rotations built by spinning `NOMINAL_SWAPI_TO_RTN_ROTATION`
    about its own +Y column (the spin axis) at the IMAP spin rate. Without
    spin-phase variation across bins the velocity component along the spin
    axis is degenerate in the proton fit — the spin sweeps azimuth across
    bins and breaks that degeneracy."""
    sweep_index = np.repeat(np.arange(_N_SWEEPS), _N_BINS_PER_SWEEP)
    bin_index_in_sweep = np.tile(np.arange(1, _N_BINS_PER_SWEEP + 1), _N_SWEEPS)
    sample_times_s = (
        sweep_index * _SWEEP_DURATION_S
        + bin_index_in_sweep * _SAMPLE_TIME_PER_BIN_S
    )
    spin_axis = NOMINAL_SWAPI_TO_RTN_ROTATION[:, 1]
    spin_axis = spin_axis / np.linalg.norm(spin_axis)
    delta_phi = (-2.0 * np.pi / _SPIN_PERIOD_S) * (
        sample_times_s - 0.5 * _SWEEP_DURATION_S
    )
    ax, ay, az = spin_axis
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return rot @ NOMINAL_SWAPI_TO_RTN_ROTATION


_ROTATION_MATRICES = _per_bin_rotation_matrices()

# Slow-wind truth. The bulk velocity is -|v|·(+Y column of the anchor SWAPI→RTN
# rotation), i.e. anti-parallel to the spin axis so the wind enters the SWAPI
# aperture along -Y_SWAPI = boresight. B̂ is taken parallel to v̂ (typical
# Parker-spiral ≈ radial direction), so a positive Δv makes alphas faster than
# protons.
_TRUE_BULK_SPEED_KM_S = 450.0
_TRUE_PROTON_DENSITY_CM3 = 5.0
_TRUE_PROTON_TEMPERATURE_K = 1.0e5
_TRUE_PROTON_VELOCITY_RTN = (
    -_TRUE_BULK_SPEED_KM_S * NOMINAL_SWAPI_TO_RTN_ROTATION[:, 1]
)
_TRUE_ALPHA_DENSITY_CM3 = 0.2
_TRUE_ALPHA_TEMPERATURE_K = 4.0e5
_TRUE_DELTA_V_KM_S = 30.0
_B_HAT_RTN = _TRUE_PROTON_VELOCITY_RTN / np.linalg.norm(_TRUE_PROTON_VELOCITY_RTN)

# Spacecraft velocity (km/s) treated as exact by the production pipeline; only
# shifts the sun-frame bulk-velocity vector, so it must not change σ.
_SC_VELOCITY_RTN = np.array([0.0, 30.0, 0.0])

# Noise model — same constants as `docs/swapi/figure_src/plot_uncertainty_mc.py`.
_LOGNORMAL_REL_SIGMA = 0.01
_NOISE_FLOOR_HZ = 10.0

_N_TRIALS = 1000

# normalized error should be 1 within 10%
_NORMALIZED_ERROR_WIDTH_TOLERANCE = 0.10
# a bit high as a consequence of the 10 Hz noise floor added
_NORMALIZED_ERROR_BIAS_TOLERANCE = 0.40
_PERIODIC_FREQUENCY = timedelta(days=7)

_RTN_AXES = ("R", "T", "N")


def _build_truth_count_rates(response, *, with_alpha: bool) -> np.ndarray:
    proton_params = SolarWindParams(
        density=_TRUE_PROTON_DENSITY_CM3,
        velocity_rtn=_TRUE_PROTON_VELOCITY_RTN.copy(),
        temperature=_TRUE_PROTON_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    proton_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(_VOLTAGE_AXIS.shape),
        esa_voltage=_VOLTAGE_AXIS,
        swapi_response=response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=_ROTATION_MATRICES,
    )
    proton_rates, _ = model_solar_wind_ideal_coincidence_rates(proton_params, proton_ctx)
    total_rates = proton_rates
    if with_alpha:
        alpha_velocity = _TRUE_PROTON_VELOCITY_RTN + _TRUE_DELTA_V_KM_S * _B_HAT_RTN
        alpha_params = SolarWindParams(
            density=_TRUE_ALPHA_DENSITY_CM3,
            velocity_rtn=alpha_velocity,
            temperature=_TRUE_ALPHA_TEMPERATURE_K,
            mass=ALPHA_PARTICLE_MASS_KG,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=np.zeros(_VOLTAGE_AXIS.shape),
            esa_voltage=_VOLTAGE_AXIS,
            swapi_response=response,
            time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
            species=Species.ALPHA,
            rotation_matrices=_ROTATION_MATRICES,
        )
        alpha_rates, _ = model_solar_wind_ideal_coincidence_rates(alpha_params, alpha_ctx)
        total_rates = proton_rates + alpha_rates
    deadtime_applied = total_rates * deadtime_factor(total_rates)
    return deadtime_applied.reshape(_VOLTAGE_AXIS.shape)


def _inject_noise(truth_rates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(np.maximum(truth_rates * SWAPI_LIVETIME_S, 0.0))
    rates = counts.astype(float) / SWAPI_LIVETIME_S
    sigma_log = np.sqrt(np.log1p(_LOGNORMAL_REL_SIGMA**2))
    log_factor = rng.normal(-0.5 * sigma_log**2, sigma_log, rates.shape)
    rates = np.maximum(rates * np.exp(log_factor), 0.0)
    floor_counts = rng.poisson(_NOISE_FLOOR_HZ * SWAPI_LIVETIME_S, size=rates.shape)
    return rates + floor_counts.astype(float) / SWAPI_LIVETIME_S


def _velocity_component_truth(base: str, velocity_rtn: np.ndarray) -> dict:
    return {
        f"{base}_{axis}": float(velocity_rtn[i]) for i, axis in enumerate(_RTN_AXES)
    }


def _proton_truth_values() -> dict:
    """Expected value of every proton CDF quantity that declares a σ."""
    velocity_sun = _TRUE_PROTON_VELOCITY_RTN + _SC_VELOCITY_RTN
    return {
        "proton_sw_density": _TRUE_PROTON_DENSITY_CM3,
        "proton_sw_temperature": _TRUE_PROTON_TEMPERATURE_K,
        "proton_sw_speed": float(np.linalg.norm(_TRUE_PROTON_VELOCITY_RTN)),
        "proton_sw_speed_sun": float(np.linalg.norm(velocity_sun)),
        **_velocity_component_truth(
            "proton_sw_velocity_rtn", _TRUE_PROTON_VELOCITY_RTN
        ),
    }


def _alpha_truth_values() -> dict:
    """Expected value of every alpha CDF quantity that declares a σ. B̂ is
    parallel to the proton bulk velocity, so the alpha velocity is the proton
    vector lengthened by the truth differential speed."""
    velocity_rtn = _TRUE_PROTON_VELOCITY_RTN + _TRUE_DELTA_V_KM_S * _B_HAT_RTN
    velocity_sun = velocity_rtn + _SC_VELOCITY_RTN
    return {
        "alpha_sw_density": _TRUE_ALPHA_DENSITY_CM3,
        "alpha_sw_temperature": _TRUE_ALPHA_TEMPERATURE_K,
        "alpha_sw_speed": float(np.linalg.norm(velocity_rtn)),
        "alpha_sw_speed_sun": float(np.linalg.norm(velocity_sun)),
        **_velocity_component_truth("alpha_sw_velocity_rtn", velocity_rtn),
    }


def _iter_value_sigma_pairs(record: dict):
    """Yield `(name, value, sigma)` for every quantity in `record` that declares
    an uncertainty — either a `<base>_uncert` scalar partner or a `<base>_covariance`
    matrix whose diagonal entries pair with a 3-vector value field (the SC-frame
    velocity, since sun-frame is an exact deterministic shift of it). Fields with
    no σ partner (epoch, quality_flags) are silently skipped."""
    for key in list(record.keys()):
        if key.endswith("_uncert"):
            base = key[: -len("_uncert")]
            yield base, float(record[base]), float(record[key])
        elif "covariance" in key:
            base = key.replace("_covariance", "")
            if base not in record:
                base = f"{base}_sc"
            assert base in record, f"no companion vector for covariance key {key!r}"
            cov = np.asarray(record[key])
            values = np.asarray(record[base])
            for i, axis in enumerate(_RTN_AXES):
                yield (
                    f"{base}_{axis}",
                    float(values[i]),
                    float(np.sqrt(max(cov[i, i], 0.0))),
                )


def _accumulate(accumulator: dict, record: dict) -> None:
    for name, value, sigma in _iter_value_sigma_pairs(record):
        accumulator.setdefault(name, []).append((value, sigma))


def _assert_calibration(
    accumulator: dict, n_good: int, *, species: Species, truth_values: dict
) -> None:
    label = species.name.lower()
    if n_good < _N_TRIALS * 0.9:
        raise AssertionError(
            f"{label}: only {n_good}/{_N_TRIALS} trials produced a good fit"
        )
    failures = []
    for name, samples in sorted(accumulator.items()):
        assert name in truth_values, f"no truth value declared for {name!r}"
        values, reported_sigmas = (
            np.asarray(column, dtype=float) for column in zip(*samples)
        )
        if not np.all(np.isfinite(reported_sigmas) & (reported_sigmas > 0)):
            failures.append(f"  {name}: non-positive or non-finite reported σ")
            continue
        normalized_errors = (values - truth_values[name]) / reported_sigmas
        width = float(normalized_errors.std(ddof=1))
        bias = float(normalized_errors.mean())
        if abs(width - 1.0) >= _NORMALIZED_ERROR_WIDTH_TOLERANCE:
            failures.append(
                f"  {name}: normalized-error width {width:.3f}, want "
                f"1 ± {_NORMALIZED_ERROR_WIDTH_TOLERANCE} "
                f"(σ {'over' if width < 1.0 else 'under'}-reported by "
                f"{abs(1.0 / width - 1.0):.1%})"
            )
        if abs(bias) >= _NORMALIZED_ERROR_BIAS_TOLERANCE:
            failures.append(
                f"  {name}: normalized-error bias {bias:+.3f} σ, want "
                f"0 ± {_NORMALIZED_ERROR_BIAS_TOLERANCE} σ"
            )
    if failures:
        raise AssertionError(
            f"{label} σ calibration: normalized errors not standard "
            f"(N={n_good}):\n" + "\n".join(failures)
        )


# Module-level worker state, populated in the parent process before workers
# are forked. Each worker inherits the snapshot via fork. Resetting between
# tests requires that the per-test setup runs in the parent before the next
# Pool is created.
_WORKER_STATE: types.SimpleNamespace | None = None


def _initialize_worker_state(*, with_alpha: bool) -> None:
    global _WORKER_STATE
    response = load_swapi_response(warm_cache_voltages=_VOLTAGE_AXIS)
    truth_rates = _build_truth_count_rates(response, with_alpha=with_alpha)
    _WORKER_STATE = types.SimpleNamespace(
        response=response,
        truth_rates=truth_rates,
        with_alpha=with_alpha,
    )


def _run_trial(trial_seed: int) -> dict | None:
    """Single MC trial. Returns a moments dict on success or `None` on bad
    fit / quality-flag rejection. Reads the forked `_WORKER_STATE`."""
    state = _WORKER_STATE
    rng = np.random.default_rng(trial_seed)
    noisy_rates = _inject_noise(state.truth_rates, rng)
    proton_ctx = build_solar_wind_fit_context(
        count_rate=noisy_rates,
        esa_voltage=_VOLTAGE_AXIS,
        swapi_response=state.response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.PROTON,
        rotation_matrices=_ROTATION_MATRICES,
    )
    proton_result = fit_solar_wind_proton_model(proton_ctx)
    if int(proton_result.quality_flag) != int(SwapiL3Flags.NONE):
        return None
    if not state.with_alpha:
        # `_proton_moments_from_fit` only touches `data_chunk` on the
        # fill-value fallback path (non-finite speed); successful fits
        # never reach it, so None is safe here.
        return _proton_moments_from_fit(
            proton_result,
            epoch=0,
            data_chunk=None,
            sc_velocity_rtn=_SC_VELOCITY_RTN,
        )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=noisy_rates,
        esa_voltage=_VOLTAGE_AXIS,
        swapi_response=state.response,
        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        species=Species.ALPHA,
        rotation_matrices=_ROTATION_MATRICES,
    )
    alpha_moments = fit_solar_wind_alpha_model(
        proton_ctx=proton_ctx,
        alpha_ctx=alpha_ctx,
        proton_moments=proton_result,
        magnetic_field_direction=_B_HAT_RTN,
    )
    if int(alpha_moments.quality_flag) != int(SwapiL3Flags.NONE):
        return None
    chunk_result = AlphaChunkFitResult(
        alpha_moments=alpha_moments,
        proton_moments=proton_result,
        b_hat_rtn=_B_HAT_RTN.copy(),
        quality_flag=int(alpha_moments.quality_flag),
    )
    return _alpha_moments_from_fit(chunk_result, epoch=0, sc_velocity_rtn=_SC_VELOCITY_RTN)


def _run_mc_in_parallel(n_trials: int) -> tuple[dict, int]:
    """Run `n_trials` independent MC trials across forked workers (one per
    CPU). Workers inherit `_WORKER_STATE` via fork — populate it in the
    parent first. Returns (accumulator, n_good)."""
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
    n_workers = max(1, os.cpu_count() or 1)
    accumulator: dict = {}
    n_good = 0
    with multiprocessing.get_context("fork").Pool(processes=n_workers) as pool:
        for record in pool.imap_unordered(_run_trial, range(n_trials), chunksize=10):
            if record is None:
                continue
            _accumulate(accumulator, record)
            n_good += 1
    return accumulator, n_good


class ProtonUncertaintyCalibration(unittest.TestCase):
    """Every proton CDF variable has standard normalized errors — unit width,
    zero mean — against its truth value under Poisson + log-normal +
    noise-floor noise."""

    @run_periodically(_PERIODIC_FREQUENCY)
    def test_proton_normalized_errors_are_standard(self):
        _initialize_worker_state(with_alpha=False)
        accumulator, n_good = _run_mc_in_parallel(_N_TRIALS)
        _assert_calibration(
            accumulator,
            n_good,
            species=Species.PROTON,
            truth_values=_proton_truth_values(),
        )


class AlphaUncertaintyCalibration(unittest.TestCase):
    """Every alpha CDF variable has standard normalized errors — unit width,
    zero mean — against its truth value under Poisson + log-normal +
    noise-floor noise."""

    @run_periodically(_PERIODIC_FREQUENCY)
    def test_alpha_normalized_errors_are_standard(self):
        _initialize_worker_state(with_alpha=True)
        accumulator, n_good = _run_mc_in_parallel(_N_TRIALS)
        _assert_calibration(
            accumulator,
            n_good,
            species=Species.ALPHA,
            truth_values=_alpha_truth_values(),
        )
