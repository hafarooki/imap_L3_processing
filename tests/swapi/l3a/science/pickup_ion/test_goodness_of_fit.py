from unittest import TestCase

import numpy as np

from imap_l3_processing.swapi.constants import SWAPI_BACKGROUND_RATE
from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import (
    MAX_CUTOFF_SPEED_KMS,
    MAX_CUTOFF_SPEED_RATIO,
    MAX_IONIZATION_RATE,
    MIN_CUTOFF_SPEED_RATIO,
    MIN_IONIZATION_RATE,
    CUTOFF_DROP_RATIO,
    goodness_of_fit_metrics,
    goodness_of_fit_upper_energy,
    is_good_fit,
)


def _make_pui_model(
    rate_at_cutoff: float = 100.0,
    sweep_count: int = 50,
) -> (np.ndarray, np.ndarray):
    esa_energies = np.geomspace(50.0, 20000.0, 62)
    peak_energy = esa_energies[-7]  # -7 so there are 6 points after

    # PUI-like model spectrum, repeated across sweeps and excluding background.

    # shape of distribution, normalized by peak value
    shape = np.zeros_like(esa_energies)

    # power law up to the cutoff
    below_cutoff = esa_energies <= peak_energy
    shape[below_cutoff] = (esa_energies[below_cutoff] / peak_energy) ** 0.5

    # custom-shaped cutoff, with only one point below CUTOFF_DROP_RATIO = 0.4
    rolloff_start = int(np.argmax(~below_cutoff))
    shape[rolloff_start : rolloff_start + 3] = np.array([0.9, 0.5, 0.2])

    # scale by peak rate
    model_rates = rate_at_cutoff * shape

    # repeat across sweeps
    return esa_energies, np.tile(model_rates, (sweep_count, 1))


class TestGoodnessOfFit(TestCase):
    def test_goodness_of_fit_upper_energy(self):
        esa_energies = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
        cases = {
            "rolloff after the peak": (
                np.array([1.0, 4.0, 2.0, 1.1, 0.0]),
                800.0,
            ),
            "rate exactly at the drop ratio is kept": (
                np.array([1.0, 4.0, 2.0, 4.0 * CUTOFF_DROP_RATIO, 0.0]),
                800.0,
            ),
            "rate just below the drop ratio is excluded": (
                np.array([1.0, 4.0, 2.0, 4.0 * CUTOFF_DROP_RATIO - 1e-9, 0.0]),
                400.0,
            ),
            "peak at the highest step": (
                np.array([0.5, 1.0, 2.0, 3.0, 4.0]),
                1600.0,
            ),
            "highest qualifying step wins past a dip": (
                np.array([1.0, 4.0, 0.1, 2.0, 0.0]),
                800.0,
            ),
            "only the peak qualifies": (
                np.array([0.0, 0.0, 4.0, 0.0, 0.0]),
                400.0,
            ),
        }
        for name, (chunk_mean_model_rates, expected_energy) in cases.items():
            with self.subTest(name):
                self.assertEqual(
                    expected_energy,
                    goodness_of_fit_upper_energy(esa_energies, chunk_mean_model_rates),
                )

    def test_goodness_of_fit_metrics(self):
        esa_energies = np.array([100.0, 200.0, 400.0, 800.0])
        # Claude: the sweeps average to [2, 4, 1, 0]; the 1 at 400 eV is exactly
        # Claude: CUTOFF_DROP_RATIO of the peak, so E_1/4 = 400 eV and only the
        # Claude: 800 eV step counts toward the past-peak ratio.
        model_rates = np.array(
            [
                [1.0, 3.0, 0.5, 0.0],
                [3.0, 5.0, 1.5, 0.0],
            ]
        )
        self.assertEqual(0.25, CUTOFF_DROP_RATIO)
        chunk_mean_model_with_background = (
            np.array([2.0, 4.0, 1.0, 0.0]) + SWAPI_BACKGROUND_RATE
        )

        # Claude: observed means sit 10% above, 10% below, and on the model at
        # Claude: the three in-range steps; the sweeps straddle those means.
        chunk_mean_observed_rates = np.array(
            [
                1.1 * chunk_mean_model_with_background[0],
                0.9 * chunk_mean_model_with_background[1],
                chunk_mean_model_with_background[2],
                0.6,
            ]
        )
        observed_rates = np.stack(
            [chunk_mean_observed_rates - 0.2, chunk_mean_observed_rates + 0.2]
        )

        mean_relative_error, past_peak_ratio = goodness_of_fit_metrics(
            esa_energies, model_rates, observed_rates
        )

        self.assertAlmostEqual((0.1 + 0.1 + 0.0) / 3, mean_relative_error)
        self.assertAlmostEqual(0.6 / 4.0, past_peak_ratio)

    def test_goodness_of_fit_metrics_perfect_fit(self):
        esa_energies, model_rates = _make_pui_model()
        observed_rates = model_rates + SWAPI_BACKGROUND_RATE
        observed_rates[:, model_rates.mean(axis=0) == 0] = 0.0

        mean_relative_error, past_peak_ratio = goodness_of_fit_metrics(
            esa_energies, model_rates, observed_rates
        )

        self.assertAlmostEqual(0.0, mean_relative_error)
        # Claude: the one nonzero past-peak step is the 0.2 rolloff point.
        self.assertAlmostEqual(
            (0.2 * 100.0 + SWAPI_BACKGROUND_RATE) / 4 / 100.0, past_peak_ratio
        )

    def test_fit_acceptance_criteria(self):
        esa_energies, model_rates = _make_pui_model()
        observed_rates = model_rates + SWAPI_BACKGROUND_RATE

        max_cutoff = MAX_CUTOFF_SPEED_KMS
        defaults = dict(
            esa_energies=esa_energies,
            model_rates=model_rates,
            observed_rates=observed_rates,
            cutoff_speed_kms=max_cutoff - 50.0,
            sw_speed_kms=400.0,
            ionization_rate=1e-7,
        )

        chunk_mean_model_rates = model_rates.mean(axis=0)
        peak_model_rate = chunk_mean_model_rates.max()
        cutoff_mask_energy = esa_energies[
            chunk_mean_model_rates >= peak_model_rate * CUTOFF_DROP_RATIO
        ].max()
        past_cutoff_mask = esa_energies > cutoff_mask_energy
        assert (
            past_cutoff_mask.sum() == 4
        )  # 4 points by construction: three zero, one nonzero

        def extra_counts_past_cutoff(fraction: float) -> dict:
            return dict(
                observed_rates=model_rates
                + past_cutoff_mask * peak_model_rate * fraction
            )

        def poisson_sampled_observed_rates() -> dict:
            livetime = 0.145
            expected_counts = observed_rates * livetime
            sampled_counts = np.random.default_rng(0).poisson(expected_counts)
            return dict(observed_rates=sampled_counts / livetime)

        cases = {
            "all good": (True, {}),
            "cutoff just below maximum": (
                True,
                dict(cutoff_speed_kms=max_cutoff - 1.0),
            ),
            "cutoff at maximum": (True, dict(cutoff_speed_kms=max_cutoff)),
            "cutoff above maximum": (False, dict(cutoff_speed_kms=max_cutoff + 1.0)),
            "model underpredicts a little": (True, dict(model_rates=model_rates * 0.9)),
            "model underpredicts a lot": (False, dict(model_rates=model_rates * 0.8)),
            "small extra counts bit past cutoff": (True, extra_counts_past_cutoff(0.3)),
            "too much past cutoff": (False, extra_counts_past_cutoff(0.5)),
            "counting noise only": (True, poisson_sampled_observed_rates()),
        }
        for case_name, (expected, overrides) in cases.items():
            with self.subTest(case_name):
                self.assertEqual(
                    expected, is_good_fit(**{**defaults, **overrides}), msg=case_name
                )

    def test_fitted_parameter_range_criteria(self):
        """A fit whose cutoff speed leaves the allowed multiple of the bulk
        speed, or whose ionization rate leaves the allowed range, is rejected.
        Both ranges are inclusive of their endpoints."""
        esa_energies, model_rates = _make_pui_model()

        # 300 km/s keeps 1.5 * v_sw under MAX_CUTOFF_SPEED_KMS
        sw_speed = 300.0
        defaults = dict(
            esa_energies=esa_energies,
            model_rates=model_rates,
            observed_rates=model_rates + SWAPI_BACKGROUND_RATE,
            cutoff_speed_kms=sw_speed,
            sw_speed_kms=sw_speed,
            ionization_rate=1e-7,
        )

        min_cutoff_speed = sw_speed * MIN_CUTOFF_SPEED_RATIO
        max_cutoff_speed = sw_speed * MAX_CUTOFF_SPEED_RATIO
        cases = {
            "all good": (True, {}),
            "cutoff speed below the lowest allowed multiple": (
                False,
                dict(cutoff_speed_kms=min_cutoff_speed - 1.0),
            ),
            "cutoff speed at the lowest allowed multiple of the bulk speed": (
                True,
                dict(cutoff_speed_kms=min_cutoff_speed),
            ),
            "cutoff speed above the lowest allowed multiple": (
                True,
                dict(cutoff_speed_kms=min_cutoff_speed + 1.0),
            ),
            "cutoff speed below the highest allowed multiple": (
                True,
                dict(cutoff_speed_kms=max_cutoff_speed - 1.0),
            ),
            "cutoff speed at the highest allowed multiple of the bulk speed": (
                True,
                dict(cutoff_speed_kms=max_cutoff_speed),
            ),
            "cutoff speed above the highest allowed multiple": (
                False,
                dict(cutoff_speed_kms=max_cutoff_speed + 1.0),
            ),
            "ionization rate below the lower limit": (
                False,
                dict(ionization_rate=MIN_IONIZATION_RATE * 0.99),
            ),
            "ionization rate at the lower limit": (
                True,
                dict(ionization_rate=MIN_IONIZATION_RATE),
            ),
            "ionization rate above the lower limit": (
                True,
                dict(ionization_rate=MIN_IONIZATION_RATE * 1.01),
            ),
            "ionization rate below the upper limit": (
                True,
                dict(ionization_rate=MAX_IONIZATION_RATE * 0.99),
            ),
            "ionization rate at the upper limit": (
                True,
                dict(ionization_rate=MAX_IONIZATION_RATE),
            ),
            "ionization rate above the upper limit": (
                False,
                dict(ionization_rate=MAX_IONIZATION_RATE * 1.01),
            ),
        }
        for case_name, (expected, overrides) in cases.items():
            with self.subTest(case_name):
                self.assertEqual(
                    expected, is_good_fit(**{**defaults, **overrides}), msg=case_name
                )
