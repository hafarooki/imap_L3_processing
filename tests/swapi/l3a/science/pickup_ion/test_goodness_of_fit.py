from unittest import TestCase

import numpy as np

from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import (
    MAX_CUTOFF_SPEED_KMS,
    MAX_CUTOFF_SPEED_RATIO,
    MAX_IONIZATION_RATE,
    MIN_CUTOFF_SPEED_RATIO,
    MIN_IONIZATION_RATE,
    CUTOFF_DROP_RATIO,
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
    shape[rolloff_start:rolloff_start + 3] = np.array([0.9, 0.5, 0.2])

    # scale by peak rate
    model_rates = rate_at_cutoff * shape

    # repeat across sweeps
    return esa_energies, np.tile(model_rates, (sweep_count, 1))


class TestGoodnessOfFit(TestCase):
    def test_fit_acceptance_criteria(self):
        esa_energies, model_rates = _make_pui_model()
        background_count_rate = 0.1
        observed_rates = model_rates + background_count_rate

        max_cutoff = MAX_CUTOFF_SPEED_KMS
        defaults = dict(
            esa_energies=esa_energies,
            model_rates=model_rates,
            observed_rates=observed_rates,
            cutoff_speed_kms=max_cutoff - 50.0,
            sw_speed_kms=400.0,
            ionization_rate=1e-7,
            background_rate=background_count_rate,
        )

        chunk_mean_model_rates = model_rates.mean(axis=0)
        peak_model_rate = chunk_mean_model_rates.max()
        cutoff_mask_energy = esa_energies[chunk_mean_model_rates >= peak_model_rate * CUTOFF_DROP_RATIO].max()
        past_cutoff_mask = esa_energies > cutoff_mask_energy
        assert past_cutoff_mask.sum() == 4  # 4 points by construction: three zero, one nonzero

        def extra_counts_past_cutoff(fraction: float) -> dict:
            return dict(observed_rates=model_rates + past_cutoff_mask * peak_model_rate * fraction)

        def poisson_sampled_observed_rates() -> dict:
            livetime = 0.145
            expected_counts = observed_rates * livetime
            sampled_counts = np.random.default_rng(0).poisson(expected_counts)
            return dict(observed_rates=sampled_counts / livetime)

        cases = {
            "all good": (True, {}),
            "cutoff just below maximum": (True, dict(cutoff_speed_kms=max_cutoff - 1.0)),
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
                self.assertEqual(expected, is_good_fit(**{**defaults, **overrides}), msg=case_name)

    def test_fitted_parameter_range_criteria(self):
        """A fit whose cutoff speed leaves the allowed multiple of the bulk
        speed, or whose ionization rate leaves the allowed range, is rejected.
        Both ranges are inclusive of their endpoints."""
        esa_energies, model_rates = _make_pui_model()
        background_count_rate = 0.1

        # 300 km/s keeps 1.5 * v_sw under MAX_CUTOFF_SPEED_KMS
        sw_speed = 300.0
        defaults = dict(
            esa_energies=esa_energies,
            model_rates=model_rates,
            observed_rates=model_rates + background_count_rate,
            cutoff_speed_kms=sw_speed,
            sw_speed_kms=sw_speed,
            ionization_rate=1e-7,
            background_rate=background_count_rate,
        )

        min_cutoff_speed = sw_speed * MIN_CUTOFF_SPEED_RATIO
        max_cutoff_speed = sw_speed * MAX_CUTOFF_SPEED_RATIO
        cases = {
            "all good": (True, {}),
            "cutoff speed below the lowest allowed multiple": (
                False, dict(cutoff_speed_kms=min_cutoff_speed - 1.0)),
            "cutoff speed at the lowest allowed multiple of the bulk speed": (
                True, dict(cutoff_speed_kms=min_cutoff_speed)),
            "cutoff speed above the lowest allowed multiple": (
                True, dict(cutoff_speed_kms=min_cutoff_speed + 1.0)),
            "cutoff speed below the highest allowed multiple": (
                True, dict(cutoff_speed_kms=max_cutoff_speed - 1.0)),
            "cutoff speed at the highest allowed multiple of the bulk speed": (
                True, dict(cutoff_speed_kms=max_cutoff_speed)),
            "cutoff speed above the highest allowed multiple": (
                False, dict(cutoff_speed_kms=max_cutoff_speed + 1.0)),
            "ionization rate below the lower limit": (
                False, dict(ionization_rate=MIN_IONIZATION_RATE * 0.99)),
            "ionization rate at the lower limit": (
                True, dict(ionization_rate=MIN_IONIZATION_RATE)),
            "ionization rate above the lower limit": (
                True, dict(ionization_rate=MIN_IONIZATION_RATE * 1.01)),
            "ionization rate below the upper limit": (
                True, dict(ionization_rate=MAX_IONIZATION_RATE * 0.99)),
            "ionization rate at the upper limit": (
                True, dict(ionization_rate=MAX_IONIZATION_RATE)),
            "ionization rate above the upper limit": (
                False, dict(ionization_rate=MAX_IONIZATION_RATE * 1.01)),
        }
        for case_name, (expected, overrides) in cases.items():
            with self.subTest(case_name):
                self.assertEqual(expected, is_good_fit(**{**defaults, **overrides}), msg=case_name)
