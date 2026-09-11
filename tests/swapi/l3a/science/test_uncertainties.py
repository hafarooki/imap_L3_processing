import unittest

import numpy as np
from uncertainties import covariance_matrix

from imap_l3_processing.swapi.l3a.science.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
)


class TestComputeHc3ParameterCovariance(unittest.TestCase):
    """Tests for `compute_hc3_parameter_covariance`: the parameter-space-agnostic sandwich estimator shared by the proton and alpha fitters."""

    def test_returns_p_by_p_symmetric_matrix_for_full_rank_jacobian(self):
        """A full-rank (n×p) Jacobian with non-zero residuals returns a finite, symmetric (p×p) covariance."""
        rng = np.random.default_rng(0)
        jacobian = rng.normal(size=(30, 4))
        residuals = rng.normal(size=30) * 0.1

        cov = compute_hc3_parameter_covariance(jacobian, residuals)

        self.assertEqual(cov.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(cov)))
        np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-12)

    def test_high_leverage_row_inflates_corresponding_diagonal(self):
        """A single row with leverage near 1 in column j produces a much larger diagonal cov[j,j] than the same fit without the leverage spike."""
        n_bins = 20
        p = 3
        # Baseline: rank-3 jacobian with no high-leverage row; uniform contribution.
        baseline_jacobian = np.zeros((n_bins, p))
        for i in range(n_bins):
            baseline_jacobian[i, i % p] = 1.0
        residuals = np.zeros(n_bins)
        residuals[0] = 0.1

        cov_baseline = compute_hc3_parameter_covariance(baseline_jacobian, residuals)

        # High-leverage variant: column 0 of row 0 dominates the JᵀJ for column 0.
        leveraged_jacobian = baseline_jacobian.copy()
        leveraged_jacobian[0, 0] = 1000.0
        cov_leveraged = compute_hc3_parameter_covariance(leveraged_jacobian, residuals)

        self.assertGreater(cov_leveraged[0, 0], cov_baseline[0, 0] * 1e3)

    def test_returns_all_nan_matrix_for_nan_jacobian(self):
        """An all-NaN Jacobian triggers the `LinAlgError` fallback and returns a (p×p) all-NaN matrix."""
        jacobian = np.full((20, 5), np.nan)
        residuals = np.zeros(20)

        cov = compute_hc3_parameter_covariance(jacobian, residuals)

        self.assertEqual(cov.shape, (5, 5))
        self.assertTrue(np.all(np.isnan(cov)))


class TestMakeCorrelatedVelocity(unittest.TestCase):
    """Tests for `make_correlated_velocity`: PSD-covariance correlated triplets vs the non-finite / non-PSD NaN-sigma fallback."""

    def setUp(self):
        self.nominal = np.array([-400.0, 5.0, -3.0])

    def test_valid_covariance_returns_three_correlated_ufloats(self):
        """A valid 3x3 PSD covariance produces a length-3 tuple of UFloats from `correlated_values`."""
        cov = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 2.0, 0.3],
                [0.5, 0.3, 1.0],
            ]
        )
        v = make_correlated_velocity(self.nominal, cov)
        self.assertEqual(len(v), 3)

    def test_valid_covariance_round_trips_through_uncertainties(self):
        """The pairwise covariance reconstructed from the returned UFloats matches the input matrix and nominal values are preserved exactly."""
        cov = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 2.0, 0.3],
                [0.5, 0.3, 1.0],
            ]
        )
        v = make_correlated_velocity(self.nominal, cov)

        recovered_cov = np.array(covariance_matrix(v))
        np.testing.assert_allclose(recovered_cov, cov, rtol=0, atol=1e-10)
        for component, expected in zip(v, self.nominal):
            self.assertAlmostEqual(component.nominal_value, expected)

    def test_non_finite_covariance_returns_nan_sigma_ufloats(self):
        """An all-NaN covariance falls back to independent UFloats that preserve the nominal values but carry NaN std_devs."""
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))

    def test_non_finite_covariance_returns_independent_ufloats(self):
        """The fallback UFloats from a NaN covariance are independent, so the recovered covariance matrix has no finite non-zero off-diagonals."""
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        # Independent UFloats have no off-diagonal correlation. Off-diagonals
        # must be either 0 or NaN — definitely not the finite values that
        # `correlated_values` would have produced.
        recovered = np.array(covariance_matrix(v))
        self.assertFalse(np.any(np.isfinite(recovered) & (recovered != 0)))

    def test_negative_eigenvalue_covariance_falls_back_to_nan_sigma_ufloats(self):
        """A finite but non-PSD covariance (diag with a negative eigenvalue) routes to the NaN-sigma fallback rather than letting `correlated_values` raise."""
        cov = np.diag([1.0, -1.0, 1.0])
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))


if __name__ == "__main__":
    unittest.main()
