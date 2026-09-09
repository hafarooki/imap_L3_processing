# SWAPI Error/Uncertainty Quantification

The [proton](./proton-sw.md), [alpha](./alpha-sw.md), and [pickup ion](./pickup-ion.md) data products all quantify their fit parameter uncertainties as described here.

Unweighted residuals are used to avoid biasing the fit by giving too much weight to off-peak points with significant contributions from populations besides the one being fit.
Even when the true variance of the residuals is not uniform (heteroscedasticity), unweighted least squares—which is optimal under homoscedastic, normally distributed errors—is still unbiased, albeit not maximally efficient.
But although the fit itself is unbiased, the Jacobian-based covariance matrix $`\Sigma_\mathbf{x} = \big(\sum_{i} r_{i}^{2} / (N-\text{DOF})\big) \cdot (J^{\top} J)^+`$ that is often used is inaccurate in the presence of heteroscedastic errors.
Instead, the heteroscedasticity-consistent HC3 method is used ([MacKinnon & White, 1985](https://doi.org/10.1016/0304-4076(85)90158-7); [Long & Ervin, 2000](https://doi.org/10.1080/00031305.2000.10474549); [Hayes & Cai, 2007](https://doi.org/10.3758/BF03192961)) to estimate the covariance matrix:
```math
\Sigma_\mathbf{x} = (J^{\top} J)^+ \thinspace  J^{\top} \mathrm{diag}\!\Big(\tfrac{r_{i}^{2}}{(1 - h_{ii})^{2}}\Big)\thinspace  J\thinspace  (J^{\top} J)^+,
```
where $`r_{i} = \text{pred}_{i} - \text{obs}_{i}`$ is the residual for measurement $i$, $`h_{ii} = J_{i}\thinspace (J^{\top} J)^+ J_{i}^{\top}`$, and $`{}^+`$ represents the Moore–Penrose pseudoinverse (used instead of a true inverse for numerical stability).

> **Important**: The uncertainties primarily represent statistical error and do not necessarily take systematic error into account.
> For example, density relies on the central effective area calibration, which has an uncertainty of up to 20%.
> This introduces systematic error that applies equally to all density measurements.
