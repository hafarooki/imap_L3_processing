import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.swapi.constants import SWAPI_BACKGROUND_RATE

MAX_CUTOFF_SPEED_KMS = 550.0  # v_cutoff <= 550 km/s
MAX_MEAN_RELATIVE_ERROR = 0.12  # Delta_rel <= 0.12
MAX_PAST_PEAK_RATIO = 0.4  # R_past_peak <= 0.4
CUTOFF_DROP_RATIO = (
    0.25  # extend up to and including the last step with model rate >= 0.25 * peak
)
MIN_CUTOFF_SPEED_RATIO = 0.5  # v_cutoff >= 0.5 * v_sw
MAX_CUTOFF_SPEED_RATIO = 1.5  # v_cutoff <= 1.5 * v_sw
MIN_IONIZATION_RATE = 0.6e-9  # beta >= 0.6 * 10^-9 s^-1
MAX_IONIZATION_RATE = 8.0e-7  # beta <= 8.0 * 10^-7 s^-1


def goodness_of_fit_upper_energy(
    esa_energies: NDArray, chunk_mean_model_rates: NDArray
) -> float:
    """
    The goodness-of-fit upper energy E_{1/4} (see [docs/swapi/pickup-ion.md]).

    Steps at or below it are scored by the mean relative error, steps above it
    by the past-peak ratio.

    Parameters
    ----------
    esa_energies : ndarray of floats
        The (n_steps,) modeled coarse-step ESA energies.
    chunk_mean_model_rates : ndarray of floats
        The (n_steps,) chunk-mean model count rates, *excluding* the
        background rate.

    Returns
    -------
    float
        The highest ESA energy whose model rate is at least
        CUTOFF_DROP_RATIO times the peak model rate.
    """
    peak_model_rate = chunk_mean_model_rates.max()
    return esa_energies[
        chunk_mean_model_rates >= peak_model_rate * CUTOFF_DROP_RATIO
    ].max()


def goodness_of_fit_metrics(
    esa_energies: NDArray, model_rates: NDArray, observed_rates: NDArray
) -> tuple[float, float]:
    """
    The PUI goodness-of-fit metrics (see [docs/swapi/pickup-ion.md]).

    Parameters
    ----------
    esa_energies : ndarray of floats
        The (n_steps,) modeled coarse-step ESA energies, starting at the lower
        edge of the fitting range.
    model_rates : ndarray of floats
        The (n_sweeps, n_steps) array of model count rates, *excluding* the
        background rate.
    observed_rates : ndarray of floats
        The (n_sweeps, n_steps) array of observed count rates.

    Returns
    -------
    mean_relative_error : float
        The mean relative error Delta_rel over the steps at or below E_{1/4}.
    past_peak_ratio : float
        The past-peak ratio R_past_peak over the steps above E_{1/4}.
    """
    chunk_mean_model_rates = model_rates.mean(axis=0)
    chunk_mean_observed_rates = observed_rates.mean(axis=0)

    upper_energy_limit = goodness_of_fit_upper_energy(
        esa_energies, chunk_mean_model_rates
    )
    past_range = esa_energies > upper_energy_limit
    in_range = ~past_range

    mean_absolute_percent_error = (
        np.abs(
            chunk_mean_model_rates + SWAPI_BACKGROUND_RATE - chunk_mean_observed_rates
        )
        / (chunk_mean_model_rates + SWAPI_BACKGROUND_RATE)
    )[in_range].mean()

    past_cutoff_ratio = (
        chunk_mean_observed_rates[past_range].mean() / chunk_mean_model_rates.max()
    )

    return float(mean_absolute_percent_error), float(past_cutoff_ratio)


def is_good_fit(
    esa_energies: NDArray,
    model_rates: NDArray,
    observed_rates: NDArray,
    cutoff_speed_kms: float,
    sw_speed_kms: float,
    ionization_rate: float,
) -> bool:
    """
    Evaluate whether a PUI fit is a good fit.

    For the goodness of fit criteria, see [docs/swapi/pickup-ion.md].

    The ESA steps should start at the lower edge of the fitting range.
    The upper edge should not be applied.

    Parameters
    ----------
    esa_energies : ndarray of floats
        The (n_steps,) modeled coarse-step ESA energies.
    model_rates : ndarray of floats
        The (n_sweeps, n_steps) array of model count rates, *excluding* the
        background rate.
    observed_rates : ndarray of floats
        The (n_sweeps, n_steps) array of observed count rates.
    cutoff_speed_kms : scalar float
        The model cutoff speed in km/s.
    sw_speed_kms : scalar float
        The chunk-mean solar wind bulk speed in km/s, used to judge the fitted
        cutoff speed as a ratio of the bulk speed.
    ionization_rate : scalar float
        The model ionization rate at 1 AU in s^-1.
    """
    if esa_energies.ndim != 1:
        raise ValueError(esa_energies.shape)

    if model_rates.shape != (len(model_rates), len(esa_energies)):
        raise ValueError(model_rates.shape)

    if observed_rates.shape != model_rates.shape:
        raise ValueError(observed_rates.shape)

    mean_absolute_percent_error, past_cutoff_ratio = goodness_of_fit_metrics(
        esa_energies, model_rates, observed_rates
    )

    cutoff_speed_ratio = cutoff_speed_kms / sw_speed_kms

    return (
        (cutoff_speed_kms <= MAX_CUTOFF_SPEED_KMS)
        and (mean_absolute_percent_error <= MAX_MEAN_RELATIVE_ERROR)
        and (past_cutoff_ratio <= MAX_PAST_PEAK_RATIO)
        and (MIN_CUTOFF_SPEED_RATIO <= cutoff_speed_ratio <= MAX_CUTOFF_SPEED_RATIO)
        and (MIN_IONIZATION_RATE <= ionization_rate <= MAX_IONIZATION_RATE)
    )
