import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)


def calculate_coincidence_rate(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
) -> NDArray:
    speed_in_sw_frame = chunk_response.speed_in_sw_frame
    bin_weights = chunk_response.bin_weights

    f_pui = vasyliunas_siscoe_distribution.f(
        speed_in_sw_frame, fitting_params, apply_cutoff=False
    )
    apply_partial_heaviside_at_cutoff(f_pui, speed_in_sw_frame, fitting_params.cutoff_speed)

    return np.tensordot(bin_weights, f_pui, axes=([2], [0]))


def apply_partial_heaviside_at_cutoff(
    f_pui: NDArray, speed_in_sw_frame: NDArray, cutoff_speed: float
) -> None:
    delta_v_prime = speed_in_sw_frame[1] - speed_in_sw_frame[0]
    cutoff_index = round((cutoff_speed - speed_in_sw_frame[0]) / delta_v_prime)

    if cutoff_index < 0:
        f_pui[:] = 0.0
        return

    if cutoff_index > speed_in_sw_frame.size - 1:
        return

    # Scale the cutoff cell by the fraction of its width below the cutoff and
    # zero everything above. This also covers cutoff_index == size - 1 (the
    # cutoff landing on the last grid node), where the cell is still partial
    # (~half) and `f_pui[cutoff_index + 1:]` is an empty, no-op slice.
    bin_min = speed_in_sw_frame[cutoff_index] - delta_v_prime / 2
    fraction_below_cutoff = (cutoff_speed - bin_min) / delta_v_prime
    f_pui[cutoff_index] *= fraction_below_cutoff
    f_pui[cutoff_index + 1:] = 0.0
