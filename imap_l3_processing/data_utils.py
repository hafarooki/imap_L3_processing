from datetime import datetime, timedelta

import numpy as np


def rebin(from_epoch: np.ndarray[float], from_data: np.ndarray[float], to_epoch: np.ndarray[float],
          to_epoch_delta: np.ndarray[float]) -> np.ndarray[float]:
    output_shape = to_epoch.shape + from_data.shape[1:]
    vector_sums = np.zeros(shape=output_shape, dtype=float)
    vector_counts = np.zeros(shape=output_shape, dtype=float)

    input_data_iter = ((time, vec) for time, vec in zip(from_epoch, from_data))
    current_epoch, current_vec = next(input_data_iter, (None, None))
    for i, (time, delta) in enumerate(zip(to_epoch, to_epoch_delta)):
        start_time = time - delta
        end_time = time + delta

        while current_epoch is not None and current_epoch < start_time:
            current_epoch, current_vec = next(input_data_iter, (None, None))

        while current_epoch is not None and start_time <= current_epoch < end_time:
            vector_sums[i] += current_vec
            vector_counts[i] += 1
            current_epoch, current_vec = next(input_data_iter, (None, None))

    return np.divide(vector_sums, vector_counts, out=np.full_like(vector_sums, fill_value=np.nan),
                     where=vector_counts != 0)


class NearestInterpolator:
    def __init__(self,
                 from_epoch: np.ndarray,
                 from_data: np.ndarray,
                 to_epoch: np.ndarray,
                 maximum_distance: timedelta | np.timedelta64
                 ):
        if len(from_data.shape) == 1:
            from_data = from_data.reshape(-1, 1)
        from_data_nan_mask = np.any(np.isnan(from_data), axis=1)
        only_valid_from_epoch = from_epoch[~from_data_nan_mask]
        self.only_valid_from_data = from_data[~from_data_nan_mask]

        from_epoch_as_dt64 = only_valid_from_epoch.astype(np.datetime64)
        to_epoch_as_dt64 = to_epoch.astype(np.datetime64)
        index = np.searchsorted(from_epoch_as_dt64, to_epoch_as_dt64)
        right_indices = np.minimum(len(from_epoch_as_dt64) - 1, index)
        left_indices = np.maximum(0, index - 1)

        right_delta = np.abs(from_epoch_as_dt64[right_indices] - to_epoch_as_dt64)
        left_delta = np.abs(from_epoch_as_dt64[left_indices] - to_epoch_as_dt64)

        self.best_indices = np.where(right_delta < left_delta, right_indices, left_indices)

        min_deltas = np.abs(from_epoch_as_dt64[self.best_indices] - to_epoch_as_dt64)
        self.min_outside_range = min_deltas > maximum_distance

    def interpolate_data(self) -> np.ndarray:
        closest_data = self.only_valid_from_data[self.best_indices].astype(float, copy=True)
        closest_data[self.min_outside_range] = np.nan
        return closest_data

    def interpolate_flags(self, flag_data: np.ndarray) -> np.ndarray:
        closest_flag = flag_data[self.best_indices].astype(bool, copy=True)
        closest_flag[self.min_outside_range] = False
        return closest_flag


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out_array = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, where=denominator != 0, out=out_array)
