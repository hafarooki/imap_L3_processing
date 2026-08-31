from typing import Callable

from imap_data_access import SPICEFilePath
from spiceypy import spiceypy, KernelPool, SpiceyError


class PredictedEphemerisTracker:
    kernels_without_predict: list[str]
    used_predict: bool
    predict_kernel_available: bool

    def __init__(self):
        self.used_predict = False
        self.predict_kernel_available = False
        self.kernels_without_predict = []
        for i in range(spiceypy.ktotal("ALL")):
            data = spiceypy.kdata(i, "ALL")
            kernel = data[0]
            try:
                kernel_type = SPICEFilePath(kernel).spice_metadata["type"]
                if kernel_type == "ephemeris_predicted":
                    self.predict_kernel_available = True
                else:
                    self.kernels_without_predict.append(kernel)
            except SPICEFilePath.InvalidImapFileError:
                self.kernels_without_predict.append(kernel)

    def run[T, **P](self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if self.used_predict or not self.predict_kernel_available:
            result = func(*args, **kwargs)
        else:
            try:
                with KernelPool(self.kernels_without_predict):
                    result = func(*args, **kwargs)
            except SpiceyError:
                result = func(*args, **kwargs)
                self.used_predict = True
        return result
