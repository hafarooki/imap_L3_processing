import numpy as np
from spacepy import pycdf


class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), ("proton efficiency", "f8"), ("alpha efficiency", "f8")])

    def get_proton_efficiency_for(self, time_as_tt2000) -> float:
        return self._get_efficiency_for_index("proton efficiency", time_as_tt2000)

    def get_alpha_efficiency_for(self, time_as_tt2000) -> float:
        return self._get_efficiency_for_index("alpha efficiency", time_as_tt2000)

    @property
    def eps_p_lab(self) -> float:
        """Proton efficiency at the lab calibration epoch.

        TODO: read from a `lab_time` field in the LUT once the cal-file format owner adds it.
        Interim: pin to the first entry whose timestamp is on or after 2025-11-01 — the
        pre-2025-11 rows in the current LUT are placeholder values (0.02348 repeated)
        and using them as the lab denominator drives the proton-fit density 6× too low."""
        cutoff = np.datetime64("2025-11-01", "ns")
        for d in self.data:
            if d["time"] >= cutoff:
                return float(d["proton efficiency"])
        return float(self.data[0]["proton efficiency"])

    def _get_efficiency_for_index(self, name, time_as_tt2000) -> float:
        for d in reversed(self.data):
            if d["time"] < np.datetime64(pycdf.lib.tt2000_to_datetime(int(time_as_tt2000)), "ns"):
                return d[name]

        raise ValueError(f"No efficiency data for {pycdf.lib.tt2000_to_datetime(time_as_tt2000)}")
