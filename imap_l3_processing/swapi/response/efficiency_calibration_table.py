import numpy as np
from spacepy import pycdf

from imap_l3_processing.swapi.species import Species

_PROTON_EFFICIENCY_COLUMN = "proton efficiency"
_HELIUM_EFFICIENCY_COLUMN = "helium efficiency"

_EFFICIENCY_COLUMN_BY_SPECIES = {
    Species.PROTON: _PROTON_EFFICIENCY_COLUMN,
    Species.ALPHA: _HELIUM_EFFICIENCY_COLUMN,
    Species.HELIUM_PLUS: _HELIUM_EFFICIENCY_COLUMN,
}


class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), (_PROTON_EFFICIENCY_COLUMN, "f8"), (_HELIUM_EFFICIENCY_COLUMN, "f8")], ndmin=1)

    def relative_efficiency(self, time_as_tt2000: int, species: Species) -> float:
        return self._get_entry_for(time_as_tt2000, _EFFICIENCY_COLUMN_BY_SPECIES[species])

    def _get_entry_for(self, time_as_tt2000, column: str) -> float:
        for d in reversed(self.data):
            if d["time"] < np.datetime64(pycdf.lib.tt2000_to_datetime(int(time_as_tt2000)), "ns"):
                return float(d[column])

        raise ValueError(f"No efficiency data for {pycdf.lib.tt2000_to_datetime(time_as_tt2000)}")
