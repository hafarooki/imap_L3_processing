"""Shared helpers for the SWAPI documentation figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imap_l3_processing.swapi.response.swapi_response import SwapiResponse

_INSTRUMENT_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "instrument_team_data" / "swapi"
)


def load_swapi_response() -> SwapiResponse:
    return SwapiResponse.from_files(
        _INSTRUMENT_DATA_DIR / "imap_swapi_azimuthal-transmission_20260425_v001.csv",
        _INSTRUMENT_DATA_DIR / "imap_swapi_central-effective-area_20260425_v001.csv",
        _INSTRUMENT_DATA_DIR / "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
    )
