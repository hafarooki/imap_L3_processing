from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import xarray as xr
from imap_data_access import ScienceFilePath
from imap_processing.ena_maps.utils.coordinates import CoordNames
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import PROBABILITY_OF_SURVIVAL_VAR_NAME
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import (
    HEALPIX_INDEX_VAR_NAME,
    ENERGY_VAR_NAME,
    EPOCH_CDF_VAR_NAME,
    GLOWS_FLAGS_VAR_NAME,
)


@dataclass
class UltraGlowsL3eData:
    epoch: datetime
    repointing: int
    energy: ndarray
    healpix_index: ndarray
    survival_probability: ndarray
    flags: ndarray

    @classmethod
    def read_from_path(cls, path_to_cdf: Path) -> UltraGlowsL3eData:
        repointing = ScienceFilePath(path_to_cdf).repointing

        with CDF(str(path_to_cdf)) as cdf:
            return UltraGlowsL3eData(
                epoch=cdf[EPOCH_CDF_VAR_NAME][0],
                repointing=repointing,
                energy=read_numeric_variable(cdf[ENERGY_VAR_NAME]),
                healpix_index=cdf[HEALPIX_INDEX_VAR_NAME][...],
                survival_probability=read_numeric_variable(cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME]),
                flags=cdf[GLOWS_FLAGS_VAR_NAME][...]
            )


@dataclass
class UltraL1CPSet:
    epoch: datetime
    epoch_delta: int
    repointing: int
    energy: ndarray
    exposure: ndarray
    healpix_index: ndarray
    latitude: ndarray
    longitude: ndarray
    sensitivity: ndarray

    @classmethod
    def read_from_path(cls, path: Path) -> UltraL1CPSet:
        repointing = ScienceFilePath(path).repointing
        with CDF(str(path)) as cdf:
            return UltraL1CPSet(
                repointing=repointing,
                epoch=cdf[CoordNames.TIME.value][0],
                epoch_delta=cdf["epoch_delta"][0],
                energy=read_numeric_variable(cdf[CoordNames.ENERGY_ULTRA_L1C.value]),
                exposure=read_numeric_variable(cdf["exposure_factor"]),
                latitude=read_numeric_variable(cdf[CoordNames.ELEVATION_L1C.value]).ravel(),
                longitude=read_numeric_variable(cdf[CoordNames.AZIMUTH_L1C.value]).ravel(),
                healpix_index=cdf[CoordNames.HEALPIX_INDEX.value][...],
                sensitivity=read_numeric_variable(cdf["sensitivity"]),
            )

    def to_xarray(self):
        return xr.Dataset(
            {
                "exposure_factor": (
                    [CoordNames.TIME.value,
                     CoordNames.ENERGY_ULTRA_L1C.value,
                     CoordNames.HEALPIX_INDEX.value],
                    self.exposure,
                ),
                "sensitivity": (
                    [
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    self.sensitivity,
                ),
                CoordNames.AZIMUTH_L1C.value: (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.longitude,
                ),
                CoordNames.ELEVATION_L1C.value: (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.latitude,
                ),
            },
            coords={
                CoordNames.TIME.value: [
                    ((self.epoch - TT2000_EPOCH).total_seconds()) * ONE_SECOND_IN_NANOSECONDS + self.epoch_delta / 2,
                ],
                CoordNames.ENERGY_ULTRA_L1C.value: self.energy,
                CoordNames.HEALPIX_INDEX.value: self.healpix_index,
            }
        )
