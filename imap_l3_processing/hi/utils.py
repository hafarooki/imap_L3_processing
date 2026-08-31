from pathlib import Path
from typing import Union

from imap_data_access import ScienceFilePath
from imap_processing.spice.time import met_to_ttj2000ns
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import (
    PROBABILITY_OF_SURVIVAL_VAR_NAME,
    EPOCH_CDF_VAR_NAME,
    ENERGY_VAR_NAME,
    SPIN_ANGLE_VAR_NAME,
    GLOWS_FLAGS_VAR_NAME,
)
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet


def read_l1c_rectangular_pointing_set_data(path: Union[Path, str]) -> InputRectangularPointingSet:
    repointing = ScienceFilePath(path).repointing
    with CDF(str(path)) as cdf:
        exposure_time_variable = cdf['exposure_time'] if 'exposure_time' in cdf else cdf['exposure_times']

        assert ("epoch_delta" in cdf) or (
                "pointing_start_met" in cdf and "pointing_end_met" in cdf), "Expected the pset to have either epoch_delta or pointing_start_met and pointing_end_met defined"

        pointing_start_met = cdf["pointing_start_met"][...] if "pointing_start_met" in cdf else None
        pointing_end_met = cdf["pointing_end_met"][...] if "pointing_end_met" in cdf else None
        epoch_delta = cdf["epoch_delta"][...] if "epoch_delta" in cdf else met_to_ttj2000ns(
            pointing_end_met) - met_to_ttj2000ns(pointing_start_met)
        return InputRectangularPointingSet(epoch=cdf["epoch"][0],
                                           epoch_delta=epoch_delta,
                                           epoch_j2000=cdf.raw_var("epoch")[...],
                                           repointing=repointing,
                                           exposure_times=exposure_time_variable[...],
                                           esa_energy_step=cdf["esa_energy_step"][...],
                                           pointing_start_met=pointing_start_met,
                                           pointing_end_met=pointing_end_met,
                                           hae_longitude=cdf['hae_longitude'][...],
                                           hae_latitude=cdf['hae_latitude'][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> GlowsL3eRectangularMapInputData:
    repointing = ScienceFilePath(cdf_path).repointing
    with CDF(str(cdf_path)) as cdf:
        return GlowsL3eRectangularMapInputData(
            epoch=cdf[EPOCH_CDF_VAR_NAME][0],
            epoch_j2000=cdf.raw_var(EPOCH_CDF_VAR_NAME)[...],
            repointing=repointing,
            energy=read_numeric_variable(cdf[ENERGY_VAR_NAME]),
            spin_angle=read_numeric_variable(cdf[SPIN_ANGLE_VAR_NAME]),
            probability_of_survival=read_numeric_variable(
                cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME]
            ),
            flags=cdf[GLOWS_FLAGS_VAR_NAME][...],
        )
