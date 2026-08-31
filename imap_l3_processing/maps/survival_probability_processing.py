from typing import Union, TypeVar

import numpy as np
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import ReferenceFrame
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData, \
    InputRectangularPointingSet, GlowsL3eRectangularMapInputData
from imap_l3_processing.maps.rectangular_survival_probability import RectangularSurvivalProbabilityPointingSet, \
    RectangularSurvivalProbabilitySkyMap
from imap_l3_processing.ultra.models import UltraGlowsL3eData, UltraL1CPSet

L1CPointingSet = TypeVar("L1CPointingSet", bound=Union[InputRectangularPointingSet, UltraL1CPSet])
GlowsL3eData = TypeVar("GlowsL3eData", bound=Union[GlowsL3eRectangularMapInputData, UltraGlowsL3eData])


def combine_glows_l3e_with_l1c_pointing(glows_l3e_data: list[GlowsL3eData], l1c_data: list[L1CPointingSet]) -> list[
    tuple[L1CPointingSet, GlowsL3eData]]:
    l1c_by_repoint = {l1c.repointing: l1c for l1c in l1c_data}
    glows_by_repoint = {l3e.repointing: l3e for l3e in glows_l3e_data}

    return [(l1c_by_repoint[repoint], glows_by_repoint[repoint])
            for repoint in l1c_by_repoint.keys() if repoint in glows_by_repoint]

def filter_bad_days(input_psets: list[L1CPointingSet]) -> list[L1CPointingSet]:
    return [pset for pset in input_psets if not np.all(pset.exposure_times == 0.0)]

def process_survival_probabilities(survival_probabilities_dependencies: HiLoL3SurvivalDependencies,
                                   spice_frame_name: SpiceFrame, cg_corrected: bool = None) \
        -> RectangularIntensityMapData:
    l2_descriptor_parts = survival_probabilities_dependencies.l2_map_descriptor_parts

    l1c_data = filter_bad_days(survival_probabilities_dependencies.l1c_data)

    combined_glows = combine_glows_l3e_with_l1c_pointing(survival_probabilities_dependencies.glows_l3e_data, l1c_data)
    pointing_sets = []

    if cg_corrected is None:
        cg_corrected = l2_descriptor_parts.reference_frame == ReferenceFrame.Heliospheric

    for l1c, glows_l3e in combined_glows:
        pointing_sets.append(RectangularSurvivalProbabilityPointingSet(
            l1c, l2_descriptor_parts.sensor, l2_descriptor_parts.spin_phase, glows_l3e,
            survival_probabilities_dependencies.l2_data.intensity_map_data.energy, cg_corrected=cg_corrected))
    assert len(pointing_sets) > 0

    survival_sky_map = RectangularSurvivalProbabilitySkyMap(pointing_sets, int(l2_descriptor_parts.grid),
                                                            spice_frame_name)

    survival_dataset = survival_sky_map.to_dataset()

    input_data = survival_probabilities_dependencies.l2_data.intensity_map_data
    survival_probabilities = survival_dataset["exposure_weighted_survival_probabilities"].values
    survival_corrected_intensity = input_data.ena_intensity / survival_probabilities
    corrected_stat_uncert = input_data.ena_intensity_stat_uncert / survival_probabilities
    corrected_sys_err = input_data.ena_intensity_sys_err / survival_probabilities

    map_data = RectangularIntensityMapData(
        intensity_map_data=IntensityMapData(
            ena_intensity_stat_uncert=corrected_stat_uncert,
            ena_intensity_sys_err=corrected_sys_err,
            ena_intensity=survival_corrected_intensity,
            epoch=input_data.epoch,
            epoch_delta=input_data.epoch_delta,
            energy=input_data.energy,
            energy_delta_plus=input_data.energy_delta_plus,
            energy_delta_minus=input_data.energy_delta_minus,
            energy_label=input_data.energy_label,
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            exposure_factor=input_data.exposure_factor,
            obs_date=input_data.obs_date,
            obs_date_range=input_data.obs_date_range,
            solid_angle=input_data.solid_angle,
            survival_probability=survival_probabilities,
            quality_flags=survival_dataset["quality_flags"].values,
        ),
        coords=survival_probabilities_dependencies.l2_data.coords,
    )

    if input_data.bg_intensity is not None:
        map_data.intensity_map_data.bg_intensity = input_data.bg_intensity / survival_probabilities
        map_data.intensity_map_data.bg_intensity_sys_err = input_data.bg_intensity_sys_err / survival_probabilities
        map_data.intensity_map_data.bg_intensity_stat_uncert = input_data.bg_intensity_stat_uncert / survival_probabilities

    if input_data.ena_intensity_sys_err_minus is not None:
        map_data.intensity_map_data.ena_intensity_sys_err_minus = input_data.ena_intensity_sys_err_minus / survival_probabilities
        map_data.intensity_map_data.ena_intensity_sys_err_plus = input_data.ena_intensity_sys_err_plus / survival_probabilities
    
    return map_data
