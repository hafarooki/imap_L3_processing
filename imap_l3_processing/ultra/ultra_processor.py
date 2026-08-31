import logging
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Optional

from imap_data_access import ScienceFilePath, ImapFilePath
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.map_combination import (
    ExposureWeightedCombination,
    UncertaintyWeightedCombination,
)
from imap_l3_processing.maps.map_descriptors import (
    MapDescriptorParts,
    MapQuantity,
    SurvivalCorrection,
    parse_map_descriptor,
    PixelSize,
    Sensor,
)
from imap_l3_processing.maps.map_models import (
    HealPixIntensityMapData,
    IntensityMapData,
    HealPixCoords,
    RectangularIntensityDataProduct,
    RectangularIntensityMapData,
    RectangularCoords,
    RectangularSpectralIndexDataProduct,
    RectangularSpectralIndexMapData,
)
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.maps.spectral_fit import calculate_spectral_index_for_multiple_ranges, \
    slice_energy_range_by_bin, fit_spectral_index_map
from imap_l3_processing.maps.survival_probability_processing import combine_glows_l3e_with_l1c_pointing
from imap_l3_processing.ultra.science.ultra_survival_probability import UltraSurvivalProbabilitySkyMap, \
    UltraSurvivalProbability
from imap_l3_processing.ultra.ultra_l3_dependencies import (
    UltraL3Dependencies,
    UltraL3SpectralIndexDependencies,
    UltraL3CombinedDependencies,
)
from imap_l3_processing.utils import (
    save_data,
    get_temp_cache_dir,
)

logger = logging.getLogger(__name__)


class UltraProcessor(MapProcessor):
    def process(self, spice_frame_name: SpiceFrame = SpiceFrame.ECLIPJ2000):
        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        parent_file_names = self.get_parent_file_names()

        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                ultra_l3_spectral_fit_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(
                    self.dependencies)
                spectral_index_map_data = self._process_spectral_index(
                    ultra_l3_spectral_fit_dependencies,
                    parsed_descriptor.spectral_index_energy_range
                )
                data_product = RectangularSpectralIndexDataProduct(input_metadata=self.input_metadata,
                                                                   data=spectral_index_map_data,
                                                                   spice_frame_name=spice_frame_name)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Ultra45 | Sensor.Ultra90,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                deps = UltraL3Dependencies.fetch_dependencies(self.dependencies)
                healpix_intensity_map_data = (
                    correct_healpix_data_for_survival_probability(
                        deps, spice_frame_name
                    )
                )
                data_product = self._process_healpix_intensity_to_rectangular(healpix_intensity_map_data,
                                                                              deps.ultra_l2_rectangular_map,
                                                                              parsed_descriptor.grid,
                                                                              spice_frame_name=spice_frame_name)
                data_product.add_paths_to_parents(deps.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.UltraCombined,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                combined_deps = UltraL3CombinedDependencies.fetch_dependencies(
                    self.dependencies
                )
                combined_healpix, combined_rectangular = (
                    self._process_combined_survival_probability(
                        combined_deps, spice_frame_name
                    )
                )
                data_product = self._process_healpix_intensity_to_rectangular(
                    combined_healpix,
                    combined_rectangular,
                    parsed_descriptor.grid,
                    spice_frame_name=spice_frame_name,
                )
                data_product.add_paths_to_parents(
                    combined_deps.u45_dependencies.dependency_file_paths
                )
                data_product.add_paths_to_parents(
                    combined_deps.u90_dependencies.dependency_file_paths
                )

            case MapDescriptorParts(sensor=Sensor.UltraCombined,
                                    survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                    grid=PixelSize.TwoDegrees | PixelSize.FourDegrees | PixelSize.SixDegrees):
                deps = UltraL3CombinedDependencies.fetch_dependencies(self.dependencies)

                combination_strategy = ExposureWeightedCombination()
                combined_healpix = (
                    combination_strategy.combine_healpix_intensity_map_data(
                        [
                            deps.u45_dependencies.ultra_l2_healpix_map,
                            deps.u90_dependencies.ultra_l2_healpix_map,
                        ]
                    )
                )
                combined_rectangular = combination_strategy.combine_rectangular_intensity_map_data(
                    [deps.u45_dependencies.ultra_l2_rectangular_map,
                     deps.u90_dependencies.ultra_l2_rectangular_map])

                data_product = self._process_healpix_intensity_to_rectangular(combined_healpix,
                                                                              combined_rectangular,
                                                                              parsed_descriptor.grid,
                                                                              spice_frame_name=spice_frame_name)
                data_product.add_paths_to_parents(deps.u45_dependencies.dependency_file_paths)
                data_product.add_paths_to_parents(deps.u90_dependencies.dependency_file_paths)
            case _:
                raise NotImplementedError

        data_product.add_filenames_to_parents(parent_file_names)
        return [save_data(data_product)]

    def _process_combined_survival_probability(
            self, deps: UltraL3CombinedDependencies, spice_frame_name: SpiceFrame,
    ) -> tuple[HealPixIntensityMapData, RectangularIntensityMapData]:
        u45_survival_corrected = correct_healpix_data_for_survival_probability(deps.u45_dependencies, spice_frame_name)
        u90_survival_corrected = correct_healpix_data_for_survival_probability(deps.u90_dependencies, spice_frame_name)

        combination_strategy = UncertaintyWeightedCombination()
        combined_healpix = combination_strategy.combine_healpix_intensity_map_data(
            [u45_survival_corrected, u90_survival_corrected]
        )
        combined_rectangular = combination_strategy.combine_rectangular_intensity_map_data(
            [deps.u45_dependencies.ultra_l2_rectangular_map, deps.u90_dependencies.ultra_l2_rectangular_map]
        )
        return combined_healpix, combined_rectangular

    def _process_spectral_index(self,
                                dependencies: UltraL3SpectralIndexDependencies,
                                spectral_index_range: Optional[tuple[int, int]]
                                ) -> RectangularSpectralIndexMapData:
        if spectral_index_range is not None:
            start, end = spectral_index_range
            sliced = slice_energy_range_by_bin(dependencies.map_data.intensity_map_data, start, end)
            map_data = fit_spectral_index_map(sliced)
        else:
            map_data = calculate_spectral_index_for_multiple_ranges(
                dependencies.map_data.intensity_map_data,
                dependencies.get_fit_energy_ranges(),
            )
        return RectangularSpectralIndexMapData(
            spectral_index_map_data=map_data, coords=dependencies.map_data.coords
        )

    def _process_healpix_intensity_to_rectangular(
        self,
        healpix_map_data: HealPixIntensityMapData,
        rect_l2_map: RectangularIntensityMapData,
        spacing_deg: int,
        spice_frame_name: SpiceFrame,
    ) -> RectangularIntensityDataProduct:
        has_survival_data = (
            healpix_map_data.intensity_map_data.survival_probability is not None
        )

        variables_to_convert_to_rectangular = [
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "predicted_ephemeris_flag",
            "nominal_alpha_proton_ratio_flag",
            "persisted_last_point_flag",
        ]

        if has_survival_data:
            variables_to_convert_to_rectangular.append("survival_probability")

        healpix_map = healpix_map_data.to_healpix_skymap()
        rectangular_map, _ = healpix_map.to_rectangular_skymap(spacing_deg, variables_to_convert_to_rectangular)
        rectangular_map_xarray_dataset = rectangular_map.to_dataset()

        rect_l2_data = rect_l2_map.intensity_map_data

        predicted_ephemeris_set = rectangular_map_xarray_dataset["predicted_ephemeris_flag"].values > 0
        nominal_alpha_proton_ratio_set = rectangular_map_xarray_dataset["nominal_alpha_proton_ratio_flag"].values > 0
        persisted_last_point_set = rectangular_map_xarray_dataset["persisted_last_point_flag"].values > 0

        quality_flags = (predicted_ephemeris_set * MapL3Flags.PREDICTIVE_EPHEMERIS) | \
                        (nominal_alpha_proton_ratio_set * MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO) | \
                        (persisted_last_point_set * MapL3Flags.PERSISTED_LAST_POINT)

        intensity_map_data = IntensityMapData(
            epoch=rect_l2_data.epoch,
            epoch_delta=rect_l2_data.epoch_delta,
            energy=rect_l2_data.energy,
            energy_delta_plus=rect_l2_data.energy_delta_plus,
            energy_delta_minus=rect_l2_data.energy_delta_minus,
            energy_label=rect_l2_data.energy_label,
            latitude=rect_l2_data.latitude,
            longitude=rect_l2_data.longitude,
            obs_date=rect_l2_data.obs_date,
            obs_date_range=rect_l2_data.obs_date_range,
            solid_angle=rect_l2_data.solid_angle,
            exposure_factor=rect_l2_data.exposure_factor,
            ena_intensity=rectangular_map_xarray_dataset["ena_intensity"].values,
            ena_intensity_stat_uncert=rectangular_map_xarray_dataset[
                "ena_intensity_stat_uncert"
            ].values,
            ena_intensity_sys_err=rectangular_map_xarray_dataset[
                "ena_intensity_sys_err"
            ].values,
            quality_flags=quality_flags,
        )

        if has_survival_data:
            intensity_map_data.survival_probability = rectangular_map_xarray_dataset["survival_probability"].values

        rect_intensity_map_data = RectangularIntensityMapData(
            intensity_map_data,
            coords=RectangularCoords(
                latitude_delta=rect_l2_map.coords.latitude_delta,
                latitude_label=rect_l2_map.coords.latitude_label,
                longitude_delta=rect_l2_map.coords.longitude_delta,
                longitude_label=rect_l2_map.coords.longitude_label,
            ),
        )

        return RectangularIntensityDataProduct(data=rect_intensity_map_data, input_metadata=self.input_metadata,
                                               spice_frame_name=spice_frame_name)

@dataclass
class UltraMapDescriptorParts:
    grid_size: int

def _make_cache_key_for_sp_corrected_data(deps: UltraL3Dependencies) -> str:
    pset_names = []
    glows_names = []
    for path in deps.dependency_file_paths:
        try:
            sfp = ScienceFilePath(path)
            if sfp.instrument == "ultra":
                if sfp.data_level == "l1c":
                    pset_names.append(path.name)
                elif sfp.data_level == "l2":
                    l2_descriptor = re.sub(r"[246]deg", "nside32", sfp.descriptor)
            elif sfp.instrument == "glows" and sfp.data_level == "l3e":
                glows_names.append(path.name)
        except ImapFilePath.InvalidImapFileError:
            pass
    return str(sorted(pset_names)) + str(sorted(glows_names)) + l2_descriptor

def correct_healpix_data_for_survival_probability(
    deps: UltraL3Dependencies, spice_frame_name: SpiceFrame
) -> HealPixIntensityMapData:
    cache_key = _make_cache_key_for_sp_corrected_data(deps)
    cache_filename = sha256(cache_key.encode("utf-8")).hexdigest()
    cache_path = get_temp_cache_dir() / cache_filename
    if cache_path.exists():
        logger.info("Loading cached HEALPix SP-corrected data from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Cached HEALPix SP-corrected data not found; building from inputs")

    combined_psets = combine_glows_l3e_with_l1c_pointing(deps.glows_l3e_sp, deps.ultra_l1c_pset)
    survival_probability_psets = [
        UltraSurvivalProbability(_l1c, _l3e, bin_groups=deps.energy_bin_group_sizes)
        for _l1c, _l3e in combined_psets
    ]

    intensity_data = deps.ultra_l2_healpix_map.intensity_map_data
    coords = deps.ultra_l2_healpix_map.coords
    corrected_skymap = UltraSurvivalProbabilitySkyMap(
        survival_probability_psets, spice_frame_name, coords.nside
    )
    skymap_dataset = corrected_skymap.to_dataset()
    survival_probability_map = skymap_dataset["exposure_weighted_survival_probabilities"].values

    healpix_intensity_data = HealPixIntensityMapData(
        intensity_map_data=IntensityMapData(
            ena_intensity=intensity_data.ena_intensity / survival_probability_map,
            ena_intensity_stat_uncert=intensity_data.ena_intensity_stat_uncert
            / survival_probability_map,
            ena_intensity_sys_err=intensity_data.ena_intensity_sys_err
            / survival_probability_map,
            epoch=intensity_data.epoch,
            epoch_delta=intensity_data.epoch_delta,
            energy=intensity_data.energy,
            energy_delta_plus=intensity_data.energy_delta_plus,
            energy_delta_minus=intensity_data.energy_delta_minus,
            energy_label=intensity_data.energy_label,
            latitude=intensity_data.latitude,
            longitude=intensity_data.longitude,
            exposure_factor=intensity_data.exposure_factor,
            obs_date=intensity_data.obs_date,
            obs_date_range=intensity_data.obs_date_range,
            solid_angle=intensity_data.solid_angle,
            survival_probability=survival_probability_map,
            quality_flags=skymap_dataset["quality_flags"].values,
        ),
        coords=HealPixCoords(
            pixel_index=coords.pixel_index,
            pixel_index_label=coords.pixel_index_label,
        ),
    )
    logger.info("Saving HEALPix SP-corrected data to cache at %s", cache_path)

    with open(cache_path, "wb") as f:
        pickle.dump(healpix_intensity_data, f)

    return healpix_intensity_data