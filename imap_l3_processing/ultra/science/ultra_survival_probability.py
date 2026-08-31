from typing import Optional

import numpy as np
import spiceypy
from astropy.units import Quantity
from astropy_healpix import HEALPix, npix_to_nside
from imap_processing.ena_maps.ena_maps import UltraPointingSet, HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.ultra.l2.ultra_l2 import bin_pset_energy_bins
from xarray import Dataset

from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.ultra.models import UltraL1CPSet, UltraGlowsL3eData


class UltraSurvivalProbability(UltraPointingSet):
    def __init__(self, l1c_pset: UltraL1CPSet, l3e_glows: UltraGlowsL3eData,
                 bin_groups: Optional[np.ndarray] = None):
        coarse_bins = bin_pset_energy_bins(l1c_pset.to_xarray(), bin_groups)
        super().__init__(coarse_bins, geometry.SpiceFrame.IMAP_DPS)

        predicted_ephemeris = (
            l3e_glows.flags & GlowsL3Flags.PREDICTIVE_EPHEMERIS > 0
        )
        nominal_alpha_proton_ratio = (
                l3e_glows.flags & GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO > 0
        )
        persisted_last_point = (
                l3e_glows.flags & GlowsL3Flags.PERSISTED_LAST_POINT > 0
        )

        l1c_epoch_in_et = spiceypy.unitim(
            self.data.coords[CoordNames.TIME.value].values[0] / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
        rotated_az_el_points = geometry.frame_transform_az_el(l1c_epoch_in_et, self.az_el_points,
                                                              geometry.SpiceFrame.IMAP_DPS,
                                                              geometry.SpiceFrame.ECLIPJ2000)
        glows_nside = npix_to_nside(len(l3e_glows.healpix_index))
        glows_healpix = HEALPix(nside=glows_nside)

        npixels = len(l1c_pset.healpix_index)

        spatially_interpolated_sp = np.zeros((len(l3e_glows.energy), npixels))

        for energy_index in range(len(l3e_glows.energy)):
            spatially_interpolated_sp[energy_index, :] = glows_healpix.interpolate_bilinear_lonlat(
                Quantity(rotated_az_el_points[:, 0], unit='deg'),
                Quantity(rotated_az_el_points[:, 1], unit='deg'),
                l3e_glows.survival_probability[0, energy_index, :])

        energy_interpolated_sp = np.zeros((len(coarse_bins.energy_bin_geometric_mean), npixels))
        for healpix_index in range(npixels):
            energy_interpolated_sp[:, healpix_index] = np.interp(
                np.log10(coarse_bins.energy_bin_geometric_mean),
                np.log10(l3e_glows.energy),
                spatially_interpolated_sp[:, healpix_index]
            )

        self.data["survival_probability_times_exposure"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            energy_interpolated_sp * coarse_bins.exposure_factor.data
        )

        self.data["predicted_ephemeris_flag"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            predicted_ephemeris * coarse_bins.exposure_factor.data
        )

        self.data["nominal_alpha_proton_ratio_flag"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            nominal_alpha_proton_ratio * coarse_bins.exposure_factor.data
        )

        self.data["persisted_last_point_flag"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            persisted_last_point * coarse_bins.exposure_factor.data
        )

        self.data["epoch_delta"] = [l1c_pset.epoch_delta]


class UltraSurvivalProbabilitySkyMap(HealpixSkyMap):
    def __init__(self, sp: list[UltraSurvivalProbability], spice_frame: geometry.SpiceFrame, nside: int):
        super().__init__(nside, spice_frame)
        for sp_pset in sp:
            values_to_project = ["survival_probability_times_exposure", "exposure_factor", "predicted_ephemeris_flag",
                      "nominal_alpha_proton_ratio_flag", "persisted_last_point_flag"]
            self.project_pset_values_to_map(sp_pset, values_to_project,
                                            pset_valid_mask=np.isfinite(
                                                sp_pset.data["survival_probability_times_exposure"]))

        predicted_ephemeris_set = self.data_1d["predicted_ephemeris_flag"] > 0
        nominal_proton_alpha_ratio_set = self.data_1d["nominal_alpha_proton_ratio_flag"] > 0
        persisted_last_point_set = self.data_1d["persisted_last_point_flag"] > 0

        quality_flags = (predicted_ephemeris_set * MapL3Flags.PREDICTIVE_EPHEMERIS) | \
                        (nominal_proton_alpha_ratio_set * MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO) | \
                        (persisted_last_point_set * MapL3Flags.PERSISTED_LAST_POINT)

        self.data_1d = Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure_factor"],
            "quality_flags": quality_flags
        })
