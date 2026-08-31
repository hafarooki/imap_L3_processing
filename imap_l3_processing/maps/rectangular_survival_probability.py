import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import apply_compton_getting_correction, \
    add_spacecraft_position_and_velocity_to_pset, calculate_ram_mask
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.maps.map_descriptors import Sensor, SpinPhase
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet
from imap_l3_processing.maps.quality_flags import MapL3Flags


def interpolate_angular_data_to_nearest_neighbor(input_azimuths: np.array, glows_azimuths: np.array,
                                                 glows_data: np.array) -> np.array:
    expanded_az = np.concatenate([glows_azimuths - 360, glows_azimuths, glows_azimuths + 360])
    expanded_glows_data = np.concatenate([glows_data, glows_data, glows_data], axis=1)
    sort = np.argsort(expanded_az)
    sorted_az = expanded_az[sort]
    sorted_data = expanded_glows_data[:, sort]
    bin_edges = sorted_az[:-1] + np.diff(sorted_az) / 2
    return sorted_data[:, np.digitize(input_azimuths, bin_edges, right=True)]


class RectangularSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: InputRectangularPointingSet, sensor: Sensor, spin_phase: SpinPhase,
                 glows_dataset: GlowsL3eRectangularMapInputData, energies: np.ndarray, cg_corrected: bool = False):
        num_spin_angle_bins = l1c_dataset.exposure_times.shape[-1]

        deg_spacing = 360 / num_spin_angle_bins
        half_bin_width = deg_spacing / 2

        self.azimuths = np.linspace(0, 360, num_spin_angle_bins,
                                    endpoint=False) + half_bin_width

        sensor_angle = Sensor.get_sensor_angle(sensor)
        self.elevations = np.repeat(sensor_angle, num_spin_angle_bins)

        hae_az_el_points = xr.DataArray(
            np.column_stack([l1c_dataset.hae_longitude[0], l1c_dataset.hae_latitude[0]]),
            dims=[CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
        )

        self.spatial_coords = (CoordNames.AZIMUTH_L1C.value,)

        if l1c_dataset.epoch_delta is not None:
            repointing_midpoint = l1c_dataset.epoch_j2000 + l1c_dataset.epoch_delta / 2
        elif l1c_dataset.pointing_start_met is not None and l1c_dataset.pointing_end_met is not None:
            pointing_duration = l1c_dataset.pointing_end_met - l1c_dataset.pointing_start_met
            repointing_midpoint = l1c_dataset.epoch_j2000 + (1e9 * pointing_duration / 2)

        initial_dataset = xr.Dataset({},
                                     coords={
                                         CoordNames.TIME.value: l1c_dataset.epoch_j2000,
                                         CoordNames.ENERGY_ULTRA_L1C.value: l1c_dataset.esa_energy_step,
                                         CoordNames.AZIMUTH_L1C.value: self.azimuths,
                                     })

        initial_dataset['epoch'] = l1c_dataset.epoch_j2000
        initial_dataset['epoch_delta'] = l1c_dataset.epoch_delta
        initial_dataset['hae_longitude'] = xr.DataArray(
            l1c_dataset.hae_longitude,
            dims=[CoordNames.TIME.value, CoordNames.AZIMUTH_L1C.value],
        )
        initial_dataset['hae_latitude'] = xr.DataArray(
            l1c_dataset.hae_latitude,
            dims=[CoordNames.TIME.value, CoordNames.AZIMUTH_L1C.value],
        )
        initial_dataset['pointing_start_met'] = l1c_dataset.pointing_start_met
        initial_dataset['pointing_end_met'] = l1c_dataset.pointing_end_met

        if sensor == Sensor.Hi90 or sensor == Sensor.Hi45:
            initial_dataset.attrs['Logical_source'] = 'imap_hi'
        elif sensor in (Sensor.Lo75, Sensor.Lo90, Sensor.Lo105):
            initial_dataset.attrs['Logical_source'] = 'imap_lo'
        else:
            raise ValueError("Unexpected sensor when performing survival probability correction!", sensor.name)
        dataset = add_spacecraft_position_and_velocity_to_pset(initial_dataset)

        if cg_corrected:
            energy_in_ev = energies * 1000

            dataset = apply_compton_getting_correction(
                dataset,
                xr.DataArray(energy_in_ev, dims=[CoordNames.ENERGY_ULTRA_L1C.value])
            )
            self.az_el_points = xr.DataArray(
                np.stack([dataset['hae_longitude'].values[0], dataset['hae_latitude'].values[0]], axis=2),
                dims=[CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
            )

            spacecraft_frame_energies_in_kev = dataset["energy_sc"].values / 1000.0

            exposure = np.full_like(l1c_dataset.exposure_times, np.nan)
            for (energy_i, spin_angle_i), cg_energy in np.ndenumerate(dataset['energy_sc'].values[0]):
                distance_to_bins = np.abs(np.log10(energy_in_ev) - np.log10(cg_energy))
                closest_energy_bin_index = np.argmin(distance_to_bins)

                exposure[0, energy_i, spin_angle_i] = l1c_dataset.exposure_times[0, closest_energy_bin_index, spin_angle_i]
        else:
            self.az_el_points = hae_az_el_points
            exposure = l1c_dataset.exposure_times

        dataset = calculate_ram_mask(dataset)

        if spin_phase == SpinPhase.RamOnly:
            dataset["directional_mask"] = dataset["ram_mask"]
        else:
            dataset["directional_mask"] = ~dataset["ram_mask"]

        if cg_corrected:
            sp_interpolated_to_pset_angles = interpolate_angular_data_to_nearest_neighbor(
                self.azimuths, glows_dataset.spin_angle, glows_dataset.probability_of_survival[0])
            log_sc_frame_energies = np.log10(spacecraft_frame_energies_in_kev[0])

            sp_final = np.empty((1, len(energies), num_spin_angle_bins))
            for spin_angle_index in range(num_spin_angle_bins):
                sp_final[0, :, spin_angle_index] = np.interp(
                    log_sc_frame_energies[:, spin_angle_index],
                    np.log10(glows_dataset.energy),
                    sp_interpolated_to_pset_angles[:, spin_angle_index]
                )
        else:
            glows_spin_bin_count = len(glows_dataset.spin_angle)
            sp_interpolated_to_hi_energies = np.empty(shape=(len(energies), glows_spin_bin_count))
            for spin_angle_index in range(glows_spin_bin_count):
                sp_interpolated_to_hi_energies[:, spin_angle_index] = np.interp(
                    np.log10(energies),
                    np.log10(glows_dataset.energy),
                    glows_dataset.probability_of_survival[0, :, spin_angle_index],
                )

            sp_interpolated_to_pset_angles = np.zeros(
                (1, len(energies), num_spin_angle_bins)
            )
            sp_interpolated_to_pset_angles[0] = (
                interpolate_angular_data_to_nearest_neighbor(
                    self.azimuths,
                    glows_dataset.spin_angle,
                    sp_interpolated_to_hi_energies,
                )
            )
            sp_final = sp_interpolated_to_pset_angles
        flag_value = glows_dataset.flags[0]

        dataset["survival_probability_times_exposure"] = xr.DataArray(
            sp_final * exposure,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ]
        )
        dataset["exposure"] = xr.DataArray(
            exposure,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ],
        )
        dataset["epoch"] = repointing_midpoint
        predict_flag_set = flag_value & GlowsL3Flags.PREDICTIVE_EPHEMERIS != 0
        dataset["predicted_ephemeris_flag"] = xr.DataArray(
            exposure * predict_flag_set,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ],
        )
        nominal_alpha_proton_flag_set = flag_value & GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO != 0
        dataset["nominal_alpha_proton_ratio_flag"] = xr.DataArray(
            exposure * nominal_alpha_proton_flag_set,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ]
        )

        persisted_last_point_flag_set = flag_value & GlowsL3Flags.PERSISTED_LAST_POINT != 0
        dataset["persisted_last_point_flag"] = xr.DataArray(
            exposure * persisted_last_point_flag_set,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ]
        )

        frame = SpiceFrame.IMAP_HAE
        super().__init__(dataset, frame)


class RectangularSurvivalProbabilitySkyMap(RectangularSkyMap):
    def __init__(self, survival_probability_pointing_sets: list[RectangularSurvivalProbabilityPointingSet],
                 spacing_degree: float, spice_frame: SpiceFrame):
        super().__init__(spacing_degree, spice_frame)
        for  sp_pset in survival_probability_pointing_sets:
            value_keys = ["survival_probability_times_exposure", "exposure", "predicted_ephemeris_flag", "nominal_alpha_proton_ratio_flag", "persisted_last_point_flag"]
            self.project_pset_values_to_map(sp_pset, value_keys, pset_valid_mask=sp_pset.data["directional_mask"])

        predicted_ephemeris_set = self.data_1d["predicted_ephemeris_flag"] != 0
        nominal_alpha_proton_ratio_set = self.data_1d["nominal_alpha_proton_ratio_flag"] != 0
        persisted_last_point_set = self.data_1d["persisted_last_point_flag"] != 0
        quality_flags_1d = (predicted_ephemeris_set * MapL3Flags.PREDICTIVE_EPHEMERIS) | \
                           (nominal_alpha_proton_ratio_set * MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO) | \
                           (persisted_last_point_set * MapL3Flags.PERSISTED_LAST_POINT)

        self.data_1d = xr.Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure"],
            "quality_flags": quality_flags_1d,
        })
