from datetime import datetime
from unittest.mock import sentinel, patch, call, Mock

import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import PixelSize, parse_map_descriptor, ReferenceFrame
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities, \
    combine_glows_l3e_with_l1c_pointing, filter_bad_days
from tests.maps.test_builders import create_rectangular_intensity_map_data, create_l1c_pset, create_l3e_pset
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import create_dataclass_mock


class TestSurvivalProbabilityProcessing(SpiceTestCase):
    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.maps.survival_probability_processing.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.maps.survival_probability_processing.filter_bad_days')
    def test_process_survival_probability(self, mock_filter_bad_days, mock_combine_glows_l3e_with_l1c_pointing,
                                          mock_survival_probability_pointing_set, mock_survival_skymap):
        rng = np.random.default_rng()
        input_map_flux = rng.random((1, 9, 90, 45))
        epoch = datetime.now()

        input_map = create_rectangular_intensity_map_data(epoch=[epoch], flux=input_map_flux,
                                                          include_ena_intensity_sys_err_minus=True)

        intensity_map_data = input_map.intensity_map_data
        input_map.intensity_map_data.energy = sentinel.hi_l2_energies

        l2_grid = PixelSize.FourDegrees
        l2_descriptor_parts = Mock(sensor=sentinel.l2_sensor, spin_phase=sentinel.l2_spin, grid=l2_grid,
                                   reference_frame=ReferenceFrame.Heliospheric)
        dependencies = HiLoL3SurvivalDependencies(l2_data=input_map,
                                                  l1c_data=sentinel.l1c_data,
                                                  glows_l3e_data=sentinel.glows_l3e_data,
                                                  l2_map_descriptor_parts=l2_descriptor_parts,
                                                  dependency_file_paths=[])

        mock_filter_bad_days.return_value = sentinel.filtered_l1c

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        computed_survival_probabilities = rng.random((1, 9, 90, 45))
        quality_flags = np.full((1, 9, 90, 45), MapL3Flags.NONE)
        quality_flags[rng.random((1, 9, 90, 45)) > 0.8] = MapL3Flags.PREDICTIVE_EPHEMERIS
        quality_flags[rng.random((1, 9, 90, 45)) > 0.7] = MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset(
            data_vars={
                "exposure_weighted_survival_probabilities": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.AZIMUTH_L2.value,
                        CoordNames.ELEVATION_L2.value,
                    ],
                    computed_survival_probabilities
                ),
                "quality_flags": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.AZIMUTH_L2.value,
                        CoordNames.ELEVATION_L2.value,
                    ],
                    quality_flags
                ),
            },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.AZIMUTH_L2.value: rng.random((90,)),
                CoordNames.ELEVATION_L2.value: rng.random((45,)),
            }
        )

        survival_data = process_survival_probabilities(dependencies, SpiceFrame.IMAP_DPS)

        mock_filter_bad_days.assert_called_once_with(sentinel.l1c_data)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_data, sentinel.filtered_l1c)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.hi_l1c_1, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_1,
                 sentinel.hi_l2_energies, cg_corrected=True),
            call(sentinel.hi_l1c_2, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_2,
                 sentinel.hi_l2_energies, cg_corrected=True),
            call(sentinel.hi_l1c_3, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_3,
                 sentinel.hi_l2_energies, cg_corrected=True)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     l2_grid,
                                                     SpiceFrame.IMAP_DPS)

        mock_survival_skymap.return_value.to_dataset.assert_called_once()

        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity,
                                      intensity_map_data.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_stat_uncert,
                                      intensity_map_data.ena_intensity_stat_uncert / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_sys_err,
                                      intensity_map_data.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data.intensity_map_data.bg_intensity,
                                      intensity_map_data.bg_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(
            survival_data.intensity_map_data.bg_intensity_sys_err,
            intensity_map_data.bg_intensity_sys_err / computed_survival_probabilities,
        )
        np.testing.assert_array_equal(
            survival_data.intensity_map_data.bg_intensity_stat_uncert,
            intensity_map_data.bg_intensity_stat_uncert
            / computed_survival_probabilities,
        )

        np.testing.assert_array_equal(
            survival_data.intensity_map_data.survival_probability,
            computed_survival_probabilities,
        )

        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_sys_err_minus,
                                      intensity_map_data.ena_intensity_sys_err_minus / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_sys_err_plus,
                                      intensity_map_data.ena_intensity_sys_err_plus / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data.intensity_map_data.epoch, intensity_map_data.epoch)
        np.testing.assert_array_equal(survival_data.intensity_map_data.epoch_delta, intensity_map_data.epoch_delta)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy, intensity_map_data.energy)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_delta_plus,
                                      intensity_map_data.energy_delta_plus)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_delta_minus,
                                      intensity_map_data.energy_delta_minus)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_label, intensity_map_data.energy_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.latitude, intensity_map_data.latitude)
        np.testing.assert_array_equal(survival_data.coords.latitude_delta, input_map.coords.latitude_delta)
        np.testing.assert_array_equal(survival_data.coords.latitude_label, input_map.coords.latitude_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.longitude, intensity_map_data.longitude)
        np.testing.assert_array_equal(survival_data.coords.longitude_delta, input_map.coords.longitude_delta)
        np.testing.assert_array_equal(survival_data.coords.longitude_label, input_map.coords.longitude_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.exposure_factor,
                                      intensity_map_data.exposure_factor)
        np.testing.assert_array_equal(survival_data.intensity_map_data.obs_date, intensity_map_data.obs_date)
        np.testing.assert_array_equal(survival_data.intensity_map_data.obs_date_range,
                                      intensity_map_data.obs_date_range)
        np.testing.assert_array_equal(survival_data.intensity_map_data.solid_angle, intensity_map_data.solid_angle)
        np.testing.assert_array_equal(survival_data.intensity_map_data.quality_flags, quality_flags)

    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.maps.survival_probability_processing.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.maps.survival_probability_processing.filter_bad_days')
    def test_process_survival_probability_cg_correction_argument(self, _mock_filter_bad_days,
                                                                 mock_combine_glows_l3e_with_l1c_pointing,
                                                                 mock_survival_probability_pointing_set,
                                                                 mock_survival_skymap):
        test_cases = [True, False]
        for cg_correction_value in test_cases:
            rng = np.random.default_rng()
            input_map_flux = rng.random((1, 9, 90, 45))
            epoch = datetime.now()

            input_map = create_rectangular_intensity_map_data(epoch=[epoch], flux=input_map_flux)

            input_map.intensity_map_data.energy = sentinel.hi_l2_energies

            l2_grid = PixelSize.FourDegrees
            l2_descriptor_parts = Mock(sensor=sentinel.l2_sensor, spin_phase=sentinel.l2_spin, grid=l2_grid,
                                       reference_frame=ReferenceFrame.Heliospheric)
            dependencies = HiLoL3SurvivalDependencies(l2_data=input_map,
                                                      l1c_data=sentinel.l1c_data,
                                                      glows_l3e_data=sentinel.glows_l3e_data,
                                                      l2_map_descriptor_parts=l2_descriptor_parts,
                                                      dependency_file_paths=[])

            mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                                     (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                                     (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

            mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

            computed_survival_probabilities = rng.random((1, 9, 90, 45))
            quality_flags = np.full_like(computed_survival_probabilities, MapL3Flags.NONE)

            mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
                "exposure_weighted_survival_probabilities": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.AZIMUTH_L2.value,
                        CoordNames.ELEVATION_L2.value,
                    ],
                    computed_survival_probabilities
                ),
                "quality_flags": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA_L1C.value,
                        CoordNames.AZIMUTH_L2.value,
                        CoordNames.ELEVATION_L2.value,
                    ],
                    quality_flags
                ),
            },
                coords={
                    CoordNames.TIME.value: [epoch],
                    CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                    CoordNames.AZIMUTH_L2.value: rng.random((90,)),
                    CoordNames.ELEVATION_L2.value: rng.random((45,)),
                })

            process_survival_probabilities(dependencies, SpiceFrame.IMAP_DPS,
                                                           cg_corrected=cg_correction_value)

            mock_survival_probability_pointing_set.assert_has_calls([
                call(sentinel.hi_l1c_1, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_1,
                     sentinel.hi_l2_energies, cg_corrected=cg_correction_value),
                call(sentinel.hi_l1c_2, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_2,
                     sentinel.hi_l2_energies, cg_corrected=cg_correction_value),
                call(sentinel.hi_l1c_3, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_3,
                     sentinel.hi_l2_energies, cg_corrected=cg_correction_value)
            ])

    def test_integration_uses_fill_values_for_missing_l3e_data(self):
        t1 = datetime(2025, 4, 29, 12)
        t2 = datetime(2025, 5, 7, 12)
        t3 = datetime(2025, 5, 11, 12)
        t4 = datetime(2025, 5, 15, 12)

        l1c_psets = [
            create_l1c_pset(epoch=t1, repointing=1,
                            hae_longitude=np.concat([np.full((1800), 304 + 2), np.full((1800), 124 + 2)])[np.newaxis,
                                          :]),
            create_l1c_pset(epoch=t2, repointing=2,
                            hae_longitude=np.concat([np.full((1800), 312 + 2), np.full((1800), 132 + 2)])[np.newaxis,
                                          :]),
            create_l1c_pset(epoch=t3, repointing=3,
                            hae_longitude=np.concat([np.full((1800), 316 + 2), np.full((1800), 136 + 2)])[np.newaxis,
                                          :]),
            create_l1c_pset(epoch=t4, repointing=4,
                            hae_longitude=np.concat([np.full((1800), 320 + 2), np.full((1800), 140 + 2)])[np.newaxis,
                                          :])
        ]
        l3e_psets = [
            create_l3e_pset(epoch=t1, repointing=1),
            create_l3e_pset(epoch=t2, repointing=2),
            create_l3e_pset(epoch=t4, repointing=4)
        ]
        l2_intensity_map = create_rectangular_intensity_map_data()

        descriptor = parse_map_descriptor("h90-ena-h-sf-nsp-ram-hae-4deg-3mo")
        survival_dependencies = HiLoL3SurvivalDependencies(l2_intensity_map, l1c_psets, l3e_psets, descriptor,
                                                           Mock())

        expected_sky_strip = np.full(45, 2.0)
        default_sp_sky_strip = np.full(45, np.nan)

        output_map = process_survival_probabilities(survival_dependencies, SpiceFrame.IMAP_HAE)
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 76, :], expected_sky_strip)
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 78, :], expected_sky_strip)
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 79, :], default_sp_sky_strip)
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 80, :], expected_sky_strip)

    def test_combine_glows_l3e_with_l1c_pointing(self):
        glows_l3e_data = [
            create_dataclass_mock(GlowsL3eRectangularMapInputData, repointing=0),
            create_dataclass_mock(GlowsL3eRectangularMapInputData, repointing=0),
            create_dataclass_mock(GlowsL3eRectangularMapInputData, repointing=1),
            create_dataclass_mock(GlowsL3eRectangularMapInputData, repointing=3),
        ]

        hi_l1c_data = [
            create_dataclass_mock(InputRectangularPointingSet, repointing=1),
            create_dataclass_mock(InputRectangularPointingSet, repointing=2),
            create_dataclass_mock(InputRectangularPointingSet, repointing=3),
            create_dataclass_mock(InputRectangularPointingSet, repointing=4)
        ]

        expected = [
            (hi_l1c_data[0], glows_l3e_data[2]),
            (hi_l1c_data[2], glows_l3e_data[3]),
        ]

        actual = combine_glows_l3e_with_l1c_pointing(glows_l3e_data, hi_l1c_data)

        self.assertEqual(expected, actual)

    def test_filter_bad_days(self):
        good_day_exposures = np.full((1, 7, 3600, 40), 100.0)
        bad_day_exposures = np.zeros((1, 7, 3600, 40))

        hi_l1c_data = [
            create_dataclass_mock(InputRectangularPointingSet, exposure_times=good_day_exposures.copy()),
            create_dataclass_mock(InputRectangularPointingSet, exposure_times=bad_day_exposures.copy()),
            create_dataclass_mock(InputRectangularPointingSet, exposure_times=good_day_exposures.copy()),
            create_dataclass_mock(InputRectangularPointingSet, exposure_times=bad_day_exposures.copy())
        ]

        filtered_psets = filter_bad_days(hi_l1c_data)

        self.assertEqual([hi_l1c_data[0], hi_l1c_data[2]], filtered_psets)
