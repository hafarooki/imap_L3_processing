import dataclasses
import copy
from datetime import datetime
from unittest.mock import patch, sentinel, call, MagicMock

import numpy as np
import spiceypy
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.maps.map_descriptors import SpinPhase
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.maps.rectangular_survival_probability import Sensor, \
    RectangularSurvivalProbabilitySkyMap, RectangularSurvivalProbabilityPointingSet, \
    interpolate_angular_data_to_nearest_neighbor
from tests.maps import test_builders
from tests.maps.test_builders import create_l1c_pset
from tests.spice_test_case import SpiceTestCase


class TestRectangularSurvivalProbability(SpiceTestCase):

    def setUp(self):
        self.num_energies = 2
        self.l1c_epoch = datetime(2025, 1, 1, 0, 30)
        self.l1c_epoch_delta = np.array([86400_000_000_000])
        self.l1c_hae_longitude = (0.05 + np.linspace(0, 360, 3600, endpoint=False) + 90).reshape((1, -1)) % 360
        self.l1c_hae_latitude = np.linspace(-90, 90, 3600, endpoint=False).reshape((1, -1))

        self.hi_energies = np.geomspace(1, 10000, self.num_energies)

        self.l1c_hi_dataset = InputRectangularPointingSet(
            epoch=self.l1c_epoch,
            epoch_delta=self.l1c_epoch_delta,
            repointing=1,
            epoch_j2000=np.array(np.array([spiceypy.datetime2et(self.l1c_epoch)]) * 1e9),
            exposure_times=np.arange(self.num_energies * 3600).reshape((1, self.num_energies, 3600)) + 1.1,
            esa_energy_step=np.arange(self.num_energies),
            pointing_start_met=None,
            pointing_end_met=None,
            hae_longitude=self.l1c_hae_longitude,
            hae_latitude=self.l1c_hae_latitude
        )

        self.glows_data = GlowsL3eRectangularMapInputData(
            epoch=datetime.now(),
            epoch_j2000=np.array([0]),
            repointing=1,
            energy=np.geomspace(1, 10000, self.num_energies + 1),
            spin_angle=np.arange(0, 360, 1) + 0.5,
            probability_of_survival=np.arange((self.num_energies + 1) * 360).reshape(
                (1, self.num_energies + 1, 360)
            )
            + 5.4,
            flags=np.array([0], dtype=np.uint16),
        )

        l1c_spin_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
        self.ram_mask = (l1c_spin_angles < 90) | (l1c_spin_angles > 270)

        def _mock_add_spacecraft_velocity(dataset):
            dataset["sc_velocity"] = xr.DataArray([1000])
            return dataset

        def _mock_calculate_ram_mask(dataset):
            dataset["ram_mask"] = xr.DataArray(self.ram_mask, dims=["longitude"])
            return dataset

        self.mock_add_spacecraft_position_and_velocity_to_pset_patcher = patch(
            'imap_l3_processing.maps.rectangular_survival_probability.add_spacecraft_position_and_velocity_to_pset')

        self.mock_add_sc_velocity_to_pset = self.mock_add_spacecraft_position_and_velocity_to_pset_patcher.start()
        self.mock_add_sc_velocity_to_pset.side_effect = _mock_add_spacecraft_velocity

        self.mock_calculate_ram_mask_patcher = patch(
            'imap_l3_processing.maps.rectangular_survival_probability.calculate_ram_mask')
        self.mock_calculate_ram_mask = self.mock_calculate_ram_mask_patcher.start()
        self.mock_calculate_ram_mask.side_effect = _mock_calculate_ram_mask

    def tearDown(self):
        self.mock_add_spacecraft_position_and_velocity_to_pset_patcher.stop()
        self.mock_calculate_ram_mask_patcher.stop()

    @patch('imap_l3_processing.maps.rectangular_survival_probability.PointingSet.__init__')
    @patch('imap_l3_processing.maps.rectangular_survival_probability.add_spacecraft_position_and_velocity_to_pset')
    @patch('imap_l3_processing.maps.rectangular_survival_probability.calculate_ram_mask')
    def test_survival_probability_pointing_set_calls_parent_constructor(self, _,
                                                                        __,
                                                                        mock_rectangular_pointing_set_constructor):
        pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi45, SpinPhase.RamOnly,
                                                                 self.glows_data,
                                                                 self.hi_energies)
        self.assertIsInstance(pointing_set, PointingSet)

        mock_rectangular_pointing_set_constructor.assert_called_once()

    @patch('imap_l3_processing.maps.rectangular_survival_probability.add_spacecraft_position_and_velocity_to_pset')
    @patch('imap_l3_processing.maps.rectangular_survival_probability.calculate_ram_mask')
    def test_survival_probability_pointing_set_no_cg(self, mock_calculate_ram_mask, mock_add_sc_velocity_to_pset):
        mock_calculate_ram_mask.return_value = xr.Dataset({
            "ram_mask": [True]
        })

        pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi45, SpinPhase.RamOnly,
                                                                 self.glows_data,
                                                                 self.hi_energies, False)

        mock_calculate_ram_mask.assert_called_once_with(mock_add_sc_velocity_to_pset.return_value)

        self.assertEqual(pointing_set.data['directional_mask'], [True])

    def test_survival_probability_pointing_set(self):
        test_cases = [
            (Sensor.Hi90, 0, SpinPhase.RamOnly, self.ram_mask),
            (Sensor.Hi90, 0, SpinPhase.AntiRamOnly, ~self.ram_mask),
            (Sensor.Hi45, -45, SpinPhase.RamOnly, self.ram_mask),
            (Sensor.Hi45, -45, SpinPhase.AntiRamOnly, ~self.ram_mask),
            (Sensor.Lo90, 0, SpinPhase.AntiRamOnly, ~self.ram_mask),
        ]

        expected_repointing_midpoint = self.l1c_hi_dataset.epoch_j2000 + self.l1c_epoch_delta / 2

        for sensor, expected_sensor_angle, spin_phase, expected_mask in test_cases:
            with self.subTest(f"{sensor.value}"):
                pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, spin_phase,
                                                                         self.glows_data,
                                                                         self.hi_energies)
                self.assertIsInstance(pointing_set, PointingSet)
                self.assertEqual(pointing_set.spice_reference_frame, geometry.SpiceFrame.IMAP_HAE)

                self.assertIn("exposure", pointing_set.data.data_vars)
                np.testing.assert_array_equal(
                    pointing_set.data["exposure"].values,
                    self.l1c_hi_dataset.exposure_times)

                np.testing.assert_array_equal(
                    pointing_set.data["directional_mask"].values,
                    expected_mask
                )

                self.assertIn(CoordNames.AZIMUTH_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_almost_equal(
                    np.arange(0, 360, 0.1) + 0.05,
                    pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)

                self.assertIn(CoordNames.ENERGY_ULTRA_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset.esa_energy_step,
                                              pointing_set.data[CoordNames.ENERGY_ULTRA_L1C.value].values)

                self.assertIn(CoordNames.TIME.value, pointing_set.data.coords)
                np.testing.assert_array_equal(pointing_set.data[CoordNames.TIME.value].values,
                                              expected_repointing_midpoint)

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 0],
                                              self.l1c_hae_longitude[0])

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 1], self.l1c_hae_latitude[0])

    @patch("imap_l3_processing.maps.rectangular_survival_probability.apply_compton_getting_correction")
    def test_hi_cg_corrected_survival_probability_pointing_set(self, mock_cg_correction):
        self.mock_add_sc_velocity_to_pset.side_effect = None
        corrected_hae_longitude = np.full((1, 3, 3600), 2)
        corrected_hae_latitude = np.full((1, 3, 3600), 1)

        l1c_dataset = test_builders.create_l1c_pset(
            epoch=self.l1c_hi_dataset.epoch,
            epoch_delta=self.l1c_hi_dataset.epoch_delta,
            exposures=np.array([[np.full((3600,), 1), np.full((3600,), 2), np.full((3600,), 3)]]),
        )

        corrected_az_el_pairs = np.stack([corrected_hae_longitude[0], corrected_hae_latitude[0]], axis=2)

        energy_sc = np.ones((1, 3, 3600))
        energy_sc[0, 0, :1800] = 10
        energy_sc[0, 0, 1800:3600] = 100
        energy_sc[0, 1, :1800] = 25
        energy_sc[0, 1, 1800:3600] = 1000
        energy_sc[0, 2, :1800] = 1000
        energy_sc[0, 2, 1800:3600] = 2000

        sc_velocity = np.arange(3) + 1000

        hi_hf_energies = np.array([.001, .1, 1])
        expected_exposures = np.array([
            [np.concatenate((l1c_dataset.exposure_times[0, 0, :1800], l1c_dataset.exposure_times[0, 1, 1800:3600])),
             np.concatenate((l1c_dataset.exposure_times[0, 1, :1800], l1c_dataset.exposure_times[0, 2, 1800:3600])),
             l1c_dataset.exposure_times[0, 2, 0:3600]],
        ])

        mock_cg_correction.return_value = xr.Dataset({
            "energy_sc": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value], energy_sc
            ),
            "hae_latitude": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value], corrected_hae_latitude
            ),
            "hae_longitude": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value],
                corrected_hae_longitude
            ),
            "sc_velocity": (
                [CoordNames.CARTESIAN_VECTOR.value],
                sc_velocity
            )
        },
            coords={
                CoordNames.TIME.value: [1_000_000_000],
                CoordNames.ENERGY_L2.value: hi_hf_energies,
                CoordNames.AZIMUTH_L1C.value: np.arange(3600),
                CoordNames.CARTESIAN_VECTOR.value: np.arange(3)
            }
        )

        cg_pointing_set = RectangularSurvivalProbabilityPointingSet(l1c_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                                    self.glows_data, hi_hf_energies,
                                                                    cg_corrected=True)

        [actual_uncorrected_pset] = self.mock_add_sc_velocity_to_pset.call_args[0]

        np.testing.assert_array_equal(actual_uncorrected_pset['hae_longitude'].values[0], l1c_dataset.hae_longitude[0])
        np.testing.assert_array_equal(actual_uncorrected_pset['hae_latitude'].values[0], l1c_dataset.hae_latitude[0])

        np.testing.assert_array_equal(actual_uncorrected_pset['epoch'].values, self.l1c_hi_dataset.epoch_j2000)
        np.testing.assert_array_equal(actual_uncorrected_pset['epoch_delta'].values, self.l1c_hi_dataset.epoch_delta)

        pset_with_sc_velocity, actual_hf_energies = mock_cg_correction.call_args[0]
        self.assertEqual(pset_with_sc_velocity, self.mock_add_sc_velocity_to_pset.return_value)
        self.mock_calculate_ram_mask.assert_called_once_with(mock_cg_correction.return_value)

        expected_energies_in_eV = hi_hf_energies * 1000
        np.testing.assert_array_equal(actual_hf_energies, expected_energies_in_eV)
        self.assertEqual((CoordNames.ENERGY_ULTRA_L1C.value,), actual_hf_energies.dims)

        np.testing.assert_array_equal(cg_pointing_set.data['epoch'],
                                      self.l1c_hi_dataset.epoch_j2000 + (self.l1c_hi_dataset.epoch_delta / 2))
        np.testing.assert_array_equal(cg_pointing_set.az_el_points, corrected_az_el_pairs)

        np.testing.assert_array_equal(cg_pointing_set.data['exposure'], expected_exposures)

    @patch("imap_l3_processing.maps.rectangular_survival_probability.add_spacecraft_position_and_velocity_to_pset")
    @patch("imap_l3_processing.maps.rectangular_survival_probability.apply_compton_getting_correction")
    def test_lo_cg_corrected_survival_probability_pointing_set(self, mock_cg_correction, mock_add_sc_velocity_to_pset):
        corrected_hae_longitude = np.full((1, 3, 3600), 2)
        corrected_hae_latitude = np.full((1, 3, 3600), 1)
        corrected_az_el_pairs = np.stack([corrected_hae_longitude[0], corrected_hae_latitude[0]], axis=2)

        uncorrected_hae_lon = np.ones((1, 3600))
        uncorrected_hae_lat = np.full((1, 3600), 2)

        l1c_dataset = test_builders.create_l1c_pset(
            epoch=self.l1c_hi_dataset.epoch,
            epoch_delta=None,
            exposures=np.array([[np.full((3600,), 1), np.full((3600,), 2), np.full((3600,), 3)]]),
            pointing_start_met=np.array([800_000_000]),
            pointing_end_met=np.array([800_100_000]),
        )

        l1c_dataset.hae_longitude = uncorrected_hae_lon
        l1c_dataset.hae_latitude = uncorrected_hae_lat

        energy_sc = np.ones((1, 3, 3600))
        energy_sc[0, 0, :1800] = 10
        energy_sc[0, 0, 1800:3600] = 100
        energy_sc[0, 1, :1800] = 100
        energy_sc[0, 1, 1800:3600] = 1000
        energy_sc[0, 2, :1800] = 1000
        energy_sc[0, 2, 1800:3600] = 2000

        sc_velocity = np.arange(3) + 1000

        lo_hf_energies = np.array([.001, .1, 1])
        expected_exposures = np.array([
            [np.concatenate((l1c_dataset.exposure_times[0, 0, :1800], l1c_dataset.exposure_times[0, 1, 1800:3600])),
             np.concatenate((l1c_dataset.exposure_times[0, 1, :1800], l1c_dataset.exposure_times[0, 2, 1800:3600])),
             l1c_dataset.exposure_times[0, 2, 0:3600]],
        ])

        mock_cg_correction.return_value = xr.Dataset({
            "energy_sc": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value], energy_sc
            ),
            "hae_latitude": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value], corrected_hae_latitude
            ),
            "hae_longitude": (
                [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.AZIMUTH_L2.value],
                corrected_hae_longitude
            ),
            "sc_velocity": (
                [CoordNames.CARTESIAN_VECTOR.value],
                sc_velocity
            )
        },
            coords={
                CoordNames.TIME.value: [1_000_000_000],
                CoordNames.ENERGY_L2.value: lo_hf_energies,
                CoordNames.AZIMUTH_L1C.value: np.arange(3600),
                CoordNames.CARTESIAN_VECTOR.value: np.arange(3)
            }
        )

        cases = {
            "lo90": Sensor.Lo90,
            "lo75": Sensor.Lo75,
            "lo105": Sensor.Lo105
        }
        for name, sensor in cases.items():
            with self.subTest(name):
                cg_pointing_set = RectangularSurvivalProbabilityPointingSet(l1c_dataset, sensor, SpinPhase.RamOnly,
                                                                            self.glows_data, lo_hf_energies,
                                                                            cg_corrected=True)

                expected_pointing_epoch_midpoint = self.l1c_hi_dataset.epoch_j2000 + (100_000 * 1e9 / 2)

                [actual_uncorrected_pset] = mock_add_sc_velocity_to_pset.call_args[0]

                np.testing.assert_array_equal(actual_uncorrected_pset['hae_longitude'], uncorrected_hae_lon)
                np.testing.assert_array_equal(actual_uncorrected_pset['hae_latitude'], uncorrected_hae_lat)

                np.testing.assert_array_equal(actual_uncorrected_pset['epoch'].values, self.l1c_hi_dataset.epoch_j2000)
                np.testing.assert_array_equal(actual_uncorrected_pset['epoch_delta'].values, None)
                np.testing.assert_array_equal(actual_uncorrected_pset['pointing_start_met'].values, [800_000_000])
                np.testing.assert_array_equal(actual_uncorrected_pset['pointing_end_met'].values, [800_100_000])
                self.assertEqual(actual_uncorrected_pset.attrs['Logical_source'], "imap_lo")

                pset_with_sc_velocity, actual_hf_energies = mock_cg_correction.call_args[0]
                self.assertEqual(pset_with_sc_velocity, mock_add_sc_velocity_to_pset.return_value)

                expected_energies_in_eV = lo_hf_energies * 1000
                np.testing.assert_array_equal(actual_hf_energies, expected_energies_in_eV)

                np.testing.assert_array_equal(cg_pointing_set.data['epoch'], expected_pointing_epoch_midpoint)
                np.testing.assert_array_equal(cg_pointing_set.az_el_points, corrected_az_el_pairs)

                np.testing.assert_array_equal(cg_pointing_set.data['exposure'], expected_exposures)

    def test_exposure_weighting_with_interpolated_survival_probabilities(self):
        test_cases = [
            (SpinPhase.RamOnly, self.ram_mask),
            (SpinPhase.AntiRamOnly, ~self.ram_mask),
        ]

        for spin_phase, expected_mask in test_cases:
            with (self.subTest(spin_phase)):
                self.hi_energies = np.array([10, 10_000])
                self.glows_data.energy = np.array([1, 100, 100_000])
                self.glows_data.probability_of_survival = np.repeat([2, 4, 7], 360).reshape(1, 3, 360)

                expected_interpolated_survival_probabilities = \
                    np.repeat([3, 6], 3600).reshape(1, 2, 3600) * self.l1c_hi_dataset.exposure_times

                sensor, expected_skygrid_elevation_index = Sensor.Hi90, 901
                pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, spin_phase,
                                                                         self.glows_data,
                                                                         self.hi_energies)

                self.assertIn("survival_probability_times_exposure", pointing_set.data.data_vars)
                self.assertEqual((1, self.num_energies, 3600),
                                 pointing_set.data["survival_probability_times_exposure"].values.shape)

                np.testing.assert_array_equal(
                    expected_interpolated_survival_probabilities,
                    pointing_set.data["survival_probability_times_exposure"].values)

                np.testing.assert_array_equal(
                    pointing_set.data["directional_mask"],
                    expected_mask
                )

    def test_exposure_weighted_survivals_are_repeated_to_match_l1c_shape(self):
        pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                                 self.glows_data,
                                                                 self.hi_energies)

        survivals = pointing_set.data[
                        "survival_probability_times_exposure"].values / self.l1c_hi_dataset.exposure_times
        every_tenth_value = survivals[:, :, ::10, np.newaxis]
        groups_of_ten = survivals.reshape(1, self.num_energies, 360, 10)
        self.assertTrue(np.all(np.isclose(every_tenth_value, groups_of_ten)))

    @patch("imap_l3_processing.maps.rectangular_survival_probability.interpolate_angular_data_to_nearest_neighbor")
    def test_survivals_matched_with_corresponding_exposures(self, mock_interpolate):
        self.hi_energies = np.array([1, 100])
        self.glows_data.energy = np.array([1, 100, 100_000])

        first_energy_corresponding_glows_data = np.linspace(0, 1, 3600)
        second_energy_corresponding_glows_data = np.linspace(0, 1, 3600) + 100.2

        mock_interpolate.return_value = np.array([first_energy_corresponding_glows_data,
                                                  second_energy_corresponding_glows_data])

        pointing_set = RectangularSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                                 self.glows_data,
                                                                 self.hi_energies)

        pset_spin_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
        pset_azimuths = np.mod(pset_spin_angles, 360)

        mock_interpolate.assert_called_once()
        get_interpolated_glows_data_args = mock_interpolate.call_args_list[0].args

        np.testing.assert_array_equal(pset_azimuths, get_interpolated_glows_data_args[0])
        np.testing.assert_array_equal(self.glows_data.spin_angle, get_interpolated_glows_data_args[1])
        np.testing.assert_array_equal(self.glows_data.probability_of_survival[0, :2],
                                      get_interpolated_glows_data_args[2])

        corresponding_glows_data = np.array(
            [first_energy_corresponding_glows_data, second_energy_corresponding_glows_data])[np.newaxis, ...]

        np.testing.assert_array_almost_equal(pointing_set.data["survival_probability_times_exposure"].values,
                                             corresponding_glows_data * self.l1c_hi_dataset.exposure_times)

    @patch("imap_l3_processing.maps.rectangular_survival_probability.interpolate_angular_data_to_nearest_neighbor")
    @patch("imap_l3_processing.maps.rectangular_survival_probability.apply_compton_getting_correction")
    def test_survivals_matched_with_corresponding_exposures_cg_corrected(self, mock_cg_correction, mock_interpolate):
        test_cases = [
            Sensor.Hi90,
            Sensor.Hi45,
            Sensor.Lo90,
            Sensor.Lo75,
            Sensor.Lo105,
        ]

        for sensor in test_cases:
            mock_cg_correction.reset_mock()
            mock_interpolate.reset_mock()
            with self.subTest(sensor):
                hf_energies = np.array([1, 2])
                self.glows_data.energy = np.array([1.25, 1.85, 3])

                energy_sc = np.array([[np.full((3600,), 1250), np.full((3600,), 1850)]])
                exposure_times = np.array(
                    [[np.full((3600,), 10), np.full((3600,), 100)]]
                )
                l1c_dataset = create_l1c_pset(exposures=exposure_times)

                corrected_hae_lon = np.ones((1, 2, 3600))
                corrected_hae_lat = np.full((1, 2, 3600), 2)

                mock_cg_correction.return_value = xr.Dataset(
                    {
                        "hae_latitude": xr.DataArray(
                            corrected_hae_lat,
                            dims=[
                                CoordNames.TIME.value,
                                CoordNames.ENERGY_L2.value,
                                CoordNames.AZIMUTH_L2.value,
                            ],
                        ),
                        "hae_longitude": xr.DataArray(
                            corrected_hae_lon,
                            dims=[
                                CoordNames.TIME.value,
                                CoordNames.ENERGY_L2.value,
                                CoordNames.AZIMUTH_L2.value,
                            ],
                        ),
                        "energy_sc": xr.DataArray(
                            energy_sc,
                            dims=[
                                CoordNames.TIME.value,
                                CoordNames.ENERGY_L2.value,
                                CoordNames.AZIMUTH_L2.value,
                            ],
                        ),
                    },
                    coords={
                        CoordNames.ENERGY_L2.value: np.array([1, 2]),
                        CoordNames.TIME.value: np.array([0]),
                        CoordNames.AZIMUTH_L2.value: np.full((3600,), 1),
                    },
                )

                first_energy_corresponding_glows_data = np.linspace(0, 1, 3600)
                second_energy_corresponding_glows_data = np.linspace(0, 1, 3600) + 100.2
                extra_glows_data = np.linspace(0, 1, 3600) + 200.2

                mock_interpolate.return_value = np.array(
                    [
                        first_energy_corresponding_glows_data,
                        second_energy_corresponding_glows_data,
                        extra_glows_data,
                    ]
                )

                pointing_set = RectangularSurvivalProbabilityPointingSet(
                    l1c_dataset,
                    sensor,
                    SpinPhase.RamOnly,
                    self.glows_data,
                    hf_energies,
                    cg_corrected=True,
                )

                pset_spin_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
                pset_azimuths = np.mod(pset_spin_angles, 360)

                self.assertEqual(1, mock_interpolate.call_count)
                mock_interpolate_call_args = mock_interpolate.call_args_list[0].args
                np.testing.assert_array_equal(
                    pset_azimuths, mock_interpolate_call_args[0]
                )
                np.testing.assert_array_equal(
                    self.glows_data.spin_angle, mock_interpolate_call_args[1]
                )
                np.testing.assert_array_equal(
                    self.glows_data.probability_of_survival[0],
                    mock_interpolate_call_args[2],
                )

                corresponding_glows_data = np.array(
                    [first_energy_corresponding_glows_data, second_energy_corresponding_glows_data])[np.newaxis, ...]

                np.testing.assert_array_almost_equal(
                    pointing_set.data["survival_probability_times_exposure"].values,
                    corresponding_glows_data * exposure_times,
                )

    def test_survival_probability_pointing_set_propagates_flags(
        self,
    ):
        cases = [
            (GlowsL3Flags.NONE, 0.0, 0.0, 0.0),
            (GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO, 0.0, 1.0, 0.0),
            (GlowsL3Flags.PREDICTIVE_EPHEMERIS, 1.0, 0.0, 0.0),
            (GlowsL3Flags.PERSISTED_LAST_POINT, 0.0, 0.0, 1.0),
            (GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PREDICTIVE_EPHEMERIS, 1.0, 1.0, 0.0),
            (GlowsL3Flags.PERSISTED_LAST_POINT | GlowsL3Flags.PREDICTIVE_EPHEMERIS, 1.0, 0.0, 1.0),
            (GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PERSISTED_LAST_POINT, 0.0, 1.0, 1.0),
            (GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PERSISTED_LAST_POINT | GlowsL3Flags.PREDICTIVE_EPHEMERIS, 1.0, 1.0, 1.0),
        ]
        for flag_value, expected_pred_ephem_value, expected_nominal_alpha_proton_value, expected_persisted_last_point_value in cases:
            with self.subTest(flag_value):
                exposure_array = np.full(self.l1c_hi_dataset.exposure_times.shape, 1.0)
                exposure_array[:, :, 1000] = 0.0
                self.glows_data.flags[0] = flag_value
                self.l1c_hi_dataset.exposure_times = exposure_array
                pointing_set = RectangularSurvivalProbabilityPointingSet(
                    self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                    glows_dataset=self.glows_data, energies=self.hi_energies)
        
                actual_pred_ephem = pointing_set.data["predicted_ephemeris_flag"].values
                expected_pred_ephem = np.full((1, self.num_energies, 3600), expected_pred_ephem_value)
                expected_pred_ephem[:, :, 1000] = 0
                np.testing.assert_array_equal(actual_pred_ephem, expected_pred_ephem, strict=True)
                self.assertEqual(pointing_set.data["predicted_ephemeris_flag"].dims,
                                 pointing_set.data["survival_probability_times_exposure"].dims)

                actual_nominal_alpha_proton = pointing_set.data["nominal_alpha_proton_ratio_flag"].values
                expected_nominal_alpha_proton = np.full((1, self.num_energies, 3600), expected_nominal_alpha_proton_value)
                expected_nominal_alpha_proton[:, :, 1000] = 0
                np.testing.assert_array_equal(actual_nominal_alpha_proton, expected_nominal_alpha_proton, strict=True)
                self.assertEqual(pointing_set.data["nominal_alpha_proton_ratio_flag"].dims,
                                 pointing_set.data["survival_probability_times_exposure"].dims)

                actual_persisted_last_point = pointing_set.data["persisted_last_point_flag"].values
                expected_persisted_last_point = np.full((1, self.num_energies, 3600), expected_persisted_last_point_value)
                expected_persisted_last_point[:, :, 1000] = 0
                np.testing.assert_array_equal(actual_persisted_last_point, expected_persisted_last_point, strict=True)
                self.assertEqual(pointing_set.data["persisted_last_point_flag"].dims,
                                 pointing_set.data["survival_probability_times_exposure"].dims)

    def test_interpolate_angular_data_to_nearest_neighbor(self):
        input_cases = [
            (0, 360),
            (0.1, 360),
            (0.4, 360),
            (0.5, 360),
            (0.6, 1),
            (120.2, 120),
            (359.4, 359),
            (359.6, 360),
            (360, 360),
        ]
        basic_glows_spin_angles = np.linspace(1, 360, 360, endpoint=True)
        basic_glows_data = np.array([np.linspace(1, 360, 360, endpoint=True) + 1000])
        glows_cases = [
            ("basic", basic_glows_spin_angles, basic_glows_data),
            ("0 instead of 360", np.mod(basic_glows_spin_angles, 360), basic_glows_data),
            ("with 0 at start", np.roll(np.mod(basic_glows_spin_angles, 360), 1), np.roll(basic_glows_data, 1)),
            ("rolled", np.roll(basic_glows_spin_angles, 90), np.roll(basic_glows_data, 90)),
        ]
        for name, glows_spin_angles, glows_data in glows_cases:
            with self.subTest(name):
                input_angles, expected_output = np.array(input_cases).T
                result = interpolate_angular_data_to_nearest_neighbor(input_angles, glows_spin_angles, glows_data)
                np.testing.assert_array_equal(expected_output + 1000, result[0])

    @patch('imap_l3_processing.maps.rectangular_survival_probability.RectangularSkyMap.project_pset_values_to_map')
    @patch('imap_l3_processing.maps.rectangular_survival_probability.RectangularSkyMap.__init__')
    def test_survival_probability_map_construction(self, mock_skymap_constructor, mock_project_pset):
        class TestableRectangularSurvivalProbabilitySkyMap(RectangularSurvivalProbabilitySkyMap):
            def __init__(self, *args):
                self.data_1d = MagicMock()
                super().__init__(*args)

        pset_1 = MagicMock()
        pset_1.data['directional_mask'] = sentinel.directional_mask_1
        pset_2 = MagicMock()
        pset_2.data['directional_mask'] = sentinel.directional_mask_2

        actual_sky_map = TestableRectangularSurvivalProbabilitySkyMap([pset_1, pset_2],
                                                                      sentinel.spacing_deg,
                                                                      sentinel.spice_frame)
        self.assertIsInstance(actual_sky_map, RectangularSkyMap)

        mock_skymap_constructor.assert_called_with(sentinel.spacing_deg, sentinel.spice_frame)

        expected_values_to_project = ["survival_probability_times_exposure", "exposure", "predicted_ephemeris_flag",
                 "nominal_alpha_proton_ratio_flag", "persisted_last_point_flag"]
        mock_project_pset.assert_has_calls([
            call(pset_1, expected_values_to_project, pset_valid_mask=pset_1.data['directional_mask']),
            call(pset_2, expected_values_to_project, pset_valid_mask=pset_2.data['directional_mask']),
        ])

    def test_survival_probability_sky_map_returns_exposure_weighted_survival_probabilities(self):
        self.l1c_hi_dataset.hae_longitude = np.concat([
            np.full((1, 1800), 30.05),
            np.full((1, 1800), 210.05)
        ], axis=1)

        self.l1c_hi_dataset.hae_latitude = np.concat([
            np.linspace(90, -90, 1800, endpoint=False) - 0.05,
            np.linspace(-90, 90, 1800, endpoint=False) + 0.05,
        ]).reshape(1, -1)
        self.ram_mask = np.concat((np.full(1800, True), np.full(1800, False)))

        pset1_glows = self.glows_data
        pset1_l1c = self.l1c_hi_dataset

        pset2_glows = dataclasses.replace(self.glows_data,
                                          probability_of_survival=4 * self.glows_data.probability_of_survival)
        pset2_l1c = dataclasses.replace(self.l1c_hi_dataset, exposure_times=2 * self.l1c_hi_dataset.exposure_times)

        pset1 = RectangularSurvivalProbabilityPointingSet(pset1_l1c, Sensor.Hi90, SpinPhase.RamOnly,
                                                          pset1_glows, self.hi_energies)

        pset2 = RectangularSurvivalProbabilityPointingSet(pset2_l1c, Sensor.Hi90, SpinPhase.RamOnly,
                                                          pset2_glows, self.hi_energies)

        spice_frame = SpiceFrame.IMAP_HAE
        actual_skymap = RectangularSurvivalProbabilitySkyMap([pset1, pset2],
                                                             0.1, spice_frame)
        survival_probability_dataset = actual_skymap.to_dataset()
        expected = 3 * np.repeat(self.glows_data.probability_of_survival[:, [0, -1], 0:180], 10, axis=2)
        expected = np.flip(expected, axis=2)

        survival_prob_in_skygrid_shape = np.full((1, 2, 3600, 1800), np.nan)
        survival_prob_in_skygrid_shape[:, :, 300, :] = expected

        self.assertIn("exposure_weighted_survival_probabilities", survival_probability_dataset)
        self.assertEqual((1, 2, 3600, 1800),
                         survival_probability_dataset["exposure_weighted_survival_probabilities"].values.shape)

        actual_sp_at_relevant_long = survival_probability_dataset["exposure_weighted_survival_probabilities"].values[:,
                                     :, 300, :]

        np.testing.assert_array_almost_equal(expected, actual_sp_at_relevant_long)

        np.testing.assert_array_almost_equal(survival_prob_in_skygrid_shape,
                                             survival_probability_dataset[
                                                 "exposure_weighted_survival_probabilities"].values)

    def test_survival_probability_sky_map_returns_flags(self):
        self.l1c_hi_dataset.hae_longitude = np.concat([
            np.full((1, 1800), 30.05),
            np.full((1, 1800), 210.05)
        ], axis=1)

        self.l1c_hi_dataset.hae_latitude = np.concat([
            np.linspace(90, -90, 1800, endpoint=False) - 0.05,
            np.linspace(-90, 90, 1800, endpoint=False) + 0.05,
        ]).reshape(1, -1)
        self.ram_mask = np.concat((np.full(1800, True), np.full(1800, False)))

        cases = {
            (
                GlowsL3Flags.NONE,
                GlowsL3Flags.NONE,
                MapL3Flags.NONE
            ),
            (
                GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                GlowsL3Flags.NONE,
                MapL3Flags.PREDICTIVE_EPHEMERIS
            ),
            (
                GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.PREDICTIVE_EPHEMERIS
            ),
            (
                GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                GlowsL3Flags.NONE,
                GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO
            ),
            (
                GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                GlowsL3Flags.NONE,
                MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PREDICTIVE_EPHEMERIS
            ),
            (
                GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO | MapL3Flags.PREDICTIVE_EPHEMERIS
            ),
            (
                GlowsL3Flags.PREDICTIVE_EPHEMERIS,
                GlowsL3Flags.PERSISTED_LAST_POINT,
                MapL3Flags.PREDICTIVE_EPHEMERIS | MapL3Flags.PERSISTED_LAST_POINT
            )
        }

        for pset1_flags, pset2_flags, expected_quality_flags in cases:
            with self.subTest(pset1_flags=pset1_flags, pset2_flags=pset2_flags, expected=expected_quality_flags):
                pset1_glows = copy.deepcopy(self.glows_data)
                pset1_glows.flags[0] = pset1_flags
                pset1_l1c = pset2_l1c = self.l1c_hi_dataset

                pset2_glows = copy.deepcopy(self.glows_data)
                pset2_glows.flags[0] = pset2_flags

                pset1 = RectangularSurvivalProbabilityPointingSet(pset1_l1c, Sensor.Hi90, SpinPhase.RamOnly,
                                                                  pset1_glows, self.hi_energies)

                pset2 = RectangularSurvivalProbabilityPointingSet(pset2_l1c, Sensor.Hi90, SpinPhase.RamOnly,
                                                                  pset2_glows, self.hi_energies)

                spice_frame = SpiceFrame.IMAP_HAE
                actual_skymap = RectangularSurvivalProbabilitySkyMap([pset1, pset2],
                                                                     0.1, spice_frame)
                survival_probability_dataset = actual_skymap.to_dataset()

                expected_flags_in_full_shape = np.full((1, 2, 3600, 1800), MapL3Flags.NONE)
                expected_flags_in_full_shape[:, :, 300, :] = expected_quality_flags

                self.assertIn("quality_flags", survival_probability_dataset)
                self.assertEqual((1, 2, 3600, 1800), survival_probability_dataset["quality_flags"].values.shape)

                np.testing.assert_array_equal(expected_flags_in_full_shape, survival_probability_dataset["quality_flags"].values)
