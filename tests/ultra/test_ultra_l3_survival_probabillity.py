import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
from astropy_healpix import HEALPix
from imap_processing.ena_maps.ena_maps import UltraPointingSet, HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.ultra.constants import UltraConstants

from imap_l3_processing.constants import (
    TT2000_EPOCH,
    ONE_SECOND_IN_NANOSECONDS,
    SECONDS_PER_DAY,
)
from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.ultra.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.ultra.science.ultra_survival_probability import UltraSurvivalProbability, \
    UltraSurvivalProbabilitySkyMap
from tests.spice_test_case import SpiceTestCase


class TestUltraSurvivalProbability(unittest.TestCase):

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    def test_ultra_survival_probability_pset_calls_super(self, _, mock_frame_transform_az_el):
        input_l1c_pset = _create_ultra_l1c_pset(np.array([2]), np.full((1, 1, 12), 1))
        glows = _build_glows_l3e_ultra()

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows, bin_groups=[0, 1])

        self.assertIsInstance(prod, UltraPointingSet)
        self.assertEqual(input_l1c_pset.to_xarray(), prod.data)
        self.assertEqual(geometry.SpiceFrame.IMAP_DPS, prod.spice_reference_frame)

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch.object(UltraConstants, 'PSET_ENERGY_BIN_EDGES', new=[0.5, 1.5, 4])
    def test_ultra_survival_probability_rotates_to_glows_frame(self, mock_unitim, mock_frame_transform_az_el):
        glows_energies = np.array([1, 2])
        input_l1c_pset = _create_ultra_l1c_pset(glows_energies, np.full((1, 2, 12), 1))
        glows_surv_prob = np.array([[[1] * 6 + [0] * 6, [2] * 6 + [3] * 6]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        rotate_el_by_180 = np.array([[1, -1]] * 12)
        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el * rotate_el_by_180

        prod = UltraSurvivalProbability(input_l1c_pset, glows, bin_groups=[0, 1, 2])

        expected_tt = (input_l1c_pset.epoch - TT2000_EPOCH).total_seconds() + 43200
        mock_unitim.assert_called_once_with(expected_tt, "TT", "ET")

        mock_frame_transform_az_el.assert_called_once_with(mock_unitim.return_value, prod.az_el_points,
                                                           geometry.SpiceFrame.IMAP_DPS,
                                                           geometry.SpiceFrame.ECLIPJ2000)
        expected_survival_probabilities_values = np.array([[[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                                                            [3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch.object(UltraConstants, 'PSET_ENERGY_BIN_EDGES', new=[.1, 10])
    def test_ultra_survival_probability_is_multiplied_by_exposure(self, mock_frame_transform_az_el, _):
        glows_energies = np.array([1])
        input_l1c_pset = _create_ultra_l1c_pset(glows_energies, np.full((1, 1, 12), 2))
        glows_surv_prob = np.array([[[1] * 6 + [0] * 6]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows, bin_groups=[0, 1])

        expected_survival_probabilities_values = np.array([[[2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)


    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch.object(UltraConstants, 'PSET_ENERGY_BIN_EDGES', new=[.1, 10])
    def test_ultra_survival_probability_pset_includes_quality_flags(self, mock_frame_transform_az_el, _):
        exposure = np.array([[[2] * 6 + [0] * 6]])

        expected_unflagged = np.array([[[0] * 12]])
        expected_flagged = exposure.copy()
        cases = [
            (GlowsL3Flags.NONE, expected_unflagged, expected_unflagged, expected_unflagged),
            (GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO, expected_unflagged, expected_flagged, expected_unflagged),
            (GlowsL3Flags.PREDICTIVE_EPHEMERIS, expected_flagged, expected_unflagged, expected_unflagged),
            (GlowsL3Flags.PERSISTED_LAST_POINT, expected_unflagged, expected_unflagged, expected_flagged),
            (GlowsL3Flags.PREDICTIVE_EPHEMERIS | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO, expected_flagged, expected_flagged, expected_unflagged),
            (GlowsL3Flags.PREDICTIVE_EPHEMERIS | GlowsL3Flags.PERSISTED_LAST_POINT, expected_flagged, expected_unflagged, expected_flagged),
            (GlowsL3Flags.PERSISTED_LAST_POINT | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO, expected_unflagged, expected_flagged, expected_flagged),
            (GlowsL3Flags.PERSISTED_LAST_POINT | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PREDICTIVE_EPHEMERIS, expected_flagged, expected_flagged, expected_flagged),
        ]
        for flag, expected_predicted_ephem_flag, expected_nominal_alpha_proton_ratio_flag, expected_persisted_last_point_flag in cases:
            with self.subTest(flag):
                glows_energies = np.array([1])
                input_l1c_pset = _create_ultra_l1c_pset(glows_energies, exposure)
                glows_surv_prob = np.full((1, 1, 12), 2)

                glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies, flags=flag)

                mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

                prod = UltraSurvivalProbability(
                    input_l1c_pset, glows, bin_groups=[0, 1]
                )

                np.testing.assert_array_equal(
                    prod.data["predicted_ephemeris_flag"], expected_predicted_ephem_flag
                )
                self.assertEqual(
                    prod.data["predicted_ephemeris_flag"].dims,
                    prod.data["survival_probability_times_exposure"].dims,
                )
                np.testing.assert_array_equal(
                    prod.data["nominal_alpha_proton_ratio_flag"], expected_nominal_alpha_proton_ratio_flag
                )
                np.testing.assert_array_equal(
                    prod.data["persisted_last_point_flag"], expected_persisted_last_point_flag
                )
                self.assertEqual(
                    prod.data["nominal_alpha_proton_ratio_flag"].dims,
                    prod.data["survival_probability_times_exposure"].dims,
                )

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch.object(UltraConstants, 'PSET_ENERGY_BIN_EDGES', new=[1, 100, 10000])
    def test_ultra_survival_probability_interpolates_over_energy(self, mock_frame_transform_az_el, _):
        glows_energies = np.array([1, 100, 10000])
        ultra_energies = np.array([10, 1000])
        input_l1c_pset = _create_ultra_l1c_pset(ultra_energies, np.full((1, 2, 12), 1))
        glows_surv_prob = np.array([[[1] * 12, [3] * 12, [5] * 12]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows, bin_groups=[0, 1, 2])

        expected_survival_probabilities_values = np.array([[[2] * 12, [4] * 12]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    @patch.object(UltraConstants, 'PSET_ENERGY_BIN_EDGES', new=[1, 1, 1, 1, 100, 100, 100, 100, 10000])
    def test_ultra_survival_probability_combines_fine_energy_bins(self, mock_frame_transform_az_el, _):
        glows_energies = np.array([1, 100, 10000])
        ultra_energies = np.array([10, 10, 10, 10, 1000, 1000, 1000, 1000])
        input_l1c_pset = _create_ultra_l1c_pset(ultra_energies, np.full((1, 2 * 4, 12), 1))
        glows_surv_prob = np.array([[[1] * 12, [3] * 12, [5] * 12]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        expected_survival_probabilities_values = np.array([[[2] * 12, [4] * 12]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)

    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.spiceypy.unitim')
    @patch('imap_l3_processing.ultra.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_handles_different_resolutions(self, mock_frame_transform_az_el, _):
        glows_energies = np.array([1])
        ultra_energies = np.array([10])
        input_l1c_pset = _create_ultra_l1c_pset(ultra_energies, np.full((1, 1, 48), 1))
        glows_surv_prob = np.array([[np.arange(12)]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows, bin_groups=[0, 1])

        expected_survival_probabilities_values = np.array([[[0.7667581, 1.25558603, 1.74441397, 2.2332419, 0.75, 0.25,
                                                             0.75, 1.25, 1.75, 2.25, 2.75, 2.25,
                                                             2.83574061, 2.4043331, 2.9043331, 3.4043331, 3.9043331,
                                                             4.4043331,
                                                             4.9043331, 4.33574061, 4.25, 4.75, 5.25, 5.75,
                                                             6.25, 6.75, 6.25, 4.75, 6.56137065, 6.12996314,
                                                             6.62996314, 7.12996314, 7.62996314, 8.12996314, 8.62996314,
                                                             8.06137065,
                                                             8.75, 8.25, 8.75, 9.25, 9.75, 10.25,
                                                             10.75, 10.25, 8.7667581, 9.25558603, 9.74441397,
                                                             10.2332419]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_almost_equal(survival_probability_times_exposure.values,
                                             expected_survival_probabilities_values)


class TestUltraSurvivalProbabilitySkyMap(SpiceTestCase):
    def test_ultra_survival_probability_skymap(self):
        pointing_set_nside = 2
        pointing_set_pixels = 12 * pointing_set_nside ** 2
        l1c_exposure_1 = np.full((1, 1, pointing_set_pixels), 0.5)
        l1c_exposure_2 = np.full((1, 1, pointing_set_pixels), 1)
        l3e_sp_1 = np.full((1, 1, pointing_set_pixels), 0.5)
        l3e_sp_2 = np.full((1, 1, pointing_set_pixels), 0.25)
        glows_energies = np.array([1])
        l1c_1 = _create_ultra_l1c_pset(energy=np.array([1]), exposure_factor=l1c_exposure_1)
        l3e_glows_1 = _build_glows_l3e_ultra(survival_probabilities=l3e_sp_1, energies=glows_energies,
                                             nside=pointing_set_nside, flags=GlowsL3Flags.PREDICTIVE_EPHEMERIS)
        l1c_2 = _create_ultra_l1c_pset(energy=np.array([1]), exposure_factor=l1c_exposure_2)
        l3e_glows_2 = _build_glows_l3e_ultra(survival_probabilities=l3e_sp_2, energies=glows_energies,
                                             nside=pointing_set_nside, flags=GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PERSISTED_LAST_POINT)

        pset_1 = UltraSurvivalProbability(l1c_1, l3e_glows_1, bin_groups=[0, 1])
        pset_2 = UltraSurvivalProbability(l1c_2, l3e_glows_2, bin_groups=[0, 1])

        output_nside = 1
        prod = UltraSurvivalProbabilitySkyMap([pset_1, pset_2], geometry.SpiceFrame.IMAP_DPS, output_nside)

        self.assertIsInstance(prod, HealpixSkyMap)

        exposure_weighted_survival_probabilities = prod.data_1d["exposure_weighted_survival_probabilities"].values
        expected_output_pixels = 12 * output_nside ** 2
        expected_exposure_weighted_survival_probabilities = np.full((1, 1, expected_output_pixels), 1 / 3)
        np.testing.assert_array_almost_equal(exposure_weighted_survival_probabilities,
                                             expected_exposure_weighted_survival_probabilities)
        actual_quality_flags = prod.data_1d["quality_flags"].values
        expected_quality_flag = GlowsL3Flags.PREDICTIVE_EPHEMERIS | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PERSISTED_LAST_POINT
        np.testing.assert_equal(actual_quality_flags, np.full((1, 1, expected_output_pixels), expected_quality_flag))


    def test_ultra_survival_probability_skymap_ignores_nan_survival_probabilities(self):
        pointing_set_nside = 2
        pointing_set_pixels = 12 * pointing_set_nside ** 2
        l1c_exposure_1 = np.full((1, 1, pointing_set_pixels), 0.5)
        l1c_exposure_2 = np.full((1, 1, pointing_set_pixels), 1)
        l3e_sp_1 = np.full((1, 1, pointing_set_pixels), 0.5)
        l3e_sp_2 = np.full((1, 1, pointing_set_pixels), np.nan)
        glows_energies = np.array([1])
        l1c_1 = _create_ultra_l1c_pset(energy=np.array([1]), exposure_factor=l1c_exposure_1)
        l3e_glows_1 = _build_glows_l3e_ultra(survival_probabilities=l3e_sp_1, energies=glows_energies,
                                             nside=pointing_set_nside)
        l1c_2 = _create_ultra_l1c_pset(energy=np.array([1]), exposure_factor=l1c_exposure_2)
        l3e_glows_2 = _build_glows_l3e_ultra(survival_probabilities=l3e_sp_2, energies=glows_energies,
                                             nside=pointing_set_nside)

        pset_1 = UltraSurvivalProbability(l1c_1, l3e_glows_1, bin_groups=[0, 1])
        pset_2 = UltraSurvivalProbability(l1c_2, l3e_glows_2, bin_groups=[0, 1])

        output_nside = 1
        prod = UltraSurvivalProbabilitySkyMap([pset_1, pset_2], geometry.SpiceFrame.IMAP_DPS, output_nside)

        self.assertIsInstance(prod, HealpixSkyMap)
        exposure_weighted_survival_probabilities = prod.data_1d["exposure_weighted_survival_probabilities"].values
        expected_output_pixels = 12 * output_nside ** 2
        expected_exposure_weighted_survival_probabilities = np.full((1, 1, expected_output_pixels), 0.5)
        np.testing.assert_array_almost_equal(exposure_weighted_survival_probabilities,
                                             expected_exposure_weighted_survival_probabilities)


def _build_glows_l3e_ultra(survival_probabilities: np.ndarray = None, energies: np.ndarray = None, nside: int = 16, flags: GlowsL3Flags = GlowsL3Flags.NONE):
    energies = np.arange(1, 21) if energies is None else energies
    num_pixels = 12 * (nside ** 2)
    survival_probabilities = np.random.rand(1, len(energies),
                                            num_pixels) if survival_probabilities is None else survival_probabilities
    healpix_pixels = np.arange(num_pixels)

    healpix_grid = HEALPix(nside)
    lon, lat = healpix_grid.healpix_to_lonlat(healpix_pixels)

    return UltraGlowsL3eData(
        epoch=datetime.now(),
        repointing=1,
        energy=energies,
        healpix_index=healpix_pixels,
        survival_probability=survival_probabilities,
        flags=np.array([flags], dtype=np.uint16)
    )


def _create_ultra_l1c_pset(energy: np.ndarray,
                           exposure_factor: np.ndarray,
                           sensitivity: np.ndarray = None,
                           counts: np.ndarray = None,
                           latitude: np.ndarray = None,
                           longitude: np.ndarray = None,
                           epoch: datetime = None,
                           epoch_delta: int = ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY):
    epoch = datetime(2025, 10, 1, 0) if epoch is None else epoch
    sensitivity = sensitivity or np.full_like(exposure_factor[0], 1)
    healpix_index = np.arange(exposure_factor.shape[-1])
    healpix = HEALPix(nside=int(np.sqrt(len(healpix_index) / 12)))
    lon_pix, lat_pix = healpix.healpix_to_lonlat(healpix_index)
    input_l1c_pset = UltraL1CPSet(
        epoch=epoch,
        epoch_delta=epoch_delta,
        energy=energy,
        exposure=exposure_factor,
        healpix_index=healpix_index,
        latitude=np.rad2deg(lat_pix.value),
        longitude=np.rad2deg(lon_pix.value),
        sensitivity=sensitivity,
        repointing=1,
    )
    return input_l1c_pset


if __name__ == '__main__':
    unittest.main()
