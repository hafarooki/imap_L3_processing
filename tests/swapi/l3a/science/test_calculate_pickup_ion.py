from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call, sentinel

import numpy as np
import spacepy.pycdf
from imap_processing.swapi.l2 import swapi_l2
from lmfit import Parameters
from spacepy.pycdf import CDF
from uncertainties import ufloat

import imap_l3_processing
from imap_l3_processing.constants import PROTON_MASS_KG, \
    PROTON_CHARGE_COULOMBS, ONE_AU_IN_KM, ONE_SECOND_IN_NANOSECONDS, \
    HE_PUI_PARTICLE_MASS_KG, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
from imap_l3_processing.swapi.l3a.science.speed_calculation import calculate_combined_sweeps
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, \
    extract_pui_energy_bins, \
    convert_velocity_relative_to_imap, calculate_velocity_vector, FittingParameters, \
    ForwardModel, convert_velocity_to_reference_frame, model_count_rate_integral, \
    calculate_pickup_ion_values, ModelCountRateCalculator, calculate_ten_minute_velocities, \
    calculate_pui_velocity_vector, calculate_solar_wind_velocity_vector, calculate_helium_pui_density, \
    calculate_helium_pui_temperature, calc_chi_squared_lm_fit
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import get_test_data_path, NumpyArrayMatcher, run_periodically

FAKE_ROTATION_MATRIX_FROM_PSP = np.array(
    [[-8.03319036e-01, -5.95067395e-01, -2.39441182e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [5.94803234e-01, -8.03675802e-01, 1.77289947e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-2.97932551e-02, 0.00000000e+00, 9.99556082e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-1.16314295e-06, 1.56750981e-06, 6.68593934e-08, -8.03319036e-01, -5.95067395e-01, -2.39441182e-02],
     [-1.56457525e-06, -1.16063465e-06, -1.21809529e-07, 5.94803234e-01, -8.03675802e-01, 1.77289947e-02],
     [1.26218156e-07, 5.29395592e-23, 3.76211978e-09, -2.97932551e-02, 0.00000000e+00, 9.99556082e-01]])

MODULE = "imap_l3_processing.swapi.l3a.science.calculate_pickup_ion"

class TestCalculatePickupIon(SpiceTestCase):
    instrument_response_collection = None

    def setUp(self) -> None:
        density_of_neutral_helium_lut_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / "imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"

        self.density_of_neutral_helium_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(
            density_of_neutral_helium_lut_path)

    def get_response_lookup_table_collection(self) -> InstrumentResponseLookupTableCollection:
        if TestCalculatePickupIon.instrument_response_collection is None:
            response_lut_path = get_test_data_path("swapi/imap_swapi_instrument-response-lut_20241023_v000.zip")
            TestCalculatePickupIon.instrument_response_collection = InstrumentResponseLookupTableCollection.from_file(
                response_lut_path)
        return TestCalculatePickupIon.instrument_response_collection

    @patch(f"{MODULE}.convert_velocity_relative_to_imap")
    @patch(f"{MODULE}.spiceypy")
    def test_calculate_pickup_ion_energy_cutoff(self, mock_spice, mock_convert_velocity):
        cases = [
            PROTON_MASS_KG,
            HE_PUI_PARTICLE_MASS_KG
        ]
        for particle_mass in cases:
            with self.subTest(particle_mass):
                expected_ephemeris_time = 100000000
                mock_light_time = 1233.002
                mock_spice.spkezr.return_value = (np.array([0, 0, 0, 4, 0, 0]), mock_light_time)
                mock_spice.latrec.return_value = np.array([0, 2, 0])

                expected_sw_velocity_in_eclipj2000_frame = np.array([1, 2, 4])
                mock_convert_velocity.return_value = expected_sw_velocity_in_eclipj2000_frame

                solar_wind_velocity_in_imap_frame = np.array([22, 33, 44])

                h_inflow_speed = 102
                hydrogen_inflow_vector = InflowVector(h_inflow_speed, sentinel.h_inflow_lon, sentinel.h_inflow_lat)

                energy_cutoff = calculate_pui_energy_cutoff(particle_mass, expected_ephemeris_time,
                                                            solar_wind_velocity_in_imap_frame,
                                                            hydrogen_inflow_vector)

                mock_spice.spkezr.assert_called_with("IMAP", expected_ephemeris_time, "ECLIPJ2000", "NONE", "SUN")
                mock_spice.latrec.assert_called_with(-h_inflow_speed,
                                                     sentinel.h_inflow_lon,
                                                     sentinel.h_inflow_lat)
                mock_convert_velocity.assert_called_with(solar_wind_velocity_in_imap_frame, expected_ephemeris_time,
                                                         "IMAP_DPS",
                                                         "ECLIPJ2000")

                velocity_cutoff_norm = 5
                self.assertAlmostEqual(
                    0.5 * (particle_mass / PROTON_CHARGE_COULOMBS) * (2 * velocity_cutoff_norm * 1000) ** 2,
                    energy_cutoff)

    def test_extract_pui_energy_bins(self):
        energies = np.array([100, 1000, 1500, 2000, 3000, 10000, 15000])
        energy_indices = np.array([50, 40, 30, 20, 12, 10, 15])
        observed_count_rates = np.array([1, 100, 100, 9, 0, 200, 300])
        lower_energy_cutoff = 1400
        upper_energy_cutoff = 11000

        extracted_energy_bin_labels, extracted_energy_bins, extracted_count_rates = extract_pui_energy_bins(
            energy_indices, energies, observed_count_rates,
            lower_energy_cutoff, upper_energy_cutoff)
        np.testing.assert_array_equal(extracted_energy_bin_labels, np.array([30, 20, 10]))
        np.testing.assert_array_equal(extracted_energy_bins, np.array([1500, 2000, 10000]))
        np.testing.assert_array_equal(extracted_count_rates, np.array([100, 9, 200]))

    @patch(f"{MODULE}.spiceypy")
    def test_convert_velocity_relative_to_imap(self, mock_spice):
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP
        mock_light_time = 12.459
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 98, 77, 66]), mock_light_time)

        input_velocity = np.array([[12, 34, 45], [67, 89, 45]])
        ephemeris_time = 2000
        from_frame = "INPUT_FRAME"
        to_frame = "OUTPUT_FRAME"
        output_velocity = convert_velocity_relative_to_imap(input_velocity, ephemeris_time, from_frame, to_frame)
        expected_velocity = np.array([[67.05039482, 57.6104663, 110.62250463],
                                      [-9.860859, 46.122475, 108.983876]])
        np.testing.assert_array_almost_equal(output_velocity, expected_velocity)
        mock_spice.sxform.assert_called_with(from_frame, to_frame, ephemeris_time)
        mock_spice.spkezr.assert_called_with("IMAP", ephemeris_time, to_frame, "NONE", "SUN")

    def test_calculate_velocity_vector(self):
        speed = [45.6787, 60.123]
        elevation_degrees = [1.7, 0.7]
        phi = [-149.0, -100.0]

        actual_pui_vector = calculate_velocity_vector(speed, elevation_degrees, phi)

        expected_pui_vectors = np.array([[-39.137055, -23.515915, 1.355115],
                                         [-10.43947, -59.205178, 0.734523]])
        np.testing.assert_array_almost_equal(actual_pui_vector, expected_pui_vectors)

    @patch(f"{MODULE}.spiceypy")
    def test_convert_velocity_to_reference_frame(self, mock_spice):
        input_vector = np.array([[1, 2, 3], [1, 2, 3]])
        ephemeris_time = 10000000

        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        result = convert_velocity_to_reference_frame(input_vector, ephemeris_time, "FROM", "TO")

        column_1 = 1 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 3]
        column_2 = 2 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 4]
        column_3 = 3 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 5]
        expected_result = column_1 + column_2 + column_3

        np.testing.assert_array_almost_equal([expected_result, expected_result], result)
        mock_spice.sxform.assert_called_with("FROM", "TO", ephemeris_time)

        result = convert_velocity_to_reference_frame(input_vector[0], ephemeris_time, "FROM", "TO")
        np.testing.assert_array_almost_equal(expected_result, result)

    @patch(f"{MODULE}.convert_velocity_relative_to_imap")
    @patch(f"{MODULE}.calculate_pui_velocity_vector")
    def test_get_speed_grid(self, mock_calculate_pui_velocity_vector, mock_convert_velocity,
                            ):

        pui_velocity_instrument_frame = np.array([8, 6, 4])
        mock_calculate_pui_velocity_vector.return_value = pui_velocity_instrument_frame
        efficiency_lut = EfficiencyCalibrationTable(
            get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat"))

        helium_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        pui_velocity_eclipj2000 = np.array([5, 7, 9])
        sw_velocity_eclipj2000 = np.array([8, 10, 12])
        mock_convert_velocity.side_effect = [
            pui_velocity_eclipj2000,
            sw_velocity_eclipj2000,
        ]

        ephemeris_time_for_epoch = 1234567.1

        speed = 240.54412560893869
        theta = 75
        phi = -135
        solar_wind_vector = np.array([0, 0, 500])
        calculator = ModelCountRateCalculator(
            response_lookup_table_collection=Mock(),
            geometric_table=Mock(),
            solar_wind_vector=solar_wind_vector,
            density_of_neutral_helium_lookup_table=Mock(),
            efficiency_table=efficiency_lut,
            helium_inflow_vector=helium_inflow_vector,
        )

        response_table = InstrumentResponseLookupTable(
            energy=1200,
            elevation=theta,
            azimuth=phi,
            d_energy=1,
            d_elevation=1,
            d_azimuth=1,
            response=0.5
        )
        result = calculator.get_speed_grid(response_table, ephemeris_time_for_epoch)
        expected = np.linalg.norm(pui_velocity_eclipj2000 - sw_velocity_eclipj2000)
        np.testing.assert_array_equal(result, expected, strict=True)

        self.assertEqual(1, mock_calculate_pui_velocity_vector.call_count)
        self.assertEqual(2, mock_convert_velocity.call_count)
        self.assertAlmostEqual(speed, mock_calculate_pui_velocity_vector.call_args.args[0], 5)
        self.assertAlmostEqual(theta, mock_calculate_pui_velocity_vector.call_args.args[1])
        self.assertAlmostEqual(phi, mock_calculate_pui_velocity_vector.call_args.args[2])
        mock_convert_velocity.assert_has_calls([
            call(NumpyArrayMatcher(pui_velocity_instrument_frame),
                 ephemeris_time_for_epoch,
                 "IMAP_SWAPI",
                 "ECLIPJ2000"),
            call(NumpyArrayMatcher(solar_wind_vector),
                 ephemeris_time_for_epoch,
                 "IMAP_DPS",
                 "ECLIPJ2000"),
        ])

    @patch(f"{MODULE}.convert_velocity_relative_to_imap")
    @patch(f"{MODULE}.calculate_pui_velocity_vector")
    def test_get_speed_grid_caches_results(self, mock_calculate_pui_velocity_vector, mock_convert_velocity,
                                           ):

        pui_velocity_instrument_frame = np.array([8, 6, 4])
        mock_calculate_pui_velocity_vector.return_value = pui_velocity_instrument_frame

        pui_velocity_eclipj2000 = np.array([5, 7, 9])
        mock_convert_velocity.return_value = pui_velocity_eclipj2000

        ephemeris_time_for_epoch = 1234567.1

        theta = 75
        phi = -135
        solar_wind_vector = np.array([0, 0, 500])

        helium_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        calculator = ModelCountRateCalculator(
            response_lookup_table_collection=Mock(),
            geometric_table=Mock(),
            solar_wind_vector=solar_wind_vector,
            density_of_neutral_helium_lookup_table=Mock(),
            efficiency_table=Mock(),
            helium_inflow_vector=helium_inflow_vector,
        )

        response_table = InstrumentResponseLookupTable(
            energy=1200,
            elevation=theta,
            azimuth=phi,
            d_energy=1,
            d_elevation=1,
            d_azimuth=1,
            response=0.5
        )
        result = calculator.get_speed_grid(response_table, ephemeris_time_for_epoch)

        self.assertEqual(1, mock_calculate_pui_velocity_vector.call_count)
        self.assertEqual(2, mock_convert_velocity.call_count)

        cached_result = calculator.get_speed_grid(response_table, ephemeris_time_for_epoch)
        np.testing.assert_array_equal(cached_result, result, strict=True)
        self.assertEqual(1, mock_calculate_pui_velocity_vector.call_count)
        self.assertEqual(2, mock_convert_velocity.call_count)

        second_response_table = InstrumentResponseLookupTable(
            energy=1000,
            elevation=theta,
            azimuth=phi,
            d_energy=1,
            d_elevation=1,
            d_azimuth=1,
            response=0.5
        )
        calculator.get_speed_grid(second_response_table, ephemeris_time_for_epoch)
        self.assertEqual(2, mock_calculate_pui_velocity_vector.call_count)
        self.assertEqual(4, mock_convert_velocity.call_count)

        new_time = ephemeris_time_for_epoch + 1
        calculator.get_speed_grid(response_table, new_time)
        self.assertEqual(3, mock_calculate_pui_velocity_vector.call_count)
        self.assertEqual(6, mock_convert_velocity.call_count)

    @patch(f"{MODULE}.DensityOfNeutralHeliumLookupTable.density")
    def test_forward_model_f(self, mock_helium_density):
        mock_helium_density.return_value = 1

        fitting_parameters = FittingParameters(
            cooling_index=0.1,
            ionization_rate=0.47,
            cutoff_speed=500,
            background_count_rate=23)
        ephemeris_time_for_epoch = 1234567.1
        solar_wind_speed_inertial_frame = 456

        distance_km = 0.99 * ONE_AU_IN_KM
        psi = 13

        forward_model = ForwardModel(fitting_parameters, ephemeris_time_for_epoch,
                                     solar_wind_speed_inertial_frame, self.density_of_neutral_helium_lookup_table,
                                     distance_km, psi)

        speed_grid = np.array([485.45, 200, 585.45, 755.45])

        result = forward_model.f(speed_grid)

        expected_term_1 = 0.1 / (4 * np.pi)
        expected_term_2 = (0.47 * ONE_AU_IN_KM ** 2) / (
                distance_km * solar_wind_speed_inertial_frame * 500 ** 3)
        expected_term_3 = (speed_grid / 500) ** (0.1 - 3)
        expected_term_4 = 1 * 1e15
        expected_term_5 = np.array([1, 1, 0, 0])

        expected = expected_term_1 * expected_term_2 * expected_term_3 * expected_term_4 * expected_term_5
        np.testing.assert_array_equal(result, expected)

        mock_helium_density.assert_called_with(
            psi,
            NumpyArrayMatcher(
                distance_km / ONE_AU_IN_KM * (speed_grid / 500) ** 0.1,
            ))

    @patch(f"{MODULE}.ForwardModel")
    @patch("imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table.calculate_sw_speed")
    def test_model_count_rate_integral(self, mock_calculate_speed, mock_forward_model):
        expected_speed_row_1 = 421
        expected_speed_row_2 = 124
        mock_calculate_speed.return_value = np.array([expected_speed_row_1, expected_speed_row_2])

        mock_forward_model.return_value.f.return_value = np.array([33.33, 35.55])

        lookup_table = InstrumentResponseLookupTable(np.array([103.07800, 105.04500]),
                                                     np.array([2.0, 1.1]),
                                                     np.array([-149.0, -132.0]),
                                                     np.array([0.97411, 0.99269]),
                                                     np.array([3.0, 2.0]),
                                                     np.array([1.0, 1.5]),
                                                     np.array([0.0160000000, 0.0230000000]),
                                                     )
        speed_grid = Mock()
        result = model_count_rate_integral(lookup_table, mock_forward_model.return_value, speed_grid)

        expected_row_1_elevation = 2.0
        expected_row_1_forward_model_f = 33.33
        expected_row_1 = 0.0160000000 * expected_row_1_forward_model_f * expected_speed_row_1 ** 4 * 0.97411 * np.cos(
            np.deg2rad(expected_row_1_elevation)) * 1.0 * 3.0

        expected_row_2_elevation = 1.1
        expected_row_2_forward_model_f = 35.55
        expected_row_2 = 0.0230000000 * expected_row_2_forward_model_f * expected_speed_row_2 ** 4 * 0.99269 * np.cos(
            np.deg2rad(expected_row_2_elevation)) * 1.5 * 2.0

        expected_denominator_row_1 = 0.97411 * np.cos(np.deg2rad(2.0)) * 3.0 * 1.0
        expected_denominator_row_2 = 0.99269 * np.cos(np.deg2rad(1.1)) * 2.0 * 1.5

        np.testing.assert_array_equal(result, (expected_row_1 + expected_row_2) /
                                      (expected_denominator_row_1 + expected_denominator_row_2))

        mock_forward_model.return_value.f.assert_called_with(speed_grid)

    @patch(f"{MODULE}.extract_pui_energy_bins")
    @patch(f"{MODULE}.calculate_pui_energy_cutoff")
    @patch(f"{MODULE}.lmfit.Minimizer")
    @patch(f"{MODULE}.spiceypy")
    @patch(f"{MODULE}.calculate_combined_sweeps")
    def test_calculate_pickup_ions_with_minimize_mocked(self, mock_calculate_combined_sweeps, mock_spice, mock_minimizer_constructor,
                                                        mock_calculate_pui_energy_cutoff, mock_extract_pui_energy_bins):
        ephemeris_time_for_epoch = 100000
        mock_spice.unitim.return_value = ephemeris_time_for_epoch

        mock_light_time = 122.0
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 4, 0, 0]), mock_light_time)
        mock_spice.latrec.return_value = np.array([0, 2, 0])
        mock_spice.reclat.return_value = np.array([1, 0.2, 0.6])
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        data_file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / "imap_swapi_l2_50-sweeps_20100101_v002.cdf"
        with CDF(str(data_file_path)) as cdf:
            energy = cdf["esa_energy"][...]
            count_rate = cdf["swp_coin_rate"][...]

            combined_counts, combined_energies = calculate_combined_sweeps(count_rate, energy)
            mock_calculate_combined_sweeps.return_value = combined_counts, combined_energies
            extracted_counts = [1.9, 1.2, 0]
            extracted_energies = [5000, 4000, 3000]
            extracted_indices = [4, 3, 2]
            expected_indices = [(4, 5000), (3, 4000), (2, 3000)]
            mock_extract_pui_energy_bins.return_value = (
                extracted_indices, extracted_energies, extracted_counts
            )
            geometric_factor_lut_path = get_test_data_path('swapi/imap_swapi_energy-gf-pui-lut_20100101_v001.csv')

            geometric_factor_lut = GeometricFactorCalibrationTable.from_file(geometric_factor_lut_path)
            input_epochs = 123456789000000000
            sw_velocity = np.array([300, 400, 0])

            mock_minimize = mock_minimizer_constructor.return_value.minimize
            mock_minimize_result = Mock()
            mock_minimize_result.uvars = {"cooling_index": 1, "ionization_rate": 2, "cutoff_speed": 3,
                                          "background_count_rate": 4}
            mock_minimize_result.redchi = 3
            mock_minimize.return_value = mock_minimize_result
            lower_energy_cutoff = 6000.0
            upper_energy_cutoff = 12000.0
            mock_calculate_pui_energy_cutoff.side_effect = [lower_energy_cutoff, upper_energy_cutoff]

            efficiency_lut = EfficiencyCalibrationTable(
                get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat"))
            h_inflow_vector = InflowVector.from_file(
                get_test_data_path("swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"))
            he_inflow_vector = InflowVector.from_file(
                get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

            actual_fitting_parameters = calculate_pickup_ion_values(self.get_response_lookup_table_collection(),
                                                                    geometric_factor_lut, energy, count_rate,
                                                                    input_epochs, sw_velocity,
                                                                    self.density_of_neutral_helium_lookup_table,
                                                                    efficiency_lut, h_inflow_vector, he_inflow_vector)

            mock_calculate_combined_sweeps.assert_called_once_with(count_rate, energy)

            mock_spice.unitim.assert_called_with(input_epochs / 1e9, "TT", "ET")
            mock_calculate_pui_energy_cutoff.assert_has_calls([
                call(PROTON_MASS_KG, ephemeris_time_for_epoch, sw_velocity, h_inflow_vector),
                call(HE_PUI_PARTICLE_MASS_KG, ephemeris_time_for_epoch, sw_velocity, he_inflow_vector)])
            mock_extract_pui_energy_bins.assert_called_with(range(62, 0, -1),
                                                            combined_energies,
                                                            combined_counts,
                                                            lower_energy_cutoff * 1.25,
                                                            upper_energy_cutoff * 1.2,
                                                            )
            actual_count_rates, indices, model_count_rates_calculator, ephemeris_time, sweep_count = \
                mock_minimizer_constructor.call_args.kwargs['fcn_args']
            self.assertEqual(calc_chi_squared_lm_fit, mock_minimizer_constructor.call_args.args[0])
            self.assertFalse(mock_minimizer_constructor.call_args.kwargs['scale_covar'])
            actual_params: Parameters = mock_minimizer_constructor.call_args.args[1]

            minimize_simplex = mock_minimizer_constructor.call_args.kwargs["options"]["initial_simplex"]

            def _transform_simplex_vertex(vertex):
                return [actual_params["cooling_index"].from_internal(vertex[0]),
                        actual_params["ionization_rate"].from_internal(vertex[1]),
                        actual_params["cutoff_speed"].from_internal(vertex[2]),
                        actual_params["background_count_rate"].from_internal(vertex[3])]

            np.testing.assert_array_almost_equal([1.5, 1e-7, 500, 0.1], _transform_simplex_vertex(minimize_simplex[0]))
            np.testing.assert_array_almost_equal([5.0, 1e-7, 500, 0.1], _transform_simplex_vertex(minimize_simplex[1]))
            np.testing.assert_array_almost_equal([1.5, 2.1e-7, 500, 0.1],
                                                 _transform_simplex_vertex(minimize_simplex[2]))
            np.testing.assert_array_almost_equal([1.5, 1e-7, 600, 0.1], _transform_simplex_vertex(minimize_simplex[3]))
            np.testing.assert_array_almost_equal([1.5, 1e-7, 500, 0.2], _transform_simplex_vertex(minimize_simplex[4]))

            self.assertEqual(1.5, actual_params["cooling_index"].value)
            self.assertEqual(1.0, actual_params["cooling_index"].min)
            self.assertEqual(5.0, actual_params["cooling_index"].max)

            self.assertEqual(1e-7, actual_params["ionization_rate"].value)
            self.assertEqual(0.6e-9, actual_params["ionization_rate"].min)
            self.assertEqual(8.0e-7, actual_params["ionization_rate"].max)

            self.assertEqual(500, actual_params["cutoff_speed"].value)
            self.assertEqual(400, actual_params["cutoff_speed"].min)
            self.assertEqual(600, actual_params["cutoff_speed"].max)

            self.assertEqual(0.1, actual_params["background_count_rate"].value)
            self.assertEqual(0, actual_params["background_count_rate"].min)
            self.assertEqual(0.2, actual_params["background_count_rate"].max)

            np.testing.assert_array_equal(extracted_counts, actual_count_rates)
            self.assertEqual(expected_indices, indices)
            np.testing.assert_array_equal(model_count_rates_calculator.solar_wind_vector, sw_velocity)
            self.assertEqual(ephemeris_time_for_epoch, ephemeris_time)
            self.assertEqual(50, sweep_count)

            self.assertEqual('nelder', mock_minimize.call_args.kwargs['method'])

            expected_fitting_params = FittingParameters(1, 2, 3, 4)
            self.assertEqual(expected_fitting_params, actual_fitting_parameters)

    @patch(f"{MODULE}.spiceypy")
    @patch(f"{MODULE}.calculate_pui_energy_cutoff")
    @patch(f"{MODULE}.extract_pui_energy_bins")
    @patch(f"{MODULE}.lmfit.Minimizer")
    @patch(f"{MODULE}.calculate_combined_sweeps")
    def test_calculate_pickup_ion_bad_fit(self, mock_calculate_combined_sweeps, mock_minimizer_class,
                                          mock_extract_pui_energy_bins, _, __):

        mock_calculate_combined_sweeps.return_value = (np.array([1]), np.array([1]))
        mock_extract_pui_energy_bins.return_value = (np.array([1]), np.array([1]), np.array([1]))

        fit_params = {
            "cooling_index": -1,
            "ionization_rate": -1,
            "cutoff_speed": -1,
            "background_count_rate": -1,
        }

        mock_minimize_result = Mock(redchi=11, uvars=fit_params)
        mock_minimize = mock_minimizer_class.return_value.minimize
        mock_minimize.return_value = mock_minimize_result

        actual_fitting_params = calculate_pickup_ion_values(instrument_response_lookup_table=Mock(),
                                                            geometric_factor_calibration_table=Mock(),
                                                            energy=np.array([1]), count_rates=np.array([1]),
                                                            center_of_epoch=1e9, sw_velocity_vector=np.array([1, 0, 0]),
                                                            density_of_neutral_helium_lookup_table=Mock(),
                                                            efficiency_table=Mock(), hydrogen_inflow_vector=Mock(),
                                                            helium_inflow_vector=Mock())

        self.assertEqual(actual_fitting_params.flags, SwapiL3Flags.HI_CHI_SQ)
        self.assertEqual(actual_fitting_params.cooling_index, fit_params["cooling_index"])
        self.assertEqual(actual_fitting_params.cutoff_speed, fit_params["cutoff_speed"])
        self.assertEqual(actual_fitting_params.ionization_rate, fit_params["ionization_rate"])
        self.assertEqual(actual_fitting_params.background_count_rate, fit_params["background_count_rate"])

    def test_calculate_pui_density(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(1.5, 1e-7, 520, 0.1)
        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))
        result = calculate_helium_pui_density(epoch, sw_velocity_vector,
                                              self.density_of_neutral_helium_lookup_table,
                                              fitting_params,
                                              he_inflow_vector)
        self.assertAlmostEqual(0.00014681078095942195, result)

    def test_calculate_pui_density_with_uncertainty(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(ufloat(1.5, 0.1),
                                           ufloat(1e-7, 1e-8),
                                           ufloat(520, 5),
                                           ufloat(0.1, 0.01))

        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))
        result = calculate_helium_pui_density(epoch, sw_velocity_vector,
                                              self.density_of_neutral_helium_lookup_table,
                                              fitting_params, he_inflow_vector)
        self.assertAlmostEqual(0.00014681078095942195, result.n)
        self.assertAlmostEqual(1.4679499690964861e-05, result.s)

    @patch('imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy.reclat')
    @patch('imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.scipy.integrate.quad')
    def test_calculate_pui_density_with_mocks(self, mock_integrate, mock_reclat):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 7, 1, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(1.5, 1e-7, 520, 0.1)
        mock_reclat.return_value = 149_000_000, 0, 0
        mock_integrate.return_value = (1000, 1)

        mock_density_of_neutral_helium_table = Mock()
        mock_density_of_neutral_helium_table.get_minimum_distance.return_value = 0.05

        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))
        result = calculate_helium_pui_density(epoch, sw_velocity_vector, mock_density_of_neutral_helium_table,
                                              fitting_params, he_inflow_vector)
        mock_integrate.assert_called_once()
        args, kwargs = mock_integrate.call_args_list[0]
        helium_table_lower_bound = (0.05 / (149_000_000 / ONE_AU_IN_KM)) ** (1 / 1.5) * 520
        expected_discontinuity_points = (0, helium_table_lower_bound, 520)
        self.assertEqual(expected_discontinuity_points, kwargs["points"])
        self.assertEqual(100, kwargs["limit"])
        self.assertEqual(0, args[1])
        self.assertEqual(520, args[2])
        self.assertEqual(4 * np.pi * 1000 / 1e15, result)

    @patch('imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy.reclat')
    @patch('imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.scipy.integrate.quad')
    def test_calculate_pui_temperature_with_mocks(self, mock_integrate, mock_reclat):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 7, 1, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(1.5, 1e-7, 520, 0.1)
        mock_reclat.return_value = 149_000_000, 0, 0
        mock_integrate.side_effect = [
            (1000, 1),
            (10, 1),
        ]

        mock_density_of_neutral_helium_table = Mock()
        mock_density_of_neutral_helium_table.get_minimum_distance.return_value = 0.15

        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        result = calculate_helium_pui_temperature(epoch, sw_velocity_vector,
                                                  mock_density_of_neutral_helium_table,
                                                  fitting_params, he_inflow_vector)

        self.assertEqual(2, mock_integrate.call_count)
        helium_table_lower_bound = (0.15 / (149_000_000 / ONE_AU_IN_KM)) ** (1 / 1.5) * 520
        expected_discontinuity_points = (0, helium_table_lower_bound, 520)
        args, kwargs = mock_integrate.call_args_list[0]
        self.assertEqual(expected_discontinuity_points, kwargs["points"])
        self.assertEqual(100, kwargs["limit"])
        self.assertEqual(0, args[1])
        self.assertEqual(520, args[2])
        args, kwargs = mock_integrate.call_args_list[1]
        self.assertEqual(expected_discontinuity_points, kwargs["points"])
        self.assertEqual(100, kwargs["limit"])
        self.assertEqual(0, args[1])
        self.assertEqual(520, args[2])
        self.assertEqual(HE_PUI_PARTICLE_MASS_KG / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN) * 1000 / 10 * 1e6, result)

    def test_calculate_pui_temperature(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(1.5, 1e-7, 500, 0.1)
        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        result = calculate_helium_pui_temperature(epoch, sw_velocity_vector,
                                                  self.density_of_neutral_helium_lookup_table,
                                                  fitting_params, he_inflow_vector)
        self.assertAlmostEqual(24456817.05142866, result)

    def test_calculate_pui_temperature_with_uncertainty(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0, 0, -500])
        fitting_params = FittingParameters(
            ufloat(1.5, .1),
            ufloat(1e-7, 1e-8),
            ufloat(500, 5),
            ufloat(0.1, 0.01))

        he_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        result = calculate_helium_pui_temperature(epoch, sw_velocity_vector,
                                                  self.density_of_neutral_helium_lookup_table,
                                                  fitting_params, he_inflow_vector)
        np.testing.assert_allclose(24456817.05142866, result.n, rtol=1e-8)
        np.testing.assert_allclose(824377.0631439432, result.s, rtol=1e-8)

    @run_periodically(timedelta(days=7))
    @patch(f"{MODULE}.spiceypy")
    def test_calculate_pickup_ions_with_minimize(self, mock_spice):
        ephemeris_time_for_epoch = int(100000 * 1e9)
        mock_spice.unitim.return_value = ephemeris_time_for_epoch
        mock_light_time = 122.0
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 0, 0, 0]), mock_light_time)
        mock_spice.latrec.return_value = np.array([0, 2, 0])
        mock_spice.reclat.return_value = np.array([0.99 * ONE_AU_IN_KM, np.deg2rad(255.7), 0.6])

        def mock_sxform(from_frame, to_frame, et):
            if from_frame == "IMAP_SWAPI":
                return np.eye(6)
            return np.array([
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
            ])

        mock_spice.sxform.side_effect = mock_sxform

        data_file_path = get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20100101_v002.cdf")
        with CDF(str(data_file_path)) as cdf:
            energy = cdf["esa_energy"][...]
            count_rate = cdf["swp_coin_rate"][...]

            geometric_factor_lut_path = get_test_data_path(
                "swapi/imap_swapi_energy-gf-sw-lut-divided-by-efficiency_20100101_v002.csv")

            geometric_factor_lut = GeometricFactorCalibrationTable.from_file(geometric_factor_lut_path)
            efficiency_lut = EfficiencyCalibrationTable(
                get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat"))
            background_count_rate_cutoff = 0.1
            epoch = 123_456_789_000_000_000
            sw_velocity_vector = np.array([0, 0, -500])

            h_inflow_vector = InflowVector.from_file(
                get_test_data_path("swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"))

            he_inflow_vector = InflowVector.from_file(
                get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

            actual_fitting_parameters = calculate_pickup_ion_values(self.get_response_lookup_table_collection(),
                                                                    geometric_factor_lut, energy, count_rate, epoch,
                                                                    sw_velocity_vector,
                                                                    self.density_of_neutral_helium_lookup_table,
                                                                    efficiency_lut, h_inflow_vector, he_inflow_vector)

            mock_spice.unitim.assert_called_with(epoch / ONE_SECOND_IN_NANOSECONDS,
                                                 "TT", "ET")

            self.assertAlmostEqual(1.5, actual_fitting_parameters.cooling_index, delta=0.1)
            self.assertAlmostEqual(1e-7, actual_fitting_parameters.ionization_rate, delta=5e-9)
            self.assertAlmostEqual(520, actual_fitting_parameters.cutoff_speed, delta=5)
            self.assertAlmostEqual(0.1, actual_fitting_parameters.background_count_rate, delta=0.05)

    def test_snapshot_model_count_rate_result(self):
        geometric_factor_lut_path = get_test_data_path(
            "swapi/imap_swapi_energy-gf-sw-lut-divided-by-efficiency_20100101_v002.csv")
        efficiency_lut = EfficiencyCalibrationTable(
            get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat"))

        geometric_factor_lut = GeometricFactorCalibrationTable.from_file(geometric_factor_lut_path)
        sw_velocity_vector = np.array([0, 0, -500])

        helium_inflow_vector = InflowVector.from_file(
            get_test_data_path("swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"))

        calculator = ModelCountRateCalculator(
            response_lookup_table_collection=self.get_response_lookup_table_collection(),
            geometric_table=geometric_factor_lut,
            solar_wind_vector=sw_velocity_vector,
            density_of_neutral_helium_lookup_table=self.density_of_neutral_helium_lookup_table,
            efficiency_table=efficiency_lut,
            helium_inflow_vector=helium_inflow_vector,
        )
        ephemeris_time = 800000000

        fit_params = FittingParameters(
            cooling_index=1.5,
            ionization_rate=1e-7,
            cutoff_speed=400,
            background_count_rate=0.1
        )
        indices_and_energy_centers = [(62, 19098.358), (61, 17541.177),
                                      (60, 16113.177), (59, 14798.38),
                                      (58, 13591.366), (57, 12485.777),
                                      (56, 11467.618), (55, 10532.608),
                                      (54, 9675.514), (53, 8885.046),
                                      (52, 8165.394), (51, 7501.76),
                                      (50, 6888.477), (49, 6327.927),
                                      (48, 5811.486), (47, 5338.868)]

        rates = calculator.model_count_rate(indices_and_energy_centers, fit_params, ephemeris_time)
        np.testing.assert_allclose(rates, [0.1, 0.10003711238804557, 0.10051147333797386, 0.11539080038179494,
                                           0.24695484811167787, 0.4658367488032966, 0.6315184089540633,
                                           0.7221186883032067, 0.7631495109243865, 0.769051763324389,
                                           0.7489970758555614, 0.7107171626571513, 0.669071369362676,
                                           0.6198514811899901, 0.5685065780319507, 0.5197560948020243],
                                   rtol=1e-12)

    @patch(f"{MODULE}.calculate_solar_wind_velocity_vector")
    def test_calculate_ten_minute_velocities(self, mock_calculate_solar_wind_velocity_vector):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])
        z = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])

        one_min_quality_flags = np.repeat(SwapiL3Flags.NONE, 21)

        mock_calculate_solar_wind_velocity_vector.return_value = np.transpose([x, y, z])

        mock_speed = Mock()
        mock_deflection_angles = Mock()
        mock_clock_angles = Mock()
        averaged_velocities, actual_quality_flags = calculate_ten_minute_velocities(mock_speed, mock_deflection_angles,
                                                                                    mock_clock_angles,
                                                                                    one_min_quality_flags)

        expected_averaged_velocities = np.array([[5.5, 55, 55], [15.5, 155, 155], [21, 210, 210]])

        expected_quality_flags = np.repeat(SwapiL3Flags.NONE, 3)

        mock_calculate_solar_wind_velocity_vector.assert_called_with(mock_speed, mock_deflection_angles,
                                                                     mock_clock_angles)

        np.testing.assert_array_equal(averaged_velocities, expected_averaged_velocities)
        np.testing.assert_array_equal(actual_quality_flags, expected_quality_flags)

    @patch(f"{MODULE}.calculate_solar_wind_velocity_vector")
    def test_calculate_ten_minute_velocities_with_swapi_quality_flag(self, mock_calculate_solar_wind_velocity_vector):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])
        z = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])

        one_min_quality_flags = np.repeat(SwapiL3Flags.NONE, 21)
        one_min_quality_flags[13] = SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED

        mock_calculate_solar_wind_velocity_vector.return_value = np.transpose([x, y, z])

        mock_speed = Mock()
        mock_deflection_angles = Mock()
        mock_clock_angles = Mock()
        averaged_velocities, actual_quality_flags = calculate_ten_minute_velocities(mock_speed, mock_deflection_angles,
                                                                                    mock_clock_angles,
                                                                                    one_min_quality_flags)

        expected_averaged_velocities = np.array([[5.5, 55, 55], [15.5, 155, 155], [21, 210, 210]])
        expected_quality_flags = np.array([
            SwapiL3Flags.NONE,
            SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED,
            SwapiL3Flags.NONE,
        ])

        mock_calculate_solar_wind_velocity_vector.assert_called_with(mock_speed, mock_deflection_angles,
                                                                     mock_clock_angles)

        np.testing.assert_array_equal(averaged_velocities, expected_averaged_velocities)
        np.testing.assert_array_equal(actual_quality_flags, expected_quality_flags)

    @patch(f"{MODULE}.calculate_solar_wind_velocity_vector")
    def test_calculate_ten_minute_velocities_with_multiple_quality_flag(self,
                                                                        mock_calculate_solar_wind_velocity_vector):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])
        z = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])

        OTHER_QUALITY_FLAG = 2 ** 3
        one_min_quality_flags = np.repeat(SwapiL3Flags.NONE, 21)

        one_min_quality_flags[13] = SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED
        one_min_quality_flags[14] = OTHER_QUALITY_FLAG

        mock_calculate_solar_wind_velocity_vector.return_value = np.transpose([x, y, z])

        mock_speed = Mock()
        mock_deflection_angles = Mock()
        mock_clock_angles = Mock()
        averaged_velocities, actual_quality_flags = calculate_ten_minute_velocities(mock_speed,
                                                                                    mock_deflection_angles,
                                                                                    mock_clock_angles,
                                                                                    one_min_quality_flags)

        expected_averaged_velocities = np.array([[5.5, 55, 55], [15.5, 155, 155], [21, 210, 210]])
        expected_quality_flags = np.array([
            SwapiL3Flags.NONE,
            SwapiL3Flags.SWP_SW_ANGLES_ESTIMATED | OTHER_QUALITY_FLAG,
            SwapiL3Flags.NONE,
        ])

        mock_calculate_solar_wind_velocity_vector.assert_called_with(mock_speed, mock_deflection_angles,
                                                                     mock_clock_angles)

        np.testing.assert_array_equal(averaged_velocities, expected_averaged_velocities)
        np.testing.assert_array_equal(actual_quality_flags, expected_quality_flags)

    def test_calculate_pickup_ion_velocities(self):
        speed = np.array([400, 390, 400, 410, 400])
        elevation = np.array([-90, 0, 0, 0, 90])
        azimuth = np.array([-180, -90, 0, 90, 180])

        actual = calculate_pui_velocity_vector(speed, elevation, azimuth)

        expected_values = [(0, 0, 400), (390, 0, 0), (0, -400, 0), (-410, 0, 0), (0, 0, -400)]

        np.testing.assert_array_almost_equal(expected_values, actual, 1)

    def test_calculate_solar_wind_velocity_vector(self):
        speed = np.array([400, 390, 400, 410, 400, 400])
        deflection_angle = np.array([90, 0, 180, 0, 90, 90])
        clock_angle = np.array([-270, -180, -90, 0, 0, -180])

        actual = calculate_solar_wind_velocity_vector(speed, deflection_angle, clock_angle)

        expected_values = [(0, -400, 0), (0, 0, -390), (0, 0, 400), (0, 0, -410), (-400, 0, 0), (400, 0, 0)]

        np.testing.assert_array_almost_equal(actual, expected_values, 1)

    def test_calc_chi_squared_lm_fit(self):
        params = MagicMock()
        observed_count_rates = np.array([2, 20, 200, 2000])
        modeled_count_rates = np.array([1, 10, 100, 1000])
        sweep_count = 50
        calculator = Mock()
        calculator.model_count_rate.return_value = modeled_count_rates
        residual_array = calc_chi_squared_lm_fit(params, observed_count_rates, Mock(), calculator, 1e10, sweep_count)
        chi_squared_formula = 2 * (modeled_count_rates - observed_count_rates
                                   + observed_count_rates * np.log(observed_count_rates / modeled_count_rates))
        np.testing.assert_almost_equal(
            np.square(residual_array),
            sweep_count * swapi_l2.SWAPI_LIVETIME * chi_squared_formula)


    @patch(f"{MODULE}.inv")
    @patch(f"{MODULE}.ndt.Hessian")
    @patch(f"{MODULE}.spiceypy")
    @patch(f"{MODULE}.calculate_pui_energy_cutoff")
    @patch(f"{MODULE}.extract_pui_energy_bins")
    @patch(f"{MODULE}.lmfit.Minimizer")
    @patch(f"{MODULE}.calculate_combined_sweeps")
    def test_calculate_pickup_ion_computes_partial_uncertainties(
            self, mock_calculate_combined_sweeps, mock_minimizer_class,
                                          mock_extract_pui_energy_bins, _, __,
            mock_hessian, mock_inv,
    ):

        mock_calculate_combined_sweeps.return_value = (np.array([1]), np.array([1]))
        mock_extract_pui_energy_bins.return_value = (np.array([1]), np.array([1]), np.array([1]))

        fit_params = {
            "cooling_index": -1,
            "ionization_rate": -1,
            "cutoff_speed": -1,
            "imaginary_non_varying_param": 0,
            "background_count_rate": -1,
        }

        mock_params = Mock()
        mock_params.valuesdict.return_value = fit_params
        mock_minimize_result = Mock(
            redchi=1, 
            uvars=None, 
            var_names=[ "cooling_index",
            "ionization_rate",
            "cutoff_speed",
            "background_count_rate",],
            params=mock_params)
        mock_minimizer = mock_minimizer_class.return_value
        (mock_minimizer.minimize).return_value = mock_minimize_result

        mock_inv.return_value = np.array([
            [-1,0,0,0],
            [0,-2,0,0],
            [0, 0, 3,0],
            [0,0,0, 4],
        ])

        mock_minimizer._int2ext_cov_x.return_value = np.array([
            [-1,0,0,0],
            [0,-2,0,0],
            [0, 0, 4,0],
            [0,0,0, 25],
        ])

        actual_fitting_params = calculate_pickup_ion_values(instrument_response_lookup_table=Mock(),
                                                            geometric_factor_calibration_table=Mock(),
                                                            energy=np.array([1]), count_rates=np.array([1]),
                                                            center_of_epoch=1e9, sw_velocity_vector=np.array([1, 0, 0]),
                                                            density_of_neutral_helium_lookup_table=Mock(),
                                                            efficiency_table=Mock(), hydrogen_inflow_vector=Mock(),
                                                            helium_inflow_vector=Mock())

        mock_hessian.assert_called_with(mock_minimizer.penalty, step=1e-4)
        mock_hfun = mock_hessian.return_value
        mock_hfun.assert_called_with(mock_minimize_result.x)
        mock_inv.assert_called_with(mock_hfun.return_value)
        mock_minimizer._int2ext_cov_x.assert_called_with(
            NumpyArrayMatcher(2*mock_inv.return_value),
            mock_minimize_result.x,
        )

        self.assertTrue(np.isnan(actual_fitting_params.cooling_index.std_dev))
        self.assertTrue(np.isnan(actual_fitting_params.ionization_rate.std_dev))
        self.assertEqual(2,actual_fitting_params.cutoff_speed.std_dev)
        self.assertEqual(5,actual_fitting_params.background_count_rate.std_dev)
        self.assertEqual(SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY, actual_fitting_params.flags)

    @patch(f"{MODULE}.inv")
    @patch(f"{MODULE}.ndt.Hessian")
    @patch(f"{MODULE}.spiceypy")
    @patch(f"{MODULE}.calculate_pui_energy_cutoff")
    @patch(f"{MODULE}.extract_pui_energy_bins")
    @patch(f"{MODULE}.lmfit.Minimizer")
    @patch(f"{MODULE}.calculate_combined_sweeps")
    def test_returns_nominal_values_when_uncertainty_fallback_fails(
            self, mock_calculate_combined_sweeps, mock_minimizer_class,
                                          mock_extract_pui_energy_bins, _, __,
            mock_hessian, mock_inv,
    ):

        mock_calculate_combined_sweeps.return_value = (np.array([1]), np.array([1]))
        mock_extract_pui_energy_bins.return_value = (np.array([1]), np.array([1]), np.array([1]))

        fit_params = {
            "cooling_index": -1,
            "ionization_rate": -1,
            "cutoff_speed": -1,
            "imaginary_non_varying_param": 0,
            "background_count_rate": -1,
        }

        mock_params = Mock()
        mock_params.valuesdict.return_value = fit_params
        mock_minimize_result = Mock(
            redchi=1,
            uvars=None,
            var_names=[ "cooling_index",
            "ionization_rate",
            "cutoff_speed",
            "background_count_rate",],
            params=mock_params)
        mock_minimizer = mock_minimizer_class.return_value
        (mock_minimizer.minimize).return_value = mock_minimize_result

        good_inv = np.array([
            [-1,0,0,0],
            [0,-2,0,0],
            [0, 0, 3,0],
            [0,0,0, 4],
        ])

        mock_minimizer._int2ext_cov_x.return_value = np.array([
            [-1,0,0,0],
            [0,-2,0,0],
            [0, 0, 4,0],
            [0,0,0, 25],
        ])
        cases = [
            ("hessian raises exception", Exception("bad matrix"), [good_inv] ),
            ("inv raises exception", [Mock()], Exception("not invertible") ),
            ]

        for name, hessian, inv in cases:
            with self.subTest(name):
                mock_hessian.return_value.side_effect = hessian
                mock_inv.side_effect = inv

                actual_fitting_params = calculate_pickup_ion_values(instrument_response_lookup_table=Mock(),
                                                                    geometric_factor_calibration_table=Mock(),
                                                                    energy=np.array([1]), count_rates=np.array([1]),
                                                                    center_of_epoch=1e9,
                                                                    sw_velocity_vector=np.array([1, 0, 0]),
                                                                    density_of_neutral_helium_lookup_table=Mock(),
                                                                    efficiency_table=Mock(),
                                                                    hydrogen_inflow_vector=Mock(),
                                                                    helium_inflow_vector=Mock())

                self.assertIsInstance(actual_fitting_params, FittingParameters)
                self.assertTrue(np.isnan(actual_fitting_params.cooling_index.std_dev))
                self.assertTrue(np.isnan(actual_fitting_params.ionization_rate.std_dev))
                self.assertTrue(np.isnan(actual_fitting_params.cutoff_speed.std_dev))
                self.assertTrue(np.isnan(actual_fitting_params.background_count_rate.std_dev))
                self.assertEqual(SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY, actual_fitting_params.flags)
