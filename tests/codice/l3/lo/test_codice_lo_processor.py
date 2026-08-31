import unittest
import warnings
from dataclasses import fields
from datetime import datetime, timedelta, date
from pathlib import Path
from unittest.mock import Mock, patch, call, sentinel, MagicMock

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup import GeometricFactorLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, CodiceLoL2SWSpeciesData, \
    CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct, \
    CODICE_LO_L2_NUM_PRIORITIES, CodiceLoL3ChargeStateDistributionsDataProduct, CodiceLoL3a3dDistributionDataProduct
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from tests.test_helpers import create_dataclass_mock, get_test_data_path, NumpyArrayMatcher

MODULE = "imap_l3_processing.codice.l3.lo.codice_lo_processor"


class TestCodiceLoProcessor(unittest.TestCase):
    def test_implements_processor(self):
        processor = CodiceLoProcessor(Mock(), Mock())
        self.assertIsInstance(processor, Processor)

    @patch(f'{MODULE}.CodiceLoL3aPartialDensitiesDependencies.fetch_dependencies')
    @patch(f'{MODULE}.CodiceLoProcessor.process_l3a_partial_densities')
    @patch(f'{MODULE}.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_partial_densities(self, mock_spiceypy, mock_save_data, mock_process_l3a_partial_densities,
                                       mock_fetch_dependencies):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-partial-densities')
        mock_spiceypy.ktotal.return_value = 0

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_partial_densities.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_partial_densities.return_value)

        self.assertEqual(['parent_file_1', 'parent_file_2'],
                         mock_process_l3a_partial_densities.return_value.parent_file_names)
        self.assertEqual([mock_save_data.return_value], product)

    @patch(f'{MODULE}.CodiceLoL3aRatiosDependencies.fetch_dependencies')
    @patch(f'{MODULE}.CodiceLoProcessor.process_l3a_ratios')
    @patch(f'{MODULE}.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_ratios(self, mock_spiceypy, mock_save_data, mock_process_l3a_ratios,
                            mock_fetch_dependencies):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-sw-ratios')
        mock_spiceypy.ktotal.return_value = 0

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_ratios.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_ratios.return_value)

        self.assertEqual(['parent_file_1'],
                         mock_process_l3a_ratios.return_value.parent_file_names)
        self.assertEqual([mock_save_data.return_value], product)

    @patch(f'{MODULE}.CodiceLoL3aRatiosDependencies.fetch_dependencies')
    @patch(f'{MODULE}.CodiceLoProcessor.process_l3a_charge_state_distributions')
    @patch(f'{MODULE}.save_data')
    @patch(f'imap_l3_processing.processor.spiceypy')
    def test_process_abundances(self, mock_spiceypy, mock_save_data, mock_process_l3a_abundances,
                                mock_fetch_dependencies):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-sw-charge-state-distributions')
        mock_spiceypy.ktotal.return_value = 0

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_abundances.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_abundances.return_value)

        self.assertEqual(['parent_file_1'],
                         mock_process_l3a_abundances.return_value.parent_file_names)
        self.assertEqual([mock_save_data.return_value], product)

    def test_process_ratios_calculate_abundance_ratios(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8),
                 now + timedelta(minutes=12), now + timedelta(minutes=16)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.array([1, 2, 3, 10, 12]),
            fe_loq_partial_density=np.array([4, 5, 6, 11, 13]),
            oplus5_partial_density=np.array([1, 2, 4, 12, 14]),
            oplus6_partial_density=np.array([5, 6, 7, 13, 15]),
            oplus7_partial_density=np.array([8, 9, 10, 14, 16]),
            oplus8_partial_density=np.array([11, 12, 13, 15, 17]),
            mg_partial_density=np.array([14, 15, 16, 16, 18]),
            cplus4_partial_density=np.array([17, 18, 19, 17, 19]),
            cplus5_partial_density=np.array([20, 21, 22, 18, 20]),
            cplus6_partial_density=np.array([23, 24, 25, 19, 21]),
            ne_partial_density=np.arange(5),
            si_partial_density=np.arange(5),
            heplusplus_partial_density=np.arange(5),
            hplus_partial_density=np.arange(5),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        ratios_data_product: CodiceLoL3aRatiosDataProduct = processor.process_l3a_ratios(dependency)
        self.assertEqual(input_metadata, ratios_data_product.input_metadata)
        np.testing.assert_array_equal(ratios_data_product.epoch,
                                      [now + timedelta(minutes=4), now + timedelta(minutes=14)])
        np.testing.assert_array_equal(ratios_data_product.epoch_delta, [360_000_000_000, 240_000_000_000])
        np.testing.assert_array_almost_equal(ratios_data_product.fe_to_o_ratio,
                                             np.array(
                                                 [(2 + 5) / (7 / 3 + 6 + 9 + 12), (11 + 12) / (13 + 14 + 15 + 16)]))
        np.testing.assert_array_almost_equal(ratios_data_product.mg_to_o_ratio,
                                             np.array([15 / (7 / 3 + 6 + 9 + 12), 17 / (13 + 14 + 15 + 16)]))
        np.testing.assert_array_almost_equal(ratios_data_product.c_to_o_ratio, np.array(
            [(18 + 21 + 24) / (7 / 3 + 6 + 9 + 12), (18 + 19 + 20) / (13 + 14 + 15 + 16)]))

        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c4_ratio, np.array([24 / 18, 20 / 18]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c5_ratio, np.array([24 / 21, 20 / 19]))

        np.testing.assert_array_almost_equal(ratios_data_product.o7_to_o6_ratio, np.array([9 / 6, 15 / 14]))
        np.testing.assert_array_almost_equal(ratios_data_product.felo_to_fehi_ratio, np.array([5 / 2, 12 / 11]))

    def test_process_ratios_calculate_abundance_ratios_using_safe_divide(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.array([0, 0, 0]),
            fe_loq_partial_density=np.array([0, 0, 0]),
            oplus5_partial_density=np.array([0, 0, 0]),
            oplus6_partial_density=np.array([0, 0, 0]),
            oplus7_partial_density=np.array([0, 0, 0]),
            oplus8_partial_density=np.array([0, 0, 0]),
            mg_partial_density=np.array([0, 0, 0]),
            cplus4_partial_density=np.array([0, 0, 0]),
            cplus5_partial_density=np.array([0, 0, 0]),
            cplus6_partial_density=np.array([0, 0, 0]),
            ne_partial_density=np.arange(3),
            si_partial_density=np.arange(3),
            heplusplus_partial_density=np.arange(3),
            hplus_partial_density=np.arange(3),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        with warnings.catch_warnings(record=True) as w:
            ratios_data_product = processor.process_l3a_ratios(dependency)
        self.assertEqual(0, len(w))

        self.assertEqual(input_metadata, ratios_data_product.input_metadata)
        np.testing.assert_array_equal(ratios_data_product.epoch,
                                      [now + timedelta(minutes=4)])
        np.testing.assert_array_equal(ratios_data_product.epoch_delta, [360_000_000_000])
        np.testing.assert_array_almost_equal(ratios_data_product.fe_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.mg_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c4_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c5_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.o7_to_o6_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.felo_to_fehi_ratio, np.array([np.nan]))

    def test_process_abundances_calculates_abundance_ratios(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8),
                 now + timedelta(minutes=12), now + timedelta(minutes=16)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.arange(5),
            fe_loq_partial_density=np.arange(5),
            oplus5_partial_density=np.array([0, 1, 2, 2, 2]),
            oplus6_partial_density=np.array([5, 5, 5, 6, 6]),
            oplus7_partial_density=np.array([8, 8, 8, 9, 9]),
            oplus8_partial_density=np.array([11, 11, 11, 12, 12, ]),
            mg_partial_density=np.arange(5),
            cplus4_partial_density=np.array([16, 17, 18, 18, 18]),
            cplus5_partial_density=np.array([20, 20, 20, 21, 21]),
            cplus6_partial_density=np.array([23, 23, 23, 24, 24]),
            ne_partial_density=np.arange(5),
            si_partial_density=np.arange(5),
            heplusplus_partial_density=np.arange(5),
            hplus_partial_density=np.arange(5),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        abundances_data_product = processor.process_l3a_charge_state_distributions(dependency)
        self.assertIsInstance(abundances_data_product, CodiceLoL3ChargeStateDistributionsDataProduct)
        self.assertEqual(input_metadata, abundances_data_product.input_metadata)
        np.testing.assert_array_equal(abundances_data_product.epoch,
                                      [now + timedelta(minutes=4), now + timedelta(minutes=14)])
        np.testing.assert_array_equal(abundances_data_product.epoch_delta, [360_000_000_000, 240_000_000_000])

        np.testing.assert_array_almost_equal(abundances_data_product.oxygen_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [1 / (1 + 5 + 8 + 11), 5 / (1 + 5 + 8 + 11), 8 / (1 + 5 + 8 + 11),
                                                      11 / (1 + 5 + 8 + 11), ],
                                                     [2 / (2 + 6 + 9 + 12), 6 / (2 + 6 + 9 + 12), 9 / (2 + 6 + 9 + 12),
                                                      12 / (2 + 6 + 9 + 12), ],
                                                 ])
                                             )
        np.testing.assert_array_almost_equal(abundances_data_product.carbon_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [17 / (17 + 20 + 23), 20 / (17 + 20 + 23), 23 / (17 + 20 + 23)],
                                                     [18 / (18 + 21 + 24), 21 / (18 + 21 + 24), 24 / (18 + 21 + 24)],
                                                 ]
                                             ))

    def test_process_abundances_uses_safe_divide(self):

        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now]),
            epoch_delta=np.array([120_000_000_000]),
            fe_hiq_partial_density=np.array([1]),
            fe_loq_partial_density=np.array([1]),
            oplus5_partial_density=np.array([0]),
            oplus6_partial_density=np.array([0]),
            oplus7_partial_density=np.array([0]),
            oplus8_partial_density=np.array([0]),
            mg_partial_density=np.array([1]),
            cplus4_partial_density=np.array([0]),
            cplus5_partial_density=np.array([0]),
            cplus6_partial_density=np.array([0]),
            ne_partial_density=np.array([1]),
            si_partial_density=np.array([1]),
            heplusplus_partial_density=np.array([1]),
            hplus_partial_density=np.array([1]),
        )
        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        with warnings.catch_warnings(record=True) as w:
            abundances_data_product = processor.process_l3a_charge_state_distributions(dependency)
        self.assertEqual(0, len(w))

        self.assertIsInstance(abundances_data_product, CodiceLoL3ChargeStateDistributionsDataProduct)
        self.assertEqual(input_metadata, abundances_data_product.input_metadata)
        np.testing.assert_array_equal(abundances_data_product.epoch,
                                      [now])
        np.testing.assert_array_equal(abundances_data_product.epoch_delta,
                                      [120_000_000_000])

        np.testing.assert_array_almost_equal(abundances_data_product.oxygen_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [np.nan, np.nan, np.nan, np.nan]
                                                 ])
                                             )
        np.testing.assert_array_almost_equal(abundances_data_product.carbon_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [np.nan, np.nan, np.nan]
                                                 ]
                                             ))

    @patch(f'{MODULE}.CodiceLoL3aDirectEventsDependencies.fetch_dependencies')
    @patch(f'{MODULE}.CodiceLoProcessor.process_l3a_direct_event_data_product')
    @patch(f'{MODULE}.save_data')
    @patch(f'imap_l3_processing.processor.spiceypy')
    def test_process_direct_events(self, mock_spiceypy, mock_save_data, mock_process_direct_event,
                                   mock_fetch_dependencies):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-direct-events')
        mock_spiceypy.ktotal.return_value = 0

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_direct_event.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_direct_event.return_value)

        self.assertEqual([mock_save_data.return_value], product)

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02', "bad-descriptor")

        processor = CodiceLoProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level and descriptor for CoDICE: L2a, bad-descriptor", str(context.exception))

    @patch(f'{MODULE}.calculate_partial_densities')
    def test_process_l3a_partial_densities(self, mock_calculate_partial_densities):
        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02',
                                       descriptor='lo-partial-densities')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)

        epochs = np.array([datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)])
        num_species = 14

        codice_lo_l2_data = CodiceLoL2SWSpeciesData(
            *[Mock() for _ in range(len(fields(CodiceLoL2SWSpeciesData)))],
        )

        mass_per_charge_lookup = MassPerChargeLookup(*[i for i in range(len(fields(MassPerChargeLookup)))])

        codice_lo_l2_data.epoch = epochs

        cplus4_partial_density = np.array([1])
        cplus5_partial_density = np.array([2])
        cplus6_partial_density = np.array([3])
        oplus5_partial_density = np.array([4])
        oplus6_partial_density = np.array([5])
        oplus7_partial_density = np.array([6])
        oplus8_partial_density = np.array([7])
        mg_partial_density = np.array([8])
        fe_loq_partial_density = np.array([9])
        fe_hiq_partial_density = np.array([10])

        mock_calculate_partial_densities.side_effect = [
            sentinel.hplus_partial_density,
            sentinel.heplusplus_partial_density,
            cplus4_partial_density,
            cplus5_partial_density,
            cplus6_partial_density,
            oplus5_partial_density,
            oplus6_partial_density,
            oplus7_partial_density,
            oplus8_partial_density,
            sentinel.ne_partial_density,
            mg_partial_density,
            sentinel.si_partial_density,
            fe_loq_partial_density,
            fe_hiq_partial_density,
        ]

        codice_lo_dependencies = CodiceLoL3aPartialDensitiesDependencies(
            codice_lo_l2_data, mass_per_charge_lookup
        )
        result_data_product = processor.process_l3a_partial_densities(
            codice_lo_dependencies
        )

        self.assertEqual(num_species, mock_calculate_partial_densities.call_count)

        mock_calculate_partial_densities.assert_has_calls(
            [
                call(
                    codice_lo_l2_data.hplus,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.hplus,
                ),
                call(
                    codice_lo_l2_data.heplusplus,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.heplusplus,
                ),
                call(
                    codice_lo_l2_data.cplus4,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.cplus4,
                ),
                call(
                    codice_lo_l2_data.cplus5,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.cplus5,
                ),
                call(
                    codice_lo_l2_data.cplus6,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.cplus6,
                ),
                call(
                    codice_lo_l2_data.oplus5,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.oplus5,
                ),
                call(
                    codice_lo_l2_data.oplus6,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.oplus6,
                ),
                call(
                    codice_lo_l2_data.oplus7,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.oplus7,
                ),
                call(
                    codice_lo_l2_data.oplus8,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.oplus8,
                ),
                call(
                    codice_lo_l2_data.ne,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.ne,
                ),
                call(
                    codice_lo_l2_data.mg,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.mg,
                ),
                call(
                    codice_lo_l2_data.si,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.si,
                ),
                call(
                    codice_lo_l2_data.fe_loq,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.fe_loq,
                ),
                call(
                    codice_lo_l2_data.fe_hiq,
                    codice_lo_l2_data.energy_per_charge,
                    mass_per_charge_lookup.fe_hiq,
                ),
            ]
        )

        self.assertIsInstance(result_data_product, CodiceLoL3aPartialDensityDataProduct)
        self.assertEqual(input_metadata, result_data_product.input_metadata)
        result_data = result_data_product.data
        self.assertIsInstance(result_data, CodiceLoPartialDensityData)

        np.testing.assert_array_equal(result_data.epoch, codice_lo_l2_data.epoch)
        np.testing.assert_array_equal(result_data.epoch_delta, codice_lo_l2_data.epoch_delta_plus)
        self.assertEqual(sentinel.hplus_partial_density, result_data.hplus_partial_density),
        self.assertEqual(sentinel.heplusplus_partial_density, result_data.heplusplus_partial_density),
        self.assertEqual(cplus4_partial_density, result_data.cplus4_partial_density),
        self.assertEqual(cplus5_partial_density, result_data.cplus5_partial_density),
        self.assertEqual(cplus6_partial_density, result_data.cplus6_partial_density),
        self.assertEqual(oplus5_partial_density, result_data.oplus5_partial_density),
        self.assertEqual(oplus6_partial_density, result_data.oplus6_partial_density),
        self.assertEqual(oplus7_partial_density, result_data.oplus7_partial_density),
        self.assertEqual(oplus8_partial_density, result_data.oplus8_partial_density),
        self.assertEqual(sentinel.ne_partial_density, result_data.ne_partial_density),
        self.assertEqual(mg_partial_density, result_data.mg_partial_density),
        self.assertEqual(sentinel.si_partial_density, result_data.si_partial_density),
        self.assertEqual(fe_loq_partial_density, result_data.fe_loq_partial_density),
        self.assertEqual(fe_hiq_partial_density, result_data.fe_hiq_partial_density),

    @patch(f'{MODULE}.SpinAngleLookup')
    @patch(f'{MODULE}.lookup_normalization_per_event', autospec=True)
    @patch(f'{MODULE}.calculate_normalization_factor', autospec=True)
    @patch(f'{MODULE}.calculate_mass_per_charge', autospec=True)
    @patch(f'{MODULE}.calculate_mass', autospec=True)
    def test_process_l3a_direct_events(self, mock_calculate_mass, mock_calculate_mass_per_charge,
                                       mock_calculate_normalization_factor,
                                       mock_lookup_normalization_per_event,
                                       mock_spin_angle_lookup_class):
        rng = np.random.default_rng()

        num_spin_angle_bins = 24
        num_energy_bins = 128
        event_buffer_size = 10
        num_priorities = 7

        mock_spin_angle_lookup = create_dataclass_mock(SpinAngleLookup)
        mock_spin_angle_lookup.num_bins = num_spin_angle_bins
        mock_spin_angle_lookup_class.return_value = mock_spin_angle_lookup

        epochs = np.array([datetime.now(), datetime.now() + timedelta(hours=1)])
        num_epochs = len(epochs)

        priority_counts_variable_shape = (num_epochs, num_energy_bins, num_spin_angle_bins // 2)
        sw_priority_rates = create_dataclass_mock(CodiceLoL1aSWPriorityRates)
        sw_priority_rates.epoch = epochs
        sw_priority_rates.p0_tcrs = rng.random(priority_counts_variable_shape)
        sw_priority_rates.p1_hplus = rng.random(priority_counts_variable_shape)
        sw_priority_rates.p2_heplusplus = rng.random(priority_counts_variable_shape)
        sw_priority_rates.p3_heavies = rng.random(priority_counts_variable_shape)
        sw_priority_rates.p4_dcrs = rng.random(priority_counts_variable_shape)
        sw_priority_rates.half_spin_per_esa_step = rng.random((num_epochs, num_energy_bins))
        sw_priority_rates.rgfo_spin_sector = rng.random(num_epochs)
        sw_priority_rates.rgfo_esa_step = rng.random(num_epochs)
        sw_priority_rates.nso_spin_sector = rng.random(num_epochs)
        sw_priority_rates.nso_esa_step = rng.random(num_epochs)
        sw_priority_rates.esa_step = np.arange(128)
        sw_priority_rates.rgfo_half_spin = rng.random(num_epochs)
        sw_priority_rates.nso_half_spin = rng.random(num_epochs)
        sw_priority_rates.acquisition_time_per_esa_step = rng.random((num_epochs, num_energy_bins))

        nsw_priority_rates = create_dataclass_mock(CodiceLoL1aNSWPriorityRates)
        nsw_priority_rates.epoch = epochs
        nsw_priority_rates.p5_heavies = rng.random(priority_counts_variable_shape)
        nsw_priority_rates.p6_hplus_heplusplus = rng.random(priority_counts_variable_shape)

        mass_per_charge = rng.random((num_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass_per_charge.return_value = mass_per_charge
        mass = rng.random((num_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass.return_value = mass

        codice_l2_variables = {f.name: rng.random((num_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size)) for f
                               in fields(CodiceLoL2DirectEventData)}

        codice_l2_variables["epoch"] = epochs
        codice_l2_variables["spin_angle"] *= 360
        codice_l2_variables["data_quality"] = rng.random((num_epochs))
        codice_l2_variables["num_events"] = rng.random((num_epochs))

        expected_energy_per_charge = codice_l2_variables["energy_per_charge"]
        expected_apd_energy = codice_l2_variables["apd_energy"]
        expected_apd_id = codice_l2_variables["apd_id"]
        expected_data_quality = codice_l2_variables["data_quality"]
        expected_energy_step = codice_l2_variables["energy_step"]
        expected_apd_gain = codice_l2_variables["gain"]
        expected_multi_flag = codice_l2_variables["multi_flag"]
        expected_elevation = codice_l2_variables["elevation_angle"]
        expected_position = codice_l2_variables["position"]
        expected_mass = mass
        expected_mass_per_charge = mass_per_charge
        expected_num_events = codice_l2_variables["num_events"]
        expected_spin_angle = codice_l2_variables["spin_angle"]
        expected_spin_sector = codice_l2_variables["spin_sector"]
        expected_tof = codice_l2_variables["tof"]
        expected_type = codice_l2_variables["type"]

        direct_events = CodiceLoL2DirectEventData(**codice_l2_variables)

        mock_energy_lookup = create_dataclass_mock(EnergyLookup)
        mock_energy_lookup.delta_minus = np.geomspace(280, 2, 128)
        mock_energy_lookup.delta_plus = np.geomspace(288, 1, 128)
        mock_energy_lookup.bin_centers = np.geomspace(14100, 88.082825, 128)
        mock_energy_lookup.num_bins = num_energy_bins

        expected_normalization = rng.random((num_epochs, num_priorities, num_energy_bins, num_spin_angle_bins))
        mock_calculate_normalization_factor.return_value = expected_normalization

        dependencies = CodiceLoL3aDirectEventsDependencies(sw_priority_rates, nsw_priority_rates, direct_events, Mock(),
                                                           mock_energy_lookup)

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        l3a_direct_event_data_product = processor.process_l3a_direct_event_data_product(dependencies)

        self.assertEqual(1, mock_calculate_mass.call_count)
        np.testing.assert_equal(mock_calculate_mass.call_args.args[0], expected_apd_energy)
        np.testing.assert_equal(mock_calculate_mass.call_args.args[1], expected_tof)
        np.testing.assert_equal(mock_calculate_mass.call_args.args[2], dependencies.mass_coefficient_lookup)

        self.assertEqual(1, mock_calculate_mass_per_charge.call_count)
        np.testing.assert_equal(mock_calculate_mass_per_charge.call_args.args[0], expected_energy_per_charge)
        np.testing.assert_equal(mock_calculate_mass_per_charge.call_args.args[1], expected_tof)

        self.assertEqual(1, mock_calculate_normalization_factor.call_count)
        actual_stacked_priorities = mock_calculate_normalization_factor.call_args.args[0]

        np.testing.assert_equal(actual_stacked_priorities[:, 0], sw_priority_rates.p0_tcrs)
        np.testing.assert_equal(actual_stacked_priorities[:, 1], sw_priority_rates.p1_hplus)
        np.testing.assert_equal(actual_stacked_priorities[:, 2], sw_priority_rates.p2_heplusplus)
        np.testing.assert_equal(actual_stacked_priorities[:, 3], sw_priority_rates.p3_heavies)
        np.testing.assert_equal(actual_stacked_priorities[:, 4], sw_priority_rates.p4_dcrs)
        np.testing.assert_equal(actual_stacked_priorities[:, 5], nsw_priority_rates.p5_heavies)
        np.testing.assert_equal(actual_stacked_priorities[:, 6], nsw_priority_rates.p6_hplus_heplusplus)

        np.testing.assert_equal(mock_calculate_normalization_factor.call_args.args[1], expected_num_events)
        np.testing.assert_equal(mock_calculate_normalization_factor.call_args.args[2], expected_energy_step)
        np.testing.assert_equal(mock_calculate_normalization_factor.call_args.args[3], expected_spin_sector)

        self.assertEqual(1, mock_lookup_normalization_per_event.call_count)
        np.testing.assert_equal(mock_lookup_normalization_per_event.call_args.args[0], expected_normalization)
        np.testing.assert_equal(mock_lookup_normalization_per_event.call_args.args[1], expected_num_events)
        np.testing.assert_equal(mock_lookup_normalization_per_event.call_args.args[2], expected_energy_step)
        np.testing.assert_equal(mock_lookup_normalization_per_event.call_args.args[3], expected_spin_sector)

        self.assertIsInstance(l3a_direct_event_data_product, CodiceLoL3aDirectEventDataProduct)
        self.assertEqual(input_metadata, l3a_direct_event_data_product.input_metadata)

        np.testing.assert_array_equal(l3a_direct_event_data_product.epoch, epochs)
        np.testing.assert_array_equal(l3a_direct_event_data_product.epoch_delta, direct_events.epoch_delta_plus)

        np.testing.assert_array_equal(
            np.arange(CODICE_LO_L2_NUM_PRIORITIES), l3a_direct_event_data_product.priority_index
        )
        np.testing.assert_array_equal(expected_mass_per_charge, l3a_direct_event_data_product.mass_per_charge)
        np.testing.assert_array_equal(expected_mass, l3a_direct_event_data_product.mass)

        np.testing.assert_array_equal(
            l3a_direct_event_data_product.normalization, np.flip(expected_normalization, axis=2)
        )
        self.assertEqual(l3a_direct_event_data_product.normalization_per_event,
                         mock_lookup_normalization_per_event.return_value)

        np.testing.assert_array_equal(expected_apd_energy, l3a_direct_event_data_product.apd_energy)
        np.testing.assert_array_equal(expected_apd_gain, l3a_direct_event_data_product.gain)
        np.testing.assert_array_equal(expected_apd_id, l3a_direct_event_data_product.apd_id)
        np.testing.assert_array_equal(expected_data_quality, l3a_direct_event_data_product.data_quality)
        np.testing.assert_array_equal(expected_elevation, l3a_direct_event_data_product.elevation)
        np.testing.assert_array_equal(expected_energy_per_charge, l3a_direct_event_data_product.energy_per_charge)
        np.testing.assert_array_equal(expected_energy_step, l3a_direct_event_data_product.energy_step)
        np.testing.assert_array_equal(expected_multi_flag, l3a_direct_event_data_product.multi_flag)
        np.testing.assert_array_equal(expected_num_events, l3a_direct_event_data_product.num_events)
        np.testing.assert_array_equal(expected_position, l3a_direct_event_data_product.position)
        np.testing.assert_array_equal(expected_spin_angle, l3a_direct_event_data_product.spin_angle)
        np.testing.assert_array_equal(expected_spin_sector, l3a_direct_event_data_product.spin_sector)
        np.testing.assert_array_equal(expected_tof, l3a_direct_event_data_product.tof)
        np.testing.assert_array_equal(expected_type, l3a_direct_event_data_product.type)

        np.testing.assert_array_equal(np.flip(mock_energy_lookup.bin_centers), l3a_direct_event_data_product.energy_bin)
        np.testing.assert_array_equal(np.flip(mock_energy_lookup.delta_plus),
                                      l3a_direct_event_data_product.energy_bin_delta_plus)
        np.testing.assert_array_equal(np.flip(mock_energy_lookup.delta_minus),
                                      l3a_direct_event_data_product.energy_bin_delta_minus)
        self.assertEqual(mock_spin_angle_lookup.bin_centers, l3a_direct_event_data_product.spin_angle_bin)
        self.assertEqual(mock_spin_angle_lookup.bin_deltas, l3a_direct_event_data_product.spin_angle_bin_delta)
        np.testing.assert_array_equal(sw_priority_rates.half_spin_per_esa_step,
                                      l3a_direct_event_data_product.half_spin_per_esa_step)
        np.testing.assert_array_equal(sw_priority_rates.rgfo_spin_sector,
                                      l3a_direct_event_data_product.rgfo_spin_sector)
        np.testing.assert_array_equal(sw_priority_rates.rgfo_esa_step, l3a_direct_event_data_product.rgfo_esa_step)
        np.testing.assert_array_equal(sw_priority_rates.nso_spin_sector, l3a_direct_event_data_product.nso_spin_sector)
        np.testing.assert_array_equal(sw_priority_rates.nso_esa_step, l3a_direct_event_data_product.nso_esa_step)
        np.testing.assert_array_equal(sw_priority_rates.esa_step, l3a_direct_event_data_product.esa_step)
        np.testing.assert_array_equal(sw_priority_rates.rgfo_half_spin, l3a_direct_event_data_product.rgfo_half_spin)
        np.testing.assert_array_equal(sw_priority_rates.nso_half_spin, l3a_direct_event_data_product.nso_half_spin)
        np.testing.assert_array_equal(sw_priority_rates.acquisition_time_per_esa_step,
                                      l3a_direct_event_data_product.acquisition_time_per_esa_step)

    @patch(f'{MODULE}.SpinAngleLookup')
    @patch(f'{MODULE}.lookup_normalization_per_event', autospec=True)
    @patch(f'{MODULE}.calculate_normalization_factor', autospec=True)
    @patch(f'{MODULE}.calculate_mass_per_charge', autospec=True)
    @patch(f'{MODULE}.calculate_mass', autospec=True)
    def test_process_l3a_direct_events_aligns_priority_to_direct_event_epochs_when_missing(
            self, mock_calculate_mass, mock_calculate_mass_per_charge,
            mock_calculate_normalization_factor, mock_lookup_normalization_per_event,
            mock_spin_angle_lookup_class):
        rng = np.random.default_rng(seed=42)

        num_spin_angle_bins = 4
        num_energy_bins = 3
        event_buffer_size = 2
        num_priorities = 7
        priority_l1_spin_sectors = num_spin_angle_bins // 2

        mock_spin_angle_lookup = create_dataclass_mock(SpinAngleLookup)
        mock_spin_angle_lookup.num_bins = num_spin_angle_bins
        mock_spin_angle_lookup_class.return_value = mock_spin_angle_lookup

        t0 = datetime(2026, 4, 23, 5, 10, 30)
        t1 = datetime(2026, 4, 23, 5, 10, 31)
        t2 = datetime(2026, 4, 23, 5, 10, 32)
        t3 = datetime(2026, 4, 23, 5, 10, 33)

        direct_event_epochs = np.array([t0, t1, t2, t3])
        num_direct_event_epochs = len(direct_event_epochs)

        sw_priority_epochs = np.array([t0, t2, t3])
        nsw_priority_epochs = np.array([t0, t1, t3])
        num_sw_epochs = len(sw_priority_epochs)
        num_nsw_epochs = len(nsw_priority_epochs)

        sw_priority_counts_shape = (num_sw_epochs, num_energy_bins, priority_l1_spin_sectors)
        nsw_priority_counts_shape = (num_nsw_epochs, num_energy_bins, priority_l1_spin_sectors)

        sw_priority_rates = create_dataclass_mock(CodiceLoL1aSWPriorityRates)
        sw_priority_rates.epoch = sw_priority_epochs
        sw_priority_rates.p0_tcrs = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p1_hplus = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p2_heplusplus = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p3_heavies = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p4_dcrs = rng.random(sw_priority_counts_shape)
        sw_priority_rates.half_spin_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))
        sw_priority_rates.rgfo_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.rgfo_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.nso_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.nso_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.esa_step = np.arange(num_energy_bins)
        sw_priority_rates.rgfo_half_spin = rng.random(num_sw_epochs)
        sw_priority_rates.nso_half_spin = rng.random(num_sw_epochs)
        sw_priority_rates.acquisition_time_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))

        nsw_priority_rates = create_dataclass_mock(CodiceLoL1aNSWPriorityRates)
        nsw_priority_rates.epoch = nsw_priority_epochs
        nsw_priority_rates.p5_heavies = rng.random(nsw_priority_counts_shape)
        nsw_priority_rates.p6_hplus_heplusplus = rng.random(nsw_priority_counts_shape)

        mass_per_charge = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass_per_charge.return_value = mass_per_charge
        mass = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass.return_value = mass

        codice_l2_variables = {
            f.name: rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
            for f in fields(CodiceLoL2DirectEventData)
        }
        codice_l2_variables["epoch"] = direct_event_epochs
        codice_l2_variables["spin_angle"] *= 360
        codice_l2_variables["data_quality"] = rng.random((num_direct_event_epochs,))
        codice_l2_variables["num_events"] = rng.random((num_direct_event_epochs,))

        direct_events = CodiceLoL2DirectEventData(**codice_l2_variables)

        mock_energy_lookup = create_dataclass_mock(EnergyLookup)
        mock_energy_lookup.delta_minus = rng.random(num_energy_bins)
        mock_energy_lookup.delta_plus = rng.random(num_energy_bins)
        mock_energy_lookup.bin_centers = rng.random(num_energy_bins)
        mock_energy_lookup.num_bins = num_energy_bins

        expected_normalization = rng.random(
            (num_direct_event_epochs, num_priorities, num_energy_bins, num_spin_angle_bins)
        )
        mock_calculate_normalization_factor.return_value = expected_normalization

        sentinel_normalization_per_event_value = 5.0
        mock_lookup_normalization_per_event.return_value = np.full(
            (num_direct_event_epochs, num_priorities, event_buffer_size),
            sentinel_normalization_per_event_value,
        )

        dependencies = CodiceLoL3aDirectEventsDependencies(
            sw_priority_rates, nsw_priority_rates, direct_events, Mock(), mock_energy_lookup
        )

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        result = processor.process_l3a_direct_event_data_product(dependencies)

        self.assertEqual(1, mock_calculate_normalization_factor.call_count)
        actual_stacked_priorities = mock_calculate_normalization_factor.call_args.args[0]
        self.assertEqual(
            (num_direct_event_epochs, num_priorities, num_energy_bins, priority_l1_spin_sectors),
            actual_stacked_priorities.shape,
        )

        np.testing.assert_equal(actual_stacked_priorities[0, 0], sw_priority_rates.p0_tcrs[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 1], sw_priority_rates.p1_hplus[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 2], sw_priority_rates.p2_heplusplus[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 3], sw_priority_rates.p3_heavies[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 4], sw_priority_rates.p4_dcrs[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 5], nsw_priority_rates.p5_heavies[0])
        np.testing.assert_equal(actual_stacked_priorities[0, 6], nsw_priority_rates.p6_hplus_heplusplus[0])

        expected_sw_nan_block = np.full((5, num_energy_bins, priority_l1_spin_sectors), np.nan)
        np.testing.assert_equal(actual_stacked_priorities[1, 0:5], expected_sw_nan_block)
        np.testing.assert_equal(actual_stacked_priorities[1, 5], nsw_priority_rates.p5_heavies[1])
        np.testing.assert_equal(actual_stacked_priorities[1, 6], nsw_priority_rates.p6_hplus_heplusplus[1])

        np.testing.assert_equal(actual_stacked_priorities[2, 0], sw_priority_rates.p0_tcrs[1])
        np.testing.assert_equal(actual_stacked_priorities[2, 1], sw_priority_rates.p1_hplus[1])
        np.testing.assert_equal(actual_stacked_priorities[2, 2], sw_priority_rates.p2_heplusplus[1])
        np.testing.assert_equal(actual_stacked_priorities[2, 3], sw_priority_rates.p3_heavies[1])
        np.testing.assert_equal(actual_stacked_priorities[2, 4], sw_priority_rates.p4_dcrs[1])
        expected_nsw_nan_block = np.full((2, num_energy_bins, priority_l1_spin_sectors), np.nan)
        np.testing.assert_equal(actual_stacked_priorities[2, 5:7], expected_nsw_nan_block)

        np.testing.assert_equal(actual_stacked_priorities[3, 0], sw_priority_rates.p0_tcrs[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 1], sw_priority_rates.p1_hplus[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 2], sw_priority_rates.p2_heplusplus[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 3], sw_priority_rates.p3_heavies[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 4], sw_priority_rates.p4_dcrs[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 5], nsw_priority_rates.p5_heavies[2])
        np.testing.assert_equal(actual_stacked_priorities[3, 6], nsw_priority_rates.p6_hplus_heplusplus[2])

        self.assertEqual(1, mock_lookup_normalization_per_event.call_count)
        np.testing.assert_equal(mock_lookup_normalization_per_event.call_args.args[0], expected_normalization)

        actual_normalization_per_event = result.normalization_per_event

        np.testing.assert_array_equal(
            actual_normalization_per_event[1, 0:5, :], np.zeros((5, event_buffer_size))
        )
        np.testing.assert_array_equal(
            actual_normalization_per_event[2, 5:7, :], np.zeros((2, event_buffer_size))
        )

        sentinel_row = np.full(event_buffer_size, sentinel_normalization_per_event_value)
        np.testing.assert_array_equal(
            actual_normalization_per_event[0],
            np.full((num_priorities, event_buffer_size), sentinel_normalization_per_event_value),
        )
        np.testing.assert_array_equal(
            actual_normalization_per_event[3],
            np.full((num_priorities, event_buffer_size), sentinel_normalization_per_event_value),
        )
        np.testing.assert_array_equal(actual_normalization_per_event[1, 5], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[1, 6], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[2, 0], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[2, 1], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[2, 2], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[2, 3], sentinel_row)
        np.testing.assert_array_equal(actual_normalization_per_event[2, 4], sentinel_row)

        np.testing.assert_array_equal(result.normalization, np.flip(expected_normalization, axis=2))

    @patch(f'{MODULE}.SpinAngleLookup')
    @patch(f'{MODULE}.lookup_normalization_per_event', autospec=True)
    @patch(f'{MODULE}.calculate_normalization_factor', autospec=True)
    @patch(f'{MODULE}.calculate_mass_per_charge', autospec=True)
    @patch(f'{MODULE}.calculate_mass', autospec=True)
    def test_process_l3a_direct_events_handles_masked_priority_counts(
            self, mock_calculate_mass, mock_calculate_mass_per_charge,
            mock_calculate_normalization_factor, mock_lookup_normalization_per_event,
            mock_spin_angle_lookup_class):
        rng = np.random.default_rng(seed=42)

        num_spin_angle_bins = 4
        num_energy_bins = 3
        event_buffer_size = 2
        num_priorities = 7
        priority_l1_spin_sectors = num_spin_angle_bins // 2

        mock_spin_angle_lookup = create_dataclass_mock(SpinAngleLookup)
        mock_spin_angle_lookup.num_bins = num_spin_angle_bins
        mock_spin_angle_lookup_class.return_value = mock_spin_angle_lookup

        t0 = datetime(2026, 4, 23, 5, 10, 30)
        t1 = datetime(2026, 4, 23, 5, 10, 31)
        t2 = datetime(2026, 4, 23, 5, 10, 32)
        t3 = datetime(2026, 4, 23, 5, 10, 33)

        direct_event_epochs = np.array([t0, t1, t2, t3])
        num_direct_event_epochs = len(direct_event_epochs)

        sw_priority_epochs = np.array([t0, t2, t3])
        nsw_priority_epochs = np.array([t0, t1, t3])
        num_sw_epochs = len(sw_priority_epochs)
        num_nsw_epochs = len(nsw_priority_epochs)

        sw_priority_counts_shape = (num_sw_epochs, num_energy_bins, priority_l1_spin_sectors)
        nsw_priority_counts_shape = (num_nsw_epochs, num_energy_bins, priority_l1_spin_sectors)
        sw_mask = np.full(sw_priority_counts_shape, False)
        sw_mask[0, 0, 0] = True
        sw_priority_rates = create_dataclass_mock(CodiceLoL1aSWPriorityRates)
        sw_priority_rates.epoch = sw_priority_epochs
        sw_priority_rates.p0_tcrs = np.ma.array(rng.random(sw_priority_counts_shape), mask=sw_mask)
        sw_priority_rates.p1_hplus = np.ma.array(rng.random(sw_priority_counts_shape), mask=sw_mask)
        sw_priority_rates.p2_heplusplus = np.ma.array(rng.random(sw_priority_counts_shape), mask=sw_mask)
        sw_priority_rates.p3_heavies = np.ma.array(rng.random(sw_priority_counts_shape), mask=sw_mask)
        sw_priority_rates.p4_dcrs = np.ma.array(rng.random(sw_priority_counts_shape), mask=sw_mask)
        sw_priority_rates.half_spin_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))
        sw_priority_rates.rgfo_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.rgfo_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.nso_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.nso_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.esa_step = np.arange(num_energy_bins)
        sw_priority_rates.rgfo_half_spin = rng.integers(0, 12, num_sw_epochs, np.uint8)
        sw_priority_rates.nso_half_spin = rng.integers(0, 12, num_sw_epochs, np.uint8)
        sw_priority_rates.acquisition_time_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))

        nsw_mask = np.full(nsw_priority_counts_shape, False)
        nsw_mask[0, 0, 0] = True
        nsw_priority_rates = create_dataclass_mock(CodiceLoL1aNSWPriorityRates)
        nsw_priority_rates.epoch = nsw_priority_epochs
        nsw_priority_rates.p5_heavies = np.ma.array(rng.random(nsw_priority_counts_shape), mask=nsw_mask)
        nsw_priority_rates.p6_hplus_heplusplus = np.ma.array(rng.random(nsw_priority_counts_shape), mask=nsw_mask)

        mass_per_charge = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass_per_charge.return_value = mass_per_charge
        mass = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass.return_value = mass

        codice_l2_variables = {
            f.name: rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
            for f in fields(CodiceLoL2DirectEventData)
        }
        codice_l2_variables["epoch"] = direct_event_epochs
        codice_l2_variables["spin_angle"] *= 360
        codice_l2_variables["data_quality"] = rng.random((num_direct_event_epochs,))
        codice_l2_variables["num_events"] = rng.random((num_direct_event_epochs,))

        direct_events = CodiceLoL2DirectEventData(**codice_l2_variables)

        mock_energy_lookup = create_dataclass_mock(EnergyLookup)
        mock_energy_lookup.delta_minus = rng.random(num_energy_bins)
        mock_energy_lookup.delta_plus = rng.random(num_energy_bins)
        mock_energy_lookup.bin_centers = rng.random(num_energy_bins)
        mock_energy_lookup.num_bins = num_energy_bins

        expected_normalization = rng.random(
            (num_direct_event_epochs, num_priorities, num_energy_bins, num_spin_angle_bins)
        )
        mock_calculate_normalization_factor.return_value = expected_normalization

        sentinel_normalization_per_event_value = 5.0
        mock_lookup_normalization_per_event.return_value = np.full(
            (num_direct_event_epochs, num_priorities, event_buffer_size),
            sentinel_normalization_per_event_value,
        )

        dependencies = CodiceLoL3aDirectEventsDependencies(
            sw_priority_rates, nsw_priority_rates, direct_events, Mock(), mock_energy_lookup
        )

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        result = processor.process_l3a_direct_event_data_product(dependencies)

        self.assertEqual(1, mock_calculate_normalization_factor.call_count)
        actual_stacked_priorities = mock_calculate_normalization_factor.call_args.args[0]
        self.assertEqual(
            (num_direct_event_epochs, num_priorities, num_energy_bins, priority_l1_spin_sectors),
            actual_stacked_priorities.shape,
        )

        np.testing.assert_equal(actual_stacked_priorities[0, :, 0, 0], np.full(7, np.nan))

    @patch(f'{MODULE}.SpinAngleLookup')
    @patch(f'{MODULE}.lookup_normalization_per_event', autospec=True)
    @patch(f'{MODULE}.calculate_normalization_factor', autospec=True)
    @patch(f'{MODULE}.calculate_mass_per_charge', autospec=True)
    @patch(f'{MODULE}.calculate_mass', autospec=True)
    def test_process_l3a_direct_events_aligns_passthrough_vars_to_direct_event_epochs_when_missing(
            self, mock_calculate_mass, mock_calculate_mass_per_charge,
            mock_calculate_normalization_factor, mock_lookup_normalization_per_event,
            mock_spin_angle_lookup_class):
        rng = np.random.default_rng(seed=42)

        num_spin_angle_bins = 4
        num_energy_bins = 3
        event_buffer_size = 2
        num_priorities = 7
        priority_l1_spin_sectors = num_spin_angle_bins // 2

        mock_spin_angle_lookup = create_dataclass_mock(SpinAngleLookup)
        mock_spin_angle_lookup.num_bins = num_spin_angle_bins
        mock_spin_angle_lookup_class.return_value = mock_spin_angle_lookup

        t0 = datetime(2026, 4, 23, 5, 10, 30)
        t1 = datetime(2026, 4, 23, 5, 10, 31)
        t2 = datetime(2026, 4, 23, 5, 10, 32)
        t3 = datetime(2026, 4, 23, 5, 10, 33)

        direct_event_epochs = np.array([t0, t1, t2, t3])
        num_direct_event_epochs = len(direct_event_epochs)

        sw_priority_epochs = np.array([t0, t2, t3])
        nsw_priority_epochs = np.array([t0, t1, t3])
        num_sw_epochs = len(sw_priority_epochs)
        num_nsw_epochs = len(nsw_priority_epochs)

        sw_priority_counts_shape = (num_sw_epochs, num_energy_bins, priority_l1_spin_sectors)
        nsw_priority_counts_shape = (num_nsw_epochs, num_energy_bins, priority_l1_spin_sectors)

        sw_priority_rates = create_dataclass_mock(CodiceLoL1aSWPriorityRates)
        sw_priority_rates.epoch = sw_priority_epochs
        sw_priority_rates.p0_tcrs = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p1_hplus = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p2_heplusplus = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p3_heavies = rng.random(sw_priority_counts_shape)
        sw_priority_rates.p4_dcrs = rng.random(sw_priority_counts_shape)
        sw_priority_rates.half_spin_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))
        sw_priority_rates.rgfo_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.rgfo_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.nso_spin_sector = rng.random(num_sw_epochs)
        sw_priority_rates.nso_esa_step = rng.random(num_sw_epochs)
        sw_priority_rates.esa_step = np.arange(num_energy_bins)
        sw_priority_rates.rgfo_half_spin = rng.integers(0, 12, num_sw_epochs, np.uint8)
        sw_priority_rates.nso_half_spin = rng.integers(0, 12, num_sw_epochs, np.uint8)
        sw_priority_rates.acquisition_time_per_esa_step = rng.random((num_sw_epochs, num_energy_bins))

        nsw_priority_rates = create_dataclass_mock(CodiceLoL1aNSWPriorityRates)
        nsw_priority_rates.epoch = nsw_priority_epochs
        nsw_priority_rates.p5_heavies = rng.random(nsw_priority_counts_shape)
        nsw_priority_rates.p6_hplus_heplusplus = rng.random(nsw_priority_counts_shape)

        mass_per_charge = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass_per_charge.return_value = mass_per_charge
        mass = rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
        mock_calculate_mass.return_value = mass

        codice_l2_variables = {
            f.name: rng.random((num_direct_event_epochs, CODICE_LO_L2_NUM_PRIORITIES, event_buffer_size))
            for f in fields(CodiceLoL2DirectEventData)
        }
        codice_l2_variables["epoch"] = direct_event_epochs
        codice_l2_variables["spin_angle"] *= 360
        codice_l2_variables["data_quality"] = rng.random((num_direct_event_epochs,))
        codice_l2_variables["num_events"] = rng.random((num_direct_event_epochs,))

        direct_events = CodiceLoL2DirectEventData(**codice_l2_variables)

        mock_energy_lookup = create_dataclass_mock(EnergyLookup)
        mock_energy_lookup.delta_minus = rng.random(num_energy_bins)
        mock_energy_lookup.delta_plus = rng.random(num_energy_bins)
        mock_energy_lookup.bin_centers = rng.random(num_energy_bins)
        mock_energy_lookup.num_bins = num_energy_bins

        expected_normalization = rng.random(
            (num_direct_event_epochs, num_priorities, num_energy_bins, num_spin_angle_bins)
        )
        mock_calculate_normalization_factor.return_value = expected_normalization

        sentinel_normalization_per_event_value = 5.0
        mock_lookup_normalization_per_event.return_value = np.full(
            (num_direct_event_epochs, num_priorities, event_buffer_size),
            sentinel_normalization_per_event_value,
        )

        dependencies = CodiceLoL3aDirectEventsDependencies(
            sw_priority_rates, nsw_priority_rates, direct_events, Mock(), mock_energy_lookup
        )

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        result = processor.process_l3a_direct_event_data_product(dependencies)

        variables_to_compare = [
            (sw_priority_rates.acquisition_time_per_esa_step, result.acquisition_time_per_esa_step),
            (sw_priority_rates.rgfo_half_spin, result.rgfo_half_spin),
            (sw_priority_rates.rgfo_spin_sector, result.rgfo_spin_sector),
            (sw_priority_rates.rgfo_esa_step, result.rgfo_esa_step),
            (sw_priority_rates.half_spin_per_esa_step, result.half_spin_per_esa_step),
            (sw_priority_rates.nso_spin_sector, result.nso_spin_sector),
            (sw_priority_rates.nso_esa_step, result.nso_esa_step),
            (sw_priority_rates.nso_half_spin, result.nso_half_spin),
        ]
        for input_var, result_var in variables_to_compare:
            np.testing.assert_equal(
                input_var[0],
                result_var[0],
            )
            np.testing.assert_equal(
                input_var[1],
                result_var[2],
            )
            np.testing.assert_equal(
                input_var[2],
                result_var[3],
            )

            self.assertTrue(np.all(result_var.mask[0] == False))
            self.assertTrue(np.all(result_var.mask[1] == True))
            self.assertTrue(np.all(result_var.mask[2] == False))
            self.assertTrue(np.all(result_var.mask[3] == False))

    @patch(f'{MODULE}.rebin_3d_distribution_azimuth_to_elevation')
    @patch(f'{MODULE}.combine_priorities_for_species_and_convert_to_rate')
    @patch(f'{MODULE}.PositionToElevationLookup')
    @patch(f'{MODULE}.rebin_to_counts_by_species_elevation_and_spin_sector')
    @patch(f'{MODULE}.convert_count_rate_to_intensity')
    def test_process_l3a_3d_distributions(self, mock_convert_count_rate_to_intensity, mock_rebin,
                                          mock_elevation_angle_lookup_class,
                                          mock_combine_priorities_for_species_and_convert_to_rate,
                                          mock_rebin_3d_distribution_azimuth_to_elevation):
        mock_elevation_lookup = mock_elevation_angle_lookup_class.return_value

        input_metadata = InputMetadata('codice', "l3a", datetime(2026, 5, 13), Mock(spec=datetime), 'v02')

        l3a_de_half_spin_per_esa_step = np.array([[3, 2, 1]])
        mock_l3a_direct_event_data = Mock(
            acquisition_time_per_esa_step=sentinel.acquisition_time,
            rgfo_half_spin=sentinel.rgfo_half_spin,
            rgfo_spin_sector=sentinel.rgfo_spin_sector,
            rgfo_esa_step=sentinel.rgfo_esa_step,
            half_spin_per_esa_step=l3a_de_half_spin_per_esa_step,
            spin_angle=sentinel.spin_angle,
            spin_angle_bin_delta=sentinel.spin_angle_bin_delta,
            spin_angle_bin=sentinel.spin_angle_bin,
        )

        mock_geometric_factor_lut = Mock(spec=GeometricFactorLookup)
        mock_efficiency_lut = Mock(spec=EfficiencyLookup)
        mock_energy_lookup = create_dataclass_mock(EnergyLookup)
        mock_energy_lookup.delta_minus = np.geomspace(280, 2, 128)
        mock_energy_lookup.delta_plus = np.geomspace(288, 1, 128)
        mock_energy_lookup.bin_centers = np.geomspace(14100, 88.082825, 128)
        mock_mass_bin_lookup = Mock()
        dependencies = CodiceLoL3a3dDistributionsDependencies(
            l3a_direct_event_data=mock_l3a_direct_event_data,
            mass_species_bin_lookup=mock_mass_bin_lookup,
            geometric_factors_lookup=mock_geometric_factor_lut,
            efficiency_factors_lut=mock_efficiency_lut,
            energy_per_charge_lut=mock_energy_lookup,
            species=sentinel.species
        )

        mock_mass_bin_lookup.get_species_index.return_value = sentinel.species_index

        expected_uncertainty = np.random.random(size=(77, 128, 24, 13))
        counts_for_species = expected_uncertainty ** 2

        counts_3d_distribution = mock_rebin.return_value
        counts_3d_distribution.__getitem__.return_value = counts_for_species

        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)
        mock_rebin_3d_distribution_azimuth_to_elevation.return_value = np.random.random(size=(77, 128, 24, 13))
        l3a_3d_distribution_data_product = processor.process_l3a_3d_distribution_product(dependencies)

        mock_elevation_angle_lookup_class.assert_called_once()
        counts_3d_distribution.__getitem__.assert_called_once_with(sentinel.species_index)
        mock_rebin.assert_called_once_with(
            direct_event_data=mock_l3a_direct_event_data,
            mass_species_bin_lookup=dependencies.mass_species_bin_lookup,
        )

        mock_combine_priorities_for_species_and_convert_to_rate.assert_called_once_with(counts_for_species,
                                                                                        sentinel.acquisition_time)

        mock_compute_geometric_factors = mock_geometric_factor_lut.get_geometric_factors
        mock_compute_geometric_factors.assert_called_once_with(
            sentinel.rgfo_half_spin,
            sentinel.rgfo_spin_sector,
            sentinel.rgfo_esa_step,
            NumpyArrayMatcher(l3a_de_half_spin_per_esa_step),
            date(2026, 5, 13)
        )
        mock_convert_count_rate_to_intensity.assert_called_once_with(
            mock_combine_priorities_for_species_and_convert_to_rate.return_value,
            mock_energy_lookup,
            mock_efficiency_lut,
            mock_compute_geometric_factors.return_value)

        mock_rebin_3d_distribution_azimuth_to_elevation.assert_called_once_with(
            mock_convert_count_rate_to_intensity.return_value,
            NumpyArrayMatcher(np.arange(1, 25)),
            mock_elevation_lookup, NumpyArrayMatcher(l3a_de_half_spin_per_esa_step),)

        self.assertIsInstance(l3a_3d_distribution_data_product, CodiceLoL3a3dDistributionDataProduct)
        self.assertEqual(processor.input_metadata, l3a_3d_distribution_data_product.input_metadata)
        self.assertEqual(mock_l3a_direct_event_data.epoch, l3a_3d_distribution_data_product.epoch)
        self.assertEqual(mock_l3a_direct_event_data.epoch_delta, l3a_3d_distribution_data_product.epoch_delta)
        self.assertEqual(mock_l3a_direct_event_data.rgfo_esa_step, l3a_3d_distribution_data_product.rgfo_esa_step)
        self.assertEqual(mock_l3a_direct_event_data.rgfo_spin_sector, l3a_3d_distribution_data_product.rgfo_spin_sector)
        self.assertEqual(mock_l3a_direct_event_data.rgfo_half_spin, l3a_3d_distribution_data_product.rgfo_half_spin)
        np.testing.assert_array_equal(l3a_3d_distribution_data_product.half_spin_per_esa_step,
                                      np.flip(mock_l3a_direct_event_data.half_spin_per_esa_step, axis=1))

        self.assertEqual(mock_elevation_lookup.bin_centers, l3a_3d_distribution_data_product.elevation)
        self.assertEqual(mock_elevation_lookup.bin_deltas, l3a_3d_distribution_data_product.elevation_delta)
        self.assertEqual(sentinel.spin_angle_bin, l3a_3d_distribution_data_product.spin_angle)
        self.assertEqual(sentinel.spin_angle_bin_delta, l3a_3d_distribution_data_product.spin_angle_delta)

        np.testing.assert_array_equal(np.flip(mock_energy_lookup.bin_centers), l3a_3d_distribution_data_product.energy)
        np.testing.assert_array_equal(np.flip(mock_energy_lookup.delta_plus),
                                      l3a_3d_distribution_data_product.energy_delta_plus)
        np.testing.assert_array_equal(np.flip(mock_energy_lookup.delta_minus),
                                      l3a_3d_distribution_data_product.energy_delta_minus)

        np.testing.assert_array_equal(np.flip(mock_rebin_3d_distribution_azimuth_to_elevation.return_value, axis=1),
                                      l3a_3d_distribution_data_product.species_data)
        np.testing.assert_array_equal(np.flip(expected_uncertainty, axis=1), l3a_3d_distribution_data_product.species_data_stat_uncert)
        self.assertEqual(sentinel.species, l3a_3d_distribution_data_product.species)

    def test_process_3d_distributions_save_for_each_species(self):

        for species in ["hplus", "heplus", "heplus2", "oplus6"]:
            with self.subTest(species=species):
                self._test_process_3d_distributions_save(species)

    @patch(f'{MODULE}.CodiceLoL3a3dDistributionsDependencies.fetch_dependencies')
    @patch(f'{MODULE}.CodiceLoProcessor.process_l3a_3d_distribution_product')
    @patch(f'{MODULE}.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def _test_process_3d_distributions_save(self, species, mock_spiceypy, mock_save_data,
                                            mock_process_l3a_3d_distribution_product,
                                            mock_fetch_dependencies):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor=f'lo-{species}-3d-distribution')
        mock_spiceypy.ktotal.return_value = 0

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies, species)
        mock_process_l3a_3d_distribution_product.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_3d_distribution_product.return_value)

        self.assertEqual(['parent_file_1', 'parent_file_2'],
                         mock_process_l3a_3d_distribution_product.return_value.parent_file_names)
        self.assertEqual([mock_save_data.return_value], product)

    def test_process_l3a_direct_events_all_fill_integration(self):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-3d-instrument-frame')

        dependencies = CodiceLoL3aDirectEventsDependencies.from_file_paths(
            sw_priority_rates_cdf=get_test_data_path(
                "codice/imap_codice_l1a_lo-sw-priority_20260307_v003-all-fill.cdf"),
            nsw_priority_rates_cdf=get_test_data_path(
                "codice/imap_codice_l1a_lo-nsw-priority_20260307_v003-all-fill.cdf"),
            direct_event_path=get_test_data_path("codice/imap_codice_l2_lo-direct-events_20260307_v003-all-fill.cdf"),
            mass_coefficients_file_path=get_test_data_path(
                "codice/imap_codice_mass-coefficient-lookup_20241110_v003.csv"),
            esa_to_energy_per_charge_file_path=get_test_data_path(
                "codice/imap_codice_lo-energy-per-charge_20241110_v002.csv"
            )
        )

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)

        try:
            processor.process_l3a_direct_event_data_product(dependencies)
        except Exception as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()
