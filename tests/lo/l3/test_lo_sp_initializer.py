import unittest
from datetime import datetime
from unittest.mock import patch, call

from imap_data_access.file_validation import Version

from imap_l3_processing.lo.l3.lo_sp_initializer import LoSPInitializer, LO_SP_MAP_KERNELS
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce, MapInitializer
from imap_l3_processing.models import InputMetadata, VersionMap
from tests.integration.integration_test_helpers import ImapQueryPatcher


class TestLoSPInitializer(unittest.TestCase):
    def setUp(self):
        self.query_patcher = patch('imap_l3_processing.lo.l3.lo_sp_initializer.imap_data_access.query')
        self.mock_query = self.query_patcher.start()

    def tearDown(self):
        self.query_patcher.stop()

    def test_is_a_map_initializer(self):
        initializer = LoSPInitializer()
        self.assertIsInstance(initializer, MapInitializer)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced(self,  mock_read_cdf_parents):
        self.mock_query.side_effect = ImapQueryPatcher([
            'imap_glows_l3e_survival-probability-lo_20100101-repoint00001_v001.cdf',
            'imap_glows_l3e_survival-probability-lo_20100102-repoint00002_v001.cdf',
            'imap_glows_l3e_survival-probability-lo_20100103-repoint00003_v001.cdf',

            'imap_glows_l3e_survival-probability-lo_20100401-repoint00101_v002.cdf',
            'imap_glows_l3e_survival-probability-lo_20100402-repoint00102_v002.cdf',
            'imap_glows_l3e_survival-probability-lo_20100403-repoint00103_v002.cdf',

            'imap_glows_l3e_survival-probability-lo_20110410-repoint00201_v001.cdf',

            'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100101_v001.cdf',
            'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100401_v001.cdf',
            'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100401_v000.cdf',

            'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100701_v001.cdf',
            'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20101001_v001.cdf',

            'imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20100101_v003.0001.cdf',
            'imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20100401_v003.0001.cdf'
        ])

        mock_read_cdf_parents.side_effect = [
            [
                "imap_lo_l1c_pset_20100101-repoint00001_v001.cdf",
                "imap_lo_l1c_pset_20100102-repoint00002_v001.cdf",
                "imap_lo_l1c_pset_20100103-repoint00003_v001.cdf"
            ],
            [
                'imap_lo_l1c_pset_20100401-repoint00101_v001.cdf',
                'imap_lo_l1c_pset_20100402-repoint00102_v001.cdf',
                'imap_lo_l1c_pset_20100403-repoint00103_v001.cdf'
            ],
            [
                'imap_lo_l1c_pset_20100701-repoint00201_v001.cdf',
                'imap_lo_l1c_pset_20100702-repoint00202_v001.cdf',
                'imap_lo_l1c_pset_20100703-repoint00203_v001.cdf'
            ],
            [
                'imap_lo_l1c_pset_20101001-repoint00301_v001.cdf',
                'imap_lo_l1c_pset_20101002-repoint00302_v001.cdf',
                'imap_lo_l1c_pset_20101003-repoint00303_v001.cdf'
            ],
            [
                'imap_glows_l3e_survival-probability-lo_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-lo_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-lo_20100103-repoint00003_v001.cdf',
                "imap_lo_l1c_pset_20100101-repoint00001_v001.cdf",
                "imap_lo_l1c_pset_20100102-repoint00002_v001.cdf",
                "imap_lo_l1c_pset_20100103-repoint00003_v001.cdf",
                'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100101_v001.cdf'
            ],
            [
                'imap_glows_l3e_survival-probability-lo_20100401-repoint00101_v001.cdf',
                'imap_glows_l3e_survival-probability-lo_20100402-repoint00102_v001.cdf',
                'imap_glows_l3e_survival-probability-lo_20100403-repoint00103_v001.cdf',
                'imap_lo_l1c_pset_20100401-repoint00101_v001.cdf',
                'imap_lo_l1c_pset_20100402-repoint00102_v001.cdf',
                'imap_lo_l1c_pset_20100403-repoint00103_v001.cdf',
                'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100401_v001.cdf'
            ]
        ]

        initializer = LoSPInitializer()
        descriptor = "l090-ena-h-sf-sp-ram-hae-6deg-12mo"
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced(descriptor,
                                                                              3)

        self.mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-lo', version="latest"),
            call(instrument='lo', data_level='l2'),
            call(instrument='lo', data_level='l3'),
        ])

        mock_read_cdf_parents.assert_has_calls([
            call('imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100101_v001.cdf'),
            call('imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100401_v001.cdf'),
            call('imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100701_v001.cdf'),
            call('imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20101001_v001.cdf'),

            call('imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20100101_v003.0001.cdf'),
            call('imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20100401_v003.0001.cdf')
        ])

        expected_possible_map_to_produce = PossibleMapToProduce(
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=datetime(2010, 4, 1),
                end_date=datetime(2011, 4, 1, 6, 0),
                version=VersionMap({descriptor: Version(3, 2)}),
                descriptor=descriptor
            ),
            input_files={
                'imap_glows_l3e_survival-probability-lo_20100401-repoint00101_v002.cdf',
                'imap_glows_l3e_survival-probability-lo_20100402-repoint00102_v002.cdf',
                'imap_glows_l3e_survival-probability-lo_20100403-repoint00103_v002.cdf',
                'imap_lo_l1c_pset_20100401-repoint00101_v001.cdf',
                'imap_lo_l1c_pset_20100402-repoint00102_v001.cdf',
                'imap_lo_l1c_pset_20100403-repoint00103_v001.cdf',
                'imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20100401_v001.cdf'
            }
        )

        self.assertIn(expected_possible_map_to_produce, actual_maps_to_produce)
        self.assertEqual(2, len(actual_maps_to_produce))

    @patch('imap_l3_processing.lo.l3.lo_sp_initializer.furnish_spice_metakernel')
    def test_furnish_spice_dependencies(self, mock_furnish_metakernel):
        start_date = datetime(2025, 4, 15)
        end_date = datetime(2025, 7, 15)

        input_metadata = InputMetadata(
            instrument="hi",
            data_level="l2",
            start_date=start_date,
            end_date=end_date,
            version="v000",
            descriptor="h90-ena-h-sf-nsp-anti-hae-4deg-3mo",
        )
        map_to_produce = PossibleMapToProduce(set(), input_metadata)

        lo_initializer = LoSPInitializer()
        lo_initializer.furnish_spice_dependencies(map_to_produce)

        mock_furnish_metakernel.assert_called_once_with(start_date=start_date, end_date=end_date,
                                                        kernel_types=LO_SP_MAP_KERNELS)
