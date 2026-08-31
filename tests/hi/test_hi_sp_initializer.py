import unittest
from datetime import datetime
from typing import Callable
from unittest.mock import patch, call, Mock, sentinel

from imap_data_access.file_validation import Version

from imap_l3_processing.hi.hi_sp_initializer import HiSPInitializer, HI_SP_SPICE_KERNELS
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, map_descriptor_parts_to_string
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.maps.sp_map_initializer import logger
from imap_l3_processing.models import InputMetadata, VersionMap
from tests.test_helpers import create_mock_query_results


class TestHiSPInitializer(unittest.TestCase):
    def setUp(self):
        self.query_patcher = patch('imap_l3_processing.hi.hi_sp_initializer.imap_data_access.query')
        self.mock_query = self.query_patcher.start()

    def tearDown(self):
        self.query_patcher.stop()

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_can_be_produced_always_returns_possible_maps_with_default_version_map(self, mock_read_cdf_parents):
        descriptor = 'h45-ena-h-sf-sp-anti-hae-4deg-3mo'
        cases = [
            (None, VersionMap({}, Version(None, 1))),
             (4, VersionMap({}, Version(None, 1))),
        ]
        for major_version, expected_version in cases:
            with self.subTest(major_version=major_version):

                self.mock_query.side_effect = [
                    create_mock_query_results([
                        'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',

                        'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',

                        'imap_glows_l3e_survival-probability-hi-45_20100702-repoint00202_v001.cdf',
                    ]),
                    create_mock_query_results([]),
                    create_mock_query_results([
                        'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                        'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                        'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                        'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'
                    ]),
                    create_mock_query_results([])
                ]

                mock_read_cdf_parents.side_effect = self.create_fake_read_cdf_parents("45")

                expected_possible_maps = [
                    PossibleMapToProduce(
                        input_files={
                            'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100101-repoint00001_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100102-repoint00002_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100103-repoint00003_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument="hi",
                            data_level="l3",
                            start_date=datetime(2010, 1, 1),
                            end_date=datetime(2010, 4, 2, 7, 30),
                            version=expected_version,
                            descriptor=descriptor,
                        )
                    ),
                    PossibleMapToProduce(
                        input_files={
                            'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100401-repoint00101_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100402-repoint00102_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100403-repoint00103_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument="hi",
                            data_level="l3",
                            start_date=datetime(2010, 4, 1),
                            end_date=datetime(2010, 7, 1, 7, 30),
                            version=expected_version,
                            descriptor=descriptor,
                        )
                    ),
                    PossibleMapToProduce(
                        input_files={
                            'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-45_20100702-repoint00202_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100401-repoint00201_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100402-repoint00202_v001.cdf',
                            'imap_hi_l1c_45sensor-pset_20100403-repoint00203_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument="hi",
                            data_level="l3",
                            start_date=datetime(2010, 7, 1),
                            end_date=datetime(2010, 9, 30, 7, 30),
                            version=expected_version,
                            descriptor=descriptor,
                        )
                    ),
                ]

                initializer = HiSPInitializer()

                self.mock_query.assert_has_calls([
                    call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
                    call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
                    call(instrument='hi', data_level='l2'),
                    call(instrument='hi', data_level='l3'),
                ])

                actual_possible_maps = initializer.get_maps_that_can_be_produced('h45-ena-h-sf-sp-anti-hae-4deg-3mo')

                mock_read_cdf_parents.assert_has_calls([
                    call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
                    call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
                    call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
                    call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'),
                ], any_order=True)

                self.assertEqual(expected_possible_maps, actual_possible_maps)


    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_can_be_produced_full_spin_descriptor(self, mock_read_cdf_parents):
        self.mock_query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100104-repoint00004_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100703-repoint00203_v001.cdf',

            ]),
            create_mock_query_results([
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20090101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20090701_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf',
            ]),
            create_mock_query_results([])
        ]

        mock_read_cdf_parents.side_effect = [
            [
                'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
            ],
            [
                'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
            ]
        ]

        initializer = HiSPInitializer()
        descriptor = 'h90-ena-h-sf-sp-full-hae-4deg-6mo'
        actual_possible_maps = initializer.get_maps_that_can_be_produced(descriptor)

        self.mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2'),
            call(instrument='hi', data_level='l3'),
        ])

        mock_read_cdf_parents.assert_has_calls([
            call('imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf'),
            call('imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf'),
        ])

        expected_possible_map_to_produce = PossibleMapToProduce(
            input_files={
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf',
            },
            input_metadata=InputMetadata(
                instrument="hi",
                data_level="l3",
                start_date=datetime(2010, 1, 1),
                end_date=datetime(2010, 7, 2, 15),
                version=VersionMap({},Version(None,1)),
                descriptor=descriptor,
            )
        )

        self.assertEqual([expected_possible_map_to_produce], actual_possible_maps)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_can_be_produced_full_spin_with_mismatched_parents(self, mock_read_cdf_parents):
        self.mock_query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100104-repoint00004_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100703-repoint00203_v001.cdf',
            ]),
            create_mock_query_results([
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf',
            ]),
            create_mock_query_results([])
        ]

        mock_read_cdf_parents.side_effect = [
            [
                'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
            ],
            [
                'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
            ]
        ]

        initializer = HiSPInitializer()

        with self.assertLogs(logger=logger, level='WARNING') as log_context:
            actual_possible_maps = initializer.get_maps_that_can_be_produced('h90-ena-h-sf-sp-full-hae-4deg-6mo')
            expected_message = ('Expected all input maps to be created from the same pointing sets! l2_file_paths: '
                                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf, '
                                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf')
            self.assertIn(expected_message, log_context.output[0])


        self.mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2'),
            call(instrument='hi', data_level='l3'),
        ])

        mock_read_cdf_parents.assert_has_calls([
            call('imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20100101_v001.cdf'),
            call('imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20100101_v001.cdf'),
        ])

        self.assertEqual([], actual_possible_maps)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced(self, mock_read_cdf_parents):
        self.mock_query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',

                'imap_glows_l3e_survival-probability-hi-90_20100401-repoint00101_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100402-repoint00102_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100403-repoint00103_v002.cdf',

                'imap_glows_l3e_survival-probability-hi-90_20100702-repoint00201_v001.cdf',
            ]),
            create_mock_query_results([
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v000.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'

            ]),
            create_mock_query_results([
                'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v000.cdf',
                'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf',
            ]),
        ]

        mock_read_cdf_parents.side_effect = self.create_fake_read_cdf_parents("90")

        descriptor = 'h90-ena-h-sf-sp-anti-hae-4deg-3mo'
        expected_possible_maps = [
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-90_20100401-repoint00101_v002.cdf',
                    'imap_glows_l3e_survival-probability-hi-90_20100402-repoint00102_v002.cdf',
                    'imap_glows_l3e_survival-probability-hi-90_20100403-repoint00103_v002.cdf',
                    'imap_hi_l1c_90sensor-pset_20100401-repoint00101_v001.cdf',
                    'imap_hi_l1c_90sensor-pset_20100402-repoint00102_v001.cdf',
                    'imap_hi_l1c_90sensor-pset_20100403-repoint00103_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 4, 1),
                    end_date=datetime(2010, 7, 1, 7, 30),
                    version=VersionMap({descriptor:Version(None,2)}),
                    descriptor=descriptor,
                )
            ),
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-90_20100702-repoint00201_v001.cdf',
                    'imap_hi_l1c_90sensor-pset_20100401-repoint00201_v001.cdf',
                    'imap_hi_l1c_90sensor-pset_20100402-repoint00202_v001.cdf',
                    'imap_hi_l1c_90sensor-pset_20100403-repoint00203_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 7, 1),
                    end_date=datetime(2010, 9, 30, 7, 30),
                    version=VersionMap({descriptor:Version(None,1)}),
                    descriptor=descriptor,
                )
            ),
        ]

        initializer = HiSPInitializer()

        self.mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2'),
            call(instrument='hi', data_level='l3'),
        ])

        actual_possible_maps = initializer.get_maps_that_should_be_produced(descriptor, None)

        mock_read_cdf_parents.assert_has_calls([
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'),

            call(f'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf'),
        ])

        self.assertEqual(expected_possible_maps, actual_possible_maps)



    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_remakes_when_major_version_changes(self, mock_read_cdf_parents):
        for major_version in [2,3]:
            with self.subTest(major_version=major_version):
                self.mock_query.reset_mock()
                mock_read_cdf_parents.reset_mock()
                self.mock_query.side_effect = [
                    create_mock_query_results([]),
                    create_mock_query_results([
                        'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                        'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                    ]),
                    create_mock_query_results([
                        'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                        'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v000.cdf',
                    ]),
                    create_mock_query_results([
                        'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf',
                        'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v000.cdf',
                    ]),
                ]

                mock_read_cdf_parents.side_effect = self.create_fake_read_cdf_parents("90")

                descriptor = 'h90-ena-h-sf-sp-anti-hae-4deg-3mo'
                expected_possible_maps = [
                    PossibleMapToProduce(
                        input_files={
                            'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                            'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                            'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                            'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
                            'imap_hi_l1c_90sensor-pset_20100103-repoint00003_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument="hi",
                            data_level="l3",
                            start_date=datetime(2010, 1, 1),
                            end_date=datetime(2010, 4, 2, 7, 30),
                            version=VersionMap({descriptor:Version(major_version,2)}),
                            descriptor=descriptor,
                        )
                    ),
                ]

                initializer = HiSPInitializer()

                self.mock_query.assert_has_calls([
                    call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
                    call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
                    call(instrument='hi', data_level='l2'),
                    call(instrument='hi', data_level='l3'),
                ])

                actual_possible_maps = initializer.get_maps_that_should_be_produced(descriptor, major_version)

                mock_read_cdf_parents.assert_has_calls([
                    call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),

                    call(f'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf'),
                ])

                self.assertEqual(expected_possible_maps, actual_possible_maps)

    def test_get_dependencies(self):
        cases = [
            ("h90-ena-h-sf-sp-anti-hae-4deg-3mo", ["h90-ena-h-sf-nsp-anti-hae-4deg-3mo"]),
            ("h45-ena-h-sf-sp-ram-hae-4deg-3mo", ["h45-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
            ("h45-ena-h-sf-sp-full-hae-4deg-3mo",
             ["h45-ena-h-sf-nsp-anti-hae-4deg-3mo", "h45-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
            ("h90-ena-h-sf-sp-full-hae-4deg-3mo",
             ["h90-ena-h-sf-nsp-anti-hae-4deg-3mo", "h90-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
        ]

        for descriptor, dependencies in cases:
            with self.subTest(descriptor):
                initializer = HiSPInitializer()
                actual_dependencies = initializer._get_l2_dependencies(parse_map_descriptor(descriptor))
                self.assertEqual(dependencies, [map_descriptor_parts_to_string(d) for d in actual_dependencies])

    @patch('imap_l3_processing.hi.hi_sp_initializer.furnish_spice_metakernel')
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

        hi_initializer = HiSPInitializer()
        hi_initializer.furnish_spice_dependencies(map_to_produce)

        mock_furnish_metakernel.assert_called_once_with(start_date=start_date, end_date=end_date,
                                                        kernel_types=HI_SP_SPICE_KERNELS)

    @patch("imap_l3_processing.maps.map_initializer.generate_imap_input")
    @patch("imap_l3_processing.maps.map_initializer.ProcessingInputCollection")
    def test_possible_maps_to_produce_constructs_processing_input_collection(self, mock_collection,
                                                                             mock_generate_imap_input):
        path1 = "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v000.cdf"
        path2 = "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100102_v000.cdf"
        path3 = "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100103_v000.cdf"
        files = {
            path1,
            path2,
            path3,
        }

        def mock_generate_input(path):
            return {path1: sentinel.file1, path2: sentinel.file2, path3: sentinel.file3}[path]

        mock_generate_imap_input.side_effect = mock_generate_input

        processing_input = PossibleMapToProduce(input_files=files, input_metadata=Mock()).processing_input_collection

        mock_collection.assert_called_once_with(sentinel.file1, sentinel.file2, sentinel.file3)

        self.assertEqual(mock_collection.return_value, processing_input)

    def create_fake_read_cdf_parents(self, sensor: str) -> Callable[[str], set[str]]:
        def fake_read_cdf_parents(file_name: str) -> set[str]:
            mapping = {
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100101-repoint00001_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100102-repoint00002_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100103-repoint00003_v001.cdf',
                    f'imap_science_0001.tf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00102_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00103_v001.cdf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00201_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00202_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00203_v001.cdf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20101001-repoint00301_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20101002-repoint00302_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20101003-repoint00303_v001.cdf',
                },
                f'imap_hi_l3_h{sensor}-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf': {
                    f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100101-repoint00001_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100102-repoint00002_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100103-repoint00003_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100101-repoint00001_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100102-repoint00002_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100103-repoint00003_v001.cdf',
                    f'imap_science_0001.tf',
                },
                f'imap_hi_l3_h{sensor}-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf': {
                    f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00102_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00103_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100401-repoint00101_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100402-repoint00102_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100403-repoint00103_v001.cdf',
                    f'imap_science_0001.tf',
                },
            }
            return mapping[file_name]

        return fake_read_cdf_parents
