import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.hit.l3 import models
from imap_l3_processing.hit.l3.models import HitDirectEventDataProduct, HitL1Data
from imap_l3_processing.models import DataProductVariable, InputMetadata
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def setUp(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_direct_events_to_data_product_variables_with_multiple_events(self):
        input_metadata = InputMetadata(
            instrument="HIT",
            data_level="L3A",
            start_date=datetime.min,
            end_date=datetime.max,
            version="1",
            descriptor="direct-events"
        )

        epoch = [datetime.now(), datetime.now() + timedelta(hours=1)]
        charge = np.array([10.0, 12.0])
        energy = np.array([100, 200])
        e_delta = np.array([30, 50])
        e_prime = np.array([60, 80])
        detected_range = np.array([2, 3])
        particle = np.array([1, 2])
        priority_buffer_number = np.array([2, 3])
        latency = np.array([20, 30])
        stim_tag = np.array([False, True])
        long_event_flag = np.array([False, True])
        haz_tag = np.array([False, True])
        a_b_side = np.array([False, True])
        has_unread_adcs = np.array([True, False])
        culling_flag = np.array([True, False])
        pha_value = np.array([[11], [12]])
        energy_at_detector = np.array([[1, 2, 3], [9, 8, 7]])
        is_low_gain = np.array([[True], [False]])
        detector_flags = np.array([None, 2])
        deindex = np.array([None, 2])
        epindex = np.array([None, True])
        stim_gain = np.array([None, 2])
        a_l_stim = np.array([None, True])
        stim_step = np.array([None, 1])
        dac_value = np.array([None, 123])
        hit_flag_values = np.array([0,0])

        direct_event = HitDirectEventDataProduct(input_metadata=input_metadata,
                                                 epoch=epoch,
                                                 charge=charge,
                                                 energy=energy,
                                                 e_delta=e_delta,
                                                 e_prime=e_prime,
                                                 detected_range=detected_range,
                                                 particle_id=particle,
                                                 priority_buffer_number=priority_buffer_number,
                                                 latency=latency,
                                                 stim_tag=stim_tag,
                                                 long_event_flag=long_event_flag,
                                                 haz_tag=haz_tag,
                                                 a_b_side=a_b_side,
                                                 has_unread_adcs=has_unread_adcs,
                                                 culling_flag=culling_flag,
                                                 pha_value=pha_value,
                                                 energy_at_detector=energy_at_detector,
                                                 is_low_gain=is_low_gain,
                                                 detector_flags=detector_flags,
                                                 deindex=deindex,
                                                 epindex=epindex,
                                                 stim_gain=stim_gain,
                                                 a_l_stim=a_l_stim,
                                                 stim_step=stim_step,
                                                 dac_value=dac_value,
                                                 hit_flags=hit_flag_values,
        )

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, epoch),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, np.array([1, 1])),
            DataProductVariable(models.CHARGE_VAR_NAME, charge),
            DataProductVariable(models.ENERGY_VAR_NAME, energy),
            DataProductVariable(models.E_DELTA_VAR_NAME, e_delta),
            DataProductVariable(models.E_PRIME_VAR_NAME, e_prime),
            DataProductVariable(models.DETECTED_RANGE_VAR_NAME, detected_range),
            DataProductVariable(models.PARTICLE_ID_VAR_NAME, particle),
            DataProductVariable(models.PRIORITY_BUFFER_NUMBER_VAR_NAME, priority_buffer_number),
            DataProductVariable(models.LATENCY_VAR_NAME, latency),
            DataProductVariable(models.STIM_TAG_VAR_NAME, stim_tag),
            DataProductVariable(models.LONG_EVENT_FLAG_VAR_NAME, long_event_flag),
            DataProductVariable(models.HAZ_TAG_VAR_NAME, haz_tag),
            DataProductVariable(models.A_B_SIDE_VAR_NAME, a_b_side),
            DataProductVariable(models.HAS_UNREAD_FLAG_VAR_NAME, has_unread_adcs),
            DataProductVariable(models.CULLING_FLAG_VAR_NAME, culling_flag),
            DataProductVariable(models.PHA_VALUE_VAR_NAME, pha_value),
            DataProductVariable(models.ENERGY_IN_DETECTOR_VAR_NAME, energy_at_detector),
            DataProductVariable(models.IS_LOW_GAIN_VAR_NAME, is_low_gain),
            DataProductVariable(models.DETECTOR_FLAGS_VAR_NAME, detector_flags),
            DataProductVariable(models.DEINDEX_VAR_NAME, deindex),
            DataProductVariable(models.EPINDEX_VAR_NAME, epindex),
            DataProductVariable(models.STIM_GAIN_VAR_NAME, stim_gain),
            DataProductVariable(models.A_L_STIM_VAR_NAME, a_l_stim),
            DataProductVariable(models.STIM_STEP_VAR_NAME, stim_step),
            DataProductVariable(models.DAC_VALUE_VAR_NAME, dac_value),
            DataProductVariable(models.HIT_FLAGS_CDF_VAR_NAME, hit_flag_values),
            DataProductVariable(models.DETECTOR_ID_VAR_NAME, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                                              45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                                              59, 60, 61, 62, 63]),
        ]

        actual_variables = direct_event.to_data_product_variables()

        self.assertEqual(28, len(actual_variables))
        for expected_variable, actual_variable in zip(expected_variables, actual_variables):
            self.assert_variable_attributes(actual_variable, expected_variable.value, expected_variable.name,
                                            expected_variable.cdf_data_type, expected_variable.record_varying)

    def test_hit_l1_data_read_from_cdf(self):
        expected_epochs = np.array([datetime(year=2020, month=1, day=1), datetime(year=2020, month=1, day=1, hour=2)])
        expected_raw_binaries = ["10101011", "10101001010101"]

        pathname = 'test_cdf'
        with CDF(pathname, '') as cdf:
            cdf["epoch"] = expected_epochs
            cdf["pha_raw"] = expected_raw_binaries

        for file_path in [pathname, Path(pathname)]:
            with self.subTest(f"Can take {type(file_path)} as file path type"):
                hit_l1_data = HitL1Data.read_from_cdf(file_path)

                np.testing.assert_array_equal(hit_l1_data.epoch, expected_epochs)
                np.testing.assert_array_equal(expected_raw_binaries, hit_l1_data.event_binary)
