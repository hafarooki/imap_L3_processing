import tempfile
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, sentinel

import numpy as np
from imap_data_access.file_validation import Version
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.models import (
    CodiceLoL2SWSpeciesData,
    CodiceLoL3aPartialDensityDataProduct,
    CodiceLoL2DirectEventData,
    CodiceLoL3aDirectEventDataProduct,
    CodiceLoL1aSWPriorityRates,
    CodiceLoL1aNSWPriorityRates,
    CODICE_LO_L2_NUM_PRIORITIES,
    CodiceLoL3aRatiosDataProduct,
    CodiceLoPartialDensityData,
    CodiceLoL3ChargeStateDistributionsDataProduct,
    CodiceLoDirectEventData,
    EPOCH_VAR_NAME,
    EPOCH_DELTA_VAR_NAME,
    ELEVATION_VAR_NAME,
    ENERGY_VAR_NAME,
    ELEVATION_DELTA_VAR_NAME,
    CodiceLoL3a3dDistributionDataProduct,
    ENERGY_DELTA_PLUS_VAR_NAME,
    ENERGY_DELTA_MINUS_VAR_NAME,
    ELEVATION_ANGLE_LABEL_VAR_NAME,
    ENERGY_LABEL_VAR_NAME, SPIN_ANGLE_DELTA_VAR_NAME, SPIN_ANGLE_VAR_NAME, SPIN_ANGLE_LABEL_VAR_NAME,
    RGFO_ESA_STEP_VAR_NAME, RGFO_SPIN_SECTOR_VAR_NAME, RGFO_HALF_SPIN_VAR_NAME, HALF_SPIN_PER_ESA_STEP_VAR_NAME,
)
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.utils import save_data
from tests.swapi.cdf_model_test_case import CdfModelTestCase
from tests.test_helpers import get_test_data_path

LO_L2_SW_SPECIES_DATA_VARS = [
    'hplus',
    'heplusplus',
    'heplus',
    'ne',
    'cplus4',
    'cplus5',
    'cplus6',
    'oplus5',
    'oplus6',
    'oplus7',
    'oplus8',
    'cnoplus',
    'mg',
    'si',
    'fe_loq',
    'fe_hiq',
]


class TestModels(CdfModelTestCase):
    def test_lo_l2_sw_species_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta_minus = rng.random(len(epoch))
                epoch_delta_plus = rng.random(len(epoch))
                energy_per_charge = np.geomspace(2, 1000)
                spin_sector = np.linspace(0, 360, 24)
                hplus = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                heplus = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                heplusplus = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                ne = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                cplus4 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                cplus5 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                cplus6 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                oplus5 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                oplus6 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                oplus7 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                oplus8 = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                mg = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                si = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                cnoplus = rng.random((len(epoch), len(energy_per_charge)))
                fe_loq = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                fe_hiq = rng.random((len(epoch), len(energy_per_charge), len(spin_sector)))
                data_quality = rng.random(len(epoch))

                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta_minus'] = epoch_delta_minus
                cdf_file['epoch_delta_plus'] = epoch_delta_plus
                cdf_file['energy_per_charge'] = energy_per_charge
                cdf_file['spin_sector'] = spin_sector
                cdf_file['hplus'] = hplus
                cdf_file['heplusplus'] = heplusplus
                cdf_file['heplus'] = heplus
                cdf_file['ne'] = ne
                cdf_file['cplus4'] = cplus4
                cdf_file['cplus5'] = cplus5
                cdf_file['cplus6'] = cplus6
                cdf_file['oplus5'] = oplus5
                cdf_file['oplus6'] = oplus6
                cdf_file['oplus7'] = oplus7
                cdf_file['oplus8'] = oplus8
                cdf_file['cnoplus'] = cnoplus
                cdf_file['mg'] = mg
                cdf_file['si'] = si
                cdf_file['fe_loq'] = fe_loq
                cdf_file['fe_hiq'] = fe_hiq
                cdf_file['data_quality'] = data_quality

                for var in LO_L2_SW_SPECIES_DATA_VARS:
                    cdf_file[var].attrs["FILLVAL"] = -1e31

            result: CodiceLoL2SWSpeciesData = CodiceLoL2SWSpeciesData.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta_minus, epoch_delta_minus)
            np.testing.assert_array_equal(result.epoch_delta_plus, epoch_delta_plus)
            np.testing.assert_array_equal(result.energy_per_charge, energy_per_charge)
            np.testing.assert_array_equal(result.hplus, hplus)
            np.testing.assert_array_equal(result.heplusplus, heplusplus)
            np.testing.assert_array_equal(result.heplus, heplus)
            np.testing.assert_array_equal(result.ne, ne)
            np.testing.assert_array_equal(result.cplus4, cplus4)
            np.testing.assert_array_equal(result.cplus5, cplus5)
            np.testing.assert_array_equal(result.cplus6, cplus6)
            np.testing.assert_array_equal(result.oplus5, oplus5)
            np.testing.assert_array_equal(result.oplus6, oplus6)
            np.testing.assert_array_equal(result.oplus7, oplus7)
            np.testing.assert_array_equal(result.oplus8, oplus8)
            np.testing.assert_array_equal(result.cnoplus, cnoplus)
            np.testing.assert_array_equal(result.mg, mg)
            np.testing.assert_array_equal(result.si, si)
            np.testing.assert_array_equal(result.fe_loq, fe_loq)
            np.testing.assert_array_equal(result.fe_hiq, fe_hiq)
            np.testing.assert_array_equal(result.data_quality, data_quality)

    def test_lo_l2_sw_species_read_from_cdf_handles_fillval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_fillval_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta_minus = rng.random(len(epoch))
                epoch_delta_plus = rng.random(len(epoch))
                energy_per_charge = np.geomspace(2, 1000)
                spin_sector = np.linspace(0, 360, 24)
                fills = np.full((len(epoch), len(energy_per_charge), len(spin_sector)), -1e31)

                data_quality = rng.random(len(epoch))

                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta_minus'] = epoch_delta_minus
                cdf_file['epoch_delta_plus'] = epoch_delta_plus
                cdf_file['energy_per_charge'] = energy_per_charge
                cdf_file['spin_sector'] = spin_sector
                cdf_file['data_quality'] = data_quality

                for var in LO_L2_SW_SPECIES_DATA_VARS:
                    cdf_file[var] = fills
                    cdf_file[var].attrs["FILLVAL"] = -1e31

            nans = np.full((len(epoch), len(energy_per_charge), len(spin_sector)), np.nan)

            result: CodiceLoL2SWSpeciesData = CodiceLoL2SWSpeciesData.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta_minus, epoch_delta_minus)
            np.testing.assert_array_equal(result.epoch_delta_plus, epoch_delta_plus)
            np.testing.assert_array_equal(result.energy_per_charge, energy_per_charge)
            np.testing.assert_array_equal(result.hplus, nans)
            np.testing.assert_array_equal(result.heplusplus, nans)
            np.testing.assert_array_equal(result.heplus, nans)
            np.testing.assert_array_equal(result.ne, nans)
            np.testing.assert_array_equal(result.cplus4, nans)
            np.testing.assert_array_equal(result.cplus5, nans)
            np.testing.assert_array_equal(result.cplus6, nans)
            np.testing.assert_array_equal(result.oplus5, nans)
            np.testing.assert_array_equal(result.oplus6, nans)
            np.testing.assert_array_equal(result.oplus7, nans)
            np.testing.assert_array_equal(result.oplus8, nans)
            np.testing.assert_array_equal(result.cnoplus, nans)
            np.testing.assert_array_equal(result.mg, nans)
            np.testing.assert_array_equal(result.si, nans)
            np.testing.assert_array_equal(result.fe_loq, nans)
            np.testing.assert_array_equal(result.fe_hiq, nans)
            np.testing.assert_array_equal(result.data_quality, data_quality)

    def test_codice_lo_l3a_partial_density_to_data_product(self):
        epoch_data = np.array([datetime.now()])
        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([1]),
            "hplus_partial_density": np.array([15]),
            "heplusplus_partial_density": np.array([15]),
            "cplus4_partial_density": np.array([15]),
            "cplus5_partial_density": np.array([15]),
            "cplus6_partial_density": np.array([15]),
            "oplus5_partial_density": np.array([15]),
            "oplus6_partial_density": np.array([15]),
            "oplus7_partial_density": np.array([15]),
            "oplus8_partial_density": np.array([15]),
            "ne_partial_density": np.array([15]),
            "mg_partial_density": np.array([15]),
            "si_partial_density": np.array([15]),
            "fe_loq_partial_density": np.array([15]),
            "fe_hiq_partial_density": np.array([15]),
        }

        data_product = CodiceLoL3aPartialDensityDataProduct(
            Mock(), CodiceLoPartialDensityData(**input_data_product_kwargs)
        )

        actual_data_product_variables = data_product.to_data_product_variables()

        self.assertEqual(16, len(actual_data_product_variables))

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product.data, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_codice_lo_l3a_ratios_to_data_product(self):
        epoch_data = np.array([datetime.now()])

        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([1]),
            "c_to_o_ratio": np.array([2]),
            "mg_to_o_ratio": np.array([3]),
            "fe_to_o_ratio": np.array([4]),
            "c6_to_c5_ratio": np.array([5]),
            "c6_to_c4_ratio": np.array([6]),
            "o7_to_o6_ratio": np.array([7]),
            "felo_to_fehi_ratio": np.array([8]),
        }

        data_product = CodiceLoL3aRatiosDataProduct(
            Mock(), **input_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        self.assertEqual(9, len(actual_data_product_variables))

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_codice_lo_l3a_partial_density_read_from_cdf(self):
        rng = np.random.default_rng()
        epoch = [datetime.now(), datetime.now() + timedelta(days=1)]
        density_data = CodiceLoPartialDensityData(epoch=np.array(epoch), epoch_delta=np.array([2, 2]),
                                                  hplus_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  heplusplus_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus4_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus5_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus6_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus5_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus6_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus7_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus8_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  ne_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  mg_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  si_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  fe_loq_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  fe_hiq_partial_density=rng.random(len(epoch), dtype='float32'), )

        input_metadata = InputMetadata("codice", "l3a", datetime.now(), datetime.now(), VersionMap({"lo-partial-densities": Version(None, 1)}), "lo-partial-densities")

        codice_lo_l3a_partial_densities = CodiceLoL3aPartialDensityDataProduct(input_metadata, density_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_path = save_data(codice_lo_l3a_partial_densities, folder_path=Path(tmpdir))

            read_in_density_data = CodiceLoPartialDensityData.read_from_cdf(cdf_path)

        for field in fields(CodiceLoPartialDensityData):
            np.testing.assert_array_equal(getattr(read_in_density_data, field.name),
                                          getattr(density_data, field.name),
                                          strict=True)

    def test_codice_lo_l3a_charge_state_distributions_to_data_product(self):
        rng = np.random.default_rng()
        epoch_data = np.array([datetime.now()])
        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([1]),
            "oxygen_charge_state_distribution": rng.random((len(epoch_data), 4)),
            "carbon_charge_state_distribution": rng.random((len(epoch_data), 3)),
        }

        data_product = CodiceLoL3ChargeStateDistributionsDataProduct(Mock(), **input_data_product_kwargs)
        actual_data_product_variables = data_product.to_data_product_variables()
        self.assertEqual(6, len(actual_data_product_variables))
        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables[:4]):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

        oxygen_charge_states, carbon_charge_states, = actual_data_product_variables[4:]
        np.testing.assert_array_equal(oxygen_charge_states.value, [5, 6, 7, 8])
        self.assertEqual("oxygen_charge_state", oxygen_charge_states.name)

        np.testing.assert_array_equal(carbon_charge_states.value, [4, 5, 6])
        self.assertEqual("carbon_charge_state", carbon_charge_states.name)

    def test_codice_lo_l3a_direct_event_read_from_cdf_reads_where_no_variables_have_fill_value(self):
        no_fill_l3a_cdf_path = get_test_data_path("codice/imap_codice_l3a_lo-direct-events_20260504_v003.cdf")

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_l3a_direct_event = tmpdir / "test_l3a_cdf.cdf"
            with CDF(str(test_l3a_direct_event), masterpath=str(no_fill_l3a_cdf_path)) as cdf:
                cdf["apd_energy"] = rng.random(cdf["apd_energy"].shape)
                cdf["elevation"] = rng.random(cdf["elevation"].shape)
                cdf["energy_per_charge"] = rng.random(cdf["energy_per_charge"].shape)
                cdf["energy_step"] = rng.random(cdf["energy_step"].shape)
                cdf["mass"] = rng.random(cdf["mass"].shape)
                cdf["mass_per_charge"] = rng.random(cdf["mass_per_charge"].shape)
                cdf["normalization"] = rng.random(cdf["normalization"].shape)
                cdf["spin_angle"] = rng.random(cdf["spin_angle"].shape)
                cdf["spin_angle_bin"] = rng.random(cdf["spin_angle_bin"].shape)
                cdf["spin_angle_bin_delta"] = rng.random(cdf["spin_angle_bin_delta"].shape)
                cdf["type"] = rng.integers(0, 5, cdf["type"].shape)
                cdf["normalization_per_event"] = rng.random(cdf["normalization_per_event"].shape)
                cdf["position"] = rng.random(cdf["position"].shape)
                cdf["spin_sector"] = rng.random(cdf["spin_sector"].shape)
                cdf["acquisition_time_per_esa_step"] = rng.random(cdf["acquisition_time_per_esa_step"].shape)

                cdf["rgfo_spin_sector"] = rng.integers(0, 24, cdf["rgfo_spin_sector"].shape)
                cdf["rgfo_half_spin"] = rng.integers(0, 32, cdf["rgfo_half_spin"].shape)
                cdf["rgfo_esa_step"] = rng.integers(0, 128, cdf["rgfo_esa_step"].shape)
                cdf["half_spin_per_esa_step"] = rng.integers(0, 32, cdf["half_spin_per_esa_step"].shape)

            actual_event_data = CodiceLoDirectEventData.read_from_cdf(test_l3a_direct_event)

            with CDF(str(test_l3a_direct_event)) as cdf:
                np.testing.assert_array_equal(actual_event_data.normalization_per_event, cdf["normalization_per_event"])
                np.testing.assert_array_equal(actual_event_data.position, cdf["position"])
                np.testing.assert_array_equal(actual_event_data.energy_step, cdf["energy_step"])
                np.testing.assert_array_equal(actual_event_data.spin_sector, cdf["spin_sector"])
                np.testing.assert_array_equal(actual_event_data.acquisition_time_per_esa_step,
                                              cdf["acquisition_time_per_esa_step"])
                np.testing.assert_array_equal(actual_event_data.apd_energy, cdf["apd_energy"])
                np.testing.assert_array_equal(actual_event_data.elevation, cdf["elevation"])
                np.testing.assert_array_equal(actual_event_data.energy_per_charge, cdf["energy_per_charge"])
                np.testing.assert_array_equal(actual_event_data.energy_step, cdf["energy_step"])
                np.testing.assert_array_equal(actual_event_data.mass, cdf["mass"])
                np.testing.assert_array_equal(actual_event_data.mass_per_charge, cdf["mass_per_charge"])
                np.testing.assert_array_equal(actual_event_data.normalization, cdf["normalization"])
                np.testing.assert_array_equal(actual_event_data.spin_angle, cdf["spin_angle"])
                np.testing.assert_array_equal(actual_event_data.spin_angle_bin, cdf["spin_angle_bin"])
                np.testing.assert_array_equal(actual_event_data.spin_angle_bin_delta, cdf["spin_angle_bin_delta"])
                np.testing.assert_array_equal(actual_event_data.type, cdf["type"])
                np.testing.assert_array_equal(actual_event_data.rgfo_spin_sector, cdf["rgfo_spin_sector"])
                np.testing.assert_array_equal(actual_event_data.rgfo_half_spin, cdf["rgfo_half_spin"])
                np.testing.assert_array_equal(actual_event_data.rgfo_esa_step, cdf["rgfo_esa_step"])
                np.testing.assert_array_equal(actual_event_data.half_spin_per_esa_step, cdf["half_spin_per_esa_step"])

    def test_codice_lo_l3a_direct_event_read_from_cdf_handles_fill_value(self):
        all_fill_l3a_cdf_path = get_test_data_path("codice/imap_codice_l3a_lo-direct-events_20260504_v003-all-fill.cdf")
        actual_event_data = CodiceLoDirectEventData.read_from_cdf(all_fill_l3a_cdf_path)

        with CDF(str(all_fill_l3a_cdf_path)) as cdf:
            # @formatter:off
            np.testing.assert_array_equal(actual_event_data.apd_energy, np.full_like(cdf["apd_energy"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.data_quality, cdf["data_quality"])
            np.testing.assert_array_equal(actual_event_data.elevation, np.full_like(cdf["elevation"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.energy_per_charge, np.full_like(cdf["energy_per_charge"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.mass, np.full_like(cdf["mass"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.mass_per_charge, np.full_like(cdf["mass_per_charge"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.normalization, np.full_like(cdf["normalization"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.normalization_per_event, np.full_like(cdf["normalization_per_event"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.acquisition_time_per_esa_step, np.full_like(cdf["acquisition_time_per_esa_step"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.spin_angle, np.full_like(cdf["spin_angle"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.spin_angle_bin_delta, np.full_like(cdf["spin_angle_bin_delta"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.rgfo_spin_sector, np.full_like(cdf["rgfo_spin_sector"][...], np.nan))
            # @formatter:on

            masked_variables = [
                "energy_step",
                "apd_id",
                "gain",
                "multi_flag",
                "num_events",
                "position",
                "spin_sector",
                "tof",
                "type",
                "half_spin_per_esa_step",
                "rgfo_half_spin",
                "rgfo_spin_sector",
                "rgfo_esa_step",
            ]

            for var in masked_variables:
                actual_data = getattr(actual_event_data, var)

                self.assertIsInstance(actual_data, np.ma.masked_array)
                np.testing.assert_array_equal(actual_data.data, cdf[var])
                self.assertTrue(np.all(actual_data.mask))

    def test_codice_lo_l3a_direct_event_to_data_product(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2022, 3, 5), datetime(2022, 3, 6)])
        epoch_delta = np.full(len(epoch), 1)
        event_num = np.array([1, 2, 3, 4])
        spin_angle = np.array([30, 60, 90])
        energy_step = np.array([5.5, 6.6, 7.7])
        priority = np.arange(CODICE_LO_L2_NUM_PRIORITIES)
        esa_step = np.arange(128)

        direct_event = CodiceLoL3aDirectEventDataProduct(
            input_metadata=Mock(),
            acquisition_time_per_esa_step=rng.random((len(epoch), len(energy_step))),
            epoch=epoch,
            epoch_delta=epoch_delta,
            normalization=rng.random((len(epoch), len(priority), len(energy_step), len(spin_angle))),
            mass_per_charge=rng.random((len(epoch), len(priority), len(event_num))),
            mass=rng.random((len(epoch), len(priority), len(event_num))),
            apd_energy=rng.random((len(epoch), len(priority), len(event_num))),
            energy_step=rng.random((len(epoch), len(priority), len(event_num))),
            gain=rng.random((len(epoch), len(priority), len(event_num))),
            apd_id=rng.random((len(epoch), len(priority), len(event_num))),
            multi_flag=rng.random((len(epoch), len(priority), len(event_num))),
            num_events=rng.random((len(epoch), len(priority), len(event_num))),
            data_quality=rng.random((len(epoch), len(priority))),
            tof=rng.random((len(epoch), len(priority), len(event_num))),
            spin_angle=rng.random((len(epoch), len(priority), len(event_num))),
            spin_sector=rng.random((len(epoch), len(priority), len(event_num))),
            elevation=rng.random((len(epoch), len(priority), len(event_num))),
            position=rng.random((len(epoch), len(priority), len(event_num))),
            spin_angle_bin=rng.random(24),
            spin_angle_bin_delta=rng.random(24),
            energy_bin=rng.random(len(energy_step)),
            energy_bin_delta_plus=rng.random(len(energy_step)),
            energy_bin_delta_minus=rng.random(len(energy_step)),
            half_spin_per_esa_step=rng.random((len(epoch), len(energy_step))),
            rgfo_half_spin=rng.random(len(epoch)),
            rgfo_spin_sector=rng.random(len(epoch)),
            rgfo_esa_step=rng.random(len(epoch)),
            nso_half_spin=rng.random(len(epoch)),
            nso_spin_sector=rng.random(len(epoch)),
            nso_esa_step=rng.random(len(epoch)),
            normalization_per_event=rng.random((len(epoch), len(priority), len(energy_step), len(spin_angle))),
            esa_step=esa_step,
            energy_per_charge=rng.random((len(epoch), len(priority), len(event_num))),
            type=rng.integers(0, 5, (len(epoch), len(priority), len(event_num))),
        )

        np.testing.assert_array_equal(direct_event.event_index, np.arange(len(event_num)))
        np.testing.assert_array_equal(direct_event.priority_index_label, np.array(["0", "1", "2", "3", "4", "5", "6"]))
        np.testing.assert_array_equal(direct_event.event_index_label, np.array([str(i) for i in range(len(event_num))]))
        np.testing.assert_array_equal(direct_event.energy_bin_label,
                                      np.array([str(e) for e in direct_event.energy_bin]))
        np.testing.assert_array_equal(direct_event.spin_angle_bin_label,
                                      np.array([str(spin_angle) for spin_angle in direct_event.spin_angle_bin]))

        data_products = direct_event.to_data_product_variables()

        non_parent_fields = [f for f in fields(CodiceLoL3aDirectEventDataProduct)
                             if f.name in CodiceLoL3aDirectEventDataProduct.__annotations__ or
                             f.name in CodiceLoDirectEventData.__annotations__]

        self.assertEqual(len(non_parent_fields), len(data_products))
        for data_product in data_products:
            np.testing.assert_array_equal(getattr(direct_event, data_product.name), data_product.value)

    def test_codice_lo_l1a_sw_priority_read_from_cdf(self):
        cdf_path = get_test_data_path("codice/imap_codice_l1a_lo-sw-priority_20260307_v004.cdf")
        actual_l1a_sw_priority_rates = CodiceLoL1aSWPriorityRates.read_from_cdf(cdf_path)

        with CDF(str(cdf_path)) as cdf:
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch, cdf["epoch"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch_delta_plus, cdf["epoch_delta_plus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch_delta_minus, cdf["epoch_delta_minus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.acquisition_time_per_esa_step,
                                          cdf["acquisition_time_per_esa_step"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.spin_sector_index, cdf["spin_sector"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.rgfo_half_spin, cdf["rgfo_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.nso_half_spin, cdf["nso_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.sw_bias_gain_mode, cdf["sw_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.st_bias_gain_mode, cdf["st_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.data_quality, cdf["data_quality"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.spin_period, cdf["spin_period"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p0_tcrs, cdf["p0_tcrs"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p1_hplus, cdf["p1_hplus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p2_heplusplus, cdf["p2_heplusplus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p3_heavies, cdf["p3_heavies"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p4_dcrs, cdf["p4_dcrs"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.half_spin_per_esa_step,
                                          cdf["half_spin_per_esa_step"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.rgfo_spin_sector, cdf["rgfo_spin_sector"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.rgfo_esa_step, cdf["rgfo_esa_step"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.nso_spin_sector, cdf["nso_spin_sector"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.nso_esa_step, cdf["nso_esa_step"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.esa_step, cdf["esa_step"][...])

    def test_codice_lo_l1a_nsw_priority_read_from_cdf(self):
        cdf_path = get_test_data_path("codice/imap_codice_l1a_lo-nsw-priority_20260307_v004.cdf")
        actual_l1a_nsw_priority_rates = CodiceLoL1aNSWPriorityRates.read_from_cdf(cdf_path)

        with CDF(str(cdf_path)) as cdf:
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch, cdf["epoch"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch_delta_plus, cdf["epoch_delta_plus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch_delta_minus,
                                          cdf["epoch_delta_minus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.acquisition_time_per_esa_step,
                                          cdf["acquisition_time_per_esa_step"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.spin_sector_index,
                                          cdf["spin_sector"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.rgfo_half_spin, cdf["rgfo_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.data_quality, cdf["data_quality"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.p5_heavies, cdf["p5_heavies"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.p6_hplus_heplusplus,
                                          cdf["p6_hplus_heplusplus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.nso_half_spin, cdf["nso_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.sw_bias_gain_mode,
                                          cdf["sw_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.st_bias_gain_mode,
                                          cdf["st_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.spin_period, cdf["spin_period"][...])

    def test_codice_lo_3d_distributions_data_product(self):
        species = "hplus"
        species_uncertainty = "hplus_stat_uncert"

        elevation = np.array([10, 20, 30])
        spin_angle = np.array([40, 50, 60])
        energy = np.array([70, 80, 90])

        data_product = CodiceLoL3a3dDistributionDataProduct(
            input_metadata=Mock(),
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            elevation=elevation,
            elevation_delta=sentinel.elevation_delta,
            spin_angle=spin_angle,
            spin_angle_delta=sentinel.spin_angle_delta,
            energy=energy,
            energy_delta_plus=sentinel.energy_delta_plus,
            energy_delta_minus=sentinel.energy_delta_minus,
            species=species,
            species_data=sentinel.species_data,
            species_data_stat_uncert=sentinel.uncertainties,
            rgfo_esa_step=sentinel.rgfo_esa_step,
            rgfo_spin_sector=sentinel.rgfo_spin_sector,
            rgfo_half_spin=sentinel.rgfo_half_spin,
            half_spin_per_esa_step=sentinel.half_spin_per_esa_step,
        )

        actual_data_product_variables = data_product.to_data_product_variables()
        self.assertEqual(18, len(actual_data_product_variables))
        actual_variables = iter(actual_data_product_variables)

        self.assert_variable_attributes(next(actual_variables), sentinel.epoch, EPOCH_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.epoch_delta, EPOCH_DELTA_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), elevation, ELEVATION_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.elevation_delta, ELEVATION_DELTA_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), spin_angle, SPIN_ANGLE_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.spin_angle_delta, SPIN_ANGLE_DELTA_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), energy, ENERGY_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.energy_delta_plus, ENERGY_DELTA_PLUS_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.energy_delta_minus,
                                        ENERGY_DELTA_MINUS_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.species_data,
                                        species)
        self.assert_variable_attributes(next(actual_variables), energy.astype(str), ENERGY_LABEL_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), spin_angle.astype(str), SPIN_ANGLE_LABEL_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), elevation.astype(str), ELEVATION_ANGLE_LABEL_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.uncertainties, species_uncertainty)
        self.assert_variable_attributes(next(actual_variables), sentinel.rgfo_esa_step, RGFO_ESA_STEP_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.rgfo_spin_sector, RGFO_SPIN_SECTOR_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.rgfo_half_spin, RGFO_HALF_SPIN_VAR_NAME)
        self.assert_variable_attributes(next(actual_variables), sentinel.half_spin_per_esa_step, HALF_SPIN_PER_ESA_STEP_VAR_NAME)

    def test_codice_lo_l2_direct_events_reads_from_correct_float_data(self):
        all_fill_l2_cdf_path = get_test_data_path('codice/imap_codice_l2_lo-direct-events_20260307_v003-all-fill.cdf')

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_cdf_path = tmpdir / "test_cdf.cdf"
            with CDF(str(test_cdf_path), masterpath=str(all_fill_l2_cdf_path)) as cdf:
                cdf["apd_energy"] = rng.random(cdf["apd_energy"].shape)
                cdf["energy_step"] = rng.random(cdf["energy_step"].shape)
                cdf["elevation_angle"] = rng.random(cdf["elevation_angle"].shape)
                cdf["energy_per_charge"] = rng.random(cdf["energy_per_charge"].shape)
                cdf["position"] = rng.random(cdf["position"].shape)
                cdf["spin_angle"] = rng.random(cdf["spin_angle"].shape)
                cdf["spin_sector"] = rng.integers(0, 24, cdf["spin_sector"].shape)
                cdf["type"] = rng.integers(0, 5, cdf["type"].shape)

            l2_direct_event = CodiceLoL2DirectEventData.read_from_cdf(test_cdf_path)

            with CDF(str(test_cdf_path)) as cdf:
                np.testing.assert_array_equal(l2_direct_event.apd_energy, cdf["apd_energy"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.elevation_angle, cdf["elevation_angle"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.energy_step, cdf["energy_step"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.energy_per_charge, cdf["energy_per_charge"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.spin_sector, cdf["spin_sector"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.spin_angle, cdf["spin_angle"][:, :7, ...])
                np.testing.assert_array_equal(l2_direct_event.type, cdf["type"][:, :7, ...])

    def test_workaround_to_convert_float_spin_sector_to_int(self):
        all_fill_l2_cdf_path = get_test_data_path('codice/imap_codice_l2_lo-direct-events_20260307_v003-all-fill.cdf')

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_cdf_path = tmpdir / "test_cdf.cdf"
            with CDF(str(test_cdf_path), masterpath=str(all_fill_l2_cdf_path)) as cdf:
                expected_spin_sector = rng.integers(0, 24, cdf["spin_sector"].shape, dtype=int)
                cdf["spin_sector"] = expected_spin_sector.astype(float)

            l2_direct_event = CodiceLoL2DirectEventData.read_from_cdf(test_cdf_path)

            np.testing.assert_array_equal(l2_direct_event.spin_sector, expected_spin_sector[:, :7, ...], strict=True)

    def test_codice_lo_l2_direct_events_read_from_cdf_handles_fill_value(self):
        all_fill_l2_cdf_path = get_test_data_path('codice/imap_codice_l2_lo-direct-events_20260307_v003-all-fill.cdf')
        l2_direct_event = CodiceLoL2DirectEventData.read_from_cdf(all_fill_l2_cdf_path)

        with CDF(str(all_fill_l2_cdf_path)) as cdf:
            np.testing.assert_array_equal(cdf["epoch"], l2_direct_event.epoch)
            np.testing.assert_array_equal(cdf["epoch_delta_plus"], l2_direct_event.epoch_delta_plus)
            np.testing.assert_array_equal(cdf["epoch_delta_minus"], l2_direct_event.epoch_delta_minus)

            np.testing.assert_array_equal(l2_direct_event.apd_energy, np.full((2, 7, 10000), np.nan))
            np.testing.assert_array_equal(l2_direct_event.energy_step, np.full((2, 7, 10000), np.nan))
            np.testing.assert_array_equal(l2_direct_event.energy_per_charge, np.full((2, 7, 10000), np.nan))
            np.testing.assert_array_equal(l2_direct_event.spin_angle, np.full((2, 7, 10000), np.nan))
            np.testing.assert_array_equal(l2_direct_event.spin_sector, np.full((2, 7, 10000), np.nan))
            np.testing.assert_array_equal(l2_direct_event.elevation_angle, np.full((2, 7, 10000), np.nan))

            self.assertIsInstance(l2_direct_event.position, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.position.data, cdf["position"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.position.mask))

            self.assertIsInstance(l2_direct_event.gain, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.gain.data, cdf["gain"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.gain.mask))

            self.assertIsInstance(l2_direct_event.apd_id, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.apd_id.data, cdf["apd_id"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.apd_id.mask))

            self.assertIsInstance(l2_direct_event.data_quality, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.data_quality.data, cdf["data_quality"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.data_quality.mask))

            self.assertIsInstance(l2_direct_event.multi_flag, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.multi_flag.data, cdf["multi_flag"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.multi_flag.mask))

            self.assertIsInstance(l2_direct_event.num_events, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.num_events.data, cdf["num_events"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.num_events.mask))

            self.assertIsInstance(l2_direct_event.tof, np.ma.masked_array)
            np.testing.assert_array_equal(l2_direct_event.tof.data, cdf["tof"][:, :7, ...])
            self.assertTrue(np.all(l2_direct_event.tof.mask))

    def test_codice_lo_l2_direct_events_read_from_cdf_masks_negative_tof_values(self):
        all_fill_l2_cdf_path = get_test_data_path('codice/imap_codice_l2_lo-direct-events_20260307_v003-all-fill.cdf')

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_cdf_path = tmpdir / "test_cdf.cdf"
            with CDF(str(test_cdf_path), masterpath=str(all_fill_l2_cdf_path)) as cdf:
                tof = rng.uniform(-100, 100, cdf["tof"].shape).astype(np.float32)
                tof[0, 0, 0] = 0.0
                cdf["tof"] = tof

            l2_direct_event = CodiceLoL2DirectEventData.read_from_cdf(test_cdf_path)

            expected_truncated_tof = tof[:, :CODICE_LO_L2_NUM_PRIORITIES, ...]
            expected_mask = expected_truncated_tof < 0

            np.testing.assert_array_equal(l2_direct_event.tof.data, expected_truncated_tof)
            np.testing.assert_array_equal(l2_direct_event.tof.mask, expected_mask)

    def test_codice_lo_l1a_sw_priority_read_from_cdf_handles_fill_value(self):
        l1a_sw_all_fill_path = get_test_data_path("codice/imap_codice_l1a_lo-sw-priority_20260307_v003-all-fill.cdf")
        l1a_sw = CodiceLoL1aSWPriorityRates.read_from_cdf(l1a_sw_all_fill_path)

        with CDF(str(l1a_sw_all_fill_path)) as cdf:
            np.testing.assert_array_equal(l1a_sw.spin_period, np.full_like(cdf['spin_period'], np.nan))

            integer_variable_names = [
                "nso_half_spin",
                "rgfo_half_spin",
                "data_quality",
                "p0_tcrs",
                "p1_hplus",
                "p2_heplusplus",
                "p3_heavies",
                "p4_dcrs",
                "st_bias_gain_mode",
                "sw_bias_gain_mode",
                "half_spin_per_esa_step",
                "rgfo_spin_sector",
                "rgfo_esa_step",
                "nso_spin_sector",
                "nso_esa_step",
            ]

            for var in integer_variable_names:
                var_data = getattr(l1a_sw, var)
                self.assertIsInstance(var_data, np.ma.masked_array)
                np.testing.assert_equal(var_data.mask, np.full_like(cdf[var], True))

    def test_codice_lo_l1a_nsw_priority_read_from_cdf_handles_fill_value(self):
        l1a_nsw_all_fill_path = get_test_data_path("codice/imap_codice_l1a_lo-nsw-priority_20260307_v003-all-fill.cdf")

        l1a_nsw_model = CodiceLoL1aNSWPriorityRates.read_from_cdf(l1a_nsw_all_fill_path)

        with CDF(str(l1a_nsw_all_fill_path)) as cdf:
            np.testing.assert_array_equal(l1a_nsw_model.spin_period, np.full_like(cdf['spin_period'], np.nan))

            self.assertIsInstance(l1a_nsw_model.rgfo_half_spin, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.rgfo_half_spin.data, cdf["rgfo_half_spin"])
            self.assertTrue(np.all(l1a_nsw_model.rgfo_half_spin.mask))

            self.assertIsInstance(l1a_nsw_model.data_quality, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.data_quality.data, cdf["data_quality"])
            self.assertTrue(np.all(l1a_nsw_model.data_quality.mask))

            self.assertIsInstance(l1a_nsw_model.p5_heavies, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.p5_heavies.data, cdf["p5_heavies"])
            self.assertTrue(np.all(l1a_nsw_model.p5_heavies.mask))

            self.assertIsInstance(l1a_nsw_model.p6_hplus_heplusplus, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.p6_hplus_heplusplus.data, cdf["p6_hplus_heplusplus"])
            self.assertTrue(np.all(l1a_nsw_model.p6_hplus_heplusplus.mask))

            self.assertIsInstance(l1a_nsw_model.nso_half_spin, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.nso_half_spin.data, cdf["nso_half_spin"])
            self.assertTrue(np.all(l1a_nsw_model.nso_half_spin.mask))

            self.assertIsInstance(l1a_nsw_model.sw_bias_gain_mode, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.sw_bias_gain_mode.data, cdf["sw_bias_gain_mode"])
            self.assertTrue(np.all(l1a_nsw_model.sw_bias_gain_mode.mask))

            self.assertIsInstance(l1a_nsw_model.st_bias_gain_mode, np.ma.masked_array)
            np.testing.assert_array_equal(l1a_nsw_model.st_bias_gain_mode.data, cdf["st_bias_gain_mode"])
            self.assertTrue(np.all(l1a_nsw_model.st_bias_gain_mode.mask))
