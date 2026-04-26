from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, TypeVar
from unittest.mock import patch, Mock

import imap_data_access
import numpy as np
import spiceypy
import xarray as xr
from imap_data_access.processing_input import AncillaryInput, ProcessingInputCollection, ScienceInput, \
    RepointInput
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.constants import ONE_AU_IN_KM, TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.hi_spectral_fit_dependencies import HiSpectralIndexDependencies
from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.hit.l3.models import HitL1Data
from imap_l3_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import HitEventTypeLookup
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies
from imap_l3_processing.lo.lo_processor import perform_spectral_fit, LoProcessor
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularIntensityDataProduct, \
    RectangularIntensityMapData
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import \
    InstrumentResponseLookupTableCollection
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.swe_processor import SweProcessor
from imap_l3_processing.ultra.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.ultra.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies
from imap_l3_processing.ultra.ultra_processor import UltraProcessor
from imap_l3_processing.utils import save_data, read_l1d_mag_data, furnish_local_spice
from scripts.codice.create_fake_efficiency_ancillary import create_efficiency_lookup
from scripts.codice.create_more_accurate_l3a_direct_event import create_more_accurate_l3a_direct_events_cdf
from scripts.codice.create_more_accurate_l3a_direct_event_input import modify_l1a_priority_counts, \
    modify_l2_direct_events
from scripts.hi.create_hi_full_spin_deps import create_hi_full_spin_deps
from scripts.ultra.create_l1c_l2_and_glows_with_matching_date_range import create_l1c_and_glows_with_matching_date_range
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, try_get_many_run_local_paths, \
    get_imap_data_dir_path


def create_codice_lo_l3a_partial_densities_cdf():
    codice_lo_l2_data = CodiceLoL2SWSpeciesData.read_from_cdf(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l2_lo-sw-species_20241110_v002.cdf'))
    mpc_lookup = MassPerChargeLookup.read_from_file(
        get_test_data_path('codice/imap_codice_mass-per-charge_20241110_v002.csv'))
    deps = CodiceLoL3aPartialDensitiesDependencies(codice_l2_lo_data=codice_lo_l2_data,
                                                   mass_per_charge_lookup=mpc_lookup)

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-partial-densities'
    )

    codice_lo_processor = CodiceLoProcessor(ProcessingInputCollection(), input_metadata)
    partial_densities_data = codice_lo_processor.process_l3a_partial_densities(deps)
    cdf_path = save_data(partial_densities_data, delete_if_present=True)
    return cdf_path


def create_codice_lo_l3a_direct_events_cdf(l1a_paths: Optional[tuple[Path, Path]] = None) -> Path:
    codice_lo_l2_direct_event_cdf_path = modify_l2_direct_events(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l2_lo-direct-events_20241110_v002.cdf'))

    if l1a_paths is None:
        codice_lo_l1a_nsw_priority_path, codice_lo_l1a_sw_priority_path = modify_l1a_priority_counts(
            get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf'),
            get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-sw-priority_20241110_v002.cdf'))
    else:
        codice_lo_l1a_nsw_priority_path, codice_lo_l1a_sw_priority_path = l1a_paths

    energy_lookup_path = get_test_data_path('codice/imap_codice_lo-energy-per-charge_20241110_v001.csv')
    mass_coefficient_path = get_test_data_path('codice/imap_codice_mass-coefficient-lookup_20241110_v002.csv')

    deps = CodiceLoL3aDirectEventsDependencies.from_file_paths(
        sw_priority_rates_cdf=codice_lo_l1a_sw_priority_path,
        nsw_priority_rates_cdf=codice_lo_l1a_nsw_priority_path,
        direct_event_path=codice_lo_l2_direct_event_cdf_path,
        mass_coefficients_file_path=mass_coefficient_path,
        esa_to_energy_per_charge_file_path=energy_lookup_path, )

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-direct-events'
    )

    codice_lo_processor = CodiceLoProcessor(Mock(), input_metadata)
    direct_event_data = codice_lo_processor.process_l3a_direct_event_data_product(deps)
    return save_data(direct_event_data, delete_if_present=True)


def create_codice_lo_l3a_ratios_cdf():
    partial_densities_file = create_codice_lo_l3a_partial_densities_cdf()

    deps = CodiceLoL3aRatiosDependencies.from_file_paths(partial_densities_file)

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-sw-ratios'
    )

    codice_lo_processor = CodiceLoProcessor(Mock(), input_metadata)
    ratios_data = codice_lo_processor.process_l3a_ratios(deps)
    return save_data(ratios_data, delete_if_present=True)


def create_codice_lo_l3a_abundances_cdf():
    partial_densities_file = create_codice_lo_l3a_partial_densities_cdf()

    deps = CodiceLoL3aRatiosDependencies.from_file_paths(partial_densities_file)

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-sw-charge-state-distributions'
    )

    codice_lo_processor = CodiceLoProcessor(Mock(), input_metadata)
    ratios_data = codice_lo_processor.process_l3a_charge_state_distributions(deps)
    return save_data(ratios_data, delete_if_present=True)


def create_codice_lo_l3a_3d_distributions_cdf(species: str):
    l1a_paths = modify_l1a_priority_counts(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf'),
        get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-sw-priority_20241110_v002.cdf'))

    codice_lo_l3a_direct_event_path = Path(create_codice_lo_l3a_direct_events_cdf(l1a_paths))
    accurate_codice_lo_l3a_direct_event_path = create_more_accurate_l3a_direct_events_cdf(
        codice_lo_l3a_direct_event_path)

    codice_lo_l1a_nsw_priority_path, codice_lo_l1a_sw_priority_path = l1a_paths

    deps = CodiceLoL3a3dDistributionsDependencies.from_file_paths(
        l3a_file_path=accurate_codice_lo_l3a_direct_event_path,
        l1a_sw_file_path=codice_lo_l1a_sw_priority_path,
        l1a_nsw_file_path=codice_lo_l1a_nsw_priority_path,
        mass_species_bin_lut=(get_test_data_path('codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv')),
        geometric_factors_lut=(get_test_data_path('codice/imap_codice_lo-geometric-factors_20241110_v001.csv')),
        efficiency_factors_lut=(create_efficiency_lookup(species)),
        energy_per_charge_lut=get_test_data_path("codice/imap_codice_lo-energy-per-charge_20241110_v001.csv"),
        species=species
    )

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor=f'lo-{species}-3d-distribution'
    )

    codice_lo_processor = CodiceLoProcessor(Mock(), input_metadata)
    l3a_3d_distributions = codice_lo_processor.process_l3a_3d_distribution_product(deps)
    return save_data(l3a_3d_distributions, delete_if_present=True)


def create_swapi_l3b_cdf(geometric_calibration_file, efficiency_calibration_file, cdf_file):
    geometric_calibration = GeometricFactorCalibrationTable.from_file(geometric_calibration_file)
    efficiency_calibration = EfficiencyCalibrationTable(efficiency_calibration_file)
    swapi_cdf_data = CDF(cdf_file)
    swapi_data = read_l2_swapi_data(swapi_cdf_data)

    swapi_l3_dependencies = SwapiL3BDependencies(swapi_data, geometric_calibration, efficiency_calibration)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3b',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2010, 1, 2),
        version='v000')
    processor = SwapiProcessor(Mock(), input_metadata)

    l3b_combined_vdf = processor.process_l3b(swapi_data, swapi_l3_dependencies)
    cdf_path = save_data(l3b_combined_vdf, delete_if_present=True)
    return cdf_path


@patch("imap_l3_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
def create_swapi_l3a_cdf(alpha_temperature_density_calibration_file,
                         geometric_factor_calibration_file,
                         instrument_response_calibration_file, density_of_neutral_helium_calibration_file,
                         imap_swapi_efficiency_lut_file, cdf_file, mock_spice):
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

    alpha_temperature_density_calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(
        alpha_temperature_density_calibration_file)
    efficiency_calibration_table = EfficiencyCalibrationTable(imap_swapi_efficiency_lut_file)
    geometric_factor_calibration_table = GeometricFactorCalibrationTable.from_file(geometric_factor_calibration_file)
    instrument_response_calibration_table = InstrumentResponseLookupTableCollection.from_file(
        instrument_response_calibration_file)
    density_of_neutral_helium_calibration_table = DensityOfNeutralHeliumLookupTable.from_file(
        density_of_neutral_helium_calibration_file)
    swapi_cdf_data = CDF(cdf_file)
    swapi_data = read_l2_swapi_data(swapi_cdf_data)
    swapi_l3_dependencies = SwapiL3ADependencies(
        data=swapi_data,
        alpha_temperature_density_calibration_table=alpha_temperature_density_calibration_table,
        efficiency_calibration_table=efficiency_calibration_table,
        geometric_factor_calibration_table=geometric_factor_calibration_table,
        instrument_response_calibration_table=instrument_response_calibration_table,
        density_of_neutral_helium_calibration_table=density_of_neutral_helium_calibration_table,
        hydrogen_inflow_vector=Mock(),
        helium_inflow_vector=Mock(),
        swapi_response=Mock(),
    )

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3a',
        start_date=datetime(2025, 10, 23),
        end_date=datetime(2025, 10, 24),
        version='v000')
    processor = SwapiProcessor(Mock(), input_metadata)

    l3a_proton_sw = processor.process_l3a_proton(swapi_data, swapi_l3_dependencies)
    l3a_alpha_sw = processor.process_l3a_alpha_solar_wind(swapi_data, swapi_l3_dependencies)
    l3a_pui_he = processor.process_l3a_pui(swapi_data, swapi_l3_dependencies)
    proton_cdf_path = save_data(l3a_proton_sw, delete_if_present=True)
    alpha_cdf_path = save_data(l3a_alpha_sw, delete_if_present=True)
    pui_he_cdf_path = save_data(l3a_pui_he, delete_if_present=True)
    return proton_cdf_path, alpha_cdf_path, pui_he_cdf_path


def create_swe_product(dependencies: SweL3Dependencies) -> Path:
    input_metadata = InputMetadata(
        instrument='swe',
        data_level='l3',
        start_date=datetime(2025, 6, 29),
        end_date=datetime(2025, 7, 1),
        version='v000')
    processor = SweProcessor(Mock(), input_metadata)
    output_data = processor.calculate_products(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


@patch("imap_l3_processing.swe.l3.science.moment_calculations.spiceypy.pxform")
def create_swe_product_with_fake_spice(dependencies: SweL3Dependencies, mock_spice_pxform) -> Path:
    data = np.loadtxt(get_test_data_path("swe/ace_attitude.dat"), skiprows=1)

    def time_to_float(t):
        return (t - datetime(1999, 1, 1, tzinfo=timezone.utc)).total_seconds()

    times = [datetime(1998, 12, 31, tzinfo=timezone.utc) + timedelta(days=row[1], seconds=row[2]) for row in data]
    times_as_floats = [time_to_float(t) for t in times]

    def fake_pxform(from_frame, to_frame, et):
        assert from_frame == "IMAP_DPS"
        assert to_frame == "IMAP_RTN"
        time = spiceypy.et2datetime(et)
        correction_factor = (datetime(2025, 6, 30) - datetime(1999, 6, 8))
        time_in_1999 = time - correction_factor
        r = np.interp(time_to_float(time_in_1999), times_as_floats, data[:, 4])
        t = np.interp(time_to_float(time_in_1999), times_as_floats, data[:, 5])
        n = np.interp(time_to_float(time_in_1999), times_as_floats, data[:, 6])

        r2 = r * r
        t2 = t * t
        n2 = n * n

        c3 = np.sqrt(r2 + t2 + n2)
        c2 = np.sqrt(t2 + n2)
        c1 = np.sqrt((t2 + n2) * (t2 + n2) + r2 * t2 + r2 * n2)

        mat = np.array([[(-n2 - t2) / c1, 0, r / c3], [r * t / c1, -n / c2, t / c3], [r * n / c1, t / c2, n / c3]])
        return mat

    mock_spice_pxform.side_effect = fake_pxform

    input_metadata = InputMetadata(
        instrument='swe',
        data_level='l3',
        start_date=datetime(2025, 6, 29),
        end_date=datetime(2025, 7, 1),
        version='v000')
    processor = SweProcessor(Mock(), input_metadata)
    output_data = processor.calculate_products(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


def create_survival_corrected_full_spin_cdf(dependencies: HiL3SingleSensorFullSpinDependencies) -> Path:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime.now(),
                                   end_date=datetime.now() + timedelta(days=1),
                                   version="v000",
                                   descriptor="h90-ena-h-sf-sp-full-hae-4deg-1yr",
                                   )
    processor = HiProcessor(Mock(), input_metadata)
    output_data = processor.process_full_spin_single_sensor(dependencies, SpiceFrame.ECLIPJ2000)

    data_product = RectangularIntensityDataProduct(data=output_data, input_metadata=input_metadata)
    cdf_path = save_data(data_product, delete_if_present=True)
    return cdf_path


def create_hi_spectral_index_cdf(dependencies: HiSpectralIndexDependencies) -> Path:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime.now(),
                                   end_date=datetime.now() + timedelta(days=1),
                                   version="v000",
                                   descriptor="h90-spx-h-hf-sp-full-hae-4deg-6mo",
                                   )
    processor = HiProcessor(Mock(), input_metadata)
    output_data = processor.process_spectral_fit_index(dependencies)
    data_product = RectangularSpectralIndexDataProduct(data=output_data, input_metadata=input_metadata)
    cdf_path = save_data(data_product, delete_if_present=True)
    return cdf_path


def create_lo_spectral_index_cdf(dependencies: LoL3SpectralFitDependencies) -> Path:
    input_metadata = InputMetadata(instrument="lo",
                                   data_level="l3",
                                   start_date=datetime.now(),
                                   end_date=datetime.now() + timedelta(days=1),
                                   version="v000",
                                   descriptor="l090-spx-h-hf-sp-ram-hae-6deg-1yr",
                                   )

    output_data = perform_spectral_fit(dependencies.map_data)
    data_product = RectangularSpectralIndexDataProduct(data=output_data, input_metadata=input_metadata)
    cdf_path = save_data(data_product, delete_if_present=True)
    return cdf_path


def create_hit_sectored_cdf(dependencies: HITL3SectoredDependencies) -> Path:
    input_metadata = InputMetadata(
        instrument='hit',
        data_level='l3',
        descriptor='macropixel',
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
        version='v000')
    processor = HitProcessor(Mock(), input_metadata)
    output_data = processor.process_pitch_angle_product(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


def create_hit_direct_event_cdf():
    cosine_table = CosineCorrectionLookupTable(
        get_test_data_path("hit/pha_events/imap_hit_range-2A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-2B-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3B-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4B-cosine-lookup_20250203_v000.csv"),
    )
    gain_table = GainLookupTable.from_file(
        get_test_data_path("hit/pha_events/imap_hit_hi-gain-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_lo-gain-lookup_20250203_v000.csv"))

    range_fit_lookup = RangeFitLookup.from_files(
        get_test_data_path("hit/pha_events/imap_hit_range-2A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-2B-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3B-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4B-charge-fit-lookup_20250319_v000.csv"),
    )

    event_type_look = HitEventTypeLookup.from_csv(
        get_test_data_path("hit/pha_events/imap_hit_hit-event-type-lookup_20250228_v000.csv"))

    hit_l1_data = HitL1Data.read_from_cdf("data/imap/hit/l1a/2010/01/imap_hit_l1a_direct-events_20100105_v001.cdf")

    direct_event_dependencies = HitL3PhaDependencies(hit_l1_data=hit_l1_data, cosine_correction_lookup=cosine_table,

                                                     gain_lookup=gain_table, range_fit_lookup=range_fit_lookup,
                                                     event_type_lookup=event_type_look)
    input_metadata = InputMetadata(
        instrument="hit",
        data_level="l3",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=1),
        version="v001",
        descriptor="direct-events"
    )
    processor = HitProcessor(Mock(), input_metadata)

    product = processor.process_direct_event_product(direct_event_dependencies)
    file_path = save_data(product, delete_if_present=True)
    return file_path


@patch('imap_l3_processing.glows.glows_initializer.query')
@patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.download_dependency_from_path')
@patch('imap_l3_processing.glows.glows_processor.imap_data_access')
@patch('imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download')
@patch('imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download_external_dependency')
def run_l3b_initializer(mock_download_external, mock_download, mock_imap_data_access,
                        mock_download_dependency_from_path, mock_query):
    local_cdfs: list[str] = os.listdir(get_test_data_path("glows/pipeline/l3a"))

    l3a_dicts = [{'file_path': f"imap/glows/l3a/{file_path}",
                  'start_date': file_path.split('_')[4].split('-')[0],
                  'repointing': int(file_path.split('_')[4].split('-repoint')[1])
                  } for file_path in local_cdfs]

    mock_query.side_effect = [
        l3a_dicts, []
    ]

    mock_download.side_effect = [
        get_test_data_path('glows/imap_glows_pipeline-settings-l3bcde_20250423_v001.json'),
        get_test_data_path('glows/imap_2026_105_01.repoint.csv'),
    ]

    mock_download_external.side_effect = [
        TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt',
        TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc',
        Path(
            r'C:\Users\Harrison\Downloads\glows_L3bc_pipeline_2025_04_11\glows_L3bc_pipeline_2025_04_11\external_dependencies\omni2_all_years.dat')
    ]

    mock_download_dependency_from_path.side_effect = [
        get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
        get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
        get_test_data_path('glows/imap_glows_bad-days-list_20100101_v004.dat'),
        get_test_data_path('glows/imap_glows_pipeline-settings-l3bcde_20250423_v001.json'),
        get_test_data_path('glows/pipeline/l3a/imap_glows_l3a_hist_20250625-repoint01071_v001.cdf'),
        get_test_data_path('glows/pipeline/l3a/imap_glows_l3a_hist_20250626-repoint01072_v001.cdf'),
        get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
        get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
        get_test_data_path('glows/imap_glows_bad-days-list_20100101_v004.dat'),
        get_test_data_path('glows/imap_glows_pipeline-settings-l3bcde_20250423_v001.json'),
        get_test_data_path('glows/pipeline/l3a/imap_glows_l3a_hist_20100521-repoint00289_v001.cdf'),
        get_test_data_path('glows/pipeline/l3a/imap_glows_l3a_hist_20100522-repoint00290_v001.cdf'),
    ]

    bad_days_list = AncillaryInput('imap_glows_bad-days-list_20100101_v004.dat')
    waw_helio_ion = AncillaryInput('imap_glows_WawHelioIonMP_20100101_v002.json')
    uv_anisotropy = AncillaryInput('imap_glows_uv-anisotropy-1CR_20100101_v001.json')
    pipeline_settings = AncillaryInput('imap_glows_pipeline-settings-l3bcde_20250423_v001.json')

    repointings = RepointInput('imap_2026_105_01.repoint.csv')

    input_collection = ProcessingInputCollection(bad_days_list, waw_helio_ion, uv_anisotropy, pipeline_settings,
                                                 repointings)
    input_metadata = InputMetadata(instrument='glows', data_level='l3b',
                                   start_date=datetime.fromisoformat('2010-01-05T00:00:00'),
                                   end_date=datetime.fromisoformat('2025-01-06T00:00:00'),
                                   version='v001')

    processor = GlowsProcessor(input_collection, input_metadata)

    print(processor.process())


@patch('imap_l3_processing.glows.glows_initializer.query')
def run_glows_l3bc_processor_and_initializer(mock_query):
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3b',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v011')

    l3a_files = imap_data_access.query(instrument='glows', version=input_metadata.version, data_level='l3a',
                                       start_date='20100422', end_date='20100625')

    l3a_files_2 = imap_data_access.query(instrument='glows', version=input_metadata.version, data_level='l3a',
                                         start_date='20100922', end_date='20101123')
    mock_query.side_effect = [l3a_files + l3a_files_2, []]

    bad_days_list = AncillaryInput('imap_glows_bad-days-list_20100101_v001.dat')
    waw_helio_ion = AncillaryInput('imap_glows_WawHelioIonMP_20100101_v002.json')
    uv_anisotropy = AncillaryInput('imap_glows_uv-anisotropy-1CR_20300101_v003.json')
    pipeline_settings = AncillaryInput('imap_glows_pipeline-settings-l3bcde_20100101_v006.json')
    input_collection = ProcessingInputCollection(bad_days_list, waw_helio_ion, uv_anisotropy, pipeline_settings)

    processor = GlowsProcessor(dependencies=input_collection, input_metadata=input_metadata)
    print(processor.process())


@patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
@patch("imap_l3_processing.glows.glows_processor.Path")
@patch("imap_l3_processing.glows.glows_processor.run")
@patch("imap_l3_processing.glows.glows_processor.get_pointing_date_range")
def run_glows_l3e_lo_with_mocks(mock_get_repoint_date_range, _, mock_path, mock_l3e_dependencies_class):
    mock_processing_input_collection = Mock()
    mock_processing_input_collection.get_file_paths.return_value = [Path("one path")]

    mock_l3e_dependencies: GlowsL3EDependencies = GlowsL3EDependencies(
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridLo.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridHi.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridUltra.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/tessXYZ8.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/tessAng16.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/lyaSeriesV4_2021b.dat"),
        Path(
            "instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/solar_uv_anisotropy_NP.1.0_SP.1.0.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/speed3D.v01.Legendre.2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/density3D.v01.Legendre.2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/phion_Hydrogen_T12F107_2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/swEqtrElectrons5_2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/ionization.files.dat"),
        pipeline_settings={"executable_dependency_paths": {
            "energy-grid-lo": "EnGridLo.dat",
            "energy-grid-hi": "EnGridHi.dat",
            "energy-grid-ultra": "EnGridUltra.dat",
            "tess-xyz-8": "tessXYZ8.dat",
            "tess-ang-16": "tessAng16.dat",
            "lya-series": "lyaSeriesV4_2021b.dat",
            "solar-uv-anistropy": "solar_uv_anisotropy_NP.1.0_SP.1.0.dat",
            "speed-3d": "speed3D.v01.Legendre.2021b.dat",
            "density-3d": "density3D.v01.Legendre.2021b.dat",
            "phion-hydrogen": "phion_Hydrogen_T12F107_2021b.dat",
            "sw-eqtr-electrons": "swEqtrElectrons5_2021b.dat",
            "ionization-files": "ionization.files.dat",
        },
            "start_cr": 2092},
        elongation=defaultdict(lambda: 90),
        repointing_file=get_test_data_path("fake_1_day_repointing_file.csv"),
    )

    mock_l3e_dependencies.copy_dependencies = Mock()
    mock_l3e_dependencies_class.fetch_dependencies.return_value = (mock_l3e_dependencies, 2094)

    mock_path.side_effect = [
        # Path(get_test_instrument_team_data_path("glows/probSur.Imap.Lo_20090101_010101_2009.000_60.00.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Hi_2009.000_135.0.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Hi_2009.000_90.00.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Ul_20090101_010101_2009.000.txt")),
    ]

    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3e',
        start_date=datetime(2015, 4, 10),
        end_date=datetime(2015, 4, 11),
        version='v001',
        descriptor='survival-probability-lo')

    mock_get_repoint_date_range.return_value = (
        np.datetime64(datetime.fromisoformat("2025-04-20T00:00:00")),
        np.datetime64(datetime.fromisoformat("2025-04-21T00:00:00")))

    glows_processor: GlowsProcessor = GlowsProcessor(mock_processing_input_collection, input_metadata)

    glows_processor.process()


@patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy")
def run_glows_l3e_with_less_mocks(mock_spiceypy):
    mock_spiceypy.spkezr = spiceypy.spkezr
    mock_spiceypy.reclat = spiceypy.reclat
    mock_spiceypy.pxform = spiceypy.pxform
    mock_spiceypy.datetime2et = lambda date: spiceypy.datetime2et(date + timedelta(days=365 * 16 + 4, hours=2))

    l3d_file = "imap_glows_l3d_solar-hist_20100101-repoint02092_v002.cdf"
    lo_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_energy-grid-lo_20100101_v002.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v002.dat"),
        AncillaryInput("imap_glows_lya-series_19470303_v002.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_19470303_v002.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_19960130_v002.dat"),
        AncillaryInput("imap_glows_speed-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_19710416_v002.dat"),
        AncillaryInput("imap_glows_tess-xyz-8_20100101_v002.dat"),
        ScienceInput(l3d_file),
        RepointInput("imap_2001_052_001.repoint.csv"),
    )

    hi_45_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_energy-grid-hi_20100101_v002.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v002.dat"),
        AncillaryInput("imap_glows_lya-series_19470303_v002.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_19470303_v002.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_19960130_v002.dat"),
        AncillaryInput("imap_glows_speed-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_19710416_v002.dat"),
        ScienceInput(l3d_file),
        RepointInput("imap_2001_052_001.repoint.csv"),
    )
    hi_90_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_energy-grid-hi_20100101_v002.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v002.dat"),
        AncillaryInput("imap_glows_lya-series_19470303_v002.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_19470303_v002.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_19960130_v002.dat"),
        AncillaryInput("imap_glows_speed-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_19710416_v002.dat"),
        ScienceInput(l3d_file),
        RepointInput("imap_2001_052_001.repoint.csv"),
    )

    ul_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_energy-grid-ultra_20100101_v002.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v002.dat"),
        AncillaryInput("imap_glows_lya-series_19470303_v002.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_19470303_v002.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_19960130_v002.dat"),
        AncillaryInput("imap_glows_speed-3d_19640117_v002.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_19710416_v002.dat"),
        AncillaryInput("imap_glows_tess-ang-16_20100101_v002.dat"),
        ScienceInput(l3d_file),
        RepointInput("imap_2001_052_001.repoint.csv"),
    )

    version = 'v007'
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 2)

    lo_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                      end_date=end_date, version=version,
                                      descriptor='survival-probability-lo')

    glows_processor: GlowsProcessor = GlowsProcessor(lo_processing_input_collection, lo_input_metadata)
    glows_processor.process()

    hi_45_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                         end_date=end_date, version=version,
                                         descriptor='survival-probability-hi-45')

    glows_processor: GlowsProcessor = GlowsProcessor(hi_45_processing_input_collection, hi_45_input_metadata)
    glows_processor.process()

    hi_90_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                         end_date=end_date, version=version,
                                         descriptor='survival-probability-hi-90')

    glows_processor: GlowsProcessor = GlowsProcessor(hi_90_processing_input_collection, hi_90_input_metadata)
    glows_processor.process()

    ul_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                      end_date=end_date, version=version,
                                      descriptor='survival-probability-ul')

    glows_processor: GlowsProcessor = GlowsProcessor(ul_processing_input_collection, ul_input_metadata)
    glows_processor.process()


def make_glows_l3a_files():
    directory = get_test_data_path("glows/pipeline/l2")
    for file_path in directory.iterdir():
        run_glows_l3a(file_path)


def run_glows_l3a(file_path):
    date_in_path = re.search(r"imap_glows_l2_hist_(\d{8}).*.cdf", file_path.name).group(1)
    start_date = datetime.strptime(date_in_path, "%Y%m%d")
    end_date = start_date + timedelta(days=1)
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3a',
        start_date=start_date,
        end_date=end_date,
        version='v001'
    )

    with CDF(str(file_path)) as cdf_data:
        l2_glows_data = read_l2_glows_data(cdf_data)

        repoint_number = int(file_path.name.split('_')[4][-5:])
        input_metadata.repointing = repoint_number

        dependencies = GlowsL3ADependencies(l2_glows_data, {
            "calibration_data": get_test_instrument_team_data_path(
                "glows/imap_glows_calibration-data_20100101_v002.dat"),
            "settings": get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings_20100101_v001.json"),
            "time_dependent_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_time-dep-bckgrd_20100101_v001.dat"),
            "extra_heliospheric_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_map-of-extra-helio-bckgrd_20100101_v002.dat"),
        })

        processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)

        l3a_data = processor.process_l3a(dependencies)

        glows_l3a_temp_path = TEMP_CDF_FOLDER_PATH / "glows" / "l3a"
        print(save_data(l3a_data, folder_path=glows_l3a_temp_path, delete_if_present=True))


def run_glows_l3bc():
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3b',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v001')

    cr = 2091
    external_files = {
        'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
        'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat')
    }
    ancillary_files = {
        'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
        'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
        'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
        'pipeline_settings': get_test_instrument_team_data_path(
            'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
    }
    l3a_data_folder_path = get_test_data_path('glows/l3a_products')
    l3a_data = []
    l3a_file_names = [f"imap_glows_l3a_hist_201004{x}-repoint{x + 90:05d}_v012.cdf" for x in range(22, 31)]
    l3a_file_names += [f"imap_glows_l3a_hist_201005{x:02d}-repoint{x + 120:05d}_v012.cdf" for x in range(1, 19)]

    for name in l3a_file_names:
        l3a_data.append(create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / name))

    dependencies = GlowsL3BCDependencies(l3a_data=l3a_data, external_files=external_files,
                                         ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                         start_date=datetime(2009, 12, 20), end_date=datetime(2009, 12, 21),
                                         zip_file_path=Path("fake/path/to/file.zip"))

    processor = GlowsProcessor(Mock(), input_metadata)

    l3b_data_product, l3c_data_product = processor.process_l3bc(dependencies)

    l3b_cdf = save_data(l3b_data_product, delete_if_present=True)
    print(l3b_cdf)

    l3c_data_product.parent_file_names.append(Path(l3b_cdf).name)
    print(save_data(l3c_data_product, delete_if_present=True))


@patch('imap_l3_processing.glows.glows_processor.shutil')
def run_glows_l3d(mock_shutil):
    mock_shutil.move = shutil.copy

    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3d',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v001',
        descriptor='solar-hist'
    )

    external_files = {
        'lya_raw_data': get_test_data_path('glows/lyman_alpha_composite.nc'),
    }

    ancillary_files = {
        'WawHelioIon': {
            'speed': get_test_instrument_team_data_path('glows/imap_glows_plasma-speed-2010a_v003.dat'),
            'p-dens': get_test_instrument_team_data_path('glows/imap_glows_proton-density-2010a_v003.dat'),
            'uv-anis': get_test_instrument_team_data_path('glows/imap_glows_uv-anisotropy-2010a_v003.dat'),
            'phion': get_test_instrument_team_data_path('glows/imap_glows_photoion-2010a_v003.dat'),
            'lya': get_test_instrument_team_data_path('glows/imap_glows_lya-2010a_v003.dat'),
            'e-dens': get_test_instrument_team_data_path('glows/imap_glows_electron-density-2010a_v003.dat'),
        },
        'pipeline_settings': get_test_data_path(
            'glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json')
    }

    l3b_file_paths = [
        get_test_data_path('glows/pipeline/l3b/imap_glows_l3b_ion-rate-profile_20100103_v001.cdf'),
        get_test_data_path('glows/pipeline/l3b/imap_glows_l3b_ion-rate-profile_20100519_v001.cdf')
    ]

    l3c_file_paths = [
        get_test_data_path('glows/pipeline/l3c/imap_glows_l3c_sw-profile_20100103_v001.cdf'),
        get_test_data_path('glows/pipeline/l3c/imap_glows_l3c_sw-profile_20100519_v001.cdf'),
    ]

    l3d_dependencies: GlowsL3DDependencies = GlowsL3DDependencies(external_files=external_files,
                                                                  ancillary_files=ancillary_files,
                                                                  l3b_file_paths=l3b_file_paths,
                                                                  l3c_file_paths=l3c_file_paths)

    processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)
    data_product, l3d_txt_paths, last_processed_cr = processor.process_l3d(l3d_dependencies)
    print("l3d_txts:")
    [print(txt_path) for txt_path in l3d_txt_paths]
    print(save_data(data_product, cr_number=last_processed_cr))


def create_empty_hi_l1c_dataset(epoch: datetime, exposures: Optional[np.ndarray] = None,
                                spin_angles: Optional[np.ndarray] = None,
                                energies: Optional[np.ndarray] = None):
    energies = energies if energies is not None else np.geomspace(1, 10000, 9)
    spin_angles = spin_angles if spin_angles is not None else np.arange(0, 360, 0.1) + 0.05
    exposures = exposures if exposures is not None else np.ones(shape=(1, len(energies), len(spin_angles)))

    return xr.Dataset({
        "exposure_times": (
            [
                "epoch",
                "esa_energy_step",
                "hi_pset_spin_angle_bin"
            ],
            exposures
        ),
    },
        coords={
            "epoch": [epoch],
            "esa_energy_step": energies,
            "hi_pset_spin_angle_bin": spin_angles,
        }
    )


def create_empty_glows_l3e_dataset(epoch: datetime, survival_probabilities: np.ndarray,
                                   spin_angles: Optional[np.ndarray] = None,
                                   energies: Optional[np.ndarray] = None):
    energies = energies or np.geomspace(1, 10000, 16)
    spin_angles = spin_angles or np.arange(0, 360, 1) + 0.5

    return xr.Dataset({
        "probability_of_survival": (
            [
                "epoch",
                "energy",
                "spin_angle_bin"
            ],
            survival_probabilities
        )
    },
        coords={
            "epoch": [epoch],
            "energy": energies,
            "spin_angle_bin": spin_angles,
        })


EPOCH = TypeVar("EPOCH")
ENERGY = TypeVar("ENERGY")
LONGITUDE = TypeVar("LONGITUDE")
LATITUDE = TypeVar("LATITUDE")


def read_glows_survival_probability_data_from_cdf() -> tuple[np.ndarray, np.ndarray]:
    l3e = CDF(str(get_test_data_path("glows/imap_glows_l3e_survival-probabilities-hi_20250324_v001.cdf")))
    return l3e["probability_of_survival"][...][:, 0], l3e["probability_of_survival"][...][:, 1]


@patch('imap_l3_processing.hi.hi_processor.HiLoL3SurvivalDependencies.fetch_dependencies')
def create_hi_l3_survival_corrected_cdf(survival_dependencies: HiLoL3SurvivalDependencies, spacing_degree: int,
                                        mock_fetch_dependencies) -> Path:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime(2025, 4, 9),
                                   end_date=datetime(2025, 4, 10),
                                   version="v001",
                                   descriptor="h90-ena-h-sf-sp-ram-hae-4deg-6mo",
                                   )

    mock_fetch_dependencies.return_value = survival_dependencies

    processing_input_collection = Mock()
    processing_input_collection.get_file_paths.return_value = []
    processor = HiProcessor(processing_input_collection, input_metadata)
    print(processor.process())


@patch('imap_l3_processing.lo.lo_processor.HiLoL3SurvivalDependencies.fetch_dependencies')
def create_lo_l3_survival_corrected_cdf(survival_dependencies: HiLoL3SurvivalDependencies, spacing_degree: int,
                                        mock_fetch_dependencies):
    input_metadata = InputMetadata(instrument="lo",
                                   data_level="l3",
                                   start_date=datetime(2025, 4, 9),
                                   end_date=datetime(2025, 4, 10),
                                   version="v001",
                                   descriptor=f"l090-ena-h-sf-sp-ram-hae-{spacing_degree}deg-6mo",
                                   )

    mock_fetch_dependencies.return_value = survival_dependencies

    processing_input_collection = Mock()
    processing_input_collection.get_file_paths.return_value = []
    processor = LoProcessor(processing_input_collection, input_metadata)
    print(processor.process())

def create_codice_hi_l3a_direct_events_cdf():
    codice_hi_de_dependencies = CodiceHiL3aDirectEventsDependencies.from_file_paths(
        tof_lookup_path=get_test_instrument_team_data_path("codice/hi/imap_codice_tof-lookup_20241110_v002.csv"),
        codice_l2_hi_cdf=get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-direct-events_20241110_v002.cdf"))

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='hi-direct-events'
    )

    codice_lo_processor = CodiceHiProcessor(ProcessingInputCollection(), input_metadata)
    direct_event_data = codice_lo_processor.process_l3a_direct_event(codice_hi_de_dependencies)
    cdf_path = save_data(direct_event_data, delete_if_present=True)
    return cdf_path


def create_codice_hi_l3b_pitch_angles_cdf():
    codice_hi_pitch_angle_dependencies = CodicePitchAngleDependencies.from_file_paths(
        mag_file_path=get_test_data_path("mag/imap_mag_l1d_norm-mago_20250814_v003.cdf"),
        codice_l2_sectored_intensities_path=get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-sectored_20250814_v002.cdf")
    )

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3b',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='hi-pitch-angle'
    )

    codice_hi_processor = CodiceHiProcessor(ProcessingInputCollection(), input_metadata)
    pitch_angle_data = codice_hi_processor.process_l3b(codice_hi_pitch_angle_dependencies)
    cdf_path = save_data(pitch_angle_data, delete_if_present=True)
    return cdf_path


if __name__ == "__main__":
    furnish_local_spice()
    logging.basicConfig(force=True, level=logging.INFO)
    if "codice-lo" in sys.argv:
        if "l3a" in sys.argv:
            if "partial-densities" in sys.argv:
                print(create_codice_lo_l3a_partial_densities_cdf())
            elif "direct-events" in sys.argv:
                print(create_codice_lo_l3a_direct_events_cdf())
            elif "ratios" in sys.argv:
                print(create_codice_lo_l3a_ratios_cdf())
            elif "abundances" in sys.argv:
                print(create_codice_lo_l3a_abundances_cdf())
            elif "3d-instrument-frame" in sys.argv:
                print(create_codice_lo_l3a_3d_distributions_cdf(sys.argv[-1]))

    if "codice-hi" in sys.argv:
        if "l3a" in sys.argv:
            print(create_codice_hi_l3a_direct_events_cdf())
        if "l3b" in sys.argv:
            print(create_codice_hi_l3b_pitch_angles_cdf())

    if "swapi" in sys.argv:
        if "l3a" in sys.argv:
            paths = create_swapi_l3a_cdf(
                str(get_test_data_path("swapi/imap_swapi_alpha-density-temperature-lut_20240920_v000.dat")),
                str(get_test_data_path("swapi/imap_swapi_energy-gf-pui-lut_20100101_v001.csv")),
                str(get_test_data_path("swapi/imap_swapi_instrument-response-lut_20241023_v000.zip")),
                str(get_test_data_path(
                    "swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf")),
                str(get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")),
                str(get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")),
            )
            print(paths)
        if "l3b" in sys.argv:
            path = create_swapi_l3b_cdf(
                "tests/test_data/swapi/imap_swapi_energy-gf-sw-lut_20100101_v001.csv",
                "tests/test_data/swapi/imap_swapi_efficiency-lut_20241020_v000.dat",
                str(get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")))
            print(path)
    if "glows" in sys.argv:
        if "pre-b" in sys.argv:
            run_l3b_initializer()
        elif "l3bc" in sys.argv:
            run_glows_l3bc()
        elif "init-l3bc" in sys.argv:
            run_glows_l3bc_processor_and_initializer()
        elif "l3d" in sys.argv:
            run_glows_l3d()
        elif "l3e" in sys.argv:
            if "mock" in sys.argv:
                run_glows_l3e_lo_with_mocks()
            else:
                run_glows_l3e_with_less_mocks()
        elif "l3a" in sys.argv:
            make_glows_l3a_files()
        else:
            raise Exception("level not specified")

    if "hit" in sys.argv:
        if "direct_event" in sys.argv:
            path = create_hit_direct_event_cdf()
            print(f"hit direct event data product: {path}")
        else:
            mag_data = read_l1d_mag_data(
                r"C:\Users\Petty\Downloads\HIT Validation-20250922T131655Z-1-001\HIT Validation\imap_mag_l1d_norm-dsrf_20100106_v001.cdf")
            hit_data = read_l2_hit_data(
                r"C:\Users\Petty\Downloads\HIT Validation-20250922T131655Z-1-001\HIT Validation\imap_hit_l2_macropixel-intensity_20100106_v001.cdf")
            dependencies = HITL3SectoredDependencies(mag_l1d_data=mag_data, data=hit_data)
            print(f"hit macropixel data product: {create_hit_sectored_cdf(dependencies)}")

    if "swe" in sys.argv:
        dependencies = SweL3Dependencies.from_file_paths(
            get_test_data_path("swe/imap_swe_l2_sci_20250630_v002.cdf"),
            get_test_data_path("swe/imap_swe_l1b_sci_20250630_v003.cdf"),
            get_test_data_path("swe/imap_mag_l1d_norm-dsrf_20250630_v001.cdf"),
            get_test_data_path("swe/imap_swapi_l3a_proton-sw_20250630_v001.cdf"),
            get_test_data_path("swe/example_swe_config.json"),
        )
        print(create_swe_product(dependencies))

    if "swe-fake-spice" in sys.argv:
        dependencies = SweL3Dependencies.from_file_paths(
            get_test_data_path("swe/imap_swe_l2_sci_20260924_v001.cdf"),
            get_test_data_path("swe/imap_swe_l1b_sci_20260924_v001.cdf"),
            get_test_data_path("swe/imap_mag_l1d_norm-dsrf_20250630_v001.cdf"),
            get_test_data_path("swe/imap_swapi_l3a_proton-sw_20250630_v001.cdf"),
            get_test_data_path("swe/example_swe_config.json"),
        )
        print(create_swe_product_with_fake_spice(dependencies))

    if "hi" in sys.argv:
        hi_targets = ["survival-probability", "spectral-index", "full-spin", "combined-sensors"]
        do_all = not np.any([t in sys.argv for t in hi_targets])

        glows_l3e_folder = get_test_data_path("hi/fake_l3e_survival_probabilities/90")
        glows_l3_paths = list(glows_l3e_folder.iterdir())
        glows_l3_paths = glows_l3_paths[:len(glows_l3_paths) // 4]

        l2_antiram_90_map_path = get_test_data_path("hi/validation/l2_maps/imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-1yr_20250415_v001.cdf")
        l2_ram_90_map_path = get_test_data_path("hi/validation/l2_maps/imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf")
        l2_antiram_45_map_path = get_test_data_path("hi/validation/l2_maps/imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-1yr_20250415_v001.cdf")
        l2_ram_45_map_path = get_test_data_path("hi/validation/l2_maps/imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf")

        l1c_paths = list((get_imap_data_dir_path() / "hi" / "l1c").rglob("*.cdf"))

        if do_all or "survival-probability" in sys.argv:
            survival_dependencies = HiLoL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_ram_90_map_path,
                l1c_paths=l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-ram-hae-4deg-1yr")
            create_hi_l3_survival_corrected_cdf(survival_dependencies, 4)

        if do_all or "spectral-index" in sys.argv:
            dependencies = HiSpectralIndexDependencies.from_file_paths(
                get_test_data_path("hi/fake_l2_maps/hi45-zirnstein-mondel-6months.cdf")
            )
            print(create_hi_spectral_index_cdf(dependencies))

        if do_all or "full-spin" in sys.argv:
            ram_survival_dependencies = HiLoL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_ram_90_map_path,
                l1c_paths=l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-ram-hae-4deg-6mo")

            antiram_survival_dependencies = HiLoL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_antiram_90_map_path,
                l1c_paths=l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-anti-hae-4deg-6mo")

            full_spin_dependencies = HiL3SingleSensorFullSpinDependencies(
                ram_dependencies=ram_survival_dependencies,
                antiram_dependencies=antiram_survival_dependencies
            )

            print(create_survival_corrected_full_spin_cdf(full_spin_dependencies))

    if "lo" in sys.argv:
        lo_targets = ['spectral-index', 'survival_probabilities']
        do_all = not np.any([t in sys.argv for t in lo_targets])

        if do_all or "spectral-index" in sys.argv:
            dependencies = LoL3SpectralFitDependencies(
                RectangularIntensityMapData.read_from_path(Path(
                    '/data/imap/lo/l3/2025/04/imap_lo_l3_l090-ena-h-sf-sp-ram-hae-4deg-6mo_20250415_v002.cdf'))
            )

            print(create_lo_spectral_index_cdf(dependencies))

        if do_all or "survival_probabilities" in sys.argv:

            glows_l3e_folder = get_test_data_path("hi/fake_l3e_survival_probabilities/90")
            glows_l3_paths = list(glows_l3e_folder.iterdir())

            missing_paths, run_local_paths = try_get_many_run_local_paths([
                "hi/full_spin_deps/l1c",
                "hi/full_spin_deps/imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf",
            ])

            if missing_paths:
                create_hi_full_spin_deps(sensor="90")

            [hi_l1c_folder, l2_ram_90_map_path] = run_local_paths
            l1c_paths = list(hi_l1c_folder.iterdir())

            if do_all or "survival-probability" in sys.argv:
                survival_dependencies = HiLoL3SurvivalDependencies.from_file_paths(
                    map_file_path=l2_ram_90_map_path,
                    l1c_paths=l1c_paths,
                    glows_l3e_paths=glows_l3_paths,
                    l2_descriptor="l090-ena-h-sf-nsp-ram-hae-4deg-6mo")
                create_lo_l3_survival_corrected_cdf(survival_dependencies, 4)

    if "ultra" in sys.argv:
        if "survival" in sys.argv:
            spacing_degree = 4
            processor_input_metadata = InputMetadata(
                instrument="ultra",
                start_date=datetime(year=2025, month=9, day=1),
                end_date=datetime(year=2025, month=9, day=1),
                data_level="l3",
                version="v001",
                descriptor=f"u90-ena-h-sf-sp-full-hae-{spacing_degree}deg-6mo"
            )
            processor = UltraProcessor(input_metadata=processor_input_metadata, dependencies=Mock())

            # @formatter:off
            missing_map_and_pset_paths, [l2_map_path, *l1c_dependency_paths] = try_get_many_run_local_paths([
                "ultra/20250415-20250419/imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-6mo_20250415_v001.cdf",
                "ultra/20250415-20250419/ultra_l1c/imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v001.cdf",
                "ultra/20250415-20250419/ultra_l1c/imap_ultra_l1c_45sensor-spacecraftpset_20250416-repoint00002_v010.cdf",
                "ultra/20250415-20250419/ultra_l1c/imap_ultra_l1c_45sensor-spacecraftpset_20250417-repoint00003_v010.cdf",
                "ultra/20250415-20250419/ultra_l1c/imap_ultra_l1c_45sensor-spacecraftpset_20250418-repoint00004_v010.cdf",
            ])
            # @formatter:on

            missing_glows_paths, [*l3e_glows_paths] = try_get_many_run_local_paths([
                "ultra/20250415-20250419/glows_l3e/imap_glows_l3e_survival-probability-ultra_20250415_v010.cdf",
                "ultra/20250415-20250419/glows_l3e/imap_glows_l3e_survival-probability-ultra_20250416_v010.cdf",
                "ultra/20250415-20250419/glows_l3e/imap_glows_l3e_survival-probability-ultra_20250417_v010.cdf",
                "ultra/20250415-20250419/glows_l3e/imap_glows_l3e_survival-probability-ultra_20250418_v010.cdf",
            ])
            if missing_glows_paths or missing_map_and_pset_paths:
                create_l1c_and_glows_with_matching_date_range(datetime(2025, 4, 15, 12), datetime(2025, 4, 19, 12))

            l1c_dependency = [UltraL1CPSet.read_from_path(l1c_dependency_path) for l1c_dependency_path in
                              l1c_dependency_paths]
            l3e_dependencies = [UltraGlowsL3eData.read_from_path(path) for path in l3e_glows_paths]

            dependencies = UltraL3Dependencies.from_file_paths(
                l2_map_path,
                l1c_dependency_paths,
                l3e_glows_paths
            )

            healpix_sp_corrected_data = processor._process_survival_probability(deps=dependencies,
                                                                                spice_frame_name=SpiceFrame.IMAP_HAE)
            rectangular_sp_data_product = processor._process_healpix_intensity_to_rectangular(healpix_sp_corrected_data,
                                                                                              spacing_degree,
                                                                                              spice_frame_name=SpiceFrame.IMAP_HAE)

            rectangular_sp_corrected_path = save_data(rectangular_sp_data_product, delete_if_present=True)
            print(rectangular_sp_corrected_path)

        if "spectral-index" in sys.argv:
            spacing_degree = 2

            ultra_l3_path = get_test_data_path('ultra/fake_ultra_map_data.cdf')
            fit_energy_ranges_path = get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
            dependencies = UltraL3SpectralIndexDependencies.from_file_paths(ultra_l3_path, fit_energy_ranges_path)

            processor_input_metadata = InputMetadata(
                instrument="ultra",
                start_date=datetime(year=2025, month=9, day=1),
                end_date=datetime(year=2025, month=9, day=1),
                data_level="l3",
                version="v001",
                descriptor=f"u90-spx-h-sf-sp-full-hae-{spacing_degree}deg-6mo"
            )

            processor = UltraProcessor(input_metadata=processor_input_metadata, dependencies=None)

            spectral_index_map_data = processor._process_spectral_index(dependencies)
            output = processor._process_healpix_spectral_index_to_rectangular(spectral_index_map_data, spacing_degree)

            print(save_data(output, True))
