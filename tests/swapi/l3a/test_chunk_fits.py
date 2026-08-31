import unittest
from datetime import datetime
from unittest.mock import Mock, patch, create_autospec
import platform

import numpy as np
from imap_processing.spice.time import str_yyyymmdd_to_ttj2000ns
from spiceypy import KernelPool
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    FIVE_MINUTES_IN_NANOSECONDS,
    ONE_SECOND_IN_NANOSECONDS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.models import InputMetadata, MagData
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import rotate_rtn_velocity_to_swapi_per_bin, \
    calculate_pui_energy_cutoff
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swapi.l3a import chunk_fits
from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitter,
    ChunkFitter,
    ParallelChunkRunner,
    ProtonChunkFitter,
    PuiChunkFitter,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    PickupIonFitResult,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters, build_vasyliunas_siscoe_distribution,
)
from imap_l3_processing.swapi.response.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.utils import (
    get_swapi_geometry,
    get_spacecraft_velocity_rtn,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.constants import (
    SWAPI_BIN_PERIOD_S,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_FINE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_LIVETIME_CENTER_OFFSET_S,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.predicted_ephemeris_tracker import PredictedEphemerisTracker
from tests.spice_test_case import SpiceTestCase
from tests.swapi._helpers import REALISTIC_ESA_VOLTAGES
from tests.test_helpers import get_test_instrument_team_data_path, get_integration_test_spice_data_path

_N_SWEEPS = 5
_N_BINS = 72
_FULL_ENERGY = np.concatenate([[1.0e4], REALISTIC_ESA_VOLTAGES]) * SWAPI_L2_K_FACTOR

_TRUE_DENSITY = 5.0
_TRUE_TEMPERATURE_K = 1.0e5
_TRUE_BULK_SPEED = 450.0
# Sunward Parker spiral, off-nominal: 55° from -R toward +T (vs. nominal
# 45° from +R toward -T), tilted 10° out of the ecliptic toward +N.
_B_HAT_RTN = np.array([
    -np.cos(np.radians(55.0)) * np.cos(np.radians(10.0)),
    np.sin(np.radians(55.0)) * np.cos(np.radians(10.0)),
    np.sin(np.radians(10.0)),
])
_TRUE_ALPHA_DENSITY = 0.2
_TRUE_ALPHA_TEMPERATURE_K = 4.0e5
_TRUE_DELTA_V_KM_S = 30.0
_SC_VELOCITY_RTN = np.array([0.0, 30.0, 0.0])
_EPOCH_TT2000 = 800_000_000_000_000_000
_CHUNK_EPOCH = _EPOCH_TT2000 + THIRTY_SECONDS_IN_NANOSECONDS
_SCI_START_TIME = _EPOCH_TT2000 + np.arange(_N_SWEEPS, dtype=np.int64) * 12_000_000_000

# Every non-flag, non-epoch field must NaN-fill on short-circuit branches.
_PROTON_SCALAR_KEYS = [
    "proton_sw_speed", "proton_sw_speed_uncert",
    "proton_sw_speed_sun", "proton_sw_speed_sun_uncert",
    "proton_sw_temperature", "proton_sw_temperature_uncert",
    "proton_sw_density", "proton_sw_density_uncert",
]
_PROTON_ARRAY_KEYS = [
    "proton_sw_velocity_rtn_sun",
    "proton_sw_velocity_rtn",
    "proton_sw_velocity_rtn_covariance",
]
_ALPHA_SCALAR_KEYS = [
    "alpha_sw_speed", "alpha_sw_speed_uncert",
    "alpha_sw_speed_sun", "alpha_sw_speed_sun_uncert",
    "alpha_sw_density", "alpha_sw_density_uncert",
    "alpha_sw_temperature", "alpha_sw_temperature_uncert",
]
_ALPHA_ARRAY_KEYS = [
    "alpha_sw_velocity_rtn_sun",
    "alpha_sw_velocity_rtn",
    "alpha_sw_velocity_rtn_covariance",
]
_ALPHA_BUMP_BINS = slice(24, 31)


def _swapi_response_with_warm_cache(voltages):
    resp = SwapiResponse.from_files(
        get_test_instrument_team_data_path("swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"),
        get_test_instrument_team_data_path("swapi/imap_swapi_central-effective-area_20260425_v001.csv"),
        get_test_instrument_team_data_path("swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"),
    )
    resp.warm_cache(voltages)
    return resp


def _spice_rotations(bin_slice):
    """SPICE-derived SWAPI→RTN rotations at the synthetic chunk's measurement
    times over `bin_slice`."""
    bin_indices = np.arange(bin_slice.start, bin_slice.stop)
    seconds_into_sweep = bin_indices * SWAPI_BIN_PERIOD_S + SWAPI_LIVETIME_CENTER_OFFSET_S
    measurement_times = (
        _SCI_START_TIME[:, np.newaxis]
        + seconds_into_sweep * ONE_SECOND_IN_NANOSECONDS
    ).flatten()
    return get_swapi_geometry(measurement_times)


def _truth_velocity_rtn(rotations):
    """Truth wind vector: `_TRUE_BULK_SPEED` km/s anti-parallel-ish to the SWAPI
    boresight (column 1 of the SWAPI→RTN rotation at the chunk's first bin),
    tilted 5° toward a stable in-plane direction. The deflection lifts the
    wind off the spin axis so clock-angle assertions are non-degenerate."""
    spin_axis_rtn = rotations[0, :, 1]
    perpendicular = np.cross(spin_axis_rtn, [1.0, 0.0, 0.0])
    perpendicular /= np.linalg.norm(perpendicular)
    deflection = np.radians(5.0)
    direction = -spin_axis_rtn * np.cos(deflection) + perpendicular * np.sin(deflection)
    return _TRUE_BULK_SPEED * direction


def _efficiency_table():
    """Synthetic `EfficiencyCalibrationTable` with realistic in-flight proton
    (0.12) and alpha (0.15) efficiencies and a lab-cal proton efficiency of 0.12."""
    table = EfficiencyCalibrationTable.__new__(EfficiencyCalibrationTable)
    table.data = np.array(
        [
            (np.datetime64("2024-01-01", "ns"), 0, 0.12, 0.15),
            (np.datetime64("2025-11-01", "ns"), 0, 0.12, 0.15),
        ],
        dtype=[
            ("time", "M8[ns]"),
            ("MET", "i8"),
            ("proton efficiency", "f8"),
            ("alpha efficiency", "f8"),
        ],
    )
    return table


def _populate_shared(response, table):
    chunk_fits._shared.update(swapi_response=response, efficiency_table=table)


def _clear_shared():
    chunk_fits._shared.clear()


def _synthesize_chunk(*, response, rotations, proton_velocity_rtn, alpha_velocity_rtn, efficiency_table):
    """Forward-model a 5-sweep proton + alpha chunk at the truth params over
    the full 71-bin science axis. Per-species effective-area scales come from
    `efficiency_table` so synthesis and the fitter share the same calibration."""
    n = SWAPI_SCIENCE_BINS.stop - SWAPI_SCIENCE_BINS.start
    voltages = np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS)

    proton_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltages)),
        esa_voltage=voltages,
        swapi_response=response,
        central_effective_area_scale=efficiency_table.central_effective_area_scale_for(_CHUNK_EPOCH, "proton"),
        rotation_matrices=rotations,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltages)),
        esa_voltage=voltages,
        swapi_response=response,
        central_effective_area_scale=efficiency_table.central_effective_area_scale_for(_CHUNK_EPOCH, "helium"),
        rotation_matrices=rotations,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )
    proton_truth = SolarWindParams(
        density=_TRUE_DENSITY,
        velocity_rtn=proton_velocity_rtn.copy(),
        temperature=_TRUE_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    alpha_truth = SolarWindParams(
        density=_TRUE_ALPHA_DENSITY,
        velocity_rtn=alpha_velocity_rtn.copy(),
        temperature=_TRUE_ALPHA_TEMPERATURE_K,
        mass=ALPHA_PARTICLE_MASS_KG,
    )
    proton_ideal, _ = model_solar_wind_ideal_coincidence_rates(proton_truth, proton_ctx)
    alpha_ideal, _ = model_solar_wind_ideal_coincidence_rates(alpha_truth, alpha_ctx)
    ideal = proton_ideal + alpha_ideal
    flat_rates = ideal * deadtime_factor(ideal)
    full_rates = np.zeros((_N_SWEEPS, _N_BINS))
    full_rates[:, SWAPI_SCIENCE_BINS] = flat_rates.reshape(_N_SWEEPS, n)
    chunk = SwapiL2Data(
        sci_start_time=_SCI_START_TIME.copy(),
        energy=np.tile(_FULL_ENERGY, (_N_SWEEPS, 1)),
        coincidence_count_rate=full_rates,
        coincidence_count_rate_uncertainty=np.full_like(full_rates, 0.1),
    )
    return chunk


def _build_truth_chunk(response, efficiency_table):
    """Forward-model a clean proton+alpha chunk from `response` over the science
    bin range, returning the chunk plus the rotations and truth velocities used
    to synthesize it. The alpha truth velocity is constructed here so all three
    fitter test classes share the same proton-to-alpha relationship."""
    science_rotations = _spice_rotations(SWAPI_SCIENCE_BINS)
    proton_velocity_rtn = _truth_velocity_rtn(science_rotations)
    alpha_velocity_rtn = proton_velocity_rtn + _TRUE_DELTA_V_KM_S * _B_HAT_RTN
    chunk = _synthesize_chunk(
        response=response,
        rotations=science_rotations,
        proton_velocity_rtn=proton_velocity_rtn,
        alpha_velocity_rtn=alpha_velocity_rtn,
        efficiency_table=efficiency_table,
    )
    return chunk, science_rotations, proton_velocity_rtn, alpha_velocity_rtn


def _with_count_rate(chunk, count_rate):
    return SwapiL2Data(
        sci_start_time=chunk.sci_start_time,
        energy=chunk.energy,
        coincidence_count_rate=count_rate,
        coincidence_count_rate_uncertainty=chunk.coincidence_count_rate_uncertainty,
    )


def _with_nan_at(chunk, sweep, bin_):
    bad = chunk.coincidence_count_rate.copy()
    bad[sweep, bin_] = np.nan
    return _with_count_rate(chunk, bad)


def _with_zero_energy_at(chunk, sweep, bin_):
    bad = chunk.energy.copy()
    bad[sweep, bin_] = 0.0
    return SwapiL2Data(
        sci_start_time=chunk.sci_start_time,
        energy=bad,
        coincidence_count_rate=chunk.coincidence_count_rate,
        coincidence_count_rate_uncertainty=chunk.coincidence_count_rate_uncertainty,
    )


def _assert_all_nan(tc, result, scalar_keys, array_keys):
    for key in scalar_keys:
        tc.assertTrue(np.isnan(result[key]), msg=f"{key} not NaN")
    for key in array_keys:
        tc.assertTrue(np.all(np.isnan(result[key])), msg=f"{key} not all-NaN")


def _assert_peak_speed_fallback(tc, result, scalar_keys, array_keys):
    """Failure-mode contract for the proton fitter: `proton_sw_speed` is the
    peak-bin ESA voltage converted to a cold-proton speed (within ~15 km/s of
    `_TRUE_BULK_SPEED` for our forward-modelled chunk), and every other science
    field NaN-fills."""
    tc.assertAlmostEqual(result["proton_sw_speed"], _TRUE_BULK_SPEED, delta=15.0)
    other_scalars = [k for k in scalar_keys if k != "proton_sw_speed"]
    _assert_all_nan(tc, result, other_scalars, array_keys)


def _assert_alpha_flag_and_all_nan(tc, result, flag):
    tc.assertEqual(int(result["quality_flags"]), int(flag))
    _assert_all_nan(tc, result, _ALPHA_SCALAR_KEYS, _ALPHA_ARRAY_KEYS)


def _assert_proton_flag_and_peak_fallback(tc, result, flag):
    tc.assertEqual(int(result["quality_flags"]), int(flag))
    _assert_peak_speed_fallback(tc, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)


def _zero_chunk():
    return SwapiL2Data(
        sci_start_time=np.array([_EPOCH_TT2000], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


def _chunk_at_time(t):
    return SwapiL2Data(
        sci_start_time=np.array([t], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


# 10**18 ns ≈ 31.7 years past `_EPOCH_TT2000` — beyond every kernel in
# `spice_kernels/`, so SPICE raises `SpiceSPKINSUFFDATA` for this chunk.
_OUT_OF_COVERAGE_START_TIME = _EPOCH_TT2000 + 10**18


def _out_of_coverage_chunk():
    return SwapiL2Data(
        sci_start_time=np.array([_OUT_OF_COVERAGE_START_TIME], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


def _predicted_ephemeris_kernel_paths():
    spice_file_names = [
        "imap_recon_20250925_20260511_v01.bsp",
        "naif0012.tls",
        "imap_sclk_0171.tsc",
        "imap_science_120.tf",
        "imap_130.tf",
        "imap_pred_od037_20260706_20260817_v01.bsp",
        "de440.bsp",
        "pck00011.tpc",
        "imap_2025_105_2026_105_01.ah.bc",
        "imap_2026_189_2026_189_001.ah.bc",
    ]
    return [str(get_integration_test_spice_data_path(name)) for name in spice_file_names]


# ----- ProtonChunkFitter ----------------------------------------------------


class TestProtonChunkFitterPrecomputeGeometry(SpiceTestCase):
    """Tests for `ProtonChunkFitter.precompute_geometry` with real SPICE kernels."""

    def test_success_returns_epoch_rotations_and_sc_velocity(self):
        """At an in-coverage chunk, precompute_geometry returns the chunk midpoint epoch, a per-bin rotation array of the right shape, and a 3-vector spacecraft velocity."""
        [(epoch, rotation_matrices, spacecraft_velocity, flags)] = (
            ProtonChunkFitter().precompute_geometry([_zero_chunk()])
        )

        self.assertEqual(epoch, _CHUNK_EPOCH)
        assert rotation_matrices is not None and spacecraft_velocity is not None
        self.assertEqual(rotation_matrices.shape, (_N_BINS - 1, 3, 3))
        self.assertEqual(spacecraft_velocity.shape, (3,))
        self.assertEqual(flags, SwapiL3Flags.NONE)

    def test_spice_failure_yields_none_for_both_outputs(self):
        """If the chunk falls outside SPICE coverage, the proton fitter returns None for both rotations and spacecraft velocity."""
        [(_, rotation_matrices, spacecraft_velocity, flags)] = (
            ProtonChunkFitter().precompute_geometry([_out_of_coverage_chunk()])
        )

        self.assertIsNone(rotation_matrices)
        self.assertIsNone(spacecraft_velocity)
        self.assertEqual(flags, SwapiL3Flags.NONE)

    def test_flags_chunks_that_need_predicted_ephemeris(self):
        spice_file_names = [
            "imap_recon_20250925_20260511_v01.bsp",
            "naif0012.tls",
            "imap_sclk_0171.tsc",
            "imap_science_120.tf",
            "imap_130.tf",
            "imap_pred_od037_20260706_20260817_v01.bsp",
            "de440.bsp",
            "pck00011.tpc",
            "imap_2025_105_2026_105_01.ah.bc",
            "imap_2026_189_2026_189_001.ah.bc"
        ]
        spice_test_paths = [str(get_integration_test_spice_data_path(file_name)) for file_name in spice_file_names]

        chunk_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260708")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        with KernelPool(spice_test_paths):
            [geom1, geom2] = ProtonChunkFitter().precompute_geometry(chunks)
            epoch1, rotation_matrices1, spacecraft_velocity1, flags1 = geom1
            epoch2, rotation_matrices2, spacecraft_velocity2, flags2 = geom2

            self.assertEqual(SwapiL3Flags.PREDICTIVE_EPHEMERIS, flags1)
            self.assertEqual(SwapiL3Flags.NONE, flags2)

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.PredictedEphemerisTracker")
    def test_uses_predicted_ephemeris_tracker(self, mock_tracker_class):
        mock_tracker_1 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_2 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_class.side_effect = [
            mock_tracker_1,
            mock_tracker_2,
        ]
        chunk_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260308")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        ProtonChunkFitter().precompute_geometry(chunks)
        self.assertEqual(2, mock_tracker_class.call_count)
        self.assertEqual(2, mock_tracker_1.run.call_count)
        self.assertEqual(get_swapi_geometry, mock_tracker_1.run.call_args_list[0].args[0])
        self.assertEqual(get_spacecraft_velocity_rtn, mock_tracker_1.run.call_args_list[1].args[0])

        self.assertEqual(2, mock_tracker_2.run.call_count)
        self.assertEqual(get_swapi_geometry, mock_tracker_2.run.call_args_list[0].args[0])
        self.assertEqual(get_spacecraft_velocity_rtn, mock_tracker_2.run.call_args_list[1].args[0])

class TestProtonChunkFitterFitChunk(SpiceTestCase):
    """Tests for `ProtonChunkFitter.fit_chunk` — end-to-end proton fit plus
    fill-value branches on a forward-modelled chunk."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, cls.rotations, cls.true_proton_velocity_rtn, _ = _build_truth_chunk(cls.response, efficiency_table)
        cls.result = ProtonChunkFitter().fit_chunk(
            cls.chunk, _CHUNK_EPOCH, cls.rotations, _SC_VELOCITY_RTN.copy(), SwapiL3Flags.NONE,
        )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_quality_flag_none_and_epoch_passthrough(self):
        """A clean synthetic chunk produces a NONE quality flag and passes the chunk epoch straight through to the result."""
        self.assertEqual(self.result["quality_flags"], SwapiL3Flags.NONE)
        self.assertEqual(self.result["epoch"], _CHUNK_EPOCH)

    def test_recovers_truth_moments(self):
        """Fitting a forward-modeled chunk back recovers the true density, temperature, and bulk velocity within a few percent."""
        self.assertAlmostEqual(
            self.result["proton_sw_density"], _TRUE_DENSITY, delta=0.05 * _TRUE_DENSITY
        )
        self.assertAlmostEqual(
            self.result["proton_sw_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            self.result["proton_sw_velocity_rtn"], self.true_proton_velocity_rtn, atol=5.0
        )

    def test_uncertainties_are_strictly_positive(self):
        """Every reported scalar uncertainty is strictly positive, so the LM Jacobian did not degenerate to zero."""
        # Bit-exact zero would mean the LM Jacobian degenerated.
        for key in _PROTON_SCALAR_KEYS:
            if key.endswith("_uncert"):
                with self.subTest(key=key):
                    self.assertGreater(self.result[key], 0.0)

    def test_speed_is_norm_of_sc_frame_velocity(self):
        """Reported proton speed equals the magnitude of the SC-frame bulk velocity (rotation to DPS is norm-preserving)."""
        np.testing.assert_allclose(
            self.result["proton_sw_speed"],
            np.linalg.norm(self.result["proton_sw_velocity_rtn"]),
            rtol=1e-9,
        )

    def test_sun_frame_velocity_is_sc_frame_plus_sc_velocity(self):
        """Sun-frame velocity is the SC-frame velocity plus the SC orbital velocity, and Sun-frame speed is its magnitude."""
        np.testing.assert_allclose(
            self.result["proton_sw_velocity_rtn_sun"],
            self.result["proton_sw_velocity_rtn"] + _SC_VELOCITY_RTN,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.result["proton_sw_speed_sun"],
            np.linalg.norm(self.result["proton_sw_velocity_rtn_sun"]),
            rtol=1e-9,
        )

    def test_speed_matches_magnitude_of_velocity_rtn(self):
        """`proton_sw_speed` is the magnitude of the SC-frame RTN bulk velocity (magnitude is rotation-invariant)."""
        np.testing.assert_allclose(
            self.result["proton_sw_speed"],
            np.linalg.norm(self.result["proton_sw_velocity_rtn"]),
            rtol=1e-12,
        )

    def test_velocity_covariance_is_symmetric_psd(self):
        """The 3x3 velocity covariance is symmetric and positive-semidefinite."""
        covariance = self.result["proton_sw_velocity_rtn_covariance"]
        self.assertEqual(covariance.shape, (3, 3))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], 0.0)

    def test_missing_sc_velocity_fills_only_sun_frame_outputs(self):
        """Calling fit_chunk with no SC velocity still runs the proton fit normally: density, temperature, SC-frame bulk velocity, covariance, and peak speed are populated from the fit; only the sun-frame outputs (`proton_sw_speed_sun`, `proton_sw_velocity_rtn_sun`) are fill values."""
        result = ProtonChunkFitter().fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, None, SwapiL3Flags.NONE,
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)

        self.assertTrue(np.isnan(result["proton_sw_speed_sun"]))
        self.assertTrue(np.isnan(result["proton_sw_speed_sun_uncert"]))
        self.assertTrue(np.all(np.isnan(result["proton_sw_velocity_rtn_sun"])))

        self.assertAlmostEqual(
            result["proton_sw_density"], _TRUE_DENSITY, delta=0.05 * _TRUE_DENSITY
        )
        self.assertAlmostEqual(
            result["proton_sw_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            result["proton_sw_velocity_rtn"],
            self.true_proton_velocity_rtn,
            atol=5.0,
        )
        self.assertTrue(
            np.all(np.isfinite(result["proton_sw_velocity_rtn_covariance"]))
        )
        self.assertTrue(np.isfinite(result["proton_sw_speed"]))

    def test_uses_quality_flag_from_geometry(self):
        result = ProtonChunkFitter().fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN.copy(), SwapiL3Flags.PREDICTIVE_EPHEMERIS,
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.PREDICTIVE_EPHEMERIS)



# ----- AlphaChunkFitter -----------------------------------------------------


class TestAlphaChunkFitterPrecomputeGeometry(SpiceTestCase):
    """Tests for `AlphaChunkFitter.precompute_geometry` with real SPICE kernels."""

    def _mag_centered_on(self, epoch_ns):
        offsets = np.array([-1_000_000_000, 0, 1_000_000_000], dtype=np.int64)
        return MagData(epoch=epoch_ns + offsets, mag_data=np.tile(_B_HAT_RTN, (3, 1)))

    def test_success_returns_rotations_sc_velocity_and_b_hat(self):
        """With both SPICE and MAG available, alpha precompute_geometry returns the chunk epoch, a science-bin rotation array of the right shape (matching the proton fit it shares with proton-sw), the spacecraft velocity, and the median B̂ in the chunk window."""
        [(epoch, rotation_matrices, sc_velocity, b_hat, flag)] = AlphaChunkFitter(
            self._mag_centered_on(_CHUNK_EPOCH)
        ).precompute_geometry([_zero_chunk()])

        self.assertEqual(epoch, _CHUNK_EPOCH)
        assert rotation_matrices is not None and sc_velocity is not None
        self.assertEqual(
            rotation_matrices.shape,
            (SWAPI_SCIENCE_BINS.stop - SWAPI_SCIENCE_BINS.start, 3, 3),
        )
        self.assertEqual(sc_velocity.shape, (3,))
        self.assertEqual(SwapiL3Flags.NONE, flag)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_spice_failure_yields_none_rotations_and_sc_velocity_but_keeps_b_hat(self):
        """When the chunk falls outside SPICE coverage, alpha precompute returns None rotations and None spacecraft velocity, but B̂ is still computed from MAG since that path is independent."""
        out_of_coverage_chunk_epoch = _OUT_OF_COVERAGE_START_TIME + THIRTY_SECONDS_IN_NANOSECONDS
        [(_, rotation_matrices, sc_velocity, b_hat, _)] = AlphaChunkFitter(
            self._mag_centered_on(out_of_coverage_chunk_epoch)
        ).precompute_geometry([_out_of_coverage_chunk()])
        self.assertIsNone(rotation_matrices)
        self.assertIsNone(sc_velocity)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_empty_mag_window_yields_nan_b_hat(self):
        """When no MAG samples fall inside the chunk window, B̂ comes back as NaN even though rotations were successfully computed."""
        far_future = _EPOCH_TT2000 + 10**18
        [(_, _, _, b_hat, _)] = AlphaChunkFitter(
            self._mag_centered_on(far_future)
        ).precompute_geometry([_zero_chunk()])
        self.assertTrue(np.all(np.isnan(b_hat)))

    @patch('imap_l3_processing.swapi.l3a.chunk_fits.compute_direction_of_mean_magnetic_field_over_chunk')
    def test_flags_chunks_that_need_predicted_ephemeris(self, _):
        spice_file_names = [
            "imap_recon_20250925_20260511_v01.bsp",
            "naif0012.tls",
            "imap_sclk_0171.tsc",
            "imap_science_120.tf",
            "imap_130.tf",
            "imap_pred_od037_20260706_20260817_v01.bsp",
            "de440.bsp",
            "pck00011.tpc",
            "imap_2025_105_2026_105_01.ah.bc",
            "imap_2026_189_2026_189_001.ah.bc"
        ]
        spice_test_paths = [str(get_integration_test_spice_data_path(file_name)) for file_name in spice_file_names]

        chunk_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260708")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        with KernelPool(spice_test_paths):
            [geom1, geom2] = AlphaChunkFitter(None).precompute_geometry(chunks)
            epoch1, rotation_matrices1, spacecraft_velocity1, bhat1, flags1 = geom1
            epoch2, rotation_matrices2, spacecraft_velocity2, bhat2, flags2 = geom2

            self.assertEqual(SwapiL3Flags.PREDICTIVE_EPHEMERIS, flags1)
            self.assertEqual(SwapiL3Flags.NONE, flags2)

    @patch('imap_l3_processing.swapi.l3a.chunk_fits.compute_direction_of_mean_magnetic_field_over_chunk')
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.PredictedEphemerisTracker")
    def test_uses_predicted_ephemeris_tracker(self, mock_tracker_class, _):
        mock_tracker_1 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_2 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_class.side_effect = [
            mock_tracker_1,
            mock_tracker_2,
        ]
        chunk_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260308")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _chunk_at_time(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        AlphaChunkFitter(None).precompute_geometry(chunks)
        self.assertEqual(2, mock_tracker_class.call_count)
        self.assertEqual(2, mock_tracker_1.run.call_count)
        self.assertEqual(get_swapi_geometry, mock_tracker_1.run.call_args_list[0].args[0])
        self.assertEqual(get_spacecraft_velocity_rtn, mock_tracker_1.run.call_args_list[1].args[0])

        self.assertEqual(2, mock_tracker_2.run.call_count)
        self.assertEqual(get_swapi_geometry, mock_tracker_2.run.call_args_list[0].args[0])
        self.assertEqual(get_spacecraft_velocity_rtn, mock_tracker_2.run.call_args_list[1].args[0])



class TestAlphaChunkFitterFitChunk(SpiceTestCase):
    """Tests for `AlphaChunkFitter.fit_chunk` — stage ordering plus fill-value branches in the alpha fit."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, _, cls.true_proton_velocity_rtn, cls.true_alpha_velocity_rtn = _build_truth_chunk(cls.response, efficiency_table)
        # AlphaChunkFitter and ProtonChunkFitter share the same proton fit on
        # `SWAPI_SCIENCE_BINS`, so the rotations passed in must span the full
        # science range; AlphaChunkFitter slices down to coarse for Stage 2.
        cls.rotations = _spice_rotations(SWAPI_SCIENCE_BINS)
        cls.fitter = AlphaChunkFitter(mag_data=None)
        cls.happy_result = cls.fitter.fit_chunk(
            cls.chunk, _CHUNK_EPOCH, cls.rotations, _SC_VELOCITY_RTN, _B_HAT_RTN, SwapiL3Flags.NONE
        )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_recovers_alpha_truth_moments(self):
        """Fitting a forward-modeled proton+alpha chunk recovers the true alpha density, temperature, and bulk velocity within a few percent."""
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_density"],
            _TRUE_ALPHA_DENSITY,
            delta=0.10 * _TRUE_ALPHA_DENSITY,
        )
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_temperature"],
            _TRUE_ALPHA_TEMPERATURE_K,
            delta=0.10 * _TRUE_ALPHA_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_velocity_rtn"], self.true_alpha_velocity_rtn, atol=5.0
        )
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_speed"],
            np.linalg.norm(self.true_alpha_velocity_rtn),
            atol=5.0,
        )

    def test_sun_frame_velocity_is_sc_frame_plus_sc_velocity(self):
        """Sun-frame alpha velocity is the SC-frame velocity plus the SC orbital velocity, and `alpha_sw_speed_sun` is its magnitude."""
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_velocity_rtn_sun"],
            self.happy_result["alpha_sw_velocity_rtn"] + _SC_VELOCITY_RTN,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_speed_sun"],
            np.linalg.norm(self.happy_result["alpha_sw_velocity_rtn_sun"]),
            atol=1e-9,
        )

    def test_missing_sc_velocity_fills_only_sun_frame_outputs(self):
        """Calling fit_chunk with no SC velocity still recovers the alpha moments; only the sun-frame outputs (`alpha_sw_speed_sun`, `alpha_sw_velocity_rtn_sun`) are fill values."""
        result = self.fitter.fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, None, _B_HAT_RTN, SwapiL3Flags.NONE
        )
        self.assertEqual(int(result["quality_flags"]), int(SwapiL3Flags.NONE))

        self.assertTrue(np.isnan(result["alpha_sw_speed_sun"]))
        self.assertTrue(np.isnan(result["alpha_sw_speed_sun_uncert"]))
        self.assertTrue(np.all(np.isnan(result["alpha_sw_velocity_rtn_sun"])))

        self.assertTrue(np.isfinite(result["alpha_sw_speed"]))
        self.assertTrue(np.isfinite(result["alpha_sw_density"]))
        self.assertTrue(np.all(np.isfinite(result["alpha_sw_velocity_rtn"])))

    def test_quality_flag_none_on_clean_chunk(self):
        """A clean chunk yields a NONE quality flag (both Stage 1 and Stage 2 converged)."""
        self.assertEqual(
            int(self.happy_result["quality_flags"]), int(SwapiL3Flags.NONE)
        )

    def test_uses_quality_flag_from_geometry(self):
        result = AlphaChunkFitter(None).fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN.copy(), _B_HAT_RTN.copy(), SwapiL3Flags.PREDICTIVE_EPHEMERIS,
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.PREDICTIVE_EPHEMERIS)



# ----- ParallelChunkRunner --------------------------------------------------


class _RecordingChunkFitter(ChunkFitter):
    """A real `ChunkFitter` that echoes the chunk's start time. Module-level
    so fork-spawned workers can locate it via inherited memory."""

    def precompute_geometry(self, chunks):
        return [(int(chunk.sci_start_time[0]),) for chunk in chunks]

    def fit_chunk(self, chunk, epoch):
        return {
            "epoch": epoch,
            "first_start_time": int(chunk.sci_start_time[0]),
        }


class _WarmCacheProbeChunkFitter(ChunkFitter):
    """A real `ChunkFitter` whose `fit_chunk` reports the `_passband_grid_cache` size
    seen by the worker process. Module-level for fork inheritance."""

    def precompute_geometry(self, chunks):
        return [(int(chunk.sci_start_time[0]),) for chunk in chunks]

    def fit_chunk(self, chunk, epoch):
        worker_response = chunk_fits._shared["swapi_response"]
        return {
            "epoch": epoch,
            "worker_cache_size": len(worker_response._passband_grid_cache),
        }


def _make_chunk_with_start_time(start_time):
    return SwapiL2Data(
        sci_start_time=np.array([start_time], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


@unittest.skipIf(platform.system() == "Windows", "fork is not available on Windows")
class TestParallelChunkRunnerOrchestration(unittest.TestCase):
    """Tests for `ParallelChunkRunner.run` against a real fork-based Pool."""

    def tearDown(self):
        _clear_shared()

    def test_dispatches_per_chunk_and_stacks_outputs_in_chunk_order(self):
        """Two chunks with distinct start times produce a stacked output where each per-key array preserves chunk order across the real fork pool."""
        chunks = [
            _make_chunk_with_start_time(_EPOCH_TT2000),
            _make_chunk_with_start_time(_EPOCH_TT2000 + 12_000_000_000),
        ]
        runner = ParallelChunkRunner(
            swapi_response=_swapi_response_with_warm_cache(np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS)),
            efficiency_table=_efficiency_table(),
        )

        result = runner.run(chunks, _RecordingChunkFitter())

        expected_epochs = np.array(
            [int(chunks[0].sci_start_time[0]), int(chunks[1].sci_start_time[0])]
        )
        np.testing.assert_array_equal(result["epoch"], expected_epochs)
        np.testing.assert_array_equal(result["first_start_time"], expected_epochs)

    def test_workers_see_parent_warm_cache_under_fork(self):
        """A passband grid cache populated in the parent before `runner.run` is visible inside each fork-spawned worker at the same size."""
        voltages = np.array([10.0, 50.0, 100.0])
        response = _swapi_response_with_warm_cache(voltages)
        parent_cache_size = len(response._passband_grid_cache)
        self.assertEqual(parent_cache_size, len(voltages))

        runner = ParallelChunkRunner(
            swapi_response=response, efficiency_table=_efficiency_table()
        )

        result = runner.run(
            [_make_chunk_with_start_time(_EPOCH_TT2000)], _WarmCacheProbeChunkFitter()
        )


        np.testing.assert_array_equal(
            result["worker_cache_size"], np.array([parent_cache_size])
        )


# ----- Proton quality flags -------------------------------------------------


class TestProtonChunkFitterQualityFlags(SpiceTestCase):
    """Quality-flag branches of `ProtonChunkFitter.fit_chunk`, ordered to match `docs/swapi/solar-wind-moments.md` §Quality flags: `BAD_FIT`, `FIT_ERROR`, then NONE-flag data-gap fills."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, cls.rotations, _, _ = _build_truth_chunk(cls.response, efficiency_table)
        cls.fitter = ProtonChunkFitter()

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def _fit(self, chunk):
        return self.fitter.fit_chunk(
            chunk, _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN.copy(), SwapiL3Flags.NONE,
        )

    def test_bad_fit_when_peak_does_not_match_maxwellian(self):
        """Scrambling the proton peak bins yields a spectrum the LM fit cannot describe (the BAD_FIT quality guard fires); the chunk fitter surfaces `BAD_FIT` with peak-bin speed and NaN-filled moments."""
        rng = np.random.default_rng(0)
        mean_count = self.chunk.coincidence_count_rate.mean(axis=0)
        peak_bin = int(np.argmax(mean_count))
        peak_window = slice(peak_bin - 2, peak_bin + 3)
        permuted = rng.permutation(np.arange(peak_window.start, peak_window.stop))
        corrupted = self.chunk.coincidence_count_rate.copy()
        corrupted[:, peak_window] = corrupted[:, permuted]

        result = self._fit(_with_count_rate(self.chunk, corrupted))

        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.BAD_FIT)

    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    def test_fit_error_when_inner_fit_returns_fit_error(self, mock_fit_proton):
        """When the inner proton fit returns `FIT_ERROR` with NaN moments (scipy LM reported `success=False`), the chunk fitter propagates the flag, falls back to peak-bin speed, and NaN-fills every other science field. Mocked because there is no clean way to force scipy LM to report `success=False` from real inputs."""
        nan = ufloat(np.nan, np.nan)
        mock_fit_proton.return_value = ProtonSolarWindFitResult(
            density=nan,
            temperature=nan,
            velocity_rtn=(nan, nan, nan),
            quality_flag=int(SwapiL3Flags.FIT_ERROR),
        )

        result = self._fit(self.chunk)

        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.FIT_ERROR)

    @patch("imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model.calculate_initial_guess")
    def test_fit_error_when_initial_guess_is_nan(self, mock_initial_guess):
        """A NaN-valued initial guess causes scipy `least_squares` to reject `x0` as infeasible; the chunk fitter catches the exception, surfaces `FIT_ERROR`, and falls back to peak-bin speed. Mocked because `calculate_initial_guess` never returns NaN from real inputs (it raises instead)."""
        mock_initial_guess.return_value = SolarWindParams(
            density=np.nan,
            velocity_rtn=np.full(3, np.nan),
            temperature=np.nan,
            mass=PROTON_MASS_KG,
        )

        result = self._fit(self.chunk)

        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.FIT_ERROR)

    def test_fit_error_when_initial_guess_raises(self):
        """A spectrum with a single non-zero bin (no Gaussian shape to fit) makes scipy `curve_fit` raise inside the initial-guess Gaussian refine; the chunk fitter catches the exception, surfaces `FIT_ERROR`, and falls back to peak-bin speed."""
        peak_bin = int(np.argmax(self.chunk.coincidence_count_rate.mean(axis=0)))
        single_bin = np.zeros_like(self.chunk.coincidence_count_rate)
        single_bin[:, peak_bin] = 100.0

        result = self._fit(_with_count_rate(self.chunk, single_bin))

        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.FIT_ERROR)

    def test_no_flag_when_rotations_missing(self):
        """Calling fit_chunk with no rotations reports a NONE quality flag (ephemeris gaps are treated as data gaps without a dedicated flag) and falls back to peak-bin ESA voltage as `proton_sw_speed`; every other science field NaN-fills."""
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, None, _SC_VELOCITY_RTN, SwapiL3Flags.NONE)
        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.NONE)

    def test_no_flag_when_count_rate_has_nan(self):
        """A NaN in the count rate is treated as an L2 data gap: the quality flag is NONE, `proton_sw_speed` reports the peak-bin speed (taken across finite bins via `nanargmax`), and every other science field NaN-fills."""
        result = self._fit(_with_nan_at(self.chunk, 0, 5))
        _assert_proton_flag_and_peak_fallback(self, result, SwapiL3Flags.NONE)

    def test_succeeds_when_fine_sweep_bin_has_zero_voltage(self):
        """Production sweeps regularly carry zero voltages in the fine-sweep bins. The proton chunk fitter drops those bins (and the matching count rates / rotations) at the call site before building the fit context, so the fit converges on the surviving science bins and the quality flag is NONE."""
        fine_bin = SWAPI_FINE_SWEEP_BINS.start + 2
        result = self._fit(_with_zero_energy_at(self.chunk, 0, fine_bin))
        self.assertEqual(int(result["quality_flags"]), int(SwapiL3Flags.NONE))
        self.assertAlmostEqual(
            result["proton_sw_density"], _TRUE_DENSITY, delta=0.05 * _TRUE_DENSITY
        )
        self.assertAlmostEqual(
            result["proton_sw_speed"], _TRUE_BULK_SPEED, delta=0.05 * _TRUE_BULK_SPEED
        )


    def test_combines_geometry_and_fitting_flags(self):
        peak_bin = int(np.argmax(self.chunk.coincidence_count_rate.mean(axis=0)))
        single_bin = np.zeros_like(self.chunk.coincidence_count_rate)
        single_bin[:, peak_bin] = 100.0

        result = self.fitter.fit_chunk(
            _with_count_rate(self.chunk, single_bin),
            _CHUNK_EPOCH,
            self.rotations,
            _SC_VELOCITY_RTN.copy(),
            SwapiL3Flags.PREDICTIVE_EPHEMERIS,
        )
        self.assertEqual(SwapiL3Flags.FIT_ERROR | SwapiL3Flags.PREDICTIVE_EPHEMERIS, result["quality_flags"])



# ----- Alpha quality flags --------------------------------------------------


class TestAlphaChunkFitterQualityFlags(SpiceTestCase):
    """Quality-flag branches of `AlphaChunkFitter.fit_chunk` and the `PRELIMINARY_MAG` bit-OR in `SwapiProcessor.process_l3a_alpha`, ordered to match `docs/swapi/solar-wind-moments.md` §Quality flags: `BAD_FIT`, `PRELIMINARY_MAG`, then NONE-flag data-gap fills."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(REALISTIC_ESA_VOLTAGES, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, cls.rotations, _, _ = _build_truth_chunk(cls.response, efficiency_table)
        cls.fitter = AlphaChunkFitter(mag_data=None)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_bad_fit_when_alpha_bump_does_not_match_maxwellian(self):
        """Scrambling and amplifying the alpha-bump bins leaves the proton peak intact but yields a Stage-2 residual the alpha LM cannot describe (the BAD_FIT quality guard fires); the chunk fitter surfaces `BAD_FIT` with every alpha moment NaN-filled."""
        rng = np.random.default_rng(0)
        permuted = rng.permutation(np.arange(_ALPHA_BUMP_BINS.start, _ALPHA_BUMP_BINS.stop))
        corrupted = self.chunk.coincidence_count_rate.copy()
        corrupted[:, _ALPHA_BUMP_BINS] = corrupted[:, permuted] * 3.0

        result = self.fitter.fit_chunk(
            _with_count_rate(self.chunk, corrupted),
            _CHUNK_EPOCH,
            self.rotations,
            _SC_VELOCITY_RTN,
            _B_HAT_RTN,
            SwapiL3Flags.NONE,
        )

        self.assertEqual(int(result["quality_flags"]), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan(self, result, _ALPHA_SCALAR_KEYS, _ALPHA_ARRAY_KEYS)

    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData")
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    def test_preliminary_mag_bit_set_iff_mag_is_preliminary(
        self, mock_chunk_l2_data, mock_runner_class, mock_alpha_data_class
    ):
        """The `PRELIMINARY_MAG` bit is OR'd onto every per-chunk quality flag when `mag_is_preliminary=True` (preserving any underlying `BAD_FIT`), and the per-chunk flags pass through untouched when `mag_is_preliminary=False`. Mocked at the processor seam because the bit-OR is processor-level wiring, not chunk-fitter behavior."""
        runner_flag = int(SwapiL3Flags.BAD_FIT)
        cases = [
            (True, runner_flag | int(SwapiL3Flags.PRELIMINARY_MAG)),
            (False, runner_flag),
        ]
        for mag_is_preliminary, expected_flag in cases:
            with self.subTest(mag_is_preliminary=mag_is_preliminary):
                mock_chunk_l2_data.return_value = [Mock()]
                mock_runner_class.return_value.run.return_value = {
                    "quality_flags": np.array([runner_flag])
                }
                dependencies = Mock(
                    mag_data=Mock(),
                    mag_is_preliminary=mag_is_preliminary,
                    swapi_response=Mock(),
                    efficiency_calibration_table=Mock(),
                )
                data = Mock(energy=np.array([[1000.0, 2000.0]]))
                metadata = InputMetadata(
                    "swapi", "l3a", datetime(2025, 1, 1), datetime(2025, 1, 2), "v001"
                )

                SwapiProcessor(Mock(), metadata).process_l3a_alpha(data, dependencies)

                passed_quality_flags = mock_alpha_data_class.call_args.kwargs[
                    "quality_flags"
                ]
                np.testing.assert_array_equal(passed_quality_flags, [expected_flag])

    def test_no_flag_when_rotations_missing(self):
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, None, None, _B_HAT_RTN, SwapiL3Flags.NONE)
        _assert_alpha_flag_and_all_nan(self, result, SwapiL3Flags.NONE)

    def test_flags_when_b_hat_is_nan(self):
        cases = (SwapiL3Flags.PREDICTIVE_EPHEMERIS, SwapiL3Flags.NONE)
        for flag in cases:
            with self.subTest(flag=flag):
                result = self.fitter.fit_chunk(
                    self.chunk, _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN, np.full(3, np.nan), flag
                )
                _assert_alpha_flag_and_all_nan(self, result, flag)

    def test_flags_when_b_hat_is_none(self):
        cases = (SwapiL3Flags.PREDICTIVE_EPHEMERIS, SwapiL3Flags.NONE)
        for flag in cases:
            with self.subTest(flag=flag):
                result = self.fitter.fit_chunk(
                    self.chunk, _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN, None, flag
                )
                _assert_alpha_flag_and_all_nan(self, result, flag)

    def test_no_flag_when_count_rate_has_nan(self):
        """A NaN in the alpha count rate is treated as an L2 data gap: every alpha field NaN-fills and bad_fit_flag is NONE."""
        result = self.fitter.fit_chunk(
            _with_nan_at(self.chunk, 0, 5),
            _CHUNK_EPOCH,
            self.rotations,
            _SC_VELOCITY_RTN,
            _B_HAT_RTN,
            SwapiL3Flags.NONE,
        )
        _assert_alpha_flag_and_all_nan(self, result, SwapiL3Flags.NONE)

    def test_fit_error_when_coarse_bin_voltage_is_zero(self):
        """Coarse-sweep bins shouldn't carry zero voltages in production, but if one does `build_solar_wind_fit_context` raises rather than silently flattening the alpha context to 1D and breaking the per-sweep aggregations. The chunk fitter's try/except catches the raise and surfaces `FIT_ERROR` with every alpha moment NaN-filled."""
        coarse_bin = SWAPI_COARSE_SWEEP_BINS.start + 5
        result = self.fitter.fit_chunk(
            _with_zero_energy_at(self.chunk, 0, coarse_bin),
            _CHUNK_EPOCH,
            self.rotations,
            _SC_VELOCITY_RTN,
            _B_HAT_RTN,
            SwapiL3Flags.NONE,
        )
        _assert_alpha_flag_and_all_nan(self, result, SwapiL3Flags.FIT_ERROR)

    def test_combines_geometry_and_fitting_flags(self):
        peak_bin = int(np.argmax(self.chunk.coincidence_count_rate.mean(axis=0)))
        single_bin = np.zeros_like(self.chunk.coincidence_count_rate)
        single_bin[:, peak_bin] = 100.0

        result = self.fitter.fit_chunk(
            _with_count_rate(self.chunk, single_bin),
            _CHUNK_EPOCH,
            self.rotations,
            _SC_VELOCITY_RTN.copy(),
            _B_HAT_RTN.copy(),
            SwapiL3Flags.PREDICTIVE_EPHEMERIS,
        )
        self.assertEqual(SwapiL3Flags.FIT_ERROR | SwapiL3Flags.PREDICTIVE_EPHEMERIS, result["quality_flags"])

# ----- PuiChunkFitter -------------------------------------------------------


_PUI_RESULT_KEYS = [
    "epoch",
    "cooling_index",
    "ionization_rate",
    "cutoff_speed",
    "background_rate",
    "density",
    "temperature",
    "quality_flags",
]


def _pui_chunk(start_time, count_rates=None):
    """50-sweep chunk shaped like a PUI input window."""
    n_sweeps = 50
    energy = np.tile(np.arange(_N_BINS) * 100.0 + 1.0, (n_sweeps, 1))
    if count_rates is None:
        count_rates = np.full((n_sweeps, _N_BINS), 5.0)
    sci_start_time = (
        start_time + np.arange(n_sweeps, dtype=np.int64) * 12 * ONE_SECOND_IN_NANOSECONDS
    )
    return SwapiL2Data(
        sci_start_time=sci_start_time,
        energy=energy,
        coincidence_count_rate=count_rates,
        coincidence_count_rate_uncertainty=np.full((n_sweeps, _N_BINS), 0.1),
    )


class TestPuiChunkFitterPrecomputeGeometry(SpiceTestCase):
    """`PuiChunkFitter.precompute_geometry` averages the 5-minute proton fits
    into the 10-minute PUI cadence, rotates each chunk into SWAPI, and
    precomputes the PUI chunk SPICE state. SPICE gaps fall back to NaN/None
    fills so they propagate to fill values in the downstream fit."""

    def _make_fitter(self, proton_results):
        return PuiChunkFitter(
            density_of_neutral_helium_lookup_table=Mock(),
            hydrogen_inflow_vector=Mock(),
            helium_inflow_vector=Mock(),
            proton_results=proton_results,
        )

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.rotate_rtn_velocity_to_swapi_per_bin")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.spiceypy")
    def test_returns_one_geometry_tuple_per_chunk(
        self,
        _,
        mock_calculate_ten_minute_velocities,
        mock_rotate_rtn_velocity_to_swapi_per_bin,
        mock_calculate_pui_energy_cutoff,
        mock_build_vasyliunas_siscoe_distribution,
    ):
        chunk = _pui_chunk(_EPOCH_TT2000)
        expected_epoch = _EPOCH_TT2000 + FIVE_MINUTES_IN_NANOSECONDS
        ten_minute_rtn = np.array([400.0, 10.0, 5.0])
        per_bin = np.full((50, 62, 3), 0.5)
        vs_dist = Mock()
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([ten_minute_rtn]),
            np.array([int(SwapiL3Flags.BAD_FIT)]),
        )
        mock_rotate_rtn_velocity_to_swapi_per_bin.return_value = per_bin
        mock_calculate_pui_energy_cutoff.side_effect = [100.0, 200.0]
        mock_build_vasyliunas_siscoe_distribution.return_value = vs_dist
        fitter = self._make_fitter({
            "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
            "quality_flags": np.array([int(SwapiL3Flags.BAD_FIT)]),
        })

        [(epoch, rtn, per_bin_swapi, flag, lower, upper, vs)] = (
            fitter.precompute_geometry([chunk])
        )

        self.assertEqual(epoch, expected_epoch)
        np.testing.assert_array_equal(rtn, ten_minute_rtn)
        np.testing.assert_array_equal(per_bin_swapi, per_bin)
        self.assertEqual(flag, int(SwapiL3Flags.BAD_FIT))
        self.assertEqual(lower, 1.25 * 100.0)
        self.assertEqual(upper, 1.2 * 200.0)
        self.assertIs(vs, vs_dist)
        self.assertEqual(
            fitter.sw_velocity_rtn_by_chunk_epoch[expected_epoch].tolist(),
            ten_minute_rtn.tolist(),
        )

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.rotate_rtn_velocity_to_swapi_per_bin")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    def test_spice_gap_on_rotate_yields_nan_per_bin_velocity(
        self,
        mock_calculate_ten_minute_velocities,
        mock_rotate_rtn_velocity_to_swapi_per_bin,
        mock_calculate_pui_energy_cutoff,
        mock_build_vasyliunas_siscoe_distribution,
    ):
        from spiceypy.utils.exceptions import SpiceyError as _SpiceyError

        chunk = _pui_chunk(_EPOCH_TT2000)
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[400.0, 10.0, 5.0]]),
            np.array([int(SwapiL3Flags.NONE)]),
        )
        mock_rotate_rtn_velocity_to_swapi_per_bin.side_effect = _SpiceyError("gap")
        mock_calculate_pui_energy_cutoff.side_effect = [1.0, 2.0]
        mock_build_vasyliunas_siscoe_distribution.return_value = Mock()
        fitter = self._make_fitter({
            "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
            "quality_flags": np.array([int(SwapiL3Flags.NONE)]),
        })

        [(_, _, per_bin_swapi, _, _, _, _)] = fitter.precompute_geometry([chunk])

        self.assertEqual(per_bin_swapi.shape, (50, 62, 3))
        self.assertTrue(np.all(np.isnan(per_bin_swapi)))

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.rotate_rtn_velocity_to_swapi_per_bin")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.spiceypy")
    def test_nan_ten_minute_velocity_skips_spice_state(
        self,
        mock_spiceypy,
        mock_calculate_ten_minute_velocities,
        mock_rotate_rtn_velocity_to_swapi_per_bin,
        mock_calculate_pui_energy_cutoff,
        mock_build_vasyliunas_siscoe_distribution,
    ):
        chunk = _pui_chunk(_EPOCH_TT2000)
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[np.nan, np.nan, np.nan]]),
            np.array([int(SwapiL3Flags.NONE)]),
        )
        mock_rotate_rtn_velocity_to_swapi_per_bin.return_value = np.full(
            (50, 62, 3), 0.0
        )
        fitter = self._make_fitter({
            "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
            "quality_flags": np.array([int(SwapiL3Flags.NONE)]),
        })

        [(_, _, _, _, lower, upper, vs)] = fitter.precompute_geometry([chunk])

        self.assertIsNone(lower)
        self.assertIsNone(upper)
        self.assertIsNone(vs)
        mock_spiceypy.unitim.assert_not_called()
        mock_calculate_pui_energy_cutoff.assert_not_called()
        mock_build_vasyliunas_siscoe_distribution.assert_not_called()


    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.rotate_rtn_velocity_to_swapi_per_bin")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    def test_spice_gap_on_precompute_spice_state_yields_none(
        self,
        mock_calculate_ten_minute_velocities,
        mock_rotate_rtn_velocity_to_swapi_per_bin,
        _,
        mock_build_vasyliunas_siscoe_distribution,
    ):
        from spiceypy.utils.exceptions import SpiceyError as _SpiceyError

        chunk = _pui_chunk(_EPOCH_TT2000)
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[400.0, 10.0, 5.0]]),
            np.array([int(SwapiL3Flags.NONE)]),
        )
        mock_rotate_rtn_velocity_to_swapi_per_bin.return_value = np.full(
            (50, 62, 3), 0.0
        )
        mock_build_vasyliunas_siscoe_distribution.side_effect = _SpiceyError("gap")
        fitter = self._make_fitter({
            "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
            "quality_flags": np.array([int(SwapiL3Flags.NONE)]),
        })

        [(_, _, _, _, lower, upper, vs)] = fitter.precompute_geometry([chunk])

        self.assertIsNone(lower)
        self.assertIsNone(upper)
        self.assertIsNone(vs)

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    def test_flags_chunks_that_need_predicted_ephemeris(
            self,
            mock_calculate_ten_minute_velocities,
            _,
            __,
    ):
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[400.0, 10.0, 5.0],[400.0, 10.0, 5.0]]),
            np.array([int(SwapiL3Flags.NONE), int(SwapiL3Flags.NONE)]),
        )

        chunk_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260708")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        with KernelPool(_predicted_ephemeris_kernel_paths()):
            fitter = self._make_fitter({
                "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
                "quality_flags": [],
            })

            [geom1, geom2] = fitter.precompute_geometry(chunks)

            [_, _, _, proton_sw_quality_flag1, _, _, _] = geom1
            [_, _, _, proton_sw_quality_flag2, _, _, _] = geom2

            self.assertEqual(SwapiL3Flags.PREDICTIVE_EPHEMERIS, proton_sw_quality_flag1)
            self.assertEqual(SwapiL3Flags.NONE, proton_sw_quality_flag2)

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    def test_combines_upstream_proton_and_predicted_ephemeris_flags(
            self,
            mock_calculate_ten_minute_velocities,
            _,
            __,
    ):
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[400.0, 10.0, 5.0], [400.0, 10.0, 5.0]]),
            np.array([int(SwapiL3Flags.BAD_FIT), int(SwapiL3Flags.BAD_FIT)]),
        )

        chunk_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260708")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        with KernelPool(_predicted_ephemeris_kernel_paths()):
            fitter = self._make_fitter({
                "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
                "quality_flags": [],
            })

            [geom1, geom2] = fitter.precompute_geometry(chunks)

            [_, _, _, proton_sw_quality_flag1, _, _, _] = geom1
            [_, _, _, proton_sw_quality_flag2, _, _, _] = geom2

            self.assertEqual(
                SwapiL3Flags.BAD_FIT | SwapiL3Flags.PREDICTIVE_EPHEMERIS,
                proton_sw_quality_flag1,
            )
            self.assertEqual(SwapiL3Flags.BAD_FIT, proton_sw_quality_flag2)

    @patch("imap_l3_processing.swapi.l3a.chunk_fits.build_vasyliunas_siscoe_distribution")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pui_energy_cutoff")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.rotate_rtn_velocity_to_swapi_per_bin")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_ten_minute_velocities")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.PredictedEphemerisTracker")
    def test_uses_predicted_ephemeris_tracker(
            self,
            mock_tracker_class,
            mock_calculate_ten_minute_velocities,
            mock_rotate_rtn_velocity_to_swapi_per_bin,
            mock_calculate_pui_energy_cutoff,
            mock_build_vasyliunas_siscoe_distribution,
    ):
        mock_calculate_ten_minute_velocities.return_value = (
        np.array([[400.0, 10.0, 5.0],[400.0, 10.0, 5.0]]),
            np.array([int(SwapiL3Flags.NONE), int(SwapiL3Flags.NONE)]),
        )

        mock_tracker_1 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_2 = create_autospec(PredictedEphemerisTracker, used_predict=False)
        mock_tracker_class.side_effect = [
            mock_tracker_1,
            mock_tracker_2,
        ]
        chunk_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260308")
            + 12*3600*1e9
        )
        chunk_not_needing_predict = _pui_chunk(
            str_yyyymmdd_to_ttj2000ns("20260120")
            + 12*3600*1e9
        )
        chunks = [chunk_needing_predict, chunk_not_needing_predict]

        fitter = self._make_fitter({
            "proton_sw_velocity_rtn": np.array([[1.0, 2.0, 3.0]]),
            "quality_flags": [],
        })

        fitter.precompute_geometry(chunks)
        self.assertEqual(2, mock_tracker_class.call_count)
        self.assertEqual(4, mock_tracker_1.run.call_count)
        self.assertEqual(mock_rotate_rtn_velocity_to_swapi_per_bin, mock_tracker_1.run.call_args_list[0].args[0])
        self.assertEqual(mock_calculate_pui_energy_cutoff, mock_tracker_1.run.call_args_list[1].args[0])
        self.assertEqual(mock_calculate_pui_energy_cutoff, mock_tracker_1.run.call_args_list[2].args[0])
        self.assertEqual(mock_build_vasyliunas_siscoe_distribution, mock_tracker_1.run.call_args_list[3].args[0])

        self.assertEqual(4, mock_tracker_2.run.call_count)
        self.assertEqual(mock_rotate_rtn_velocity_to_swapi_per_bin, mock_tracker_2.run.call_args_list[0].args[0])
        self.assertEqual(mock_calculate_pui_energy_cutoff, mock_tracker_2.run.call_args_list[1].args[0])
        self.assertEqual(mock_calculate_pui_energy_cutoff, mock_tracker_2.run.call_args_list[2].args[0])
        self.assertEqual(mock_build_vasyliunas_siscoe_distribution, mock_tracker_2.run.call_args_list[3].args[0])


@patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_helium_pui_temperature")
@patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_helium_pui_density")
@patch("imap_l3_processing.swapi.l3a.chunk_fits.calculate_pickup_ion_values")
class TestPuiChunkFitterFitChunk(unittest.TestCase):
    """`PuiChunkFitter.fit_chunk` wraps `calculate_pickup_ion_values` and the
    two helium-moment helpers in a try/except, then OR-combines the upstream
    proton quality flag with the per-fit flag."""

    def setUp(self):
        self.epoch = _EPOCH_TT2000 + FIVE_MINUTES_IN_NANOSECONDS
        self.sw_velocity_rtn = np.array([400.0, 10.0, 5.0])
        self.bulk_sw_per_bin_swapi = np.full((50, 62, 3), 0.5)
        self.lower_energy_cutoff = 1234.0
        self.upper_energy_cutoff = 5678.0
        self.vasyliunas_siscoe_distribution = Mock()
        self.swapi_response = Mock()
        self.efficiency_table = Mock()
        self.efficiency_table.central_effective_area_scale_for.return_value = 0.42
        _populate_shared(self.swapi_response, self.efficiency_table)
        self.density_lut = Mock()
        self.hydrogen_inflow = Mock()
        self.helium_inflow = Mock()
        self.fitter = PuiChunkFitter(
            density_of_neutral_helium_lookup_table=self.density_lut,
            hydrogen_inflow_vector=self.hydrogen_inflow,
            helium_inflow_vector=self.helium_inflow,
            proton_results={},
        )

    def tearDown(self):
        _clear_shared()

    def test_clean_chunk_passes_through_fit_parameters_and_moment_helpers(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """On a clean chunk the fitter forwards the fit parameters and the moment-helper outputs verbatim, and the helium-channel effective-area scale lookup happens at the PUI chunk-center epoch."""
        fit_params = FittingParameters(1.5, 1e-7, 450.0, 0.3, int(SwapiL3Flags.NONE))
        mock_calculate_pickup_ion.return_value = PickupIonFitResult(
            fitting_params=fit_params,
            chunk_response=Mock(),
            vasyliunas_siscoe_distribution=Mock(),
        )
        density_result = ufloat(5.0, 0.5)
        temperature_result = ufloat(1e6, 1e5)
        mock_density.return_value = density_result
        mock_temperature.return_value = temperature_result

        result = self.fitter.fit_chunk(
            _pui_chunk(_EPOCH_TT2000),
            self.epoch,
            self.sw_velocity_rtn,
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.NONE),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            self.vasyliunas_siscoe_distribution,
        )

        self.assertEqual(result["epoch"], self.epoch)
        self.assertEqual(result["cooling_index"], 1.5)
        self.assertEqual(result["ionization_rate"], 1e-7)
        self.assertEqual(result["cutoff_speed"], 450.0)
        self.assertEqual(result["background_rate"], 0.3)
        self.assertIs(result["density"], density_result)
        self.assertIs(result["temperature"], temperature_result)
        self.assertEqual(result["quality_flags"], int(SwapiL3Flags.NONE))

        self.efficiency_table.central_effective_area_scale_for.assert_called_once_with(
            self.epoch, "helium"
        )
        pickup_kwargs = mock_calculate_pickup_ion.call_args.kwargs
        self.assertEqual(pickup_kwargs["central_effective_area_scale"], 0.42)

    def test_combines_pui_fit_flag_with_upstream_proton_flag(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """The output quality flag is the bitwise OR of the per-fit flag returned by `calculate_pickup_ion_values` and the upstream proton-SW quality flag."""
        nan = ufloat(np.nan, np.nan)
        mock_calculate_pickup_ion.return_value = PickupIonFitResult(
            fitting_params=FittingParameters(
                nan, nan, nan, nan, int(SwapiL3Flags.BAD_FIT)
            ),
            chunk_response=Mock(),
            vasyliunas_siscoe_distribution=Mock(),
        )

        result = self.fitter.fit_chunk(
            _pui_chunk(_EPOCH_TT2000),
            self.epoch,
            self.sw_velocity_rtn,
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.FIT_ERROR),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            self.vasyliunas_siscoe_distribution,
        )

        self.assertEqual(
            result["quality_flags"],
            int(SwapiL3Flags.BAD_FIT) | int(SwapiL3Flags.FIT_ERROR),
        )

    def test_bad_fit_skips_moment_helpers_and_fills_density_and_temperature(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """A `BAD_FIT` fit returns NaN-filled parameters, so the chunk fitter skips the moment helpers entirely (a NaN cutoff speed would otherwise crash them) and leaves density and temperature as NaN fill."""
        nan = ufloat(np.nan, np.nan)
        mock_calculate_pickup_ion.return_value = PickupIonFitResult(
            fitting_params=FittingParameters(
                nan, nan, nan, nan, int(SwapiL3Flags.BAD_FIT)
            ),
            chunk_response=Mock(),
            vasyliunas_siscoe_distribution=Mock(),
        )

        result = self.fitter.fit_chunk(
            _pui_chunk(_EPOCH_TT2000),
            self.epoch,
            self.sw_velocity_rtn,
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.NONE),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            self.vasyliunas_siscoe_distribution,
        )

        mock_density.assert_not_called()
        mock_temperature.assert_not_called()
        self.assertTrue(np.isnan(result["density"].nominal_value))
        self.assertTrue(np.isnan(result["temperature"].nominal_value))
        self.assertEqual(result["quality_flags"], int(SwapiL3Flags.BAD_FIT))

    def test_nan_in_count_rates_fills_outputs_and_skips_fit(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """A NaN in the coarse-bin count rate slice short-circuits the fit: every scalar output is NaN, no science helper is called, and the only flag in the result is the upstream proton flag."""
        count_rates = np.full((50, _N_BINS), 5.0)
        count_rates[3, 30] = np.nan
        chunk = _pui_chunk(_EPOCH_TT2000, count_rates=count_rates)

        result = self.fitter.fit_chunk(
            chunk,
            self.epoch,
            self.sw_velocity_rtn,
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.NONE),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            self.vasyliunas_siscoe_distribution,
        )

        mock_calculate_pickup_ion.assert_not_called()
        mock_density.assert_not_called()
        mock_temperature.assert_not_called()
        for key in ("cooling_index", "ionization_rate", "cutoff_speed",
                    "background_rate", "density", "temperature"):
            self.assertTrue(np.isnan(result[key].nominal_value), msg=key)
            self.assertTrue(np.isnan(result[key].std_dev), msg=key)
        self.assertEqual(result["quality_flags"], int(SwapiL3Flags.NONE))

    def test_nan_in_sw_velocity_fills_outputs_and_skips_fit(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """A NaN-valued RTN SW velocity (from upstream proton fit failure) short-circuits the fit the same way as a missing count rate."""
        result = self.fitter.fit_chunk(
            _pui_chunk(_EPOCH_TT2000),
            self.epoch,
            np.array([np.nan, np.nan, np.nan]),
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.NONE),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            self.vasyliunas_siscoe_distribution,
        )

        mock_calculate_pickup_ion.assert_not_called()
        for key in ("cooling_index", "ionization_rate", "cutoff_speed",
                    "background_rate", "density", "temperature"):
            self.assertTrue(np.isnan(result[key].nominal_value), msg=key)
        self.assertEqual(result["quality_flags"], int(SwapiL3Flags.NONE))

    def test_nan_vasyliunas_siscoe_distribution_none_fills_outputs_and_skips_fit(
        self, mock_calculate_pickup_ion, mock_density, mock_temperature
    ):
        """A `None` Vasyliunas-Siscoe distribution (SPICE gap building the chunk state, with finite velocity) short-circuits the fit just like a missing input."""
        result = self.fitter.fit_chunk(
            _pui_chunk(_EPOCH_TT2000),
            self.epoch,
            self.sw_velocity_rtn,
            self.bulk_sw_per_bin_swapi,
            int(SwapiL3Flags.NONE),
            self.lower_energy_cutoff,
            self.upper_energy_cutoff,
            None,
        )

        mock_calculate_pickup_ion.assert_not_called()
        for key in ("cooling_index", "ionization_rate", "cutoff_speed",
                    "background_rate", "density", "temperature"):
            self.assertTrue(np.isnan(result[key].nominal_value), msg=key)
        self.assertEqual(result["quality_flags"], int(SwapiL3Flags.NONE))


if __name__ == "__main__":
    unittest.main()
