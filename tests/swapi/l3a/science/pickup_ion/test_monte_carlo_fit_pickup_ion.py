"""MC parameter-recovery test for the PUI fit via the production
`PuiChunkFitter`.

Dispatches Poisson-resampled chunks through the production
`ParallelChunkRunner` → `PuiChunkFitter` path, so the test exercises the same
chunk fan-out, precompute-geometry SPICE chain, and per-chunk fit code that
the pipeline runs end-to-end.

Per-minute proton SW velocities are synthesized so that
`calculate_ten_minute_velocities` averages each chunk's 10-entry window back
to the truth velocity stored in the h5 fixture. The fitter therefore derives
its chunk geometry (RTN→IMAP_SWAPI rotation, energy cutoffs, Vasyliunas-
Siscoe distribution) from real production code instead of test-side stand-ins.
"""
import os
import tempfile
import unittest
from datetime import datetime
from unittest import skip

import h5py
import numpy as np
import spiceypy

from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_LIVETIME_S,
)
from imap_l3_processing.swapi.l3a.chunk_fits import (
    ParallelChunkRunner,
    PuiChunkFitter,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector
from imap_l3_processing.swapi.response.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.utils import SpiceKernelTypes, furnish_spice_metakernel
from tests.swapi._helpers import load_swapi_response
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

_REFERENCE_50SWEEP_H5_PATH = get_test_data_path(
    "swapi/pui_count_rate_reference_50sweep.h5"
)
_DENSITY_LUT_PATH = get_test_instrument_team_data_path(
    "swapi/density-of-neutral-helium-lut.dat"
)
_HYDROGEN_INFLOW_PATH = get_test_data_path(
    "swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
)
_HELIUM_INFLOW_PATH = get_test_data_path(
    "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
)

_MC_N_SAMPLES = 1000
_MC_BIAS_TOLERANCE = 0.03
_MC_SIGMA_TOLERANCE = 0.10

_SPICE_KERNEL_TYPES = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.IMAPFrames,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.EphemerisReconstructed,
    SpiceKernelTypes.PlanetaryEphemeris,
    SpiceKernelTypes.PlanetaryConstants,
]


@skip("temporarily disabled until PUI algorithm changes are fully implemented")
# @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
class MonteCarloFitPickupIonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Kernels stay furnished for the run. `PuiChunkFitter.precompute_geometry`
        # queries SPICE in the parent process; the PUI worker path
        # (`calculate_pickup_ion_values` and the moment helpers) does not, so
        # fork-state corruption of SPICE handles is not a concern.
        furnish_spice_metakernel(
            start_date=datetime(2026, 4, 25, 0, 0, 0),
            end_date=datetime(2026, 4, 25, 0, 11, 0),
            kernel_types=_SPICE_KERNEL_TYPES,
        )

    @classmethod
    def tearDownClass(cls):
        spiceypy.kclear()

    def test_recovers_parameter_means_and_uncertainties(self):
        with h5py.File(_REFERENCE_50SWEEP_H5_PATH, "r") as h5:
            voltage_v_ascending = h5["voltage_v"][...].astype(float)
            expected_rate_ascending = h5["expected_coincidence_rate_hz"][...].astype(
                float
            )
            energy_ev_ascending = h5["energy_ev"][...].astype(float)
            sci_start_time = h5["sci_start_time_tt2000_ns"][...].astype(np.int64)
            sw_velocity_rtn = np.array(h5.attrs["bulk_sw_rtn_kms"], dtype=float)

            cooling_index_truth = float(h5.attrs["cooling_index"])
            cutoff_speed_truth_kms = float(h5.attrs["cutoff_speed_kms"])
            ionization_rate_truth_hz = float(h5.attrs["ionization_rate_hz"])
            background_rate_truth_hz = float(h5.attrs["background_rate_hz"])
            helium_efficiency_ratio = float(h5.attrs["helium_efficiency_ratio"])

        n_sweeps = expected_rate_ascending.shape[0]

        # Production SWAPI sweeps go high-to-low voltage (descending). The h5
        # stores ascending order; flip along the bin axis.
        voltage_v_descending = voltage_v_ascending[::-1].copy()
        energy_ev_descending = energy_ev_ascending[::-1].copy()
        expected_rate_descending = expected_rate_ascending[:, ::-1].copy()

        # 72-wide energy template with the 62 coarse bins at indices 1:63.
        energy_full_template = np.zeros((n_sweeps, 72))
        energy_full_template[:, SWAPI_COARSE_SWEEP_BINS] = energy_ev_descending[
            np.newaxis, :
        ]
        expected_counts_coarse = np.maximum(
            expected_rate_descending * SWAPI_LIVETIME_S, 0.0
        )

        # One chunk per MC sample. Each gets a slightly bumped start time so
        # `PuiChunkFitter`'s per-epoch dict lookups stay per-chunk and the
        # runner can stack results in chunk order.
        chunks = []
        for seed in range(_MC_N_SAMPLES):
            mc_rng = np.random.default_rng(seed)
            observed_counts = mc_rng.poisson(expected_counts_coarse)
            observed_rate = observed_counts.astype(float) / SWAPI_LIVETIME_S
            rate_full = np.zeros((n_sweeps, 72))
            rate_full[:, SWAPI_COARSE_SWEEP_BINS] = observed_rate

            chunk_start_time = sci_start_time + (seed * 1_000_000)  # +1ms per seed
            chunks.append(
                SwapiL2Data(
                    sci_start_time=chunk_start_time,
                    energy=energy_full_template.copy(),
                    coincidence_count_rate=rate_full,
                    coincidence_count_rate_uncertainty=np.zeros_like(rate_full),
                )
            )

        # Per-minute proton SW velocities feeding `PuiChunkFitter`.
        # `calculate_ten_minute_velocities` averages 10 consecutive entries
        # into the 10-minute mean used per PUI chunk, so 10 copies of the
        # truth velocity per chunk recovers truth exactly. Quality flags are
        # all NONE.
        proton_velocities_per_minute = np.tile(
            sw_velocity_rtn, (_MC_N_SAMPLES * 10, 1)
        )
        proton_quality_flags_per_minute = [0] * (_MC_N_SAMPLES * 10)
        proton_results = {
            "proton_sw_velocity_rtn": proton_velocities_per_minute,
            "quality_flags": proton_quality_flags_per_minute,
        }

        # Efficiency table with alpha/proton ratio matching the h5 fixture.
        proton_eff = 0.02348
        alpha_eff = proton_eff * helium_efficiency_ratio
        efficiency_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        )
        efficiency_file.write(
            f"2000-01-01T11:00:00  0  {proton_eff:.10f}  {alpha_eff:.10f}\n"
        )
        efficiency_file.close()
        efficiency_table = EfficiencyCalibrationTable(efficiency_file.name)
        os.unlink(efficiency_file.name)

        swapi_response = load_swapi_response(warm_cache_voltages=voltage_v_descending)
        density_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(
            _DENSITY_LUT_PATH
        )
        hydrogen_inflow_vector = InflowVector.from_file(_HYDROGEN_INFLOW_PATH)
        helium_inflow_vector = InflowVector.from_file(_HELIUM_INFLOW_PATH)

        fitter = PuiChunkFitter(
            density_of_neutral_helium_lookup_table=density_lookup_table,
            hydrogen_inflow_vector=hydrogen_inflow_vector,
            helium_inflow_vector=helium_inflow_vector,
            proton_results=proton_results,
        )
        runner = ParallelChunkRunner(
            swapi_response=swapi_response, efficiency_table=efficiency_table
        )

        result = runner.run(chunks, fitter)

        cooling_index = result["cooling_index"]
        ionization_rate = result["ionization_rate"]
        cutoff_speed = result["cutoff_speed"]
        background_rate = result["background_rate"]

        nominal = np.array(
            [
                [u.nominal_value for u in cooling_index],
                [u.nominal_value for u in ionization_rate],
                [u.nominal_value for u in cutoff_speed],
                [u.nominal_value for u in background_rate],
            ]
        ).T  # (n_samples, 4)
        sigma = np.array(
            [
                [u.std_dev for u in cooling_index],
                [u.std_dev for u in ionization_rate],
                [u.std_dev for u in cutoff_speed],
                [u.std_dev for u in background_rate],
            ]
        ).T

        param_names = (
            "cooling_index",
            "ionization_rate",
            "cutoff_speed",
            "background_count_rate",
        )
        truth = np.array(
            [
                cooling_index_truth,
                ionization_rate_truth_hz,
                cutoff_speed_truth_kms,
                background_rate_truth_hz,
            ]
        )

        good = np.all(np.isfinite(nominal) & np.isfinite(sigma), axis=1)
        n_good = int(good.sum())
        self.assertGreater(
            n_good, _MC_N_SAMPLES // 2, "too many fits returned NaN"
        )

        for k, name in enumerate(param_names):
            mean_fit = float(np.mean(nominal[good, k]))
            std_fit = float(np.std(nominal[good, k], ddof=1))
            mean_sigma = float(np.mean(sigma[good, k]))
            rel_bias = (mean_fit - truth[k]) / truth[k]
            rel_sigma_error = (mean_sigma - std_fit) / std_fit
            self.assertLess(
                abs(rel_bias),
                _MC_BIAS_TOLERANCE,
                msg=(
                    f"{name}: mean(fit)={mean_fit:g} truth={truth[k]:g} "
                    f"relative bias={rel_bias:+.2%}"
                ),
            )
            self.assertLess(
                abs(rel_sigma_error),
                _MC_SIGMA_TOLERANCE,
                msg=(
                    f"{name}: mean(sigma)={mean_sigma:g} empirical "
                    f"std={std_fit:g} relative error={rel_sigma_error:+.2%}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
