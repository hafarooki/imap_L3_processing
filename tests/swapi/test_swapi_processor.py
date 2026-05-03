"""Tests for `SwapiProcessor`.

These tests focus on the *processor's* responsibilities — descriptor dispatch,
assembly of per-chunk worker outputs into data-product dataclasses, and the
fill-value behavior on bad inputs. The numerics inside `ProtonChunkFitter`,
`AlphaChunkFitter`, `PuiProtonChunkFitter`, `calculate_pickup_ion_values`, and
the L3b VDF/flux helpers are covered by their own unit-test files; the
end-to-end real-fitter path (including subprocess + SPICE + CDF write) is
covered by `tests/integration/test_swapi_processor_integration.py`.

For each `process_l3a_*` method we therefore stub `ParallelChunkRunner.run` so
the assembly is exercised against a canned-dict result without paying for an LM
fit per chunk. The L3b method runs end-to-end against a small synthetic L2
sweep — the science there is closed-form and fast.
"""

import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from uncertainties import ufloat

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.l3a.models import (
    SwapiL2Data,
    SwapiL3AlphaSolarWindData,
    SwapiL3PickupIonData,
    SwapiL3ProtonSolarWindData,
)
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import FittingParameters
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


# Real fixtures — same set used by test_swapi_l3a_dependencies.py.
_L2_SCIENCE = get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")
_EFFICIENCY = get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")
_GEOMETRIC_FACTOR_PUI = get_test_data_path(
    "swapi/imap_swapi_energy-gf-pui-lut_20100101_v001.csv"
)
_GEOMETRIC_FACTOR_SW = get_test_data_path(
    "swapi/imap_swapi_energy-gf-sw-lut_20100101_v001.csv"
)
_INSTRUMENT_RESPONSE = get_test_data_path(
    "swapi/imap_swapi_instrument-response-lut_20241023_v000.zip"
)
_NEUTRAL_HELIUM = get_test_data_path(
    "swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
)
_HYDROGEN_INFLOW = get_test_data_path(
    "swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
)
_HELIUM_INFLOW = get_test_data_path(
    "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
)
_AZIMUTHAL_TRANSMISSION = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def _input_metadata(descriptor: str, data_level: str = "l3a") -> InputMetadata:
    return InputMetadata(
        instrument="swapi",
        data_level=data_level,
        start_date=datetime(2024, 9, 8),
        end_date=datetime(2024, 9, 9),
        version="v001",
        descriptor=descriptor,
    )


def _load_l3a_deps() -> SwapiL3ADependencies:
    return SwapiL3ADependencies.from_file_paths(
        _L2_SCIENCE,
        _EFFICIENCY,
        _GEOMETRIC_FACTOR_PUI,
        _INSTRUMENT_RESPONSE,
        _NEUTRAL_HELIUM,
        _HYDROGEN_INFLOW,
        _HELIUM_INFLOW,
        _AZIMUTHAL_TRANSMISSION,
        _CENTRAL_EFFECTIVE_AREA,
        _PASSBAND_FIT_COEFFICIENTS,
    )


def _load_l3b_deps() -> SwapiL3BDependencies:
    return SwapiL3BDependencies.from_file_paths(
        _L2_SCIENCE, _GEOMETRIC_FACTOR_SW, _EFFICIENCY
    )


# Canned dicts that mirror the real fitter outputs — assembled by ParallelChunkRunner.run
# into ndarray-of-shape (N, ...). The test only cares that `process_l3a_*` builds the
# right dataclass from this dict; the dict shape and key set are the contract.
def _proton_chunk_results(n_chunks: int = 10) -> dict:
    return {
        "epoch": np.arange(n_chunks, dtype=np.int64),
        "proton_sw_speed": np.full(n_chunks, 450.0),
        "proton_sw_speed_uncert": np.full(n_chunks, 0.5),
        "proton_sw_speed_sun": np.full(n_chunks, 480.0),
        "proton_sw_speed_sun_uncert": np.full(n_chunks, 0.5),
        "proton_sw_temperature": np.full(n_chunks, 50_000.0),
        "proton_sw_temperature_uncert": np.full(n_chunks, 1_000.0),
        "proton_sw_density": np.full(n_chunks, 5.0),
        "proton_sw_density_uncert": np.full(n_chunks, 0.1),
        "proton_sw_clock_angle": np.full(n_chunks, 90.0),
        "proton_sw_clock_angle_uncert": np.full(n_chunks, 1.0),
        "proton_sw_deflection_angle": np.full(n_chunks, 5.0),
        "proton_sw_deflection_angle_uncert": np.full(n_chunks, 0.1),
        "proton_sw_bulk_velocity_rtn_sun": np.tile([450.0, 0.0, 0.0], (n_chunks, 1)),
        "proton_sw_bulk_velocity_rtn_sun_covariance": np.tile(
            np.eye(3), (n_chunks, 1, 1)
        ),
        "proton_sw_bulk_velocity_rtn_sc": np.tile([450.0, 0.0, 0.0], (n_chunks, 1)),
        "proton_sw_bulk_velocity_rtn_sc_covariance": np.tile(
            np.eye(3), (n_chunks, 1, 1)
        ),
        "quality_flags": np.full(n_chunks, int(SwapiL3Flags.NONE), dtype=np.uint16),
    }


def _alpha_chunk_results(n_chunks: int = 10) -> dict:
    return {
        "epoch": np.arange(n_chunks, dtype=np.int64),
        "alpha_sw_density": np.full(n_chunks, 0.2),
        "alpha_sw_density_uncert": np.full(n_chunks, 0.01),
        "alpha_sw_temperature": np.full(n_chunks, 200_000.0),
        "alpha_sw_temperature_uncert": np.full(n_chunks, 5_000.0),
        "alpha_sw_velocity_rtn": np.tile([450.0, 0.0, 0.0], (n_chunks, 1)),
        "alpha_sw_velocity_covariance_rtn": np.tile(np.eye(3), (n_chunks, 1, 1)),
        "alpha_sw_delta_v": np.full(n_chunks, 30.0),
        "alpha_sw_delta_v_uncert": np.full(n_chunks, 1.0),
        "alpha_sw_b_hat_rtn": np.tile([1.0, 0.0, 0.0], (n_chunks, 1)),
        "alpha_sw_reference_proton_density": np.full(n_chunks, 5.0),
        "alpha_sw_reference_proton_temperature": np.full(n_chunks, 50_000.0),
        "alpha_sw_reference_proton_velocity_rtn": np.tile(
            [450.0, 0.0, 0.0], (n_chunks, 1)
        ),
        "bad_fit_flag": np.full(n_chunks, int(SwapiL3Flags.NONE), dtype=np.uint16),
    }


def _pui_proton_chunk_results(n_chunks: int = 10) -> dict:
    return {
        "proton_sw_speed": np.array([ufloat(450.0, 1.0)] * n_chunks, dtype=object),
        "proton_sw_clock_angle": np.array([ufloat(90.0, 1.0)] * n_chunks, dtype=object),
        "proton_sw_deflection_angle": np.array(
            [ufloat(5.0, 0.1)] * n_chunks, dtype=object
        ),
        "quality_flags": np.full(n_chunks, int(SwapiL3Flags.NONE), dtype=np.uint16),
    }


class TestProcessDispatch(unittest.TestCase):
    """`SwapiProcessor.process` dispatches on `(data_level, descriptor)`."""

    def setUp(self):
        self.collection = MagicMock(spec=ProcessingInputCollection)

    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch.object(SwapiL3ADependencies, "fetch_dependencies")
    def test_process_dispatches_to_proton_for_proton_sw(
        self, fetch_deps, save_data_mock
    ):
        fetch_deps.return_value = MagicMock()
        save_data_mock.return_value = sentinel.cdf_path
        proc = SwapiProcessor(self.collection, _input_metadata("proton-sw"))
        with (
            patch.object(
                proc, "process_l3a_proton", return_value=MagicMock()
            ) as proton,
            patch.object(proc, "process_l3a_alpha") as alpha,
            patch.object(proc, "process_l3a_pui") as pui,
            patch.object(proc, "get_parent_file_names", return_value=[]),
        ):
            paths = proc.process()
        proton.assert_called_once()
        alpha.assert_not_called()
        pui.assert_not_called()
        self.assertEqual(paths, [sentinel.cdf_path])

    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch.object(SwapiL3ADependencies, "fetch_dependencies")
    def test_process_dispatches_to_alpha_for_alpha_sw(self, fetch_deps, save_data_mock):
        fetch_deps.return_value = MagicMock()
        save_data_mock.return_value = sentinel.cdf_path
        proc = SwapiProcessor(self.collection, _input_metadata("alpha-sw"))
        with (
            patch.object(proc, "process_l3a_proton") as proton,
            patch.object(proc, "process_l3a_alpha", return_value=MagicMock()) as alpha,
            patch.object(proc, "process_l3a_pui") as pui,
            patch.object(proc, "get_parent_file_names", return_value=[]),
        ):
            proc.process()
        proton.assert_not_called()
        alpha.assert_called_once()
        pui.assert_not_called()

    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch.object(SwapiL3ADependencies, "fetch_dependencies")
    def test_process_dispatches_to_pui_for_pui_he(self, fetch_deps, save_data_mock):
        fetch_deps.return_value = MagicMock()
        save_data_mock.return_value = sentinel.cdf_path
        proc = SwapiProcessor(self.collection, _input_metadata("pui-he"))
        with (
            patch.object(proc, "process_l3a_proton") as proton,
            patch.object(proc, "process_l3a_alpha") as alpha,
            patch.object(proc, "process_l3a_pui", return_value=MagicMock()) as pui,
            patch.object(proc, "get_parent_file_names", return_value=[]),
        ):
            proc.process()
        proton.assert_not_called()
        alpha.assert_not_called()
        pui.assert_called_once()

    @patch.object(SwapiL3ADependencies, "fetch_dependencies")
    def test_process_raises_for_unknown_l3a_descriptor(self, fetch_deps):
        fetch_deps.return_value = MagicMock()
        proc = SwapiProcessor(self.collection, _input_metadata("unknown-descriptor"))
        with self.assertRaises(NotImplementedError):
            proc.process()

    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch.object(SwapiL3BDependencies, "fetch_dependencies")
    def test_process_dispatches_to_l3b_for_l3b_data_level(
        self, fetch_deps, save_data_mock
    ):
        fetch_deps.return_value = MagicMock()
        save_data_mock.return_value = sentinel.cdf_path
        proc = SwapiProcessor(self.collection, _input_metadata("combined", "l3b"))
        with (
            patch.object(proc, "process_l3b", return_value=MagicMock()) as l3b,
            patch.object(proc, "get_parent_file_names", return_value=[]),
        ):
            paths = proc.process()
        l3b.assert_called_once()
        self.assertEqual(paths, [sentinel.cdf_path])

    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch.object(SwapiL3ADependencies, "fetch_dependencies")
    def test_process_attaches_parent_files_before_save(
        self, fetch_deps, save_data_mock
    ):
        fetch_deps.return_value = MagicMock()
        save_data_mock.return_value = sentinel.cdf_path
        proc = SwapiProcessor(self.collection, _input_metadata("proton-sw"))
        product = MagicMock()
        with (
            patch.object(proc, "process_l3a_proton", return_value=product),
            patch.object(
                proc,
                "get_parent_file_names",
                return_value=["parent_a.cdf", "parent_b.cdf"],
            ),
        ):
            proc.process()
        self.assertEqual(product.parent_file_names, ["parent_a.cdf", "parent_b.cdf"])
        save_data_mock.assert_called_once_with(product)


class _L3ADepsTestCase(unittest.TestCase):
    """Base class — load `SwapiL3ADependencies` once for all subclass tests."""

    @classmethod
    def setUpClass(cls):
        cls.deps = _load_l3a_deps()
        cls.collection = MagicMock(spec=ProcessingInputCollection)


class TestProcessL3AProtonAssembly(_L3ADepsTestCase):
    """`process_l3a_proton` chunks the L2 data, runs the parallel fitter, and packs
    the results into a `SwapiL3ProtonSolarWindData`. We stub the runner so the
    assembly is exercised in isolation."""

    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_returns_proton_data_product(self, runner_class):
        # 50 sweeps / 5 per chunk = 10 chunks.
        runner_class.return_value.run.return_value = _proton_chunk_results(10)
        proc = SwapiProcessor(self.collection, _input_metadata("proton-sw"))

        product = proc.process_l3a_proton(self.deps.data, self.deps)

        self.assertIsInstance(product, SwapiL3ProtonSolarWindData)
        self.assertEqual(product.input_metadata.descriptor, "proton-sw")
        self.assertEqual(product.epoch.shape, (10,))
        self.assertEqual(product.proton_sw_speed.shape, (10,))
        np.testing.assert_array_equal(product.proton_sw_speed, np.full(10, 450.0))
        # Original input metadata is not mutated by `replace(...)`.
        self.assertEqual(proc.input_metadata.descriptor, "proton-sw")

    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_warms_response_cache_before_dispatch(self, runner_class):
        runner_class.return_value.run.return_value = _proton_chunk_results(10)
        proc = SwapiProcessor(self.collection, _input_metadata("proton-sw"))
        with patch.object(self.deps.swapi_response, "warm_cache") as warm:
            proc.process_l3a_proton(self.deps.data, self.deps)
        # warm_cache is called with the unique ESA voltages — i.e. all energies / k.
        warm.assert_called_once()
        (arg,) = warm.call_args.args
        self.assertEqual(arg.shape, self.deps.data.energy.shape)


class TestProcessL3AAlphaAssembly(_L3ADepsTestCase):
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_returns_alpha_data_product(self, runner_class):
        runner_class.return_value.run.return_value = _alpha_chunk_results(10)
        proc = SwapiProcessor(self.collection, _input_metadata("alpha-sw"))

        product = proc.process_l3a_alpha(self.deps.data, self.deps)

        self.assertIsInstance(product, SwapiL3AlphaSolarWindData)
        self.assertEqual(product.input_metadata.descriptor, "alpha-sw")
        self.assertEqual(product.epoch.shape, (10,))
        np.testing.assert_array_equal(product.alpha_sw_density, np.full(10, 0.2))

    @patch("imap_l3_processing.swapi.swapi_processor.AlphaChunkFitter")
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_passes_mag_data_to_alpha_chunk_fitter(self, runner_class, fitter_class):
        runner_class.return_value.run.return_value = _alpha_chunk_results(10)
        deps = replace(self.deps, mag_l1d_data=sentinel.mag_data)
        proc = SwapiProcessor(self.collection, _input_metadata("alpha-sw"))

        proc.process_l3a_alpha(deps.data, deps)

        fitter_class.assert_called_once_with(sentinel.mag_data)


class TestProcessL3APuiAssembly(_L3ADepsTestCase):
    """`process_l3a_pui` runs the proton chunk fitter, derives 10-minute-averaged
    velocities, then loops over 50-sweep chunks calling `calculate_pickup_ion_values`
    and the helium PUI density/temperature helpers. We stub the heavy components and
    verify that fill values, flag composition, and dataclass shape are correct."""

    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values")
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_returns_pui_data_product_with_finite_values(
        self, runner_class, pui_values, density_fn, temp_fn
    ):
        runner_class.return_value.run.return_value = _pui_proton_chunk_results(10)
        pui_values.return_value = FittingParameters(
            cooling_index=ufloat(1.5, 0.1),
            ionization_rate=ufloat(1e-7, 1e-9),
            cutoff_speed=ufloat(500.0, 5.0),
            background_count_rate=ufloat(0.1, 0.01),
            flags=SwapiL3Flags.NONE,
        )
        density_fn.return_value = ufloat(1e-4, 1e-6)
        temp_fn.return_value = ufloat(1e6, 1e3)

        proc = SwapiProcessor(self.collection, _input_metadata("pui-he"))
        product = proc.process_l3a_pui(self.deps.data, self.deps)

        self.assertIsInstance(product, SwapiL3PickupIonData)
        self.assertEqual(product.input_metadata.descriptor, "pui-he")
        # 50 sweeps / 50 per pui-chunk = 1 entry; ten-minute averaging also
        # produces one velocity. Verify shape consistency.
        self.assertEqual(product.epoch.shape, product.cooling_index.shape)
        self.assertGreater(len(product.epoch), 0)
        # Each entry should be the canned ufloat (not a fill-NaN), since
        # calculate_pickup_ion_values + density/temp succeeded.
        for entry in product.cooling_index:
            self.assertEqual(entry.nominal_value, 1.5)
        for entry in product.density:
            self.assertEqual(entry.nominal_value, 1e-4)

    @patch("imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values")
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_outputs_fill_when_input_count_rates_have_nan(
        self, runner_class, pui_values
    ):
        runner_class.return_value.run.return_value = _pui_proton_chunk_results(10)
        # Inject NaN into the L2 chunk so the fill-value branch trips.
        broken_data = replace(
            self.deps.data,
            coincidence_count_rate=np.full_like(
                self.deps.data.coincidence_count_rate, np.nan
            ),
        )
        proc = SwapiProcessor(self.collection, _input_metadata("pui-he"))

        product = proc.process_l3a_pui(broken_data, self.deps)

        # Bypass branch raises → cooling_index stays NaN; pui_values is never called.
        pui_values.assert_not_called()
        self.assertTrue(np.isnan(product.cooling_index[0].nominal_value))
        self.assertTrue(np.isnan(product.density[0].nominal_value))
        self.assertTrue(np.isnan(product.temperature[0].nominal_value))

    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values")
    @patch("imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner")
    def test_or_combines_proton_and_pui_quality_flags(
        self, runner_class, pui_values, density_fn, temp_fn
    ):
        proton_results = _pui_proton_chunk_results(10)
        proton_results["quality_flags"] = np.full(
            10, int(SwapiL3Flags.STALE_PROTON), dtype=np.uint16
        )
        runner_class.return_value.run.return_value = proton_results
        pui_values.return_value = FittingParameters(
            cooling_index=ufloat(1.5, 0.1),
            ionization_rate=ufloat(1e-7, 1e-9),
            cutoff_speed=ufloat(500.0, 5.0),
            background_count_rate=ufloat(0.1, 0.01),
            flags=SwapiL3Flags.BAD_FIT,
        )
        density_fn.return_value = ufloat(1e-4, 1e-6)
        temp_fn.return_value = ufloat(1e6, 1e3)

        proc = SwapiProcessor(self.collection, _input_metadata("pui-he"))
        product = proc.process_l3a_pui(self.deps.data, self.deps)

        # OR of STALE_PROTON (proton-fit-derived 10-minute flag) and BAD_FIT
        # (pui fit flag) is a single value covering both bits.
        expected = int(SwapiL3Flags.STALE_PROTON) | int(SwapiL3Flags.BAD_FIT)
        np.testing.assert_array_equal(
            product.quality_flags, np.full_like(product.quality_flags, expected)
        )


class TestProcessL3B(unittest.TestCase):
    """`process_l3b` runs end-to-end against a small synthetic L2 input — the L3b
    science is closed-form (VDF arithmetic) and finishes in milliseconds, so we
    don't stub the ChunkRunner here."""

    @classmethod
    def setUpClass(cls):
        cls.deps = _load_l3b_deps()
        cls.collection = MagicMock(spec=ProcessingInputCollection)

    def test_assembles_combined_vdf_with_expected_shapes(self):
        proc = SwapiProcessor(self.collection, _input_metadata("combined", "l3b"))

        product = proc.process_l3b(self.deps.data, self.deps)

        self.assertIsInstance(product, SwapiL3BCombinedVDF)
        self.assertEqual(product.input_metadata.descriptor, "combined")
        # 50 sweeps / 50-per-chunk = 1 epoch.
        self.assertEqual(product.epoch.shape, (1,))
        # 62 coarse-sweep bins per epoch.
        self.assertEqual(product.proton_sw_velocities.shape, (1, 62))
        self.assertEqual(product.alpha_sw_velocities.shape, (1, 62))
        self.assertEqual(product.pui_sw_velocities.shape, (1, 62))
        self.assertEqual(product.combined_energy.shape, (1, 62))
        # Probabilities are uarray (object) — values must be finite/positive on
        # a real spectrum.
        from uncertainties.unumpy import nominal_values

        self.assertTrue(
            np.all(np.isfinite(nominal_values(product.proton_sw_combined_vdf)))
        )

    def test_epoch_is_first_sci_start_plus_five_minutes(self):
        proc = SwapiProcessor(self.collection, _input_metadata("combined", "l3b"))

        product = proc.process_l3b(self.deps.data, self.deps)

        expected = self.deps.data.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
        self.assertEqual(product.epoch[0], expected)


if __name__ == "__main__":
    unittest.main()
