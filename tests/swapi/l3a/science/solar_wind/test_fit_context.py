import unittest
from unittest.mock import MagicMock

import numpy as np

from imap_l3_processing.constants import ALPHA_PARTICLE_MASS_KG, PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid
from imap_l3_processing.swapi.species import Species
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000


def _swapi_response_returning_per_voltage_response_grid():
    """Mock SwapiResponse whose `get_response_grid(t, v, species)` returns a real
    `ResponseGrid` NamedTuple tagged with the voltage in its `central_speed`
    field — lets tests assert on order and per-sweep identity without fighting
    numba's typed-list element-type inference (which rejects MagicMock)."""
    response = MagicMock()

    def _build_response_grid(time_as_tt2000, voltage, species):
        # Tag the voltage in `central_speed` so tests can recover which
        # voltage produced which grid. Other fields use shape-correct
        # placeholders that satisfy numba's type system.
        return ResponseGrid(
            sg_passband=None,
            oa_passband=None,
            central_speed=float(voltage),
            central_effective_area=0.0,
            azimuthal_transmission=AzimuthalTransmissionGrid(
                values=np.zeros(1), spacing=0.1
            ),
        )

    response.get_response_grid.side_effect = _build_response_grid
    return response


class TestSolarWindFitContextSubset(unittest.TestCase):
    """Tests for `SolarWindFitContext.subset` — selects the named bin columns from every sweep, preserving the (n_sweeps, n_kept_bins) 2D layout of `count_rate`/`esa_voltage` while expanding the bin indices to the flat sweep-major positions used by `response_grids` and `rotation_matrices`."""

    def setUp(self):
        # 2 sweeps × 3 bins, with distinguishable values per (sweep, bin)
        # so the subset operation's axis handling is observable. The
        # "rotation matrices" here are not real rotations (det ≠ 1) —
        # they're just stand-ins keyed by their flat sweep-major bin-minor
        # position so picks are easy to verify.
        self.full_ctx = SolarWindFitContext(
            count_rate=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            esa_voltage=np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]),
            response_grids=[
                "s0b0", "s0b1", "s0b2",
                "s1b0", "s1b1", "s1b2",
            ],
            rotation_matrices=np.stack(
                [np.eye(3) * (i + 1) for i in range(6)]
            ),
            mass_kg=PROTON_MASS_KG,
        )

    def test_subset_with_all_bin_indices_returns_equivalent_context(self):
        """Subsetting with every bin index reproduces all per-bin arrays unchanged."""
        kept = self.full_ctx.subset(np.array([0, 1, 2]))
        np.testing.assert_array_equal(kept.count_rate, self.full_ctx.count_rate)
        np.testing.assert_array_equal(kept.esa_voltage, self.full_ctx.esa_voltage)
        # `subset` rebuilds response_grids as a numba.typed.List; cast to a
        # plain list for value-level comparison.
        self.assertEqual(list(kept.response_grids), self.full_ctx.response_grids)
        np.testing.assert_array_equal(
            kept.rotation_matrices, self.full_ctx.rotation_matrices
        )

    def test_subset_picks_bin_axis_at_the_given_indices_preserving_sweeps(self):
        """Selecting bin indices [0, 2] yields a context with those two bin columns from each sweep; count_rate and esa_voltage stay 2D as (n_sweeps, n_kept_bins)."""
        kept = self.full_ctx.subset(np.array([0, 2]))
        np.testing.assert_array_equal(
            kept.count_rate, [[10.0, 30.0], [40.0, 60.0]]
        )
        np.testing.assert_array_equal(
            kept.esa_voltage, [[100.0, 300.0], [400.0, 600.0]]
        )
        # Flat sweep-major bin-minor order: s0b0, s0b2, s1b0, s1b2
        self.assertEqual(
            list(kept.response_grids), ["s0b0", "s0b2", "s1b0", "s1b2"]
        )
        np.testing.assert_array_equal(
            kept.rotation_matrices,
            self.full_ctx.rotation_matrices[[0, 2, 3, 5]],
        )

    def test_subset_preserves_mass(self):
        """Subsetting to a single bin leaves the scalar `mass_kg` field untouched."""
        kept = self.full_ctx.subset(np.array([0]))
        self.assertEqual(kept.mass_kg, PROTON_MASS_KG)


class TestBuildSolarWindFitContext(unittest.TestCase):
    """Tests for `build_solar_wind_fit_context` — rejects invalid voltages and materializes one `ResponseGrid` per bin in input order."""

    def _call_factory(self, *, count_rate, esa_voltage, rotation_matrices=None):
        response = _swapi_response_returning_per_voltage_response_grid()
        if rotation_matrices is None:
            rotation_matrices = np.stack([np.eye(3)] * len(count_rate))
        ctx = build_solar_wind_fit_context(
            count_rate=count_rate,
            esa_voltage=esa_voltage,
            swapi_response=response,
            time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
            species=Species.PROTON,
            rotation_matrices=rotation_matrices,
        )
        return ctx, response

    def test_keeps_all_inputs_unchanged_when_voltages_are_finite_and_positive(self):
        """When every voltage is finite and positive, the context preserves count_rate and esa_voltage verbatim and builds one ResponseGrid per bin."""
        count_rate = np.array([10.0, 20.0, 30.0])
        esa_voltage = np.array([100.0, 200.0, 300.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        np.testing.assert_array_equal(ctx.count_rate, count_rate)
        np.testing.assert_array_equal(ctx.esa_voltage, esa_voltage)
        self.assertEqual(len(ctx.response_grids), 3)

    def test_raises_on_zero_or_negative_voltage(self):
        """Any voltage <= 0 raises `ValueError`; the caller is expected to drop invalid bins (and the matching count_rate / rotation_matrices entries) before building the context, since the right pre-filter depends on whether the downstream fit needs per-sweep shape preserved."""
        with self.assertRaises(ValueError):
            self._call_factory(
                count_rate=np.array([10.0, 20.0]),
                esa_voltage=np.array([100.0, 0.0]),
            )
        with self.assertRaises(ValueError):
            self._call_factory(
                count_rate=np.array([10.0, 20.0]),
                esa_voltage=np.array([100.0, -50.0]),
            )

    def test_raises_on_non_finite_voltage(self):
        """NaN or inf voltages raise `ValueError` for the same reason as the <=0 case — the response-grid factory can't sensibly evaluate at those voltages, and the caller must decide how to handle them."""
        with self.assertRaises(ValueError):
            self._call_factory(
                count_rate=np.array([10.0, 20.0]),
                esa_voltage=np.array([100.0, np.nan]),
            )
        with self.assertRaises(ValueError):
            self._call_factory(
                count_rate=np.array([10.0, 20.0]),
                esa_voltage=np.array([100.0, np.inf]),
            )

    def test_preserves_input_shape_when_count_rate_is_2d(self):
        """A 2D count_rate (n_sweeps, n_bins) is stored as-is so the alpha path's per-sweep aggregations keep working — `build_solar_wind_fit_context` does not reshape its inputs."""
        count_rate = np.array([[10.0, 20.0], [30.0, 40.0]])
        esa_voltage = np.array([[100.0, 200.0], [300.0, 400.0]])
        rotations = np.stack([np.eye(3)] * 4)
        ctx, _ = self._call_factory(
            count_rate=count_rate,
            esa_voltage=esa_voltage,
            rotation_matrices=rotations,
        )
        self.assertEqual(ctx.count_rate.shape, (2, 2))
        self.assertEqual(ctx.esa_voltage.shape, (2, 2))
        self.assertEqual(len(ctx.response_grids), 4)

    def test_creates_response_grid_for_each_bin_in_input_order(self):
        """The factory calls `get_response_grid` once per voltage (in ravel order) and stores the resulting grids in the same order."""
        count_rate = np.array([10.0, 20.0])
        esa_voltage = np.array([100.0, 200.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        self.assertEqual(response.get_response_grid.call_count, 2)
        self.assertEqual(ctx.response_grids[0].central_speed, 100.0)
        self.assertEqual(ctx.response_grids[1].central_speed, 200.0)

    def test_passes_epoch_and_species_to_create_response_grid(self):
        """The epoch and species supplied to the factory are forwarded into the `get_response_grid` call for each bin, so the response picks up the right mass-per-charge and relative efficiency."""
        response = _swapi_response_returning_per_voltage_response_grid()
        ctx = build_solar_wind_fit_context(
            count_rate=np.array([10.0]),
            esa_voltage=np.array([100.0]),
            swapi_response=response,
            time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
            species=Species.ALPHA,
            rotation_matrices=np.eye(3)[np.newaxis],
        )
        self.assertEqual(response.get_response_grid.call_count, 1)
        call = response.get_response_grid.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        self.assertIn(NOMINAL_TEST_EPOCH_TT2000, all_args)
        self.assertIn(100.0, all_args)
        self.assertIn(Species.ALPHA, all_args)
        self.assertEqual(ctx.mass_kg, ALPHA_PARTICLE_MASS_KG)


if __name__ == "__main__":
    unittest.main()
