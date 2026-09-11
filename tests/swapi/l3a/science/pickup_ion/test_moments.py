from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.integrate
import spacepy.pycdf
import spiceypy
from uncertainties import ufloat

import imap_l3_processing
from imap_l3_processing.constants import (
    CENTIMETERS_PER_METER,
    METERS_PER_KILOMETER,
    ONE_AU_IN_KM,
    ONE_SECOND_IN_NANOSECONDS,
)
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.science.pickup_ion.moments import (
    calculate_helium_pui_density,
    calculate_helium_pui_temperature,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
    build_vasyliunas_siscoe_distribution,
)
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import get_test_data_path


_CHUNK_GRID_POINTS = 256


def _build_moment_chunk_response(sw_speed_kms: float) -> ChunkCollapsedResponse:
    speed_in_sw_frame = np.linspace(1, sw_speed_kms * 1.2, _CHUNK_GRID_POINTS)
    bin_weights = np.zeros((1, 1, _CHUNK_GRID_POINTS))
    return ChunkCollapsedResponse(
        speed_in_sw_frame=speed_in_sw_frame, bin_weights=bin_weights
    )


def _quad_discontinuity_points(
    distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
    lut: DensityOfNeutralHeliumLookupTable,
) -> tuple[float, float, float]:
    radius_au = distribution.distance_km / ONE_AU_IN_KM
    lower = (lut.grid[1][0] / radius_au) ** (
        1.0 / SWAPI_PUI_COOLING_INDEX
    ) * fitting_params.cutoff_speed
    return (0.0, lower, fitting_params.cutoff_speed)


def _quad_density_reference(
    distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
    lut: DensityOfNeutralHeliumLookupTable,
) -> float:
    integral, _ = scipy.integrate.quad(
        lambda v: distribution.f(v, fitting_params) * v * v,
        0.0,
        fitting_params.cutoff_speed,
        points=_quad_discontinuity_points(distribution, fitting_params, lut),
        limit=100,
    )
    return 4 * np.pi * integral / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3


def _quad_temperature_reference(
    distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
    lut: DensityOfNeutralHeliumLookupTable,
) -> float:
    from imap_l3_processing.constants import (
        BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
        HE_PUI_PARTICLE_MASS_KG,
    )

    points = _quad_discontinuity_points(distribution, fitting_params, lut)
    numerator, _ = scipy.integrate.quad(
        lambda v: distribution.f(v, fitting_params) * v**4,
        0.0,
        fitting_params.cutoff_speed,
        points=points,
        limit=100,
    )
    denominator, _ = scipy.integrate.quad(
        lambda v: distribution.f(v, fitting_params) * v**2,
        0.0,
        fitting_params.cutoff_speed,
        points=points,
        limit=100,
    )
    return (
        HE_PUI_PARTICLE_MASS_KG
        / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
        * numerator
        / denominator
        * METERS_PER_KILOMETER**2
    )


class CalculatePuiDensityAndTemperatureTest(SpiceTestCase):
    def setUp(self) -> None:
        density_lut_path = (
            Path(imap_l3_processing.__file__).parent.parent
            / "tests"
            / "test_data"
            / "swapi"
            / "imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
        )
        self.density_of_neutral_helium_lookup_table = (
            DensityOfNeutralHeliumLookupTable.from_file(density_lut_path)
        )
        self.he_inflow_vector = InflowVector.from_file(
            get_test_data_path(
                "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
            )
        )

    def _build_distribution(
        self, epoch: int, sw_velocity_vector: np.ndarray
    ) -> VasyliunasSiscoeDistribution:
        return build_vasyliunas_siscoe_distribution(
            spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"),
            sw_velocity_vector,
            self.density_of_neutral_helium_lookup_table,
            self.he_inflow_vector,
        )

    def test_density_matches_scipy_quad_reference(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0.0, 0.0, -500.0])
        fitting_params = FittingParameters(1e-7, 520.0)

        distribution = self._build_distribution(epoch, sw_velocity_vector)
        chunk_response = _build_moment_chunk_response(
            float(np.linalg.norm(sw_velocity_vector))
        )

        expected = _quad_density_reference(
            distribution, fitting_params, self.density_of_neutral_helium_lookup_table
        )
        actual = calculate_helium_pui_density(
            chunk_response, distribution, fitting_params
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_density_propagates_uncertainty_from_ufloat_fit_params(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0.0, 0.0, -500.0])
        fitting_params = FittingParameters(
            ufloat(1e-7, 1e-8),
            ufloat(520, 5),
        )
        distribution = self._build_distribution(epoch, sw_velocity_vector)
        chunk_response = _build_moment_chunk_response(
            float(np.linalg.norm(sw_velocity_vector))
        )

        result = calculate_helium_pui_density(
            chunk_response, distribution, fitting_params
        )

        nominal_params = FittingParameters(1e-7, 520.0)
        expected_nominal = _quad_density_reference(
            distribution, nominal_params, self.density_of_neutral_helium_lookup_table
        )
        np.testing.assert_allclose(result.n, expected_nominal, rtol=1e-3)
        self.assertGreater(result.s, 0.0)

    def test_temperature_matches_scipy_quad_reference(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0.0, 0.0, -500.0])
        fitting_params = FittingParameters(1e-7, 500.0)

        distribution = self._build_distribution(epoch, sw_velocity_vector)
        chunk_response = _build_moment_chunk_response(
            float(np.linalg.norm(sw_velocity_vector))
        )

        expected = _quad_temperature_reference(
            distribution, fitting_params, self.density_of_neutral_helium_lookup_table
        )
        actual = calculate_helium_pui_temperature(
            chunk_response, distribution, fitting_params
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_temperature_propagates_uncertainty_from_ufloat_fit_params(self):
        epoch = spacepy.pycdf.lib.datetime_to_tt2000(datetime(2025, 6, 6, 12))
        sw_velocity_vector = np.array([0.0, 0.0, -500.0])
        fitting_params = FittingParameters(
            ufloat(1e-7, 1e-8),
            ufloat(500, 5),
        )
        distribution = self._build_distribution(epoch, sw_velocity_vector)
        chunk_response = _build_moment_chunk_response(
            float(np.linalg.norm(sw_velocity_vector))
        )

        result = calculate_helium_pui_temperature(
            chunk_response, distribution, fitting_params
        )

        nominal_params = FittingParameters(1e-7, 500.0)
        expected_nominal = _quad_temperature_reference(
            distribution, nominal_params, self.density_of_neutral_helium_lookup_table
        )
        np.testing.assert_allclose(result.n, expected_nominal, rtol=1e-3)
        self.assertGreater(result.s, 0.0)
