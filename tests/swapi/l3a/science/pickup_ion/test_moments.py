import unittest
from pathlib import Path

import numpy as np
import scipy.integrate
from uncertainties import ufloat

import imap_l3_processing
from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    CENTIMETERS_PER_METER,
    HE_PUI_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
    ONE_AU_IN_KM,
)
from imap_l3_processing.swapi.constants import SWAPI_PUI_COOLING_INDEX
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.moments import (
    calculate_helium_pui_density,
    calculate_helium_pui_temperature,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.uniform_speed_grid import (
    UniformSpeedGrid,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    _filled_shell_vdf_without_cutoff,
)

_SOLAR_WIND_SPEED_KMS = 500.0
_DISTANCE_KM = ONE_AU_IN_KM
_INFLOW_ANGLE_DEG = 75.0
_IONIZATION_RATE_HZ = 1e-7


def _speed_grid() -> UniformSpeedGrid:
    return UniformSpeedGrid(_SOLAR_WIND_SPEED_KMS * 1.2)


def _quad_discontinuity_points(
    cutoff_speed: float, lut: DensityOfNeutralHeliumLookupTable
) -> tuple[float, float, float]:
    radius_au = _DISTANCE_KM / ONE_AU_IN_KM
    lower = (lut.grid[1][0] / radius_au) ** (
        1.0 / SWAPI_PUI_COOLING_INDEX
    ) * cutoff_speed
    return (0.0, lower, cutoff_speed)


def _quad_density_reference(
    cutoff_speed: float, lut: DensityOfNeutralHeliumLookupTable
) -> float:
    integral, _ = scipy.integrate.quad(
        lambda v: _reference_vdf(v, cutoff_speed, lut) * v * v,
        0.0,
        cutoff_speed,
        points=_quad_discontinuity_points(cutoff_speed, lut),
        limit=100,
    )
    return 4 * np.pi * integral / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3


def _quad_temperature_reference(
    cutoff_speed: float, lut: DensityOfNeutralHeliumLookupTable
) -> float:
    points = _quad_discontinuity_points(cutoff_speed, lut)
    numerator, _ = scipy.integrate.quad(
        lambda v: _reference_vdf(v, cutoff_speed, lut) * v**4,
        0.0,
        cutoff_speed,
        points=points,
        limit=100,
    )
    denominator, _ = scipy.integrate.quad(
        lambda v: _reference_vdf(v, cutoff_speed, lut) * v**2,
        0.0,
        cutoff_speed,
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


def _reference_vdf(
    speed_in_sw_frame: float,
    cutoff_speed: float,
    lut: DensityOfNeutralHeliumLookupTable,
) -> float:
    return float(
        _filled_shell_vdf_without_cutoff(
            np.atleast_1d(float(speed_in_sw_frame)),
            ionization_rate=_IONIZATION_RATE_HZ,
            cutoff_speed=cutoff_speed,
            distance=_DISTANCE_KM,
            inflow_angle=_INFLOW_ANGLE_DEG,
            solar_wind_speed_inertial_frame=_SOLAR_WIND_SPEED_KMS,
            density_of_neutral_helium_lookup_table=lut,
        )[0]
    )


class CalculatePuiDensityAndTemperatureTest(unittest.TestCase):
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

    def _calculate_density(self, cutoff_speed):
        return calculate_helium_pui_density(
            _speed_grid(),
            ionization_rate=ufloat(_IONIZATION_RATE_HZ, 0.1 * _IONIZATION_RATE_HZ),
            cutoff_speed=cutoff_speed,
            distance=_DISTANCE_KM,
            inflow_angle=_INFLOW_ANGLE_DEG,
            solar_wind_speed_inertial_frame=_SOLAR_WIND_SPEED_KMS,
            density_of_neutral_helium_lookup_table=self.density_of_neutral_helium_lookup_table,
        )

    def _calculate_temperature(self, cutoff_speed):
        return calculate_helium_pui_temperature(
            _speed_grid(),
            ionization_rate=ufloat(_IONIZATION_RATE_HZ, 0.1 * _IONIZATION_RATE_HZ),
            cutoff_speed=cutoff_speed,
            distance=_DISTANCE_KM,
            inflow_angle=_INFLOW_ANGLE_DEG,
            solar_wind_speed_inertial_frame=_SOLAR_WIND_SPEED_KMS,
            density_of_neutral_helium_lookup_table=self.density_of_neutral_helium_lookup_table,
        )

    def test_density_matches_scipy_quad_reference_and_propagates_uncertainty(self):
        result = self._calculate_density(ufloat(520, 5))

        expected_nominal = _quad_density_reference(
            520.0, self.density_of_neutral_helium_lookup_table
        )
        np.testing.assert_allclose(result.n, expected_nominal, rtol=1e-3)
        self.assertGreater(result.s, 0.0)

    def test_temperature_matches_scipy_quad_reference_and_propagates_uncertainty(self):
        result = self._calculate_temperature(ufloat(500, 5))

        expected_nominal = _quad_temperature_reference(
            500.0, self.density_of_neutral_helium_lookup_table
        )
        np.testing.assert_allclose(result.n, expected_nominal, rtol=1e-3)
        self.assertGreater(result.s, 0.0)


if __name__ == "__main__":
    unittest.main()
