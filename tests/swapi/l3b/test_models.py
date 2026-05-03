from datetime import datetime

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swapi.l3a.models import (
    EPOCH_CDF_VAR_NAME,
    EPOCH_DELTA_CDF_VAR_NAME,
)
from imap_l3_processing.swapi.l3b.models import (
    SwapiL3BCombinedVDF,
    COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME,
    COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME,
    SOLAR_WIND_ENERGY_CDF_VAR_NAME,
    SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME,
    SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME,
)
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_combined_vdf_data_products(self):
        input_metadata = UpstreamDataDependency(
            "swapi", "l3b", datetime(2024, 9, 8), datetime(2024, 9, 9), "v001", ""
        )
        epoch = np.array([1, 2, 3])
        epoch_delta = np.full_like(epoch, FIVE_MINUTES_IN_NANOSECONDS)
        proton_velocities = np.array([4, 5, 6])
        proton_velocities_delta_plus = np.array([0.4, 0.5, 0.6])
        proton_velocities_delta_minus = 1 + np.array([0.4, 0.5, 0.6])
        proton_vdf = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
        proton_vdf_uncertainties = np.array(
            [[0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]]
        )
        alpha_velocities = np.array([11, 12, 13])
        alpha_velocities_delta_plus = np.array([0.11, 0.12, 0.13])
        alpha_velocities_delta_minus = 1 + np.array([0.11, 0.12, 0.13])
        alpha_vdf = np.array([[14, 15, 16], [17, 18, 19], [20, 21, 22]])
        alpha_vdf_uncertainties = np.array(
            [[0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]]
        )
        pui_velocities = np.array([23, 24, 25])
        pui_velocities_delta_plus = np.array([0.23, 0.24, 0.25])
        pui_velocities_delta_minus = 1 + np.array([0.23, 0.24, 0.25])
        pui_vdf = np.array([[26, 27, 28], [29, 30, 31], [32, 33, 34]])
        pui_vdf_uncertainties = np.array(
            [[0.72, 0.8, 0.9], [0.10, 0.121, 0.12], [0.13, 0.142, 0.15]]
        )
        combined_energies = np.array([230, 240, 250])
        combined_energies_delta_plus = np.array([44, 55, 66])
        combined_energies_delta_minus = 1 + np.array([44, 55, 66])
        combined_differential_flux = np.array(
            [[26, 27.2, 28], [29.2, 30, 31], [32, 33.5, 34]]
        )
        combined_differential_flux_uncertainties = np.array(
            [[0.725, 0.8, 0.9], [0.105, 0.121, 0.124], [0.13, 0.1425, 0.15]]
        )

        vdf = SwapiL3BCombinedVDF(
            input_metadata=input_metadata,
            epoch=epoch,
            proton_sw_velocities=proton_velocities,
            proton_sw_velocities_delta_minus=proton_velocities_delta_minus,
            proton_sw_velocities_delta_plus=proton_velocities_delta_plus,
            proton_sw_combined_vdf=uarray(proton_vdf, proton_vdf_uncertainties),
            alpha_sw_velocities=alpha_velocities,
            alpha_sw_velocities_delta_minus=alpha_velocities_delta_minus,
            alpha_sw_velocities_delta_plus=alpha_velocities_delta_plus,
            alpha_sw_combined_vdf=uarray(alpha_vdf, alpha_vdf_uncertainties),
            pui_sw_velocities=pui_velocities,
            pui_sw_velocities_delta_minus=pui_velocities_delta_minus,
            pui_sw_velocities_delta_plus=pui_velocities_delta_plus,
            pui_sw_combined_vdf=uarray(pui_vdf, pui_vdf_uncertainties),
            combined_energy=combined_energies,
            combined_energy_delta_minus=combined_energies_delta_minus,
            combined_energy_delta_plus=combined_energies_delta_plus,
            combined_differential_flux=uarray(
                combined_differential_flux, combined_differential_flux_uncertainties
            ),
        )

        variables = vdf.to_data_product_variables()

        self.assertEqual(7, len(variables))
        self.assert_variable_attributes(
            variables[0], epoch, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000
        )
        self.assert_variable_attributes(
            variables[1], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME
        )

        self.assert_variable_attributes(
            variables[2], combined_energies, SOLAR_WIND_ENERGY_CDF_VAR_NAME
        )
        self.assert_variable_attributes(
            variables[3],
            combined_energies_delta_minus,
            SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[4],
            combined_energies_delta_plus,
            SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[5],
            combined_differential_flux,
            COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME,
        )
        self.assert_variable_attributes(
            variables[6],
            combined_differential_flux_uncertainties,
            COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME,
        )

    def test_combined_vdf_uses_real_calculate_delta_minus_plus_for_round_trip(self):
        """The L3b processor populates `delta_minus`/`delta_plus` via
        `calculate_delta_minus_plus`. Verify the dataclass round-trips those
        fields without mutating them so the bin edges stay consistent."""
        from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import (
            calculate_delta_minus_plus,
        )

        velocities = np.array([100.0, 200.0, 400.0])
        deltas = calculate_delta_minus_plus(velocities)
        # Bin edges: [v - delta_minus, v + delta_plus]. For a geometric series, the
        # right edge of bin i should equal the left edge of bin i+1.
        right_edges = velocities + deltas.delta_plus
        left_edges = velocities - deltas.delta_minus
        np.testing.assert_allclose(right_edges[:-1], left_edges[1:], rtol=1e-12)

        vdf = SwapiL3BCombinedVDF(
            input_metadata=UpstreamDataDependency(
                "swapi", "l3b", datetime(2024, 9, 8), datetime(2024, 9, 9), "v001", ""
            ),
            epoch=np.array([1]),
            proton_sw_velocities=velocities,
            proton_sw_velocities_delta_minus=deltas.delta_minus,
            proton_sw_velocities_delta_plus=deltas.delta_plus,
            proton_sw_combined_vdf=np.zeros((1, 3)),
            alpha_sw_velocities=velocities,
            alpha_sw_velocities_delta_minus=deltas.delta_minus,
            alpha_sw_velocities_delta_plus=deltas.delta_plus,
            alpha_sw_combined_vdf=np.zeros((1, 3)),
            pui_sw_velocities=velocities,
            pui_sw_velocities_delta_minus=deltas.delta_minus,
            pui_sw_velocities_delta_plus=deltas.delta_plus,
            pui_sw_combined_vdf=np.zeros((1, 3)),
            combined_energy=velocities,
            combined_energy_delta_minus=deltas.delta_minus,
            combined_energy_delta_plus=deltas.delta_plus,
            combined_differential_flux=np.zeros((1, 3)),
        )
        # The data class doesn't transform deltas — confirm the round-trip is identity.
        np.testing.assert_array_equal(
            vdf.combined_energy_delta_minus, deltas.delta_minus
        )
        np.testing.assert_array_equal(vdf.combined_energy_delta_plus, deltas.delta_plus)
