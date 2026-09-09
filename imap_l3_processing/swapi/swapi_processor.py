import logging
from dataclasses import replace

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitter,
    ParallelChunkRunner,
    ProtonChunkFitter,
    PuiChunkFitter,
)
from imap_l3_processing.swapi.l3a.models import (
    SwapiL3ProtonSolarWindData,
    SwapiL3AlphaSolarWindData,
    SwapiL3PickupIonData,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import (
    chunk_l2_data,
)
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import (
    calculate_combined_solar_wind_differential_flux,
)
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import (
    calculate_proton_solar_wind_vdf,
    calculate_alpha_solar_wind_vdf,
    calculate_pui_solar_wind_vdf,
    calculate_delta_minus_plus,
)
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.utils import save_data

logger = logging.getLogger(__name__)


class SwapiProcessor(Processor):
    def __init__(
        self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata
    ):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = SwapiL3ADependencies.fetch_dependencies(
                self.dependencies
            )

            if self.input_metadata.descriptor == "proton-sw":
                data = self.process_l3a_proton(l3a_dependencies.data, l3a_dependencies)
            elif self.input_metadata.descriptor == "alpha-sw":
                data = self.process_l3a_alpha(l3a_dependencies.data, l3a_dependencies)
            elif self.input_metadata.descriptor == "pui-he":
                data = self.process_l3a_pui(l3a_dependencies.data, l3a_dependencies)
            else:
                raise NotImplementedError(
                    "unknown descriptor", self.input_metadata.descriptor
                )
            data.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(data)
            return [cdf_path]
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(
                self.dependencies
            )
            l3b_combined_vdf = self.process_l3b(l3b_dependencies.data, l3b_dependencies)
            l3b_combined_vdf.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(l3b_combined_vdf)
            return [cdf_path]

    def process_l3a_proton(self, data, dependencies) -> SwapiL3ProtonSolarWindData:
        chunks = list(chunk_l2_data(data, 5))
        dependencies.swapi_response.warm_cache(data.energy / SWAPI_L2_K_FACTOR)
        runner = ParallelChunkRunner(
            dependencies.swapi_response, dependencies.efficiency_calibration_table
        )

        return SwapiL3ProtonSolarWindData(
            replace(self.input_metadata, descriptor="proton-sw"),
            **runner.run(chunks, ProtonChunkFitter()),
        )

    def process_l3a_alpha(self, data, dependencies) -> SwapiL3AlphaSolarWindData:
        if dependencies.mag_data is None:
            raise ValueError(
                "alpha-sw requires MAG RTN data (L2 preferred, L1D fallback); "
                "none was provided in the dependency collection."
            )
        chunks = list(chunk_l2_data(data, 5))
        dependencies.swapi_response.warm_cache(data.energy / SWAPI_L2_K_FACTOR)
        runner = ParallelChunkRunner(
            dependencies.swapi_response, dependencies.efficiency_calibration_table
        )

        fitter = AlphaChunkFitter(dependencies.mag_data)
        result = runner.run(chunks, fitter)
        
        if dependencies.mag_is_preliminary:
            result["quality_flags"] = result["quality_flags"] | int(SwapiL3Flags.PRELIMINARY_MAG)
        
        metadata = replace(self.input_metadata, descriptor="alpha-sw")
        return SwapiL3AlphaSolarWindData(metadata, **result)

    def process_l3a_pui(
        self, data, dependencies: SwapiL3ADependencies
    ) -> SwapiL3PickupIonData:
        dependencies.swapi_response.warm_cache(data.energy / SWAPI_L2_K_FACTOR)
        runner = ParallelChunkRunner(
            dependencies.swapi_response, dependencies.efficiency_calibration_table
        )

        proton_chunks = list(chunk_l2_data(data, 5))
        pui_chunks = list(chunk_l2_data(data, 50))
        proton_results = runner.run(proton_chunks, ProtonChunkFitter())

        fitter = PuiChunkFitter(
            density_of_neutral_helium_lookup_table=dependencies.density_of_neutral_helium_calibration_table,
            hydrogen_inflow_vector=dependencies.hydrogen_inflow_vector,
            helium_inflow_vector=dependencies.helium_inflow_vector,
            proton_results=proton_results,
        )
        result = runner.run(pui_chunks, fitter)

        metadata = replace(self.input_metadata, descriptor="pui-he")
        return SwapiL3PickupIonData(metadata, **result)

    def process_l3b(self, data, dependencies):
        epochs = []
        cdf_proton_velocities = []
        cdf_proton_probabilities = []
        cdf_alpha_velocities = []
        cdf_alpha_probabilities = []
        cdf_pui_velocities = []
        cdf_pui_probabilities = []
        combined_differential_fluxes = []
        combined_energies = []
        cdf_proton_deltas = []
        cdf_alpha_deltas = []
        cdf_pui_deltas = []
        combined_energy_deltas = []

        for data_chunk in chunk_l2_data(data, 50):
            center_of_epoch = data_chunk.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
            instrument_efficiency = (
                dependencies.efficiency_calibration_table.relative_proton_efficiency(
                    center_of_epoch
                )
            )
            coincidence_count_rates_with_uncertainty = uarray(
                data_chunk.coincidence_count_rate,
                data_chunk.coincidence_count_rate_uncertainty,
            )
            coarse_rates = coincidence_count_rates_with_uncertainty[:, SWAPI_COARSE_SWEEP_BINS]
            average_coincident_count_rates = np.sum(coarse_rates, axis=0) / len(coarse_rates)
            energies = np.mean(data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS], axis=0)
            proton_velocities, proton_probabilities = calculate_proton_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            alpha_velocities, alpha_probabilities = calculate_alpha_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            pui_velocities, pui_probabilities = calculate_pui_solar_wind_vdf(
                energies,
                average_coincident_count_rates,
                instrument_efficiency,
                dependencies.geometric_factor_calibration_table,
            )
            combined_differential_flux = (
                calculate_combined_solar_wind_differential_flux(
                    energies,
                    average_coincident_count_rates,
                    instrument_efficiency,
                    dependencies.geometric_factor_calibration_table,
                )
            )
            epochs.append(center_of_epoch)
            cdf_proton_velocities.append(proton_velocities)
            cdf_proton_probabilities.append(proton_probabilities)
            cdf_proton_deltas.append(calculate_delta_minus_plus(proton_velocities))

            cdf_alpha_velocities.append(alpha_velocities)
            cdf_alpha_probabilities.append(alpha_probabilities)
            cdf_alpha_deltas.append(calculate_delta_minus_plus(alpha_velocities))

            cdf_pui_velocities.append(pui_velocities)
            cdf_pui_probabilities.append(pui_probabilities)
            cdf_pui_deltas.append(calculate_delta_minus_plus(pui_velocities))

            combined_differential_fluxes.append(combined_differential_flux)
            combined_energies.append(energies)
            combined_energy_deltas.append(calculate_delta_minus_plus(energies))

        l3b_combined_metadata = self.input_metadata
        l3b_combined_metadata.descriptor = "combined"
        l3b_combined_vdf = SwapiL3BCombinedVDF(
            input_metadata=l3b_combined_metadata,
            epoch=np.array(epochs),
            proton_sw_velocities=np.array(cdf_proton_velocities),
            proton_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_proton_deltas]
            ),
            proton_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_proton_deltas]
            ),
            proton_sw_combined_vdf=np.array(cdf_proton_probabilities),
            alpha_sw_velocities=np.array(cdf_alpha_velocities),
            alpha_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_alpha_deltas]
            ),
            alpha_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_alpha_deltas]
            ),
            alpha_sw_combined_vdf=np.array(cdf_alpha_probabilities),
            pui_sw_velocities=np.array(cdf_pui_velocities),
            pui_sw_velocities_delta_minus=np.array(
                [delta.delta_minus for delta in cdf_pui_deltas]
            ),
            pui_sw_velocities_delta_plus=np.array(
                [delta.delta_plus for delta in cdf_pui_deltas]
            ),
            pui_sw_combined_vdf=np.array(cdf_pui_probabilities),
            combined_energy=np.array(combined_energies),
            combined_energy_delta_minus=np.array(
                [delta.delta_minus for delta in combined_energy_deltas]
            ),
            combined_energy_delta_plus=np.array(
                [delta.delta_plus for delta in combined_energy_deltas]
            ),
            combined_differential_flux=np.array(combined_differential_fluxes),
        )
        return l3b_combined_vdf
