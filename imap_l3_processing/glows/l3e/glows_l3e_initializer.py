import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import (
    ProcessingInputCollection,
    AncillaryInput,
    ScienceInput,
    RepointInput,
)

from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3d.utils import get_most_recently_uploaded_ancillary
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_utils import (
    find_first_updated_cr,
    identify_versions_for_l3e_output_files,
    GlowsL3eVersionsForRepointings,
)
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo
from imap_l3_processing.models import VersionMap
from imap_l3_processing.utils import FurnishMetakernelOutput

logger = logging.getLogger(__name__)

@dataclass
class GlowsL3EInitializerOutput:
    dependencies: GlowsL3EDependencies
    repointings: GlowsL3eVersionsForRepointings
    l3d_cdf_path: Path
    metakernel_with_predict_ephem: FurnishMetakernelOutput
    metakernel_without_predict_ephem: FurnishMetakernelOutput


class GlowsL3EInitializer:
    @staticmethod
    def get_repointings_to_process(
            l3d_output: GlowsL3DProcessorOutput,
            previous_l3d: Optional[str],
            repointing_file_path: Path,
            version_map: VersionMap,
            reprocess_info: ReprocessInfo,
    ) -> Optional[GlowsL3EInitializerOutput]:
        pipeline_settings_l3bcde = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='pipeline-settings-l3bcde'))
        energy_grid_lo = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-lo'))
        tess_xyz_8 = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='tess-xyz-8'))
        energy_grid_hi = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-hi'))
        energy_grid_ultra = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-ultra'))
        tess_ang_16 = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='tess-ang-16'))

        processing_input_collection = ProcessingInputCollection(
            ScienceInput(l3d_output.l3d_cdf_file_path.name),
            *[AncillaryInput(file.name) for file in l3d_output.l3d_text_file_paths],
            AncillaryInput(str(pipeline_settings_l3bcde["file_path"])),
            AncillaryInput(str(energy_grid_lo["file_path"])),
            AncillaryInput(str(tess_xyz_8["file_path"])),
            AncillaryInput(str(energy_grid_hi["file_path"])),
            AncillaryInput(str(energy_grid_ultra["file_path"])),
            AncillaryInput(str(tess_ang_16["file_path"])),
            RepointInput(str(repointing_file_path))
        )

        l3e_deps = GlowsL3EDependencies.fetch_dependencies(processing_input_collection)
        l3e_deps.copy_dependencies()

        first_cr = l3e_deps.pipeline_settings["start_cr"]

        first_updated_cr = first_cr
        if previous_l3d is not None:
            first_updated_cr = find_first_updated_cr(l3d_output.l3d_cdf_file_path, previous_l3d)
            if first_updated_cr is not None:
                first_updated_cr -= 1

        last_cr = l3d_output.last_processed_cr
        files_to_produce = identify_versions_for_l3e_output_files(
            first_cr,
            last_cr,
            first_updated_cr,
            repointing_file_path,
            version_map,
            reprocess_info,
        )

        if len(files_to_produce.repointing_numbers) == 0:
            return None

        earliest_repointing_start, _ = get_pointing_date_range(
            min(files_to_produce.repointing_numbers)
        )
        _, latest_repointing_end = get_pointing_date_range(
            max(files_to_produce.repointing_numbers)
        )

        furnished_metakernels = GlowsL3EDependencies.collect_spice_dependencies(
            start_date=earliest_repointing_start, end_date=latest_repointing_end
        )

        return GlowsL3EInitializerOutput(
            dependencies=l3e_deps,
            repointings=files_to_produce,
            l3d_cdf_path=l3d_output.l3d_cdf_file_path,
            metakernel_with_predict_ephem=furnished_metakernels[0],
            metakernel_without_predict_ephem=furnished_metakernels[1],
        )

