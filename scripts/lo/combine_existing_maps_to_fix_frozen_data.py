import abc
import dataclasses
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Self, Optional, override, Literal

import imap_data_access
import numpy as np
import pandas
import xarray as xr
from imap_data_access import ProcessingInputCollection, ScienceFilePath
from imap_data_access.file_validation import Version
from imap_data_access.processing_input import ScienceInput
from imap_processing.cdf.utils import write_cdf as write_l2_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.hit.l1b.constants import FILLVAL_INT64
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.lo.l3.lo_sp_initializer import LO_SP_MAP_KERNELS
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.utils import furnished_metakernel
from tests.test_helpers import get_run_local_data_path


def copy_to_output_directory(
        release_directory: Path, output_maps: list[Path]
):
    release_directory.mkdir(parents=True, exist_ok=True)

    for generated_path in output_maps:
        shutil.copy(generated_path, release_directory)


if __name__ == "__main__":

    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    output_data_path = get_run_local_data_path("lo_txt_pipeline")
    shutil.rmtree(output_data_path / "imap" / "lo" / "l2", ignore_errors=True)
    shutil.rmtree(output_data_path / "imap" / "lo" / "l3", ignore_errors=True)
    imap_data_access.config["DATA_DIR"] = output_data_path

    lo_input_data_dir = get_run_local_data_path("input_lo_txt_pipeline")
    maps_to_combine = lo_input_data_dir / "maps_to_combine"
    if not maps_to_combine.exists():
        raise Exception(f"Place maps to combine in {maps_to_combine}")

    maps_needed_for_combination = [
        (
            "ilo-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
            [
                "l090-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
                "l075-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
            [
                "l090-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
                "l075-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l090-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l075-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l090-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l075-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l090-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l075-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
    ]
    input_descriptors = []
    input_files = {}
    for combined_descriptor, combined_dependencies in maps_needed_for_combination:
        input_descriptors.extend(combined_dependencies)
    for file in maps_to_combine.rglob("*.cdf"):
        sfp = ScienceFilePath(file.name)
        if sfp.descriptor in input_descriptors:
            output_path = sfp.construct_path()
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(file, output_path)
            assert input_files.get(sfp.descriptor) is None
            input_files[sfp.descriptor] = file.name
    output_maps = []
    for combined_descriptor, combined_dependencies in maps_needed_for_combination:

        [combined_sp_map] = LoProcessor(
            dependencies=ProcessingInputCollection(*[ScienceInput(input_files[desc]) for desc in combined_dependencies]),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=datetime(2025,11,25),
                end_date=None,
                version=VersionMap({}, Version(None, 2)),
                descriptor=combined_descriptor
            )
        ).process()

        output_maps.extend([
            combined_sp_map
        ])

    release_directory = get_run_local_data_path("IMAP-Lo July 31st 2026 Maps")
    copy_to_output_directory(release_directory, output_maps)
