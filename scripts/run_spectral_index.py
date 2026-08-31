import dataclasses
import shutil
from datetime import datetime
from pathlib import Path

from imap_data_access import (
    ScienceFilePath,
    ProcessingInputCollection,
    ScienceInput,
    AncillaryInput,
)
from imap_data_access.file_validation import Version

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.maps.map_descriptors import (
    parse_map_descriptor,
    MapQuantity,
    map_descriptor_parts_to_string,
)
from imap_l3_processing.models import InputMetadata, VersionMap
from imap_l3_processing.ultra.ultra_processor import UltraProcessor


def run_spectral_index(input: Path):
    parsed = ScienceFilePath(input.name)

    processors = {"hi": HiProcessor, "lo": LoProcessor, "ultra": UltraProcessor}
    parsed_descriptor = parse_map_descriptor(parsed.descriptor)

    output_descriptor = map_descriptor_parts_to_string(
        dataclasses.replace(parsed_descriptor, quantity=MapQuantity.SpectralIndex, quantity_suffix="")
    )

    dependencies = ProcessingInputCollection(ScienceInput(input.name))
    if parsed.instrument == 'ultra':
        dependencies.add(AncillaryInput("imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat"))

    data_path = parsed.construct_path()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input, data_path)
    processor = processors[parsed.instrument](
        dependencies,
        InputMetadata(
            instrument=parsed.instrument,
            data_level="l3",
            start_date=datetime.strptime(parsed.start_date, "%Y%m%d"),
            end_date=None,
            version=VersionMap({}, Version.from_version("v000.0000")),
            descriptor=output_descriptor,
        ),
    )
    return processor.process()


if __name__ == "__main__":
    outputs = []
    for file in Path(
            '/Users/harrison/Development/imap_L3_processing/run_local_input_data/lo/integration_data/imap/lo/l3/2025/04').iterdir():
        outputs.extend(run_spectral_index(file))
    print(outputs)
