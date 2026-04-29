import enum
import json
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Union, TypeVar
from urllib.parse import urlparse

import imap_data_access
import requests
import spiceypy
from imap_data_access import ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection
from requests import RequestException
from spacepy.pycdf import CDF

import imap_l3_processing
from imap_l3_processing.cdf.cdf_utils import write_cdf, read_numeric_variable
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet, \
    RectangularSpectralIndexMapData, RectangularIntensityMapData, \
    HealPixIntensityMapData, HealPixSpectralIndexMapData, RectangularSpectralIndexDataProduct, \
    RectangularIntensityDataProduct, HealPixSpectralIndexDataProduct, HealPixIntensityDataProduct, MapDataProduct, \
    ISNBackgroundSubtractedDataProduct, ISNBackgroundSubtractedMapData
from imap_l3_processing.models import UpstreamDataDependency, DataProduct, MagData, InputMetadata
from imap_l3_processing.ultra.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.version import VERSION

logger = logging.getLogger(__name__)


class SpiceKernelTypes(enum.Enum):
    Leapseconds = "leapseconds"
    IMAPFrames = "imap_frames"
    ScienceFrames = "science_frames"
    EphemerisReconstructed = "ephemeris_reconstructed"
    AttitudeHistory = "attitude_history"
    PointingAttitude = "pointing_attitude"
    PlanetaryEphemeris = "planetary_ephemeris"
    SpacecraftClock = "spacecraft_clock"
    EphemerisPredicted = "ephemeris_predicted"
    PlanetaryConstants = "planetary_constants"


def save_data(data: DataProduct, delete_if_present: bool = False, folder_path: Path = None,
              cr_number=None) -> Path:
    assert data.input_metadata.repointing is None or cr_number is None, "You cannot call save_data with both a repointing in the metadata while passing in a CR number"
    formatted_start_date = data.input_metadata.start_date.strftime("%Y%m%d")
    science_file_path = ScienceFilePath.generate_from_inputs(
        instrument=data.input_metadata.instrument,
        data_level=data.input_metadata.data_level,
        descriptor=data.input_metadata.descriptor,
        start_time=formatted_start_date,
        repointing=data.input_metadata.repointing,
        cr=cr_number,
        version=data.input_metadata.version,
    )

    file_path = science_file_path.construct_path()
    if folder_path is not None:
        file_path = folder_path / file_path.name

    logical_source = f"imap_{data.input_metadata.instrument}_{data.input_metadata.data_level}_{data.input_metadata.descriptor}"
    logical_file_id = file_path.stem

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if delete_if_present:
        file_path.unlink(missing_ok=True)

    attribute_manager = ImapAttributeManager()
    attribute_manager.add_global_attribute("Data_version", data.input_metadata.version.replace('v', ''))
    attribute_manager.add_instrument_attrs(data.input_metadata.instrument, data.input_metadata.data_level,
                                           data.input_metadata.descriptor)
    attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
    attribute_manager.add_global_attribute("Logical_source", logical_source)
    attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
    attribute_manager.add_global_attribute("ground_software_version", VERSION)

    map_instruments = ["hi", "lo", "ultra"]
    if data.input_metadata.instrument in map_instruments:

        if isinstance(data, (RectangularSpectralIndexDataProduct, RectangularIntensityDataProduct,
                             HealPixSpectralIndexDataProduct, HealPixIntensityDataProduct,
                             ISNBackgroundSubtractedDataProduct)):
            attrs = generate_map_global_metadata(data)
            for key, value in attrs.items():
                attribute_manager.add_global_attribute(key, value)

            if attribute_manager.try_load_global_metadata(logical_source) is None:
                logical_source_global_attrs = generate_global_metadata_for_undefined_logical_source(data.input_metadata)
                attribute_manager.add_global_attribute(logical_source, logical_source_global_attrs)

            attribute_manager.add_global_attribute("Spice_reference_frame", data.spice_frame_name.name)
        else:
            raise AssertionError(f"Found an unsupported map data product of type: {type(data)}")
    elif data.parent_file_names:
        attribute_manager.add_global_attribute("Parents", data.parent_file_names)

    if data.global_metadata_attrs:
        for key, value in data.global_metadata_attrs.items():
            attribute_manager.add_global_attribute(key, value)

    file_path_str = str(file_path)
    write_cdf(file_path_str, data, attribute_manager)
    return file_path


def generate_map_global_metadata(data_product: MapDataProduct) -> dict:
    attrs = {}

    match data_product.data:
        case (
        RectangularSpectralIndexMapData(spectral_index_map_data=map_data) |
        HealPixSpectralIndexMapData(spectral_index_map_data=map_data) |
        RectangularIntensityMapData(intensity_map_data=map_data) |
        HealPixIntensityMapData(intensity_map_data=map_data) |
        ISNBackgroundSubtractedMapData(isn_rate_map_data=map_data)
        ):
            [start_date] = map_data.epoch
            [epoch_delta] = map_data.epoch_delta

            if not isinstance(start_date, datetime):
                start_date = TT2000_EPOCH + timedelta(seconds=start_date / 1e9)

            end_date = start_date + timedelta(seconds=epoch_delta / 1e9)
        case _:
            raise ValueError(f"Found an unsupported map data product of type: {type(data_product.data)}")

    attrs["start_date"] = start_date.isoformat()
    attrs["end_date"] = end_date.isoformat()

    if data_product.parent_file_names:
        science_files = []
        ancillary_files = []
        for name in data_product.parent_file_names:
            try:
                ScienceFilePath(name)
                science_files.append(name)
            except ScienceFilePath.InvalidImapFileError:
                ancillary_files.append(name)

        attrs["Parents"] = science_files
        attrs["Ancillary_files"] = ancillary_files

    return attrs


def generate_global_metadata_for_undefined_logical_source(input_metadata: InputMetadata) -> dict:
    level = input_metadata.data_level.replace('l', '')
    data_type_string = f"Level-{level}"
    if "spx" in input_metadata.descriptor:
        data_type_string += f" Spectral Fit Index Map"
    elif "ena" in input_metadata.descriptor and "-sp-" in input_metadata.descriptor:
        data_type_string += f" Survival Corrected"
    elif "ena" in input_metadata.descriptor:
        data_type_string += f" ENA Intensity"

    logical_source_global_attrs = {
        "Data_level": level,
        "Data_type": f"L{level}_{input_metadata.descriptor}>{data_type_string}",
        "Logical_source_description": f"IMAP-{input_metadata.instrument} {data_type_string}",
    }
    return logical_source_global_attrs


def format_time(t: Optional[datetime]) -> Optional[str]:
    if t is not None:
        return t.strftime("%Y%m%d")
    return None


def download_dependency(dependency: UpstreamDataDependency) -> Path:
    files_to_download = [result['file_path'] for result in
                         imap_data_access.query(instrument=dependency.instrument,
                                                data_level=dependency.data_level,
                                                descriptor=dependency.descriptor,
                                                start_date=format_time(dependency.start_date),
                                                end_date=format_time(dependency.end_date),
                                                version=dependency.version
                                                )]
    if len(files_to_download) != 1:
        raise ValueError(f"{files_to_download}. Expected one file to download, found {len(files_to_download)}.")

    return imap_data_access.download(files_to_download[0])


def download_dependency_with_repointing(dependency: UpstreamDataDependency) -> (Path, int):
    files_with_repointing_to_download = [(result['file_path'], result['repointing']) for result in
                                         imap_data_access.query(instrument=dependency.instrument,
                                                                data_level=dependency.data_level,
                                                                descriptor=dependency.descriptor,
                                                                start_date=format_time(dependency.start_date),
                                                                end_date=format_time(dependency.end_date),
                                                                version=dependency.version
                                                                )]
    if len(files_with_repointing_to_download) != 1:
        raise ValueError(
            f"{[file[0] for file in files_with_repointing_to_download]}. Expected one file to download, found {len(files_with_repointing_to_download)}.")
    repointing_number = files_with_repointing_to_download[0][1]
    return imap_data_access.download(files_with_repointing_to_download[0][0]), repointing_number


def download_external_dependency(dependency_url: str, file_path: Path) -> Optional[Path]:
    try:
        response = requests.get(dependency_url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            return Path(file_path)
        else:
            logger.error(f"Failed to download {dependency_url} with status code {response.status_code}")
    except RequestException:
        logger.exception(f"Failed to download {dependency_url}")
    return None


def read_mag_data(cdf_path: Union[str, Path]) -> MagData:
    with CDF(str(cdf_path)) as cdf:
        return MagData(
            epoch=cdf['epoch'][...],
            mag_data=read_numeric_variable(cdf["b_dsrf"])[:, :3])


L1CPointingSet = TypeVar("L1CPointingSet", bound=Union[InputRectangularPointingSet, UltraL1CPSet])
GlowsL3eData = TypeVar("GlowsL3eData", bound=Union[GlowsL3eRectangularMapInputData, UltraGlowsL3eData])


def combine_glows_l3e_with_l1c_pointing(glows_l3e_data: list[GlowsL3eData], l1c_data: list[L1CPointingSet]) -> list[
    tuple[L1CPointingSet, Optional[GlowsL3eData]]]:
    l1c_by_repoint = {l1c.repointing: l1c for l1c in l1c_data}
    glows_by_repoint = {l3e.repointing: l3e for l3e in glows_l3e_data}

    return [(l1c_by_repoint[repoint], glows_by_repoint.get(repoint, None))
            for repoint in l1c_by_repoint.keys()]


def get_dependency_paths_by_descriptor(deps: ProcessingInputCollection, descriptors: list[str]) -> dict[
    str, list[Path]]:
    descriptor_to_paths = {key: [] for key in descriptors}
    for input_file in deps.get_science_inputs():
        for descriptor in descriptors:
            if descriptor in input_file.descriptor:
                descriptor_to_paths[descriptor].append(input_file.filename_list[0])
                continue

    return descriptor_to_paths


def furnish_local_spice():
    kernels = Path(imap_l3_processing.__file__).parent.parent.joinpath("spice_kernels")

    total_kernels = spiceypy.ktotal('ALL')
    current_kernels = []
    for i in range(0, total_kernels):
        current_kernels.append(Path(spiceypy.kdata(i, 'ALL')[0]).name)

    for file in kernels.iterdir():
        if file.name not in current_kernels:
            spiceypy.furnsh(str(file))


def get_spice_parent_file_names() -> list[str]:
    count = spiceypy.ktotal('ALL')
    return [Path(spiceypy.kdata(i, 'ALL')[0]).name for i in range(0, count)]


@dataclass
class FurnishMetakernelOutput:
    metakernel_path: Path
    spice_kernel_paths: list[Path]


def get_spice_kernels_file_names(start_date: datetime, end_date: datetime, kernel_types: list[SpiceKernelTypes]) -> \
        list[str]:
    metakernel_url = urlparse(imap_data_access.config['DATA_ACCESS_URL'])._replace(path="metakernel").geturl()

    parameters: dict = {
        'file_types': [kernel_type.value for kernel_type in kernel_types],
        'start_time': f"{int((start_date - datetime(2000, 1, 1, 12)).total_seconds())}",
        'end_time': f"{int((end_date - datetime(2000, 1, 1, 12)).total_seconds())}",
    }

    kernels_res = requests.get(metakernel_url, params={**parameters, 'list_files': 'true'})
    kernels = json.loads(kernels_res.text)

    return kernels


def furnish_spice_metakernel(start_date: datetime, end_date: datetime, kernel_types: list[SpiceKernelTypes]):
    metakernel_path = imap_data_access.config.get("DATA_DIR") / "metakernel" / "metakernel.txt"
    kernel_path = imap_data_access.config.get("DATA_DIR") / "imap" / "spice"

    parameters: dict = {
        'spice_path': kernel_path,
        'file_types': [kernel_type.value for kernel_type in kernel_types],
        'start_time': str(int((start_date - datetime(2000, 1, 1, 12)).total_seconds())),
        'end_time': str(int((end_date - datetime(2000, 1, 1, 12)).total_seconds())),
    }

    metakernel_url = urlparse(imap_data_access.config['DATA_ACCESS_URL'])._replace(path="metakernel").geturl()

    logger.info(f"Getting SPICE Metakernel from: {metakernel_url}, with params: {parameters}")

    metakernel_res = requests.get(metakernel_url, params=parameters)

    metakernel_path.parent.mkdir(parents=True, exist_ok=True)
    metakernel_path.write_bytes(metakernel_res.content)

    kernels = get_spice_kernels_file_names(start_date, end_date, kernel_types)
    logger.info(f"Metakernel API returned the following kernels: {kernels}")

    downloaded_paths = [imap_data_access.download(kernel) for kernel in kernels]

    spiceypy.furnsh(str(metakernel_path))

    return FurnishMetakernelOutput(metakernel_path=metakernel_path, spice_kernel_paths=downloaded_paths)


def read_cdf_parents(server_file_name: str) -> set[str]:
    downloaded_path = imap_data_access.download(server_file_name)

    with CDF(str(downloaded_path)) as cdf:
        parents = set(cdf.attrs["Parents"])
    return parents
