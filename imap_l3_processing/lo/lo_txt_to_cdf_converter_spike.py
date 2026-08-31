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

LO_ENERGIES_IN_KEV = np.array([16.33, 30.47, 55.76, 106.3, 200.0, 405.0, 787.3]) / 1000.0
LO_ENERGY_BIN_LOWERS = np.array([10.9, 20.4, 36.8, 71.6, 135.0, 269.0, 504.7]) / 1000.0
LO_ENERGY_BIN_UPPERS = np.array([21.7, 40.5, 74.7, 140.9, 265.0, 541.0, 1069.9]) / 1000.0


class CsvNameToProduct:
    @abc.abstractmethod
    def get_energy_and_quantity(self, filename) -> Optional[tuple[int, str]]:
        raise NotImplementedError


@dataclasses.dataclass
class NBSNameMapping(CsvNameToProduct):
    @override
    def get_energy_and_quantity(self, filename: str) -> Optional[tuple[int, str]]:
        if fn_match := re.match("map_([a-zA-Z]+)_esa(\d{1}).csv", filename):
            [quantity, energy] = fn_match.groups()

            csv_to_l2_cdf_mapping = {
                "flux": "ena_intensity",
                "fvar": "ena_intensity_stat_var",
                "fsel": "ena_intensity_sys_err_minus",
                "fseu": "ena_intensity_sys_err_plus",
                "fser": "ena_intensity_sys_err",
                "bflux": "bg_intensity",
                "bfvar": "bg_intensity_stat_var",
                "bfunc": "bg_intensity_sys_err",
                "expo": "exposure_factor"
            }

            if quantity in csv_to_l2_cdf_mapping:
                return int(energy), csv_to_l2_cdf_mapping[quantity]


class CGNameMapping(CsvNameToProduct):
    @override
    def get_energy_and_quantity(self, filename: str) -> Optional[tuple[int, str]]:
        if fn_match := re.match("([a-zA-Z]+_[a-zA-Z]+)_esa(\d{1}).csv", filename):
            [quantity, energy] = fn_match.groups()

            csv_to_l2_cdf_mapping = {
                "map_cgflux": "ena_intensity",
                "map_cgfvar": "ena_intensity_stat_var",
                "map_cgfunl": "ena_intensity_sys_err_minus",
                "map_cgfunu": "ena_intensity_sys_err_plus",
                "map_cgfunc": "ena_intensity_sys_err",
                "bkg_cgflux": "bg_intensity",
                "bkg_cgfvar": "bg_intensity_stat_var",
                "bkg_cgfunc": "bg_intensity_sys_err",
            }

            if quantity in csv_to_l2_cdf_mapping:
                return int(energy), csv_to_l2_cdf_mapping[quantity]


class SputterOrBootStrapNameMapping(CsvNameToProduct):
    def __init__(self, correction: Literal["sput", "boot"]):
        self.correction = correction

    @override
    def get_energy_and_quantity(self, filename: str) -> Optional[tuple[int, str]]:
        if fn_match := re.match("map_flux_(\d{1})_Hy_([a-zA-Z]+)_([a-zA-Z]+).csv", filename):
            [energy, correction, quantity] = fn_match.groups()

            csv_to_l2_cdf_mapping = {
                "cor": "ena_intensity",
                "var": "ena_intensity_stat_var",
                "unc": "ena_intensity_sys_err",
                "unu": "ena_intensity_sys_err_plus",
                "unl": "ena_intensity_sys_err_minus",
            }

            if correction == self.correction and quantity in csv_to_l2_cdf_mapping:
                return int(energy), csv_to_l2_cdf_mapping.get(quantity, None)


@dataclasses.dataclass
class Manifest:
    repoints: list[int]
    l1b_filenames: list[str]
    start_date: datetime
    end_date: datetime

    @classmethod
    def load(cls, manifest_path: Path) -> Self:
        manifest_df = pandas.read_csv(manifest_path)
        l1c_times = pandas.to_datetime(manifest_df["date_yyyymmdd"], format="%Y%m%d")
        start_date = min(l1c_times)
        end_date = max(l1c_times)

        return cls(
            repoints=list(manifest_df["repoint"]),
            l1b_filenames=list(manifest_df["l1b_filename"]),
            start_date=start_date,
            end_date=end_date,
        )


@dataclasses.dataclass
class LoProcessingInput:
    repoints: list[int]
    start_date: datetime
    end_date: datetime
    version: int
    l2_descriptor: str
    dataset: xr.Dataset
    l2_cdf_path: Path

    @staticmethod
    def load_data_dir(data_dir: Path, name_mapping: CsvNameToProduct, **kwargs):
        temp_data = {}
        for data_file_path in data_dir.iterdir():
            if energy_and_quantity := name_mapping.get_energy_and_quantity(data_file_path.name):
                esa_step, data_type = energy_and_quantity
                if data_type not in temp_data:
                    temp_data[data_type] = np.full((1, 7, 60, 30), np.nan)
                temp_data[data_type][0, esa_step - 1] = np.loadtxt(data_file_path, delimiter=",", **kwargs).T
        return temp_data

    @classmethod
    def load(cls,
             manifest: Manifest,
             l2_descriptor: str,
             version: int,
             input_path: Path,
             name_mapping: CsvNameToProduct,
             use_masked_data: bool,
             **additional_map_data
             ) -> Self:
        if use_masked_data:
            loaded_data = {
                **LoProcessingInput.load_data_dir(input_path / "maps", name_mapping, skiprows=1),
                **LoProcessingInput.load_data_dir(input_path / "masked_maps", name_mapping)
            }
        else:
            loaded_data = {
                **LoProcessingInput.load_data_dir(input_path / "maps", name_mapping, skiprows=1),
            }

        start_date_nanoseconds = (manifest.start_date - TT2000_EPOCH).total_seconds() * 1e9
        end_date_nanoseconds = (manifest.end_date - TT2000_EPOCH).total_seconds() * 1e9

        skymap = RectangularSkyMap(6, SpiceFrame.ECLIPJ2000)
        skymap.max_epoch = int(end_date_nanoseconds)
        skymap.min_epoch = int(start_date_nanoseconds)

        map_data_coords = ["epoch", "energy", "longitude", "latitude"]
        l2_dataset = xr.Dataset(
            data_vars=
            {
                "obs_date": (map_data_coords, np.full((1, 7, 60, 30), FILLVAL_INT64)),
                "obs_date_range": (map_data_coords, np.full((1, 7, 60, 30), np.nan)),
                "solid_angle": (["epoch", "longitude", "latitude"], np.full((1, 60, 30), np.nan)),
                "energy_delta_minus": (["energy"], LO_ENERGIES_IN_KEV - LO_ENERGY_BIN_LOWERS),
                "energy_delta_plus": (["energy"], LO_ENERGY_BIN_UPPERS - LO_ENERGIES_IN_KEV),

                **{quantity: (map_data_coords, data) for quantity, data in loaded_data.items()},
                **additional_map_data
            },
            coords={
                "epoch": np.array([start_date_nanoseconds]),
                "energy": LO_ENERGIES_IN_KEV,
                "longitude": skymap.sky_grid.az_bin_midpoints,
                "latitude": skymap.sky_grid.el_bin_midpoints,
            },
        )

        if "ena_intensity_stat_var" in l2_dataset.data_vars:
            l2_dataset["ena_intensity_stat_uncert"] = np.sqrt(l2_dataset["ena_intensity_stat_var"])
            l2_dataset.drop_vars(["ena_intensity_stat_var"])

        if "bg_intensity_stat_var" in l2_dataset.data_vars:
            l2_dataset["bg_intensity_stat_uncert"] = np.sqrt(l2_dataset["bg_intensity_stat_var"])
            l2_dataset.drop_vars(["bg_intensity_stat_var"])

        variables_to_mask = [
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "ena_intensity_sys_err_plus",
            "ena_intensity_sys_err_minus",
            "bg_intensity",
            "bg_intensity_stat_uncert",
            "bg_intensity_sys_err",
        ]

        mask = l2_dataset["exposure_factor"] != 0.0
        for var in variables_to_mask:
            if var in l2_dataset:
                l2_dataset[var] = l2_dataset[var].where(mask, np.nan)

        l2_dataset_with_metadata = skymap.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=l2_descriptor,
            external_map_dataset=l2_dataset,
        )
        l2_dataset_with_metadata.attrs["Data_version"] = f"{version:03}"
        l2_dataset_with_metadata.attrs["Parents"] = manifest.l1b_filenames

        return cls(
            dataset=l2_dataset_with_metadata,
            l2_descriptor=l2_descriptor,
            version=version,
            l2_cdf_path=write_l2_cdf(l2_dataset_with_metadata),
            repoints=manifest.repoints,
            start_date=manifest.start_date,
            end_date=manifest.end_date,
        )

    def get_spx_dependencies(self):
        return ProcessingInputCollection(ScienceInput(self.l2_cdf_path.name))

    def get_survival_corrected_dependencies(self, l1c_files: list[Path]):
        # l1c_results = imap_data_access.query(instrument="lo", data_level="l1c", version="latest")
        # l1c_inputs = [Path(l1c["file_path"]) for l1c in l1c_results if l1c["repointing"] in self.repoints]
        l1c_inputs = [p for p in l1c_files if ScienceFilePath(p.name).repointing in self.repoints]

        glows_query_results = imap_data_access.query(
            instrument="glows",
            data_level="l3e",
            descriptor="survival-probability-lo",
            version="latest",
        )
        glows_paths = [
            Path(glows["file_path"]) for glows in glows_query_results if int(glows["repointing"]) in self.repoints
        ]

        return ProcessingInputCollection(*(ScienceInput(p.name) for p in [self.l2_cdf_path, *glows_paths, *l1c_inputs]))

    def make_l3_input_metadata(self, l3_descriptor: str) -> InputMetadata:
        return InputMetadata(
            instrument="lo",
            data_level="l3",
            start_date=self.start_date,
            end_date=self.end_date,
            version=VersionMap({}, Version(None, 2)),
            descriptor=l3_descriptor,
        )


def copy_to_output_directory_and_rename_for_initial_release(
        release_directory: Path, output_maps: list[Path]
):
    for generated_path in output_maps:
        science_file_path = ScienceFilePath(generated_path.name)

        new_name = (
                "_".join(
                    [
                        "imap",
                        science_file_path.instrument,
                        science_file_path.data_level,
                        science_file_path.descriptor + "-INITIAL",
                        science_file_path.start_date,
                        science_file_path.version,
                    ]
                )
                + ".cdf"
        )

        output_dir = release_directory / science_file_path.data_level
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / new_name

        with CDF(str(output_path), masterpath=str(generated_path), readonly=False) as cdf:
            cdf.attrs["Logical_file_id"] = output_path.stem


def download_all_l1c(path: Path):
    original_data_dir = imap_data_access.config["DATA_DIR"]
    try:
        imap_data_access.config["DATA_DIR"] = path
        all_l1c = imap_data_access.query(instrument="lo", data_level="l1c", version="latest")

        with ThreadPoolExecutor() as pool:
            for l1c in all_l1c:
                pool.submit(imap_data_access.download, l1c["file_path"])

    finally:
        imap_data_access.config["DATA_DIR"] = original_data_dir


if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    output_data_path = get_run_local_data_path("lo_txt_pipeline")
    shutil.rmtree(output_data_path / "imap" / "lo" / "l2", ignore_errors=True)
    shutil.rmtree(output_data_path / "imap" / "lo" / "l3", ignore_errors=True)
    imap_data_access.config["DATA_DIR"] = output_data_path

    lo_input_data_dir = get_run_local_data_path("input_lo_txt_pipeline")
    cg_corrected_input_path = lo_input_data_dir / "3S8_l1b_cg_corrected"
    sputter_or_bootstrap_input_path = lo_input_data_dir / "3S7_l1b_sputterbootstrap_ram"
    ram_nbs_input_path = lo_input_data_dir / "3S5_l1b_ram_maps"

    l1c_paths = []
    local_l1c_input = lo_input_data_dir / "l1c"
    
    download_all_l1c(local_l1c_input)
    for file in local_l1c_input.rglob("*.cdf"):
        sfp = ScienceFilePath(file.name)

        output_path = sfp.construct_path()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file, output_path)
        l1c_paths.append(output_path)
    if len(l1c_paths) == 0:
        raise Exception(f"Expected l1c files to be provided in {local_l1c_input}")

    output_maps = []
    for masked, descriptor_suffix in [(False, ""), (True, "Msk")]:
        start_dates = []
        end_dates = []
        for pivot in (90, 105, 75):
            ram_nbs_path_map_path = ram_nbs_input_path / "outdir" / f"pivot_{pivot}"
            manifest_path = ram_nbs_path_map_path / "maps" / "map_l1b_manifest_esa1.csv"
            manifest = Manifest.load(manifest_path)

            start_dates.append(manifest.start_date)
            end_dates.append(manifest.end_date)

            nbs_processing_input = LoProcessingInput.load(
                manifest,
                f"l{pivot:03d}-enansnbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo",
                1,
                ram_nbs_path_map_path,
                NBSNameMapping(),
                masked,
            )

            [spx_nsnbs_map] = LoProcessor(
                input_metadata=nbs_processing_input.make_l3_input_metadata(
                    f"l{pivot:03d}-spxnsnbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo"),
                dependencies=nbs_processing_input.get_spx_dependencies(),
            ).process()

            cg_processing_input = LoProcessingInput.load(
                manifest,
                f"l{pivot:03d}-enasbs{descriptor_suffix}-h-hf-nsp-ram-hae-6deg-6mo",
                1,
                cg_corrected_input_path / "outdir" / f"pivot_{pivot}",
                CGNameMapping(),
                masked,
                exposure_factor=nbs_processing_input.dataset["exposure_factor"]
            )

            [spx_cg_nsp_map] = LoProcessor(
                input_metadata=cg_processing_input.make_l3_input_metadata(
                    f"l{pivot:03d}-spxsbs{descriptor_suffix}-h-hf-nsp-ram-hae-6deg-6mo"),
                dependencies=cg_processing_input.get_spx_dependencies()
            ).process()

            with furnished_metakernel(cg_processing_input.start_date, cg_processing_input.end_date, LO_SP_MAP_KERNELS):
                [sp_map] = LoProcessor(
                    input_metadata=cg_processing_input.make_l3_input_metadata(
                        f"l{pivot:03d}-enasbs{descriptor_suffix}-h-hf-sp-ram-hae-6deg-6mo"),
                    dependencies=cg_processing_input.get_survival_corrected_dependencies(l1c_paths),
                ).process()

            [spx_cg_sp_map] = LoProcessor(
                input_metadata=cg_processing_input.make_l3_input_metadata(
                    f"l{pivot:03d}-spxsbs{descriptor_suffix}-h-hf-sp-ram-hae-6deg-6mo"),
                dependencies=ProcessingInputCollection(ScienceInput(sp_map.name)),
            ).process()

            sputter_no_bootstrap_map = LoProcessingInput.load(
                manifest=manifest,
                l2_descriptor=f"l{pivot:03d}-enasnbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo",
                version=1,
                input_path=sputter_or_bootstrap_input_path / "outdir" / f"pivot_{pivot}",
                name_mapping=SputterOrBootStrapNameMapping("sput"),
                use_masked_data=masked,
                exposure_factor=nbs_processing_input.dataset["exposure_factor"]
            )

            sputter_and_bootstrap_map = LoProcessingInput.load(
                manifest=manifest,
                l2_descriptor=f"l{pivot:03d}-enasbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo",
                version=1,
                input_path=sputter_or_bootstrap_input_path / "outdir" / f"pivot_{pivot}",
                name_mapping=SputterOrBootStrapNameMapping("boot"),
                use_masked_data=masked,
                exposure_factor=nbs_processing_input.dataset["exposure_factor"]
            )

            output_maps.extend([
                nbs_processing_input.l2_cdf_path,
                spx_nsnbs_map,
                cg_processing_input.l2_cdf_path,
                spx_cg_nsp_map,
                sp_map,
                spx_cg_sp_map,
                sputter_no_bootstrap_map.l2_cdf_path,
                sputter_and_bootstrap_map.l2_cdf_path
            ])

    maps_needed_for_combination = [
        (
            "ilo-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
            [
                "l075-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
                "l090-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-hf-sp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
            [
                "l075-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
                "l090-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-hf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l075-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l090-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enasbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l075-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l090-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enasnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
        (
            "ilo-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            [
                "l075-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l090-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
                "l105-enansnbsMsk-h-sf-nsp-ram-hae-6deg-6mo",
            ],
        ),
    ]

    combined_start_date = min(start_dates)
    combined_end_date = max(end_dates)
    for combined_descriptor, combined_dependencies in maps_needed_for_combination:
        combined_inputs = []
        for output_map in output_maps:
            if any([f"_{combined_dep}_" in output_map.name for combined_dep in combined_dependencies]):
                combined_inputs.append(output_map)

        [combined_sp_map] = LoProcessor(
            dependencies=ProcessingInputCollection(*[ScienceInput(map_path.name) for map_path in combined_inputs]),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=combined_start_date,
                end_date=combined_end_date,
                version=VersionMap({}, Version(None, 2)),
                descriptor=combined_descriptor
            )
        ).process()

        combined_spx_descriptor = combined_descriptor.replace(
            "-enasbsMsk-", "-spxsbsMsk-"
        ).replace(
            "-enansnbsMsk-", "-spxnsnbsMsk-"
        ).replace(
            "-enasnbsMsk-", "-spxsnbsMsk-"
        )

        [combined_spx_map] = LoProcessor(
            dependencies=ProcessingInputCollection(ScienceInput(combined_sp_map.name)),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=combined_start_date,
                end_date=combined_end_date,
                version=VersionMap({}, Version(None, 2)),
                descriptor=combined_spx_descriptor,
            ),
        ).process()

        output_maps.extend([
            combined_sp_map,
            combined_spx_map
        ])

    release_directory = get_run_local_data_path("IMAP-Lo June 2nd 2026 Maps")
    copy_to_output_directory_and_rename_for_initial_release(release_directory, output_maps)
