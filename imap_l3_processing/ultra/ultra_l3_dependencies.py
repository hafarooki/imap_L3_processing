from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Optional

import imap_data_access
import numpy as np
from imap_data_access import ScienceInput
from imap_data_access.processing_input import ProcessingInputCollection
from imap_processing.ultra.l2.ultra_l2 import ultra_l2

from imap_l3_processing.maps.map_models import HealPixIntensityMapData, RectangularIntensityMapData, \
    SpectralIndexDependencies
from imap_l3_processing.ultra.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.utils import get_dependency_paths_by_descriptor, get_temp_cache_dir

logger = logging.getLogger(__name__)


@dataclass
class UltraL3Dependencies:
    ultra_l2_healpix_map: HealPixIntensityMapData
    ultra_l2_rectangular_map: RectangularIntensityMapData
    ultra_l1c_pset: list[UltraL1CPSet]
    glows_l3e_sp: list[UltraGlowsL3eData]
    energy_bin_group_sizes: Optional[np.ndarray]
    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3Dependencies:
        ultra_l2_names = deps.get_file_paths("ultra", data_type="l2")
        assert len(ultra_l2_names) == 1, f"Incorrect number of map dependencies: {len(ultra_l2_names)}"
        ultra_l2_name = ultra_l2_names[0]


        ultra_l1c_names = deps.get_file_paths("ultra", data_type="l1c")
        glows_l3e_names = deps.get_file_paths("glows")
        energy_bin_group_sizes_names = deps.get_file_paths(data_type='ancillary',
                                                           descriptor='l2-energy-bin-group-sizes')

        l2_map_path = imap_data_access.download(ultra_l2_name)
        ultra_l1c_downloaded_paths = [imap_data_access.download(l1c) for l1c in ultra_l1c_names]
        glows_l3e_download_paths = [imap_data_access.download(path) for path in glows_l3e_names]

        if energy_bin_group_sizes_names:
            energy_bin_group_sizes_path = imap_data_access.download(energy_bin_group_sizes_names[0])
        else:
            energy_bin_group_sizes_path = None

        return cls.from_file_paths(l2_map_path, ultra_l1c_downloaded_paths, glows_l3e_download_paths,
                                   energy_bin_group_sizes_path)

    @classmethod
    def from_file_paths(cls, l2_map_path: Path, l1c_file_paths: list[Path], glows_file_paths: list[Path],
                        energy_bin_path: Optional[Path]) -> UltraL3Dependencies:
        ultra_l1c_data = []
        glows_l3e_data = []
        for file_path in l1c_file_paths:
            ultra_l1c_data.append(UltraL1CPSet.read_from_path(file_path))
        for file_path in glows_file_paths:
            glows_l3e_data.append(UltraGlowsL3eData.read_from_path(file_path))
        paths = [l2_map_path] + l1c_file_paths + glows_file_paths

        l2_healpix_map_data = load_or_create_healpix_l2(l2_map_path, l1c_file_paths)

        l2_rectangular_map_data = RectangularIntensityMapData.read_from_path(l2_map_path)
        energy_bin_group_sizes = None
        if energy_bin_path:
            paths.append(energy_bin_path)
            energy_bin_group_sizes = np.loadtxt(energy_bin_path, delimiter=',', dtype=np.uint8)

        return cls(ultra_l2_healpix_map=l2_healpix_map_data, ultra_l2_rectangular_map=l2_rectangular_map_data, ultra_l1c_pset=ultra_l1c_data, glows_l3e_sp=glows_l3e_data,
                   dependency_file_paths=paths, energy_bin_group_sizes=energy_bin_group_sizes)


@dataclass
class UltraL3SpectralIndexDependencies(SpectralIndexDependencies):
    fit_energy_ranges: np.ndarray

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3SpectralIndexDependencies:
        energy_fit_ranges_ancillary_file_path = deps.get_file_paths(source="ultra", descriptor="spx-energy-ranges")
        ultra_map_file_paths = deps.get_file_paths(source="ultra", data_type="l3") + deps.get_file_paths(source="ultra", data_type="l2")

        if len(ultra_map_file_paths) != 1:
            raise ValueError(f"Expected 1 input map, got {len(ultra_map_file_paths)}: [{', '.join(p.name for p in ultra_map_file_paths)}]")

        if len(energy_fit_ranges_ancillary_file_path) != 1:
            raise ValueError("Missing fit energy ranges ancillary file")

        map_file_path = imap_data_access.download(ultra_map_file_paths[0].name)
        energy_ranges_file_path = imap_data_access.download(energy_fit_ranges_ancillary_file_path[0].name)

        return cls.from_file_paths(map_file_path, energy_ranges_file_path)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, energy_fit_ranges_path: Path):
        map_data = RectangularIntensityMapData.read_from_path(map_file_path)
        energy_fit_ranges = np.loadtxt(energy_fit_ranges_path)
        return cls(map_data, energy_fit_ranges)

    def get_fit_energy_ranges(self) -> np.ndarray:
        return self.fit_energy_ranges


@dataclass
class UltraL3CombinedDependencies:
    u45_dependencies: UltraL3Dependencies
    u90_dependencies: UltraL3Dependencies

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3CombinedDependencies:
        descriptors = ["u45", "u90", "45sensor", "90sensor", "survival-probability-ul"]
        file_paths = get_dependency_paths_by_descriptor(deps=deps, descriptors=descriptors)

        assert len(file_paths['u45']) == 1
        assert len(file_paths['u90']) == 1

        u45_pset_paths = [imap_data_access.download(pset) for pset in file_paths['45sensor']]
        u90_pset_paths = [imap_data_access.download(pset) for pset in file_paths['90sensor']]
        glows_l3e_pset_paths = [imap_data_access.download(pset) for pset in file_paths['survival-probability-ul']]
        u45_map_path = imap_data_access.download(file_paths['u45'][0])
        u90_map_path = imap_data_access.download(file_paths['u90'][0])

        energy_bin_group_sizes_names = deps.get_file_paths(data_type='ancillary',
                                                           descriptor='l2-energy-bin-group-sizes')
        if energy_bin_group_sizes_names:
            energy_bin_group_sizes_path = imap_data_access.download(energy_bin_group_sizes_names[0].name)
        else:
            energy_bin_group_sizes_path = None

        return cls.from_file_paths(u45_pset_paths, u90_pset_paths, glows_l3e_pset_paths, u45_map_path, u90_map_path,
                                   energy_bin_group_sizes_path)

    @classmethod
    def from_file_paths(cls, u45_pset_paths: list[Path], u90_pset_paths: list[Path], glows_l3e_pset_paths: list[Path],
                        u45_map_path: Path, u90_map_path: Path,
                        energy_bin_group_sizes_path: Optional[Path]) -> UltraL3CombinedDependencies:

        return cls(
            u45_dependencies=UltraL3Dependencies.from_file_paths(
                l2_map_path=u45_map_path,
                l1c_file_paths=u45_pset_paths,
                glows_file_paths=glows_l3e_pset_paths,
                energy_bin_path=energy_bin_group_sizes_path
            ),
            u90_dependencies=UltraL3Dependencies.from_file_paths(
                l2_map_path=u90_map_path,
                l1c_file_paths=u90_pset_paths,
                glows_file_paths=glows_l3e_pset_paths,
                energy_bin_path=energy_bin_group_sizes_path
            ),
        )


def load_or_create_healpix_l2(
    l2_map_path: Path, l1c_file_paths: list[Path]
) -> HealPixIntensityMapData:
    l2_descriptor = ScienceInput(l2_map_path.name).descriptor
    l2_descriptor = re.sub(r"[246]deg", "nside32", l2_descriptor)
    sorted_l1c_paths = sorted(l1c_file_paths, key=lambda p: p.name)
    l1c_paths_dict = {path.stem: path for path in sorted_l1c_paths}

    cache_key = str([p.name for p in sorted_l1c_paths]) + l2_descriptor
    cache_filename = sha256(cache_key.encode("utf-8")).hexdigest()
    cache_path = get_temp_cache_dir() / cache_filename
    if cache_path.exists():
        logger.info("Loading cached HEALPix L2 from %s", cache_path)
        with open(cache_path, "rb") as f:
            l2_healpix_data = pickle.load(f)
    else:
        logger.info("Cached HEALPix L2 not found; building from pointing sets")
        l2_healpix_datasets = ultra_l2(l1c_paths_dict, descriptor=l2_descriptor)
        l2_healpix_data = HealPixIntensityMapData.read_from_xarray(l2_healpix_datasets[0])
        logger.info("Saving HEALPix L2 to cache at %s", cache_path)
        with open(cache_path, "wb") as f:
            pickle.dump(l2_healpix_data, f)

    return l2_healpix_data