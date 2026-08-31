from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import imap_data_access
from imap_data_access import ProcessingInputCollection

from imap_l3_processing.glows.descriptors import (
    GLOWS_L3D_DESCRIPTOR,
    GLOWS_L3E_DESCRIPTORS,
    GLOWS_REPROCESSING_DESCRIPTOR,
)
from imap_l3_processing.glows.l3e.glows_l3e_utils import (
    get_repoint_numbers_within_cr_window,
)

SUPPORTED_DATA_LEVELS = ["l3e", "l3d"]
REPOINT_PREFIX = "repoint"
CARRINGTON_ROTATION_PREFIX = "cr"
IGNORE_AFTER_PREFIX = "ignore-after:"


@dataclass
class ReprocessTargets:
    repoints: list[int]
    carrington_rotations: list[int]


@dataclass
class ReprocessInfo:
    products_to_reprocess: dict[str, ReprocessTargets]

    @classmethod
    def parse_from_ancillary(cls, path_to_ancillary: Path) -> ReprocessInfo:
        with open(path_to_ancillary) as file:
            ignore_after = cls._parse_header(file.readline())
            parsed_products = [cls._parse_line(line) for line in file if line.strip()]

        if datetime.now(timezone.utc) >= ignore_after:
            products = {}
        else:
            products = {
                descriptor: target
                for data_level, descriptor, target in parsed_products
                if data_level in SUPPORTED_DATA_LEVELS
            }
        return cls(products)

    @staticmethod
    def _parse_header(line: str) -> datetime:
        ignore_after = datetime.fromisoformat(
            line.removeprefix(IGNORE_AFTER_PREFIX).strip()
        )
        if ignore_after.tzinfo is None:
            ignore_after = ignore_after.replace(tzinfo=timezone.utc)
        return ignore_after


    @staticmethod
    def _parse_line(line: str) -> tuple[str, str, ReprocessTargets]:
        product_name, _, remainder = line.strip().partition(" ")
        data_level, _, descriptor = product_name.partition("_")

        targets = remainder.replace(",", " ").split()
        repoints = [int(target.removeprefix(REPOINT_PREFIX)) for target in targets if target.startswith(REPOINT_PREFIX)]
        carrington_rotations = [int(target.removeprefix(CARRINGTON_ROTATION_PREFIX)) for target in targets if
                                target.startswith(CARRINGTON_ROTATION_PREFIX)]

        return data_level, descriptor, ReprocessTargets(repoints, carrington_rotations)

    def should_reprocess_l3d(self) -> bool:
        return GLOWS_L3D_DESCRIPTOR in self.products_to_reprocess or any(
            set(self.products_to_reprocess.keys()).intersection(
                set(GLOWS_L3E_DESCRIPTORS)
            )
        )

    def get_repoints_for_descriptor(
        self, descriptor: str, repointing_data
    ) -> list[int]:
        reprocess_targets = self.products_to_reprocess.get(descriptor, None)
        if reprocess_targets is None:
            return []

        repoints = set(reprocess_targets.repoints)

        for cr in reprocess_targets.carrington_rotations:
            repoints.update(get_repoint_numbers_within_cr_window(cr, cr, repointing_data))

        return sorted(repoints)

def fetch_reprocess_info(input_collection: ProcessingInputCollection) -> ReprocessInfo:
    [reprocessing_file_path] = input_collection.get_file_paths(source="glows", descriptor=GLOWS_REPROCESSING_DESCRIPTOR)
    path_to_downloaded_ancillary = imap_data_access.download(reprocessing_file_path.name)
    return ReprocessInfo.parse_from_ancillary(path_to_downloaded_ancillary)