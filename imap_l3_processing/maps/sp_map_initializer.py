import logging
from abc import abstractmethod
from datetime import datetime

from imap_data_access import ScienceFilePath
from imap_data_access.file_validation import Version

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, \
    get_duration_from_map_descriptor, map_descriptor_parts_to_string, Sensor
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce, MapInitializer
from imap_l3_processing.models import InputMetadata, VersionMap

logger = logging.getLogger(__name__)


class SPMapInitializer(MapInitializer):
    def __init__(self, instrument: str, l2_query_results: list[dict[str, str]]):
        super().__init__(instrument)
        self.input_maps_by_descriptor = self.get_latest_version_by_descriptor_and_start_date(l2_query_results)

    @abstractmethod
    def _collect_glows_psets_by_repoint(self, descriptor: MapDescriptorParts) -> dict[int, str]:
        raise NotImplementedError()

    @abstractmethod
    def _get_ancillary_files(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        raise NotImplementedError()

    def get_maps_that_can_be_produced(self, descriptor: str) -> list[PossibleMapToProduce]:
        l3_descriptor_parts = parse_map_descriptor(descriptor)
        map_duration = get_duration_from_map_descriptor(l3_descriptor_parts)

        glows_file_by_repointing = self._collect_glows_psets_by_repoint(l3_descriptor_parts)

        if len(glows_file_by_repointing) == 0:
            logger.info(f"No GLOWS data available for descriptor {descriptor}, no maps will be produced!")

        l2_descriptors = self._get_l2_dependencies(l3_descriptor_parts)
        l2_descriptor_strs = [map_descriptor_parts_to_string(parts) for parts in l2_descriptors]
        assert l2_descriptor_strs, f"Expected at least one L2 dependency for l3 map: {descriptor}"

        possible_start_dates = set(self.input_maps_by_descriptor[l2_descriptor_strs[0]].keys())
        for l2_descriptor in l2_descriptor_strs[1:]:
            possible_start_dates.intersection_update(self.input_maps_by_descriptor[l2_descriptor].keys())

        possible_maps = []
        for str_start_date in sorted(list(possible_start_dates)):
            start_date = datetime.strptime(str_start_date, "%Y%m%d")

            l2_file_paths = []
            for l2_descriptor in l2_descriptor_strs:
                l2_file_paths.append(self.input_maps_by_descriptor[l2_descriptor][str_start_date])

            if l3_descriptor_parts.sensor in [Sensor.UltraCombined, Sensor.HiCombined]:
                l1c_names = [l1c for l2 in l2_file_paths for l1c in self.get_l1c_parents_from_map(l2)]

            else:
                l1c_names = self.get_l1c_parents_from_map(l2_file_paths[0])

                mismatch = False
                for l2_path in l2_file_paths[1:]:
                    if set(self.get_l1c_parents_from_map(l2_path)) != set(l1c_names):
                        mismatch = True
                        break

                if mismatch:
                    logger.warning(
                        f"Expected all input maps to be created from the same pointing sets! l2_file_paths: "
                        f"{', '.join(l2_file_paths)}"
                    )
                    continue

            l1c_repointings = [ScienceFilePath(l1c).repointing for l1c in l1c_names]

            glows_files = [glows_file_by_repointing[repoint] for repoint in l1c_repointings if
                           repoint in glows_file_by_repointing]

            if len(glows_files) == 0:
                logger.info(f"No GLOWS data available for l3 map: {descriptor} with start date: {str_start_date}. Skipping")
                continue

            input_metadata = InputMetadata(instrument=self.instrument, data_level='l3', start_date=start_date,
                                           end_date=start_date + map_duration, version=VersionMap({}, Version(None, 1)),
                                           descriptor=descriptor)
            possible_map_to_produce = PossibleMapToProduce(
                input_files=set(l2_file_paths + glows_files + l1c_names + self._get_ancillary_files()),
                input_metadata=input_metadata
            )
            possible_maps.append(possible_map_to_produce)
        return possible_maps
