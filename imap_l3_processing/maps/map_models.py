from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import timedelta, datetime
from pathlib import Path
from typing import Generic, Optional

import numpy as np
import xarray
from imap_processing.ena_maps.ena_maps import HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable_and_mask_fill_values, read_numeric_variable
from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.maps.quality_flags import MapL3Flags
from imap_l3_processing.models import DataProduct, DataProductVariable, D

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
ENERGY_LABEL_VAR_NAME = "energy_label"

LONGITUDE_VAR_NAME = "longitude"
LONGITUDE_DELTA_VAR_NAME = "longitude_delta"
LONGITUDE_LABEL_VAR_NAME = "longitude_label"
LATITUDE_VAR_NAME = "latitude"
LATITUDE_DELTA_VAR_NAME = "latitude_delta"
LATITUDE_LABEL_VAR_NAME = "latitude_label"

EXPOSURE_FACTOR_VAR_NAME = "exposure_factor"
OBS_DATE_VAR_NAME = "obs_date"
OBS_DATE_RANGE_VAR_NAME = "obs_date_range"
SOLID_ANGLE_VAR_NAME = "solid_angle"
ENA_SPECTRAL_INDEX_VAR_NAME = "ena_spectral_index"
ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME = "ena_spectral_index_stat_uncert"
ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_VAR_NAME = "ena_spectral_scalar"
ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_STAT_UNCERT_VAR_NAME = "ena_spectral_scalar_stat_uncert"
ENA_SPECTRAL_INDEX_CHISQ_VAR_NAME = "ena_spectral_index_chisq"

ENA_INTENSITY_VAR_NAME = "ena_intensity"
ENA_INTENSITY_STAT_UNCERT_VAR_NAME = "ena_intensity_stat_uncert"
ENA_INTENSITY_SYS_ERR_VAR_NAME = "ena_intensity_sys_err"

BG_INTENSITY_VAR_NAME = "bg_intensity"
BG_INTENSITY_STAT_UNC_VAR_NAME = "bg_intensity_stat_uncert"
BG_INTENSITY_SYS_ERR_VAR_NAME = "bg_intensity_sys_err"

SURVIVAL_PROBABILITY_VAR_NAME = "survival_probability"

PIXEL_INDEX_VAR_NAME = "pixel_index"
PIXEL_INDEX_LABEL_VAR_NAME = "pixel_index_label"

BG_RATE_VAR_NAME = "bg_rate"
BG_RATE_STAT_UNCERT_VAR_NAME = "bg_rate_stat_uncert"
BG_RATE_SYS_ERR_VAR_NAME = "bg_rate_sys_err"
ENA_COUNT_RATE_VAR_NAME = "ena_count_rate"
ENA_COUNT_RATE_STAT_UNCERT_VAR_NAME = "ena_count_rate_stat_uncert"
ISN_BG_RATE_SUBTRACTED_VAR_NAME = "isn_rate_bg_subtracted"
ISN_BG_RATE_SUBTRACTED_STAT_UNCERT_VAR_NAME = "isn_rate_bg_subtracted_stat_uncert"
ISN_BG_RATE_SUBTRACTED_VAR_SYS_ERR_NAME = "isn_rate_bg_subtracted_sys_err"

ENA_INTENSITY_SYS_ERR_MINUS_VAR_NAME = "ena_intensity_sys_err_minus"
ENA_INTENSITY_SYS_ERR_PLUS_VAR_NAME = "ena_intensity_sys_err_plus"

QUALITY_FLAGS_VAR_NAME = "quality_flags"


@dataclass
class MapData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    energy_label: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    exposure_factor: np.ndarray
    obs_date: np.ndarray
    obs_date_range: np.ndarray
    solid_angle: np.ndarray


@dataclass
class HealPixCoords:
    pixel_index: np.ndarray
    pixel_index_label: np.ndarray

    @property
    def nside(self) -> int:
        return int(np.sqrt(len(self.pixel_index) / 12))


@dataclass
class RectangularCoords:
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray


@dataclass
class IntensityMapData(MapData):
    ena_intensity: np.ndarray
    ena_intensity_stat_uncert: np.ndarray
    ena_intensity_sys_err: np.ndarray
    quality_flags: np.ndarray
    ena_intensity_sys_err_minus: Optional[np.ndarray] = None
    ena_intensity_sys_err_plus: Optional[np.ndarray] = None
    bg_intensity: Optional[np.ndarray] = None
    bg_intensity_stat_uncert: Optional[np.ndarray] = None
    bg_intensity_sys_err: Optional[np.ndarray] = None
    survival_probability: Optional[np.ndarray] = None


@dataclass
class ISNBackgroundSubtractedData(MapData):
    bg_rate: np.ndarray
    bg_rate_stat_uncert: np.ndarray
    bg_rate_sys_err: np.ndarray
    isn_bg_rate_subtracted_stat_uncert: np.ndarray
    isn_bg_rate_subtracted_sys_err: np.ndarray
    isn_bg_rate_subtracted: np.ndarray
    latitude_label: np.ndarray
    latitude_delta: np.ndarray
    longitude_label: np.ndarray
    longitude_delta: np.ndarray


@dataclass
class SpectralIndexMapData(MapData):
    ena_spectral_index: np.ndarray
    ena_spectral_index_stat_uncert: np.ndarray
    ena_spectral_index_scalar_coefficient: np.ndarray
    ena_spectral_index_scalar_coefficient_stat_uncert: np.ndarray
    ena_spectral_index_chisq: np.ndarray
    quality_flags: np.ndarray


@dataclass
class RectangularIntensityMapData:
    intensity_map_data: IntensityMapData
    coords: RectangularCoords

    @classmethod
    def read_from_path(cls, cdf_path: Path | str) -> RectangularIntensityMapData:
        with CDF(str(cdf_path)) as cdf:
            return RectangularIntensityMapData(
                intensity_map_data=_read_intensity_map_data_from_open_cdf(cdf),
                coords=_read_rectangular_coords_from_open_cdf(cdf),
            )


@dataclass
class ISNBackgroundSubtractedMapData:
    isn_rate_map_data: ISNBackgroundSubtractedData


@dataclass
class ISNRateData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    ena_intensity: np.ndarray
    ena_intensity_stat_uncert: np.ndarray
    ena_intensity_sys_err: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    energy_label: np.ndarray
    exposure_factor: np.ndarray
    solid_angle: np.ndarray
    bg_rate: np.ndarray
    bg_rate_stat_uncert: np.ndarray
    bg_rate_sys_err: np.ndarray
    ena_count_rate: np.ndarray
    ena_count_rate_stat_uncert: np.ndarray
    latitude: np.ndarray
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray
    obs_date: np.ndarray
    obs_date_range: np.ndarray

    @classmethod
    def read_from_path(cls, cdf_path: Path | str) -> ISNRateData:

        with CDF(str(cdf_path)) as cdf:
            if "obs_date" in cdf and "obs_date_range" in cdf:
                masked_obs_date = read_variable_and_mask_fill_values(cdf["obs_date"])
                if np.issubdtype(cdf["obs_date"].dtype, np.number):
                    obs_date = convert_tt2000_time_to_datetime(masked_obs_date.filled(0))
                    masked_obs_date = np.ma.masked_array(data=obs_date, mask=masked_obs_date.mask)
                obs_date_range = read_variable_and_mask_fill_values(cdf["obs_date_range"])
            else:
                obs_date_shape = cdf["bg_rate"].shape
                all_mask_array = np.full(obs_date_shape, True)
                masked_obs_date = np.ma.masked_array(np.full(obs_date_shape, TT2000_EPOCH), mask=all_mask_array)
                obs_date_range = np.ma.masked_array(np.full(obs_date_shape, 0),
                                                    mask=all_mask_array,
                                                    dtype=np.int64)

            return ISNRateData(
                epoch=cdf['epoch'][...],
                epoch_delta=read_variable_and_mask_fill_values(cdf['epoch_delta']),
                obs_date=masked_obs_date,
                obs_date_range=obs_date_range,
                bg_rate=read_numeric_variable(cdf['bg_rate']),
                bg_rate_stat_uncert=read_numeric_variable(cdf['bg_rate_stat_uncert']),
                bg_rate_sys_err=read_numeric_variable(cdf['bg_rate_sys_err']),
                ena_count_rate=read_numeric_variable(cdf['ena_count_rate']),
                ena_count_rate_stat_uncert=read_numeric_variable(cdf['ena_count_rate_stat_uncert']),
                latitude=(cdf['latitude'][...]),
                latitude_delta=(cdf['latitude_delta'][...]),
                latitude_label=(cdf['latitude_label'][...]),
                longitude=(cdf['longitude'][...]),
                longitude_delta=(cdf['longitude_delta'][...]),
                longitude_label=(cdf['longitude_label'][...]),
                ena_intensity=read_numeric_variable(cdf['ena_intensity']),
                ena_intensity_stat_uncert=read_numeric_variable(cdf['ena_intensity_stat_uncert']),
                ena_intensity_sys_err=read_numeric_variable(cdf['ena_intensity_sys_err']),
                energy=(cdf['energy'][...]),
                energy_delta_plus=(cdf['energy_delta_plus'][...]),
                energy_delta_minus=(cdf['energy_delta_minus'][...]),
                energy_label=(cdf['energy_label'][...]),
                exposure_factor=read_numeric_variable(cdf['exposure_factor']),
                solid_angle=read_numeric_variable(cdf['solid_angle'])
            )


@dataclass
class RectangularSpectralIndexMapData:
    spectral_index_map_data: SpectralIndexMapData
    coords: RectangularCoords


@dataclass
class HealPixIntensityMapData:
    intensity_map_data: IntensityMapData
    coords: HealPixCoords

    @classmethod
    def read_from_path(cls, cdf_path: Path | str) -> HealPixIntensityMapData:
        with CDF(str(cdf_path)) as cdf:
            return HealPixIntensityMapData(
                intensity_map_data=_read_intensity_map_data_from_open_cdf(cdf),
                coords=_read_healpix_coords_from_open_cdf(cdf),
            )

    def to_healpix_skymap(self) -> HealpixSkyMap:
        healpix_map = HealpixSkyMap(self.coords.nside, SpiceFrame.ECLIPJ2000)

        full_shape = [
            CoordNames.TIME.value,
            CoordNames.ENERGY_L2.value,
            CoordNames.HEALPIX_INDEX.value,
        ]
        healpix_map.data_1d = xarray.Dataset(
            data_vars={
                "latitude": (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.intensity_map_data.latitude,
                ),
                "longitude": (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.intensity_map_data.longitude,
                ),
                "solid_angle": (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.intensity_map_data.solid_angle,
                ),
                "obs_date_range": (full_shape, self.intensity_map_data.obs_date_range),
                "obs_date": (full_shape, self.intensity_map_data.obs_date),
                "exposure_factor": (
                    full_shape,
                    self.intensity_map_data.exposure_factor,
                ),
                "ena_intensity": (full_shape, self.intensity_map_data.ena_intensity),
                "ena_intensity_stat_uncert": (
                    full_shape,
                    self.intensity_map_data.ena_intensity_stat_uncert,
                ),
                "ena_intensity_sys_err": (
                    full_shape,
                    self.intensity_map_data.ena_intensity_sys_err,
                ),
                "predicted_ephemeris_flag": (
                    full_shape,
                    self.intensity_map_data.quality_flags & MapL3Flags.PREDICTIVE_EPHEMERIS,
                ),
                "nominal_alpha_proton_ratio_flag": (
                    full_shape,
                    self.intensity_map_data.quality_flags & MapL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
                ),
                "persisted_last_point_flag": (
                    full_shape,
                    self.intensity_map_data.quality_flags & MapL3Flags.PERSISTED_LAST_POINT,
                )

            },
            coords={
                CoordNames.TIME.value: self.intensity_map_data.epoch,
                CoordNames.ENERGY_L2.value: self.intensity_map_data.energy,
                CoordNames.HEALPIX_INDEX.value: self.coords.pixel_index,
            },
        )

        if self.intensity_map_data.survival_probability is not None:
            data_1d_with_sp = healpix_map.data_1d.assign(
                survival_probability=(full_shape, self.intensity_map_data.survival_probability))
            healpix_map.data_1d = data_1d_with_sp

        healpix_map.data_1d = healpix_map.data_1d \
            .assign({"obs_date": (full_shape, healpix_map.data_1d["obs_date"].values.astype(np.float64))}) \
            .rename({CoordNames.HEALPIX_INDEX.value: CoordNames.GENERIC_PIXEL.value})

        return healpix_map

    @classmethod
    def read_from_xarray(cls, input_dataset):
        return HealPixIntensityMapData(
            intensity_map_data=_read_intensity_map_data_from_xarray(input_dataset),
            coords=_read_healpix_coords_from_xarray(input_dataset),
        )


@dataclass
class SpectralIndexDependencies(metaclass=abc.ABCMeta):
    map_data: RectangularIntensityMapData | HealPixIntensityMapData

    @abc.abstractmethod
    def get_fit_energy_ranges(self) -> np.ndarray:
        raise NotImplementedError


def convert_tt2000_time_to_datetime(time: np.ndarray) -> np.ndarray:
    return time / 1e9 * timedelta(seconds=1) + TT2000_EPOCH


def _read_intensity_map_data_from_open_cdf(cdf: CDF) -> IntensityMapData:
    if "obs_date" in cdf and "obs_date_range" in cdf:
        masked_obs_date = read_variable_and_mask_fill_values(cdf["obs_date"])
        if np.issubdtype(cdf["obs_date"].dtype, np.number):
            obs_date = convert_tt2000_time_to_datetime(masked_obs_date.filled(0))
            masked_obs_date = np.ma.masked_array(
                data=obs_date, mask=masked_obs_date.mask
            )
        obs_date_range = read_variable_and_mask_fill_values(cdf["obs_date_range"])
    else:
        obs_date_shape = cdf["ena_intensity"].shape
        all_mask_array = np.full(obs_date_shape, True)
        masked_obs_date = np.ma.masked_array(
            np.full(obs_date_shape, TT2000_EPOCH), mask=all_mask_array
        )
        obs_date_range = np.ma.masked_array(
            np.full(obs_date_shape, 0), mask=all_mask_array, dtype=np.int64
        )
    quality_flag_data = np.full(cdf["ena_intensity"].shape, MapL3Flags.NONE)
    if QUALITY_FLAGS_VAR_NAME in cdf:
        quality_flag_data = cdf[QUALITY_FLAGS_VAR_NAME][...]
    map_intensity_data = IntensityMapData(
        epoch=cdf["epoch"][...],
        epoch_delta=read_variable_and_mask_fill_values(cdf["epoch_delta"]),
        energy=read_numeric_variable(cdf["energy"]),
        energy_delta_plus=read_numeric_variable(cdf["energy_delta_plus"]),
        energy_delta_minus=read_numeric_variable(cdf["energy_delta_minus"]),
        energy_label=cdf["energy_label"][...],
        latitude=read_numeric_variable(cdf["latitude"]),
        longitude=read_numeric_variable(cdf["longitude"]),
        exposure_factor=read_numeric_variable(cdf["exposure_factor"]),
        obs_date=masked_obs_date,
        obs_date_range=obs_date_range,
        solid_angle=read_numeric_variable(cdf["solid_angle"]),
        ena_intensity=read_numeric_variable(cdf["ena_intensity"]),
        ena_intensity_stat_uncert=read_numeric_variable(
            cdf["ena_intensity_stat_uncert"]
        ),
        ena_intensity_sys_err=read_numeric_variable(cdf["ena_intensity_sys_err"]),
        quality_flags=quality_flag_data,
    )

    if "survival_probability" in cdf:
        map_intensity_data.survival_probability = read_numeric_variable(cdf["survival_probability"])

    if "bg_intensity" in cdf:
        map_intensity_data.bg_intensity = read_numeric_variable(cdf["bg_intensity"])
        map_intensity_data.bg_intensity_sys_err = read_numeric_variable(cdf["bg_intensity_sys_err"])
        map_intensity_data.bg_intensity_stat_uncert = read_numeric_variable(cdf["bg_intensity_stat_uncert"])

    if "ena_intensity_sys_err_minus" in cdf:
        map_intensity_data.ena_intensity_sys_err_minus = read_numeric_variable(cdf["ena_intensity_sys_err_minus"])
        map_intensity_data.ena_intensity_sys_err_plus = read_numeric_variable(cdf["ena_intensity_sys_err_plus"])

    return map_intensity_data


def _read_healpix_coords_from_open_cdf(cdf: CDF) -> HealPixCoords:
    return HealPixCoords(
        pixel_index=cdf["pixel_index"][...],
        pixel_index_label=cdf["pixel_index_label"][...]
    )


def _read_intensity_map_data_from_xarray(dataset: xarray.Dataset) -> IntensityMapData:
    intensity = _replace_fill_values_in_xarray(dataset, "ena_intensity")
    return IntensityMapData(
        epoch=dataset[CoordNames.TIME.value].values,
        epoch_delta=_replace_fill_values_in_xarray(dataset, "epoch_delta"),
        energy=dataset.coords[CoordNames.ENERGY_L2.value].values,
        energy_delta_plus=_replace_fill_values_in_xarray(dataset, "energy_delta_plus"),
        energy_delta_minus=_replace_fill_values_in_xarray(dataset, "energy_delta_minus"),
        energy_label=_replace_fill_values_in_xarray(dataset, "energy_label"),
        latitude=_replace_fill_values_in_xarray(dataset, "latitude"),
        longitude=_replace_fill_values_in_xarray(dataset, "longitude"),
        exposure_factor=_replace_fill_values_in_xarray(dataset, "exposure_factor"),
        obs_date=_replace_fill_values_in_xarray(dataset, "obs_date"),
        obs_date_range=_replace_fill_values_in_xarray(dataset, "obs_date_range"),
        solid_angle=np.squeeze(_replace_fill_values_in_xarray(dataset, "solid_angle")),
        ena_intensity=intensity,
        ena_intensity_stat_uncert=_replace_fill_values_in_xarray(dataset, "ena_intensity_stat_uncert"),
        ena_intensity_sys_err=_replace_fill_values_in_xarray(dataset, "ena_intensity_sys_err"),
        quality_flags=np.full(intensity.shape, MapL3Flags.NONE),
    )


def _replace_fill_values_in_xarray(dataset, variable):
    if 'FILLVAL' in dataset[variable].attrs:
        return np.where(dataset[variable].values == dataset[variable].attrs['FILLVAL'], np.nan,
                        dataset[variable].values)
    return dataset[variable].values


def _read_healpix_coords_from_xarray(dataset: xarray.Dataset) -> HealPixCoords:
    return HealPixCoords(
        pixel_index=dataset[CoordNames.HEALPIX_INDEX.value].values,
        pixel_index_label=dataset["pixel_index_label"].values
    )


def _read_rectangular_coords_from_open_cdf(cdf: CDF) -> RectangularCoords:
    return RectangularCoords(
        latitude_delta=cdf["latitude_delta"][...],
        latitude_label=cdf["latitude_label"][...],
        longitude_delta=cdf["longitude_delta"][...],
        longitude_label=cdf["longitude_label"][...],
    )


@dataclass
class MapDataProduct(DataProduct[D], Generic[D]):
    data: D
    spice_frame_name: SpiceFrame

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplementedError


class RectangularSpectralIndexDataProduct(MapDataProduct[RectangularSpectralIndexMapData]):
    data: RectangularSpectralIndexMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _spectral_index_data_variables(self.data.spectral_index_map_data) \
            + _rectangular_coords_to_variables(self.data.coords)


class RectangularIntensityDataProduct(MapDataProduct[RectangularIntensityMapData]):
    data: RectangularIntensityMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _intensity_data_variables(self.data.intensity_map_data) \
            + _rectangular_coords_to_variables(self.data.coords)


class ISNBackgroundSubtractedDataProduct(MapDataProduct[ISNBackgroundSubtractedMapData]):
    data: ISNBackgroundSubtractedMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.data.isn_rate_map_data.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.data.isn_rate_map_data.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.data.isn_rate_map_data.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.data.isn_rate_map_data.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.data.isn_rate_map_data.energy_delta_minus),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, self.data.isn_rate_map_data.energy_label),
            DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, self.data.isn_rate_map_data.exposure_factor),
            DataProductVariable(SOLID_ANGLE_VAR_NAME, self.data.isn_rate_map_data.solid_angle),
            DataProductVariable(BG_RATE_VAR_NAME, self.data.isn_rate_map_data.bg_rate),
            DataProductVariable(BG_RATE_STAT_UNCERT_VAR_NAME, self.data.isn_rate_map_data.bg_rate_stat_uncert),
            DataProductVariable(BG_RATE_SYS_ERR_VAR_NAME, self.data.isn_rate_map_data.bg_rate_sys_err),
            DataProductVariable(LATITUDE_VAR_NAME, self.data.isn_rate_map_data.latitude),
            DataProductVariable(LATITUDE_DELTA_VAR_NAME, self.data.isn_rate_map_data.latitude_delta),
            DataProductVariable(LATITUDE_LABEL_VAR_NAME, self.data.isn_rate_map_data.latitude_label),
            DataProductVariable(LONGITUDE_VAR_NAME, self.data.isn_rate_map_data.longitude),
            DataProductVariable(LONGITUDE_DELTA_VAR_NAME, self.data.isn_rate_map_data.longitude_delta),
            DataProductVariable(LONGITUDE_LABEL_VAR_NAME, self.data.isn_rate_map_data.longitude_label),
            DataProductVariable(ISN_BG_RATE_SUBTRACTED_VAR_NAME,
                                self.data.isn_rate_map_data.isn_bg_rate_subtracted),
            DataProductVariable(ISN_BG_RATE_SUBTRACTED_VAR_SYS_ERR_NAME,
                                self.data.isn_rate_map_data.isn_bg_rate_subtracted_sys_err),
            DataProductVariable(ISN_BG_RATE_SUBTRACTED_STAT_UNCERT_VAR_NAME,
                                self.data.isn_rate_map_data.isn_bg_rate_subtracted_stat_uncert)
        ]


def _map_data_to_variables(data: MapData) -> list[DataProductVariable]:
    return [
        DataProductVariable(EPOCH_VAR_NAME, data.epoch),
        DataProductVariable(EPOCH_DELTA_VAR_NAME, data.epoch_delta),
        DataProductVariable(ENERGY_VAR_NAME, data.energy),
        DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, data.energy_delta_plus),
        DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, data.energy_delta_minus),
        DataProductVariable(ENERGY_LABEL_VAR_NAME, data.energy_label),
        DataProductVariable(LATITUDE_VAR_NAME, data.latitude),
        DataProductVariable(LONGITUDE_VAR_NAME, data.longitude),
        DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, data.exposure_factor),
        DataProductVariable(OBS_DATE_VAR_NAME, data.obs_date),
        DataProductVariable(OBS_DATE_RANGE_VAR_NAME, data.obs_date_range),
        DataProductVariable(SOLID_ANGLE_VAR_NAME, data.solid_angle),
    ]


def _spectral_index_data_variables(data: SpectralIndexMapData) -> list[DataProductVariable]:
    variables = _map_data_to_variables(data) + [
        DataProductVariable(ENA_SPECTRAL_INDEX_VAR_NAME, data.ena_spectral_index),
        DataProductVariable(
            ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, data.ena_spectral_index_stat_uncert
        ),
        DataProductVariable(
            ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_VAR_NAME,
            data.ena_spectral_index_scalar_coefficient,
        ),
        DataProductVariable(
            ENA_SPECTRAL_INDEX_SCALAR_COEFFICIENT_STAT_UNCERT_VAR_NAME,
            data.ena_spectral_index_scalar_coefficient_stat_uncert,
        ),
        DataProductVariable(
            ENA_SPECTRAL_INDEX_CHISQ_VAR_NAME, data.ena_spectral_index_chisq
        ),
        DataProductVariable(QUALITY_FLAGS_VAR_NAME, data.quality_flags),
    ]
    return variables


def _intensity_data_variables(data: IntensityMapData) -> list[DataProductVariable]:
    intensity_variables = [
        DataProductVariable(ENA_INTENSITY_VAR_NAME, data.ena_intensity),
        DataProductVariable(ENA_INTENSITY_STAT_UNCERT_VAR_NAME, data.ena_intensity_stat_uncert),
        DataProductVariable(ENA_INTENSITY_SYS_ERR_VAR_NAME, data.ena_intensity_sys_err),
        DataProductVariable(QUALITY_FLAGS_VAR_NAME, data.quality_flags),
    ]
    if data.bg_intensity is not None:
        intensity_variables.extend(
            [
                DataProductVariable(BG_INTENSITY_VAR_NAME, data.bg_intensity),
                DataProductVariable(
                    BG_INTENSITY_STAT_UNC_VAR_NAME, data.bg_intensity_stat_uncert
                ),
                DataProductVariable(
                    BG_INTENSITY_SYS_ERR_VAR_NAME, data.bg_intensity_sys_err
                ),
            ]
        )
    if data.survival_probability is not None:
        intensity_variables.extend([
            DataProductVariable(SURVIVAL_PROBABILITY_VAR_NAME, data.survival_probability),
        ])
    if data.ena_intensity_sys_err_minus is not None:
        intensity_variables.extend(
            [DataProductVariable(ENA_INTENSITY_SYS_ERR_MINUS_VAR_NAME, data.ena_intensity_sys_err_minus),
             DataProductVariable(ENA_INTENSITY_SYS_ERR_PLUS_VAR_NAME, data.ena_intensity_sys_err_plus)])
    return _map_data_to_variables(data) + intensity_variables


def _rectangular_coords_to_variables(coords: RectangularCoords) -> list[DataProductVariable]:
    return [
        DataProductVariable(LATITUDE_DELTA_VAR_NAME, coords.latitude_delta),
        DataProductVariable(LATITUDE_LABEL_VAR_NAME, coords.latitude_label),
        DataProductVariable(LONGITUDE_DELTA_VAR_NAME, coords.longitude_delta),
        DataProductVariable(LONGITUDE_LABEL_VAR_NAME, coords.longitude_label),
    ]



def calculate_datetime_weighted_average(data: np.ndarray, weights: np.ndarray, axis: int,
                                        **kwargs) -> np.ma.masked_array:
    if isinstance(np.ravel(np.ma.getdata(data))[0], datetime):
        masked_indices = np.ma.getmask(data)
        masked_weights = weights.copy()
        masked_weights[masked_indices] = 0

        epoch_based_dates = np.array((np.ma.getdata(data) - TT2000_EPOCH) / timedelta(seconds=1),
                                     dtype=float)

        averaged_dates_as_seconds = np.ma.average(
            epoch_based_dates, weights=masked_weights, axis=axis, **kwargs
        )

        return np.ma.array(
            averaged_dates_as_seconds.data * timedelta(seconds=1) + TT2000_EPOCH,
            mask=averaged_dates_as_seconds.mask,
        )
    else:
        return np.ma.average(data, weights=weights, axis=axis, **kwargs)


@dataclass
class GlowsL3eRectangularMapInputData:
    epoch: datetime
    epoch_j2000: np.ndarray
    repointing: int
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray
    flags: np.ndarray


@dataclass
class InputRectangularPointingSet:
    epoch: datetime
    epoch_delta: Optional[np.ndarray]
    epoch_j2000: np.ndarray
    repointing: int
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray
    pointing_start_met: Optional[float]
    pointing_end_met: Optional[float]
    hae_longitude: np.ndarray
    hae_latitude: np.ndarray
