from datetime import datetime
from typing import Optional

import numpy as np
import spiceypy

from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData, RectangularCoords, \
    InputRectangularPointingSet, GlowsL3eRectangularMapInputData, RectangularSpectralIndexMapData, SpectralIndexMapData
from imap_l3_processing.maps.quality_flags import MapL3Flags


def create_intensity_map_data(epoch=None, epoch_delta=None, lon=None, lat=None, energy=None,
                              energy_delta=None, flux=None,
                              intensity_stat_uncert=None) -> IntensityMapData:
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    epoch_delta = epoch_delta if epoch_delta is not None else np.ma.array([86400 * 1e9])
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=1)
    intensity_stat_uncert = intensity_stat_uncert if intensity_stat_uncert is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)

    if isinstance(flux, np.ndarray):

        more_real_flux = flux

    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return IntensityMapData(
        epoch=epoch,
        epoch_delta=epoch_delta,
        energy=energy,
        energy_delta_plus=energy_delta,
        energy_delta_minus=energy_delta,
        energy_label=np.array(["energy"]),
        latitude=lat,
        longitude=lon,
        exposure_factor=np.full_like(flux, 0),
        obs_date=np.ma.array(
            np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))
        ),
        obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
        solid_angle=np.full_like(more_real_flux, 0),
        ena_intensity=flux,
        ena_intensity_stat_uncert=intensity_stat_uncert,
        ena_intensity_sys_err=flux * 0.001,
        bg_intensity=flux * 0.01,
        bg_intensity_stat_uncert=intensity_stat_uncert * 0.01,
        bg_intensity_sys_err=flux * 0.01 * 0.001,
        quality_flags=np.full_like(flux, MapL3Flags.NONE),
    )


def create_rectangular_intensity_map_data(epoch=None, epoch_delta=None, lon=None, lat=None, energy=None,
                                          energy_delta=None, flux=None,
                                          intensity_stat_uncert=None,
                                          include_ena_intensity_sys_err_minus=False) -> RectangularIntensityMapData:
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    epoch_delta = epoch_delta if epoch_delta is not None else np.ma.array([86400 * 1e9])
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=1)
    intensity_stat_uncert = intensity_stat_uncert if intensity_stat_uncert is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)
    ena_intensity_sys_err_minus = None
    ena_intensity_sys_err_plus = None
    if include_ena_intensity_sys_err_minus:
        ena_intensity_sys_err_minus = np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=2)
        ena_intensity_sys_err_plus = np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=3)

    if isinstance(flux, np.ndarray):
        more_real_flux = flux

    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return RectangularIntensityMapData(
        intensity_map_data=IntensityMapData(
            epoch=epoch,
            epoch_delta=epoch_delta,
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(flux, 0),
            obs_date=np.ma.array(np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))),
            obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_intensity=flux,
            ena_intensity_stat_uncert=intensity_stat_uncert,
            ena_intensity_sys_err=flux * .001,
            bg_intensity=flux * .01,
            bg_intensity_stat_uncert=intensity_stat_uncert * .01,
            bg_intensity_sys_err=flux * .01 * .001,
            ena_intensity_sys_err_minus=ena_intensity_sys_err_minus,
            ena_intensity_sys_err_plus=ena_intensity_sys_err_plus,
            quality_flags=np.full_like(flux, MapL3Flags.NONE),
        ),
        coords=RectangularCoords(
            latitude_delta=np.full_like(lat, 0),
            latitude_label=lat.astype(str),
            longitude_delta=np.full_like(lon, 0),
            longitude_label=lon.astype(str),
        )
    )


def create_rectangular_intensity_map(intensity_map_data: IntensityMapData = None,
                                     coords: RectangularCoords = None) -> RectangularIntensityMapData:
    intensity_map = intensity_map_data if intensity_map_data is not None else create_intensity_map_data()

    if coords is None:
        lon = np.array([1.0])
        lat = np.array([1.0])
        coords = RectangularCoords(
            latitude_delta=np.full_like(lat, 0),
            latitude_label=lat.astype(str),
            longitude_delta=np.full_like(np.array([1.0]), 0),
            longitude_label=lon.astype(str),
        )

    return RectangularIntensityMapData(intensity_map_data=intensity_map, coords=coords)


def construct_intensity_data_with_all_zero_fields(num_pixels: int = 1) -> IntensityMapData:
    return IntensityMapData(
        epoch=np.array([0]),
        epoch_delta=np.array([0]),
        energy=np.array([0]),
        energy_delta_plus=np.array([0]),
        energy_delta_minus=np.array([0]),
        energy_label=np.array([0]),
        latitude=np.array([0] * num_pixels),
        longitude=np.array([0]),
        exposure_factor=np.array([1] * num_pixels),
        obs_date=np.array([datetime(2025, 5, 6)] * num_pixels),
        obs_date_range=np.array([0] * num_pixels),
        solid_angle=np.array([0] * num_pixels),
        ena_intensity=np.array([0] * num_pixels),
        ena_intensity_stat_uncert=np.array([0] * num_pixels),
        ena_intensity_sys_err=np.array([0] * num_pixels),
        bg_intensity=np.array([0] * num_pixels),
        bg_intensity_sys_err=np.array([0] * num_pixels),
        bg_intensity_stat_uncert=np.array([0] * num_pixels),
        survival_probability=np.array([0] * num_pixels),
        quality_flags=np.array([MapL3Flags.NONE] * num_pixels),
    )


def create_rectangular_spectral_index_map_data(epoch=None, epoch_delta=None, lon=None, lat=None, energy=None,
                                               energy_delta=None, spectral_index=None,
                                               spectral_index_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    epoch_delta = epoch_delta if epoch_delta is not None else np.ma.array([86400 * 1e9])
    spectral_index = spectral_index if spectral_index is not None else np.full((len(epoch), len(lon), len(lat)),
                                                                               fill_value=1)
    spectral_index_stat_unc = spectral_index_stat_unc if spectral_index_stat_unc is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)
    ena_spectral_index_scalar_coefficient = np.full_like(spectral_index, 3)
    ena_spectral_index_scalar_coefficient_stat_unc = np.full_like(spectral_index, 4)
    ena_spectral_index_chisq = np.full_like(spectral_index, 5)

    if isinstance(spectral_index, np.ndarray):
        more_real_flux = spectral_index
    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return RectangularSpectralIndexMapData(
        spectral_index_map_data=SpectralIndexMapData(
            epoch=epoch,
            epoch_delta=epoch_delta,
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(spectral_index, 0),
            obs_date=np.ma.array(
                np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))
            ),
            obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_spectral_index=spectral_index,
            ena_spectral_index_stat_uncert=spectral_index_stat_unc,
            ena_spectral_index_scalar_coefficient=ena_spectral_index_scalar_coefficient,
            ena_spectral_index_scalar_coefficient_stat_uncert=ena_spectral_index_scalar_coefficient_stat_unc,
            ena_spectral_index_chisq=ena_spectral_index_chisq,
            quality_flags=np.full(more_real_flux.shape, MapL3Flags.NONE),
        ),
        coords=RectangularCoords(
            latitude_delta=np.full_like(lat, 0),
            latitude_label=lat.astype(str),
            longitude_delta=np.full_like(lon, 0),
            longitude_label=lon.astype(str),
        ),
    )


def create_l1c_pset(
        epoch: datetime = datetime(2025, 4, 15, 12),
        epoch_delta: np.ndarray = np.array([43200_000_000_000]),
        repointing: int = 1,
        energy_steps: np.ndarray = np.array([1]),
        exposures: Optional[np.ndarray] = None,
        pointing_start_met=None,
        pointing_end_met=None,
        hae_longitude=None,
        hae_latitude=None,
) -> InputRectangularPointingSet:
    epoch_j2000 = np.array([spiceypy.datetime2et(epoch)]) * 1e9
    exposures = exposures if exposures is not None else np.full(shape=(1, energy_steps.shape[0], 3600), fill_value=1.)
    ram_latitudes = np.linspace(89.95, -89.95, 1800)
    anti_latitudes = np.linspace(-89.95, 89.95, 1800)
    full_rotation_latitudes = np.concat([ram_latitudes, anti_latitudes])
    return InputRectangularPointingSet(
        epoch,
        epoch_delta,
        epoch_j2000,
        repointing,
        exposures,
        energy_steps,
        pointing_start_met,
        pointing_end_met,
        hae_longitude=hae_longitude
        if hae_longitude is not None
        else ((np.linspace(0, 360, 3600, endpoint=False) + 90.05) % 360)[np.newaxis, :],
        hae_latitude=hae_latitude
        if hae_latitude is not None
        else (full_rotation_latitudes[np.newaxis, :]),
    )


def create_l3e_pset(
    epoch: datetime = datetime(2025, 4, 15, 12),
    repointing: int = 1,
    energy_steps=np.array([0.5, 5.0, 12.0]),
    spin_angle=np.arange(0, 360),
    sp: Optional[np.array] = None,
    flags: Optional[np.array] = None,
) -> GlowsL3eRectangularMapInputData:
    epoch_j2000 = np.array([spiceypy.datetime2et(epoch)]) * 1e9
    sp = sp or np.full(shape=(1, len(energy_steps), len(spin_angle)), fill_value=0.5)
    flags = flags or np.full(1, 0, dtype=np.uint16)
    return GlowsL3eRectangularMapInputData(epoch, epoch_j2000, repointing, energy_steps, spin_angle, sp, flags)
