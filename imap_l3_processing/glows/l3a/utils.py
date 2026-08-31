import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from imap_data_access import AncillaryFilePath
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3a.models import GlowsL2Data, GlowsL2LightCurve, GlowsLatLon, GlowsL3LightCurve, XYZ, \
    GlowsL2Header
from imap_l3_processing.models import InputMetadata, Instrument

MAX_SPIN_ANGLE_BINS = 90


def _pad_lightcurve(values) -> np.ndarray:
    """Pad a lightcurve variable to the fixed L3a spin-angle dimension."""
    values = np.asarray(values)
    if len(values) > MAX_SPIN_ANGLE_BINS:
        raise ValueError(
                f"GLOWS L3a lightcurve bins ({len(values)}) exceed "
                f"the allowed maximum ({MAX_SPIN_ANGLE_BINS})."
        )

    if np.issubdtype(values.dtype, np.integer):
        padded_values = np.ma.masked_all(MAX_SPIN_ANGLE_BINS, dtype=values.dtype)
        padded_values[:len(values)] = values
        return padded_values.reshape(1, -1)

    return np.pad(
        values,
        (0, MAX_SPIN_ANGLE_BINS - len(values)),
        constant_values=np.nan,
    ).reshape(1, -1)


def read_l2_glows_data(cdf: CDF) -> GlowsL2Data:
    assert 1 == cdf['photon_flux'].shape[0], "Level 2 file should have only one histogram"

    light_curve = GlowsL2LightCurve(photon_flux=cdf['photon_flux'][0],
                                    spin_angle=cdf['spin_angle'][0],
                                    histogram_flag_array=cdf['histogram_flag_array'][0],
                                    exposure_times=cdf['exposure_times'][0],
                                    flux_uncertainties=cdf['flux_uncertainties'][0],
                                    raw_histogram=cdf['raw_histograms'][0],
                                    ecliptic_lon=cdf['ecliptic_lon'][0],
                                    ecliptic_lat=cdf['ecliptic_lat'][0])
    spin_axis_average = GlowsLatLon(lon=cdf['spin_axis_orientation_average'][0, 0],
                                    lat=cdf['spin_axis_orientation_average'][0, 1])
    spin_axis_std_dev = GlowsLatLon(lon=cdf['spin_axis_orientation_std_dev'][0, 0],
                                    lat=cdf['spin_axis_orientation_std_dev'][0, 1])
    spacecraft_location_average = XYZ(x=cdf['spacecraft_location_average'][0, 0],
                                      y=cdf['spacecraft_location_average'][0, 1],
                                      z=cdf['spacecraft_location_average'][0, 2], )
    spacecraft_location_std_dev = XYZ(x=cdf['spacecraft_location_std_dev'][0, 0],
                                      y=cdf['spacecraft_location_std_dev'][0, 1],
                                      z=cdf['spacecraft_location_std_dev'][0, 2], )
    spacecraft_velocity_average = XYZ(x=cdf['spacecraft_velocity_average'][0, 0],
                                      y=cdf['spacecraft_velocity_average'][0, 1],
                                      z=cdf['spacecraft_velocity_average'][0, 2], )
    spacecraft_velocity_std_dev = XYZ(x=cdf['spacecraft_velocity_std_dev'][0, 0],
                                      y=cdf['spacecraft_velocity_std_dev'][0, 1],
                                      z=cdf['spacecraft_velocity_std_dev'][0, 2], )

    ancillary_files = []
    for file in list(cdf.attrs["Parents"][...]):
        try:
            ancillary_file = AncillaryFilePath(file)
            if ancillary_file.instrument == Instrument.GLOWS.value:
                ancillary_files.append(file)
        except AncillaryFilePath.InvalidImapFileError:
            pass

    return GlowsL2Data(identifier=cdf['identifier'][0],
                       start_time=cdf['start_time'][0],
                       end_time=cdf['end_time'][0],
                       daily_lightcurve=light_curve,
                       number_of_bins=cdf['number_of_bins'][0],
                       spin_axis_orientation_average=spin_axis_average,
                       spin_axis_orientation_std_dev=spin_axis_std_dev,
                       filter_temperature_average=cdf['filter_temperature_average'][0],
                       filter_temperature_std_dev=cdf['filter_temperature_std_dev'][0],
                       hv_voltage_average=cdf['hv_voltage_average'][0],
                       hv_voltage_std_dev=cdf['hv_voltage_std_dev'][0],
                       spin_period_average=cdf['spin_period_average'][0],
                       spin_period_std_dev=cdf['spin_period_std_dev'][0],
                       spin_period_ground_average=cdf['spin_period_ground_average'][0],
                       spin_period_ground_std_dev=cdf['spin_period_ground_std_dev'][0],
                       pulse_length_average=cdf['pulse_length_average'][0],
                       pulse_length_std_dev=cdf['pulse_length_std_dev'][0],
                       position_angle_offset_average=cdf['position_angle_offset_average'][0],
                       position_angle_offset_std_dev=cdf['position_angle_offset_std_dev'][0],
                       spacecraft_location_average=spacecraft_location_average,
                       spacecraft_location_std_dev=spacecraft_location_std_dev,
                       spacecraft_velocity_average=spacecraft_velocity_average,
                       spacecraft_velocity_std_dev=spacecraft_velocity_std_dev,
                       header=GlowsL2Header(
                           flight_software_version=str(cdf.attrs["flight_software_version"][0]),
                           pkts_file_name=cdf.attrs["pkts_file_name"][0],
                           ancillary_data_files=ancillary_files,
                       ),
                       l2_file_name=Path(cdf.pathname.decode('utf-8')).name
                       )


def _read_xyz(cdf: CDF, variable_name: str) -> XYZ:
    return {k: cdf[f'{variable_name}_{k}'][0] for k in "xyz"}


def create_glows_l3a_from_dictionary(data: dict, input_metadata: InputMetadata) -> GlowsL3LightCurve:
    start_time = datetime.fromisoformat(data["start_time"])
    end_time = datetime.fromisoformat(data["end_time"])
    total_time = end_time - start_time
    epoch = start_time + total_time / 2
    return GlowsL3LightCurve(
        global_metadata_attrs=data["header"],
        input_metadata=input_metadata,
        identifier=input_metadata.repointing,
        epoch=np.array([epoch]),
        epoch_delta=np.array([total_time.total_seconds() / 2 * 1e9]),
        start_time=data["start_time"],
        end_time=data["end_time"],
        photon_flux=_pad_lightcurve(data["daily_lightcurve"]["photon_flux"]),
        photon_flux_uncertainty=_pad_lightcurve(data["daily_lightcurve"]["flux_uncertainties"]),
        raw_histogram=_pad_lightcurve(data["daily_lightcurve"]["raw_histogram"]),
        exposure_times=_pad_lightcurve(data["daily_lightcurve"]["exposure_times"]),
        spin_angle=_pad_lightcurve(data["daily_lightcurve"]["spin_angle"]),
        spin_angle_delta=_pad_lightcurve(data["daily_lightcurve"]["spin_angle_delta"]),
        latitude=_pad_lightcurve(data["daily_lightcurve"]["ecliptic_lat"]),
        longitude=_pad_lightcurve(data["daily_lightcurve"]["ecliptic_lon"]),
        extra_heliospheric_background=_pad_lightcurve(data["daily_lightcurve"]["extra_heliospheric_bckgrd"]),
        time_dependent_background=_pad_lightcurve(data["daily_lightcurve"]["time_dependent_bckgrd"]),
        filter_temperature_average=np.array([data["filter_temperature_average"]]),
        filter_temperature_std_dev=np.array([data["filter_temperature_std_dev"]]),
        hv_voltage_average=np.array([data["hv_voltage_average"]]),
        hv_voltage_std_dev=np.array([data["hv_voltage_std_dev"]]),
        spin_period_average=np.array([data["spin_period_average"]]),
        spin_period_std_dev=np.array([data["spin_period_std_dev"]]),
        spin_period_ground_average=np.array([data["spin_period_ground_average"]]),
        spin_period_ground_std_dev=np.array([data["spin_period_ground_std_dev"]]),
        pulse_length_average=np.array([data["pulse_length_average"]]),
        pulse_length_std_dev=np.array([data["pulse_length_std_dev"]]),
        position_angle_offset_average=np.array([data["position_angle_offset_average"]]),
        position_angle_offset_std_dev=np.array([data["position_angle_offset_std_dev"]]),
        spin_axis_orientation_average=get_lon_lat(data["spin_axis_orientation_average"]),
        spin_axis_orientation_std_dev=get_lon_lat(data["spin_axis_orientation_std_dev"]),
        spacecraft_location_average=get_xyz(data["spacecraft_location_average"]),
        spacecraft_location_std_dev=get_xyz(data["spacecraft_location_std_dev"]),
        spacecraft_velocity_average=get_xyz(data["spacecraft_velocity_average"]),
        spacecraft_velocity_std_dev=get_xyz(data["spacecraft_velocity_std_dev"]),
        number_of_bins=np.array([data['daily_lightcurve']['number_of_bins']]),
    )


def create_glows_l3a_dictionary_from_cdf(cdf_file_path: Path) -> dict:
    cdf = CDF(str(cdf_file_path))
    time_delta = timedelta(seconds=cdf['epoch_delta'][0] / 1e9)
    start_time = cdf['epoch'][0] - time_delta
    end_time = cdf['epoch'][0] + time_delta
    valid_bin_count = cdf['number_of_bins'][0]
    return {
        'filename': f'{os.path.basename(cdf_file_path)}',
        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
        'daily_lightcurve': {
            'ecliptic_lat': cdf['ecliptic_lat'][0][:valid_bin_count],
            'ecliptic_lon': cdf['ecliptic_lon'][0][:valid_bin_count],
            'exposure_times': cdf['exposure_times'][0][:valid_bin_count],
            'photon_flux': cdf['photon_flux'][0][:valid_bin_count],
            'flux_uncertainties': cdf['photon_flux_uncertainty'][0][:valid_bin_count],
            'extra_heliospheric_bckgrd': cdf['extra_heliospheric_bckgrd'][0][:valid_bin_count],
            'time_dependent_bckgrd': cdf['time_dependent_bckgrd'][0][:valid_bin_count],
            'spin_angle': cdf['spin_angle'][0][:valid_bin_count],
            'raw_histogram': cdf['raw_histogram'][0][:valid_bin_count],
            'number_of_bins': valid_bin_count
        },
        'filter_temperature_average': cdf['filter_temperature_average'][0],
        'filter_temperature_std_dev': cdf['filter_temperature_std_dev'][0],
        'hv_voltage_average': cdf['hv_voltage_average'][0],
        'hv_voltage_std_dev': cdf['hv_voltage_std_dev'][0],
        'spin_period_average': cdf['spin_period_average'][0],
        'spin_period_std_dev': cdf['spin_period_std_dev'][0],
        'spin_period_ground_average': cdf['spin_period_ground_average'][0],
        'spin_period_ground_std_dev': cdf['spin_period_ground_std_dev'][0],
        'pulse_length_average': cdf['pulse_length_average'][0],
        'pulse_length_std_dev': cdf['pulse_length_std_dev'][0],
        'position_angle_offset_average': cdf['position_angle_offset_average'][0],
        'position_angle_offset_std_dev': cdf['position_angle_offset_std_dev'][0],
        'spin_axis_orientation_average': {
            'lon': cdf['spin_axis_orientation_average'][0][0],
            'lat': cdf['spin_axis_orientation_average'][0][1],
        },
        'spin_axis_orientation_std_dev': {
            'lon': cdf['spin_axis_orientation_std_dev'][0][0],
            'lat': cdf['spin_axis_orientation_std_dev'][0][1],
        },
        'spacecraft_location_average': {
            'x': cdf['spacecraft_location_average'][0][0],
            'y': cdf['spacecraft_location_average'][0][1],
            'z': cdf['spacecraft_location_average'][0][2],
        },
        'spacecraft_location_std_dev': {
            'x': cdf['spacecraft_location_std_dev'][0][0],
            'y': cdf['spacecraft_location_std_dev'][0][1],
            'z': cdf['spacecraft_location_std_dev'][0][2],
        },
        'spacecraft_velocity_average': {
            'x': cdf['spacecraft_velocity_average'][0][0],
            'y': cdf['spacecraft_velocity_average'][0][1],
            'z': cdf['spacecraft_velocity_average'][0][2],
        },
        'spacecraft_velocity_std_dev': {
            'x': cdf['spacecraft_velocity_std_dev'][0][0],
            'y': cdf['spacecraft_velocity_std_dev'][0][1],
            'z': cdf['spacecraft_velocity_std_dev'][0][2],
        },
    }


def get_lon_lat(data: dict) -> np.ndarray:
    return np.array([[data["lon"], data["lat"]]])


def get_xyz(data: dict) -> np.ndarray:
    return np.array([[data["x"], data["y"], data["z"]]])
