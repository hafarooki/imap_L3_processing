import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import imap_data_access
import numpy as np
from imap_data_access.file_validation import Version
from spacepy.pycdf import CDF

import imap_l3_processing
from imap_l3_processing.glows.l3bc.utils import get_midpoint_of_cr
from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.cdf.cdf_utils import read_numeric_variable

PATH_TO_L3D_TOOLKIT = Path(imap_l3_processing.__file__).parent / 'glows' / 'l3d' / 'science'


def create_glows_l3c_json_file_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        cr_number = cdf['cr'][...][0]

        json_dict = {
            'header': {
                'filename': cdf_file_path.name
            },
            'solar_wind_profile': {
                'proton_density': read_numeric_variable(cdf["proton_density_profile"])[0].tolist(),
                'plasma_speed': read_numeric_variable(cdf["plasma_speed_profile"])[0].tolist(),
            },
            'solar_wind_ecliptic': {
                'proton_density': float(read_numeric_variable(cdf["proton_density_ecliptic"])[0]),
                'alpha_abundance': float(read_numeric_variable(cdf["alpha_abundance_ecliptic"])[0]),
            },
            'CR': float(cr_number)
        }
        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3c_cr_{cr_number}_{version}.json'

        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3c', exist_ok=True)

        with open(PATH_TO_L3D_TOOLKIT / 'data_l3c' / json_file_name, 'w') as fp:
            json.dump(json_dict, fp)


def create_glows_l3b_json_file_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        cr_number = int(cdf['cr'][...][0])
        json_dict = {
            'header': {
                'filename': cdf_file_path.name,
            },
            'date': cdf['mean_time'][0].isoformat(),
            'CR': cr_number,
            'uv_anisotropy_factor': read_numeric_variable(cdf['uv_anisotropy_factor'])[0].tolist(),
            'ion_rate_profile': {
                'lat_grid': read_numeric_variable(cdf['lat_grid']).tolist(),
                'ph_rate': read_numeric_variable(cdf['ph_rate'])[0].tolist()
            },
            'uv_anisotropy_flag': cdf['uv_anisotropy_flag'][0],
            'glows_flags': int(cdf['glows_flags'][0]),
        }

        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3b_cr_{cr_number}_{version}.json'

        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3b', exist_ok=True)
        with open(PATH_TO_L3D_TOOLKIT / 'data_l3b' / json_file_name, 'w') as fp:
            json.dump(json_dict, fp)


def convert_json_to_l3d_data_product(json_file_path: Path, input_metadata: InputMetadata,
                                     parent_file_names: list[str]) -> GlowsL3DSolarParamsHistory:
    with open(json_file_path, 'r') as json_file:
        l3d_json_dict = json.load(json_file)

    new_start = datetime.fromisoformat(l3d_json_dict['time_grid'][0])
    input_metadata.start_date = new_start.date()

    return GlowsL3DSolarParamsHistory(
        input_metadata=input_metadata,
        parent_file_names=parent_file_names,
        latitude=l3d_json_dict['lat_grid'],
        cr=l3d_json_dict['cr_grid'],
        epoch=np.array([datetime.fromisoformat(time) for time in l3d_json_dict['time_grid']]),
        plasma_speed=l3d_json_dict['solar_params']['speed'],
        proton_density=l3d_json_dict['solar_params']['p-dens'],
        ultraviolet_anisotropy=l3d_json_dict['solar_params']['uv-anis'],
        phion=l3d_json_dict['solar_params']['phion'],
        lyman_alpha=l3d_json_dict['solar_params']['lya'],
        electron_density=l3d_json_dict['solar_params']['e-dens'],
        plasma_speed_flag=l3d_json_dict['flags']['speed'],
        uv_anisotropy_flag=l3d_json_dict['flags']['uv-anis'],
        proton_density_flag=l3d_json_dict['flags']['p-dens'],
        glows_flags=np.array(l3d_json_dict['glows_flags'], dtype=np.uint16),
    )


def get_parent_file_names_from_l3d_json(l3d_folder: Path) -> list[str]:
    parent_file_names = set()
    l3d_file_paths = sorted(os.listdir(l3d_folder))

    if len(l3d_file_paths) == 0:
        return []

    with open(l3d_folder / l3d_file_paths[0]) as l3d:
        l3d_data = json.load(l3d)
        ancillary_files = [Path(file).name for file in l3d_data['header']['ancillary_data_files']]
        external_dependencies = Path(l3d_data['header']['external_dependeciens']).name
        parent_file_names.update(ancillary_files)
        parent_file_names.add(external_dependencies)

    for file_path in l3d_file_paths:
        with open(l3d_folder / file_path, 'r') as l3d:
            l3d_data = json.load(l3d)
            l3b_parents = [Path(file).name for file in l3d_data['header']['l3b_input_filename']]
            l3c_parents = [Path(file).name for file in l3d_data['header']['l3c_input_filename']]
            parent_file_names.update(l3b_parents)
            parent_file_names.update(l3c_parents)

    return list(parent_file_names)

def get_most_recently_uploaded_ancillary(query_result: list[dict]) -> dict:
    query_result = max(query_result, key=lambda x: x['ingestion_date'], default=None)
    return query_result

def rename_l3d_text_outputs(paths: list[Path], version: str) -> list[Path]:
    out_paths = []
    for path in paths:
        filename_without_extension, extension = path.name.split('.')
        original_name_components = filename_without_extension.split('_')
        descriptor = original_name_components[3]
        start_date_with_cr = original_name_components[4]
        start_date, cr = start_date_with_cr.split('-cr')
        midpoint_of_cr = get_midpoint_of_cr(int(cr))
        new_path = path.parent / f"imap_glows_{descriptor}_{start_date}_{midpoint_of_cr.strftime('%Y%m%d')}_{version}.{extension}"
        os.rename(path, new_path)
        out_paths.append(new_path)
    return out_paths


def query_for_most_recent_l3d(descriptor: str) -> Optional[dict]:
    query_result = imap_data_access.query(instrument="glows", data_level="l3d", descriptor=descriptor)
    sorted_query_result = sorted(
        query_result,
        key=lambda qr: (
            qr["cr"],
            (qr["minor_version"] if "minor_version" in qr else Version.from_version(qr["version"]).minor)
        ),
        reverse=True)
    return next(iter(sorted_query_result), None)
