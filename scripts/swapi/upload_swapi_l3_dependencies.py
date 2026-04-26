import shutil
from pathlib import Path

import imap_data_access
from imap_data_access.file_validation import generate_imap_file_path

from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

_swapi = get_test_data_path('swapi')
files_to_upload = [
    _swapi / "imap_swapi_alpha-density-temperature-lut_20240920_v000.dat",
    _swapi / "imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat",
    _swapi / "imap_swapi_efficiency-lut_20241020_v000.dat",
    _swapi / "imap_swapi_energy-gf-pui-lut_20100101_v001.csv",
    _swapi / "imap_swapi_energy-gf-sw-lut_20100101_v001.csv",
    _swapi / "imap_swapi_instrument-response-lut_20241023_v000.zip",
    get_test_instrument_team_data_path("swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"),
    get_test_instrument_team_data_path("swapi/imap_swapi_central-effective-area_20260425_v001.csv"),
    get_test_instrument_team_data_path("swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"),
]
upload_to_local_dir = False
for file in files_to_upload:
    try:
        if upload_to_local_dir:
            write_path = generate_imap_file_path(file.name).construct_path()
            shutil.copy(file, write_path)
        else:
            imap_data_access.upload(file)
            print("Successfully uploaded", file.name)
    except Exception as e:
        print(f'File "{file.name}" failed to upload with exception {e}')
