import shutil
from pathlib import Path

import imap_data_access
from imap_data_access.file_validation import generate_imap_file_path

from tests.test_helpers import get_test_data_path

files_to_upload = [
    "imap_swapi_alpha-density-temperature-lut_20240920_v000.dat",
    "imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat",
    "imap_swapi_efficiency-lut_20241020_v000.dat",
    "imap_swapi_energy-gf-pui-lut_20100101_v001.csv",
    "imap_swapi_energy-gf-sw-lut_20100101_v001.csv",
    "imap_swapi_instrument-response-lut_20241023_v000.zip",
]
upload_to_local_dir = False
for file in files_to_upload:
    try:
        if upload_to_local_dir:
            write_path = generate_imap_file_path(file).construct_path()
            shutil.copy(get_test_data_path('swapi') / file, write_path)
        else:
            imap_data_access.upload(Path(__file__).parent.parent / "swapi" / "test_data" / file)
            print("Successfully uploaded", file)
    except Exception as e:
        print(f'File "{file}" failed to upload with exception {e}')
