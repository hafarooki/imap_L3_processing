source scripts/update_version.sh

docker run --rm --mount type=bind,src="$(pwd)/spice_kernels",dst="/mnt/spice" \
--mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" \
$(docker build -q .) --instrument swapi --data-level l3a \
--start-date 20250606 --descriptor proton-sw --version v000 --dependency \
'[{"type":"science","files":["imap_swapi_l2_sci_20250606_v007.cdf"]},{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v000.dat"]},{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"]},{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]}]'


git restore imap_l3_processing/version.py