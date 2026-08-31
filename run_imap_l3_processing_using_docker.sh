source scripts/update_version.sh

docker run --rm \
--mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" \
--mount type=bind,src="$(pwd)/data",dst="/data" \
-e IMAP_API_KEY=$IMAP_API_KEY \
$(docker build -q .) $@

git restore imap_l3_processing/version.py