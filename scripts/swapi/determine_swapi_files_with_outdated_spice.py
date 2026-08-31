import imap_data_access
from spacepy.pycdf import CDF

data_level = "l3a"
descriptors = ["pui-he"]

for descriptor in descriptors:
    query_results = imap_data_access.query(instrument="swapi", data_level=data_level, descriptor=descriptor, version="latest")

    dates_that_used_predict = []
    for result in sorted(query_results, key=lambda r: r["start_date"]):
        file_path = imap_data_access.download(result["file_path"])

        with CDF(str(file_path)) as cdf:
            used_predict = any(["pred" in entry for entry in cdf.attrs["Parents"]])

            if used_predict:
                dates_that_used_predict.append(result["start_date"])

    print(descriptor)
    for date in dates_that_used_predict:
        print(f"\t{date}")
