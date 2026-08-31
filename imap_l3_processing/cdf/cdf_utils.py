import re

import numpy as np
from imap_processing.ena_maps.utils import naming
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.models import DataProduct

_CDF_FLOAT_NUMPY_DTYPE = {
    pycdf.const.CDF_REAL4.value: np.float32,
    pycdf.const.CDF_REAL8.value: np.float64,
}


def write_cdf(file_path: str, data: DataProduct, attribute_manager: ImapAttributeManager):
    with CDF(file_path, '') as cdf:
        cdf.col_major(True)
        try:
            global_attrs = attribute_manager.get_global_attributes(data.input_metadata.logical_source)
        except KeyError:
            trimmed_source = re.sub(r"-\d+$", "-", data.input_metadata.logical_source)
            global_attrs = attribute_manager.get_global_attributes(trimmed_source)

        for k, v in global_attrs.items():
            cdf.attrs[k] = v

        map_descriptor = None
        if data.input_metadata.instrument in ("lo", "hi", "ultra", "glows",):
            try:
                map_descriptor = naming.MapDescriptor.from_string(data.input_metadata.descriptor)
            except ValueError:
                # Assume this is not actually a map
                pass

        for data_product in data.to_data_product_variables():
            var_name = data_product.name
            variable_attributes = attribute_manager.get_variable_attributes(var_name)
            data_type = getattr(pycdf.const, variable_attributes["DATA_TYPE"])
            data_array = np.asanyarray(data_product.value)

            if map_descriptor is not None:
                if map_var_catdesc := map_descriptor.build_map_var_catdesc(var_name):
                    variable_attributes["CATDESC"] = map_var_catdesc

            record_varying = variable_attributes["RECORD_VARYING"].lower() == "rv"
            if record_varying:
                dims = data_array.shape[1:]
            else:
                dims = data_array.shape

            if data_array.size == 0:
                data_array = None
            else:
                target_float_dtype = _CDF_FLOAT_NUMPY_DTYPE.get(data_type.value)
                if target_float_dtype is not None and np.issubdtype(data_array.dtype, np.floating):
                    data_array = data_array.astype(target_float_dtype)

                if 'FILLVAL' in variable_attributes:
                    if np.issubdtype(data_array.dtype, np.floating):
                        data_array = np.ma.masked_invalid(data_array)
                    data_array = np.ma.filled(data_array, variable_attributes['FILLVAL'])
                else:
                    assert not np.ma.isMaskedArray(data_array)

            if (
                    data_type
                    not in (pycdf.const.CDF_TIME_TT2000, pycdf.const.CDF_EPOCH,
                            pycdf.const.CDF_EPOCH16)
                    and record_varying
            ):
                compress = pycdf.const.GZIP_COMPRESSION
                compress_param = 7
            else:
                compress = None
                compress_param = None
            cdf.new(var_name, data_array,
                    recVary=record_varying,
                    type=data_type,
                    dims=dims,
                    compress=compress,
                    compress_param=compress_param)
            for k, v in variable_attributes.items():
                if v == '':
                    continue
                if k in ['DATA_TYPE', 'RECORD_VARYING']:
                    continue
                if k == 'FILLVAL' and data_type is not None:
                    cdf[var_name].attrs.new(k, v, data_type)
                else:
                    cdf[var_name].attrs[k] = v


def read_variable_and_mask_fill_values(var: pycdf.Var) -> np.ma.masked_array:
    return np.ma.masked_equal(var[...], var.attrs['FILLVAL'])


def read_numeric_variable(var: pycdf.Var) -> np.ndarray:
    assert np.issubdtype(var.dtype, np.number)
    return np.where(var[...] == var.attrs['FILLVAL'], np.nan, var[...])
