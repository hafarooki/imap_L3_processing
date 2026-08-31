import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import yaml

from imap_l3_processing import cdf


class TestCdfUtils(TestCase):

    def setUp(self):
        yaml_path = Path(cdf.__file__).parent / "config"

        variable_attrs_filenames = [str(filename) for filename in os.listdir(yaml_path) if
                                    "variable_attrs" in str(filename)]
        self.test_cases_variable = []
        for filename in variable_attrs_filenames:
            with open(yaml_path / filename) as file:
                yaml_data = yaml.safe_load(file)
                for variable_key, variable in yaml_data.items():
                    if 'NAME' in variable:
                        self.test_cases_variable.append((filename, yaml_data, variable_key, variable))

        self.test_cases_file = []
        for filename in variable_attrs_filenames:
            with open(yaml_path / filename) as file:
                yaml_data = yaml.safe_load(file)
                self.test_cases_file.append((filename, yaml_data))

    def test_variable_name_equals_key(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(msg=f"{filename}:{variable_key}"):
                self.assertEqual(variable_key, variable['NAME'],
                                 f"{filename}:{variable_key} variable key and NAME do not match")

    def test_epoch(self):
        for filename, yaml_data in self.test_cases_file:
            with self.subTest(filename):
                self.assertIn('epoch', yaml_data.keys(), "no entry in file for 'epoch'")
                self.assertEqual('epoch', yaml_data['epoch']['NAME'], "epoch must be lowercase")

                self._epoch_meets_schema(yaml_data)
                if "epoch_delta" in yaml_data.keys():
                    self._epoch_delta_meets_schema(yaml_data)

    def test_type_of_min_max_attributes(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            for attr in ("SCALEMIN", "SCALEMAX", "VALIDMIN", "VALIDMAX"):
                if value := variable.get(attr):
                    with self.subTest(msg=f"{filename}:{variable_key}"):
                        if variable["DATA_TYPE"] == "CDF_TIME_TT2000":
                            self.assertEqual(datetime, type(value))
                        else:
                            self.assertIn(type(value), (int, float))

    def test_delta_vars_have_same_units(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(msg=f"{filename}:{variable_key}"):
                if 'DELTA_PLUS_VAR' in variable:
                    self.assertEqual(variable['UNITS'], yaml_data[variable['DELTA_PLUS_VAR']]['UNITS'])
                if 'DELTA_MINUS_VAR' in variable:
                    self.assertEqual(variable['UNITS'], yaml_data[variable['DELTA_MINUS_VAR']]['UNITS'])

    def test_lablaxis_is_defined_for_required_types(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if 'NAME' in variable and not variable['VAR_TYPE'] == 'metadata' and not variable.get(
                        "DISPLAY_TYPE") == 'no_plot':
                    self.assertIn('LABLAXIS', variable.keys(), f'LABLAXIS should exist for {variable_key}')

    def test_var_type_exists(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                self.assertIn('VAR_TYPE', variable.keys(), f'VAR_TYPE should exist for {variable_key}')

    def test_each_data_variable_has_a_display_type(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if variable['VAR_TYPE'] == 'data':
                    self.assertIn('DISPLAY_TYPE', variable.keys(),
                                  f'DISPLAY_TYPE should exist for data variables {variable_key}')

    def test_units_exists_for_data_and_support_data_variables(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if variable['VAR_TYPE'] in ("data", "support_data"):
                    self.assertIn("UNITS", variable.keys())

    def test_metadata_never_has_empty_string_as_value(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                for attr in variable.values():
                    self.assertNotEqual("", attr)

    def test_each_variable_has_data_type_and_fill_value(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                self.assertIn("DATA_TYPE", variable.keys(), f'{variable_key} requires "DATA_TYPE"')
                self.assertIn("FILLVAL", variable.keys(), f'{variable_key} requires "FILLVAL"')

    def test_all_angle_variables_in_degrees(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            if "angle" in variable_key and "UNITS" in variable and "deg" in variable["UNITS"].lower():
                with self.subTest(f"{filename}:{variable_key}"):
                    self.assertIn("UNITS", variable.keys(), f'{variable_key}: Expected units for angle variable ')
                    self.assertEqual("degrees", variable["UNITS"])

    def test_variables_that_depend_on_epoch_are_record_varying(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            if variable.get("DEPEND_0") == "epoch":
                with self.subTest(f"{filename}:{variable_key}"):
                    self.assertIn("RECORD_VARYING", variable)
                    self.assertEqual("RV", variable["RECORD_VARYING"])

    def test_attrs_that_point_to_another_var_actually_point_to_something(self):
        def _is_pointer_attr(attr: str) -> bool:
            return attr.startswith("DEPEND_") or \
                attr.startswith("LABL_PTR_") or \
                attr in ["DELTA_PLUS_VAR", "DELTA_MINUS_VAR"]

        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                pointed_to_vars = [(attr, variable[attr]) for attr in variable if _is_pointer_attr(attr)]
                for attr, pointed_to_var in pointed_to_vars:
                    self.assertIn(pointed_to_var, yaml_data.keys(),
                                  f"\n{pointed_to_var} is the {attr} of {variable_key} but does not exist!")

    def test_variable_fill_value_matches_data_type(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                fill_val = variable.get('FILLVAL')
                match variable['DATA_TYPE']:
                    case "CDF_DOUBLE":
                        self.assertEqual(fill_val, -1e31)
                    case "CDF_FLOAT":
                        self.assertEqual(fill_val, -1e31)
                    case "CDF_REAL4":
                        self.assertEqual(fill_val, -1e31)
                    case "CDF_UINT1":
                        self.assertEqual(fill_val, 255)
                    case "CDF_UINT2":
                        self.assertEqual(fill_val, 65535)
                    case "CDF_UINT4":
                        self.assertEqual(fill_val, 4294967295)
                    case "CDF_INT1":
                        self.assertEqual(fill_val, -128)
                    case "CDF_INT2":
                        self.assertEqual(fill_val, -32768)
                    case "CDF_INT4":
                        self.assertEqual(fill_val, -2147483648)
                    case "CDF_INT8":
                        self.assertEqual(fill_val, -9223372036854775808)
                    case "CDF_CHAR":
                        self.assertEqual(fill_val, ' ')
                    case "CDF_TIME_TT2000":
                        self.assertEqual(fill_val, datetime.fromisoformat('9999-12-31T23:59:59.999999'))
                    case _:
                        self.assertFalse(True, f"Found unknown DATA_TYPE: {variable['DATA_TYPE']}")

    def test_all_record_varying_data_depends_on_epoch(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if "RECORD_VARYING" in variable and variable["RECORD_VARYING"].lower() == "rv" and variable_key != "epoch":
                    self.assertEqual("epoch", variable.get("DEPEND_0"))

    def _epoch_meets_schema(self, yaml_data: dict):
        epoch_delta_types = ["epoch_delta", "epoch_delta_plus", "epoch_delta_minus"]
        if any([epoch_delta_type in yaml_data.keys() for epoch_delta_type in epoch_delta_types]):
            self.assertEqual(20, len(yaml_data['epoch'].keys()))
        else:
            self.assertEqual(18, len(yaml_data['epoch'].keys()))
        self.assertEqual('CDF_TIME_TT2000', yaml_data['epoch']['DATA_TYPE'],
                         "epoch type must be CDF_TIME_TT2000")
        self.assertEqual('support_data', yaml_data['epoch']['VAR_TYPE'],
                         "epoch VAR_TYPE should be support_data")
        self.assertEqual('RV', yaml_data['epoch']['RECORD_VARYING'],
                         "epoch RECORD_VARYING should be RV")
        self.assertIn(yaml_data['epoch']['FIELDNAM'], ['Epoch', 'J2000 Nanoseconds'],
                      "epoch FIELDNAM should be Epoch")
        self.assertEqual(' ', yaml_data['epoch']['FORMAT'],
                         "epoch FORMAT should be ' '")
        self.assertEqual('Epoch', yaml_data['epoch']['LABLAXIS'],
                         "epoch LABLAXIS should be Epoch")
        self.assertEqual('ns', yaml_data['epoch']['UNITS'],
                         "epoch UNITS should be ns")
        self.assertEqual(datetime.fromisoformat('2099-12-31T00:00:00.000000'),
                         yaml_data['epoch']['VALIDMAX'],
                         "epoch VALIDMAX should be '2099-12-31T00:00:00.000000'")
        self.assertEqual(datetime.fromisoformat('9999-12-31T23:59:59.999999'),
                         yaml_data['epoch']['FILLVAL'],
                         "epoch FILLVAL should be '9999-12-31T23:59:59.999999'")
        self.assertEqual("linear", yaml_data['epoch']['SCALETYP'],
                         "epoch SCALETYP should be linear")
        self.assertEqual("J2000", yaml_data['epoch']['TIME_BASE'],
                         "epoch TIME_BASE should be J2000")
        self.assertEqual("TT", yaml_data['epoch']['TIME_SCALE'],
                         "epoch TIME_SCALE should be TT")
        self.assertEqual("INCREASE", yaml_data['epoch']['MONOTON'],
                         "epoch MONOTON should be INCREASE")
        self.assertEqual("Rotating Earth Geoid", yaml_data['epoch']['REFERENCE_POSITION'],
                         "epoch REFERENCE_POSITION should be Rotating Earth Geoid")
        self.assertEqual("1e-9>seconds", yaml_data['epoch']['SI_CONVERSION'],
                         "epoch SI_CONVERSION should be 1e-9>seconds")

    def _epoch_delta_meets_schema(self, yaml_data: dict):
        self.assertEqual('CDF_INT8', yaml_data['epoch_delta']['DATA_TYPE'],
                         "epoch_delta type must be CDF_INT8")
        self.assertEqual('epoch', yaml_data['epoch_delta']['DEPEND_0'],
                         "DEPEND_0 type must be epoch")
        self.assertEqual('support_data', yaml_data['epoch_delta']['VAR_TYPE'],
                         "epoch_delta VAR_TYPE should be 'support_data'")
        self.assertEqual('RV', yaml_data['epoch_delta']['RECORD_VARYING'],
                         "epoch_delta RECORD_VARYING should be RV")
        self.assertEqual('Epoch Delta', yaml_data['epoch_delta']['FIELDNAM'],
                         "epoch_delta FIELDNAM should be 'Epoch Delta'")
        self.assertIn(yaml_data['epoch_delta']['FORMAT'], ['I19', 'I15'],
                      "epoch_delta FORMAT should be 'I19'")
        self.assertEqual('Epoch delta', yaml_data['epoch_delta']['LABLAXIS'],
                         "epoch_delta LABLAXIS should be 'Epoch delta'")
        self.assertEqual('ns', yaml_data['epoch_delta']['UNITS'],
                         "epoch_delta UNITS should be ns")
        self.assertEqual(0,
                         yaml_data['epoch_delta']['VALIDMIN'],
                         "epoch_delta VALIDMIN should be '0'")
        self.assertEqual(-9223372036854775808,
                         yaml_data['epoch_delta']['FILLVAL'],
                         "epoch_delta FILLVAL should be '-9223372036854775808'")
        self.assertEqual("linear", yaml_data['epoch_delta']['SCALETYP'],
                         "epoch_delta SCALETYP should be linear")
