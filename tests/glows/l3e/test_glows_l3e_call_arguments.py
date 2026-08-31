import unittest

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo


class TestGlowsL3eCallArguments(unittest.TestCase):

    def test_to_argument_list(self):
        glows_l3e_call_arguments = GlowsL3eCallArguments(
            elongation=80.0001,
            decimal_date="2024.0001",
            formatted_date="20240101",
            spacecraft_info=GlowsL3eSpacecraftInfo(
                spacecraft_radius=np.float32(987),
                spin_axis_latitude=np.float32(.01234),
                spin_axis_longitude=np.float32(87.11213),
                spacecraft_latitude=np.float32(1.134141),
                spacecraft_longitude=np.float32(84.12314),
                spacecraft_velocity_x=np.float32(1.23143),
                spacecraft_velocity_y=np.float32(12.34141),
                spacecraft_velocity_z=np.float32(123),
            )
        )

        call_args_list = glows_l3e_call_arguments.to_argument_list()

        self.assertEqual("20240101", call_args_list[0])
        self.assertEqual("2024.0001", call_args_list[1])
        self.assertEqual("987.0", call_args_list[2])
        self.assertEqual("84.1231", call_args_list[3])
        self.assertEqual("1.1341", call_args_list[4])
        self.assertEqual("1.2314", call_args_list[5])
        self.assertEqual("12.3414", call_args_list[6])
        self.assertEqual("123.0", call_args_list[7])
        self.assertEqual("87.1121", call_args_list[8])
        self.assertEqual("0.0123", call_args_list[9])
        self.assertEqual("80.000", call_args_list[10])
