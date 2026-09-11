import unittest

import numpy as np

from imap_l3_processing.swapi.l3a.science.pickup_ion.uniform_speed_grid import (
    UniformSpeedGrid,
)


class UniformSpeedGridTest(unittest.TestCase):
    def test_uniformly_spans_from_just_above_zero_up_to_the_max_speed(self):
        max_speed = 600.0

        grid = UniformSpeedGrid(max_speed)

        self.assertEqual(grid.size, grid.centers.size)
        self.assertGreater(float(grid.centers[0]), 0.0)
        self.assertLess(float(grid.centers[0]), max_speed * 1e-2)
        self.assertAlmostEqual(max_speed, float(grid.centers[-1]))
        np.testing.assert_allclose(np.diff(grid.centers), grid.spacing, rtol=1e-12)

    def test_weights_each_cell_by_its_shell_volume(self):
        grid = UniformSpeedGrid(600.0)

        np.testing.assert_allclose(
            grid.spherical_shell_integration_weights,
            grid.centers**2 * grid.spacing,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
