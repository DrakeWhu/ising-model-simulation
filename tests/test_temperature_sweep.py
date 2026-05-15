import unittest

import numpy as np

import temperature_sweep


class TestTemperatureSweep(unittest.TestCase):
    def test_build_temperature_grid_from_explicit_temperatures(self) -> None:
        result = temperature_sweep.build_temperature_grid(
            explicit_temperatures=[1.5, 2.0, 2.5],
            t_min=0.0,
            t_max=0.0,
            num_temperatures=0,
        )

        np.testing.assert_allclose(result, np.array([1.5, 2.0, 2.5]))

    def test_build_temperature_grid_from_range(self) -> None:
        result = temperature_sweep.build_temperature_grid(
            explicit_temperatures=None,
            t_min=1.0,
            t_max=3.0,
            num_temperatures=3,
        )

        np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))

    def test_build_temperature_grid_rejects_non_positive_temperatures(self) -> None:
        with self.assertRaises(ValueError):
            temperature_sweep.build_temperature_grid(
                explicit_temperatures=[1.0, 0.0],
                t_min=0.0,
                t_max=0.0,
                num_temperatures=0,
            )

    def test_build_temperature_grid_rejects_too_few_range_points(self) -> None:
        with self.assertRaises(ValueError):
            temperature_sweep.build_temperature_grid(
                explicit_temperatures=None,
                t_min=1.0,
                t_max=2.0,
                num_temperatures=1,
            )


if __name__ == "__main__":
    unittest.main()
