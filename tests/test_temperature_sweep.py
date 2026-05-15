import unittest

import numpy as np

import exact_square_lattice
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

    def test_build_exact_reference_curves_returns_none_for_nonzero_field(self) -> None:
        temperatures = np.array([1.5, 2.0, 2.5])

        result = temperature_sweep.build_exact_reference_curves(
            temperatures=temperatures,
            coupling=1.0,
            field=0.1,
        )

        self.assertIsNone(result)

    def test_build_exact_reference_curves_for_zero_field(self) -> None:
        temperatures = np.array([1.5, 2.0, 2.5])

        result = temperature_sweep.build_exact_reference_curves(
            temperatures=temperatures,
            coupling=1.0,
            field=0.0,
            n_points=50,
            n_critical_points=100,
        )

        self.assertIsNotNone(result)
        assert result is not None

        self.assertIn("energy_temperature", result)
        self.assertIn("energy_density", result)
        self.assertIn("magnetization_temperature", result)
        self.assertIn("spontaneous_magnetization", result)

        self.assertEqual(
            result["energy_temperature"].shape,
            result["energy_density"].shape,
        )
        self.assertEqual(
            result["magnetization_temperature"].shape,
            result["spontaneous_magnetization"].shape,
        )

        self.assertAlmostEqual(result["energy_temperature"][0], 1.5)
        self.assertAlmostEqual(result["energy_temperature"][-1], 2.5)

        tc = exact_square_lattice.critical_temperature()
        self.assertLessEqual(result["magnetization_temperature"][-1], tc)

        self.assertTrue(np.all(result["energy_density"] < 0.0))
        self.assertTrue(np.all(result["spontaneous_magnetization"] >= 0.0))
        self.assertTrue(np.all(result["spontaneous_magnetization"] <= 1.0))

    def test_build_exact_reference_curves_has_matching_shapes(self) -> None:
        temperatures = np.array([2.0, 2.25, 2.5])

        result = temperature_sweep.build_exact_reference_curves(
            temperatures=temperatures,
            coupling=1.0,
            field=0.0,
            n_points=50,
            n_critical_points=100,
        )

        self.assertIsNotNone(result)
        assert result is not None

        self.assertEqual(
            result["energy_temperature"].shape,
            result["energy_density"].shape,
        )
        self.assertEqual(
            result["magnetization_temperature"].shape,
            result["spontaneous_magnetization"].shape,
        )

    def test_build_exact_reference_curves_has_matching_plot_shapes(self) -> None:
        temperatures = np.array([2.0, 2.25, 2.5])

        result = temperature_sweep.build_exact_reference_curves(
            temperatures=temperatures,
            coupling=1.0,
            field=0.0,
            n_points=50,
            n_critical_points=100,
        )

        self.assertIsNotNone(result)
        assert result is not None

        self.assertEqual(
            result["energy_temperature"].shape,
            result["energy_density"].shape,
        )
        self.assertEqual(
            result["magnetization_temperature"].shape,
            result["spontaneous_magnetization"].shape,
        )


if __name__ == "__main__":
    unittest.main()
