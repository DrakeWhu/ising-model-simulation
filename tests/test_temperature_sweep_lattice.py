import tempfile
import unittest
from pathlib import Path

import numpy as np

import temperature_sweep_lattice


class TestTemperatureSweepLattice(unittest.TestCase):
    def test_build_temperature_grid(self) -> None:
        temperatures = temperature_sweep_lattice.build_temperature_grid(
            t_min=1.0,
            t_max=3.0,
            num_temperatures=3,
        )

        np.testing.assert_allclose(temperatures, np.array([1.0, 2.0, 3.0]))

    def test_run_square2d_wolff_sweep_smoke(self) -> None:
        temperatures = np.array([2.0, 2.5])

        results = temperature_sweep_lattice.run_temperature_sweep(
            lattice_kind="square2d",
            size=4,
            periodic=True,
            backend="wolff",
            temperatures=temperatures,
            coupling=1.0,
            thermalization_sweeps=1,
            measurement_sweeps=2,
            sample_every=1,
            initial_state="random",
            seed=123,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].lattice, "square2d")
        self.assertEqual(results[0].backend, "wolff")
        self.assertEqual(results[0].n_sites, 16)
        self.assertEqual(results[0].energy_error, 0.0)

    def test_save_results_csv(self) -> None:
        temperatures = np.array([2.0])

        results = temperature_sweep_lattice.run_temperature_sweep(
            lattice_kind="square2d",
            size=4,
            periodic=True,
            backend="metropolis",
            temperatures=temperatures,
            coupling=1.0,
            thermalization_sweeps=1,
            measurement_sweeps=2,
            sample_every=1,
            initial_state="random",
            seed=123,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sweep.csv"

            temperature_sweep_lattice.save_results_csv(results, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_temperature_sweep_smoke(self) -> None:
        temperatures = np.array([2.0, 2.5])

        results = temperature_sweep_lattice.run_temperature_sweep(
            lattice_kind="square2d",
            size=4,
            periodic=True,
            backend="wolff",
            temperatures=temperatures,
            coupling=1.0,
            thermalization_sweeps=1,
            measurement_sweeps=2,
            sample_every=1,
            initial_state="random",
            seed=123,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sweep.png"

            temperature_sweep_lattice.plot_temperature_sweep(
                results,
                critical_temperature=2.2691853,
                save_path=output_path,
                show=False,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
