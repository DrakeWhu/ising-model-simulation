import unittest

import measure_lattice_equilibrium


class TestMeasureLatticeEquilibrium(unittest.TestCase):
    def test_make_checkerboard_spins_for_square_lattice(self) -> None:
        lattice = measure_lattice_equilibrium.lattice_factory.build_lattice(
            "square2d",
            size=4,
            periodic=True,
        )

        spins = measure_lattice_equilibrium.make_initial_spins(
            lattice,
            initial_state="checkerboard",
            seed=123,
        )

        self.assertEqual(spins.shape, (16,))
        self.assertEqual(spins[0], 1)
        self.assertEqual(spins[1], -1)
        self.assertEqual(spins[4], -1)

    def test_run_square2d_metropolis_smoke(self) -> None:
        result = measure_lattice_equilibrium.run_lattice_equilibrium(
            lattice_kind="square2d",
            size=4,
            periodic=True,
            backend="metropolis",
            temperature=2.5,
            coupling=1.0,
            thermalization_sweeps=1,
            measurement_sweeps=2,
            sample_every=1,
            initial_state="random",
            seed=123,
        )

        self.assertEqual(result.lattice, "square2d")
        self.assertEqual(result.backend, "metropolis")
        self.assertEqual(result.n_sites, 16)
        self.assertEqual(result.n_samples, 2)
        self.assertEqual(result.energy_error, 0.0)

    def test_run_cubic3d_wolff_smoke(self) -> None:
        result = measure_lattice_equilibrium.run_lattice_equilibrium(
            lattice_kind="cubic3d",
            size=4,
            periodic=True,
            backend="wolff",
            temperature=4.5,
            coupling=1.0,
            thermalization_sweeps=1,
            measurement_sweeps=2,
            sample_every=1,
            initial_state="random",
            seed=123,
        )

        self.assertEqual(result.lattice, "cubic3d")
        self.assertEqual(result.backend, "wolff")
        self.assertEqual(result.n_sites, 64)
        self.assertEqual(result.n_samples, 2)
        self.assertEqual(result.energy_error, 0.0)
        self.assertIsNotNone(result.mean_cluster_fraction)


if __name__ == "__main__":
    unittest.main()
