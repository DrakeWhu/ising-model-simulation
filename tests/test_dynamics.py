import unittest

import numpy as np

import dynamics
import nearest_neighbour as nn


class TestDynamics(unittest.TestCase):
    def test_metropolis_backend_returns_expected_shapes(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        dynamics.seed_backend_rng("metropolis", 123)
        result = dynamics.run_dynamics(
            lattice=lattice,
            n_sweeps=2,
            beta=1.0 / 2.0,
            energy=energy,
            coupling=1.0,
            field=0.0,
            backend="metropolis",
        )

        self.assertEqual(result.magnetizations.shape, (3,))
        self.assertEqual(result.energies.shape, (3,))
        self.assertTrue(np.all(np.isin(result.lattice, [-1, 1])))

    def test_wolff_backend_returns_expected_shapes(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        dynamics.seed_backend_rng("wolff", 123)
        result = dynamics.run_dynamics(
            lattice=lattice,
            n_sweeps=2,
            beta=1.0 / 2.0,
            energy=energy,
            coupling=1.0,
            field=0.0,
            backend="wolff",
        )

        self.assertEqual(result.magnetizations.shape, (3,))
        self.assertEqual(result.energies.shape, (3,))
        self.assertTrue(np.all(np.isin(result.lattice, [-1, 1])))

    def test_wolff_backend_rejects_nonzero_field(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        with self.assertRaises(ValueError):
            dynamics.run_dynamics(
                lattice=lattice,
                n_sweeps=1,
                beta=1.0 / 2.0,
                energy=energy,
                coupling=1.0,
                field=0.1,
                backend="wolff",
            )

    def test_unknown_backend_is_rejected(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        with self.assertRaises(ValueError):
            dynamics.run_dynamics(
                lattice=lattice,
                n_sweeps=1,
                beta=1.0 / 2.0,
                energy=energy,
                coupling=1.0,
                field=0.0,
                backend="glauber_but_spicy",
            )


if __name__ == "__main__":
    unittest.main()
