import unittest

import numpy as np

import nearest_neighbour as nn
import wolff_numba


class TestWolffNumba(unittest.TestCase):
    def test_bond_probability_is_zero_at_infinite_temperature(self) -> None:
        result = wolff_numba.bond_probability(beta=0.0, coupling=1.0)

        self.assertEqual(result, 0.0)

    def test_bond_probability_is_between_zero_and_one(self) -> None:
        result = wolff_numba.bond_probability(beta=0.5, coupling=1.0)

        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_single_step_preserves_ising_spins(self) -> None:
        lattice = np.array(
            [
                [1, 1, -1, 1],
                [-1, 1, -1, -1],
                [1, -1, 1, 1],
                [-1, -1, 1, -1],
            ],
            dtype=np.int8,
        )
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        wolff_numba.seed_numba_rng(123)
        updated_lattice, cluster_size, updated_energy, magnetization = wolff_numba.wolff_step_numba(
            lattice,
            beta=1.0 / 2.0,
            energy=energy,
            coupling=1.0,
        )

        self.assertGreaterEqual(cluster_size, 1)
        self.assertLessEqual(cluster_size, lattice.size)
        self.assertTrue(np.all(np.isin(updated_lattice, [-1, 1])))
        self.assertEqual(magnetization, float(updated_lattice.sum()))
        self.assertAlmostEqual(
            updated_energy,
            nn.get_energy(updated_lattice, coupling=1.0, field=0.0),
        )

    def test_zero_beta_flips_single_seed_spin(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        wolff_numba.seed_numba_rng(123)
        updated_lattice, cluster_size, updated_energy, magnetization = wolff_numba.wolff_step_numba(
            lattice,
            beta=0.0,
            energy=energy,
            coupling=1.0,
        )

        self.assertEqual(cluster_size, 1)
        self.assertEqual(updated_lattice.sum(), 14)
        self.assertEqual(magnetization, 14.0)
        self.assertAlmostEqual(updated_energy, -24.0)
        self.assertAlmostEqual(
            updated_energy,
            nn.get_energy(updated_lattice, coupling=1.0, field=0.0),
        )

    def test_probability_one_flips_full_ordered_lattice(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        wolff_numba.seed_numba_rng(123)
        updated_lattice, cluster_size, updated_energy, magnetization = wolff_numba.wolff_step_numba(
            lattice,
            beta=1000.0,
            energy=energy,
            coupling=1.0,
        )

        self.assertEqual(cluster_size, lattice.size)
        self.assertTrue(np.all(updated_lattice == -1))
        self.assertEqual(magnetization, -16.0)
        self.assertAlmostEqual(updated_energy, energy)

    def test_wolff_numba_returns_one_sample_per_effective_sweep(self) -> None:
        lattice = np.ones((8, 8), dtype=np.int8)
        energy = nn.get_energy(lattice, coupling=1.0, field=0.0)

        wolff_numba.seed_numba_rng(123)
        updated_lattice, spins, energies = wolff_numba.wolff_numba(
            lattice,
            n_sweeps=3,
            beta=1.0 / 2.0,
            energy=energy,
            coupling=1.0,
        )

        self.assertEqual(spins.shape, (4,))
        self.assertEqual(energies.shape, (4,))
        self.assertTrue(np.all(np.isin(updated_lattice, [-1, 1])))
        self.assertAlmostEqual(
            energies[-1],
            nn.get_energy(updated_lattice, coupling=1.0, field=0.0),
        )


if __name__ == "__main__":
    unittest.main()
