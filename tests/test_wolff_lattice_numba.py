import unittest

import numpy as np

import ising_energy
import lattices
import wolff_lattice_numba


class TestWolffLatticeNumba(unittest.TestCase):
    def test_generic_wolff_returns_expected_shapes(self) -> None:
        lattice = lattices.square_lattice_2d(size=6, periodic=True)
        spins = np.ones(lattice.n_sites, dtype=np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        wolff_lattice_numba.seed_numba_rng(123)
        updated_spins, magnetizations, energies = wolff_lattice_numba.run_wolff_lattice(
            spins,
            lattice,
            n_sweeps=5,
            beta=1.0 / 2.5,
            energy=energy,
            coupling=1.0,
        )

        self.assertEqual(updated_spins.shape, (lattice.n_sites,))
        self.assertEqual(magnetizations.shape, (6,))
        self.assertEqual(energies.shape, (6,))
        self.assertTrue(np.all(np.isin(updated_spins, [-1, 1])))

    def test_generic_wolff_energy_matches_recomputed_energy(self) -> None:
        lattice = lattices.square_lattice_2d(size=8, periodic=True)

        rng = np.random.default_rng(123)
        spins = rng.choice([-1, 1], size=lattice.n_sites).astype(np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        wolff_lattice_numba.seed_numba_rng(456)
        updated_spins, _, energies = wolff_lattice_numba.run_wolff_lattice(
            spins,
            lattice,
            n_sweeps=10,
            beta=1.0 / 2.5,
            energy=energy,
            coupling=1.0,
        )

        recomputed_energy = ising_energy.get_lattice_energy(
            updated_spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        self.assertAlmostEqual(energies[-1], recomputed_energy)

    def test_generic_wolff_supports_cubic_3d_lattice(self) -> None:
        lattice = lattices.cubic_lattice_3d(size=4, periodic=True)

        rng = np.random.default_rng(123)
        spins = rng.choice([-1, 1], size=lattice.n_sites).astype(np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        wolff_lattice_numba.seed_numba_rng(789)
        updated_spins, magnetizations, energies = wolff_lattice_numba.run_wolff_lattice(
            spins,
            lattice,
            n_sweeps=3,
            beta=1.0 / 4.5,
            energy=energy,
            coupling=1.0,
        )

        recomputed_energy = ising_energy.get_lattice_energy(
            updated_spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        self.assertEqual(updated_spins.shape, (lattice.n_sites,))
        self.assertEqual(magnetizations.shape, (4,))
        self.assertEqual(energies.shape, (4,))
        self.assertAlmostEqual(energies[-1], recomputed_energy)

    def test_generic_wolff_with_cluster_stats_returns_per_sweep_stats(self) -> None:
        lattice = lattices.square_lattice_2d(size=8, periodic=True)
        spins = np.ones(lattice.n_sites, dtype=np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        wolff_lattice_numba.seed_numba_rng(123)
        (
            updated_spins,
            magnetizations,
            energies,
            cluster_flips,
            mean_cluster_sizes,
            max_cluster_sizes,
        ) = wolff_lattice_numba.run_wolff_lattice_with_cluster_stats(
            spins,
            lattice,
            n_sweeps=3,
            beta=1.0 / 2.0,
            energy=energy,
            coupling=1.0,
        )

        self.assertEqual(updated_spins.shape, (lattice.n_sites,))
        self.assertEqual(magnetizations.shape, (4,))
        self.assertEqual(energies.shape, (4,))
        self.assertEqual(cluster_flips.shape, (3,))
        self.assertEqual(mean_cluster_sizes.shape, (3,))
        self.assertEqual(max_cluster_sizes.shape, (3,))

        self.assertTrue(np.all(cluster_flips >= 1))
        self.assertTrue(np.all(mean_cluster_sizes >= 1.0))
        self.assertTrue(np.all(mean_cluster_sizes <= lattice.n_sites))
        self.assertTrue(np.all(max_cluster_sizes >= 1))
        self.assertTrue(np.all(max_cluster_sizes <= lattice.n_sites))

        recomputed_energy = ising_energy.get_lattice_energy(
            updated_spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        self.assertAlmostEqual(energies[-1], recomputed_energy)

    def test_zero_sweeps_returns_initial_state_observables(self) -> None:
        lattice = lattices.square_lattice_2d(size=4, periodic=True)
        spins = np.ones(lattice.n_sites, dtype=np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        updated_spins, magnetizations, energies = wolff_lattice_numba.run_wolff_lattice(
            spins,
            lattice,
            n_sweeps=0,
            beta=1.0 / 2.5,
            energy=energy,
            coupling=1.0,
        )

        np.testing.assert_array_equal(updated_spins, spins)
        self.assertEqual(magnetizations.shape, (1,))
        self.assertEqual(energies.shape, (1,))
        self.assertEqual(magnetizations[0], lattice.n_sites)
        self.assertEqual(energies[0], energy)

    def test_rejects_wrong_number_of_spins(self) -> None:
        lattice = lattices.square_lattice_2d(size=4, periodic=True)
        spins = np.ones(lattice.n_sites - 1, dtype=np.int8)

        with self.assertRaises(ValueError):
            wolff_lattice_numba.run_wolff_lattice(
                spins,
                lattice,
                n_sweeps=1,
                beta=1.0,
                energy=0.0,
                coupling=1.0,
            )


if __name__ == "__main__":
    unittest.main()
