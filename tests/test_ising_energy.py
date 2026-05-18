import unittest

import numpy as np

import ising_energy
import lattices
from nearest_neighbour import get_energy as get_square_lattice_energy


class TestIsingEnergy(unittest.TestCase):
    def test_periodic_square_lattice_matches_existing_energy(self) -> None:
        size = 6
        coupling = 1.0
        field = 0.25

        rng = np.random.default_rng(123)
        spins_2d = rng.choice([-1, 1], size=(size, size)).astype(np.int8)

        lattice = lattices.square_lattice_2d(size=size, periodic=True)

        general_energy = ising_energy.get_lattice_energy(
            spins_2d,
            lattice,
            coupling=coupling,
            field=field,
        )
        specialized_energy = get_square_lattice_energy(
            spins_2d,
            coupling=coupling,
            field=field,
        )

        self.assertAlmostEqual(general_energy, specialized_energy)

    def test_periodic_square_checkerboard_energy(self) -> None:
        size = 4
        indices = np.indices((size, size)).sum(axis=0)
        spins_2d = np.where(indices % 2 == 0, 1, -1).astype(np.int8)

        lattice = lattices.square_lattice_2d(size=size, periodic=True)

        energy = ising_energy.get_lattice_energy(
            spins_2d,
            lattice,
            coupling=1.0,
            field=0.0,
        )

        self.assertEqual(energy, 2.0 * size**2)

    def test_periodic_cubic_all_up_energy(self) -> None:
        size = 4
        coupling = 1.5
        field = 0.25

        lattice = lattices.cubic_lattice_3d(size=size, periodic=True)
        spins = np.ones(lattice.n_sites, dtype=np.int8)

        energy = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=coupling,
            field=field,
        )

        expected_energy = -coupling * len(lattice.edge_u) - field * lattice.n_sites

        self.assertEqual(energy, expected_energy)

    def test_delta_energy_matches_total_energy_difference(self) -> None:
        size = 4
        coupling = 1.0
        field = 0.25

        spins_2d = np.array(
            [
                [1, 1, -1, 1],
                [-1, 1, -1, -1],
                [1, -1, 1, 1],
                [-1, -1, 1, -1],
            ],
            dtype=np.int8,
        )

        lattice = lattices.square_lattice_2d(size=size, periodic=True)
        spins = spins_2d.reshape(-1).copy()

        site_index = 1 * size + 2

        energy_before = ising_energy.get_lattice_energy(
            spins,
            lattice,
            coupling=coupling,
            field=field,
        )

        delta_energy = ising_energy.delta_energy_for_site(
            spins,
            lattice,
            site_index=site_index,
            coupling=coupling,
            field=field,
        )

        flipped_spins = spins.copy()
        flipped_spins[site_index] *= -1

        energy_after = ising_energy.get_lattice_energy(
            flipped_spins,
            lattice,
            coupling=coupling,
            field=field,
        )

        self.assertAlmostEqual(energy_after - energy_before, delta_energy)

    def test_rejects_wrong_number_of_spins(self) -> None:
        lattice = lattices.square_lattice_2d(size=4, periodic=True)
        spins = np.ones(15, dtype=np.int8)

        with self.assertRaises(ValueError):
            ising_energy.get_lattice_energy(spins, lattice)


if __name__ == "__main__":
    unittest.main()
