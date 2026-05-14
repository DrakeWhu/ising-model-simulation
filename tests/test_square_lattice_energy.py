import unittest

import numpy as np

from metropolis import _delta_energy_periodic
from nearest_neighbour import get_energy


class TestSquareLatticeIsingEnergy(unittest.TestCase):
    def test_all_up_2x2_periodic_energy(self) -> None:
        lattice = np.ones((2, 2), dtype=np.int8)

        energy = get_energy(lattice, coupling=1.0, field=0.0)

        self.assertEqual(energy, -8.0)

    def test_all_up_lattice_energy_with_field(self) -> None:
        size = 4
        coupling = 1.0
        field = 0.5
        lattice = np.ones((size, size), dtype=np.int8)

        energy = get_energy(lattice, coupling=coupling, field=field)

        expected_interaction_energy = -2.0 * coupling * size**2
        expected_field_energy = -field * size**2
        self.assertEqual(energy, expected_interaction_energy + expected_field_energy)

    def test_checkerboard_energy(self) -> None:
        size = 4
        indices = np.indices((size, size)).sum(axis=0)
        lattice = np.where(indices % 2 == 0, 1, -1).astype(np.int8)

        energy = get_energy(lattice, coupling=1.0, field=0.0)

        self.assertEqual(energy, 2.0 * size**2)

    def test_delta_energy_for_ordered_lattice_spin_flip(self) -> None:
        lattice = np.ones((4, 4), dtype=np.int8)

        delta_energy = _delta_energy_periodic(
            lattice,
            x_index=0,
            y_index=0,
            coupling=1.0,
            field=0.0,
        )

        self.assertEqual(delta_energy, 8.0)

    def test_delta_energy_matches_total_energy_difference(self) -> None:
        lattice = np.array(
            [
                [1, 1, -1, 1],
                [-1, 1, -1, -1],
                [1, -1, 1, 1],
                [-1, -1, 1, -1],
            ],
            dtype=np.int8,
        )
        x_index = 1
        y_index = 2

        energy_before = get_energy(lattice, coupling=1.0, field=0.25)
        delta_energy = _delta_energy_periodic(
            lattice,
            x_index=x_index,
            y_index=y_index,
            coupling=1.0,
            field=0.25,
        )

        flipped_lattice = lattice.copy()
        flipped_lattice[x_index, y_index] *= -1
        energy_after = get_energy(flipped_lattice, coupling=1.0, field=0.25)

        self.assertEqual(energy_after - energy_before, delta_energy)


if __name__ == "__main__":
    unittest.main()