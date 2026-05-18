import unittest

import numpy as np

import lattices


class TestLattices(unittest.TestCase):
    def test_periodic_square_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.square_lattice_2d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 16)
        self.assertEqual(lattice.shape, (4, 4))
        self.assertTrue(np.all(degrees == 4))
        self.assertEqual(len(lattice.edge_u), 32)
        self.assertEqual(len(lattice.edge_v), 32)

    def test_open_square_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.square_lattice_2d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 16)
        self.assertEqual(len(lattice.edge_u), 24)
        self.assertEqual(len(lattice.edge_v), 24)

    def test_periodic_cubic_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.cubic_lattice_3d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 64)
        self.assertEqual(lattice.shape, (4, 4, 4))
        self.assertTrue(np.all(degrees == 6))
        self.assertEqual(len(lattice.edge_u), 192)
        self.assertEqual(len(lattice.edge_v), 192)

    def test_open_cubic_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.cubic_lattice_3d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 64)
        self.assertEqual(len(lattice.edge_u), 144)
        self.assertEqual(len(lattice.edge_v), 144)


if __name__ == "__main__":
    unittest.main()
