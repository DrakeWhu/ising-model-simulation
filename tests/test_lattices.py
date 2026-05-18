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

    def test_periodic_triangular_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.triangular_lattice_2d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 16)
        self.assertEqual(lattice.shape, (4, 4))
        self.assertTrue(np.all(degrees == 6))
        self.assertEqual(len(lattice.edge_u), 48)
        self.assertEqual(len(lattice.edge_v), 48)

    def test_open_triangular_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.triangular_lattice_2d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 16)
        self.assertEqual(len(lattice.edge_u), 33)
        self.assertEqual(len(lattice.edge_v), 33)

    def test_periodic_hexagonal_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.hexagonal_lattice_2d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 32)
        self.assertEqual(lattice.shape, (4, 4, 2))
        self.assertTrue(np.all(degrees == 3))
        self.assertEqual(len(lattice.edge_u), 48)
        self.assertEqual(len(lattice.edge_v), 48)

    def test_open_hexagonal_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.hexagonal_lattice_2d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 32)
        self.assertEqual(len(lattice.edge_u), 40)
        self.assertEqual(len(lattice.edge_v), 40)

    def test_periodic_bcc_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.bcc_lattice_3d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 128)
        self.assertEqual(lattice.shape, (4, 4, 4, 2))
        self.assertTrue(np.all(degrees == 8))
        self.assertEqual(len(lattice.edge_u), 512)
        self.assertEqual(len(lattice.edge_v), 512)

    def test_open_bcc_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.bcc_lattice_3d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 128)
        self.assertEqual(len(lattice.edge_u), 343)
        self.assertEqual(len(lattice.edge_v), 343)

    def test_periodic_fcc_lattice_has_expected_connectivity(self) -> None:
        lattice = lattices.fcc_lattice_3d(size=4, periodic=True)

        degrees = np.diff(lattice.neighbor_offsets)

        self.assertEqual(lattice.n_sites, 256)
        self.assertEqual(lattice.shape, (4, 4, 4, 4))
        self.assertTrue(np.all(degrees == 12))
        self.assertEqual(len(lattice.edge_u), 1536)
        self.assertEqual(len(lattice.edge_v), 1536)

    def test_open_fcc_lattice_has_expected_edge_count(self) -> None:
        lattice = lattices.fcc_lattice_3d(size=4, periodic=False)

        self.assertEqual(lattice.n_sites, 256)
        self.assertEqual(len(lattice.edge_u), 1176)
        self.assertEqual(len(lattice.edge_v), 1176)


if __name__ == "__main__":
    unittest.main()
