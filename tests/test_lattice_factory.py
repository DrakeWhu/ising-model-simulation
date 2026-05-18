import unittest

import lattice_factory


class TestLatticeFactory(unittest.TestCase):
    def test_build_square2d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("square2d", size=4, periodic=True)

        self.assertEqual(lattice.name, "square_2d_periodic")
        self.assertEqual(lattice.shape, (4, 4))
        self.assertEqual(lattice.n_sites, 16)

    def test_build_cubic3d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("cubic3d", size=4, periodic=True)

        self.assertEqual(lattice.name, "cubic_3d_periodic")
        self.assertEqual(lattice.shape, (4, 4, 4))
        self.assertEqual(lattice.n_sites, 64)

    def test_rejects_unknown_lattice(self) -> None:
        with self.assertRaises(ValueError):
            lattice_factory.build_lattice("not_a_lattice", size=4, periodic=True)

    def test_build_triangular2d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("triangular2d", size=4, periodic=True)

        self.assertEqual(lattice.name, "triangular_2d_periodic")
        self.assertEqual(lattice.shape, (4, 4))
        self.assertEqual(lattice.n_sites, 16)

    def test_build_hexagonal2d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("hexagonal2d", size=4, periodic=True)

        self.assertEqual(lattice.name, "hexagonal_2d_periodic")
        self.assertEqual(lattice.shape, (4, 4, 2))
        self.assertEqual(lattice.n_sites, 32)

    def test_build_bcc3d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("bcc3d", size=4, periodic=True)

        self.assertEqual(lattice.name, "bcc_3d_periodic")
        self.assertEqual(lattice.shape, (4, 4, 4, 2))
        self.assertEqual(lattice.n_sites, 128)

    def test_build_fcc3d_lattice(self) -> None:
        lattice = lattice_factory.build_lattice("fcc3d", size=4, periodic=True)

        self.assertEqual(lattice.name, "fcc_3d_periodic")
        self.assertEqual(lattice.shape, (4, 4, 4, 4))
        self.assertEqual(lattice.n_sites, 256)


if __name__ == "__main__":
    unittest.main()
