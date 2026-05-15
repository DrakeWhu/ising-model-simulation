import unittest

import numpy as np

import creation


class TestCreation(unittest.TestCase):
    def test_create_all_up_lattice(self) -> None:
        lattice = creation.create_spin_lattice(4, initial_state="all-up")

        self.assertEqual(lattice.dtype, np.int8)
        self.assertTrue(np.all(lattice == 1))

    def test_create_all_down_lattice(self) -> None:
        lattice = creation.create_spin_lattice(4, initial_state="all-down")

        self.assertEqual(lattice.dtype, np.int8)
        self.assertTrue(np.all(lattice == -1))

    def test_create_checkerboard_lattice(self) -> None:
        lattice = creation.create_spin_lattice(4, initial_state="checkerboard")

        expected = np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int8,
        )

        np.testing.assert_array_equal(lattice, expected)

    def test_create_random_lattice_has_valid_spin_values(self) -> None:
        lattice = creation.create_spin_lattice(16, initial_state="random")

        self.assertEqual(lattice.dtype, np.int8)
        self.assertTrue(set(np.unique(lattice)).issubset({-1, 1}))

    def test_create_lattice_rejects_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            creation.create_spin_lattice(0, initial_state="random")

    def test_create_lattice_rejects_unknown_initial_state(self) -> None:
        with self.assertRaises(ValueError):
            creation.create_spin_lattice(4, initial_state="wat")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
