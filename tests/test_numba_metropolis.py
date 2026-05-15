import unittest

import numpy as np

from metropolis import _delta_energy_periodic
from metropolis_numba import delta_energy_periodic_numba


class TestNumbaMetropolis(unittest.TestCase):
    def test_numba_delta_energy_matches_reference(self) -> None:
        lattice = np.array(
            [
                [1, 1, -1, 1],
                [-1, 1, -1, -1],
                [1, -1, 1, 1],
                [-1, -1, 1, -1],
            ],
            dtype=np.int8,
        )

        reference = _delta_energy_periodic(
            lattice,
            x_index=1,
            y_index=2,
            coupling=1.0,
            field=0.25,
        )
        accelerated = delta_energy_periodic_numba(
            lattice,
            x_index=1,
            y_index=2,
            coupling=1.0,
            field=0.25,
        )

        self.assertEqual(accelerated, reference)


if __name__ == "__main__":
    unittest.main()
