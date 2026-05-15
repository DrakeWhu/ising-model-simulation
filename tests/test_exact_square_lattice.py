import math
import unittest

import exact_square_lattice


class TestExactSquareLattice(unittest.TestCase):
    def test_critical_temperature_for_unit_coupling(self) -> None:
        result = exact_square_lattice.critical_temperature()

        self.assertAlmostEqual(result, 2.269185314213022)

    def test_critical_temperature_scales_with_coupling(self) -> None:
        result = exact_square_lattice.critical_temperature(coupling=2.0)

        self.assertAlmostEqual(result, 2.0 * 2.269185314213022)

    def test_critical_temperature_rejects_non_positive_coupling(self) -> None:
        with self.assertRaises(ValueError):
            exact_square_lattice.critical_temperature(coupling=0.0)

    def test_spontaneous_magnetization_is_zero_at_tc(self) -> None:
        tc = exact_square_lattice.critical_temperature()

        result = exact_square_lattice.exact_spontaneous_magnetization_density(tc)

        self.assertEqual(result, 0.0)

    def test_spontaneous_magnetization_is_zero_above_tc(self) -> None:
        result = exact_square_lattice.exact_spontaneous_magnetization_density(3.0)

        self.assertEqual(result, 0.0)

    def test_spontaneous_magnetization_tends_to_one_at_low_temperature(self) -> None:
        result = exact_square_lattice.exact_spontaneous_magnetization_density(0.5)

        self.assertAlmostEqual(result, 1.0, places=4)

    def test_energy_density_at_critical_temperature(self) -> None:
        tc = exact_square_lattice.critical_temperature()

        result = exact_square_lattice.exact_energy_density(tc)

        self.assertAlmostEqual(result, -math.sqrt(2.0))

    def test_energy_density_tends_to_ground_state_at_low_temperature(self) -> None:
        result = exact_square_lattice.exact_energy_density(0.5)

        self.assertAlmostEqual(result, -2.0, places=3)

    def test_energy_density_tends_to_zero_at_high_temperature(self) -> None:
        result = exact_square_lattice.exact_energy_density(100.0)

        self.assertLess(abs(result), 0.05)

    def test_exact_functions_reject_non_positive_temperature(self) -> None:
        with self.assertRaises(ValueError):
            exact_square_lattice.exact_energy_density(0.0)

        with self.assertRaises(ValueError):
            exact_square_lattice.exact_spontaneous_magnetization_density(0.0)

    def test_exact_functions_reject_non_positive_coupling(self) -> None:
        with self.assertRaises(ValueError):
            exact_square_lattice.exact_energy_density(1.0, coupling=0.0)

        with self.assertRaises(ValueError):
            exact_square_lattice.exact_spontaneous_magnetization_density(1.0, coupling=0.0)


if __name__ == "__main__":
    unittest.main()
