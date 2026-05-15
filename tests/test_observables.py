import math
import unittest

import numpy as np

import observables


class TestObservables(unittest.TestCase):
    def test_energy_density(self) -> None:
        energies = np.array([-8.0, -4.0, 0.0])

        result = observables.energy_density(energies, n_sites=4)

        np.testing.assert_allclose(result, np.array([-2.0, -1.0, 0.0]))

    def test_magnetization_density(self) -> None:
        magnetizations = np.array([-4.0, 0.0, 4.0])

        result = observables.magnetization_density(magnetizations, n_sites=4)

        np.testing.assert_allclose(result, np.array([-1.0, 0.0, 1.0]))

    def test_specific_heat_per_spin(self) -> None:
        energies = np.array([-2.0, 0.0, 2.0])
        beta = 0.5
        n_sites = 2

        result = observables.specific_heat_per_spin(energies, beta, n_sites)

        expected_variance = np.mean(energies**2) - np.mean(energies) ** 2
        expected = beta**2 * expected_variance / n_sites
        self.assertAlmostEqual(result, expected)

    def test_susceptibility_per_spin(self) -> None:
        magnetizations = np.array([-2.0, 0.0, 2.0])
        beta = 0.5
        n_sites = 2

        result = observables.susceptibility_per_spin(magnetizations, beta, n_sites)

        expected_variance = np.mean(magnetizations**2) - np.mean(magnetizations) ** 2
        expected = beta * expected_variance / n_sites
        self.assertAlmostEqual(result, expected)

    def test_binder_cumulant_for_constant_nonzero_magnetization(self) -> None:
        magnetizations = np.array([4.0, 4.0, 4.0, 4.0])

        result = observables.binder_cumulant(magnetizations)

        self.assertAlmostEqual(result, 2.0 / 3.0)

    def test_binder_cumulant_returns_nan_for_zero_second_moment(self) -> None:
        magnetizations = np.zeros(4)

        result = observables.binder_cumulant(magnetizations)

        self.assertTrue(math.isnan(result))

    def test_summarize_observables(self) -> None:
        energies = np.array([-8.0, -4.0, -4.0])
        magnetizations = np.array([4.0, 2.0, -2.0])
        beta = 0.5
        n_sites = 4

        summary = observables.summarize_observables(
            energies,
            magnetizations,
            beta,
            n_sites,
        )

        self.assertAlmostEqual(summary.energy_mean, np.mean(energies))
        self.assertAlmostEqual(summary.energy_density_mean, np.mean(energies) / n_sites)
        self.assertAlmostEqual(summary.magnetization_mean, np.mean(magnetizations) / n_sites)
        self.assertAlmostEqual(
            summary.abs_magnetization_mean,
            np.mean(np.abs(magnetizations)) / n_sites,
        )


if __name__ == "__main__":
    unittest.main()
