from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservableSummary:
    energy_mean: float
    energy_density_mean: float
    magnetization_mean: float
    abs_magnetization_mean: float
    specific_heat_per_spin: float
    susceptibility_per_spin: float
    abs_susceptibility_per_spin: float
    binder_cumulant: float


def energy_density(energies: np.ndarray, n_sites: int) -> np.ndarray:
    """Return E/N for each sampled energy."""
    return np.asarray(energies, dtype=np.float64) / n_sites


def magnetization_density(magnetizations: np.ndarray, n_sites: int) -> np.ndarray:
    """Return M/N for each sampled total magnetization."""
    return np.asarray(magnetizations, dtype=np.float64) / n_sites


def specific_heat_per_spin(energies: np.ndarray, beta: float, n_sites: int) -> float:
    """Compute C_V/N = beta^2 / N * (⟨E²⟩ - ⟨E⟩²), with k_B = 1."""
    energies = np.asarray(energies, dtype=np.float64)
    variance = np.mean(energies**2) - np.mean(energies) ** 2
    return float(beta**2 * variance / n_sites)


def susceptibility_per_spin(magnetizations: np.ndarray, beta: float, n_sites: int) -> float:
    """Compute chi/N = beta / N * (⟨M²⟩ - ⟨M⟩²)."""
    magnetizations = np.asarray(magnetizations, dtype=np.float64)
    variance = np.mean(magnetizations**2) - np.mean(magnetizations) ** 2
    return float(beta * variance / n_sites)


def abs_susceptibility_per_spin(magnetizations: np.ndarray, beta: float, n_sites: int) -> float:
    """Compute chi_abs/N = beta / N * (⟨M²⟩ - ⟨|M|⟩²).

    This is often useful for finite systems at h=0, where the magnetization can
    flip sign between symmetry-related phases.
    """
    magnetizations = np.asarray(magnetizations, dtype=np.float64)
    variance_like = np.mean(magnetizations**2) - np.mean(np.abs(magnetizations)) ** 2
    return float(beta * variance_like / n_sites)


def binder_cumulant(magnetizations: np.ndarray) -> float:
    """Compute U_4 = 1 - ⟨M⁴⟩ / (3 ⟨M²⟩²)."""
    magnetizations = np.asarray(magnetizations, dtype=np.float64)
    second_moment = np.mean(magnetizations**2)

    if second_moment == 0.0:
        return float("nan")

    fourth_moment = np.mean(magnetizations**4)
    return float(1.0 - fourth_moment / (3.0 * second_moment**2))


def summarize_observables(
    energies: np.ndarray,
    magnetizations: np.ndarray,
    beta: float,
    n_sites: int,
) -> ObservableSummary:
    """Summarize equilibrium observables from sampled energies and magnetizations."""
    energies = np.asarray(energies, dtype=np.float64)
    magnetizations = np.asarray(magnetizations, dtype=np.float64)

    return ObservableSummary(
        energy_mean=float(np.mean(energies)),
        energy_density_mean=float(np.mean(energy_density(energies, n_sites))),
        magnetization_mean=float(np.mean(magnetization_density(magnetizations, n_sites))),
        abs_magnetization_mean=float(np.mean(np.abs(magnetizations)) / n_sites),
        specific_heat_per_spin=specific_heat_per_spin(energies, beta, n_sites),
        susceptibility_per_spin=susceptibility_per_spin(magnetizations, beta, n_sites),
        abs_susceptibility_per_spin=abs_susceptibility_per_spin(
            magnetizations,
            beta,
            n_sites,
        ),
        binder_cumulant=binder_cumulant(magnetizations),
    )
