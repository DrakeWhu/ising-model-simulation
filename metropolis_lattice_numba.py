from __future__ import annotations

import numpy as np
from numba import njit

import ising_energy
from lattices import Lattice


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """Seed Numba's internal random number generator."""
    np.random.seed(seed)


@njit(cache=True)
def metropolis_lattice_numba(
    spins: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Metropolis dynamics on a generic lattice.

    The lattice connectivity is encoded through CSR-style neighbor arrays.

    One sweep is N_sites attempted spin flips.
    """
    if n_sweeps < 0:
        raise ValueError("n_sweeps cannot be negative.")

    spins = spins.copy()
    n_sites = spins.size
    n_attempts = n_sweeps * n_sites

    net_spins = np.empty(n_sweeps + 1, dtype=np.float64)
    net_energy = np.empty(n_sweeps + 1, dtype=np.float64)

    net_spins[0] = spins.sum()
    net_energy[0] = energy

    for attempt in range(1, n_attempts + 1):
        site_index = np.random.randint(0, n_sites)

        delta_energy = ising_energy.delta_energy_from_neighbors_numba(
            spins,
            site_index,
            neighbor_offsets,
            neighbors,
            coupling,
            field,
        )

        if delta_energy <= 0.0 or np.random.random() < np.exp(-beta * delta_energy):
            spins[site_index] *= -1
            energy += delta_energy

        if attempt % n_sites == 0:
            sweep = attempt // n_sites
            net_spins[sweep] = spins.sum()
            net_energy[sweep] = energy

    return spins, net_spins, net_energy


def run_metropolis_lattice(
    spins: np.ndarray,
    lattice: Lattice,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float = 1.0,
    field: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run generic Metropolis dynamics using a Lattice object."""
    flat_spins = np.asarray(spins, dtype=np.int8).reshape(-1)

    if flat_spins.size != lattice.n_sites:
        raise ValueError(
            f"Expected {lattice.n_sites} spins for lattice {lattice.name!r}, got {flat_spins.size}."
        )

    return metropolis_lattice_numba(
        flat_spins,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        neighbor_offsets=lattice.neighbor_offsets,
        neighbors=lattice.neighbors,
        coupling=coupling,
        field=field,
    )
