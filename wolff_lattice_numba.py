from __future__ import annotations

import numpy as np
from numba import njit

from lattices import Lattice


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """Seed Numba's internal random number generator."""
    np.random.seed(seed)


@njit(cache=True)
def _wolff_lattice_step_inplace(
    spins: np.ndarray,
    beta: float,
    energy: float,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    coupling: float,
) -> tuple[float, float, int]:
    """Perform one Wolff cluster flip on a generic lattice.

    Assumes ferromagnetic coupling and zero external field.
    """
    n_sites = spins.size

    if coupling < 0.0:
        raise ValueError("Wolff dynamics requires non-negative coupling.")

    add_probability = 1.0 - np.exp(-2.0 * beta * coupling)

    cluster_mask = np.zeros(n_sites, dtype=np.bool_)
    stack = np.empty(n_sites, dtype=np.int64)
    cluster_sites = np.empty(n_sites, dtype=np.int64)

    seed = np.random.randint(0, n_sites)
    target_spin = spins[seed]

    cluster_mask[seed] = True
    stack[0] = seed
    cluster_sites[0] = seed

    stack_size = 1
    cluster_size = 1

    while stack_size > 0:
        stack_size -= 1
        site = stack[stack_size]

        start = neighbor_offsets[site]
        end = neighbor_offsets[site + 1]

        for cursor in range(start, end):
            neighbor = neighbors[cursor]

            if cluster_mask[neighbor]:
                continue

            if spins[neighbor] != target_spin:
                continue

            if np.random.random() < add_probability:
                cluster_mask[neighbor] = True
                stack[stack_size] = neighbor
                stack_size += 1
                cluster_sites[cluster_size] = neighbor
                cluster_size += 1

    delta_energy = 0.0

    for cluster_cursor in range(cluster_size):
        site = cluster_sites[cluster_cursor]
        spin = spins[site]

        start = neighbor_offsets[site]
        end = neighbor_offsets[site + 1]

        for cursor in range(start, end):
            neighbor = neighbors[cursor]

            if not cluster_mask[neighbor]:
                delta_energy += 2.0 * coupling * spin * spins[neighbor]

    for cluster_cursor in range(cluster_size):
        site = cluster_sites[cluster_cursor]
        spins[site] *= -1

    energy += delta_energy
    magnetization = spins.sum()

    return energy, magnetization, cluster_size


@njit(cache=True)
def wolff_lattice_numba_with_cluster_stats(
    spins: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Wolff dynamics on a generic lattice with cluster stats.

    One effective Wolff sweep is defined as enough cluster flips for the sum of
    flipped cluster sizes to reach at least N_sites.
    """
    if n_sweeps < 0:
        raise ValueError("n_sweeps cannot be negative.")

    spins = spins.copy()
    n_sites = spins.size

    net_spins = np.empty(n_sweeps + 1, dtype=np.float64)
    net_energy = np.empty(n_sweeps + 1, dtype=np.float64)

    cluster_flips = np.empty(n_sweeps, dtype=np.int64)
    mean_cluster_sizes = np.empty(n_sweeps, dtype=np.float64)
    max_cluster_sizes = np.empty(n_sweeps, dtype=np.int64)

    net_spins[0] = spins.sum()
    net_energy[0] = energy

    for sweep in range(1, n_sweeps + 1):
        flipped_sites = 0
        sweep_cluster_flips = 0
        sweep_cluster_size_sum = 0.0
        sweep_max_cluster_size = 0

        magnetization = net_spins[sweep - 1]

        while flipped_sites < n_sites:
            energy, magnetization, cluster_size = _wolff_lattice_step_inplace(
                spins,
                beta=beta,
                energy=energy,
                neighbor_offsets=neighbor_offsets,
                neighbors=neighbors,
                coupling=coupling,
            )

            flipped_sites += cluster_size
            sweep_cluster_flips += 1
            sweep_cluster_size_sum += cluster_size

            if cluster_size > sweep_max_cluster_size:
                sweep_max_cluster_size = cluster_size

        net_spins[sweep] = magnetization
        net_energy[sweep] = energy

        cluster_flips[sweep - 1] = sweep_cluster_flips
        mean_cluster_sizes[sweep - 1] = sweep_cluster_size_sum / sweep_cluster_flips
        max_cluster_sizes[sweep - 1] = sweep_max_cluster_size

    return (
        spins,
        net_spins,
        net_energy,
        cluster_flips,
        mean_cluster_sizes,
        max_cluster_sizes,
    )


@njit(cache=True)
def wolff_lattice_numba(
    spins: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Wolff dynamics on a generic lattice."""
    (
        spins,
        net_spins,
        net_energy,
        _,
        _,
        _,
    ) = wolff_lattice_numba_with_cluster_stats(
        spins,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        neighbor_offsets=neighbor_offsets,
        neighbors=neighbors,
        coupling=coupling,
    )

    return spins, net_spins, net_energy


def run_wolff_lattice(
    spins: np.ndarray,
    lattice: Lattice,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run generic Wolff dynamics using a Lattice object."""
    flat_spins = np.asarray(spins, dtype=np.int8).reshape(-1)

    if flat_spins.size != lattice.n_sites:
        raise ValueError(
            f"Expected {lattice.n_sites} spins for lattice {lattice.name!r}, got {flat_spins.size}."
        )

    return wolff_lattice_numba(
        flat_spins,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        neighbor_offsets=lattice.neighbor_offsets,
        neighbors=lattice.neighbors,
        coupling=coupling,
    )


def run_wolff_lattice_with_cluster_stats(
    spins: np.ndarray,
    lattice: Lattice,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run generic Wolff dynamics using a Lattice object and return cluster stats."""
    flat_spins = np.asarray(spins, dtype=np.int8).reshape(-1)

    if flat_spins.size != lattice.n_sites:
        raise ValueError(
            f"Expected {lattice.n_sites} spins for lattice {lattice.name!r}, got {flat_spins.size}."
        )

    return wolff_lattice_numba_with_cluster_stats(
        flat_spins,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        neighbor_offsets=lattice.neighbor_offsets,
        neighbors=lattice.neighbors,
        coupling=coupling,
    )
