import numpy as np
from numba import njit


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """Seed Numba's internal random number generator."""
    np.random.seed(seed)


@njit(cache=True)
def bond_probability(beta: float, coupling: float) -> float:
    """Return Wolff bond activation probability for ferromagnetic Ising at h=0."""
    if beta < 0.0:
        raise ValueError("beta cannot be negative.")
    if coupling <= 0.0:
        raise ValueError("coupling must be positive for Wolff dynamics.")

    return 1.0 - np.exp(-2.0 * beta * coupling)


@njit(cache=True)
def _wolff_step_inplace(
    spin_arr: np.ndarray,
    beta: float,
    energy: float,
    coupling: float,
) -> tuple[float, float, int]:
    n_rows, n_cols = spin_arr.shape
    n_sites = spin_arr.size
    probability = bond_probability(beta, coupling)

    seed_x = np.random.randint(0, n_rows)
    seed_y = np.random.randint(0, n_cols)
    cluster_spin = spin_arr[seed_x, seed_y]

    in_cluster = np.zeros((n_rows, n_cols), dtype=np.bool_)
    stack_x = np.empty(n_sites, dtype=np.int64)
    stack_y = np.empty(n_sites, dtype=np.int64)
    cluster_x = np.empty(n_sites, dtype=np.int64)
    cluster_y = np.empty(n_sites, dtype=np.int64)

    stack_size = 1
    cluster_size = 1

    stack_x[0] = seed_x
    stack_y[0] = seed_y
    cluster_x[0] = seed_x
    cluster_y[0] = seed_y
    in_cluster[seed_x, seed_y] = True

    while stack_size > 0:
        stack_size -= 1
        x_index = stack_x[stack_size]
        y_index = stack_y[stack_size]

        for direction in range(4):
            if direction == 0:
                neighbour_x = (x_index - 1) % n_rows
                neighbour_y = y_index
            elif direction == 1:
                neighbour_x = (x_index + 1) % n_rows
                neighbour_y = y_index
            elif direction == 2:
                neighbour_x = x_index
                neighbour_y = (y_index - 1) % n_cols
            else:
                neighbour_x = x_index
                neighbour_y = (y_index + 1) % n_cols

            if in_cluster[neighbour_x, neighbour_y]:
                continue
            if spin_arr[neighbour_x, neighbour_y] != cluster_spin:
                continue
            if np.random.random() >= probability:
                continue

            in_cluster[neighbour_x, neighbour_y] = True
            stack_x[stack_size] = neighbour_x
            stack_y[stack_size] = neighbour_y
            stack_size += 1
            cluster_x[cluster_size] = neighbour_x
            cluster_y[cluster_size] = neighbour_y
            cluster_size += 1

    delta_energy = 0.0

    for cluster_index in range(cluster_size):
        x_index = cluster_x[cluster_index]
        y_index = cluster_y[cluster_index]
        spin = spin_arr[x_index, y_index]

        for direction in range(4):
            if direction == 0:
                neighbour_x = (x_index - 1) % n_rows
                neighbour_y = y_index
            elif direction == 1:
                neighbour_x = (x_index + 1) % n_rows
                neighbour_y = y_index
            elif direction == 2:
                neighbour_x = x_index
                neighbour_y = (y_index - 1) % n_cols
            else:
                neighbour_x = x_index
                neighbour_y = (y_index + 1) % n_cols

            if not in_cluster[neighbour_x, neighbour_y]:
                delta_energy += 2.0 * coupling * spin * spin_arr[neighbour_x, neighbour_y]

    for cluster_index in range(cluster_size):
        x_index = cluster_x[cluster_index]
        y_index = cluster_y[cluster_index]
        spin_arr[x_index, y_index] *= -1

    energy += delta_energy
    magnetization = spin_arr.sum()

    return energy, float(magnetization), cluster_size


@njit(cache=True)
def wolff_step_numba(
    spin_arr: np.ndarray,
    beta: float,
    energy: float,
    coupling: float,
) -> tuple[np.ndarray, int, float, float]:
    """Run one Wolff cluster flip on a copy of the input lattice."""
    spin_arr = spin_arr.copy()
    energy, magnetization, cluster_size = _wolff_step_inplace(
        spin_arr,
        beta=beta,
        energy=energy,
        coupling=coupling,
    )

    return spin_arr, cluster_size, energy, magnetization


@njit(cache=True)
def wolff_numba(
    spin_arr: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Wolff dynamics using effective sweeps.

    One effective Wolff sweep is defined as enough cluster flips for the sum of
    flipped cluster sizes to reach at least N_sites.
    """
    if n_sweeps < 0:
        raise ValueError("n_sweeps cannot be negative.")

    spin_arr = spin_arr.copy()
    n_sites = spin_arr.size

    net_spins = np.empty(n_sweeps + 1, dtype=np.float64)
    net_energy = np.empty(n_sweeps + 1, dtype=np.float64)

    net_spins[0] = spin_arr.sum()
    net_energy[0] = energy

    for sweep in range(1, n_sweeps + 1):
        flipped_sites = 0
        magnetization = net_spins[sweep - 1]

        while flipped_sites < n_sites:
            energy, magnetization, cluster_size = _wolff_step_inplace(
                spin_arr,
                beta=beta,
                energy=energy,
                coupling=coupling,
            )
            flipped_sites += cluster_size

        net_spins[sweep] = magnetization
        net_energy[sweep] = energy

    return spin_arr, net_spins, net_energy
