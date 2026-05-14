import numpy as np


def _delta_energy_periodic(
    spin_arr: np.ndarray,
    x_index: int,
    y_index: int,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    """Energy change for flipping one spin in a periodic square lattice."""
    n_rows, n_cols = spin_arr.shape
    spin = spin_arr[x_index, y_index]

    neighbour_sum = (
        spin_arr[(x_index - 1) % n_rows, y_index]
        + spin_arr[(x_index + 1) % n_rows, y_index]
        + spin_arr[x_index, (y_index - 1) % n_cols]
        + spin_arr[x_index, (y_index + 1) % n_cols]
    )

    return float(2.0 * coupling * spin * neighbour_sum + 2.0 * field * spin)


def metropolis(
    spin_arr: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float = 1.0,
    field: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Metropolis dynamics on a 2D square Ising lattice.

    One sweep is defined as N_sites attempted spin flips.
    """
    spin_arr = spin_arr.copy()
    n_rows, n_cols = spin_arr.shape
    n_sites = spin_arr.size
    n_attempts = n_sweeps * n_sites

    net_spins = np.empty(n_sweeps + 1)
    net_energy = np.empty(n_sweeps + 1)

    net_spins[0] = spin_arr.sum()
    net_energy[0] = energy

    for attempt in range(1, n_attempts + 1):
        x_index = np.random.randint(0, n_rows)
        y_index = np.random.randint(0, n_cols)

        delta_energy = _delta_energy_periodic(
            spin_arr,
            x_index,
            y_index,
            coupling=coupling,
            field=field,
        )

        if delta_energy <= 0.0 or np.random.random() < np.exp(-beta * delta_energy):
            spin_arr[x_index, y_index] *= -1
            energy += delta_energy

        if attempt % n_sites == 0:
            sweep = attempt // n_sites
            net_spins[sweep] = spin_arr.sum()
            net_energy[sweep] = energy

    return spin_arr, net_spins, net_energy