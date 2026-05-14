import numpy as np


def get_energy(lattice: np.ndarray, coupling: float = 1.0, field: float = 0.0) -> float:
    """Compute the 2D square-lattice Ising energy with periodic boundaries.

    Hamiltonian:
        H = -J sum_<ij> s_i s_j - h sum_i s_i

    Bonds are counted once by summing only right and down neighbours.
    """
    right_neighbours = np.roll(lattice, shift=-1, axis=1)
    down_neighbours = np.roll(lattice, shift=-1, axis=0)

    interaction_energy = -coupling * np.sum(lattice * (right_neighbours + down_neighbours))
    field_energy = -field * np.sum(lattice)

    return float(interaction_energy + field_energy)