import numpy as np


def create_random_distribution(size: int) -> np.ndarray:
    """Create a square Ising lattice with spins s_i in {-1, +1}."""
    return np.random.choice([-1, 1], size=(size, size)).astype(np.int8)


def create_ones_distribution(size: int) -> np.ndarray:
    """Create a square Ising lattice with all spins up."""
    return np.ones((size, size), dtype=np.int8)