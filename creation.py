from typing import Literal

import numpy as np

InitialState = Literal["random", "all-up", "all-down", "checkerboard"]


def create_random_distribution(size: int) -> np.ndarray:
    """Create a square Ising lattice with random spins s_i in {-1, +1}."""
    return np.random.choice([-1, 1], size=(size, size)).astype(np.int8)


def create_ones_distribution(size: int) -> np.ndarray:
    """Create a square Ising lattice with all spins up."""
    return np.ones((size, size), dtype=np.int8)


def create_spin_lattice(size: int, initial_state: InitialState = "random") -> np.ndarray:
    """Create a square Ising lattice from a named initial condition."""
    if size <= 0:
        raise ValueError("size must be positive.")

    if initial_state == "random":
        return create_random_distribution(size)

    if initial_state == "all-up":
        return np.ones((size, size), dtype=np.int8)

    if initial_state == "all-down":
        return -np.ones((size, size), dtype=np.int8)

    if initial_state == "checkerboard":
        indices = np.indices((size, size)).sum(axis=0)
        return np.where(indices % 2 == 0, 1, -1).astype(np.int8)

    raise ValueError(f"Unknown initial_state: {initial_state!r}")
