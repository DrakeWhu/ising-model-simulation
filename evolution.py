import matplotlib.pyplot as plt
import numpy as np


def average_spin_and_energy(
    spins: np.ndarray,
    energies: np.ndarray,
    n_sites: int,
    beta: float | None = None,
    coupling: float = 1.0,
) -> None:
    """Plot magnetization per spin and total energy over Monte Carlo sweeps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(spins / n_sites)
    ax.set_xlabel("Monte Carlo sweep")
    ax.set_ylabel(r"Magnetization per spin $m$")
    ax.grid()

    ax = axes[1]
    ax.plot(energies)
    ax.set_xlabel("Monte Carlo sweep")
    ax.set_ylabel(r"Energy $E$")
    ax.grid()

    title = "Evolution of magnetization and energy"
    if beta is not None:
        title += rf" ($\beta J = {beta * coupling:.3g}$)"

    fig.suptitle(title, y=1.07, size=16)
    fig.tight_layout()
    plt.show()