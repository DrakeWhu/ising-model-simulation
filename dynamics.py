from dataclasses import dataclass

import numpy as np

import metropolis_numba
import wolff_numba

BACKENDS = ["metropolis", "wolff"]


@dataclass(frozen=True)
class DynamicsResult:
    lattice: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray


def validate_backend(backend: str, field: float) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Expected one of {BACKENDS}.")

    if backend == "wolff" and field != 0.0:
        raise ValueError("Wolff backend is only implemented for zero external field.")


def seed_backend_rng(backend: str, seed: int | None) -> None:
    if seed is None:
        return

    if backend == "metropolis":
        metropolis_numba.seed_numba_rng(seed)
    elif backend == "wolff":
        wolff_numba.seed_numba_rng(seed)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Expected one of {BACKENDS}.")


def run_dynamics(
    lattice: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float,
    field: float,
    backend: str,
) -> DynamicsResult:
    validate_backend(backend=backend, field=field)

    if backend == "metropolis":
        updated_lattice, magnetizations, energies = metropolis_numba.metropolis_numba(
            lattice,
            n_sweeps=n_sweeps,
            beta=beta,
            energy=energy,
            coupling=coupling,
            field=field,
        )
    elif backend == "wolff":
        updated_lattice, magnetizations, energies = wolff_numba.wolff_numba(
            lattice,
            n_sweeps=n_sweeps,
            beta=beta,
            energy=energy,
            coupling=coupling,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Expected one of {BACKENDS}.")

    return DynamicsResult(
        lattice=updated_lattice,
        magnetizations=magnetizations,
        energies=energies,
    )


def warm_up_backend(
    backend: str,
    lattice: np.ndarray,
    beta: float,
    energy: float,
    coupling: float,
    field: float,
) -> None:
    run_dynamics(
        lattice=lattice.copy(),
        n_sweeps=1,
        beta=beta,
        energy=energy,
        coupling=coupling,
        field=field,
        backend=backend,
    )
