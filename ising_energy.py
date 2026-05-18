from __future__ import annotations

import numpy as np
from numba import njit

from lattices import Lattice


@njit(cache=True)
def get_energy_from_edges_numba(
    spins: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    coupling: float,
    field: float,
) -> float:
    interaction_sum = 0.0
    spin_sum = 0.0

    for edge_index in range(edge_u.size):
        interaction_sum += spins[edge_u[edge_index]] * spins[edge_v[edge_index]]

    for site_index in range(spins.size):
        spin_sum += spins[site_index]

    return -coupling * interaction_sum - field * spin_sum


def get_energy_from_edges(
    spins: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    flat_spins = np.asarray(spins).reshape(-1)
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)

    if edge_u.shape != edge_v.shape:
        raise ValueError("edge_u and edge_v must have the same shape.")

    if edge_u.size > 0:
        min_index = min(int(edge_u.min()), int(edge_v.min()))
        max_index = max(int(edge_u.max()), int(edge_v.max()))

        if min_index < 0:
            raise ValueError("edge indices cannot be negative.")
        if max_index >= flat_spins.size:
            raise ValueError("edge indices exceed number of spins.")

    return float(
        get_energy_from_edges_numba(
            flat_spins,
            edge_u,
            edge_v,
            coupling,
            field,
        )
    )


def get_lattice_energy(
    spins: np.ndarray,
    lattice: Lattice,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    flat_spins = np.asarray(spins).reshape(-1)

    if flat_spins.size != lattice.n_sites:
        raise ValueError(
            f"Expected {lattice.n_sites} spins for lattice {lattice.name!r}, got {flat_spins.size}."
        )

    return get_energy_from_edges(
        flat_spins,
        edge_u=lattice.edge_u,
        edge_v=lattice.edge_v,
        coupling=coupling,
        field=field,
    )


@njit(cache=True)
def delta_energy_from_neighbors_numba(
    spins: np.ndarray,
    site_index: int,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    coupling: float,
    field: float,
) -> float:
    spin = spins[site_index]
    neighbor_sum = 0.0

    start = neighbor_offsets[site_index]
    end = neighbor_offsets[site_index + 1]

    for cursor in range(start, end):
        neighbor_sum += spins[neighbors[cursor]]

    return 2.0 * coupling * spin * neighbor_sum + 2.0 * field * spin


def delta_energy_for_site(
    spins: np.ndarray,
    lattice: Lattice,
    site_index: int,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    flat_spins = np.asarray(spins).reshape(-1)

    if flat_spins.size != lattice.n_sites:
        raise ValueError(
            f"Expected {lattice.n_sites} spins for lattice {lattice.name!r}, got {flat_spins.size}."
        )

    if not 0 <= site_index < lattice.n_sites:
        raise ValueError("site_index out of bounds.")

    return float(
        delta_energy_from_neighbors_numba(
            flat_spins,
            site_index,
            lattice.neighbor_offsets,
            lattice.neighbors,
            coupling,
            field,
        )
    )
