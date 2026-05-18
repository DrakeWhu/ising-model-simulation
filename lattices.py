from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lattice:
    name: str
    shape: tuple[int, ...]
    n_sites: int
    neighbor_offsets: np.ndarray
    neighbors: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray


def _build_lattice_from_neighbor_lists(
    name: str,
    shape: tuple[int, ...],
    neighbor_lists: list[list[int]],
) -> Lattice:
    n_sites = len(neighbor_lists)

    offsets = np.empty(n_sites + 1, dtype=np.int64)
    offsets[0] = 0

    total_neighbors = 0
    for i, site_neighbors in enumerate(neighbor_lists):
        total_neighbors += len(site_neighbors)
        offsets[i + 1] = total_neighbors

    neighbors = np.empty(total_neighbors, dtype=np.int64)

    cursor = 0
    for site_neighbors in neighbor_lists:
        for neighbor in site_neighbors:
            neighbors[cursor] = neighbor
            cursor += 1

    edge_pairs: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()

    for i, site_neighbors in enumerate(neighbor_lists):
        for j in site_neighbors:
            edge = (min(i, j), max(i, j))
            if edge not in seen_edges:
                seen_edges.add(edge)
                edge_pairs.append(edge)

    edge_u = np.array([edge[0] for edge in edge_pairs], dtype=np.int64)
    edge_v = np.array([edge[1] for edge in edge_pairs], dtype=np.int64)

    return Lattice(
        name=name,
        shape=shape,
        n_sites=n_sites,
        neighbor_offsets=offsets,
        neighbors=neighbors,
        edge_u=edge_u,
        edge_v=edge_v,
    )


def _build_3d_basis_lattice_from_nearest_neighbors(
    name: str,
    shape: tuple[int, int, int, int],
    basis: np.ndarray,
    nearest_neighbor_distance_squared: float,
    periodic: bool,
    tolerance: float = 1.0e-12,
) -> Lattice:
    size_x, size_y, size_z, n_basis = shape

    if basis.shape != (n_basis, 3):
        raise ValueError("basis must have shape (n_basis, 3).")

    def index(x: int, y: int, z: int, basis_index: int) -> int:
        return (((x * size_y + y) * size_z + z) * n_basis) + basis_index

    neighbor_lists: list[list[int]] = [[] for _ in range(size_x * size_y * size_z * n_basis)]

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                for basis_index in range(n_basis):
                    site_index = index(x, y, z, basis_index)
                    site_position = basis[basis_index]

                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                for neighbor_basis_index in range(n_basis):
                                    if (
                                        dx == 0
                                        and dy == 0
                                        and dz == 0
                                        and neighbor_basis_index == basis_index
                                    ):
                                        continue

                                    delta = (
                                        np.array([dx, dy, dz], dtype=np.float64)
                                        + basis[neighbor_basis_index]
                                        - site_position
                                    )
                                    distance_squared = float(np.dot(delta, delta))

                                    if (
                                        abs(distance_squared - nearest_neighbor_distance_squared)
                                        > tolerance
                                    ):
                                        continue

                                    nx = x + dx
                                    ny = y + dy
                                    nz = z + dz

                                    if periodic:
                                        nx %= size_x
                                        ny %= size_y
                                        nz %= size_z
                                    elif not (
                                        0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z
                                    ):
                                        continue

                                    neighbor_index = index(
                                        nx,
                                        ny,
                                        nz,
                                        neighbor_basis_index,
                                    )

                                    if (
                                        neighbor_index != site_index
                                        and neighbor_index not in neighbor_lists[site_index]
                                    ):
                                        neighbor_lists[site_index].append(neighbor_index)

    return _build_lattice_from_neighbor_lists(
        name=name,
        shape=shape,
        neighbor_lists=neighbor_lists,
    )


def square_lattice_2d(size: int, periodic: bool = True) -> Lattice:
    if size <= 0:
        raise ValueError("size must be positive.")

    def index(row: int, col: int) -> int:
        return row * size + col

    neighbor_lists: list[list[int]] = []

    for row in range(size):
        for col in range(size):
            site_neighbors: list[int] = []

            candidate_positions = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ]

            for neighbor_row, neighbor_col in candidate_positions:
                if periodic:
                    neighbor_row %= size
                    neighbor_col %= size
                elif not (0 <= neighbor_row < size and 0 <= neighbor_col < size):
                    continue

                neighbor = index(neighbor_row, neighbor_col)
                if neighbor != index(row, col) and neighbor not in site_neighbors:
                    site_neighbors.append(neighbor)

            neighbor_lists.append(site_neighbors)

    boundary = "periodic" if periodic else "open"
    return _build_lattice_from_neighbor_lists(
        name=f"square_2d_{boundary}",
        shape=(size, size),
        neighbor_lists=neighbor_lists,
    )


def triangular_lattice_2d(size: int, periodic: bool = True) -> Lattice:
    if size <= 0:
        raise ValueError("size must be positive.")

    def index(row: int, col: int) -> int:
        return row * size + col

    neighbor_lists: list[list[int]] = []

    for row in range(size):
        for col in range(size):
            site_neighbors: list[int] = []

            candidate_positions = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
                (row - 1, col + 1),
                (row + 1, col - 1),
            ]

            for neighbor_row, neighbor_col in candidate_positions:
                if periodic:
                    neighbor_row %= size
                    neighbor_col %= size
                elif not (0 <= neighbor_row < size and 0 <= neighbor_col < size):
                    continue

                neighbor = index(neighbor_row, neighbor_col)
                if neighbor != index(row, col) and neighbor not in site_neighbors:
                    site_neighbors.append(neighbor)

            neighbor_lists.append(site_neighbors)

    boundary = "periodic" if periodic else "open"
    return _build_lattice_from_neighbor_lists(
        name=f"triangular_2d_{boundary}",
        shape=(size, size),
        neighbor_lists=neighbor_lists,
    )


def hexagonal_lattice_2d(size: int, periodic: bool = True) -> Lattice:
    """Build a 2D honeycomb/hexagonal lattice.

    The lattice has size x size unit cells and two sites per unit cell.
    """
    if size <= 0:
        raise ValueError("size must be positive.")

    def index(row: int, col: int, sublattice: int) -> int:
        return 2 * (row * size + col) + sublattice

    neighbor_lists: list[list[int]] = []

    for row in range(size):
        for col in range(size):
            # A sublattice site.
            site_neighbors: list[int] = []

            candidate_b_positions = [
                (row, col),
                (row - 1, col),
                (row, col - 1),
            ]

            for neighbor_row, neighbor_col in candidate_b_positions:
                if periodic:
                    neighbor_row %= size
                    neighbor_col %= size
                elif not (0 <= neighbor_row < size and 0 <= neighbor_col < size):
                    continue

                neighbor = index(neighbor_row, neighbor_col, 1)
                if neighbor not in site_neighbors:
                    site_neighbors.append(neighbor)

            neighbor_lists.append(site_neighbors)

            # B sublattice site.
            site_neighbors = []

            candidate_a_positions = [
                (row, col),
                (row + 1, col),
                (row, col + 1),
            ]

            for neighbor_row, neighbor_col in candidate_a_positions:
                if periodic:
                    neighbor_row %= size
                    neighbor_col %= size
                elif not (0 <= neighbor_row < size and 0 <= neighbor_col < size):
                    continue

                neighbor = index(neighbor_row, neighbor_col, 0)
                if neighbor not in site_neighbors:
                    site_neighbors.append(neighbor)

            neighbor_lists.append(site_neighbors)

    boundary = "periodic" if periodic else "open"
    return _build_lattice_from_neighbor_lists(
        name=f"hexagonal_2d_{boundary}",
        shape=(size, size, 2),
        neighbor_lists=neighbor_lists,
    )


def cubic_lattice_3d(size: int, periodic: bool = True) -> Lattice:
    if size <= 0:
        raise ValueError("size must be positive.")

    def index(x: int, y: int, z: int) -> int:
        return (x * size + y) * size + z

    neighbor_lists: list[list[int]] = []

    for x in range(size):
        for y in range(size):
            for z in range(size):
                site_neighbors: list[int] = []

                candidate_positions = [
                    (x - 1, y, z),
                    (x + 1, y, z),
                    (x, y - 1, z),
                    (x, y + 1, z),
                    (x, y, z - 1),
                    (x, y, z + 1),
                ]

                for nx, ny, nz in candidate_positions:
                    if periodic:
                        nx %= size
                        ny %= size
                        nz %= size
                    elif not (0 <= nx < size and 0 <= ny < size and 0 <= nz < size):
                        continue

                    neighbor = index(nx, ny, nz)
                    if neighbor != index(x, y, z) and neighbor not in site_neighbors:
                        site_neighbors.append(neighbor)

                neighbor_lists.append(site_neighbors)

    boundary = "periodic" if periodic else "open"
    return _build_lattice_from_neighbor_lists(
        name=f"cubic_3d_{boundary}",
        shape=(size, size, size),
        neighbor_lists=neighbor_lists,
    )


def bcc_lattice_3d(size: int, periodic: bool = True) -> Lattice:
    """Build a 3D body-centered cubic lattice.

    The lattice has size x size x size conventional cubic cells and two sites
    per cell.
    """
    if size <= 0:
        raise ValueError("size must be positive.")

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    boundary = "periodic" if periodic else "open"

    return _build_3d_basis_lattice_from_nearest_neighbors(
        name=f"bcc_3d_{boundary}",
        shape=(size, size, size, 2),
        basis=basis,
        nearest_neighbor_distance_squared=0.75,
        periodic=periodic,
    )


def fcc_lattice_3d(size: int, periodic: bool = True) -> Lattice:
    """Build a 3D face-centered cubic lattice.

    The lattice has size x size x size conventional cubic cells and four sites
    per cell.
    """
    if size <= 0:
        raise ValueError("size must be positive.")

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    boundary = "periodic" if periodic else "open"

    return _build_3d_basis_lattice_from_nearest_neighbors(
        name=f"fcc_3d_{boundary}",
        shape=(size, size, size, 4),
        basis=basis,
        nearest_neighbor_distance_squared=0.5,
        periodic=periodic,
    )
