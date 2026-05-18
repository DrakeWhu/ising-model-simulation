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
