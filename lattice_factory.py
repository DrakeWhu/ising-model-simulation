from __future__ import annotations

from typing import Literal

import lattices
from lattices import Lattice

LatticeKind = Literal[
    "square2d",
    "triangular2d",
    "hexagonal2d",
    "cubic3d",
    "bcc3d",
    "fcc3d",
]

SUPPORTED_LATTICES = (
    "square2d",
    "triangular2d",
    "hexagonal2d",
    "cubic3d",
    "bcc3d",
    "fcc3d",
)


def build_lattice(
    lattice_kind: str,
    size: int,
    periodic: bool = True,
) -> Lattice:
    if lattice_kind == "square2d":
        return lattices.square_lattice_2d(size=size, periodic=periodic)

    if lattice_kind == "triangular2d":
        return lattices.triangular_lattice_2d(size=size, periodic=periodic)

    if lattice_kind == "hexagonal2d":
        return lattices.hexagonal_lattice_2d(size=size, periodic=periodic)

    if lattice_kind == "cubic3d":
        return lattices.cubic_lattice_3d(size=size, periodic=periodic)

    if lattice_kind == "bcc3d":
        return lattices.bcc_lattice_3d(size=size, periodic=periodic)

    if lattice_kind == "fcc3d":
        return lattices.fcc_lattice_3d(size=size, periodic=periodic)

    supported = ", ".join(SUPPORTED_LATTICES)
    raise ValueError(f"Unknown lattice {lattice_kind!r}. Supported: {supported}.")
