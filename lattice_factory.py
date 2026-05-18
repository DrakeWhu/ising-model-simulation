from __future__ import annotations

from typing import Literal

import lattices
from lattices import Lattice

LatticeKind = Literal["square2d", "cubic3d"]

SUPPORTED_LATTICES = ("square2d", "cubic3d")


def build_lattice(
    lattice_kind: str,
    size: int,
    periodic: bool = True,
) -> Lattice:
    if lattice_kind == "square2d":
        return lattices.square_lattice_2d(size=size, periodic=periodic)

    if lattice_kind == "cubic3d":
        return lattices.cubic_lattice_3d(size=size, periodic=periodic)

    supported = ", ".join(SUPPORTED_LATTICES)
    raise ValueError(f"Unknown lattice {lattice_kind!r}. Supported: {supported}.")
