import argparse
import sys
import time
from collections.abc import Callable

import creation
import metropolis
import metropolis_numba
import nearest_neighbour as nn

DEFAULT_LATTICE_SIZE = 128
DEFAULT_N_SWEEPS = 200
DEFAULT_CHUNKS = 40

COUPLING = 1.0
TEMPERATURE = 2.0
FIELD = 0.0
BETA = 1.0 / TEMPERATURE


def print_progress(label: str, completed: int, total: int, start_time: float) -> None:
    width = 32
    fraction = completed / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start_time

    sys.stdout.write(
        f"\r{label}: [{bar}] {completed}/{total} sweeps "
        f"({100 * fraction:5.1f}%) elapsed={elapsed:7.2f}s"
    )
    sys.stdout.flush()

    if completed >= total:
        sys.stdout.write("\n")


def time_run(
    label: str,
    function: Callable,
    lattice,
    energy: float,
    n_sweeps: int,
    chunks: int,
    progress: bool,
) -> float:
    current_lattice = lattice.copy()
    current_energy = energy

    chunk_size = max(1, n_sweeps // chunks)
    completed_sweeps = 0

    start = time.perf_counter()

    if progress:
        print_progress(label, completed_sweeps, n_sweeps, start)

    while completed_sweeps < n_sweeps:
        sweeps_this_chunk = min(chunk_size, n_sweeps - completed_sweeps)

        current_lattice, _spins, energies = function(
            current_lattice,
            n_sweeps=sweeps_this_chunk,
            beta=BETA,
            energy=current_energy,
            coupling=COUPLING,
            field=FIELD,
        )

        current_energy = energies[-1]
        completed_sweeps += sweeps_this_chunk

        if progress:
            print_progress(label, completed_sweeps, n_sweeps, start)

    elapsed = time.perf_counter() - start

    if not progress:
        print(f"{label}: {elapsed:.4f} s")
    else:
        print(f"{label} total: {elapsed:.4f} s")

    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Python and Numba Metropolis backends.")
    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_N_SWEEPS)
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--backend",
        choices=["both", "python", "numba"],
        default="both",
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a simple sweep-level progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lattice = creation.create_random_distribution(args.size)
    energy = nn.get_energy(lattice, coupling=COUPLING, field=FIELD)

    print("Warming up Numba compilation...")
    metropolis_numba.metropolis_numba(
        lattice,
        n_sweeps=1,
        beta=BETA,
        energy=energy,
        coupling=COUPLING,
        field=FIELD,
    )

    print(f"Benchmark: L={args.size}, sweeps={args.sweeps}, backend={args.backend}")

    python_time = None
    numba_time = None

    if args.backend in {"both", "python"}:
        python_time = time_run(
            "Python reference",
            metropolis.metropolis,
            lattice,
            energy,
            n_sweeps=args.sweeps,
            chunks=args.chunks,
            progress=args.progress,
        )

    if args.backend in {"both", "numba"}:
        numba_time = time_run(
            "Numba backend",
            metropolis_numba.metropolis_numba,
            lattice,
            energy,
            n_sweeps=args.sweeps,
            chunks=args.chunks,
            progress=args.progress,
        )

    if python_time is not None and numba_time is not None:
        print(f"Speedup: {python_time / numba_time:.2f}x")


if __name__ == "__main__":
    main()
