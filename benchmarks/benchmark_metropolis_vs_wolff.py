import argparse
import time
from dataclasses import dataclass

import numpy as np

import creation
import exact_square_lattice
import metropolis_numba
import nearest_neighbour as nn
import wolff_numba

DEFAULT_LATTICE_SIZE = 128
DEFAULT_N_SWEEPS = 500
DEFAULT_CHUNKS = 50
DEFAULT_SEED = 123

COUPLING = 1.0
FIELD = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    algorithm: str
    temperature: float
    sweeps: int
    elapsed_seconds: float
    seconds_per_sweep: float
    energy_density: float
    magnetization_density: float
    energy_error: float


def print_progress(label: str, completed: int, total: int, start_time: float) -> None:
    width = 32
    fraction = completed / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start_time

    print(
        f"\r{label}: [{bar}] {completed}/{total} sweeps "
        f"({100 * fraction:5.1f}%) elapsed={elapsed:7.2f}s",
        end="",
        flush=True,
    )

    if completed >= total:
        print()


def run_metropolis_numba_benchmark(
    lattice: np.ndarray,
    temperature: float,
    n_sweeps: int,
    chunks: int,
    progress: bool,
) -> BenchmarkResult:
    beta = 1.0 / temperature
    current_lattice = lattice.copy()
    current_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    chunk_size = max(1, n_sweeps // chunks)
    completed_sweeps = 0
    start = time.perf_counter()

    if progress:
        print_progress("Metropolis Numba", completed_sweeps, n_sweeps, start)

    while completed_sweeps < n_sweeps:
        sweeps_this_chunk = min(chunk_size, n_sweeps - completed_sweeps)

        current_lattice, spins, energies = metropolis_numba.metropolis_numba(
            current_lattice,
            n_sweeps=sweeps_this_chunk,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
            field=FIELD,
        )

        current_energy = float(energies[-1])
        current_magnetization = float(spins[-1])
        completed_sweeps += sweeps_this_chunk

        if progress:
            print_progress("Metropolis Numba", completed_sweeps, n_sweeps, start)

    elapsed = time.perf_counter() - start
    checked_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    return BenchmarkResult(
        algorithm="metropolis_numba",
        temperature=temperature,
        sweeps=n_sweeps,
        elapsed_seconds=elapsed,
        seconds_per_sweep=elapsed / n_sweeps,
        energy_density=current_energy / current_lattice.size,
        magnetization_density=current_magnetization / current_lattice.size,
        energy_error=abs(current_energy - checked_energy),
    )


def run_wolff_numba_benchmark(
    lattice: np.ndarray,
    temperature: float,
    n_sweeps: int,
    chunks: int,
    progress: bool,
) -> BenchmarkResult:
    beta = 1.0 / temperature
    current_lattice = lattice.copy()
    current_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    chunk_size = max(1, n_sweeps // chunks)
    completed_sweeps = 0
    start = time.perf_counter()

    if progress:
        print_progress("Wolff Numba", completed_sweeps, n_sweeps, start)

    while completed_sweeps < n_sweeps:
        sweeps_this_chunk = min(chunk_size, n_sweeps - completed_sweeps)

        current_lattice, spins, energies = wolff_numba.wolff_numba(
            current_lattice,
            n_sweeps=sweeps_this_chunk,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
        )

        current_energy = float(energies[-1])
        current_magnetization = float(spins[-1])
        completed_sweeps += sweeps_this_chunk

        if progress:
            print_progress("Wolff Numba", completed_sweeps, n_sweeps, start)

    elapsed = time.perf_counter() - start
    checked_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    return BenchmarkResult(
        algorithm="wolff_numba",
        temperature=temperature,
        sweeps=n_sweeps,
        elapsed_seconds=elapsed,
        seconds_per_sweep=elapsed / n_sweeps,
        energy_density=current_energy / current_lattice.size,
        magnetization_density=current_magnetization / current_lattice.size,
        energy_error=abs(current_energy - checked_energy),
    )


def print_results_table(results: list[BenchmarkResult]) -> None:
    print()
    print(
        f"{'algorithm':<18} {'T':>9} {'sweeps':>8} {'time [s]':>10} "
        f"{'s/sweep':>12} {'E/N':>12} {'m/N':>12} {'|dE|':>10}"
    )
    print("-" * 101)

    for result in results:
        print(
            f"{result.algorithm:<18} "
            f"{result.temperature:>9.6g} "
            f"{result.sweeps:>8d} "
            f"{result.elapsed_seconds:>10.4f} "
            f"{result.seconds_per_sweep:>12.6e} "
            f"{result.energy_density:>12.6g} "
            f"{result.magnetization_density:>12.6g} "
            f"{result.energy_error:>10.3e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Metropolis Numba against Wolff Numba dynamics."
    )
    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_N_SWEEPS)
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Temperatures to benchmark. Defaults to 2.0, Tc, 2.5.",
    )
    parser.add_argument(
        "--backend",
        choices=["both", "metropolis", "wolff"],
        default="both",
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--initial-state",
        choices=["random", "all-up", "all-down", "checkerboard"],
        default="random",
        help="Initial lattice state.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.size <= 0:
        raise ValueError("size must be positive.")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive.")
    if args.chunks <= 0:
        raise ValueError("chunks must be positive.")

    temperatures = args.temperatures
    if temperatures is None:
        temperatures = [
            2.0,
            exact_square_lattice.critical_temperature(COUPLING),
            2.5,
        ]

    np.random.seed(args.seed)
    lattice = creation.create_spin_lattice(args.size, args.initial_state)
    energy = nn.get_energy(lattice, coupling=COUPLING, field=FIELD)

    print("Warming up Numba compilation...")
    metropolis_numba.seed_numba_rng(args.seed)
    metropolis_numba.metropolis_numba(
        lattice,
        n_sweeps=1,
        beta=1.0 / temperatures[0],
        energy=energy,
        coupling=COUPLING,
        field=FIELD,
    )

    wolff_numba.seed_numba_rng(args.seed)
    wolff_numba.wolff_numba(
        lattice,
        n_sweeps=1,
        beta=1.0 / temperatures[0],
        energy=energy,
        coupling=COUPLING,
    )

    print(
        f"Benchmark: L={args.size}, sweeps={args.sweeps}, "
        f"initial={args.initial_state}, backend={args.backend}"
    )

    results: list[BenchmarkResult] = []

    for temperature in temperatures:
        print(f"\nT = {temperature:.8g}")

        if args.backend in {"both", "metropolis"}:
            metropolis_numba.seed_numba_rng(args.seed)
            results.append(
                run_metropolis_numba_benchmark(
                    lattice=lattice,
                    temperature=temperature,
                    n_sweeps=args.sweeps,
                    chunks=args.chunks,
                    progress=args.progress,
                )
            )

        if args.backend in {"both", "wolff"}:
            wolff_numba.seed_numba_rng(args.seed)
            results.append(
                run_wolff_numba_benchmark(
                    lattice=lattice,
                    temperature=temperature,
                    n_sweeps=args.sweeps,
                    chunks=args.chunks,
                    progress=args.progress,
                )
            )

    print_results_table(results)

    paired_results = {}
    for result in results:
        paired_results.setdefault(result.temperature, {})[result.algorithm] = result

    print()
    for temperature, by_algorithm in paired_results.items():
        if "metropolis_numba" not in by_algorithm or "wolff_numba" not in by_algorithm:
            continue

        metropolis_time = by_algorithm["metropolis_numba"].elapsed_seconds
        wolff_time = by_algorithm["wolff_numba"].elapsed_seconds

        print(
            f"T={temperature:.6g}: "
            f"Metropolis/Wolff wall-time ratio = {metropolis_time / wolff_time:.3g}x"
        )


if __name__ == "__main__":
    main()
