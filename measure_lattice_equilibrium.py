from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import ising_energy
import lattice_factory
import metropolis_lattice_numba
import wolff_lattice_numba
from lattices import Lattice


DEFAULT_LATTICE = "square2d"
DEFAULT_BACKEND = "wolff"
DEFAULT_SIZE = 32
DEFAULT_TEMPERATURE = 2.2691853
DEFAULT_COUPLING = 1.0
DEFAULT_THERMALIZATION_SWEEPS = 1000
DEFAULT_MEASUREMENT_SWEEPS = 5000
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_INITIAL_STATE = "random"
DEFAULT_SEED = 123


@dataclass(frozen=True)
class LatticeEquilibriumResult:
    lattice: str
    lattice_name: str
    size: int
    n_sites: int
    periodic: bool
    backend: str
    temperature: float
    coupling: float
    thermalization_sweeps: int
    measurement_sweeps: int
    sample_every: int
    n_samples: int
    elapsed_seconds: float
    energy_density_mean: float
    energy_density_std: float
    abs_magnetization_density_mean: float
    abs_magnetization_density_std: float
    specific_heat_per_spin: float
    abs_susceptibility_per_spin: float
    energy_error: float
    mean_cluster_flips_per_sweep: float | None = None
    mean_cluster_fraction: float | None = None
    max_cluster_fraction: float | None = None


def make_initial_spins(
    lattice: Lattice,
    initial_state: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if initial_state == "random":
        return rng.choice([-1, 1], size=lattice.n_sites).astype(np.int8)

    if initial_state == "all-up":
        return np.ones(lattice.n_sites, dtype=np.int8)

    if initial_state == "all-down":
        return -np.ones(lattice.n_sites, dtype=np.int8)

    if initial_state == "checkerboard":
        indices = np.indices(lattice.shape).sum(axis=0).reshape(-1)
        return np.where(indices % 2 == 0, 1, -1).astype(np.int8)

    raise ValueError(f"Unknown initial state {initial_state!r}.")


def run_backend(
    backend: str,
    spins: np.ndarray,
    lattice: Lattice,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if backend == "metropolis":
        updated_spins, magnetizations, energies = metropolis_lattice_numba.run_metropolis_lattice(
            spins,
            lattice,
            n_sweeps=n_sweeps,
            beta=beta,
            energy=energy,
            coupling=coupling,
            field=0.0,
        )

        return updated_spins, magnetizations, energies, None, None, None

    if backend == "wolff":
        return wolff_lattice_numba.run_wolff_lattice_with_cluster_stats(
            spins,
            lattice,
            n_sweeps=n_sweeps,
            beta=beta,
            energy=energy,
            coupling=coupling,
        )

    raise ValueError(f"Unknown backend {backend!r}.")


def summarize_samples(
    lattice_kind: str,
    lattice: Lattice,
    size: int,
    periodic: bool,
    backend: str,
    temperature: float,
    coupling: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    elapsed_seconds: float,
    energy_density_samples: np.ndarray,
    magnetization_samples: np.ndarray,
    current_energy: float,
    checked_energy: float,
    total_cluster_flips: int = 0,
    total_cluster_size: float = 0.0,
    max_cluster_size: int = 0,
) -> LatticeEquilibriumResult:
    n_sites = lattice.n_sites
    beta = 1.0 / temperature
    n_samples = energy_density_samples.size

    abs_magnetization_density_samples = np.abs(magnetization_samples) / n_sites
    total_energy_samples = energy_density_samples * n_sites

    if n_samples > 1:
        energy_density_std = float(np.std(energy_density_samples, ddof=1))
        abs_magnetization_density_std = float(np.std(abs_magnetization_density_samples, ddof=1))
        total_energy_variance = float(np.var(total_energy_samples, ddof=1))
        abs_magnetization_variance = float(np.var(np.abs(magnetization_samples), ddof=1))
    else:
        energy_density_std = 0.0
        abs_magnetization_density_std = 0.0
        total_energy_variance = 0.0
        abs_magnetization_variance = 0.0

    mean_cluster_flips_per_sweep = None
    mean_cluster_fraction = None
    max_cluster_fraction = None

    if total_cluster_flips > 0:
        mean_cluster_flips_per_sweep = total_cluster_flips / (
            thermalization_sweeps + measurement_sweeps
        )
        mean_cluster_fraction = (total_cluster_size / total_cluster_flips) / n_sites
        max_cluster_fraction = max_cluster_size / n_sites

    return LatticeEquilibriumResult(
        lattice=lattice_kind,
        lattice_name=lattice.name,
        size=size,
        n_sites=n_sites,
        periodic=periodic,
        backend=backend,
        temperature=temperature,
        coupling=coupling,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        n_samples=n_samples,
        elapsed_seconds=elapsed_seconds,
        energy_density_mean=float(np.mean(energy_density_samples)),
        energy_density_std=energy_density_std,
        abs_magnetization_density_mean=float(np.mean(abs_magnetization_density_samples)),
        abs_magnetization_density_std=abs_magnetization_density_std,
        specific_heat_per_spin=beta**2 * total_energy_variance / n_sites,
        abs_susceptibility_per_spin=beta * abs_magnetization_variance / n_sites,
        energy_error=abs(current_energy - checked_energy),
        mean_cluster_flips_per_sweep=mean_cluster_flips_per_sweep,
        mean_cluster_fraction=mean_cluster_fraction,
        max_cluster_fraction=max_cluster_fraction,
    )


def run_lattice_equilibrium(
    lattice_kind: str,
    size: int,
    periodic: bool,
    backend: str,
    temperature: float,
    coupling: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    initial_state: str,
    seed: int,
) -> LatticeEquilibriumResult:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if coupling < 0.0 and backend == "wolff":
        raise ValueError("Wolff backend requires non-negative coupling.")

    if thermalization_sweeps < 0:
        raise ValueError("thermalization_sweeps must be non-negative.")

    if measurement_sweeps <= 0:
        raise ValueError("measurement_sweeps must be positive.")

    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")

    if measurement_sweeps < sample_every:
        raise ValueError("measurement_sweeps must be at least sample_every.")

    lattice = lattice_factory.build_lattice(
        lattice_kind,
        size=size,
        periodic=periodic,
    )

    spins = make_initial_spins(
        lattice,
        initial_state=initial_state,
        seed=seed,
    )

    if backend == "metropolis":
        metropolis_lattice_numba.seed_numba_rng(seed)
    elif backend == "wolff":
        wolff_lattice_numba.seed_numba_rng(seed)
    else:
        raise ValueError(f"Unknown backend {backend!r}.")

    beta = 1.0 / temperature
    current_energy = ising_energy.get_lattice_energy(
        spins,
        lattice,
        coupling=coupling,
        field=0.0,
    )

    total_cluster_flips = 0
    total_cluster_size = 0.0
    max_cluster_size = 0

    start = time.perf_counter()

    if thermalization_sweeps > 0:
        (
            spins,
            magnetizations,
            energies,
            cluster_flips,
            mean_cluster_sizes,
            max_cluster_sizes,
        ) = run_backend(
            backend,
            spins,
            lattice,
            n_sweeps=thermalization_sweeps,
            beta=beta,
            energy=current_energy,
            coupling=coupling,
        )

        current_energy = float(energies[-1])

        if cluster_flips is not None:
            total_cluster_flips += int(np.sum(cluster_flips))
            total_cluster_size += float(np.sum(mean_cluster_sizes * cluster_flips))
            max_cluster_size = max(max_cluster_size, int(np.max(max_cluster_sizes)))

    n_samples = measurement_sweeps // sample_every

    energy_density_samples = np.empty(n_samples, dtype=np.float64)
    magnetization_samples = np.empty(n_samples, dtype=np.float64)

    for sample_index in range(n_samples):
        (
            spins,
            magnetizations,
            energies,
            cluster_flips,
            mean_cluster_sizes,
            max_cluster_sizes,
        ) = run_backend(
            backend,
            spins,
            lattice,
            n_sweeps=sample_every,
            beta=beta,
            energy=current_energy,
            coupling=coupling,
        )

        current_energy = float(energies[-1])
        current_magnetization = float(magnetizations[-1])

        energy_density_samples[sample_index] = current_energy / lattice.n_sites
        magnetization_samples[sample_index] = current_magnetization

        if cluster_flips is not None:
            total_cluster_flips += int(np.sum(cluster_flips))
            total_cluster_size += float(np.sum(mean_cluster_sizes * cluster_flips))
            max_cluster_size = max(max_cluster_size, int(np.max(max_cluster_sizes)))

    elapsed = time.perf_counter() - start

    checked_energy = ising_energy.get_lattice_energy(
        spins,
        lattice,
        coupling=coupling,
        field=0.0,
    )

    return summarize_samples(
        lattice_kind=lattice_kind,
        lattice=lattice,
        size=size,
        periodic=periodic,
        backend=backend,
        temperature=temperature,
        coupling=coupling,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        elapsed_seconds=elapsed,
        energy_density_samples=energy_density_samples,
        magnetization_samples=magnetization_samples,
        current_energy=current_energy,
        checked_energy=checked_energy,
        total_cluster_flips=total_cluster_flips,
        total_cluster_size=total_cluster_size,
        max_cluster_size=max_cluster_size,
    )


def print_result(result: LatticeEquilibriumResult) -> None:
    print()
    print(f"lattice:     {result.lattice_name}")
    print(f"backend:     {result.backend}")
    print(f"T:           {result.temperature:.8g}")
    print(f"N sites:     {result.n_sites}")
    print(f"samples:     {result.n_samples}")
    print(f"time [s]:    {result.elapsed_seconds:.4f}")
    print(f"<E/N>:       {result.energy_density_mean:.8g}")
    print(f"std(E/N):    {result.energy_density_std:.3e}")
    print(f"<|m|/N>:     {result.abs_magnetization_density_mean:.8g}")
    print(f"std(|m|/N):  {result.abs_magnetization_density_std:.3e}")
    print(f"C/N:         {result.specific_heat_per_spin:.8g}")
    print(f"chi_abs/N:   {result.abs_susceptibility_per_spin:.8g}")
    print(f"|dE check|:  {result.energy_error:.3e}")

    if result.mean_cluster_flips_per_sweep is not None:
        print(f"cl/sweep:    {result.mean_cluster_flips_per_sweep:.8g}")
        print(f"<|C|>/N:     {result.mean_cluster_fraction:.8g}")
        print(f"max|C|/N:    {result.max_cluster_fraction:.8g}")


def save_result_csv(result: LatticeEquilibriumResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure equilibrium observables on a generic Ising lattice."
    )

    parser.add_argument(
        "--lattice",
        choices=lattice_factory.SUPPORTED_LATTICES,
        default=DEFAULT_LATTICE,
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--open-boundary", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["metropolis", "wolff"],
        default=DEFAULT_BACKEND,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument(
        "--thermalization",
        type=int,
        default=DEFAULT_THERMALIZATION_SWEEPS,
    )
    parser.add_argument(
        "--measurement",
        type=int,
        default=DEFAULT_MEASUREMENT_SWEEPS,
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
    )
    parser.add_argument(
        "--initial-state",
        choices=["random", "all-up", "all-down", "checkerboard"],
        default=DEFAULT_INITIAL_STATE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-csv", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_lattice_equilibrium(
        lattice_kind=args.lattice,
        size=args.size,
        periodic=not args.open_boundary,
        backend=args.backend,
        temperature=args.temperature,
        coupling=args.coupling,
        thermalization_sweeps=args.thermalization,
        measurement_sweeps=args.measurement,
        sample_every=args.sample_every,
        initial_state=args.initial_state,
        seed=args.seed,
    )

    print_result(result)

    if args.save_csv is not None:
        save_result_csv(result, args.save_csv)
        print(f"\nSaved result to {args.save_csv}")


if __name__ == "__main__":
    main()
