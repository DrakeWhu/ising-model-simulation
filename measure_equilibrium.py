import argparse
import time
from dataclasses import dataclass

import numpy as np

import creation
import dynamics
import nearest_neighbour as nn
import observables

DEFAULT_LATTICE_SIZE = 64
DEFAULT_TEMPERATURE = 2.0
DEFAULT_COUPLING = 1.0
DEFAULT_FIELD = 0.0

DEFAULT_THERMALIZATION_SWEEPS = 1000
DEFAULT_MEASUREMENT_SWEEPS = 5000
DEFAULT_SAMPLE_EVERY = 10


@dataclass(frozen=True)
class SweepResult:
    lattice: np.ndarray
    energy: float
    magnetization: float


@dataclass(frozen=True)
class EquilibriumMeasurement:
    temperature: float
    beta: float
    coupling: float
    field: float
    lattice_size: int
    n_sites: int
    thermalization_sweeps: int
    measurement_sweeps: int
    sample_every: int
    n_samples: int
    elapsed_seconds: float
    summary: observables.ObservableSummary
    initial_state: str
    backend: str


def print_progress(label: str, completed: int, total: int, start_time: float) -> None:
    width = 32
    fraction = completed / total if total > 0 else 1.0
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


def run_sweeps(
    lattice: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
    coupling: float,
    field: float,
    backend: str,
) -> SweepResult:
    result = dynamics.run_dynamics(
        lattice=lattice,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        coupling=coupling,
        field=field,
        backend=backend,
    )

    return SweepResult(
        lattice=result.lattice,
        energy=float(result.energies[-1]),
        magnetization=float(result.magnetizations[-1]),
    )


def run_equilibrium_measurement(
    lattice_size: int,
    temperature: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    coupling: float = DEFAULT_COUPLING,
    field: float = DEFAULT_FIELD,
    seed: int | None = None,
    progress: bool = True,
    backend: str = "metropolis",
    initial_state: creation.InitialState = "random",
) -> EquilibriumMeasurement:
    if lattice_size <= 0:
        raise ValueError("lattice_size must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if thermalization_sweeps < 0:
        raise ValueError("thermalization_sweeps cannot be negative.")
    if measurement_sweeps <= 0:
        raise ValueError("measurement_sweeps must be positive.")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")

    dynamics.validate_backend(backend=backend, field=field)

    beta = 1.0 / temperature

    if seed is not None:
        np.random.seed(seed)

    lattice = creation.create_spin_lattice(lattice_size, initial_state=initial_state)
    current_energy = nn.get_energy(lattice, coupling=coupling, field=field)
    n_sites = lattice.size

    # Compile Numba before timing / before seeded production dynamics.
    dynamics.warm_up_backend(
        backend=backend,
        lattice=lattice,
        beta=beta,
        energy=current_energy,
        coupling=coupling,
        field=field,
    )

    if seed is not None:
        dynamics.seed_backend_rng(backend, seed)

    start_time = time.perf_counter()

    completed_thermalization = 0
    thermalization_chunk = max(1, sample_every)

    if progress and thermalization_sweeps > 0:
        print_progress("Thermalization", 0, thermalization_sweeps, start_time)

    while completed_thermalization < thermalization_sweeps:
        sweeps_this_chunk = min(
            thermalization_chunk,
            thermalization_sweeps - completed_thermalization,
        )

        sweep_result = run_sweeps(
            lattice=lattice,
            n_sweeps=sweeps_this_chunk,
            beta=beta,
            energy=current_energy,
            coupling=coupling,
            field=field,
            backend=backend,
        )

        lattice = sweep_result.lattice
        current_energy = sweep_result.energy

        completed_thermalization += sweeps_this_chunk

        if progress:
            print_progress(
                "Thermalization",
                completed_thermalization,
                thermalization_sweeps,
                start_time,
            )

    sampled_magnetizations: list[float] = []
    sampled_energies: list[float] = []

    completed_measurement = 0
    measurement_start = time.perf_counter()

    if progress:
        print_progress("Measurement", 0, measurement_sweeps, measurement_start)

    while completed_measurement < measurement_sweeps:
        sweeps_this_sample = min(sample_every, measurement_sweeps - completed_measurement)

        sweep_result = run_sweeps(
            lattice=lattice,
            n_sweeps=sweeps_this_sample,
            beta=beta,
            energy=current_energy,
            coupling=coupling,
            field=field,
            backend=backend,
        )

        lattice = sweep_result.lattice
        current_energy = sweep_result.energy

        sampled_magnetizations.append(sweep_result.magnetization)
        sampled_energies.append(sweep_result.energy)

        completed_measurement += sweeps_this_sample

        if progress:
            print_progress(
                "Measurement",
                completed_measurement,
                measurement_sweeps,
                measurement_start,
            )

    energies = np.asarray(sampled_energies, dtype=np.float64)
    magnetizations = np.asarray(sampled_magnetizations, dtype=np.float64)

    summary = observables.summarize_observables(
        energies=energies,
        magnetizations=magnetizations,
        beta=beta,
        n_sites=n_sites,
    )

    elapsed_seconds = time.perf_counter() - start_time

    return EquilibriumMeasurement(
        temperature=temperature,
        beta=beta,
        coupling=coupling,
        field=field,
        lattice_size=lattice_size,
        n_sites=n_sites,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        n_samples=len(sampled_energies),
        elapsed_seconds=elapsed_seconds,
        summary=summary,
        initial_state=initial_state,
        backend=backend,
    )


def print_summary(result: EquilibriumMeasurement) -> None:
    summary = result.summary

    print()
    print("Equilibrium measurement")
    print("=======================")
    print(f"L                    = {result.lattice_size}")
    print(f"N_sites              = {result.n_sites}")
    print(f"T                    = {result.temperature:g}")
    print(f"beta                 = {result.beta:g}")
    print(f"J                    = {result.coupling:g}")
    print(f"h                    = {result.field:g}")
    print(f"initial_state        = {result.initial_state}")
    print(f"thermalization       = {result.thermalization_sweeps} sweeps")
    print(f"measurement          = {result.measurement_sweeps} sweeps")
    print(f"sample_every         = {result.sample_every} sweeps")
    print(f"n_samples            = {result.n_samples}")
    print()
    print(f"<E>                  = {summary.energy_mean:.8g}")
    print(f"<E>/N                = {summary.energy_density_mean:.8g}")
    print(f"<m>                  = {summary.magnetization_mean:.8g}")
    print(f"<|m|>                = {summary.abs_magnetization_mean:.8g}")
    print(f"C_V/N                = {summary.specific_heat_per_spin:.8g}")
    print(f"chi/N                = {summary.susceptibility_per_spin:.8g}")
    print(f"chi_abs/N            = {summary.abs_susceptibility_per_spin:.8g}")
    print(f"Binder U4            = {summary.binder_cumulant:.8g}")
    print()
    print(f"elapsed              = {result.elapsed_seconds:.3f} s")
    print(f"backend              = {result.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure equilibrium observables for 2D Ising.")
    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument("--thermalization", type=int, default=DEFAULT_THERMALIZATION_SWEEPS)
    parser.add_argument("--measurement", type=int, default=DEFAULT_MEASUREMENT_SWEEPS)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")

    (
        parser.add_argument(
            "--initial-state",
            choices=["random", "all-up", "all-down", "checkerboard"],
            default="random",
        ),
    )

    parser.add_argument(
        "--backend",
        choices=dynamics.BACKENDS,
        default="metropolis",
        help="Dynamics backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_equilibrium_measurement(
        lattice_size=args.size,
        temperature=args.temperature,
        thermalization_sweeps=args.thermalization,
        measurement_sweeps=args.measurement,
        sample_every=args.sample_every,
        coupling=args.coupling,
        field=args.field,
        seed=args.seed,
        progress=not args.no_progress,
        initial_state=args.initial_state,
        backend=args.backend,
    )

    print_summary(result)


if __name__ == "__main__":
    main()
