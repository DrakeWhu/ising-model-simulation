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
DEFAULT_THERMALIZATION_SWEEPS = 1000
DEFAULT_MEASUREMENT_SWEEPS = 5000
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_CHUNKS = 50
DEFAULT_SEED = 123

COUPLING = 1.0
FIELD = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    algorithm: str
    temperature: float
    thermalization_sweeps: int
    measurement_sweeps: int
    sample_every: int
    n_samples: int
    elapsed_seconds: float
    seconds_per_sample: float

    energy_density_mean: float
    energy_density_std: float
    energy_density_stderr: float

    abs_magnetization_density_mean: float
    abs_magnetization_density_std: float
    abs_magnetization_density_stderr: float

    tau_int_energy: float
    tau_int_abs_magnetization: float
    effective_energy_samples_per_second: float
    effective_abs_magnetization_samples_per_second: float

    energy_error: float

    mean_cluster_flips_per_sweep: float | None = None
    mean_cluster_fraction: float | None = None
    max_cluster_fraction: float | None = None


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


def normalized_autocorrelation(series: np.ndarray) -> np.ndarray:
    """Return the normalized autocorrelation function rho(t)."""
    values = np.asarray(series, dtype=np.float64)

    if values.size == 0:
        raise ValueError("Cannot compute autocorrelation of an empty series.")

    centered = values - np.mean(values)
    variance = np.mean(centered**2)

    if variance == 0.0:
        return np.ones(1, dtype=np.float64)

    raw = np.correlate(centered, centered, mode="full")[values.size - 1 :]
    normalization = variance * np.arange(values.size, 0, -1)

    return raw / normalization


def integrated_autocorrelation_time(
    series: np.ndarray,
    max_lag: int | None = None,
) -> float:
    """Estimate tau_int = 1/2 + sum rho(t), stopping at first non-positive rho."""
    values = np.asarray(series, dtype=np.float64)

    if values.size < 2:
        return 0.5

    autocorr = normalized_autocorrelation(values)

    if max_lag is None:
        max_lag = min(values.size - 1, values.size // 2)

    tau_int = 0.5

    for lag in range(1, max_lag + 1):
        if autocorr[lag] <= 0.0:
            break
        tau_int += autocorr[lag]

    return max(float(tau_int), 0.5)


def standard_error_with_autocorrelation(
    series: np.ndarray,
    tau_int: float,
) -> float:
    """Estimate the standard error of the mean including autocorrelation."""
    values = np.asarray(series, dtype=np.float64)

    if values.size < 2:
        return 0.0

    return float(np.std(values, ddof=1) * np.sqrt(2.0 * tau_int / values.size))


def effective_samples_per_second(
    n_samples: int,
    tau_int: float,
    elapsed_seconds: float,
) -> float:
    if elapsed_seconds <= 0.0:
        return float("nan")

    effective_samples = n_samples / (2.0 * tau_int)
    return effective_samples / elapsed_seconds


def summarize_benchmark_samples(
    algorithm: str,
    temperature: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    elapsed_seconds: float,
    energy_density_samples: np.ndarray,
    abs_magnetization_density_samples: np.ndarray,
    energy_error: float,
    mean_cluster_flips_per_sweep: float | None = None,
    mean_cluster_fraction: float | None = None,
    max_cluster_fraction: float | None = None,
) -> BenchmarkResult:
    n_samples = energy_density_samples.size

    tau_energy = integrated_autocorrelation_time(energy_density_samples)
    tau_abs_magnetization = integrated_autocorrelation_time(abs_magnetization_density_samples)

    return BenchmarkResult(
        algorithm=algorithm,
        temperature=temperature,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        n_samples=n_samples,
        elapsed_seconds=elapsed_seconds,
        seconds_per_sample=elapsed_seconds / n_samples,
        energy_density_mean=float(np.mean(energy_density_samples)),
        energy_density_std=float(np.std(energy_density_samples, ddof=1)),
        energy_density_stderr=standard_error_with_autocorrelation(
            energy_density_samples,
            tau_energy,
        ),
        abs_magnetization_density_mean=float(np.mean(abs_magnetization_density_samples)),
        abs_magnetization_density_std=float(np.std(abs_magnetization_density_samples, ddof=1)),
        abs_magnetization_density_stderr=standard_error_with_autocorrelation(
            abs_magnetization_density_samples,
            tau_abs_magnetization,
        ),
        tau_int_energy=tau_energy,
        tau_int_abs_magnetization=tau_abs_magnetization,
        effective_energy_samples_per_second=effective_samples_per_second(
            n_samples,
            tau_energy,
            elapsed_seconds,
        ),
        effective_abs_magnetization_samples_per_second=effective_samples_per_second(
            n_samples,
            tau_abs_magnetization,
            elapsed_seconds,
        ),
        energy_error=energy_error,
        mean_cluster_flips_per_sweep=mean_cluster_flips_per_sweep,
        mean_cluster_fraction=mean_cluster_fraction,
        max_cluster_fraction=max_cluster_fraction,
    )


def run_metropolis_numba_benchmark(
    lattice: np.ndarray,
    temperature: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    progress: bool,
) -> BenchmarkResult:
    beta = 1.0 / temperature
    current_lattice = lattice.copy()
    current_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)
    n_sites = current_lattice.size

    start = time.perf_counter()

    if thermalization_sweeps > 0:
        if progress:
            print_progress(
                "Metropolis thermalization",
                0,
                thermalization_sweeps,
                start,
            )

        current_lattice, spins, energies = metropolis_numba.metropolis_numba(
            current_lattice,
            n_sweeps=thermalization_sweeps,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
            field=FIELD,
        )

        current_energy = float(energies[-1])

        if progress:
            print_progress(
                "Metropolis thermalization",
                thermalization_sweeps,
                thermalization_sweeps,
                start,
            )

    n_samples = measurement_sweeps // sample_every
    if n_samples <= 0:
        raise ValueError("measurement_sweeps must be at least sample_every.")

    energy_density_samples = np.empty(n_samples, dtype=np.float64)
    abs_magnetization_density_samples = np.empty(n_samples, dtype=np.float64)

    measurement_start = time.perf_counter()

    if progress:
        print_progress("Metropolis measurement", 0, measurement_sweeps, measurement_start)

    for sample_index in range(n_samples):
        current_lattice, spins, energies = metropolis_numba.metropolis_numba(
            current_lattice,
            n_sweeps=sample_every,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
            field=FIELD,
        )

        current_energy = float(energies[-1])
        current_magnetization = float(spins[-1])

        energy_density_samples[sample_index] = current_energy / n_sites
        abs_magnetization_density_samples[sample_index] = abs(current_magnetization) / n_sites

        if progress:
            print_progress(
                "Metropolis measurement",
                (sample_index + 1) * sample_every,
                measurement_sweeps,
                measurement_start,
            )

    elapsed = time.perf_counter() - start
    checked_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    return summarize_benchmark_samples(
        algorithm="metropolis_numba",
        temperature=temperature,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        elapsed_seconds=elapsed,
        energy_density_samples=energy_density_samples,
        abs_magnetization_density_samples=abs_magnetization_density_samples,
        energy_error=abs(current_energy - checked_energy),
    )


def run_wolff_numba_benchmark(
    lattice: np.ndarray,
    temperature: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    progress: bool,
) -> BenchmarkResult:
    beta = 1.0 / temperature
    current_lattice = lattice.copy()
    current_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)
    n_sites = current_lattice.size

    total_cluster_flips = 0
    total_cluster_size = 0.0
    max_cluster_size = 0

    start = time.perf_counter()

    if thermalization_sweeps > 0:
        if progress:
            print_progress("Wolff thermalization", 0, thermalization_sweeps, start)

        (
            current_lattice,
            spins,
            energies,
            cluster_flips,
            mean_cluster_sizes,
            max_cluster_sizes,
        ) = wolff_numba.wolff_numba_with_cluster_stats(
            current_lattice,
            n_sweeps=thermalization_sweeps,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
        )

        current_energy = float(energies[-1])

        total_cluster_flips += int(np.sum(cluster_flips))
        total_cluster_size += float(np.sum(mean_cluster_sizes * cluster_flips))
        max_cluster_size = max(max_cluster_size, int(np.max(max_cluster_sizes)))

        if progress:
            print_progress(
                "Wolff thermalization",
                thermalization_sweeps,
                thermalization_sweeps,
                start,
            )

    n_samples = measurement_sweeps // sample_every
    if n_samples <= 0:
        raise ValueError("measurement_sweeps must be at least sample_every.")

    energy_density_samples = np.empty(n_samples, dtype=np.float64)
    abs_magnetization_density_samples = np.empty(n_samples, dtype=np.float64)

    measurement_start = time.perf_counter()

    if progress:
        print_progress("Wolff measurement", 0, measurement_sweeps, measurement_start)

    for sample_index in range(n_samples):
        (
            current_lattice,
            spins,
            energies,
            cluster_flips,
            mean_cluster_sizes,
            max_cluster_sizes,
        ) = wolff_numba.wolff_numba_with_cluster_stats(
            current_lattice,
            n_sweeps=sample_every,
            beta=beta,
            energy=current_energy,
            coupling=COUPLING,
        )

        current_energy = float(energies[-1])
        current_magnetization = float(spins[-1])

        total_cluster_flips += int(np.sum(cluster_flips))
        total_cluster_size += float(np.sum(mean_cluster_sizes * cluster_flips))
        max_cluster_size = max(max_cluster_size, int(np.max(max_cluster_sizes)))

        energy_density_samples[sample_index] = current_energy / n_sites
        abs_magnetization_density_samples[sample_index] = abs(current_magnetization) / n_sites

        if progress:
            print_progress(
                "Wolff measurement",
                (sample_index + 1) * sample_every,
                measurement_sweeps,
                measurement_start,
            )

    elapsed = time.perf_counter() - start
    checked_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    return summarize_benchmark_samples(
        algorithm="wolff_numba",
        temperature=temperature,
        thermalization_sweeps=thermalization_sweeps,
        measurement_sweeps=measurement_sweeps,
        sample_every=sample_every,
        elapsed_seconds=elapsed,
        energy_density_samples=energy_density_samples,
        abs_magnetization_density_samples=abs_magnetization_density_samples,
        energy_error=abs(current_energy - checked_energy),
        mean_cluster_flips_per_sweep=(
            total_cluster_flips / (thermalization_sweeps + measurement_sweeps)
        ),
        mean_cluster_fraction=(total_cluster_size / total_cluster_flips) / n_sites,
        max_cluster_fraction=max_cluster_size / n_sites,
    )


def _format_optional(value: float | None, width: int, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:>{width}.{precision}g}"


def _format_optional(value: float | None, width: int, precision: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:>{width}.{precision}g}"


def print_results_table(results: list[BenchmarkResult]) -> None:
    print()
    print(
        f"{'algorithm':<18} {'T':>9} {'samples':>8} {'time [s]':>10} "
        f"{'s/sample':>12} {'<E/N>':>12} {'stderr E':>11} "
        f"{'<|m|/N>':>12} {'stderr m':>11} "
        f"{'tau_E':>9} {'tau_m':>9} {'ESS_E/s':>10} {'ESS_m/s':>10} "
        f"{'cl/sweep':>10} {'<|C|>/N':>10} {'max|C|/N':>10} {'|dE|':>10}"
    )
    print("-" * 190)

    for result in results:
        print(
            f"{result.algorithm:<18} "
            f"{result.temperature:>9.6g} "
            f"{result.n_samples:>8d} "
            f"{result.elapsed_seconds:>10.4f} "
            f"{result.seconds_per_sample:>12.6e} "
            f"{result.energy_density_mean:>12.6g} "
            f"{result.energy_density_stderr:>11.3e} "
            f"{result.abs_magnetization_density_mean:>12.6g} "
            f"{result.abs_magnetization_density_stderr:>11.3e} "
            f"{result.tau_int_energy:>9.3g} "
            f"{result.tau_int_abs_magnetization:>9.3g} "
            f"{result.effective_energy_samples_per_second:>10.3g} "
            f"{result.effective_abs_magnetization_samples_per_second:>10.3g} "
            f"{_format_optional(result.mean_cluster_flips_per_sweep, width=10)} "
            f"{_format_optional(result.mean_cluster_fraction, width=10)} "
            f"{_format_optional(result.max_cluster_fraction, width=10)} "
            f"{result.energy_error:>10.3e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Metropolis Numba against Wolff Numba dynamics."
    )
    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
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
    if args.thermalization < 0:
        raise ValueError("thermalization must be non-negative.")
    if args.measurement <= 0:
        raise ValueError("measurement must be positive.")
    if args.sample_every <= 0:
        raise ValueError("sample_every must be positive.")
    if args.measurement < args.sample_every:
        raise ValueError("measurement must be at least sample_every.")

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
        f"Benchmark: L={args.size}, thermalization={args.thermalization}, "
        f"measurement={args.measurement}, sample_every={args.sample_every}, "
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
                    thermalization_sweeps=args.thermalization,
                    measurement_sweeps=args.measurement,
                    sample_every=args.sample_every,
                    progress=args.progress,
                )
            )

        if args.backend in {"both", "wolff"}:
            wolff_numba.seed_numba_rng(args.seed)
            results.append(
                run_wolff_numba_benchmark(
                    lattice=lattice,
                    temperature=temperature,
                    thermalization_sweeps=args.thermalization,
                    measurement_sweeps=args.measurement,
                    sample_every=args.sample_every,
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

        metropolis_result = by_algorithm["metropolis_numba"]
        wolff_result = by_algorithm["wolff_numba"]

        print(
            f"T={temperature:.6g}: "
            f"time ratio M/W = "
            f"{metropolis_result.elapsed_seconds / wolff_result.elapsed_seconds:.3g}x, "
            f"ESS_E/s ratio M/W = "
            f"{metropolis_result.effective_energy_samples_per_second / wolff_result.effective_energy_samples_per_second:.3g}x, "
            f"ESS_|m|/s ratio M/W = "
            f"{metropolis_result.effective_abs_magnetization_samples_per_second / wolff_result.effective_abs_magnetization_samples_per_second:.3g}x"
        )


if __name__ == "__main__":
    main()
