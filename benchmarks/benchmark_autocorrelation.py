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
DEFAULT_CHUNKS = 50
DEFAULT_SEED = 123

COUPLING = 1.0
FIELD = 0.0


@dataclass(frozen=True)
class AutocorrelationSummary:
    algorithm: str
    temperature: float
    measurement_sweeps: int
    sample_count: int
    elapsed_seconds: float
    samples_per_second: float
    energy_density_mean: float
    abs_magnetization_mean: float
    tau_energy: float
    tau_abs_magnetization: float
    effective_samples_per_second_energy: float
    effective_samples_per_second_abs_magnetization: float


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
    """Return normalized autocorrelation rho(t) using an FFT estimator."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if values.size < 2:
        raise ValueError("series must contain at least two samples.")

    centered = values - np.mean(values)
    variance = np.var(centered)

    if variance == 0.0:
        autocorrelation = np.zeros(values.size, dtype=np.float64)
        autocorrelation[0] = 1.0
        return autocorrelation

    fft_size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)
    autocovariance = autocovariance[: values.size]

    normalization = np.arange(values.size, 0, -1, dtype=np.float64)
    autocovariance = autocovariance / normalization

    return autocovariance / autocovariance[0]


def integrated_autocorrelation_time(
    series: np.ndarray,
    max_lag: int | None = None,
) -> float:
    """Estimate tau_int = 1/2 + sum rho(t), cutting at first negative rho."""
    autocorrelation = normalized_autocorrelation(series)

    if max_lag is None:
        max_lag = autocorrelation.size // 2

    max_lag = max(1, min(max_lag, autocorrelation.size - 1))

    window = max_lag
    for lag in range(1, max_lag + 1):
        if autocorrelation[lag] <= 0.0:
            window = lag - 1
            break

    tau = 0.5 + float(np.sum(autocorrelation[1 : window + 1]))
    return max(tau, 0.5)


def run_metropolis_chunk(
    lattice: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return metropolis_numba.metropolis_numba(
        lattice,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        coupling=COUPLING,
        field=FIELD,
    )


def run_wolff_chunk(
    lattice: np.ndarray,
    n_sweeps: int,
    beta: float,
    energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return wolff_numba.wolff_numba(
        lattice,
        n_sweeps=n_sweeps,
        beta=beta,
        energy=energy,
        coupling=COUPLING,
    )


def run_chunks(
    algorithm: str,
    lattice: np.ndarray,
    temperature: float,
    n_sweeps: int,
    chunks: int,
    progress: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    beta = 1.0 / temperature
    current_lattice = lattice.copy()
    current_energy = nn.get_energy(current_lattice, coupling=COUPLING, field=FIELD)

    chunk_size = max(1, n_sweeps // chunks)
    completed_sweeps = 0
    all_spins: list[np.ndarray] = []
    all_energies: list[np.ndarray] = []

    start = time.perf_counter()

    if progress:
        print_progress(label, completed_sweeps, n_sweeps, start)

    while completed_sweeps < n_sweeps:
        sweeps_this_chunk = min(chunk_size, n_sweeps - completed_sweeps)

        if algorithm == "metropolis":
            current_lattice, spins, energies = run_metropolis_chunk(
                lattice=current_lattice,
                n_sweeps=sweeps_this_chunk,
                beta=beta,
                energy=current_energy,
            )
        elif algorithm == "wolff":
            current_lattice, spins, energies = run_wolff_chunk(
                lattice=current_lattice,
                n_sweeps=sweeps_this_chunk,
                beta=beta,
                energy=current_energy,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

        current_energy = float(energies[-1])
        all_spins.append(spins[1:].copy())
        all_energies.append(energies[1:].copy())

        completed_sweeps += sweeps_this_chunk

        if progress:
            print_progress(label, completed_sweeps, n_sweeps, start)

    elapsed = time.perf_counter() - start

    spin_series = np.concatenate(all_spins)
    energy_series = np.concatenate(all_energies)

    return current_lattice, spin_series, energy_series, elapsed


def run_algorithm_autocorrelation(
    algorithm: str,
    initial_lattice: np.ndarray,
    temperature: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    chunks: int,
    progress: bool,
    max_lag: int | None,
    seed: int,
) -> AutocorrelationSummary:
    if algorithm == "metropolis":
        metropolis_numba.seed_numba_rng(seed)
        label_prefix = "Metropolis"
    elif algorithm == "wolff":
        wolff_numba.seed_numba_rng(seed)
        label_prefix = "Wolff"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    thermalized_lattice, _, _, _ = run_chunks(
        algorithm=algorithm,
        lattice=initial_lattice,
        temperature=temperature,
        n_sweeps=thermalization_sweeps,
        chunks=chunks,
        progress=progress,
        label=f"{label_prefix} thermalization",
    )

    measured_lattice, spins, energies, elapsed = run_chunks(
        algorithm=algorithm,
        lattice=thermalized_lattice,
        temperature=temperature,
        n_sweeps=measurement_sweeps,
        chunks=chunks,
        progress=progress,
        label=f"{label_prefix} measurement",
    )

    checked_energy = nn.get_energy(measured_lattice, coupling=COUPLING, field=FIELD)
    if not np.isclose(float(energies[-1]), checked_energy):
        raise RuntimeError(
            f"{algorithm} energy drift detected: tracked={energies[-1]}, checked={checked_energy}"
        )

    n_sites = initial_lattice.size
    energy_density = energies / n_sites
    abs_magnetization_density = np.abs(spins) / n_sites

    tau_energy = integrated_autocorrelation_time(
        energy_density,
        max_lag=max_lag,
    )
    tau_abs_magnetization = integrated_autocorrelation_time(
        abs_magnetization_density,
        max_lag=max_lag,
    )

    sample_count = energy_density.size
    samples_per_second = sample_count / elapsed

    effective_samples_energy = sample_count / (2.0 * tau_energy)
    effective_samples_abs_magnetization = sample_count / (2.0 * tau_abs_magnetization)

    return AutocorrelationSummary(
        algorithm=f"{algorithm}_numba",
        temperature=temperature,
        measurement_sweeps=measurement_sweeps,
        sample_count=sample_count,
        elapsed_seconds=elapsed,
        samples_per_second=samples_per_second,
        energy_density_mean=float(np.mean(energy_density)),
        abs_magnetization_mean=float(np.mean(abs_magnetization_density)),
        tau_energy=tau_energy,
        tau_abs_magnetization=tau_abs_magnetization,
        effective_samples_per_second_energy=effective_samples_energy / elapsed,
        effective_samples_per_second_abs_magnetization=(
            effective_samples_abs_magnetization / elapsed
        ),
    )


def print_summary_table(results: list[AutocorrelationSummary]) -> None:
    print()
    print(
        f"{'algorithm':<18} {'T':>9} {'time [s]':>10} {'samples/s':>12} "
        f"{'<E>/N':>12} {'<|m|>':>12} {'tau_E':>10} {'tau_|m|':>10} "
        f"{'ESS_E/s':>12} {'ESS_|m|/s':>12}"
    )
    print("-" * 125)

    for result in results:
        print(
            f"{result.algorithm:<18} "
            f"{result.temperature:>9.6g} "
            f"{result.elapsed_seconds:>10.4f} "
            f"{result.samples_per_second:>12.3f} "
            f"{result.energy_density_mean:>12.6g} "
            f"{result.abs_magnetization_mean:>12.6g} "
            f"{result.tau_energy:>10.3f} "
            f"{result.tau_abs_magnetization:>10.3f} "
            f"{result.effective_samples_per_second_energy:>12.3f} "
            f"{result.effective_samples_per_second_abs_magnetization:>12.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark autocorrelation of Metropolis and Wolff Numba dynamics."
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
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-lag", type=int, default=None)
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
    )
    parser.add_argument(
        "--initial-state",
        choices=["random", "all-up", "all-down", "checkerboard"],
        default="random",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.size <= 0:
        raise ValueError("size must be positive.")
    if args.thermalization < 0:
        raise ValueError("thermalization cannot be negative.")
    if args.measurement <= 1:
        raise ValueError("measurement must be greater than one.")
    if args.chunks <= 0:
        raise ValueError("chunks must be positive.")
    if args.max_lag is not None and args.max_lag <= 0:
        raise ValueError("max-lag must be positive.")

    temperatures = args.temperatures
    if temperatures is None:
        temperatures = [
            2.0,
            exact_square_lattice.critical_temperature(COUPLING),
            2.5,
        ]

    np.random.seed(args.seed)
    initial_lattice = creation.create_spin_lattice(args.size, args.initial_state)
    initial_energy = nn.get_energy(initial_lattice, coupling=COUPLING, field=FIELD)

    print("Warming up Numba compilation...")
    metropolis_numba.seed_numba_rng(args.seed)
    metropolis_numba.metropolis_numba(
        initial_lattice,
        n_sweeps=1,
        beta=1.0 / temperatures[0],
        energy=initial_energy,
        coupling=COUPLING,
        field=FIELD,
    )

    wolff_numba.seed_numba_rng(args.seed)
    wolff_numba.wolff_numba(
        initial_lattice,
        n_sweeps=1,
        beta=1.0 / temperatures[0],
        energy=initial_energy,
        coupling=COUPLING,
    )

    print(
        f"Autocorrelation benchmark: L={args.size}, "
        f"thermalization={args.thermalization}, measurement={args.measurement}, "
        f"initial={args.initial_state}, backend={args.backend}"
    )

    results: list[AutocorrelationSummary] = []

    for temperature in temperatures:
        print(f"\nT = {temperature:.8g}")

        if args.backend in {"both", "metropolis"}:
            results.append(
                run_algorithm_autocorrelation(
                    algorithm="metropolis",
                    initial_lattice=initial_lattice,
                    temperature=temperature,
                    thermalization_sweeps=args.thermalization,
                    measurement_sweeps=args.measurement,
                    chunks=args.chunks,
                    progress=args.progress,
                    max_lag=args.max_lag,
                )
            )

        if args.backend in {"both", "wolff"}:
            results.append(
                run_algorithm_autocorrelation(
                    algorithm="wolff",
                    initial_lattice=initial_lattice,
                    temperature=temperature,
                    thermalization_sweeps=args.thermalization,
                    measurement_sweeps=args.measurement,
                    chunks=args.chunks,
                    progress=args.progress,
                    max_lag=args.max_lag,
                )
            )

    print_summary_table(results)

    print()
    print("tau values are measured in sampled sweeps.")
    print("ESS/s = approximate effective independent samples per second.")


if __name__ == "__main__":
    main()
