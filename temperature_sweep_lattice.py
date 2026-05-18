from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lattice_factory
import measure_lattice_equilibrium


DEFAULT_LATTICE = "square2d"
DEFAULT_BACKEND = "wolff"
DEFAULT_SIZE = 32
DEFAULT_T_MIN = 1.5
DEFAULT_T_MAX = 3.5
DEFAULT_NUM_TEMPERATURES = 21
DEFAULT_COUPLING = 1.0
DEFAULT_THERMALIZATION_SWEEPS = 1000
DEFAULT_MEASUREMENT_SWEEPS = 5000
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_INITIAL_STATE = "random"
DEFAULT_SEED = 123


def default_critical_temperature(
    lattice_kind: str,
    coupling: float,
) -> float | None:
    if lattice_kind == "square2d":
        return float(2.0 * coupling / np.log(1.0 + np.sqrt(2.0)))

    if lattice_kind == "triangular2d":
        return float(4.0 * coupling / np.log(3.0))

    if lattice_kind == "hexagonal2d":
        return float(2.0 * coupling / np.log(2.0 + np.sqrt(3.0)))

    if lattice_kind == "cubic3d":
        return float(4.5115 * coupling)

    return None


def build_temperature_grid(
    t_min: float,
    t_max: float,
    num_temperatures: int,
) -> np.ndarray:
    if t_min <= 0.0:
        raise ValueError("t_min must be positive.")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")
    if t_max < t_min:
        raise ValueError("t_max must be greater than or equal to t_min.")
    if num_temperatures <= 0:
        raise ValueError("num_temperatures must be positive.")

    return np.linspace(t_min, t_max, num_temperatures)


def run_temperature_sweep(
    lattice_kind: str,
    size: int,
    periodic: bool,
    backend: str,
    temperatures: np.ndarray,
    coupling: float,
    thermalization_sweeps: int,
    measurement_sweeps: int,
    sample_every: int,
    initial_state: str,
    seed: int,
) -> list[measure_lattice_equilibrium.LatticeEquilibriumResult]:
    results: list[measure_lattice_equilibrium.LatticeEquilibriumResult] = []

    for index, temperature in enumerate(temperatures):
        print(f"T = {temperature:.8g}")

        result = measure_lattice_equilibrium.run_lattice_equilibrium(
            lattice_kind=lattice_kind,
            size=size,
            periodic=periodic,
            backend=backend,
            temperature=float(temperature),
            coupling=coupling,
            thermalization_sweeps=thermalization_sweeps,
            measurement_sweeps=measurement_sweeps,
            sample_every=sample_every,
            initial_state=initial_state,
            seed=seed + index,
        )

        results.append(result)

    return results


def save_results_csv(
    results: list[measure_lattice_equilibrium.LatticeEquilibriumResult],
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("Cannot save an empty result list.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(results[0]).keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def plot_temperature_sweep(
    results: list[measure_lattice_equilibrium.LatticeEquilibriumResult],
    critical_temperature: float | None,
    save_path: Path | None,
    show: bool,
) -> None:
    if not results:
        raise ValueError("Cannot plot an empty result list.")

    temperatures = np.array([result.temperature for result in results])
    energy = np.array([result.energy_density_mean for result in results])
    magnetization = np.array([result.abs_magnetization_density_mean for result in results])
    specific_heat = np.array([result.specific_heat_per_spin for result in results])
    susceptibility = np.array([result.abs_susceptibility_per_spin for result in results])

    cluster_fraction = np.array(
        [
            np.nan if result.mean_cluster_fraction is None else result.mean_cluster_fraction
            for result in results
        ]
    )

    first = results[0]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(9, 13),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"{first.lattice_name}, backend={first.backend}, L={first.size}, N={first.n_sites}"
    )

    axes[0].plot(temperatures, energy, marker="o", linestyle="none")
    axes[0].set_ylabel(r"$\langle E\rangle/N$")
    axes[0].grid()

    axes[1].plot(temperatures, magnetization, marker="o", linestyle="none")
    axes[1].set_ylabel(r"$\langle |m| \rangle/N$")
    axes[1].grid()

    axes[2].plot(temperatures, specific_heat, marker="o", linestyle="none")
    axes[2].set_ylabel(r"$C/N$")
    axes[2].grid()

    axes[3].plot(temperatures, susceptibility, marker="o", linestyle="none")
    axes[3].set_ylabel(r"$\chi_{|m|}/N$")
    axes[3].grid()

    axes[4].plot(temperatures, cluster_fraction, marker="o", linestyle="none")
    axes[4].set_ylabel(r"$\langle |C| \rangle/N$")
    axes[4].set_xlabel(r"$T$")
    axes[4].grid()

    if critical_temperature is not None:
        for axis in axes:
            axis.axvline(
                critical_temperature,
                linestyle="--",
                linewidth=1,
                label=r"$T_c$",
            )

        axes[0].legend(loc="best")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a temperature sweep on a generic Ising lattice."
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
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument(
        "--num-temperatures",
        type=int,
        default=DEFAULT_NUM_TEMPERATURES,
    )
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
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--critical-temperature", type=float, default=None)
    parser.add_argument("--no-critical-line", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temperatures = build_temperature_grid(
        t_min=args.t_min,
        t_max=args.t_max,
        num_temperatures=args.num_temperatures,
    )

    results = run_temperature_sweep(
        lattice_kind=args.lattice,
        size=args.size,
        periodic=not args.open_boundary,
        backend=args.backend,
        temperatures=temperatures,
        coupling=args.coupling,
        thermalization_sweeps=args.thermalization,
        measurement_sweeps=args.measurement,
        sample_every=args.sample_every,
        initial_state=args.initial_state,
        seed=args.seed,
    )

    if args.save_csv is not None:
        save_results_csv(results, args.save_csv)
        print(f"\nSaved results to {args.save_csv}")

    if args.plot or args.save_plot is not None:
        critical_temperature = args.critical_temperature

        if critical_temperature is None and not args.no_critical_line:
            critical_temperature = default_critical_temperature(
                args.lattice,
                coupling=args.coupling,
            )

        plot_temperature_sweep(
            results,
            critical_temperature=critical_temperature,
            save_path=args.save_plot,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
