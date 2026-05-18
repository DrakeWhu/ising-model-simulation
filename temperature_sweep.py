import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dynamics
import measure_equilibrium
from exact_square_lattice import (
    critical_temperature,
    exact_energy_density,
    exact_spontaneous_magnetization_density,
)

DEFAULT_LATTICE_SIZE = 64
DEFAULT_T_MIN = 1.5
DEFAULT_T_MAX = 3.5
DEFAULT_NUM_TEMPERATURES = 21
DEFAULT_THERMALIZATION_SWEEPS = 1000
DEFAULT_MEASUREMENT_SWEEPS = 5000
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_INITIAL_STATE = "all-up"
DEFAULT_OUTPUT = Path("outputs/temperature_sweep.csv")


CSV_FIELDNAMES = [
    "temperature",
    "beta",
    "coupling",
    "field",
    "lattice_size",
    "n_sites",
    "initial_state",
    "thermalization_sweeps",
    "measurement_sweeps",
    "sample_every",
    "n_samples",
    "energy_mean",
    "energy_density_mean",
    "magnetization_mean",
    "abs_magnetization_mean",
    "specific_heat_per_spin",
    "susceptibility_per_spin",
    "abs_susceptibility_per_spin",
    "binder_cumulant",
    "elapsed_seconds",
    "seed",
    "backend",
]


def build_exact_reference_curves(
    temperatures: np.ndarray,
    coupling: float,
    field: float,
    n_points: int = 1000,
    n_critical_points: int = 2000,
) -> dict[str, np.ndarray] | None:
    """Return exact thermodynamic-limit reference curves when available.

    The implemented exact references are only valid for the isotropic
    ferromagnetic 2D square-lattice Ising model at zero external field.
    """
    if field != 0.0:
        return None

    temperature_min = float(np.min(temperatures))
    temperature_max = float(np.max(temperatures))

    exact_temperatures = np.linspace(temperature_min, temperature_max, n_points)

    tc = critical_temperature(coupling)
    if temperature_min < tc < temperature_max:
        span = temperature_max - temperature_min
        critical_half_width = min(
            0.25 * span,
            tc - temperature_min,
            temperature_max - tc,
        )

        critical_coordinate = np.linspace(-1.0, 1.0, n_critical_points)
        critical_temperatures = (
            tc
            + np.sign(critical_coordinate) * critical_half_width * np.abs(critical_coordinate) ** 3
        )

        exact_temperatures = np.unique(
            np.concatenate(
                [
                    exact_temperatures,
                    critical_temperatures,
                    np.array([tc]),
                ]
            )
        )

    exact_energy = np.asarray(
        [exact_energy_density(float(temperature), coupling) for temperature in exact_temperatures],
        dtype=np.float64,
    )

    magnetization_temperatures = exact_temperatures[exact_temperatures <= tc]
    if temperature_min < tc < temperature_max:
        magnetization_temperatures = np.unique(
            np.concatenate([magnetization_temperatures, np.array([tc])])
        )

    exact_magnetization = np.asarray(
        [
            exact_spontaneous_magnetization_density(float(temperature), coupling)
            for temperature in magnetization_temperatures
        ],
        dtype=np.float64,
    )

    return {
        "energy_temperature": exact_temperatures,
        "energy_density": exact_energy,
        "magnetization_temperature": magnetization_temperatures,
        "spontaneous_magnetization": exact_magnetization,
    }


def build_temperature_grid(
    explicit_temperatures: list[float] | None,
    t_min: float,
    t_max: float,
    num_temperatures: int,
) -> np.ndarray:
    if explicit_temperatures is not None:
        temperatures = np.asarray(explicit_temperatures, dtype=np.float64)
    else:
        if num_temperatures < 2:
            raise ValueError("num_temperatures must be at least 2.")
        temperatures = np.linspace(t_min, t_max, num_temperatures)

    if np.any(temperatures <= 0.0):
        raise ValueError("All temperatures must be positive.")

    return temperatures


def row_from_result(
    result: measure_equilibrium.EquilibriumMeasurement,
    seed: int | None,
) -> dict[str, float | int | str | None]:
    summary = result.summary

    return {
        "temperature": result.temperature,
        "beta": result.beta,
        "coupling": result.coupling,
        "field": result.field,
        "lattice_size": result.lattice_size,
        "n_sites": result.n_sites,
        "initial_state": result.initial_state,
        "thermalization_sweeps": result.thermalization_sweeps,
        "measurement_sweeps": result.measurement_sweeps,
        "sample_every": result.sample_every,
        "n_samples": result.n_samples,
        "energy_mean": summary.energy_mean,
        "energy_density_mean": summary.energy_density_mean,
        "magnetization_mean": summary.magnetization_mean,
        "abs_magnetization_mean": summary.abs_magnetization_mean,
        "specific_heat_per_spin": summary.specific_heat_per_spin,
        "susceptibility_per_spin": summary.susceptibility_per_spin,
        "abs_susceptibility_per_spin": summary.abs_susceptibility_per_spin,
        "binder_cumulant": summary.binder_cumulant,
        "elapsed_seconds": result.elapsed_seconds,
        "seed": seed,
        "backend": result.backend,
    }


def plot_temperature_sweep(
    rows: list[dict[str, float | int | str | None]],
    coupling: float,
    field: float,
    save_path: Path | None,
    show: bool,
) -> None:
    temperatures = np.asarray([row["temperature"] for row in rows], dtype=np.float64)
    energy = np.asarray([row["energy_density_mean"] for row in rows], dtype=np.float64)
    abs_magnetization = np.asarray(
        [row["abs_magnetization_mean"] for row in rows],
        dtype=np.float64,
    )
    heat_capacity = np.asarray(
        [row["specific_heat_per_spin"] for row in rows],
        dtype=np.float64,
    )
    susceptibility = np.asarray(
        [row["abs_susceptibility_per_spin"] for row in rows],
        dtype=np.float64,
    )

    binder = np.asarray([row["binder_cumulant"] for row in rows], dtype=np.float64)

    backend = str(rows[0].get("backend", "metropolis"))
    backend_label = {
        "metropolis": "Metropolis",
        "wolff": "Wolff",
    }.get(backend, backend)

    exact_curves = build_exact_reference_curves(
        temperatures=temperatures,
        coupling=coupling,
        field=field,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].plot(temperatures, energy, marker="o", linestyle="none", label=backend_label)
    if exact_curves is not None:
        axes[0].plot(
            exact_curves["energy_temperature"],
            exact_curves["energy_density"],
            linestyle="-",
            linewidth=2,
            label="exact energy, thermodynamic limit",
        )

    axes[0].set_ylabel(r"$\langle E\rangle/N$")
    axes[0].grid()
    axes[0].legend(loc="upper left")

    axes[1].plot(
        temperatures,
        abs_magnetization,
        marker="o",
        linestyle="none",
        label=backend_label,
    )
    if exact_curves is not None:
        axes[1].plot(
            exact_curves["magnetization_temperature"],
            exact_curves["spontaneous_magnetization"],
            linewidth=2,
            label="exact $m_0$, thermodynamic limit",
        )
    axes[1].set_ylabel(r"$\langle |m| \rangle$")
    axes[1].grid()
    axes[1].legend(loc="upper right")

    axes[2].plot(temperatures, heat_capacity, marker="o", linestyle="none")
    axes[2].set_ylabel(r"$C_V/N$")
    axes[2].grid()

    axes[3].plot(temperatures, susceptibility, marker="o", linestyle="none")
    axes[3].set_ylabel(r"$\chi_{|m|}/N$")
    axes[3].grid()

    axes[4].plot(temperatures, binder, marker="o", linestyle="none")
    axes[4].set_ylabel(r"$U_4$")
    axes[4].grid()

    axes[5].axis("off")

    if field == 0.0:
        tc = critical_temperature(coupling)
        for ax in axes[:5]:
            ax.axvline(tc, linestyle="--", linewidth=1)
        axes[5].text(
            0.05,
            0.8,
            rf"$T_c = {tc:.6g}$",
            transform=axes[5].transAxes,
            fontsize=12,
        )

    for ax in axes[:5]:
        ax.set_xlabel(r"$T$")

    fig.suptitle(f"2D square-lattice Ising temperature sweep ({backend_label})")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2D Ising temperature sweep.")

    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument("--num-temperatures", type=int, default=DEFAULT_NUM_TEMPERATURES)

    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=0.0)
    parser.add_argument(
        "--initial-state",
        choices=["random", "all-up", "all-down", "checkerboard"],
        default=DEFAULT_INITIAL_STATE,
    )

    parser.add_argument("--thermalization", type=int, default=DEFAULT_THERMALIZATION_SWEEPS)
    parser.add_argument("--measurement", type=int, default=DEFAULT_MEASUREMENT_SWEEPS)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-output", type=Path, default=Path("outputs/temperature_sweep.png"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--backend", choices=dynamics.BACKENDS, default="metropolis")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temperatures = build_temperature_grid(
        explicit_temperatures=args.temperatures,
        t_min=args.t_min,
        t_max=args.t_max,
        num_temperatures=args.num_temperatures,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sweep_start_time = time.perf_counter()
    rows: list[dict[str, float | int | str | None]] = []

    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for index, temperature in enumerate(temperatures, start=1):
            run_seed = None if args.seed is None else args.seed + index - 1

            print()
            print(
                f"[{index}/{len(temperatures)}] "
                f"T={temperature:.6g}, L={args.size}, "
                f"initial={args.initial_state}, "
                f"backend={args.backend}, "
                f"seed={run_seed}"
            )

            result = measure_equilibrium.run_equilibrium_measurement(
                lattice_size=args.size,
                temperature=float(temperature),
                thermalization_sweeps=args.thermalization,
                measurement_sweeps=args.measurement,
                sample_every=args.sample_every,
                coupling=args.coupling,
                field=args.field,
                seed=run_seed,
                progress=not args.no_progress,
                initial_state=args.initial_state,
                backend=args.backend,
            )

            row = row_from_result(result, seed=run_seed)
            rows.append(row)

            writer.writerow(row)
            csv_file.flush()

            print(
                "summary: "
                f"E/N={result.summary.energy_density_mean:.6g}, "
                f"|m|={result.summary.abs_magnetization_mean:.6g}, "
                f"Cv/N={result.summary.specific_heat_per_spin:.6g}, "
                f"chi_abs/N={result.summary.abs_susceptibility_per_spin:.6g}, "
                f"U4={result.summary.binder_cumulant:.6g}"
            )

    sweep_elapsed = time.perf_counter() - sweep_start_time
    average_time_per_temperature = sweep_elapsed / len(rows)

    print()
    print(f"Wrote sweep data to: {args.output}")
    print(
        f"Completed {len(rows)} temperature points in {sweep_elapsed:.3f} s "
        f"({average_time_per_temperature:.3f} s/temperature)"
    )

    if args.plot:
        plot_temperature_sweep(
            rows=rows,
            coupling=args.coupling,
            field=args.field,
            save_path=args.plot_output,
            show=not args.no_show,
        )
        print(f"Wrote plot to: {args.plot_output}")


if __name__ == "__main__":
    main()
