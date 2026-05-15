import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import creation
import metropolis_numba
import nearest_neighbour as nn

DEFAULT_LATTICE_SIZE = 256
DEFAULT_N_SWEEPS = 5000
DEFAULT_SWEEPS_PER_FRAME = 10

DEFAULT_COUPLING = 1.0
DEFAULT_TEMPERATURE = 2.0
DEFAULT_FIELD = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 2D square-lattice Ising simulation.")
    parser.add_argument("--size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_N_SWEEPS)
    parser.add_argument("--sweeps-per-frame", type=int, default=DEFAULT_SWEEPS_PER_FRAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument(
        "--pause",
        type=float,
        default=0.001,
        help="Matplotlib pause time after each frame.",
    )
    return parser.parse_args()


def setup_figure(
    lattice: np.ndarray,
    total_sweeps: int,
    initial_magnetization: float,
    initial_energy_density: float,
    temperature: float,
):
    plt.ion()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_lattice, ax_magnetization, ax_energy = axes

    image = ax_lattice.imshow(
        lattice,
        interpolation="nearest",
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax_lattice.set_title("Spin lattice")
    ax_lattice.set_xticks([])
    ax_lattice.set_yticks([])

    (magnetization_line,) = ax_magnetization.plot([0], [initial_magnetization])
    ax_magnetization.set_title("Magnetization")
    ax_magnetization.set_xlabel("Monte Carlo sweep")
    ax_magnetization.set_ylabel(r"$m$")
    ax_magnetization.set_xlim(0, total_sweeps)
    ax_magnetization.set_ylim(-1.05, 1.05)
    ax_magnetization.grid()

    (energy_line,) = ax_energy.plot([0], [initial_energy_density])
    ax_energy.set_title("Energy density")
    ax_energy.set_xlabel("Monte Carlo sweep")
    ax_energy.set_ylabel(r"$E/N$")
    ax_energy.set_xlim(0, total_sweeps)
    ax_energy.grid()

    fig.suptitle(f"2D Ising model, T={temperature:g}")
    fig.tight_layout()

    return fig, image, magnetization_line, energy_line, ax_energy


def main() -> None:
    args = parse_args()

    if args.temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    if args.sweeps <= 0:
        raise ValueError("Number of sweeps must be positive.")
    if args.sweeps_per_frame <= 0:
        raise ValueError("sweeps-per-frame must be positive.")

    beta = 1.0 / args.temperature

    lattice = creation.create_random_distribution(args.size)
    current_energy = nn.get_energy(lattice, coupling=args.coupling, field=args.field)
    n_sites = lattice.size

    print("Warming up Numba compilation...")
    metropolis_numba.metropolis_numba(
        lattice.copy(),
        n_sweeps=1,
        beta=beta,
        energy=current_energy,
        coupling=args.coupling,
        field=args.field,
    )

    sweep_history = [0]
    magnetization_history = [lattice.sum() / n_sites]
    energy_density_history = [current_energy / n_sites]

    fig, image, magnetization_line, energy_line, ax_energy = setup_figure(
        lattice=lattice,
        total_sweeps=args.sweeps,
        initial_magnetization=magnetization_history[0],
        initial_energy_density=energy_density_history[0],
        temperature=args.temperature,
    )

    completed_sweeps = 0
    start_time = time.perf_counter()

    while completed_sweeps < args.sweeps:
        if not plt.fignum_exists(fig.number):
            print("\nFigure closed. Stopping simulation.")
            break

        sweeps_this_frame = min(args.sweeps_per_frame, args.sweeps - completed_sweeps)

        lattice, chunk_spins, chunk_energies = metropolis_numba.metropolis_numba(
            lattice,
            n_sweeps=sweeps_this_frame,
            beta=beta,
            energy=current_energy,
            coupling=args.coupling,
            field=args.field,
        )

        current_energy = float(chunk_energies[-1])

        new_sweeps = np.arange(
            completed_sweeps + 1,
            completed_sweeps + sweeps_this_frame + 1,
        )
        sweep_history.extend(new_sweeps.tolist())
        magnetization_history.extend((chunk_spins[1:] / n_sites).tolist())
        energy_density_history.extend((chunk_energies[1:] / n_sites).tolist())

        completed_sweeps += sweeps_this_frame

        image.set_data(lattice)

        magnetization_line.set_data(sweep_history, magnetization_history)

        energy_line.set_data(sweep_history, energy_density_history)
        ax_energy.relim()
        ax_energy.autoscale_view(scalex=False, scaley=True)

        elapsed = time.perf_counter() - start_time
        fig.suptitle(
            f"2D Ising model, T={args.temperature:g}, "
            f"sweep={completed_sweeps}/{args.sweeps}, elapsed={elapsed:.2f}s"
        )

        fig.canvas.draw_idle()
        plt.pause(args.pause)

    elapsed = time.perf_counter() - start_time
    print(f"\nDone: {completed_sweeps} sweeps in {elapsed:.2f} s")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
