import creation
import evolution
import metropolis_numba
import nearest_neighbour as nn

LATTICE_SIZE = 128
N_SWEEPS = 5000

COUPLING = 1.0
TEMPERATURE = 2.0
FIELD = 0.0

BETA = 1.0 / TEMPERATURE


def main() -> None:
    lattice = creation.create_random_distribution(LATTICE_SIZE)
    initial_energy = nn.get_energy(lattice, coupling=COUPLING, field=FIELD)

    _final_lattice, spins, energies = metropolis_numba.metropolis_numba(
        lattice,
        n_sweeps=N_SWEEPS,
        beta=BETA,
        energy=initial_energy,
        coupling=COUPLING,
        field=FIELD,
    )

    evolution.average_spin_and_energy(
        spins,
        energies,
        n_sites=lattice.size,
        beta=BETA,
        coupling=COUPLING,
    )


if __name__ == "__main__":
    main()
