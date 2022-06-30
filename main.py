import evolution
import metropolis
import creation
import scipy.ndimage as ndimage
import nearest_neighbour as nn

N = 500 # Size of the grid
steps = 10 # Steps of the simulation
J = 1 # Interaction energy between spins
T = 300 # Temperature in Kelvin
kb = 1.38e-23 # Boltzmann constant
BJ = J/(kb*T)

lattice_n = creation.create_random_distribution(N)

spins, energies = metropolis(lattice_n, steps, BJ, nn.get_energy(lattice_n))

evolution.average_spin_and_energy(spins, energies)