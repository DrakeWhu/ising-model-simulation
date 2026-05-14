import matplotlib.pyplot as plt
import numpy as np

import creation
import nearest_neighbour as nn

LATTICE_SIZE = 64
FRAMES = 50


def main() -> None:
    energies = np.empty(FRAMES)

    for frame in range(FRAMES):
        lattice = creation.create_random_distribution(LATTICE_SIZE)
        energy = nn.get_energy(lattice)

        energies[frame] = energy

        plt.imshow(lattice)
        plt.pause(0.001)

    plt.show()

    plt.figure()
    plt.scatter(range(FRAMES), energies)
    plt.xlabel("Random configuration")
    plt.ylabel("Energy")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()