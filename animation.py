import matplotlib.pyplot as plt
import creation
import nearest_neighbour as nn
import numpy as np

N = 500 # Size of the grid
frames = 50
energies = np.empty(frames)

for _ in range(frames):

    arr  = creation.create_random_distribution(N)
    energy = nn.get_energy(arr)
    energies[_]=energy

    plt.imshow(arr)
    plt.pause(0.001)

plt.show()

plt.figure(1)
plt.scatter(range(frames),energies)
plt.show()