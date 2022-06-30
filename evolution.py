import matplotlib.pyplot as plt

def average_spin_and_energy(spins, energies):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    ax = axes[0]
    ax.plot(spins/N**2)
    ax.set_xlabel('Algorithm time steps')
    ax.set_ylabel(r'Average spin $\bar{m}$')
    ax.grid()
    ax = axes[1]
    ax.plot(energies)
    ax.set_xlabel('Algorithm time step')
    ax.set_ylabel(r'Energy $E/J$')
    ax.grid()
    fig.tight_layout()
    fig.suptitle(r'Evolution of average spin and energy for $\beta J=$0.7', y=1.07, size=18)
    plt.show()