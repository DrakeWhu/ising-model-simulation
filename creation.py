import matplotlib.pyplot as plt
import numpy as np
import random

# Creation of the array of ones and zeroes representing the spin up or spin down state of each lattice point

def create_random_distribution(N): # Here N is the number of rows/columns in the lattice. N^2 is the number of lattice points
    arr = np.random.rand(N,N)
    
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i,j]<0.5:
                arr[i,j] = -1
            else:
                arr[i,j] = 1

    return arr

def create_ones_distribution(N):
    arr = np.ones(N,N)
    return arr