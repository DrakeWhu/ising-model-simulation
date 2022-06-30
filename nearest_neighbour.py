import numpy as np
import scipy.ndimage as ndimage

eps = 0.5 # Interaction energy between spins, its negative when they are parallel

def spin_energy(i,u,d,r,l):
    E = -eps*(i*(u+d+r+l))
    return E

def microstate_energy(arr,N):

    E = 0

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == N-1 and j == N-1:
                e = spin_energy(arr[i,j],arr[i-1,j],arr[0,j],arr[i,0],arr[i,j-1])
            elif i == N-1 and j != N-1:
                e = spin_energy(arr[i,j],arr[i-1,j],arr[0,j],arr[i,j+1],arr[i,j-1])   
            elif i != N-1 and j == N-1:
                e = spin_energy(arr[i,j],arr[i-1,j],arr[i+1,j],arr[i,0],arr[i,j-1]) 
            elif i != N-1 and j != N-1:
                e = spin_energy(arr[i,j],arr[i-1,j],arr[i+1,j],arr[i,j+1],arr[i,j-1])
            else:
                print("Algo ha salido mal")
                break
            
            E += e
    
    return E

# This function comes from other person's code, and it's way optimal

def get_energy(lattice):
    # applies the nearest neighbour summation
    kern =  ndimage.generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice*ndimage.convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()