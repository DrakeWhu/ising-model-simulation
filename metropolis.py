"""
The Idea: Find the equilibrium state mu in the magnet at a particular temperature beta=1/kT
(how many spins are +1 and how many are -1).

1. Call the current state mu
2. Pick a random particle on the lattice and flip the spin sign. Call the state nu. We want to find P(mu -> nu)
which is the probability that we'll accept this new state
3. If E_nu>E_mu then P(nu->mu)=1 and P(mu->nu)=exp(-beta(E_nu-E_mu))
4. If E_nu<E_mu then P(mu->nu)=1
5. Change to state nu (flip the sign) with the probabilities above
6. Go back to step 1. Repeat the whole thing many times and eventually you'll force out an equilibrium state
"""

import numba
from numba import njit
import random
import numpy as np

#@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip sign
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] # initial spin
        spin_f = spin_i*-1

        # compute change in energy
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -spin_i*spin_arr[x-1,y]
            E_f += -spin_f*spin_arr[x-1,y]
        if x<N-1:
            E_i += -spin_i*spin_arr[x+1,y]
            E_f += -spin_f*spin_arr[x+1,y]
        if y>0:
            E_i += -spin_i*spin_arr[x,y-1]
            E_f += -spin_f*spin_arr[x,y-1]
        if y<N-1:
            E_i += -spin_i*spin_arr[x,y+1]
            E_f += -spin_f*spin_arr[x,y+1]
        
        # 3 / 4. Change state with designated probabilities
        dE = E_f-E_i
        if (dE>0)*(np.random.random()<np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy += dE
        
        net_spins[t]=spin_arr.sum()
        net_energy[t]= energy
    
    return net_spins, net_energy