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