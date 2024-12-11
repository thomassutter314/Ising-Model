Ising main: ising simulation in which the entire simulation is updated together, but some random fraction of sites are skipped so as to avoid time crystal solutions

Ising fast: this uses numba to do a fast simulation in which sites are updated individually. This does seem to still be slower than using Ising main.